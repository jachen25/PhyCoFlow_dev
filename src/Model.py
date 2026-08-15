

import math, os, torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from neuralop.models import FNO as NeuralOpFNO
except ImportError:
    NeuralOpFNO = None

# KeOps is optional; unavailable installations use the torch neighbor backend.
try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

from obs_consistency import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

# Prior distributions.
class IIDGaussianPrior(nn.Module):
    """IID standard-normal prior on a point cloud."""

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels,
                           device=coords.device, dtype=coords.dtype)

    def sample(self, shape, device=None, dtype=None):
        return torch.randn(*shape, device=device, dtype=dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256,
                 lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega",
            torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat,
                              device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)

    def sample(self, shape, device=None, dtype=None):
        # Fallback for dense-grid callers that don't carry coords.
        return torch.randn(*shape, device=device, dtype=dtype)


# Deterministic Fourier positional encoding.
class FourierPositionalEncoding(nn.Module):
    """Encode coordinates with sine-cosine frequency bands.

    The output dimension is ``2 * coord_dim * num_bands``.
    """

    def __init__(self, coord_dim: int, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.out_dim = coord_dim * num_bands * 2
        freqs = torch.linspace(1.0, max_freq / 2.0, num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0   # [-1, 1]
        x = coords.unsqueeze(-1) * self.freqs * math.pi      # [..., D, F]
        enc = torch.cat([x.sin(), x.cos()], dim=-1)          # [..., D, 2F]
        return enc.reshape(*coords.shape[:-1], self.out_dim)


# MLP-RBF backbone.
def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)

# GL-RBF gather helpers.
def batched_gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M] using idx with shape [B, N, K].
    Returns shape [B, N, K].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]


def batched_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M, C] using idx with shape [B, N, K].
    Returns shape [B, N, K, C].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]

class ConditionalPointFFM(nn.Module):
    """Point-cloud backbone with field-aware sparse observations."""
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(coord_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    # RBF attention over sensors: softmax of negative squared distance, so
    # nearby sensors dominate each query point and distant ones are ignored.
    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)          # zero padded rows

        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


class ConditionalPointMLPRBF(nn.Module):
    """MLP backbone with RBF sensor aggregation and global pooling."""
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(coord_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Embed the physical field identity for each sparse sensor.
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)

        # Encode sparse sensor tokens.
        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        # RBF weighting from each query point to each sparse sensor.
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


# Perceiver backbone.
class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block used after attention.
    """
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with residual connection and FFN.

    q  : [B, Tq, D]
    kv : [B, Tk, D]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize queries and keys/values independently.
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        # key_padding_mask: True means "ignore this token".
        attn_out, _ = self.attn(
            q_in,
            kv_in,
            kv_in,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )

        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class SelfAttentionBlock(nn.Module):
    """
    Standard latent self-attention block with residual connection and FFN.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.norm_attn(x)
        attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class ConditionalPointPerceiver(nn.Module):
    """Perceiver backbone for conditional point-cloud velocity prediction."""
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 8,
        num_latent_blocks: int = 4,
        field_embed_dim: int = 32,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        decode_chunk_size: Optional[int] = 4096,
        share_query_proj: bool = False,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.decode_chunk_size = decode_chunk_size
        self.use_fourier_pe = bool(use_fourier_pe)
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if self.use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim

        # Field-id embedding lets the model know which physical quantity
        # each sparse sensor measures.
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Query-state token = [coords, x_t, t]
        self.query_in_proj = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Sparse sensor token = [obs_coords, obs_value, field_embedding]
        self.sensor_proj = make_mlp(
            in_dim=self.coord_feat_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Decoder queries can either share or not share the encoder projection.
        if share_query_proj:
            self.query_out_proj = self.query_in_proj
        else:
            self.query_out_proj = make_mlp(
                in_dim=self.coord_feat_dim + n_fields + 1,
                hidden_dim=latent_dim,
                out_dim=latent_dim,
                depth=3,
            )

        # Learned latent array used by the Perceiver bottleneck.
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Encoder: latents attend to all input tokens.
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Latent processing blocks.
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Decoder: output query points attend to latent memory.
        self.output_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Final pointwise velocity head.
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(latent_dim, n_fields),
        )

    def _build_query_tokens(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        proj: nn.Module,
    ) -> torch.Tensor:
        """
        Build per-point query tokens from coordinates, current field state, and flow time.
        """
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        token_in = torch.cat([coord_feat, x_t, t_feat], dim=-1)
        return proj(token_in)

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build sparse sensor tokens from:
          - sensor location
          - observed scalar value
          - field-id embedding
        """
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)
        field_feat = field_feat * obs_mask.unsqueeze(-1)

        obs_coord_feat = self.pos_enc(obs_coords) if self.pos_enc is not None else obs_coords
        sensor_in = torch.cat([obs_coord_feat, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_proj(sensor_in)

        # Zero padded sensor slots so they do not inject junk features.
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        query_tokens: torch.Tensor,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all input information into the latent bottleneck.

        query_tokens : [B, N, D]
        sensor_tokens: [B, M, D]
        obs_mask     : [B, M]
        """
        bsz, n_query, _ = query_tokens.shape

        # Concatenate query-state tokens and sparse sensor tokens.
        input_tokens = torch.cat([query_tokens, sensor_tokens], dim=1)  # [B, N+M, D]

        # Query tokens are always valid; only sensor tokens may be padded.
        query_keep_mask = torch.zeros(
            bsz, n_query, device=query_tokens.device, dtype=torch.bool
        )
        sensor_padding_mask = ~obs_mask.bool()
        kv_padding_mask = torch.cat([query_keep_mask, sensor_padding_mask], dim=1)

        # Expand learned latent array across the batch.
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)

        # Encode into latents.
        latents = self.input_cross_attn(
            q=latents,
            kv=input_tokens,
            kv_padding_mask=kv_padding_mask,
        )

        # Continue processing in latent space.
        for block in self.latent_blocks:
            latents = block(latents)

        return latents

    def _decode_queries_chunked(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Decode pointwise outputs in bounded-size query chunks."""
        n_pts = coords.shape[1]

        if self.decode_chunk_size is None or n_pts <= self.decode_chunk_size:
            query_tokens = self._build_query_tokens(t, x_t, coords, self.query_out_proj)
            decoded = self.output_cross_attn(q=query_tokens, kv=latents, kv_padding_mask=None)
            return self.head(decoded)

        outputs = []
        for start in range(0, n_pts, self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, n_pts)

            coords_chunk = coords[:, start:end]
            x_t_chunk = x_t[:, start:end]

            query_tokens = self._build_query_tokens(t, x_t_chunk, coords_chunk, self.query_out_proj)
            decoded = self.output_cross_attn(q=query_tokens, kv=latents, kv_padding_mask=None)
            outputs.append(self.head(decoded))

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Build query-state tokens for the encoder.
        query_tokens = self._build_query_tokens(t, x_t, coords, self.query_in_proj)

        # Build sparse sensor tokens.
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )

        # Encode all information into latent memory.
        latents = self._encode_latents(
            query_tokens=query_tokens,
            sensor_tokens=sensor_tokens,
            obs_mask=obs_mask,
        )

        # Decode the per-point velocity field from latent memory.
        return self._decode_queries_chunked(
            latents=latents,
            t=t,
            x_t=x_t,
            coords=coords,
        )


# Global-local backbone.
class ConditionalPointHybridLocalGlobalRBF(nn.Module):
    """Hybrid latent-global and RBF-local point-cloud backbone.

    Sensor tokens are encoded through latent attention and gathered at query
    points with dense or top-k RBF variants. Neighbor search supports torch and
    optional KeOps backends.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",   # ["cls", "mean"]

        gather_mode: str = "rbf",    # ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        fieldwise_rbf_gather: bool = False,
        rbf_sigma_per_field: Optional[Sequence[float]] = None,
        periodic_coord_periods: Optional[Sequence[float]] = None,
        adaptive_rbf_sigma: bool = False,
        adaptive_rbf_scale: float = 1.0,
        neighbor_backend: str = "torch",      # ["auto", "torch", "keops"]

        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,

        use_fourier_pe: bool = False,
        pe_num_bands: int = 32,
        pe_max_freq: float = 64.0,

        # Enhanced GL-RBF options.
        enhanced_backbone: bool = False,
        sensor_coord_encoding: str = "raw",
        latent_sensor_reinject: bool = False,
        latent_reinject_every: int = 1,
        query_latent_readout: bool = False,
        query_readout_type: str = "point",
        query_readout_scale_init: float = 0.0,
        enhanced_head_norm: bool = False,
        glres_scale_init: float = 0.0,

        # Generating-parameter conditioning.
        n_params: int = 0,                                   # 0 disables the whole path
        param_log_mask: Optional[Sequence[bool]] = None,
        param_mu: Optional[Sequence[float]] = None,
        param_sigma: Optional[Sequence[float]] = None,
        param_n_freq: int = 4,
        param_jitter: float = 0.1,
        param_dropout: float = 0.1,
        param_embed_hidden: int = 128,
    ) -> None:
        super().__init__()

        if summary_type not in ["cls", "mean"]:
            raise ValueError(f"summary_type must be 'cls' or 'mean', got {summary_type}")
        if sensor_coord_encoding not in ["raw", "fourier"]:
            raise ValueError(
                f"sensor_coord_encoding must be one of ['raw', 'fourier'], got {sensor_coord_encoding}"
            )
        if query_readout_type not in ["point", "coord"]:
            raise ValueError(
                f"query_readout_type must be one of ['point', 'coord'], got {query_readout_type}"
            )
        if latent_reinject_every < 1:
            raise ValueError(f"latent_reinject_every must be >= 1, got {latent_reinject_every}")

        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.summary_type = summary_type

        if gather_mode not in ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]:
            raise ValueError(
                f"gather_mode must be one of ['rbf', 'topk_rbf', 'topk_rbf_gate', 'topk_rbf_ptlocal', 'topk_rbf_glres'], got {gather_mode}"
            )
        if neighbor_backend not in ["auto", "torch", "keops"]:
            raise ValueError(
                f"neighbor_backend must be one of ['auto', 'torch', 'keops'], got {neighbor_backend}"
            )
        self.gather_mode = gather_mode
        self.gather_topk = int(gather_topk)
        self.gather_query_chunk_size = gather_query_chunk_size
        self.learnable_rbf_sigma = learnable_rbf_sigma
        self.fieldwise_rbf_gather = bool(fieldwise_rbf_gather)
        self.adaptive_rbf_sigma = bool(adaptive_rbf_sigma)
        self.adaptive_rbf_scale = float(adaptive_rbf_scale)
        self.neighbor_backend = neighbor_backend

        if periodic_coord_periods is None:
            periodic_coord_periods = [0.0] * int(coord_dim)
        periodic_coord_periods = [float(v) for v in periodic_coord_periods]
        if len(periodic_coord_periods) != int(coord_dim):
            raise ValueError(
                "periodic_coord_periods must have one entry per coordinate "
                f"dimension ({coord_dim}); use 0 for non-periodic dimensions, "
                f"got {periodic_coord_periods}.")
        if any(not math.isfinite(v) or v < 0.0
               for v in periodic_coord_periods):
            raise ValueError(
                "periodic_coord_periods entries must be finite and non-negative, "
                f"got {periodic_coord_periods}.")
        # Python metadata (rather than a persistent buffer) preserves strict
        # compatibility with existing checkpoints. Positive entries use the
        # exact minimum-image distance on that periodic axis; zero entries
        # retain ordinary Euclidean distance.
        self.periodic_coord_periods = tuple(periodic_coord_periods)

        if self.fieldwise_rbf_gather and self.adaptive_rbf_sigma:
            raise ValueError(
                "fieldwise_rbf_gather and adaptive_rbf_sigma are mutually "
                "exclusive: fieldwise gathering already supplies a normalized-coordinate "
                "bandwidth for each sensor lattice."
            )
        if self.fieldwise_rbf_gather and gather_mode in {
                "topk_rbf_gate", "topk_rbf_ptlocal"}:
            raise ValueError(
                "fieldwise_rbf_gather supports rbf, topk_rbf, and "
                "topk_rbf_glres. The gate/ptlocal modes add learned or "
                "field-blind neighbor reweighting that defeats the guaranteed "
                "per-field geometric interpolation."
            )

        if rbf_sigma_per_field is not None:
            rbf_sigma_per_field = [float(v) for v in rbf_sigma_per_field]
            if not self.fieldwise_rbf_gather:
                raise ValueError(
                    "rbf_sigma_per_field requires fieldwise_rbf_gather=true."
                )
            if len(rbf_sigma_per_field) != int(n_fields):
                raise ValueError(
                    "rbf_sigma_per_field must have one positive value per "
                    f"model field ({n_fields}), got {rbf_sigma_per_field}."
                )
            if any(not math.isfinite(v) or v <= 0.0
                   for v in rbf_sigma_per_field):
                raise ValueError(
                    "rbf_sigma_per_field values must all be finite and positive, got "
                    f"{rbf_sigma_per_field}."
                )
        elif self.fieldwise_rbf_gather:
            rbf_sigma_per_field = [float(rbf_sigma)] * int(n_fields)

        if self.fieldwise_rbf_gather:
            if not math.isfinite(float(rbf_sigma)) or float(rbf_sigma) <= 0.0:
                raise ValueError(
                    f"rbf_sigma must be finite and positive, got {rbf_sigma}.")
            # Store only dimensionless coordinate-scale ratios and keep them out
            # of state_dict. This lets legacy scalar-sigma checkpoints strict-
            # load into the opt-in fieldwise gather for controlled ablations.
            self.register_buffer(
                "_fieldwise_rbf_scale",
                torch.tensor(rbf_sigma_per_field, dtype=torch.float32)
                / float(rbf_sigma),
                persistent=False,
            )

        if self.gather_mode == "rbf": print(f"\nThe gather mode is {gather_mode} as default choice.\n")
        else: print(f"\nNOTICE: The gather mode is {gather_mode} with top-k {gather_topk} !!!\n")

        # Build the query-side gate only for gated aggregation.
        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(hidden_dim, cond_dim, bias=False)

            # Scalar query-neighbor reweighting.
            gate_in_dim = cond_dim + cond_dim + coord_dim + 1
            self.gather_gate = nn.Sequential(
                nn.Linear(gate_in_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )

        if self.gather_topk < 1:
            raise ValueError(f"gather_topk must be >= 1, got {self.gather_topk}")
        # Optional learnable locality scale.
        if learnable_rbf_sigma:
            self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(float(rbf_sigma))))

        self.sensor_local_topk = int(sensor_local_topk)
        self.sensor_local_dropout_p = float(sensor_local_dropout)

        if self.sensor_local_topk < 1:
            raise ValueError(f"sensor_local_topk must be >= 1, got {self.sensor_local_topk}")

        # Fourier features do not affect coordinate-space neighbor distances.
        self.use_fourier_pe = bool(use_fourier_pe)
        if self.use_fourier_pe:
            self.pos_enc = FourierPositionalEncoding(
                coord_dim=coord_dim,
                num_bands=pe_num_bands,
                max_freq=pe_max_freq,
            )
            coord_feat_dim = self.pos_enc.out_dim
        else:
            self.pos_enc = None
            coord_feat_dim = coord_dim
        self.coord_feat_dim = coord_feat_dim

        # Enhanced options default to the base checkpoint schema.
        self.enhanced_backbone = bool(enhanced_backbone)
        self.sensor_coord_encoding = sensor_coord_encoding
        self.latent_sensor_reinject = bool(latent_sensor_reinject)
        self.latent_reinject_every = int(latent_reinject_every)
        self.query_latent_readout_enabled = bool(query_latent_readout)
        self.query_readout_type = query_readout_type
        self.enhanced_head_norm = bool(enhanced_head_norm)

        # Point/query branch
        # Query point token from [pos_feat(coords), x_t, t]
        self.point_encoder = make_mlp(
            in_dim=coord_feat_dim + n_fields + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
        )

        # Sparse sensor branch
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Sensor token: position, value, and field embedding.
        sensor_coord_dim = (
            coord_feat_dim
            if sensor_coord_encoding == "fourier" and self.pos_enc is not None
            else coord_dim
        )
        self.sensor_in_proj = make_mlp(
            in_dim=sensor_coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Project sensor tokens to the RBF-gather width.
        self.sensor_out_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=cond_dim,
            out_dim=cond_dim,
            depth=2,
        )

        # Local sensor refinement operates at cond_dim.
        if self.gather_mode == "topk_rbf_ptlocal":
            self.sensor_local_q = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_k = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_v = nn.Linear(cond_dim, cond_dim, bias=False)
            # Relative position encoding: [dx, dy, dz, ||d||]
            self.sensor_local_pos = make_mlp(
                in_dim=coord_dim + 1,
                hidden_dim=cond_dim,
                out_dim=cond_dim,
                depth=2,
            )
            # Scalar attention over local neighbors.
            self.sensor_local_attn = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_local_out = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_dropout = nn.Dropout(sensor_local_dropout)
            self.sensor_local_norm = nn.LayerNorm(cond_dim)

        # Optional query-to-latent readout; glres enables it automatically.
        self.use_query_latent_readout = self.query_latent_readout_enabled or self.gather_mode == "topk_rbf_glres"
        if self.use_query_latent_readout:
            # Read queries from L latents.
            if self.query_readout_type == "coord":
                # Coordinate decoder tokens.
                self.query_decoder_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
                self.query_readout_in = nn.Linear(self.coord_feat_dim + hidden_dim, latent_dim, bias=False)
            else:
                self.query_decoder_token = None
                self.query_readout_in = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.query_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.query_readout_out = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.query_readout_scale = nn.Parameter(torch.tensor(float(query_readout_scale_init)))

        if self.gather_mode == "topk_rbf_glres":
            # Pointwise coarse residual from the latent summary.
            self.coarse_film = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.coarse_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_dim, n_fields),
            )
            self.coarse_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))

            # Per-sensor bias for top-k gathering.
            self.sensor_importance = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_importance_scale = nn.Parameter(torch.tensor(float(glres_scale_init)))
            if self.fieldwise_rbf_gather:
                # Fieldwise local weights are intentionally pure RBF geometry.
                # Keep these tensors in state_dict for strict checkpoint
                # compatibility, but mark the dead branch non-trainable (also
                # safe for DDP with find_unused_parameters=False).
                self.sensor_importance_scale.requires_grad_(False)
                for parameter in self.sensor_importance.parameters():
                    parameter.requires_grad_(False)

        # Latent global processor
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Latents attend to sparse sensor tokens
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Process latents in latent space
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Refine sensor tokens from processed latents.
        self.sensor_back_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Global latent summary.
        self.summary_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
        )

        # Final velocity head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )
        # Enhanced mode normalizes the fused [query, global, local] head input.
        head_in_dim = hidden_dim + hidden_dim + cond_dim
        self.head_in_norm = nn.LayerNorm(head_in_dim) if enhanced_head_norm else nn.Identity()

        # Zero-initialized generating-parameter conditioning.
        self.n_params = int(n_params)
        self.param_n_freq = int(param_n_freq)
        self.param_jitter = float(param_jitter)
        self.param_dropout = float(param_dropout)
        if self.n_params > 0:
            if param_mu is None or param_sigma is None:
                raise ValueError(
                    "n_params > 0 requires param_mu and param_sigma (the TRAIN-split "
                    "standardization of the transformed parameters). They are registered "
                    "as buffers so a checkpoint is self-describing at inference.")
            mu_t = torch.as_tensor(param_mu, dtype=torch.float32).reshape(-1)
            sd_t = torch.as_tensor(param_sigma, dtype=torch.float32).reshape(-1)
            if mu_t.numel() != self.n_params or sd_t.numel() != self.n_params:
                raise ValueError(
                    f"param_mu/param_sigma must each have n_params={self.n_params} "
                    f"entries, got {mu_t.numel()}/{sd_t.numel()}")
            if bool((sd_t <= 0).any()):
                raise ValueError(f"param_sigma must be strictly positive, got {sd_t.tolist()}")
            log_mask = ([False] * self.n_params if param_log_mask is None
                        else [bool(b) for b in param_log_mask])
            if len(log_mask) != self.n_params:
                raise ValueError(
                    f"param_log_mask must have n_params={self.n_params} entries, "
                    f"got {len(log_mask)}")
            self.register_buffer("param_mu", mu_t)
            self.register_buffer("param_sigma", sd_t)
            self.register_buffer("param_log_mask", torch.tensor(log_mask, dtype=torch.bool))
            # Unknown token for parameter-slot dropout.
            self.param_null = nn.Parameter(torch.zeros(self.n_params))
            in_dim = self.n_params * (1 + 2 * self.param_n_freq)
            self.param_mlp = nn.Sequential(
                nn.Linear(in_dim, param_embed_hidden),
                nn.SiLU(),
                nn.Linear(param_embed_hidden, hidden_dim),
            )
            nn.init.zeros_(self.param_mlp[-1].weight)
            nn.init.zeros_(self.param_mlp[-1].bias)
            # Bounded Fourier bands for generating parameters.
            self.register_buffer(
                "param_freqs",
                2.0 ** torch.arange(self.param_n_freq, dtype=torch.float32) * math.pi)

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build sensor tokens from position, value, and field identity."""
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        field_feat = field_feat * obs_mask.unsqueeze(-1)             # zero padded rows

        # Optionally share Fourier coordinates with query tokens.
        if self.sensor_coord_encoding == "fourier" and self.pos_enc is not None:
            sensor_coord_feat = self.pos_enc(
                self._positional_coordinates(obs_coords))
        else:
            sensor_coord_feat = obs_coords

        sensor_in = torch.cat([sensor_coord_feat, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_in_proj(sensor_in)               # [B, M, D]
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sparse sensors into the learned latent array."""
        bsz = sensor_tokens.shape[0]

        # Expand learned latents across the batch.
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)      # [B, L, D]

        # True masks padded sensor tokens.
        sensor_padding_mask = ~obs_mask.bool()

        # Latents attend to sparse sensor tokens
        latents = self.input_cross_attn(
            q=latents,
            kv=sensor_tokens,
            kv_padding_mask=sensor_padding_mask,
        )

        # Optionally re-read sensors between latent blocks.
        for i, block in enumerate(self.latent_blocks):
            if (
                self.latent_sensor_reinject
                and i > 0
                and i % self.latent_reinject_every == 0
            ):
                # Reinject sensors at cost O(L*M).
                latents = self.input_cross_attn(
                    q=latents,
                    kv=sensor_tokens,
                    kv_padding_mask=sensor_padding_mask,
                )
            latents = block(latents)

        return latents

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent array into one global summary vector.

        If summary_type == 'cls', the last latent slot is treated as the summary token.
        If summary_type == 'mean', use the mean of all latent slots.
        """
        if self.summary_type == "cls":
            summary = latents[:, -1]         # [B, D]
        else:
            summary = latents.mean(dim=1)    # [B, D]

        return self.summary_proj(summary)    # [B, H]

    def _encode_params(self, params: torch.Tensor) -> torch.Tensor:
        """Encode raw physical parameters as ``[B, hidden_dim]`` features.

        Log-scaled slots are transformed and standardized before optional
        training jitter, slot dropout, and Fourier encoding.
        """
        if self.n_params <= 0:
            raise RuntimeError("_encode_params called on a model built with n_params=0")
        if params.dim() != 2 or params.shape[-1] != self.n_params:
            raise ValueError(
                f"params must be [B, {self.n_params}], got {tuple(params.shape)}")
        z = params.to(dtype=self.param_mu.dtype)
        if bool(self.param_log_mask.any()):
            if bool((z[:, self.param_log_mask] <= 0).any()):
                raise ValueError(
                    "a log-scaled parameter slot received a non-positive value; "
                    "params must be RAW PHYSICAL units (the model applies log10 and "
                    "standardization itself).")
            z = torch.where(self.param_log_mask, torch.log10(z.clamp_min(1e-30)), z)
        z = (z - self.param_mu) / self.param_sigma

        if self.training:
            if self.param_jitter > 0:
                z = z + self.param_jitter * torch.randn_like(z)
            if self.param_dropout > 0:
                drop = torch.rand_like(z) < self.param_dropout
                z = torch.where(drop, self.param_null.expand_as(z), z)

        feats = [z]
        if self.param_n_freq > 0:
            ang = z.unsqueeze(-1) * self.param_freqs            # [B, P, F]
            feats.append(ang.sin().flatten(1))
            feats.append(ang.cos().flatten(1))
        return self.param_mlp(torch.cat(feats, dim=-1))          # [B, hidden_dim]

    def null_param_embedding(self, bsz: int, device=None, dtype=None) -> torch.Tensor:
        """Return the embedding used when all parameter slots are unknown."""
        z = self.param_null.expand(bsz, self.n_params)
        feats = [z]
        if self.param_n_freq > 0:
            ang = z.unsqueeze(-1) * self.param_freqs
            feats.append(ang.sin().flatten(1))
            feats.append(ang.cos().flatten(1))
        out = self.param_mlp(torch.cat(feats, dim=-1))
        if device is not None or dtype is not None:
            out = out.to(device=device, dtype=dtype)
        return out

    def _use_keops(self) -> bool:
        """
        Decide whether to use KeOps.

        - rbf mode can benefit a lot from KeOps soft reductions
        - topk modes can use KeOps KNN search
        """
        if self.neighbor_backend == "torch":
            return False

        if self.neighbor_backend == "keops":
            if LazyTensor is None:
                raise ImportError(
                    "neighbor_backend='keops' was requested, but pykeops is not installed."
                )
            return True

        # auto
        return LazyTensor is not None

    def _minimum_image_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Wrap tensor coordinate differences into each periodic half-cell."""
        if not any(period > 0.0 for period in self.periodic_coord_periods):
            return delta
        parts = []
        for dim, period in enumerate(self.periodic_coord_periods):
            value = delta[..., dim:dim + 1]
            if period > 0.0:
                half_period = 0.5 * period
                value = torch.remainder(value + half_period, period) - half_period
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def _pairwise_sqdist_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise squared distance with exact periodic minimum images.

        The all-Euclidean fast path deliberately remains the historical
        ``torch.cdist`` expression so existing configurations retain their
        original numerical behavior.
        """
        if not any(period > 0.0 for period in self.periodic_coord_periods):
            return torch.cdist(query_coords, obs_coords, p=2.0).square()
        d2 = query_coords.new_zeros(
            query_coords.shape[0], query_coords.shape[1], obs_coords.shape[1])
        for dim, period in enumerate(self.periodic_coord_periods):
            delta = (query_coords[..., dim].unsqueeze(2)
                     - obs_coords[..., dim].unsqueeze(1))
            if period > 0.0:
                half_period = 0.5 * period
                delta = torch.remainder(
                    delta + half_period, period) - half_period
            d2 = d2 + delta.square()
        return d2

    def _pairwise_sqdist_keops(self, x_i, y_j):
        """KeOps counterpart of :meth:`_pairwise_sqdist_torch`."""
        delta = x_i - y_j
        if not any(period > 0.0 for period in self.periodic_coord_periods):
            return (delta ** 2).sum(-1)
        d2 = None
        for dim, period in enumerate(self.periodic_coord_periods):
            component = delta[dim]
            if period > 0.0:
                # LazyTensor.mod(P, -P/2) maps to [-P/2, P/2), the exact
                # minimum-image displacement even when inputs differ by more
                # than one period.
                component = component.mod(period, -0.5 * period)
            term = component ** 2
            d2 = term if d2 is None else d2 + term
        return d2

    def _positional_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """Scale periodic axes so Fourier features use the configured period."""
        if not any(period > 0.0 for period in self.periodic_coord_periods):
            return coords
        return torch.cat([
            coords[..., dim:dim + 1] / period
            if period > 0.0 else coords[..., dim:dim + 1]
            for dim, period in enumerate(self.periodic_coord_periods)
        ], dim=-1)

    def _aggregate_rbf_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full RBF gather using KeOps sumsoftmaxweight, without building the dense [B, N, M] matrix.
        """
        if sigma is None:
            sigma = (torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
                     if self.learnable_rbf_sigma else self.rbf_sigma)

        # KeOps full softmax masking needs a large finite logit penalty. Promote
        # half/bfloat16 inputs so that 1e6 stays finite (0 * inf would otherwise
        # produce NaNs on valid mask entries under mixed precision).
        output_dtype = refined_sensor_feat.dtype
        compute_dtype = query_coords.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        query_coords = query_coords.to(dtype=compute_dtype)
        obs_coords = obs_coords.to(dtype=compute_dtype)
        refined_sensor_feat = refined_sensor_feat.to(dtype=compute_dtype)
        sigma_t = torch.as_tensor(
            sigma, device=query_coords.device, dtype=compute_dtype)
        gamma = 1.0 / (2 * sigma_t ** 2 + 1e-12)

        # KeOps requires contiguous inputs.
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        refined_sensor_feat = refined_sensor_feat.contiguous()

        # KeOps symbolic tensors
        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]
        v_j = LazyTensor(refined_sensor_feat[:, None, :, :])          # [B, 1, M, Cc]

        # Scalar logits: -gamma * ||x_i - y_j||^2
        sqdist_ij = self._pairwise_sqdist_keops(x_i, y_j)             # [B, N, M, 1]
        logits_ij = -gamma * sqdist_ij

        # Mask invalid sensor slots below every valid normalized-coordinate
        # logit. A fixed -1e6 is insufficient when sigma becomes very small,
        # because valid -d^2/(2 sigma^2) values can be even more negative.
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())   # [B, 1, M, 1]
        sigma_for_mask = sigma_t.detach()
        invalid_penalty = 1e6 + 1e6 / (2 * sigma_for_mask ** 2 + 1e-12)
        logits_ij = logits_ij + (mask_j - 1.0) * invalid_penalty

        # Softmax-weighted sum over the sensor axis.
        # With one batch dimension, the j-axis is dim=2.
        local_cond = logits_ij.sumsoftmaxweight(v_j, dim=2)           # [B, N, Cc]
        return local_cond.to(dtype=output_dtype)

    def _knn_search_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-k neighbor search using KeOps Kmin_argKmin.
        """

        # Distance search must remain finite for explicit half-precision
        # coordinates. Both the 1e6 invalid-slot sentinel and squared
        # distances can overflow fp16.
        obs_coords_out = obs_coords
        distance_dtype = query_coords.dtype
        if distance_dtype in (torch.float16, torch.bfloat16):
            distance_dtype = torch.float32
        query_coords_dist = query_coords.to(dtype=distance_dtype).contiguous()
        obs_coords_dist = obs_coords.to(dtype=distance_dtype).contiguous()

        x_i = LazyTensor(query_coords_dist[:, :, None, :])            # [B, N, 1, D]
        y_j = LazyTensor(obs_coords_dist[:, None, :, :])              # [B, 1, M, D]

        sqdist_ij = self._pairwise_sqdist_keops(x_i, y_j)             # [B, N, M, 1]

        # Mask invalid sensor slots
        mask_j = LazyTensor(
            obs_mask[:, None, :, None].to(distance_dtype).contiguous())
        sqdist_ij = sqdist_ij + (1.0 - mask_j) * 1e6

        # With one batch dimension, the j-axis is dim=2.
        topk_d2, topk_idx = sqdist_ij.Kmin_argKmin(K=k, dim=2)

        # KeOps can return indices in a non-long dtype; convert explicitly.
        topk_idx = topk_idx.long()

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords_out, topk_idx)
        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()

        return topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _knn_search_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback KNN search using torch.cdist + torch.topk.
        """
        obs_coords_out = obs_coords
        distance_dtype = query_coords.dtype
        if distance_dtype in (torch.float16, torch.bfloat16):
            distance_dtype = torch.float32
        query_coords_dist = query_coords.to(dtype=distance_dtype)
        obs_coords_dist = obs_coords.to(dtype=distance_dtype)
        d2 = self._pairwise_sqdist_torch(query_coords_dist, obs_coords_dist)
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        topk_d2, topk_idx = torch.topk(d2, k=k, dim=-1, largest=False)

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords_out, topk_idx)
        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()

        return topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _get_topk_neighbors(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified top-k neighbor retrieval.
        """
        if self._use_keops():
            return self._knn_search_keops(
                query_coords=query_coords,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                k=k,
            )

        return self._knn_search_torch(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

    def _sensor_local_refine(
        self,
        sensor_coords: torch.Tensor,      # [B, M, D]
        sensor_feat: torch.Tensor,        # [B, M, Cc]
        obs_mask: torch.Tensor,           # [B, M]
    ) -> torch.Tensor:
        """Refine sensor tokens over a local Point-Transformer graph.

        Neighbor search requests one extra entry and removes the sensor's own
        entry before attention.
        """
        # Search one extra neighbor and discard the self-neighbor.
        k_search = min(self.sensor_local_topk + 1, sensor_coords.shape[1])

        nbr_d2, nbr_feat, nbr_coords, nbr_valid = self._get_topk_neighbors(
            query_coords=sensor_coords,
            obs_coords=sensor_coords,
            refined_sensor_feat=sensor_feat,
            obs_mask=obs_mask,
            k=k_search,
        )

        # Drop the first neighbor slot, which is typically the point itself.
        if k_search > 1:
            nbr_d2 = nbr_d2[:, :, 1:]
            nbr_feat = nbr_feat[:, :, 1:]
            nbr_coords = nbr_coords[:, :, 1:]
            nbr_valid = nbr_valid[:, :, 1:]

        # If there was only one valid sensor total, keep the feature unchanged.
        if nbr_feat.shape[2] == 0:
            return sensor_feat

        q = self.sensor_local_q(sensor_feat).unsqueeze(2)   # [B, M, 1, Cc]
        k = self.sensor_local_k(nbr_feat)                   # [B, M, Ks, Cc]
        v = self.sensor_local_v(nbr_feat)                   # [B, M, Ks, Cc]

        rel = self._minimum_image_delta(
            sensor_coords.unsqueeze(2) - nbr_coords)        # [B, M, Ks, D]
        rel_dist = torch.sqrt(nbr_d2.clamp_min(0.0)).unsqueeze(-1)  # [B, M, Ks, 1]
        pos = self.sensor_local_pos(torch.cat([rel, rel_dist], dim=-1))  # [B, M, Ks, Cc]

        # Lightweight Point-Transformer-style attention:
        # attention is driven by query-key difference plus relative position.
        attn_logits = self.sensor_local_attn(torch.tanh(q - k + pos)).squeeze(-1)  # [B, M, Ks]
        attn_logits = attn_logits.masked_fill(~nbr_valid, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        update = torch.sum(attn.unsqueeze(-1) * (v + pos), dim=2)       # [B, M, Cc]
        out = self.sensor_local_norm(sensor_feat + self.sensor_local_dropout(self.sensor_local_out(update)))

        # Keep padded sensor rows zeroed out.
        out = out * obs_mask.unsqueeze(-1)
        return out

    def _build_query_readout_tokens(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Build point- or coordinate-based latent readout tokens."""
        if self.query_readout_type == "coord":
            bsz, n_query, _ = coords.shape
            coord_feat = (self.pos_enc(self._positional_coordinates(coords))
                          if self.pos_enc is not None else coords)
            dq = self.query_decoder_token.view(1, 1, -1).expand(bsz, n_query, -1)
            return self.query_readout_in(torch.cat([coord_feat, dq], dim=-1))
        return self.query_readout_in(point_feat)

    def _readout_query_global_chunked(
        self,
        point_feat: torch.Tensor,
        coords: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Read query-global features from latents in bounded-size chunks."""
        n_query = point_feat.shape[1]
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096

        if chunk_size is None or n_query <= chunk_size:
            q = self._build_query_readout_tokens(point_feat, coords)
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            return self.query_readout_out(readout)

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            q = self._build_query_readout_tokens(point_feat[:, start:end], coords[:, start:end])
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            outputs.append(self.query_readout_out(readout))
        return torch.cat(outputs, dim=1)

    def _predict_global_coarse(
        self,
        point_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the FiLM-conditioned coarse field for global-residual gather."""
        gamma, beta = self.coarse_film(global_feat).chunk(2, dim=-1)
        coarse_feat = (
            point_feat * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.coarse_head(coarse_feat)

    @staticmethod
    def _masked_neighbor_softmax(
        logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax over valid neighbors, returning zero for an empty field."""
        logits = logits.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        # An observation batch may omit a physical field. softmax(all -inf) is
        # NaN, so explicitly make that field contribute a zero feature.
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights * valid.to(dtype=weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fieldwise_sigma(self, field_id: int) -> torch.Tensor:
        """Return one field's RBF width in normalized model coordinates."""
        if not self.fieldwise_rbf_gather:
            raise RuntimeError("_fieldwise_sigma requires fieldwise_rbf_gather")
        if self.learnable_rbf_sigma:
            base = torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
        else:
            base = self._fieldwise_rbf_scale.new_tensor(float(self.rbf_sigma))
        return (base * self._fieldwise_rbf_scale[field_id]).clamp_min(1e-6)

    def _aggregate_chunk_fieldwise(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate every physical field independently, then fuse.

        Mixed-resolution observations must not share a neighbor search or RBF
        normalization: otherwise the dense field takes spatially varying
        probability mass from the coarse fields. Here each field receives its
        own top-k set, bandwidth, and unit-mass softmax. The resulting features
        are combined by a balanced mean, independent of sensor count.
        """
        field_features = []
        field_present = []
        dense_d2 = None
        if self.gather_mode == "rbf" and not self._use_keops():
            # Full fieldwise RBF is intentionally evaluated with a shared dense
            # distance matrix. N19 uses top-k/KeOps; this path keeps the option
            # correct for small point-cloud configurations as well.
            distance_dtype = query_coords.dtype
            if distance_dtype in (torch.float16, torch.bfloat16):
                distance_dtype = torch.float32
            dense_d2 = self._pairwise_sqdist_torch(
                query_coords.to(distance_dtype),
                obs_coords.to(distance_dtype))

        for field_id in range(self.n_fields):
            field_mask = (
                obs_mask.bool() & (obs_field_ids.long() == int(field_id))
            )
            present = field_mask.any(dim=-1)
            sigma = self._fieldwise_sigma(field_id).to(
                device=query_coords.device, dtype=query_coords.dtype)

            if self.gather_mode == "rbf":
                if self._use_keops():
                    local = self._aggregate_rbf_keops(
                        query_coords=query_coords,
                        obs_coords=obs_coords,
                        refined_sensor_feat=refined_sensor_feat,
                        obs_mask=field_mask,
                        sigma=sigma,
                    )
                else:
                    valid = field_mask.unsqueeze(1).expand(
                        -1, query_coords.shape[1], -1)
                    logits = -dense_d2 / (2 * sigma.square() + 1e-12)
                    weights = self._masked_neighbor_softmax(logits, valid)
                    local = torch.einsum(
                        "bnm,bmd->bnd", weights,
                        refined_sensor_feat.to(weights.dtype)).to(
                            refined_sensor_feat.dtype)
            else:
                k = min(self.gather_topk, obs_coords.shape[1])
                topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid = \
                    self._get_topk_neighbors(
                        query_coords=query_coords,
                        obs_coords=obs_coords,
                        refined_sensor_feat=refined_sensor_feat,
                        obs_mask=field_mask,
                        k=k,
                    )
                sigma_compute = sigma.to(dtype=topk_d2.dtype)
                logits = -topk_d2 / (2 * sigma_compute.square() + 1e-12)

                if self.gather_mode == "topk_rbf_gate":
                    query_cond = self.query_to_cond(query_feat)
                    query_cond = query_cond.unsqueeze(2).expand(-1, -1, k, -1)
                    rel = self._minimum_image_delta(
                        query_coords.unsqueeze(2) - topk_sensor_coords)
                    rel_dist = torch.sqrt(
                        topk_d2.clamp_min(0.0)).unsqueeze(-1)
                    gate_in = torch.cat(
                        [query_cond, topk_sensor_feat, rel, rel_dist], dim=-1)
                    logits = logits + self.gather_gate(gate_in).squeeze(-1)

                # Deliberately omit topk_rbf_glres's learned per-sensor logit
                # bias here. Fieldwise mode keeps the local weights purely
                # geometric; glres retains its global residual paths.
                weights = self._masked_neighbor_softmax(logits, topk_valid)
                local = torch.sum(
                    weights.unsqueeze(-1) * topk_sensor_feat, dim=2).to(
                        dtype=topk_sensor_feat.dtype)

            local = local * present[:, None, None].to(dtype=local.dtype)
            field_features.append(local)
            field_present.append(present)

        # Missing fields remain zero and are excluded from the balanced mean.
        stacked = torch.stack(field_features, dim=2)
        present = torch.stack(field_present, dim=1)
        denom = present.sum(dim=1).clamp_min(1).to(stacked.dtype)
        balanced = stacked.sum(dim=2) / denom[:, None, None]
        return balanced

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,         # [B, Nc, D]
        query_feat: torch.Tensor,           # [B, Nc, H]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        obs_field_ids: Optional[torch.Tensor] = None,  # [B, M]
    ) -> torch.Tensor:
        """Aggregate one query chunk."""
        if self.fieldwise_rbf_gather:
            if obs_field_ids is None:
                raise ValueError(
                    "fieldwise_rbf_gather requires obs_field_ids for every sensor.")
            return self._aggregate_chunk_fieldwise(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )

        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma

        # Full RBF gather.
        if self.gather_mode == "rbf":
            if self._use_keops():
                return self._aggregate_rbf_keops(
                    query_coords=query_coords,
                    obs_coords=obs_coords,
                    refined_sensor_feat=refined_sensor_feat,
                    obs_mask=obs_mask,
                )

            distance_dtype = query_coords.dtype
            if distance_dtype in (torch.float16, torch.bfloat16):
                distance_dtype = torch.float32
            d2 = self._pairwise_sqdist_torch(
                query_coords.to(distance_dtype),
                obs_coords.to(distance_dtype))
            large = torch.full_like(d2, 1e6)
            d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

            logits = -d2 / (2 * sigma ** 2 + 1e-12)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum(
                "bnm,bmd->bnd", weights,
                refined_sensor_feat.to(weights.dtype)).to(
                    refined_sensor_feat.dtype)

        # Top-k gather modes.
        k = min(self.gather_topk, obs_coords.shape[1])

        topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid = self._get_topk_neighbors(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

        if self.adaptive_rbf_sigma:
            # Scale bandwidth by each query's farthest valid top-k neighbor.
            masked_d2 = torch.where(
                topk_valid, topk_d2, torch.zeros_like(topk_d2)
            )
            ref_d2 = masked_d2.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            sigma_q = ref_d2.sqrt() * self.adaptive_rbf_scale * sigma
            logits = -topk_d2 / (2 * sigma_q ** 2 + 1e-12)
        else:
            logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)

        if self.gather_mode == "topk_rbf_gate":
            query_cond = self.query_to_cond(query_feat)                    # [B, Nc, Cc]
            query_cond = query_cond.unsqueeze(2).expand(-1, -1, k, -1)    # [B, Nc, k, Cc]

            rel = self._minimum_image_delta(
                query_coords.unsqueeze(2) - topk_sensor_coords)            # [B, Nc, k, D]
            rel_dist = torch.sqrt(topk_d2.clamp_min(0.0)).unsqueeze(-1)    # [B, Nc, k, 1]

            gate_in = torch.cat([query_cond, topk_sensor_feat, rel, rel_dist], dim=-1)
            gate_logits = self.gather_gate(gate_in).squeeze(-1)            # [B, Nc, k]

            logits = logits + gate_logits

        if self.gather_mode == "topk_rbf_glres":
            # Pointwise sensor importance on gathered neighbors.
            topk_sensor_bias = self.sensor_importance(topk_sensor_feat).squeeze(-1)  # [B, Nc, k]
            logits = logits + self.sensor_importance_scale * topk_sensor_bias

        logits = logits.masked_fill(~topk_valid, -1e9)
        weights = torch.softmax(logits, dim=-1)
        local_cond = torch.sum(
            weights.unsqueeze(-1) * topk_sensor_feat, dim=2).to(
                dtype=topk_sensor_feat.dtype)
        return local_cond

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather enriched sensor features at query points."""
        n_query = query_coords.shape[1]

        if self.gather_mode == "topk_rbf_gate":
            # Gate tensors scale as [B,N,K,...].
            chunk_size = self.gather_query_chunk_size if self.gather_query_chunk_size is not None else 2048
        else:
            # Other modes use the configured chunk size.
            chunk_size = self.gather_query_chunk_size

        if chunk_size is None or n_query <= chunk_size:
            return self._aggregate_chunk(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)

            local_chunk = self._aggregate_chunk(
                query_coords=query_coords[:, start:end],
                query_feat=query_feat[:, start:end],
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )
            outputs.append(local_chunk)

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        params: optional raw physical parameters with shape [B, n_params].

        Output:
            velocity field of shape [B, N, C]
        """
        bsz, n_pts, _ = x_t.shape
        # Require parameter inputs and model configuration to agree.
        if params is not None and self.n_params <= 0:
            raise ValueError(
                "received params but this model was built with n_params=0; rebuild it "
                "with n_params/param_mu/param_sigma set, or stop passing params.")
        if params is None and self.n_params > 0:
            raise ValueError(
                f"this model was built with n_params={self.n_params} and needs params "
                "on every call; got None.")

        # Query-point features.
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = (self.pos_enc(self._positional_coordinates(coords))
                      if self.use_fourier_pe else coords)
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))  # [B, N, H]

        # Local sensor tokens.
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, M, D]

        # Global latent processing.
        latents = self._encode_latents(sensor_tokens=sensor_tokens, obs_mask=obs_mask)  # [B, L, D]

        # Refine sensor tokens against latent memory.
        refined_sensor_tokens = self.sensor_back_attn(
            q=sensor_tokens,
            kv=latents,
            kv_padding_mask=None,
        )  # [B, M, D]

        # Clear padded sensor rows after attention.
        refined_sensor_tokens = refined_sensor_tokens * obs_mask.unsqueeze(-1)

        # Project refined sensor tokens to the local conditioning width.
        refined_sensor_feat = self.sensor_out_proj(refined_sensor_tokens)   # [B, M, cond_dim]
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)

        # Global summary + optional per-query latent readout.
        global_feat = self._extract_global_summary(latents)                 # [B, H]
        if self.n_params > 0:
            # This global feature also feeds the coarse prediction path.
            global_feat = global_feat + self._encode_params(params)         # [B, H]
        if self.use_query_latent_readout:
            query_global = self._readout_query_global_chunked(point_feat, coords, latents)  # [B, N, H]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
        else:
            global_for_head = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)  # [B, N, H]

        # Global-residual gather.
        if self.gather_mode == "topk_rbf_glres":
            # Local top-k gather (sensor-importance bias is applied inside _aggregate_chunk).
            local_cond = self.aggregate_sparse_obs(
                query_coords=coords,
                query_feat=point_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
            )  # [B, N, cond_dim]

            # Add a learned coarse residual from the global summary.
            coarse_pred = self.coarse_scale * self._predict_global_coarse(point_feat, global_feat)

            head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
            residual = self.head(self.head_in_norm(head_in))
            return coarse_pred + residual

        # Optional sensor-side local graph refinement.
        if self.gather_mode == "topk_rbf_ptlocal":
            refined_sensor_feat = self._sensor_local_refine(
                sensor_coords=obs_coords,
                sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,)

        # Gather sensor features at query points.
        local_cond = self.aggregate_sparse_obs(
            query_coords=coords,
            query_feat=point_feat,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, N, cond_dim]

        # Final velocity prediction.
        head_in = torch.cat([point_feat, global_for_head, local_cond], dim=-1)
        out = self.head(self.head_in_norm(head_in))
        return out


# FNO backbone
class FNO(nn.Module):
    """Regular-grid FNO with rasterized sparse conditioning.

    Inputs use point layout and are permuted to row-major grids internally.
    """

    def __init__(
        self,
        n_fields: int,
        Num_x: int,
        Num_y: int,
        n_modes_x: int = 32,
        n_modes_y: int = 8,
        hidden_channels: int = 64,
        n_layers: int = 4,
        use_grid_positional_embedding: bool = True,
        condition_blur: bool = False,
        condition_blur_kernel: int = 5,
        condition_blur_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        if NeuralOpFNO is None:
            raise ImportError(
                "The FNO backbone requires the optional 'neuraloperator' package.")

        self.n_fields = n_fields
        self.Num_x = int(Num_x)
        self.Num_y = int(Num_y)
        self.condition_blur = bool(condition_blur)
        self.condition_blur_kernel = int(condition_blur_kernel)
        self.condition_blur_sigma = float(condition_blur_sigma)

        if self.condition_blur_kernel < 1 or self.condition_blur_kernel % 2 == 0:
            raise ValueError(
                f"condition_blur_kernel must be a positive odd integer, got {self.condition_blur_kernel}."
            )
        if self.condition_blur_sigma <= 0.0:
            raise ValueError(
                f"condition_blur_sigma must be > 0, got {self.condition_blur_sigma}."
            )

        # Non-persistent caches are rebuilt after loading or device changes.
        self.register_buffer("_condition_blur_kernel_cache", torch.empty(0), persistent=False)
        self.register_buffer("_grid_order_cache", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_point_to_grid_cache", torch.empty(0, dtype=torch.long), persistent=False)

        # FNO input channels:
        #   current state x_t         -> C
        #   scalar time channel       -> 1
        #   normalized observed values        -> C
        #   support-weighted observed values  -> C
        #   soft observation support maps     -> C
        # total = 4C + 1
        in_channels = 4 * n_fields + 1

        self.fno = NeuralOpFNO(
            n_modes=(n_modes_y, n_modes_x),   # tensor layout is [B, C, Num_y, Num_x]
            in_channels=in_channels,
            out_channels=n_fields,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid" if use_grid_positional_embedding else None,
        )

    def _get_grid_permutation(
        self,
        coords: torch.Tensor,
        decimals: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return maps between dataset point order and row-major grid order.

        grid_order[g] = original point index for row-major grid cell g.
        point_to_grid[p] = row-major grid cell for original point index p.

        The non-persistent cache assumes a fixed mesh for each model instance.
        """
        n_pts = coords.shape[1]
        expected = self.Num_x * self.Num_y
        if n_pts != expected:
            raise ValueError(
                f"FNO backbone expected N = Num_x * Num_y = {expected}, got {n_pts}."
            )

        cached_order = self._grid_order_cache
        cached_point_to_grid = self._point_to_grid_cache
        if (
            cached_order.numel() == n_pts
            and cached_point_to_grid.numel() == n_pts
            and cached_order.device == coords.device
            and cached_point_to_grid.device == coords.device
        ):
            return cached_order, cached_point_to_grid

        coords0 = coords[0, :, :2].detach()
        scale = float(10 ** decimals)
        x = torch.round(coords0[:, 0] * scale) / scale
        y = torch.round(coords0[:, 1] * scale) / scale

        unique_x, x_rank = torch.unique(x, sorted=True, return_inverse=True)
        unique_y, y_rank = torch.unique(y, sorted=True, return_inverse=True)
        if unique_x.numel() != self.Num_x or unique_y.numel() != self.Num_y:
            raise ValueError(
                "FNO could not infer the requested grid from coords: "
                f"detected unique (x, y)=({unique_x.numel()}, {unique_y.numel()}), "
                f"but expected (Num_x, Num_y)=({self.Num_x}, {self.Num_y})."
            )

        point_to_grid = (y_rank.long() * self.Num_x + x_rank.long()).contiguous()
        if torch.unique(point_to_grid).numel() != n_pts:
            raise ValueError(
                "FNO could not infer a complete tensor-product grid from coords. "
                "The coordinate set has duplicate or missing (x, y) grid cells, so "
                "an internal row-major permutation would be ambiguous."
            )

        grid_order = torch.argsort(point_to_grid).contiguous()
        self._grid_order_cache = grid_order
        self._point_to_grid_cache = point_to_grid
        return grid_order, point_to_grid

    def _get_condition_blur_kernel(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build and cache a depthwise 2D Gaussian kernel used to splat sparse
        conditioning impulses into a small local neighborhood.
        """
        kernel = self._condition_blur_kernel_cache
        if kernel.numel() > 0 and kernel.dtype == dtype and kernel.device == device:
            return kernel

        radius = self.condition_blur_kernel // 2
        coords_1d = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-0.5 * (coords_1d / self.condition_blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-12)

        kernel = kernel_2d.view(1, 1, self.condition_blur_kernel, self.condition_blur_kernel)
        kernel = kernel.expand(self.n_fields, 1, -1, -1).contiguous()
        self._condition_blur_kernel_cache = kernel
        return kernel

    def _blur_condition_maps(
        self,
        obs_value_maps: torch.Tensor,
        obs_mask_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace one-pixel conditioning maps with Gaussian splats.

        Returns three semantically different condition maps:
          - normalized/interpolated values on the original field scale;
          - support-weighted values, i.e. the blurred numerator;
          - soft support, i.e. the blurred mask.

        The support-weighted channel is deliberately separate from the
        normalized channel. It avoids presenting an unqualified finite-support
        normalized plateau/boundary to the spectral model, and lets the FNO
        distinguish "interpolated value" from "confidence/support."
        """
        kernel = self._get_condition_blur_kernel(
            dtype=obs_value_maps.dtype,
            device=obs_value_maps.device,
        )
        padding = self.condition_blur_kernel // 2

        blurred_mask_raw = F.conv2d(
            obs_mask_maps,
            kernel,
            padding=padding,
            groups=self.n_fields,
        )
        blurred_value_num = F.conv2d(
            obs_value_maps,
            kernel,
            padding=padding,
            groups=self.n_fields,
        )

        blurred_value_norm = blurred_value_num / blurred_mask_raw.clamp_min(1e-6)
        blurred_value_norm = torch.where(
            blurred_mask_raw > 0,
            blurred_value_norm,
            torch.zeros_like(blurred_value_norm),
        )
        return blurred_value_norm, blurred_value_num, blurred_mask_raw

    def _pointcloud_to_grid(
        self,
        x: torch.Tensor,
        grid_order: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert original point order [B, N, C] -> row-major grid [B, C, Num_y, Num_x].
        """
        bsz, n_pts, n_fields = x.shape
        expected = self.Num_x * self.Num_y
        if n_pts != expected:
            raise ValueError(
                f"FNO backbone expected N = Num_x * Num_y = {expected}, got {n_pts}."
            )

        x = x[:, grid_order, :]
        x_grid = x.reshape(bsz, self.Num_y, self.Num_x, n_fields)
        x_grid = x_grid.permute(0, 3, 1, 2).contiguous()
        return x_grid

    def _grid_to_pointcloud(
        self,
        x_grid: torch.Tensor,
        point_to_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert row-major grid [B, C, Num_y, Num_x] -> original point order [B, N, C].
        """
        bsz, n_fields, _, _ = x_grid.shape
        x = x_grid.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, self.Num_x * self.Num_y, n_fields)
        x = x[:, point_to_grid, :]
        return x

    def _build_condition_maps(
        self,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rasterize sparse observations into dense grid-aligned maps.

        Returns:
            obs_value_norm_maps    : [B, C, Num_y, Num_x]
            obs_value_weighted_maps: [B, C, Num_y, Num_x]
            obs_support_maps       : [B, C, Num_y, Num_x]
        """
        bsz, _, _ = obs_values.shape
        n_pts = self.Num_x * self.Num_y

        obs_value_maps = torch.zeros(
            bsz, self.n_fields, n_pts, dtype=dtype, device=device
        )
        obs_mask_maps = torch.zeros(
            bsz, self.n_fields, n_pts, dtype=dtype, device=device
        )

        # Scatter sparse sensor values into the appropriate field-channel grid.
        for b in range(bsz):
            valid = obs_mask[b].bool()
            if not valid.any():
                continue

            idx = obs_indices[b, valid].long()
            fld = obs_field_ids[b, valid].long()
            val = obs_values[b, valid, 0]

            obs_value_maps[b, fld, idx] = val
            obs_mask_maps[b, fld, idx] = 1.0

        obs_value_maps = obs_value_maps.reshape(bsz, self.n_fields, self.Num_y, self.Num_x)
        obs_mask_maps = obs_mask_maps.reshape(bsz, self.n_fields, self.Num_y, self.Num_x)

        if self.condition_blur:
            return self._blur_condition_maps(
                obs_value_maps=obs_value_maps,
                obs_mask_maps=obs_mask_maps,
            )

        # Without blur, a point observation is both the normalized value and
        # the support-weighted value, with the binary mask carrying support.
        return obs_value_maps, obs_value_maps, obs_mask_maps

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict the velocity field on the full regular grid.

        obs_indices is required because the sparse sensor values must be
        rasterized onto the fixed grid before being fed into the FNO.
        """
        if obs_indices is None:
            raise ValueError(
                "FNO.forward requires obs_indices so sparse observations can be "
                "placed onto the regular grid."
            )

        bsz = x_t.shape[0]
        grid_order, point_to_grid = self._get_grid_permutation(coords)

        # Convert the current state to a row-major grid. If the dataset is
        # stored in a different point order, the permutation is applied here
        # rather than mutating the shared point-cloud dataset.
        x_grid = self._pointcloud_to_grid(x_t, grid_order=grid_order)  # [B, C, Num_y, Num_x]
        # Broadcast time to a full grid channel.
        t_map = t.view(bsz, 1, 1, 1).expand(bsz, 1, self.Num_y, self.Num_x)

        # Convert sparse observations into dense field-aligned maps.
        obs_grid_indices = point_to_grid[obs_indices.long()]
        obs_value_norm_maps, obs_value_weighted_maps, obs_support_maps = self._build_condition_maps(
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_grid_indices,
            dtype=x_t.dtype,
            device=x_t.device,
        )

        # Concatenate:
        #   [current fields, time channel, normalized observed values,
        #    support-weighted observed values, soft support maps]
        fno_in = torch.cat(
            [x_grid, t_map, obs_value_norm_maps, obs_value_weighted_maps, obs_support_maps],
            dim=1,
        )
        # FNO predicts the velocity field on the regular grid.
        vel_grid = self.fno(fno_in)
        # Convert back to the standard point-cloud layout expected by the wrapper.
        vel = self._grid_to_pointcloud(vel_grid, point_to_grid=point_to_grid)
        return vel



# Flow-model wrappers.
class PointCloudFFM(nn.Module):
    """Rectified-flow wrapper for point-cloud velocity models."""
    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.prior = prior

        # Stored for checkpoint and configuration compatibility.
        self.sigma_min = sigma_min

    def sample_source(self, coords: torch.Tensor) -> torch.Tensor:
        """Draw the rectified-flow source endpoint from the configured prior."""
        return self.prior(coords, self.model.n_fields)

    def simulate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Interpolate as ``x_t = (1 - t) * x0 + t * x1``."""
        alpha = t.view(-1, 1, 1)
        return (1.0 - alpha) * x0 + alpha * x1

    def target_vector_field(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return the rectified-flow target velocity ``x1 - x0``."""
        return x1 - x0

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
        *,
        params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Sample x0 from the source prior for the current query coordinates.
        x0 = self.sample_source(coords)

        # Uniform time for standard 1-RF training.
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # Straight interpolation and constant target velocity.
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        # Predict the velocity under sparse conditioning.
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                          params=params)

        # Standard supervised regression loss used in 1-RF.
        loss = F.mse_loss(pred, target)

        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
        }

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 8,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the rectified-flow ODE from the source prior.

        ``params`` contains raw physical generating parameters and is required
        when the backbone was configured with parameter conditioning.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        bsz = coords.shape[0]
        x = self.sample_source(coords)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)
        if obs_consistency_mode != "none" and clamp_indices is None:
            if obs_consistency_mode in ("default_hard", "endpoint"):
                raise ValueError(
                    f"obs_consistency_mode={obs_consistency_mode!r} requires clamp_indices."
                )

        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(
            0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype
        )

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            # Velocity at the current state.
            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                            params=params)
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking: guide x1_hat, then
                # convert the consistent endpoint back to a velocity.
                v0 = apply_endpoint_observation_consistency(
                    x_t=x,
                    v=v0,
                    t=t0,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )

            if ode_solver == "heun":
                # Optional predictor-corrector step.
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids,
                                params=params)
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                    v1 = apply_endpoint_observation_consistency(
                        x_t=x_euler,
                        v=v1,
                        t=t1,
                        value_map=value_map,
                        mask_map=mask_map,
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                x = x + 0.5 * dt * (v0 + v1)
            else:
                # Default 1-RF benchmark solver.
                x = x + dt * v0

            # Hard consistency replaces observed entries after each step.
            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none" and clamp_indices is not None:
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x

class FNOFFM(PointCloudFFM):
    """Rectified-flow wrapper for a full-grid FNO backbone.

    ``obs_indices`` rasterizes sparse measurements into grid channels.
    """

    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__(model=model, prior=prior, sigma_min=sigma_min)
        self.requires_full_grid = True

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rectified-flow loss with rasterized sparse sensors."""
        if obs_indices is None:
            raise ValueError("FNOFFM.training_loss requires obs_indices.")

        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # RF source sample
        x0 = self.sample_source(coords)

        # Straight interpolation
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        pred = self.model(
            t=t,
            x_t=x_t,
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        loss = F.mse_loss(pred, target)
        return loss, {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 100,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
    ) -> torch.Tensor:
        """Sample with FNO conditioning and optional observation clamping.

        ``clamp_indices`` locates observations for rasterization and hard
        consistency. Euler uses one evaluation per step and Heun uses two.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if ode_solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported ode_solver={ode_solver!r}; expected 'euler' or 'heun'."
            )
        if clamp_indices is None:
            raise ValueError(
                "FNOFFM.sample requires clamp_indices so sparse observations can be "
                "rasterized onto the grid and clamped during generation."
            )

        bsz = coords.shape[0]
        x = self.prior(coords, self.model.n_fields)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)

        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype)

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            v0 = self.model(
                t=t0,
                x_t=x,
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                obs_indices=clamp_indices,
            )
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking for the FNO backbone.
                v0 = apply_endpoint_observation_consistency(
                    x_t=x,
                    v=v0,
                    t=t0,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )

            if ode_solver == "heun":
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(
                    t=t1,
                    x_t=x_euler,
                    coords=coords,
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    obs_indices=clamp_indices,
                )
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                    v1 = apply_endpoint_observation_consistency(
                        x_t=x_euler,
                        v=v1,
                        t=t1,
                        value_map=value_map,
                        mask_map=mask_map,
                        strength=obs_consistency_strength,
                        schedule_power=obs_consistency_schedule_power,
                    )
                x = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0

            if obs_consistency_mode == "default_hard":
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none":
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x

# Compatibility implementation of the base hybrid GL-RBF backbone.
class _ConditionalPointHybridLocalGlobalRBF(nn.Module):
    """Base hybrid backbone with latent processing and RBF sensor gather."""
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",   # choices: ["cls", "mean"]
    ) -> None:
        super().__init__()

        if summary_type not in ["cls", "mean"]:
            raise ValueError(f"summary_type must be 'cls' or 'mean', got {summary_type}")

        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.summary_type = summary_type

        # Point/query branch.
        # Query point token from [coords, x_t, t]
        self.point_encoder = make_mlp(
            in_dim=coord_dim + n_fields + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
        )

        # Sparse sensor branch.
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Initial sensor token from position, value, and field embedding.
        self.sensor_in_proj = make_mlp(
            in_dim=coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Project refined sensor tokens to the RBF conditioning width.
        self.sensor_out_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=cond_dim,
            out_dim=cond_dim,
            depth=2,
        )

        # Latent global processor.
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Latents attend to sparse sensor tokens.
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Process latents in latent space.
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Refine sensor tokens against the processed latents.
        self.sensor_back_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Project the latent summary to the global feature width.
        self.summary_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
        )

        # Final velocity head.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build sensor tokens from coordinates, values, and field embeddings."""
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        field_feat = field_feat * obs_mask.unsqueeze(-1)             # zero padded rows

        sensor_in = torch.cat([obs_coords, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_in_proj(sensor_in)               # [B, M, D]
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode sparse sensor tokens into the latent array."""
        bsz = sensor_tokens.shape[0]

        # Expand learned latents across the batch.
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)      # [B, L, D]

        # True mask entries exclude padded tokens.
        sensor_padding_mask = ~obs_mask.bool()

        # Latents attend to sparse sensor tokens.
        latents = self.input_cross_attn(
            q=latents,
            kv=sensor_tokens,
            kv_padding_mask=sensor_padding_mask,
        )

        # Process in latent space.
        for block in self.latent_blocks:
            latents = block(latents)

        return latents

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent array into one global summary vector.

        If summary_type == 'cls', the last latent slot is treated as the summary token.
        If summary_type == 'mean', use the mean of all latent slots.
        """
        if self.summary_type == "cls":
            summary = latents[:, -1]         # [B, D]
        else:
            summary = latents.mean(dim=1)    # [B, D]

        return self.summary_proj(summary)    # [B, H]

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather refined sensor features at query points with RBF weights."""
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2        # [B, N, M]
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        local_cond = torch.einsum("bnm,bmd->bnd", weights, refined_sensor_feat)
        return local_cond

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output:
            velocity field of shape [B, N, C]
        """
        bsz, n_pts, _ = x_t.shape

        # Query-point features.
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))  # [B, N, H]

        # Local sensor tokens.
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, M, D]

        # Global latent processing.
        latents = self._encode_latents(sensor_tokens=sensor_tokens, obs_mask=obs_mask)  # [B, L, D]

        # Refine sensor tokens against latent memory.
        refined_sensor_tokens = self.sensor_back_attn(
            q=sensor_tokens,
            kv=latents,
            kv_padding_mask=None,
        )  # [B, M, D]

        # Clear padded sensor rows after attention.
        refined_sensor_tokens = refined_sensor_tokens * obs_mask.unsqueeze(-1)

        # Project refined sensor tokens to the local conditioning width.
        refined_sensor_feat = self.sensor_out_proj(refined_sensor_tokens)   # [B, M, cond_dim]
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)

        # Gather sensor features at query points.
        local_cond = self.aggregate_sparse_obs(
            query_coords=coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
        )  # [B, N, cond_dim]

        # Global summary.
        global_feat = self._extract_global_summary(latents)                 # [B, H]
        global_feat = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)      # [B, N, H]

        # Final velocity prediction.
        out = self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))
        return out
