"""Gates for the Senseiver active-emulsion joint-SR wiring.

The Det-baseline stack gained: an active_emulsion branch in build_dataset,
SR conditioning keys in validate_and_normalize_config, and
resolve_sr_condition_kwargs threading obs_grid_stride_list / obs_grid_pool /
pool_value_transform into run_epoch_senseiver and the deterministic
visualizer. Checked here, without touching the real dataset:

  1. The shipped config_baseline_Det.yaml validates and resolves; its
     architecture block matches the FFM N19 capacity rule (D/N/H/L/FF).
  2. Legacy configs (no SR keys) resolve to sr_kwargs == {} so every pre-SR
     call site keeps byte-identical behavior (RNG parity).
  3. On a stub AE-like dataset, the resolved kwargs drive
     build_sparse_condition to the exact N19 token layout: 256 pooled phi +
     64 + 64 pooled colocated vx/vy = 384 tokens, all valid.
  4. obs_grid_pool_physical demands the dataset's physical-pooling contract
     (a dataset without it must raise, not silently pool in model space).
  5. run_epoch_senseiver runs end-to-end on the stub loader (CPU, tiny
     model) with a finite loss.
  6. The build_dataset active_emulsion branch's kwargs bind to the real
     ActiveEmulsionDataset constructor signature (no data load).
"""

# Make src/ importable when run from any cwd.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

import copy
import inspect
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model_baseline import (
    BaselineBundle,
    Senseiver,
    build_sparse_condition,
    load_yaml,
    resolve_sr_condition_kwargs,
    resolve_stage_config,
    run_epoch_senseiver,
    validate_and_normalize_config,
)
from helpers_baseline import collate_snapshots

DEMO_ROOT = Path(_SRC_DIR).parent
CONFIG_PATH = DEMO_ROOT / "Save_config/active_emulsion/config_baseline_Det.yaml"

N = 32          # stub grid edge (matches 128 / 4 to keep the test light)
STRIDES = [8, 16, 16]


class _StubAEDataset(Dataset):
    """Minimal stand-in honoring the ActiveEmulsionDataset contracts used
    by the SR path: grid_shape, num_fields/field_names/mean/std, and the
    physical-pooling transform (pool_observations_physical + denormalize +
    normalize_field, asinh on the flow channels)."""

    def __init__(self, n_samples=4, physical=True):
        self.grid_Ny = self.grid_Nx = N
        self.grid_shape = (N, N)
        self.field_names = ("phi", "vx", "vy")
        self.num_fields = 3
        self.mean = torch.zeros(3)
        self.std = torch.ones(3)
        self.asinh_scale = torch.tensor([1.0, 0.5, 0.5])
        self.lin_mask = torch.tensor([True, False, False])
        self.pool_observations_physical = bool(physical)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, N), torch.linspace(0, 1, N), indexing="ij")
        coords2 = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.coords_raw = coords2
        self.coords = torch.cat([coords2, torch.zeros(N * N, 1)], dim=-1)
        g = torch.Generator().manual_seed(0)
        self._fields = torch.randn(n_samples, N * N, 3, generator=g)

    def denormalize(self, z):
        m = self.mean.to(z.device, z.dtype)
        s = self.std.to(z.device, z.dtype)
        sc = self.asinh_scale.to(z.device, z.dtype)
        lin = self.lin_mask.to(z.device)
        u = z * s + m
        return torch.where(lin, u, sc * torch.sinh(u))

    def normalize_field(self, x, field_id: int):
        field_id = int(field_id)
        t = x if bool(self.lin_mask[field_id]) else torch.asinh(
            x / self.asinh_scale[field_id].to(x.device, x.dtype))
        return (t - self.mean[field_id].to(x.device, x.dtype)) / \
            self.std[field_id].to(x.device, x.dtype)

    def __len__(self):
        return self._fields.shape[0]

    def __getitem__(self, i):
        return {
            "coords": self.coords.clone(),
            "coords_raw": self.coords_raw.clone(),
            "fields": self._fields[i].clone(),
            "time_index": torch.tensor(i, dtype=torch.long),
            "physical_time": torch.tensor(0.0),
        }


def _load_cfg():
    cfg = load_yaml(CONFIG_PATH)
    return validate_and_normalize_config(cfg)


def _stub_cfg(cfg=None):
    """The shipped config, retargeted at the stub's smaller grid."""
    cfg = copy.deepcopy(cfg or _load_cfg())
    cfg["shared"]["data"]["num_x"] = N
    cfg["shared"]["data"]["num_y"] = N
    return cfg


def test_config_validates_and_matches_n19_capacity():
    cfg = _load_cfg()
    assert cfg["shared"]["data"]["dataset_name"] == "active_emulsion"
    cond = cfg["shared"]["conditioning"]
    assert cond["cond_fields"] == [0, 1, 2]
    assert cond["obs_grid_stride_list"] == STRIDES
    assert cond["obs_grid_pool"] is True
    assert cond["obs_grid_pool_physical"] is True

    stage = resolve_stage_config(cfg)
    arch = stage["architecture"]
    # FFM N19 capacity rule: D=256, N=4, H=8, L=128, FF=4.
    assert arch["latent_dim"] == 256
    assert arch["num_encoder_layers"] * arch["num_self_attn_per_block"] == 4
    assert arch["num_cross_attn_heads"] == 8
    assert arch["num_self_attn_heads"] == 8
    assert arch["num_latents"] == 128
    assert arch["ff_mult"] == 4
    assert arch["latent_dim"] % arch["num_self_attn_heads"] == 0


def test_legacy_config_resolves_to_empty_sr_kwargs():
    # A pre-SR config is the same surface without the coarse-grid keys; the
    # resolver must return {} for it so every legacy call site keeps its
    # random-sensor behavior (and RNG parity) unchanged.
    legacy = load_yaml(CONFIG_PATH)
    for key in ("obs_grid_stride_list", "obs_grid_pool", "obs_grid_pool_physical"):
        legacy["shared"]["conditioning"].pop(key, None)
    legacy = validate_and_normalize_config(legacy)
    cond = legacy["shared"]["conditioning"]
    assert cond["obs_grid_stride_list"] is None
    assert cond["obs_grid_pool"] is False
    assert resolve_sr_condition_kwargs(legacy, _StubAEDataset()) == {}


def test_sr_kwargs_produce_n19_token_layout():
    cfg = _stub_cfg()
    ds = _StubAEDataset()
    kw = resolve_sr_condition_kwargs(cfg, ds)
    assert kw["Ny"] == N and kw["Nx"] == N
    assert kw["obs_grid_strides"] == STRIDES
    assert kw["obs_grid_pool"] is True
    assert kw["pool_value_transform"] is ds

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_snapshots)
    batch = next(iter(loader))
    cond = cfg["shared"]["conditioning"]
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = \
        build_sparse_condition(
            coords_full=batch["coords"], fields_full=batch["fields"],
            cond_fields=cond["cond_fields"],
            n_obs_min=cond["n_obs_min_list"], n_obs_max=cond["n_obs_max_list"],
            **kw)
    n_phi = (N // STRIDES[0]) ** 2
    n_v = (N // STRIDES[1]) ** 2
    assert obs_mask.shape[1] == n_phi + 2 * n_v
    assert bool(obs_mask.all())
    assert (obs_field_ids == 0).sum(dim=1).eq(n_phi).all()
    assert (obs_field_ids == 1).sum(dim=1).eq(n_v).all()
    assert (obs_field_ids == 2).sum(dim=1).eq(n_v).all()
    # vx/vy pooled sites are colocated by construction (same lattice).
    vx = obs_coords[0, obs_field_ids[0] == 1]
    vy = obs_coords[0, obs_field_ids[0] == 2]
    assert torch.allclose(vx, vy)
    # Physical pooling actually differs from model-space pooling on the
    # asinh channels.
    kw_model_space = dict(kw, pool_value_transform=None)
    _, obs_values_ms, _, _, _ = build_sparse_condition(
        coords_full=batch["coords"], fields_full=batch["fields"],
        cond_fields=cond["cond_fields"],
        n_obs_min=cond["n_obs_min_list"], n_obs_max=cond["n_obs_max_list"],
        **kw_model_space)
    v_sel = obs_field_ids[0] >= 1
    assert not torch.allclose(obs_values[:, v_sel], obs_values_ms[:, v_sel])


def test_physical_pooling_requires_dataset_contract():
    cfg = _stub_cfg()
    ds = _StubAEDataset(physical=False)
    try:
        resolve_sr_condition_kwargs(cfg, ds)
    except ValueError as exc:
        assert "obs_grid_pool_physical" in str(exc)
    else:
        raise AssertionError(
            "physical pooling without the dataset contract must raise")


def test_run_epoch_senseiver_end_to_end_on_stub():
    cfg = _stub_cfg()
    # Tiny architecture: this gate checks wiring, not capacity.
    arch = cfg["senseiver_params"]["architecture"]
    arch.update(num_latents=8, latent_dim=32, num_encoder_layers=1,
                num_self_attn_per_block=1, num_cross_attn_heads=2,
                num_self_attn_heads=2, dec_num_cross_attn_heads=2,
                field_embed_dim=4, space_bands=4)
    cfg["senseiver_params"]["training"]["n_query_points"] = 64

    ds = _StubAEDataset()
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_snapshots)
    model = Senseiver(
        n_fields=ds.num_fields, coord_dim=2, num_latents=8, latent_dim=32,
        num_encoder_layers=1, num_self_attn_per_block=1,
        num_cross_attn_heads=2, num_self_attn_heads=2,
        dec_num_cross_attn_heads=2, field_embed_dim=4, space_bands=4,
        max_freq=8.0, ff_mult=2, dropout=0.0)
    bundle = BaselineBundle(
        baseline_model="senseiver", training_stage=1, model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None, ema=None, device=torch.device("cpu"),
        run_dir=Path("."), config=cfg, dataset_train=ds, dataset_val=ds)
    loss = run_epoch_senseiver(bundle, loader, training=True, epoch=0)
    assert loss == loss and loss < float("inf")


def test_build_dataset_kwargs_bind_to_real_ctor():
    from helpers import ActiveEmulsionDataset
    cfg = _load_cfg()
    data_cfg = cfg["shared"]["data"]
    sig = inspect.signature(ActiveEmulsionDataset.__init__)
    # Mirror both branches of the build_dataset active_emulsion arm.
    train_kwargs = dict(
        data_root="/nonexistent", split="train",
        protocol=str(data_cfg["ae_protocol"]),
        splits_path=data_cfg.get("ae_splits_path"),
        fields=tuple(data_cfg["ae_fields"]),
        seed=int(cfg["shared"]["seed"]),
        flow_transform=str(data_cfg["ae_flow_transform"]),
        pool_observations_physical=True,
        frame_downsample=bool(data_cfg["ae_frame_downsample"]),
        frame_tau=float(data_cfg["ae_frame_tau"]),
        frame_min=int(data_cfg["ae_frame_min"]),
        augment=str(data_cfg["ae_augment"]),
    )
    sig.bind(None, **train_kwargs)
    val_kwargs = {k: v for k, v in train_kwargs.items()
                  if not k.startswith("frame_") and k != "augment"}
    val_kwargs["split"] = "val"
    sig.bind(None, **val_kwargs)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [PASS] {name}")
            except Exception as exc:  # noqa: BLE001 - gate runner
                failures += 1
                print(f"  [FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
