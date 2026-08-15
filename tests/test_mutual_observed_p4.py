"""Test configuration propagation and validation for observed anchors.

All observed-anchor fields are checked across argparse, direct-coherence, and
topology-loss configurations.
"""

# Support direct execution from any working directory.
import os as _os
import sys as _sys
_SRC_DIR = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)
import sys
from argparse import Namespace

from topo_coherence_training.topo_loss import TopoLossConfig
from direct_coherence_loss import TopoDirectCoherenceConfig
from train_pointcloud_ffm import build_topo_direct_coherence_config

ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")
    ok = ok and bool(cond)


def raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


# Map direct-coherence configuration to topology-loss configuration.
b = TopoDirectCoherenceConfig(
    mode="betti_self_mutual",
    mutual_anchor_source="observed", mutual_anchor_channels=[2, 3],
    mutual_anchor_provider="vorticity", mutual_carrier_gauge="interface",
    mutual_reduction="curve", bifilt_carrier_channel=0)
a = b.to_topo_loss_config()
check("T-map source", a.mutual_anchor_source == "observed")
check("T-map channels", list(a.mutual_anchor_channels) == [2, 3])
check("T-map provider", a.mutual_anchor_provider == "vorticity")
check("T-map gauge", a.mutual_carrier_gauge == "interface")
check("T-map reduction", a.mutual_reduction == "curve")


# Build direct-coherence configuration from argparse values.
def _args(**over):
    """Create a complete, overridable namespace for the builder."""
    d = dict(
        _yaml_keys=set(), direct_coherence_enabled=True,
        topo_grid_h=64, topo_grid_w=64, topo_n_points=8192, topo_idx_seed=0,
        topo_mode="betti_self_mutual", topo_saliency="zscore", topo_quantiles=[0.5],
        topo_pairs=None, topo_presmooth_sigma=1.0, topo_channels=[0],
        topo_beta=8.0, topo_connectivity=4, topo_contact_sigma=1.0,
        topo_contain_center=False, topo_contain_sharp=1.0, topo_count_beta=8.0,
        topo_count_weight=0.0, topo_dilate_ksize=3, topo_ec_beta=8.0,
        topo_ec_quantiles=[0.5], topo_ec_weight=0.0, topo_epi_fields=[0],
        topo_epi_weight=0.0, topo_euler_open_ksize=3,
        topo_landscape_crossfield_weight=0.0, topo_landscape_h0_weight=0.0,
        topo_landscape_k_layers=3, topo_landscape_resolution=64,
        topo_landscape_slice_weights=(), topo_max_coverage=1.0,
        topo_min_bar_persistence=0.0, topo_min_threshold_gap=0.0,
        topo_missing_mass_weight=0.0, topo_pi_grid_birth=32, topo_pi_grid_pers=32,
        topo_pi_sigma=1.0, topo_rcc_stride_frac=0.5, topo_rcc_window_frac=0.5,
        topo_reference_path=None, topo_region_relations_weight=0.0,
        topo_run_global_checks=False, topo_t_max=1.0, topo_t_min=0.0,
        topo_workers=0, topo_wrcc_weight=0.0,
        # Observed-anchor arguments.
        topo_mutual_anchor_source="observed", topo_mutual_anchor_channels=None,
        topo_mutual_anchor_provider="vorticity", topo_mutual_carrier_gauge="interface",
        topo_mutual_reduction="match",
        cond_fields=[2, 3],
        # Enable the observed path during validation.
        topo_bifilt_carrier_channel=0, topo_mutual_h1_weight=1.0, topo_mutual_h0_weight=0.0,
        topo_self_h0_weight=0.0, topo_self_h1_weight=0.0, topo_homology_dims=[1],
    )
    d.update(over)
    return Namespace(**d)


cfg = build_topo_direct_coherence_config(_args())
check("T-build source dest -> b", cfg.mutual_anchor_source == "observed")
check("T-build provider dest -> b", cfg.mutual_anchor_provider == "vorticity")
check("T-build gauge dest -> b", cfg.mutual_carrier_gauge == "interface")
check("T-build reduction dest -> b", cfg.mutual_reduction == "match")
check("T-build anchor_channels DEFAULTS to cond_fields", list(cfg.mutual_anchor_channels) == [2, 3],
      f"got {cfg.mutual_anchor_channels}")
# Explicit anchor channels override conditioned fields.
cfg2 = build_topo_direct_coherence_config(_args(topo_mutual_anchor_channels=[1, 2]))
check("T-build explicit anchor_channels override", list(cfg2.mutual_anchor_channels) == [1, 2])
# The two mapping checks cover the full configuration path.


# Configuration validation.
def _tlc(**kw):
    base = dict(mode="betti_self_mutual", bifilt_carrier_channel=0,
                mutual_h1_weight=1.0, mutual_h0_weight=0.0,
                self_h0_weight=0.0, self_h1_weight=0.0, homology_dims=(1,))
    base.update(kw)
    return TopoLossConfig(**base)


check("T-valid OK config constructs", not raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_anchor_provider="vorticity",
                 mutual_anchor_channels=[2, 3])))
check("T-valid bad provider raises", raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_anchor_provider="nope",
                 mutual_anchor_channels=[2, 3])))
check("T-valid bad gauge raises", raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_carrier_gauge="nope",
                 mutual_anchor_channels=[2, 3])))
check("T-valid bad reduction raises", raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_reduction="nope",
                 mutual_anchor_channels=[2, 3])))
check("T-valid vorticity wrong arity raises", raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_anchor_provider="vorticity",
                 mutual_anchor_channels=[2])))
check("T-valid self-anchoring (carrier in channels) raises", raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_anchor_provider="vector_magnitude",
                 mutual_anchor_channels=[0, 2])))       # carrier=0 in channels
check("T-valid source='generated' is INERT (bad provider ignored)", not raises(
    lambda: _tlc(mutual_anchor_source="generated", mutual_anchor_provider="nope")))
check("T-valid channels=None (trainer fills later) does NOT raise", not raises(
    lambda: _tlc(mutual_anchor_source="observed", mutual_anchor_provider="vorticity",
                 mutual_anchor_channels=None)))

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
