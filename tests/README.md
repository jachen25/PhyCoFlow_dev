# Tests

These self-contained scripts test the modules in `../src`. Run any script
directly from any working directory:

```bash
python tests/test_coarse_grid_condition.py
```

Each script adds `../src` to `sys.path`. The `SRC` constant used by
source-inspection tests refers to that directory.

Launch scripts run a subset as preflight checks (see
`run_pointcloud_ffm_active_emulsion_*.sbatch`): `test_ae_augment.py`,
`test_param_injection.py`, and for N18 `test_coarse_grid_condition.py` and
`test_pooled_grid_condition.py`.

## Conventions

- Tests use synthetic CPU fixtures and do not read datasets or require a GPU.
- RNG parity tests such as `test_coarse_grid_condition.py` and
  `test_constrained_topo.py` compare bit-exact tensors.
