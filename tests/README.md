# tests/

Gate suites for `../src` (moved out of `src/` on 2026-08-14; they previously
lived alongside the sources).

Every file is a self-contained script with a `__main__` runner and PASS/FAIL
prints — run any of them directly, from any working directory:

```bash
python tests/test_coarse_grid_condition.py
```

Each file starts with a bootstrap shim that puts `../src` on `sys.path`, so
imports like `from helpers import ...` resolve without needing `cd src` or
`PYTHONPATH`. Inside the tests, the `SRC` constant (where present) points at
the `src/` directory — it is used by source-reading gates
(e.g. `test_n12_review_fixes.py` greps `train_pointcloud_ffm.py`) and must
keep meaning "the src dir", not "this file's dir".

Launch scripts run a subset as pre-flight gates (see
`run_pointcloud_ffm_active_emulsion_*.sbatch`): `test_ae_augment.py`,
`test_param_injection.py`, and for N16 `test_coarse_grid_condition.py`.

Conventions (see the memory of past incidents for why):
- Tests are synthetic/CPU-only; they must not read dataset files from
  `/orcd/pool` or require a GPU.
- Gates that protect RNG parity (`test_coarse_grid_condition.py`,
  `test_constrained_topo.py`) compare bit-exact tensors — keep them green
  before touching conditioning or post-train code paths.
