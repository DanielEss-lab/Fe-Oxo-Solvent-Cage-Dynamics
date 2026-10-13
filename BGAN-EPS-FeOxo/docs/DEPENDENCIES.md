# Dependency environments

## Recorded production environment

The project notes describe BYU runs using Python 3.9 and CUDA/PyTorch in the
`bgan_eps` Conda environment. The upstream README lists:

| Component | Listed version |
|---|---|
| Python | 3.9.18 |
| NumPy | 1.26.0 |
| PyTorch | 2.1.0 |
| NetworkX | 3.1 |
| scikit-learn | 1.3.0 |
| MDTraj | 1.9.9 |

These are historical requirements recorded with the project, not a verified
freeze of every library on the production GPU node. A complete production
environment export was not present. Upstream requirements:
[BGAN-EPS repository](https://github.com/rshin1209/bgan_eps).

For a historical Linux setup, first create a Python 3.9 or 3.10 environment,
install compatible versions above, then install this package with `pip install
-e . --no-deps`. Choose the PyTorch build for your CUDA installation.

## Reuse and local verification

Package metadata keeps NumPy below 2 and allows compatible versions of PyTorch and
scikit-learn. Preprocessing, plotting, and historical raster figures are optional
extras. The CPU verification environment for this release used Python 3.12,
NumPy 1.26.4, PyTorch 2.5.1, scikit-learn 1.5.2, NetworkX 3.3, and matplotlib 3.9.2.
See `VERIFICATION.md` for what was actually tested.

MDTraj availability depends on Python and platform. Its optional dependency range
is intentionally separate from core training. For conversion, a Linux Conda
environment with a compatible MDTraj build is a practical route. Follow the
[MDTraj project documentation](https://github.com/mdtraj/mdtraj)
and maintain the NumPy range required by this package.

With a compatible NVIDIA GPU build, the code selects `cuda:0`; otherwise it runs
on CPU. Fixed seeds do not guarantee bitwise reproducibility across library
versions, devices, or hardware. Record the automatically generated
`run_settings.json` with each scientific calculation.
