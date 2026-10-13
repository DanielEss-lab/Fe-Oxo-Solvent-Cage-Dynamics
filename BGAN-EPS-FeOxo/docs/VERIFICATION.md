# Release verification

Checks performed for version 0.1.0 on Windows with Python 3.12.14, NumPy 1.26.4,
PyTorch 2.5.1 (CPU), scikit-learn 1.5.2, NetworkX 3.3, and matplotlib 3.9.2:

- Installed the package into an isolated environment and exercised training,
  summary, plot, and converter-help entry points.
- Completed the synthetic two-epoch, two-loop training example, generated a
  reference-window CSV summary, and rendered its PNG plot.
- Compared that same CPU training calculation against the supplied working
  `bgan_eps_feoxo` code: the complete `(2, 4)` output arrays matched bit-for-bit.
- Compared real-only EPS calculations against the working code for both increasing
  and decreasing windows: the output arrays matched bit-for-bit.
- Confirmed that the network definitions are byte-identical and the loss functions,
  reconstruction filter, validation-report routine, Jacobians, and one-/two-dimensional
  entropy functions are unchanged at the Python syntax-tree level.
- Passed eight included regression/workflow tests, covering both-direction entropy
  values, custom coordinate paths, reference-window subtraction, input immutability,
  overwrite protection, nonfinite inputs, topology/coordinate mismatches,
  single-loop uncertainty, and empty-window summary rejection.

Run the portable checks from the package directory after installation:

```bash
python -m unittest discover -s tests -v
```

The synthetic data test software behavior. They do not establish scientific
convergence or reproduce the publication's numerical profile. The original
publication trajectories were not rerun, and CUDA training was not tested in this
release check. MDTraj XYZ conversion was not executed in this Windows environment;
its path/import changes were inspected and its command-line help was checked.
Use the production workflow with compatible MDTraj dependencies for actual
conversion and validate atom ordering/topology on your system.

The Fe(PDP) independent validation scripts retain their calculation code and
require the separate publication inputs. They are not tests of an arbitrary
new molecular system. The historical figure scripts similarly require the
documented publication-table schemas.
