# Publication workflows

The commands below use the recorded production settings. The archive contains
code and small references; each user supplies the corresponding trajectory XYZ
and, for independent validation, metadata. Do not apply either system's atom map
to a different atom order.

## Fe(PDP): intramolecular carboxylate arm

Atom numbering is 1-based: Fe52, O25, radical carbon C60, transferred H64.
Entropy includes atoms 53–63 and 65–74, excluding transferred H64 and Fe52.
The supplied manual topology contains 20 bonds, 36 angles, and 42 torsions.
There are 98 entropy coordinates plus the windowing coordinate.

```bash
python examples/prepare_reference.py --system feoxo --workdir work/feoxo
```

Place the pooled multi-frame trajectory in
`work/feoxo/dataset/feoxo_implicit_all.xyz`. For the publication dataset, it has
41,144 frames from 60 trajectories: 38 rebound, 18 desaturation, and 4 direct
lactonization. No trajectory files are included in this package.

```bash
bgan-eps-xyz2bat --workdir work/feoxo --reaction feoxo_implicit_all --ts mecp_fixed --atom1 25 --atom2 60 --topology_file dataset/mecp_carboxylate_arm_noH64_topology.txt
bgan-eps --workdir work/feoxo --reaction feoxo_implicit_all --rcmin 1.3 --rcmax 3.9 --rc_direction decreasing --ensemble 8 --window_points 8 --bins 15 --temperature 298.15 --batch_size 32 --epochs 50 --lr 1e-4 --hidden_nodes 32 --latent_dim 64 --loop 5 --seed 42
bgan-eps-summary --run-dir work/feoxo/log/feoxo_implicit_all --reference-window 4 --output-dir work/feoxo/summary
bgan-eps-plot --profile work/feoxo/summary/entropy_profile.csv --output work/feoxo/summary/dCO_entropy.svg --xlabel "C-O distance (angstrom)" --reference-coordinate 2.754
```

Here the converter directly calculates `d(O25,C60)` in column 0, so a separate
custom-coordinate file is unnecessary. To use precomputed internal coordinates
whose first column contains a different coordinate, supply the matching
`feoxo_implicit_all_d_OC.npy` with `--rc_file`.

MECP is approximately 2.754 angstrom and lies in window 4 (2.925–2.600 angstrom).
The six-window sensitivity run uses `--ensemble 6 --window_points 6`; its MECP
reference is window 3. Keep distinct reaction/run directories for different runs.

## TQACl: internal fragment entropy

Atom numbering: Fe59, Cl60, O61, radical carbon C69, transferred H70.
Internal entropy uses atoms 69 and 71–79, excluding H70. The manual topology
contains 9 bonds, 13 angles, and 13 torsions: 35 entropy coordinates plus the RC.

```bash
python examples/prepare_reference.py --system tqacl --workdir work/tqacl
```

Place the pooled trajectory in `work/tqacl/dataset/tqacl_all.xyz`.

```bash
bgan-eps-xyz2bat --workdir work/tqacl --reaction tqacl_all --ts tqacl_ts_first_frame --atom1 69 --atom2 61 --topology_file dataset/tqacl_entropy_atoms_69_71-79_noH70_topology.txt
bgan-eps --workdir work/tqacl --reaction tqacl_all --rcmin 0.8666010187307134 --rcmax 6.2266010187307135 --rc_direction increasing --ensemble 8 --window_points 8 --bins 15 --temperature 298.15 --batch_size 8192 --epochs 50 --lr 1e-4 --hidden_nodes 32 --latent_dim 64 --loop 5 --seed 42
bgan-eps-summary --run-dir work/tqacl/log/tqacl_all --reference-window 3 --output-dir work/tqacl/summary
```

The first-frame TS distance is approximately 2.541601 angstrom, centered in
window 3. This internal-coordinate estimator does not capture the full relative
translation and orientation of a dissociating fragment. Do not interpret its
internal entropy alone as a complete dissociation entropy.

The exploratory TQACl 3D fragment-COM post-processing in the project was not
present as a saved source script and is not represented here as a reproducible
implementation. The core and internal-entropy commands above are available.

## Fe(PDP) independent real-frame validation

`publication_scripts/validation/analyze_real_frame_entropy.py` contains the saved
real-only histogram/MIST and Cartesian Schlitter quasi-harmonic analyses. It is
specific to the Fe(PDP) atom definitions and eight windows. Its equations were
retained; input/output directories were made explicit.

The validation expects this supplied data layout:

```text
my_feoxo_project/
  dataset/
    feoxo_implicit_all.xyz
    feoxo_implicit_all_d_OC.npy
    feoxo_implicit_all_metadata.csv
  log/
    feoxo_implicit_all_dOC_8windows/
      dof.npy
      topology.txt
      d_CO_entropy_profile_MECP_reference.csv
      d_CO_loop_profiles.csv
```

The MIST input array's first column must contain the C–O distance, since this
script windows directly on that column. The metadata CSV must contain
`global_frame_index`, `source_trajectory`, `product_class`, `snapshot_index`, and
`atom_count` in frame order. The supplied data/trajectory naming conventions
are part of this system-specific workflow; this script is not a general topology
builder for unrelated systems.

```bash
python publication_scripts/validation/analyze_real_frame_entropy.py --project-dir my_feoxo_project --output-dir work/validation --methods mist qh --bins 10 15 20 --mist-equal-sample-repeats 50 --mist-equal-sample-size 1000 --mist-jackknife --bootstrap 1000 --qh-equal-sample-repeats 500 --qh-equal-sample-size 1000 --seed 20260916
python publication_scripts/validation/summarize_entropy_validation.py --results-dir work/validation --bgan-dir my_feoxo_project/log/feoxo_implicit_all_dOC_8windows
```

The summarizer expects the publication-specific CSV column names and exactly
five BGAN loop columns (`bgan1_MECP_ref` through `bgan5_MECP_ref`), rather than the
generic filenames/schema produced by `bgan-eps-summary`. Use the original
publication analysis tables as its BGAN inputs. It reports both the broad
windows-5–8 minus windows-1–3 contrast and the localized windows-5–6 minus
windows-2–3 contrast.

Quasi-harmonic uncertainty comes from whole-trajectory bootstrap sampling;
real-only MIST uses a leave-one-trajectory-out jackknife. These differ from the
SD or SEM of training loops. The reference-window value is zero by construction.

## GPU allocation

On a scheduler-managed GPU node, activate the environment before running the
commands above. The recorded BYU workflow used `PYTHONNOUSERSITE=1` to avoid
user-site packages and assigned `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to allocated
CPU resources. Allocation commands and CUDA module names are cluster-specific;
they are not embedded as defaults in the reusable package.
