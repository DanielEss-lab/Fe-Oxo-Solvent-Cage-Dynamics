# Modified BGAN-EPS for Fe-oxo dynamics

This package adapts bidirectional generative adversarial network entropic path
sampling (BGAN-EPS) to custom reaction coordinates and selected molecular
fragments. It contains the modified code used for Fe-oxo configurational-entropy
analysis, with a reusable command-line interface and accompanying analysis
scripts. The same core supports the intramolecular Fe(PDP) and TQACl systems.

The original method and software are credited in [CITATION.md](CITATION.md) and
[NOTICE.md](NOTICE.md). The original MIT license is retained. This is a modified
research implementation, not an official upstream release.

## What is included

- BGAN training, coordinate conversion, and the existing histogram/MIST entropy estimator.
- Custom reaction coordinates, increasing/decreasing windows, configurable model dimensions, seeds, and real/generated sample diagnostics.
- Utilities for referencing an entropy profile and plotting it.
- Small PDB and manual-topology references for the two systems.
- A generator for synthetic installation-test inputs.
- Fe(PDP) real-frame MIST, Schlitter quasi-harmonic validation, and the original figure-generation scripts.
- Source hashes and patches documenting the scientific and packaging modifications.

Trajectory datasets, trained outputs, manuscript figures, and electronic-structure
outputs are supplied separately by the publication repository, if available.
The synthetic example is generated locally; it does not contain publication data.

## Installation

Use an isolated Python environment. Python 3.10 or 3.11 is recommended for the
historical preprocessing dependencies. Core CPU checks were also run on Python
3.12; see [docs/VERIFICATION.md](docs/VERIFICATION.md).

From this directory:

```bash
python -m pip install -e ".[plot]"
```

This installs the core and plotting commands. XYZ conversion additionally requires
MDTraj and NetworkX:

```bash
python -m pip install -e ".[preprocess,plot,publication]"
```

On platforms without an MDTraj wheel, use a Conda environment with MDTraj from
conda-forge, then install the package into that same environment. NumPy must remain
below version 2 for the dependency range declared here. CUDA-enabled PyTorch should
be installed using the appropriate command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) before installing
this package. The code selects a GPU when available and otherwise uses the CPU.

The dependency ranges permit reuse; they are not a complete lock of the original
production environment. The recorded production versions and a tested CPU
environment are described in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Quick installation check

From this directory, run these commands in order. Each command is one line, so it
can be used in a terminal on Windows, macOS, or Linux.

```bash
python examples/make_synthetic_example.py --workdir work/synthetic
bgan-eps --workdir work/synthetic --reaction synthetic --rc_file dataset/synthetic_rc.npy --rcmin 1.3 --rcmax 3.9 --rc_direction decreasing --ensemble 4 --window_points 4 --bins 5 --batch_size 32 --epochs 2 --hidden_nodes 16 --latent_dim 8 --loop 2 --seed 42
bgan-eps-summary --run-dir work/synthetic/log/synthetic --reference-window 2 --output-dir work/synthetic/summary
bgan-eps-plot --profile work/synthetic/summary/entropy_profile.csv --output work/synthetic/summary/entropy_profile.png
```

This short run checks the software, not scientific convergence. For a second run,
use another work directory or explicitly pass `--overwrite` to the training command.

## Using your own system

Create a work directory with this structure:

```text
my_work/
  dataset/
    my_reaction.xyz
    reference.pdb
    subset_topology.txt
    my_coordinate.npy       # optional custom coordinate
  log/                     # created by coordinate conversion
```

The multi-frame XYZ and reference PDB must have the same atom count and atom order.
Topology indices are 1-based indices in the full XYZ/PDB, even when entropy is
limited to a subset. See [docs/INPUT_FORMAT.md](docs/INPUT_FORMAT.md).

Convert coordinates:

```bash
bgan-eps-xyz2bat --workdir my_work --reaction my_reaction --ts reference --atom1 25 --atom2 60 --topology_file dataset/subset_topology.txt
```

The atom numbers above are the Fe(PDP) example; replace them for another system.
This writes `log/my_reaction/dof.npy` and `log/my_reaction/topology.txt`.
The first array column is the reaction coordinate. Remaining columns contain the
entropy coordinates in bond, angle, torsion order. Distances are in angstroms;
angles and torsions are in radians.

Train and calculate an entropy profile:

```bash
bgan-eps --workdir my_work --reaction my_reaction --rcmin 1.3 --rcmax 3.9 --rc_direction decreasing --ensemble 8 --window_points 8 --bins 15 --temperature 298.15 --batch_size 32 --epochs 50 --hidden_nodes 32 --latent_dim 64 --loop 5 --seed 42
```

To replace column 0 at runtime, add `--rc_file dataset/my_coordinate.npy`.
The file must contain one finite value per snapshot in exactly the same order.
It may be a one-column `.npy`, headerless `.csv`, or whitespace text file.
The saved input array is not overwritten by the training command.

Use `bgan-eps --help` and `bgan-eps-xyz2bat --help` for all options. Compatibility
wrappers `python main.py` and `python xyz2bat.py` are also included.
`--workdir` paths are resolved from the current directory. Relative `--rc_file`
and `--topology_file` paths are resolved inside the work directory.

## Outputs and interpretation

Each `log/<reaction>/bganN/` directory contains `eps.npy`, `eps.log`, detailed
window logs, and `validation_summary.txt`. `run_settings.json` records the command
settings, dependency versions, device, and loop seeds. The combined result is
`log/<reaction>/<reaction>_eps.npy`. `--save_augmented` also writes generated/real
coordinate arrays and a real-frame mask; these can be large.

Reference to a chemically chosen window, rather than assuming that the first
window is the appropriate zero:

```bash
bgan-eps-summary --run-dir my_work/log/my_reaction --reference-window 4 --output-dir my_work/summary
bgan-eps-plot --profile my_work/summary/entropy_profile.csv --output my_work/summary/profile.svg --xlabel "C-O distance (angstrom)" --reference-coordinate 2.754
```

The reported values are relative `-TΔS` contributions in kcal/mol. A negative
relative value indicates greater entropy than the reference; a positive value
indicates an entropic penalty. They are not absolute thermodynamic entropies or a
complete equilibrium free-energy surface. The original `eps.log` relative column
uses its first window as zero; the summary utility changes that reference.

Summary SD and SEM describe variation across training loops. They do not measure
uncertainty from independent trajectories. A single-loop summary has undefined
SD/SEM; plot it with `--error none`. Assess sparse windows, topology choice,
histogram resolution, and generated configurations before interpreting a result.

## Publication workflows and development

- [docs/PUBLICATION_WORKFLOWS.md](docs/PUBLICATION_WORKFLOWS.md): system atom maps, recorded production commands, and independent validation.
- [publication_scripts/README.md](publication_scripts/README.md): figure scripts and their required input layout.
- [docs/MODIFICATIONS.md](docs/MODIFICATIONS.md): changes relative to upstream and packaging changes.
- [docs/VERIFICATION.md](docs/VERIFICATION.md): checks performed on this release and their limits.

Run the included software checks with `python -m unittest discover -s tests`.
For archiving, upload the extracted directory contents to the publication
repository and keep the version number attached to the corresponding release.
