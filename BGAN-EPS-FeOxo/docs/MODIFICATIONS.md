# Modifications relative to the local upstream snapshot

## Research changes

- `main.py`: optional custom coordinate file, coordinate limits/direction, configurable
  hidden/latent dimensions, bins, moving windows, sample warning threshold, seed,
  and optional augmented-array saves.
- `bgan.py`: configurable network dimensions; real/generated sample tracking;
  global validation reports; real and augmented counts per window; configurable
  histogram/window settings; increasing and decreasing coordinate windows; an
  explicit error for the unsupported `max` estimator path.
- `xyz2bat.py`: optional manual topology for metal coordination and entropy subsets;
  explicit reaction-coordinate arguments and safe file handling.
- `model.py`: retained from the supplied local upstream snapshot.

## Release packaging changes

- Imports use the package namespace; paths support an explicit `--workdir`.
- Existing output directories are protected unless `--overwrite` is supplied.
- Training checks coordinate dimensions, finiteness, topology ordering, estimator
  coordinate counts, and basic run settings before computation.
- Run metadata records settings, loop seeds, dependencies, and device.
- The converter no longer deletes an unrelated `<reaction>.pdb` file that it does
  not itself create.
- Validation scripts accept explicit input/output directories rather than requiring
  the developer's project checkout.
- Generic summary/plot commands and synthetic software checks were added.

The BGAN network architecture, loss functions, optimizer updates, configuration
acceptance filter, histogram/Jacobian arithmetic, MIST correction, and EPS window
calculation remain those of the modified working snapshot. Research results are
not regenerated merely by installing this release.

The patches in `provenance/` distinguish the research modifications from packaging
edits. SHA-256 hashes identify the input snapshots even though their original
upstream Git commit is unknown.
