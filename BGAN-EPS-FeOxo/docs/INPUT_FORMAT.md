# Input formats and assumptions

## Cartesian inputs

`dataset/<reaction>.xyz` is a standard multi-frame XYZ: an atom-count line, one
comment line, and one element/x/y/z row per atom, repeated for each frame.
Coordinates are in angstroms. `dataset/<ts>.pdb` supplies the full atom ordering
and, when no manual topology is used, the connectivity.

Manual topology is recommended for metal complexes. Inspect coordination bonds
and reactive atoms explicitly; a recognizable molecular picture does not establish
correct connectivity. The included reference PDBs are specific to their systems.

## Manual topology

The first line is the number of selected entropy atoms. Each subsequent line lists
two, three, or four full-system atom indices, defining a bond, angle, or torsion.
Indices are 1-based and are not renumbered for an entropy subset. All bond entries
must come first, then angles, then torsions. Do not include the windowing coordinate
as an additional topology entry unless you intentionally want it in the entropy basis.

The average estimator uses the number of selected atoms to weight each coordinate
class. It requires at least four selected atoms and at least two entries in each
class, because its existing formulas divide by `n_class` and `n_class - 1`.
A topology containing distances only is not supported by this estimator.

## Precomputed arrays

You may supply `log/<reaction>/dof.npy` and `topology.txt` directly without MDTraj.
The array must be finite, floating-point, and shaped `(snapshots, 1 + coordinates)`.
Column 0 is the windowing coordinate; columns 1 onward match the topology entries.
Distances use angstroms, angles use radians in the physical bond-angle domain,
and torsions use radians. The package does not change torsion wrapping or introduce
additional periodic-boundary treatment.

`--rc_file` replaces column 0 in memory. Keep frame order identical to `dof.npy`.
Use a one-dimensional `.npy` or a one-column text file without a header.

## Windows

`--rcmin` and `--rcmax` are aliases for `--bondmin` and `--bondmax`. Always provide
minimum < maximum; `--rc_direction` determines the order in which windows are
reported. The estimator uses strict inequalities at both window boundaries.

`--ensemble` controls window width: `(maximum - minimum) / ensemble`.
`--window_points` controls how many window positions are evaluated. Set these equal
for adjacent windows; other combinations produce moving windows with different
overlap or spacing. Window bounds in `eps.log` are rounded to four decimal places.

The entire supplied array is scaled and used for model training. Accepted real
and generated configurations are then filtered to the specified open coordinate
range before entropy estimation. The windowing coordinate participates in training
but is excluded from the entropy basis.

Changing physical units, entropy-coordinate choices, topology redundancy, or
histogram resolution can change an entropy estimate. The Jacobian-corrected MIST
implementation is inherited from the research code; it is not an independent
thermodynamic standard-state treatment.
