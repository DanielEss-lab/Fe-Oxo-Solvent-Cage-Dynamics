# Fe-Oxo Solvent-Cage Dynamics

Computational inputs, molecular structures, force-field parameters, and analysis code accompanying:

**Solvent Cage Suppresses Dynamically Induced Entropic Pathways to Provide High Selectivity in Intramolecular Fe-Oxo Radical Rebound Reactions**  
Jyothish Joy, Anthony J. Schaefer, and Daniel H. Ess

This study examines how explicit acetonitrile solvation controls pathway selectivity in Fe(PDP)-mediated intramolecular C-H functionalization.

## Contents

- `sample-implicit-trajectory-input.in`: Milo input for quasiclassical trajectories with implicit acetonitrile solvation.
- `sample-equilibration-input-file.in`: Milo input for equilibration with 200 explicit acetonitrile molecules.
- `sample-explicit-solvent-trajectory-input.in`: Milo input for ONIOM(QM/MM) reaction trajectories with explicit acetonitrile.
- `opls-acn+Feoxo.prm`: Custom OPLS-based force-field parameters, including UFF-derived Lennard-Jones parameters for Fe.
- `cartesian_coordinates.xyz`: Cartesian coordinates of molecular structures.
- `BGAN-EPS-FeOxo/`: Modified BGAN-EPS package for configurational-entropy analysis, with installation instructions and examples.
- `traj_vol_extract_selected_range.py`: MSMS-based solvent-excluded volume and surface-area analysis of the Fe-oxo unit and carboxylate arm.
- `acn_number_density_analysis.py`: Radial acetonitrile number-density analysis.

## Software and use

Dynamics calculations use [Milo-1.2](https://github.com/DanielEss-lab/milo-1.2) interfaced with Gaussian 16; initial solvent packing uses [Packmol](https://m3g.github.io/packmol/). Volume analysis requires NumPy, MDAnalysis, and MSMS. Number-density analysis requires NumPy, Matplotlib, and ASE. BGAN-EPS installation and usage are documented within `BGAN-EPS-FeOxo/`.

Additional computational details are provided in the manuscript and Supporting Information.
