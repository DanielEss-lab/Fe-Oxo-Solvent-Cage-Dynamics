#!/usr/bin/env python3
"""Analyze the radial molecular number density of ACN in a spherical cage.

The script assumes that the Fe-oxo complex is listed first in every XYZ frame
and that it is followed by N_ACN consecutive, all-atom ACN molecules.  Each
ACN is represented by its molecular center of mass.  Densities are calculated
in spherical shells around the actual spherical-boundary center.

Outputs
-------
acn_radial_number_density.csv
    Mean radial density, block SEM, and density relative to the bulk target.
acn_radial_density_blocks.csv
    Individual contiguous-block density profiles.
acn_shell_density_summary.csv
    Broad-shell densities and contiguous-block values.
acn_radial_number_density.png
    Mean density profile with block SEM.
acn_radial_density_blocks.png
    Individual block profiles for equilibration assessment.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread


# ============================================================================
# USER SETTINGS
# ============================================================================

TRAJECTORY = "sample-explicit-solvent-trajectory.xyz"

N_ACN = 200
ATOMS_PER_ACN = 6  # All-atom CH3CN: C2 H3 N
SPHERE_RADIUS = 16.4  # Angstrom

# Discard this many initial saved frames.  Convert the desired discarded time
# to frames using the trajectory output interval.
SKIP_FRAMES = 0

# Analyze every STRIDE-th saved frame after SKIP_FRAMES.  Increasing STRIDE
# reduces runtime and file-reading cost, but block averages should still be
# defined by contiguous time intervals.
STRIDE = 1

# A width of 0.25-0.50 A is reasonable for 200 ACN molecules.
DESIRED_DR = 0.25  # Angstrom

# Use "fixed" when the spherical wall is centered at fixed coordinates.
# Use "fe" only when the wall follows the Fe atom.
CENTER_MODE = "fixed"
FIXED_CENTER = np.array([0.0, 0.0, 0.0])  # Angstrom

# Bulk-density reference.  The default is experimental ACN at 298.15 K.
# Replace this with the density of the chosen ACN force field from neat-solvent
# NPT if that value is available.
TARGET_MASS_DENSITY = 0.77588  # g cm^-3
ACN_MOLAR_MASS = 41.0519  # g mol^-1
AVOGADRO = 6.02214076e23  # mol^-1

# Contiguous trajectory blocks used to assess time stability.
N_BLOCKS = 5

# Broad radial intervals used to average over molecular layering.  The 9-14 A
# range is appropriate only if it lies outside the solute-excluded region and
# inside the wall-affected region for this system.
REPORT_SHELLS = [
    (9.0, 12.0),
    (12.0, 14.0),
    (9.0, 14.0),
]

# Output names
RADIAL_CSV = "acn_radial_number_density.csv"
BLOCK_CSV = "acn_radial_density_blocks.csv"
SHELL_CSV = "acn_shell_density_summary.csv"
MEAN_PLOT = "acn_radial_number_density.png"
BLOCK_PLOT = "acn_radial_density_blocks.png"


def validate_settings():
    """Check user settings before reading a potentially large trajectory."""
    if not Path(TRAJECTORY).is_file():
        raise FileNotFoundError(f"Trajectory not found: {TRAJECTORY}")
    if N_ACN <= 0 or ATOMS_PER_ACN <= 0:
        raise ValueError("N_ACN and ATOMS_PER_ACN must be positive.")
    if SPHERE_RADIUS <= 0.0 or DESIRED_DR <= 0.0:
        raise ValueError("SPHERE_RADIUS and DESIRED_DR must be positive.")
    if SKIP_FRAMES < 0:
        raise ValueError("SKIP_FRAMES cannot be negative.")
    if STRIDE <= 0:
        raise ValueError("STRIDE must be a positive integer.")
    if N_BLOCKS <= 0:
        raise ValueError("N_BLOCKS must be positive.")
    if CENTER_MODE not in {"fixed", "fe"}:
        raise ValueError("CENTER_MODE must be 'fixed' or 'fe'.")
    for r_min, r_max in REPORT_SHELLS:
        if not (0.0 <= r_min < r_max <= SPHERE_RADIUS):
            raise ValueError(
                f"Invalid reporting shell ({r_min}, {r_max}); it must lie "
                f"within 0-{SPHERE_RADIUS} A."
            )


def block_mean_profiles(density_per_frame, requested_blocks):
    """Return mean profiles for contiguous, nonempty trajectory blocks."""
    n_blocks = min(requested_blocks, len(density_per_frame))
    blocks = np.array_split(density_per_frame, n_blocks, axis=0)
    profiles = np.array([block.mean(axis=0) for block in blocks if len(block)])
    return profiles


def block_sem(block_values):
    """Calculate SEM from statistically meaningful contiguous-block means."""
    block_values = np.asarray(block_values, dtype=float)
    if len(block_values) <= 1:
        return np.zeros_like(block_values[0])
    return block_values.std(axis=0, ddof=1) / np.sqrt(len(block_values))


def analyze_shell(
    r_min,
    r_max,
    bin_edges,
    shell_volumes,
    frame_histograms,
    requested_blocks,
    target_density_nm3,
):
    """Return a broad-shell density summary and its block means."""
    # Include only complete radial bins inside the requested interval.
    mask = (bin_edges[:-1] >= r_min) & (bin_edges[1:] <= r_max)
    if not np.any(mask):
        raise ValueError(
            f"No complete radial bins lie within {r_min:.2f}-{r_max:.2f} A."
        )

    selected_volume_A3 = shell_volumes[mask].sum()
    counts_per_frame = frame_histograms[:, mask].sum(axis=1)
    density_per_frame_nm3 = counts_per_frame / selected_volume_A3 * 1000.0

    n_blocks = min(requested_blocks, len(density_per_frame_nm3))
    time_blocks = np.array_split(density_per_frame_nm3, n_blocks)
    shell_block_means = np.array(
        [block.mean() for block in time_blocks if len(block)]
    )

    mean_count = counts_per_frame.mean()
    mean_density_nm3 = density_per_frame_nm3.mean()
    sem_nm3 = (
        shell_block_means.std(ddof=1) / np.sqrt(len(shell_block_means))
        if len(shell_block_means) > 1
        else 0.0
    )

    return {
        "r_min_A": r_min,
        "r_max_A": r_max,
        "shell_volume_A3": selected_volume_A3,
        "mean_acn_count": mean_count,
        "mean_density_nm3": mean_density_nm3,
        "block_sem_nm3": sem_nm3,
        "percent_of_bulk": 100.0 * mean_density_nm3 / target_density_nm3,
        "block_means_nm3": shell_block_means,
    }


def main():
    validate_settings()

    n_bins = int(np.ceil(SPHERE_RADIUS / DESIRED_DR))
    bin_edges = np.linspace(0.0, SPHERE_RADIUS, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    actual_bin_width = bin_edges[1] - bin_edges[0]

    shell_volumes = (
        4.0
        * np.pi
        / 3.0
        * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    )

    target_density_A3 = (
        TARGET_MASS_DENSITY
        * AVOGADRO
        / ACN_MOLAR_MASS
        / 1.0e24
    )
    target_density_nm3 = target_density_A3 * 1000.0

    frame_histograms = []
    solute_maximum_radii = []
    fe_center_displacements = []
    outside_count = 0
    total_acn_observations = 0

    acn_groups = None
    fe_index = None
    n_complex_atoms = None
    reference_symbols = None

    for raw_frame_number, atoms in enumerate(iread(TRAJECTORY, index=":")):
        if raw_frame_number < SKIP_FRAMES:
            continue
        if (raw_frame_number - SKIP_FRAMES) % STRIDE != 0:
            continue

        positions = atoms.get_positions()
        symbols = np.asarray(atoms.get_chemical_symbols())
        masses = atoms.get_masses()

        if acn_groups is None:
            n_atoms = len(atoms)
            n_acn_atoms = N_ACN * ATOMS_PER_ACN
            n_complex_atoms = n_atoms - n_acn_atoms

            if n_complex_atoms <= 0:
                raise ValueError(
                    "The inferred number of complex atoms is not positive. "
                    "Check N_ACN and ATOMS_PER_ACN."
                )

            # Assumes complex first, followed by consecutive ACN molecules.
            acn_groups = np.arange(n_complex_atoms, n_atoms).reshape(
                N_ACN, ATOMS_PER_ACN
            )

            expected_composition = Counter({"C": 2, "H": 3, "N": 1})
            incorrect_groups = [
                index
                for index, group in enumerate(acn_groups)
                if Counter(symbols[group]) != expected_composition
            ]
            if incorrect_groups:
                preview = incorrect_groups[:10]
                raise ValueError(
                    "The inferred solvent groups are not consecutive six-atom "
                    f"ACN molecules. Incorrect group indices include {preview}."
                )

            fe_indices = np.where(symbols == "Fe")[0]
            if len(fe_indices) != 1:
                raise ValueError(
                    f"Expected exactly one Fe atom, but found {len(fe_indices)}."
                )
            fe_index = int(fe_indices[0])
            reference_symbols = symbols.copy()

            print(f"Total atoms: {n_atoms}")
            print(f"Inferred complex atoms: {n_complex_atoms}")
            print(f"Fe atom index (zero-based): {fe_index}")
            print(
                f"Target ACN density: {target_density_A3:.6f} "
                "molecules/A^3"
            )
            print(f"Actual radial-bin width: {actual_bin_width:.4f} A")

        elif len(atoms) != len(reference_symbols) or not np.array_equal(
            symbols, reference_symbols
        ):
            raise ValueError(
                f"Atom count or ordering changed at raw frame {raw_frame_number}."
            )

        if CENTER_MODE == "fixed":
            center = FIXED_CENTER
        else:
            center = positions[fe_index]

        # Monitor Fe motion even when the density origin is the fixed wall center.
        fe_center_displacements.append(
            np.linalg.norm(positions[fe_index] - FIXED_CENTER)
        )

        group_positions = positions[acn_groups]  # shape: (N_ACN, atoms/ACN, 3)
        group_masses = masses[acn_groups]  # shape: (N_ACN, atoms/ACN)
        acn_centers = (
            (group_positions * group_masses[:, :, None]).sum(axis=1)
            / group_masses.sum(axis=1)[:, None]
        )

        radial_distances = np.linalg.norm(acn_centers - center[None, :], axis=1)
        outside_count += int(np.count_nonzero(radial_distances > SPHERE_RADIUS))
        total_acn_observations += len(radial_distances)

        histogram, _ = np.histogram(radial_distances, bins=bin_edges)
        frame_histograms.append(histogram)

        complex_distances = np.linalg.norm(
            positions[:n_complex_atoms] - center[None, :], axis=1
        )
        solute_maximum_radii.append(complex_distances.max())

    frame_histograms = np.asarray(frame_histograms, dtype=float)
    if len(frame_histograms) == 0:
        raise RuntimeError("No trajectory frames were analyzed.")

    density_per_frame_A3 = frame_histograms / shell_volumes[None, :]
    mean_density_A3 = density_per_frame_A3.mean(axis=0)
    radial_block_profiles_A3 = block_mean_profiles(
        density_per_frame_A3, N_BLOCKS
    )
    density_sem_A3 = block_sem(radial_block_profiles_A3)

    mean_density_nm3 = mean_density_A3 * 1000.0
    density_sem_nm3 = density_sem_A3 * 1000.0
    normalized_density = mean_density_A3 / target_density_A3
    radial_block_profiles_nm3 = radial_block_profiles_A3 * 1000.0

    average_solute_radius = float(np.mean(solute_maximum_radii))
    fe_center_displacements = np.asarray(fe_center_displacements)

    # ------------------------------------------------------------------------
    # Save mean radial profile.
    # ------------------------------------------------------------------------
    radial_output = np.column_stack(
        [
            bin_centers,
            mean_density_A3,
            mean_density_nm3,
            density_sem_nm3,
            normalized_density,
        ]
    )
    np.savetxt(
        RADIAL_CSV,
        radial_output,
        delimiter=",",
        header=(
            "r_A,density_molecules_per_A3,density_molecules_per_nm3,"
            "block_SEM_molecules_per_nm3,density_over_bulk"
        ),
        comments="",
    )

    # Save individual contiguous-block profiles.
    block_header = "r_A," + ",".join(
        f"block_{index + 1}_density_molecules_per_nm3"
        for index in range(len(radial_block_profiles_nm3))
    )
    block_output = np.column_stack(
        [bin_centers, radial_block_profiles_nm3.T]
    )
    np.savetxt(
        BLOCK_CSV,
        block_output,
        delimiter=",",
        header=block_header,
        comments="",
    )

    # ------------------------------------------------------------------------
    # Broad-shell averages.
    # ------------------------------------------------------------------------
    shell_results = [
        analyze_shell(
            r_min,
            r_max,
            bin_edges,
            shell_volumes,
            frame_histograms,
            N_BLOCKS,
            target_density_nm3,
        )
        for r_min, r_max in REPORT_SHELLS
    ]

    max_shell_blocks = max(
        len(result["block_means_nm3"]) for result in shell_results
    )
    shell_header_fields = [
        "r_min_A",
        "r_max_A",
        "shell_volume_A3",
        "mean_acn_count",
        "mean_density_molecules_per_nm3",
        "block_SEM_molecules_per_nm3",
        "percent_of_bulk",
    ] + [
        f"block_{index + 1}_density_molecules_per_nm3"
        for index in range(max_shell_blocks)
    ]

    shell_rows = []
    for result in shell_results:
        block_values = list(result["block_means_nm3"])
        block_values.extend([np.nan] * (max_shell_blocks - len(block_values)))
        shell_rows.append(
            [
                result["r_min_A"],
                result["r_max_A"],
                result["shell_volume_A3"],
                result["mean_acn_count"],
                result["mean_density_nm3"],
                result["block_sem_nm3"],
                result["percent_of_bulk"],
                *block_values,
            ]
        )

    np.savetxt(
        SHELL_CSV,
        np.asarray(shell_rows),
        delimiter=",",
        header=",".join(shell_header_fields),
        comments="",
    )

    # ------------------------------------------------------------------------
    # Mean radial-density plot.
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        bin_centers,
        mean_density_nm3,
        color="navy",
        linewidth=2.2,
        label="ACN COM density",
    )
    ax.fill_between(
        bin_centers,
        np.maximum(mean_density_nm3 - density_sem_nm3, 0.0),
        mean_density_nm3 + density_sem_nm3,
        color="navy",
        alpha=0.20,
        label="Block SEM",
    )
    ax.axhline(
        target_density_nm3,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Bulk target: {target_density_nm3:.2f} nm$^{{-3}}$",
    )
    ax.axvline(
        average_solute_radius,
        color="gray",
        linestyle=":",
        linewidth=2.0,
        label=(
            "Maximum solute-atom radius: "
            f"{average_solute_radius:.2f} A"
        ),
    )
    ax.set_xlabel("Distance from spherical-boundary center (A)")
    ax.set_ylabel("ACN number density (molecules nm$^{-3}$)")
    ax.set_xlim(0.0, SPHERE_RADIUS)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MEAN_PLOT, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Individual block-profile plot.
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(
        np.linspace(0.05, 0.95, len(radial_block_profiles_nm3))
    )
    for index, (profile, color) in enumerate(
        zip(radial_block_profiles_nm3, colors)
    ):
        ax.plot(
            bin_centers,
            profile,
            color=color,
            linewidth=1.5,
            label=f"Block {index + 1}",
        )
    ax.axhline(
        target_density_nm3,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Bulk target: {target_density_nm3:.2f} nm$^{{-3}}$",
    )
    ax.axvline(
        average_solute_radius,
        color="gray",
        linestyle=":",
        linewidth=2.0,
        label=f"Maximum solute-atom radius: {average_solute_radius:.2f} A",
    )
    ax.set_xlabel("Distance from spherical-boundary center (A)")
    ax.set_ylabel("ACN number density (molecules nm$^{-3}$)")
    ax.set_xlim(0.0, SPHERE_RADIUS)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(BLOCK_PLOT, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Terminal summary.
    # ------------------------------------------------------------------------
    sphere_volume_A3 = 4.0 * np.pi / 3.0 * SPHERE_RADIUS**3
    target_acn_volume_A3 = N_ACN / target_density_A3
    implied_complex_volume_A3 = sphere_volume_A3 - target_acn_volume_A3
    outside_fraction = outside_count / total_acn_observations

    print(f"Frames analyzed: {len(frame_histograms)}")
    print(f"Average maximum solute-atom radius: {average_solute_radius:.3f} A")
    print(f"ACN COM observations outside sphere: {outside_count}")
    print(f"Fraction of ACN COM observations outside: {outside_fraction:.8f}")
    print(f"Mean Fe displacement from fixed center: {fe_center_displacements.mean():.4f} A")
    print(f"Maximum Fe displacement from fixed center: {fe_center_displacements.max():.4f} A")
    print(f"Sphere volume: {sphere_volume_A3:.2f} A^3")
    print(f"Target volume required by {N_ACN} ACN: {target_acn_volume_A3:.2f} A^3")
    print(
        "Implied effective complex volume at the target ACN density: "
        f"{implied_complex_volume_A3:.2f} A^3"
    )

    print("\nBroad-shell density results")
    print("---------------------------")
    for result in shell_results:
        block_text = ", ".join(
            f"{value:.3f}" for value in result["block_means_nm3"]
        )
        print(
            f"{result['r_min_A']:.1f}-{result['r_max_A']:.1f} A: "
            f"count = {result['mean_acn_count']:.2f}, "
            f"density = {result['mean_density_nm3']:.3f} +/- "
            f"{result['block_sem_nm3']:.3f} molecules/nm^3, "
            f"bulk = {result['percent_of_bulk']:.2f}%"
        )
        print(f"  Block densities: [{block_text}] molecules/nm^3")

    print("\nFiles written")
    for filename in [RADIAL_CSV, BLOCK_CSV, SHELL_CSV, MEAN_PLOT, BLOCK_PLOT]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()


