#!/usr/bin/env python3
"""Real-frame entropy validation for the Fe(PDP) BGAN-EPS profile.

This script performs two calculations on the original trajectory frames:

1. The same Jacobian-corrected histogram/MIST estimator used by BGAN-EPS,
   without any generated configurations.
2. A Cartesian quasi-harmonic (Schlitter) entropy estimate for the same
   carboxylate-arm atom set after alignment of the Fe-PDP scaffold.

All profiles use the eight published d(C60-O25) windows and are referenced
to the MECP-containing window (window 4).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


WINDOW_EDGES_DESC = np.array(
    [3.900, 3.575, 3.250, 2.925, 2.600, 2.275, 1.950, 1.625, 1.300],
    dtype=np.float64,
)
REFERENCE_WINDOW = 3  # zero-based window 4, 2.925-2.600 A
TEMPERATURE_K = 298.15
R_KCAL_MOL_K = 1.987204259e-3

# 1-based atom definitions from the project atom map.
ALIGN_ATOMS_1BASED = list(range(1, 25)) + [52]
ARM_ATOMS_1BASED = list(range(53, 64)) + list(range(65, 75))

ATOMIC_MASS_U = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "FE": 55.845,
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", type=Path, default=project_dir, help="bgan_eps_feoxo directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "validation_results",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("mist", "qh"),
        default=("mist", "qh"),
    )
    parser.add_argument("--bins", nargs="+", type=int, default=(10, 15, 20))
    parser.add_argument("--mist-equal-sample-repeats", type=int, default=0)
    parser.add_argument("--mist-equal-sample-size", type=int, default=1000)
    parser.add_argument("--mist-jackknife", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--qh-equal-sample-repeats", type=int, default=500)
    parser.add_argument("--qh-equal-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def window_masks(rc: np.ndarray) -> list[np.ndarray]:
    masks = []
    for high, low in zip(WINDOW_EDGES_DESC[:-1], WINDOW_EDGES_DESC[1:]):
        # Match the strict inequalities in bgan.py.
        masks.append((rc > low) & (rc < high))
    return masks


def read_topology(topology_path: Path) -> tuple[int, list[str], int, int, int]:
    lines = [line.split() for line in topology_path.read_text().splitlines() if line.strip()]
    atom_count = int(lines[0][0])
    dofs = lines[1:]
    types = [{2: "bond", 3: "angle", 4: "torsion"}[len(items)] for items in dofs]
    bond_count = types.count("bond")
    angle_count = types.count("angle")
    torsion_count = types.count("torsion")
    return atom_count, types, bond_count, angle_count, torsion_count


def jacobian_1d(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "bond":
        return x * x
    if kind == "angle":
        return np.sin(x)
    return np.ones_like(x)


def jacobian_2d(x: np.ndarray, xkind: str, y: np.ndarray, ykind: str) -> np.ndarray:
    jx = jacobian_1d(x, xkind)
    jy = jacobian_1d(y, ykind)
    return jx[:, None] * jy[None, :]


def entropy_1d(values: np.ndarray, bins: int, kind: str) -> float:
    counts, edges = np.histogram(values, bins=bins)
    n = float(counts.sum())
    if n == 0:
        return float("nan")
    dx = edges[1] - edges[0]
    if dx == 0:
        return float("nan")
    # Reproduce the original implementation, which shifts the complete edge
    # array by dx/2 and then uses the first `bins` entries as bin centers.
    centers = edges[:bins] + dx / 2.0
    p = counts.astype(np.float64) / n
    nonzero = p > 0
    denom = jacobian_1d(centers, kind) * dx
    valid = nonzero & (denom > 0)
    value = -np.sum(p[valid] * np.log(p[valid] / denom[valid]))
    value += (np.count_nonzero(nonzero) - 1) / (2.0 * n)
    return float(value)


def entropy_2d(
    x: np.ndarray, y: np.ndarray, bins: int, xkind: str, ykind: str
) -> float:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    n = float(hist.sum())
    if n == 0:
        return float("nan")
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    if dx == 0 or dy == 0:
        return float("nan")
    xcenters = xedges[:bins] + dx / 2.0
    ycenters = yedges[:bins] + dy / 2.0
    p = hist.astype(np.float64) / n
    denom = jacobian_2d(xcenters, xkind, ycenters, ykind) * dx * dy
    valid = (p > 0) & (denom > 0)
    value = -np.sum(p[valid] * np.log(p[valid] / denom[valid]))
    value += (np.count_nonzero(p) - 1) / (2.0 * n)
    return float(value)


def mist_entropy(
    coordinates: np.ndarray,
    types: list[str],
    atom_count: int,
    bond_count: int,
    angle_count: int,
    torsion_count: int,
    bins: int,
) -> float:
    n_dof = coordinates.shape[1]
    s1 = [entropy_1d(coordinates[:, i], bins, types[i]) for i in range(n_dof)]
    mist = []
    for i in range(n_dof - 1):
        mutual_information = []
        for j in range(i + 1, n_dof):
            s2 = entropy_2d(
                coordinates[:, i], coordinates[:, j], bins, types[i], types[j]
            )
            mutual_information.append(s1[i] + s1[j] - s2)
        mist.append(float(np.nanmax(mutual_information)))

    bond_entropy = sum(s1[:bond_count]) * (atom_count - 1) / bond_count
    angle_entropy = (
        sum(s1[bond_count : bond_count + angle_count])
        * (atom_count - 2)
        / angle_count
    )
    torsion_entropy = (
        sum(s1[bond_count + angle_count :])
        * (atom_count - 3)
        / torsion_count
    )
    bond_mist = sum(mist[:bond_count]) * (atom_count - 2) / (bond_count - 1)
    angle_mist = (
        sum(mist[bond_count : bond_count + angle_count])
        * (atom_count - 3)
        / (angle_count - 1)
    )
    torsion_mist = (
        sum(mist[bond_count + angle_count :])
        * (atom_count - 4)
        / (torsion_count - 1)
    )
    dimensionless_entropy = (
        bond_entropy + angle_entropy + torsion_entropy
        - bond_mist - angle_mist - torsion_mist
    )
    return -R_KCAL_MOL_K * TEMPERATURE_K * dimensionless_entropy


def run_real_only_mist(
    dof_path: Path,
    topology_path: Path,
    metadata_path: Path,
    output_dir: Path,
    bins_values: list[int],
    equal_sample_repeats: int,
    equal_sample_size: int,
    run_jackknife: bool,
    seed: int,
) -> list[dict[str, float]]:
    dof = np.load(dof_path).astype(np.float64, copy=False)
    rc = dof[:, 0]
    coordinates = dof[:, 1:]
    atom_count, types, nbond, nangle, ntorsion = read_topology(topology_path)
    if coordinates.shape[1] != len(types):
        raise ValueError(
            "DOF/topology mismatch: %d coordinate columns versus %d topology entries"
            % (coordinates.shape[1], len(types))
        )

    masks = window_masks(rc)
    rows: list[dict[str, float]] = []
    for bins in bins_values:
        raw = []
        for index, mask in enumerate(masks, start=1):
            print(
                "Real-only MIST: bins=%d, window=%d, frames=%d"
                % (bins, index, np.count_nonzero(mask)),
                flush=True,
            )
            raw.append(
                mist_entropy(
                    coordinates[mask], types, atom_count, nbond, nangle, ntorsion, bins
                )
            )
        referenced = np.asarray(raw) - raw[REFERENCE_WINDOW]
        for index, (high, low, mask, raw_value, relative) in enumerate(
            zip(
                WINDOW_EDGES_DESC[:-1],
                WINDOW_EDGES_DESC[1:],
                masks,
                raw,
                referenced,
            ),
            start=1,
        ):
            rows.append(
                {
                    "method": "real_only_mist",
                    "bins": bins,
                    "window": index,
                    "d_CO_high_A": high,
                    "d_CO_low_A": low,
                    "d_CO_mid_A": (high + low) / 2.0,
                    "n_frames": int(np.count_nonzero(mask)),
                    "minus_TS_raw_kcal_mol": raw_value,
                    "delta_minus_TS_MECP_kcal_mol": relative,
                }
            )

    output_path = output_dir / "real_only_mist_profiles.csv"
    write_csv(output_path, rows)

    sensitivity_bins = 15 if 15 in bins_values else bins_values[0]
    if equal_sample_repeats > 0:
        rng = np.random.RandomState(seed)
        indices_by_window = [np.flatnonzero(mask) for mask in masks]
        profiles = np.full(
            (equal_sample_repeats, len(masks)), np.nan, dtype=np.float64
        )
        for repeat in range(equal_sample_repeats):
            raw = []
            for indices in indices_by_window:
                size = min(equal_sample_size, len(indices))
                chosen = rng.choice(indices, size=size, replace=False)
                raw.append(
                    mist_entropy(
                        coordinates[chosen],
                        types,
                        atom_count,
                        nbond,
                        nangle,
                        ntorsion,
                        sensitivity_bins,
                    )
                )
            raw = np.asarray(raw)
            profiles[repeat] = raw - raw[REFERENCE_WINDOW]
            if (repeat + 1) % 10 == 0:
                print(
                    "Real-only MIST equal-size repeats: %d/%d"
                    % (repeat + 1, equal_sample_repeats),
                    flush=True,
                )
        np.save(output_dir / "real_only_mist_equal_sample_profiles.npy", profiles)
        summary_rows = []
        for window, (high, low) in enumerate(
            zip(WINDOW_EDGES_DESC[:-1], WINDOW_EDGES_DESC[1:])
        ):
            summary_rows.append(
                {
                    "window": window + 1,
                    "d_CO_high_A": high,
                    "d_CO_low_A": low,
                    "bins": sensitivity_bins,
                    "sample_size": min(equal_sample_size, len(indices_by_window[window])),
                    "repeats": equal_sample_repeats,
                    "median_kcal_mol": np.nanpercentile(profiles[:, window], 50.0),
                    "95ci_low_kcal_mol": np.nanpercentile(profiles[:, window], 2.5),
                    "95ci_high_kcal_mol": np.nanpercentile(profiles[:, window], 97.5),
                }
            )
        write_csv(output_dir / "real_only_mist_equal_sample_summary.csv", summary_rows)

    if run_jackknife:
        trajectory_ids, trajectory_names = read_trajectory_ids(metadata_path)
        jackknife = np.full((len(trajectory_names), len(masks)), np.nan, dtype=np.float64)
        for trajectory in range(len(trajectory_names)):
            retained = trajectory_ids != trajectory
            raw = []
            for mask in masks:
                raw.append(
                    mist_entropy(
                        coordinates[retained & mask],
                        types,
                        atom_count,
                        nbond,
                        nangle,
                        ntorsion,
                        sensitivity_bins,
                    )
                )
            raw = np.asarray(raw)
            jackknife[trajectory] = raw - raw[REFERENCE_WINDOW]
            if (trajectory + 1) % 5 == 0:
                print(
                    "Real-only MIST trajectory jackknife: %d/%d"
                    % (trajectory + 1, len(trajectory_names)),
                    flush=True,
                )
        np.save(output_dir / "real_only_mist_trajectory_jackknife.npy", jackknife)
        full = np.asarray(
            [
                next(
                    row["delta_minus_TS_MECP_kcal_mol"]
                    for row in rows
                    if row["bins"] == sensitivity_bins and row["window"] == window
                )
                for window in range(1, len(masks) + 1)
            ]
        )
        jack_mean = np.nanmean(jackknife, axis=0)
        n = len(trajectory_names)
        jack_se = np.sqrt((n - 1.0) / n * np.nansum((jackknife - jack_mean) ** 2, axis=0))
        jack_rows = []
        for window, (high, low) in enumerate(
            zip(WINDOW_EDGES_DESC[:-1], WINDOW_EDGES_DESC[1:])
        ):
            jack_rows.append(
                {
                    "window": window + 1,
                    "d_CO_high_A": high,
                    "d_CO_low_A": low,
                    "bins": sensitivity_bins,
                    "full_profile_kcal_mol": full[window],
                    "jackknife_se_kcal_mol": jack_se[window],
                    "normal_95ci_low_kcal_mol": full[window] - 1.96 * jack_se[window],
                    "normal_95ci_high_kcal_mol": full[window] + 1.96 * jack_se[window],
                }
            )
        write_csv(output_dir / "real_only_mist_trajectory_jackknife_summary.csv", jack_rows)
    return rows


def parse_pdb_reference(pdb_path: Path) -> tuple[np.ndarray, list[str]]:
    coordinates = []
    elements = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        coordinates.append([float(fields[-4]), float(fields[-3]), float(fields[-2])])
        elements.append(fields[-1].upper())
    return np.asarray(coordinates, dtype=np.float64), elements


def read_xyz_and_align_arm(
    xyz_path: Path,
    reference: np.ndarray,
    align_indices: np.ndarray,
    arm_indices: np.ndarray,
    expected_frames: int,
) -> np.ndarray:
    aligned_arm = np.empty((expected_frames, len(arm_indices), 3), dtype=np.float64)
    ref_align = reference[align_indices]
    ref_center = ref_align.mean(axis=0)
    ref_centered = ref_align - ref_center

    with xyz_path.open("r") as handle:
        frame = 0
        while True:
            first = handle.readline()
            if not first:
                break
            atom_count = int(first.strip())
            handle.readline()  # comment
            coords = np.empty((atom_count, 3), dtype=np.float64)
            for atom in range(atom_count):
                fields = handle.readline().split()
                coords[atom] = [float(fields[1]), float(fields[2]), float(fields[3])]

            moving_align = coords[align_indices]
            moving_center = moving_align.mean(axis=0)
            moving_centered = moving_align - moving_center
            covariance = moving_centered.T.dot(ref_centered)
            u, _, vt = np.linalg.svd(covariance)
            rotation = u.dot(vt)
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1.0
                rotation = u.dot(vt)
            aligned_arm[frame] = (coords[arm_indices] - moving_center).dot(rotation) + ref_center
            frame += 1
            if frame % 5000 == 0:
                print("Aligned %d XYZ frames" % frame, flush=True)

    if frame != expected_frames:
        raise ValueError("Read %d XYZ frames; expected %d" % (frame, expected_frames))
    return aligned_arm


def read_trajectory_ids(metadata_path: Path) -> tuple[np.ndarray, list[str]]:
    names = []
    with metadata_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["source_trajectory"])
    unique_names = sorted(set(names))
    mapping = {name: index for index, name in enumerate(unique_names)}
    return np.asarray([mapping[name] for name in names], dtype=np.int32), unique_names


def schlitter_minus_ts(coordinates: np.ndarray, masses_u: np.ndarray) -> tuple[float, dict]:
    if coordinates.shape[0] < 2:
        return float("nan"), {"rank": 0, "min_eigenvalue": float("nan")}
    centered = coordinates - coordinates.mean(axis=0)
    covariance_a2 = centered.T.dot(centered) / (coordinates.shape[0] - 1)

    # SI constants. The mass-weighted covariance has units kg*m^2.
    k_b = 1.380649e-23
    hbar = 1.054571817e-34
    avogadro = 6.02214076e23
    amu_kg = 1.66053906660e-27
    angstrom_m = 1.0e-10
    factor = k_b * TEMPERATURE_K * math.e**2 / (hbar * hbar)
    sqrt_mass = np.sqrt(np.repeat(masses_u, 3) * amu_kg)
    mass_cov = covariance_a2 * (angstrom_m**2)
    mass_cov *= sqrt_mass[:, None]
    mass_cov *= sqrt_mass[None, :]
    eigenvalues = np.linalg.eigvalsh(mass_cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    logdet = np.log1p(factor * eigenvalues).sum()
    entropy_j_per_k_molecule = 0.5 * k_b * logdet
    ts_kcal_mol = entropy_j_per_k_molecule * avogadro * TEMPERATURE_K / 4184.0
    qc = {
        "rank": int(np.count_nonzero(eigenvalues > eigenvalues.max() * 1e-12)),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "logdet": float(logdet),
    }
    return -float(ts_kcal_mol), qc


def sufficient_statistics_by_trajectory(
    coordinates: np.ndarray,
    trajectory_ids: np.ndarray,
    masks: list[np.ndarray],
    n_trajectories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = coordinates.shape[1]
    counts = np.zeros((n_trajectories, len(masks)), dtype=np.int64)
    sums = np.zeros((n_trajectories, len(masks), dimension), dtype=np.float64)
    cross = np.zeros(
        (n_trajectories, len(masks), dimension, dimension), dtype=np.float64
    )
    for trajectory in range(n_trajectories):
        trajectory_mask = trajectory_ids == trajectory
        for window, window_mask in enumerate(masks):
            values = coordinates[trajectory_mask & window_mask]
            if len(values) == 0:
                continue
            counts[trajectory, window] = len(values)
            sums[trajectory, window] = values.sum(axis=0)
            cross[trajectory, window] = values.T.dot(values)
    return counts, sums, cross


def schlitter_from_statistics(
    count: int, total: np.ndarray, cross: np.ndarray, masses_u: np.ndarray
) -> float:
    if count < 2:
        return float("nan")
    covariance_a2 = (cross - np.outer(total, total) / count) / (count - 1)
    covariance_a2 = (covariance_a2 + covariance_a2.T) / 2.0
    k_b = 1.380649e-23
    hbar = 1.054571817e-34
    avogadro = 6.02214076e23
    amu_kg = 1.66053906660e-27
    factor = k_b * TEMPERATURE_K * math.e**2 / (hbar * hbar)
    sqrt_mass = np.sqrt(np.repeat(masses_u, 3) * amu_kg)
    mass_cov = covariance_a2 * 1.0e-20
    mass_cov *= sqrt_mass[:, None]
    mass_cov *= sqrt_mass[None, :]
    eigenvalues = np.maximum(np.linalg.eigvalsh(mass_cov), 0.0)
    logdet = np.log1p(factor * eigenvalues).sum()
    entropy = 0.5 * k_b * logdet
    return -float(entropy * avogadro * TEMPERATURE_K / 4184.0)


def bootstrap_qh(
    coordinates: np.ndarray,
    masses_u: np.ndarray,
    trajectory_ids: np.ndarray,
    masks: list[np.ndarray],
    replicates: int,
    seed: int,
) -> np.ndarray:
    n_trajectories = int(trajectory_ids.max()) + 1
    counts, sums, cross = sufficient_statistics_by_trajectory(
        coordinates, trajectory_ids, masks, n_trajectories
    )
    rng = np.random.RandomState(seed)
    profiles = np.full((replicates, len(masks)), np.nan, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.randint(0, n_trajectories, size=n_trajectories)
        multiplicity = np.bincount(selected, minlength=n_trajectories).astype(np.float64)
        raw = []
        for window in range(len(masks)):
            count = int(np.dot(multiplicity, counts[:, window]))
            total = np.tensordot(multiplicity, sums[:, window], axes=(0, 0))
            total_cross = np.tensordot(multiplicity, cross[:, window], axes=(0, 0))
            raw.append(schlitter_from_statistics(count, total, total_cross, masses_u))
        raw = np.asarray(raw)
        profiles[replicate] = raw - raw[REFERENCE_WINDOW]
    return profiles


def run_quasi_harmonic(
    project_dir: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    equal_sample_repeats: int,
    equal_sample_size: int,
    seed: int,
) -> list[dict[str, float]]:
    dataset_dir = project_dir / "dataset"
    log_dir = project_dir / "log" / "feoxo_implicit_all_dOC_8windows"
    rc = np.load(dataset_dir / "feoxo_implicit_all_d_OC.npy").astype(np.float64)
    masks = window_masks(rc)
    reference, elements = parse_pdb_reference(dataset_dir / "mecp_fixed.pdb")
    align_indices = np.asarray([x - 1 for x in ALIGN_ATOMS_1BASED], dtype=np.int64)
    arm_indices = np.asarray([x - 1 for x in ARM_ATOMS_1BASED], dtype=np.int64)

    cache_path = output_dir / "aligned_arm_coordinates.npy"
    if cache_path.exists():
        aligned_arm = np.load(cache_path)
    else:
        aligned_arm = read_xyz_and_align_arm(
            dataset_dir / "feoxo_implicit_all.xyz",
            reference,
            align_indices,
            arm_indices,
            len(rc),
        )
        np.save(cache_path, aligned_arm)

    trajectory_ids, trajectory_names = read_trajectory_ids(
        dataset_dir / "feoxo_implicit_all_metadata.csv"
    )
    if len(trajectory_ids) != len(rc):
        raise ValueError("Metadata length does not match coordinate arrays")

    arm_elements = [elements[i] for i in arm_indices]
    atom_sets = {
        "all_arm_atoms_no_H64": np.arange(len(arm_indices), dtype=np.int64),
        "arm_heavy_atoms": np.asarray(
            [i for i, element in enumerate(arm_elements) if element != "H"],
            dtype=np.int64,
        ),
    }
    rows: list[dict[str, float]] = []
    qc_rows = []
    for set_index, (label, selected_atoms) in enumerate(atom_sets.items()):
        xyz = aligned_arm[:, selected_atoms, :].reshape(len(rc), -1)
        masses = np.asarray(
            [ATOMIC_MASS_U[arm_elements[i]] for i in selected_atoms], dtype=np.float64
        )
        raw = []
        qcs = []
        for mask in masks:
            value, qc = schlitter_minus_ts(xyz[mask], masses)
            raw.append(value)
            qcs.append(qc)
        relative = np.asarray(raw) - raw[REFERENCE_WINDOW]
        bootstrap = bootstrap_qh(
            xyz,
            masses,
            trajectory_ids,
            masks,
            bootstrap_replicates,
            seed + set_index,
        )
        lower = np.nanpercentile(bootstrap, 2.5, axis=0)
        median = np.nanpercentile(bootstrap, 50.0, axis=0)
        upper = np.nanpercentile(bootstrap, 97.5, axis=0)
        standard_error = np.nanstd(bootstrap, axis=0, ddof=1)
        rng = np.random.RandomState(seed + 100 + set_index)
        equal_profiles = np.full(
            (equal_sample_repeats, len(masks)), np.nan, dtype=np.float64
        )
        window_indices = [np.flatnonzero(mask) for mask in masks]
        for repeat in range(equal_sample_repeats):
            equal_raw = []
            for indices in window_indices:
                size = min(equal_sample_size, len(indices))
                chosen = rng.choice(indices, size=size, replace=False)
                value, _ = schlitter_minus_ts(xyz[chosen], masses)
                equal_raw.append(value)
            equal_raw = np.asarray(equal_raw)
            equal_profiles[repeat] = equal_raw - equal_raw[REFERENCE_WINDOW]
        equal_median = np.nanpercentile(equal_profiles, 50.0, axis=0)
        equal_lower = np.nanpercentile(equal_profiles, 2.5, axis=0)
        equal_upper = np.nanpercentile(equal_profiles, 97.5, axis=0)
        for index, (high, low, mask) in enumerate(
            zip(WINDOW_EDGES_DESC[:-1], WINDOW_EDGES_DESC[1:], masks), start=1
        ):
            rows.append(
                {
                    "method": "schlitter_quasi_harmonic",
                    "atom_set": label,
                    "window": index,
                    "d_CO_high_A": high,
                    "d_CO_low_A": low,
                    "d_CO_mid_A": (high + low) / 2.0,
                    "n_frames": int(np.count_nonzero(mask)),
                    "minus_TS_raw_kcal_mol": raw[index - 1],
                    "delta_minus_TS_MECP_kcal_mol": relative[index - 1],
                    "bootstrap_median_kcal_mol": median[index - 1],
                    "bootstrap_se_kcal_mol": standard_error[index - 1],
                    "bootstrap_95ci_low_kcal_mol": lower[index - 1],
                    "bootstrap_95ci_high_kcal_mol": upper[index - 1],
                    "bootstrap_replicates": bootstrap_replicates,
                    "equal_sample_size": min(equal_sample_size, int(np.count_nonzero(mask))),
                    "equal_sample_median_kcal_mol": equal_median[index - 1],
                    "equal_sample_95ci_low_kcal_mol": equal_lower[index - 1],
                    "equal_sample_95ci_high_kcal_mol": equal_upper[index - 1],
                    "equal_sample_repeats": equal_sample_repeats,
                }
            )
            qc_rows.append(
                {
                    "atom_set": label,
                    "window": index,
                    "n_dimensions": xyz.shape[1],
                    "n_frames": int(np.count_nonzero(mask)),
                    **qcs[index - 1],
                }
            )
        np.save(output_dir / ("qh_bootstrap_%s.npy" % label), bootstrap)
        np.save(output_dir / ("qh_equal_sample_%s.npy" % label), equal_profiles)

    write_csv(output_dir / "quasi_harmonic_profiles.csv", rows)
    write_csv(output_dir / "quasi_harmonic_qc.csv", qc_rows)
    (output_dir / "trajectory_index.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(trajectory_names)}, indent=2)
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = project_dir / "log" / "feoxo_implicit_all_dOC_8windows"

    if "mist" in args.methods:
        run_real_only_mist(
            log_dir / "dof.npy",
            log_dir / "topology.txt",
            project_dir / "dataset" / "feoxo_implicit_all_metadata.csv",
            output_dir,
            args.bins,
            args.mist_equal_sample_repeats,
            args.mist_equal_sample_size,
            args.mist_jackknife,
            args.seed,
        )
    if "qh" in args.methods:
        run_quasi_harmonic(
            project_dir,
            output_dir,
            args.bootstrap,
            args.qh_equal_sample_repeats,
            args.qh_equal_sample_size,
            args.seed,
        )

    settings = {
        "temperature_K": TEMPERATURE_K,
        "window_edges_descending_A": WINDOW_EDGES_DESC.tolist(),
        "reference_window_1based": REFERENCE_WINDOW + 1,
        "alignment_atoms_1based": ALIGN_ATOMS_1BASED,
        "arm_atoms_1based": ARM_ATOMS_1BASED,
        "bins": list(args.bins),
        "mist_equal_sample_repeats": args.mist_equal_sample_repeats,
        "mist_equal_sample_size": args.mist_equal_sample_size,
        "mist_jackknife": args.mist_jackknife,
        "bootstrap_replicates": args.bootstrap,
        "qh_equal_sample_repeats": args.qh_equal_sample_repeats,
        "qh_equal_sample_size": args.qh_equal_sample_size,
        "seed": args.seed,
    }
    (output_dir / "analysis_settings.json").write_text(json.dumps(settings, indent=2))
    print("Results written to %s" % output_dir, flush=True)


if __name__ == "__main__":
    main()
