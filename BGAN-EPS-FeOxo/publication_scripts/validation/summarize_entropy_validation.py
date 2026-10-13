#!/usr/bin/env python3
"""Combine BGAN-EPS, real-only MIST, and quasi-harmonic validation results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "real_frame_entropy_results"
BGAN_DIR = PROJECT / "log" / "feoxo_implicit_all_dOC_8windows"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def contrasts(profile: np.ndarray) -> tuple[float, float]:
    # Pre-specified overall contrast and a transition-region contrast that
    # excludes the sparsest expanded endpoint and relaxed product basin.
    overall = float(np.mean(profile[4:8]) - np.mean(profile[0:3]))
    transition = float(np.mean(profile[4:6]) - np.mean(profile[1:3]))
    return overall, transition


def main() -> None:
    global RESULTS, BGAN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bgan-dir", type=Path, required=True)
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    BGAN_DIR = args.bgan_dir.resolve()
    bgan_rows = read_csv(BGAN_DIR / "d_CO_entropy_profile_MECP_reference.csv")
    bgan = np.asarray(
        [float(row["mean_minus_T_delta_S_MECP_ref"]) for row in bgan_rows]
    )
    bgan_sem = np.asarray([float(row["sem"]) for row in bgan_rows])

    mist_rows = read_csv(RESULTS / "real_only_mist_profiles.csv")
    mist15_rows = [row for row in mist_rows if int(row["bins"]) == 15]
    mist15 = np.asarray(
        [float(row["delta_minus_TS_MECP_kcal_mol"]) for row in mist15_rows]
    )
    jack_rows = read_csv(RESULTS / "real_only_mist_trajectory_jackknife_summary.csv")
    mist_low = np.asarray([float(row["normal_95ci_low_kcal_mol"]) for row in jack_rows])
    mist_high = np.asarray([float(row["normal_95ci_high_kcal_mol"]) for row in jack_rows])

    qh_rows = read_csv(RESULTS / "quasi_harmonic_profiles.csv")
    qh_all_rows = [row for row in qh_rows if row["atom_set"] == "all_arm_atoms_no_H64"]
    qh = np.asarray(
        [float(row["delta_minus_TS_MECP_kcal_mol"]) for row in qh_all_rows]
    )
    qh_low = np.asarray(
        [float(row["bootstrap_95ci_low_kcal_mol"]) for row in qh_all_rows]
    )
    qh_high = np.asarray(
        [float(row["bootstrap_95ci_high_kcal_mol"]) for row in qh_all_rows]
    )

    combined = []
    for index, source in enumerate(bgan_rows):
        shared = {
            "window": index + 1,
            "d_CO_high_A": source["d_CO_high"],
            "d_CO_low_A": source["d_CO_low"],
            "d_CO_mid_A": source["d_CO_mid"],
        }
        combined.extend(
            [
                {
                    **shared,
                    "method": "BGAN-EPS augmented MIST",
                    "delta_minus_TS_kcal_mol": bgan[index],
                    "uncertainty_type": "SEM across 5 BGAN loops",
                    "uncertainty_low": bgan[index] - bgan_sem[index],
                    "uncertainty_high": bgan[index] + bgan_sem[index],
                },
                {
                    **shared,
                    "method": "Real-frame-only MIST (15 bins)",
                    "delta_minus_TS_kcal_mol": mist15[index],
                    "uncertainty_type": "95% normal interval from trajectory jackknife",
                    "uncertainty_low": mist_low[index],
                    "uncertainty_high": mist_high[index],
                },
                {
                    **shared,
                    "method": "Schlitter quasi-harmonic (all arm atoms)",
                    "delta_minus_TS_kcal_mol": qh[index],
                    "uncertainty_type": "95% trajectory-bootstrap interval",
                    "uncertainty_low": qh_low[index],
                    "uncertainty_high": qh_high[index],
                },
            ]
        )
    write_csv(RESULTS / "combined_entropy_validation_profiles.csv", combined)

    metric_rows = []
    for name, profile in (
        ("BGAN-EPS augmented MIST", bgan),
        ("Real-frame-only MIST (15 bins)", mist15),
        ("Schlitter quasi-harmonic (all arm atoms)", qh),
    ):
        overall, transition = contrasts(profile)
        metric_rows.append(
            {
                "method": name,
                "overall_contrast_W5to8_minus_W1to3_kcal_mol": overall,
                "transition_contrast_W5to6_minus_W2to3_kcal_mol": transition,
                "spearman_vs_BGAN_all_nonreference_windows": 1.0
                if name == "BGAN-EPS augmented MIST"
                else spearman(
                    np.delete(profile, 3),
                    np.delete(bgan, 3),
                ),
                "rms_difference_vs_BGAN_kcal_mol": 0.0
                if name == "BGAN-EPS augmented MIST"
                else float(np.sqrt(np.mean((profile - bgan) ** 2))),
            }
        )
    write_csv(RESULTS / "entropy_validation_method_metrics.csv", metric_rows)

    distribution_rows = []
    distributions = {
        "BGAN-EPS five loops": np.asarray(
            [
                [float(row["bgan%d_MECP_ref" % loop]) for row in read_csv(BGAN_DIR / "d_CO_loop_profiles.csv")]
                for loop in range(1, 6)
            ]
        ),
        "Real-only MIST equal-size repeats": np.load(
            RESULTS / "real_only_mist_equal_sample_profiles.npy"
        ),
        "Schlitter QH trajectory bootstrap": np.load(
            RESULTS / "qh_bootstrap_all_arm_atoms_no_H64.npy"
        ),
        "Schlitter QH equal-size repeats": np.load(
            RESULTS / "qh_equal_sample_all_arm_atoms_no_H64.npy"
        ),
    }
    for name, profiles in distributions.items():
        values = np.asarray([contrasts(profile) for profile in profiles])
        for column, label in enumerate(("overall", "transition")):
            distribution_rows.append(
                {
                    "method": name,
                    "contrast": label,
                    "n_replicates": len(values),
                    "median_kcal_mol": np.nanpercentile(values[:, column], 50.0),
                    "95ci_low_kcal_mol": np.nanpercentile(values[:, column], 2.5),
                    "95ci_high_kcal_mol": np.nanpercentile(values[:, column], 97.5),
                    "fraction_positive": np.nanmean(values[:, column] > 0),
                }
            )
    write_csv(RESULTS / "entropy_validation_contrast_distributions.csv", distribution_rows)

    print("Wrote combined profiles and validation metrics to %s" % RESULTS)


if __name__ == "__main__":
    main()
