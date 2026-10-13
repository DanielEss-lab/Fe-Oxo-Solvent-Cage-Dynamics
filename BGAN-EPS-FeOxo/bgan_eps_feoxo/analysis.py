"""Summarize EPS logs with a user-selected reference window."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_eps_log(path):
    """Read full window geometry/counts from an EPS log (1-based window IDs)."""
    lines = path.read_text().splitlines()[1:]
    rows = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 7:
            raise ValueError("Expected seven EPS log columns in %s" % path)
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                     float(parts[3]), int(parts[5]), int(parts[6])))
    if not rows or [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("EPS windows must be consecutive and start at 1")
    return rows


def summarize(run_dir, reference_window, output_dir):
    folders = sorted((p for p in run_dir.glob("bgan*")
                      if p.is_dir() and p.name[4:].isdigit()),
                     key=lambda p: int(p.name[4:]))
    if not folders:
        raise FileNotFoundError("No bgan loop directories found in %s" % run_dir)
    logs = [read_eps_log(folder / "eps.log") for folder in folders]
    edges = [(r[1], r[2]) for r in logs[0]]
    for folder, rows in zip(folders, logs):
        if [(r[1], r[2]) for r in rows] != edges:
            raise ValueError("Window bounds differ between loops")
        saved = np.asarray(np.load(folder / "eps.npy"), dtype=float)
        if saved.shape != (len(edges),):
            raise ValueError("eps.npy must contain one value per window")
        if not np.allclose(saved, [r[3] for r in rows], atol=5.1e-5, rtol=0, equal_nan=True):
            raise ValueError("EPS log and array disagree in %s" % folder)
    profiles = np.vstack([np.load(folder / "eps.npy") for folder in folders])
    if not np.isfinite(profiles).all():
        raise ValueError("Nonfinite entropy values found; inspect sparse/empty window logs")
    if not 1 <= reference_window <= profiles.shape[1]:
        raise ValueError("Reference window is outside the profile")
    relative = profiles - profiles[:, [reference_window - 1]]
    mean = relative.mean(axis=0)
    sd = relative.std(axis=0, ddof=1) if len(folders) > 1 else np.full(len(edges), np.nan)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ("window", "rc_high", "rc_low", "rc_mid", "mean_minus_T_delta_S",
              "std", "sem", "n_loops", "real_snapshots", "augmented_snapshots_min",
              "augmented_snapshots_mean", "augmented_snapshots_max", "is_reference")
    with (output_dir / "entropy_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (high, low) in enumerate(edges):
            counts = [rows[index][4] for rows in logs]
            real = [rows[index][5] for rows in logs]
            if len(set(real)) != 1:
                raise ValueError("Real-frame counts differ between loops")
            writer.writerow(dict(zip(fields, (index + 1, high, low, (high + low) / 2,
                mean[index], sd[index], sd[index] / np.sqrt(len(folders)), len(folders),
                real[0], min(counts), np.mean(counts), max(counts), index + 1 == reference_window))))
    with (output_dir / "loop_profiles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window", "rc_mid"] + [p.name + "_ref" for p in folders])
        for index, (high, low) in enumerate(edges):
            writer.writerow([index + 1, (high + low) / 2] + list(relative[:, index]))
    (output_dir / "summary_settings.json").write_text(json.dumps({
        "reference_window_1based": reference_window,
        "loop_directories": [p.name for p in folders],
        "uncertainty": "Sample SD and SEM across training loops; not trajectory sampling uncertainty",
        "window_bounds": "Taken from eps.log (rounded to four decimals by the original code)",
    }, indent=2) + "\n")
    return mean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-window", type=int, required=True, help="1-based window number")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize(args.run_dir.resolve(), args.reference_window, args.output_dir.resolve())
    print("Wrote entropy_profile.csv and loop_profiles.csv to %s" % args.output_dir)


if __name__ == "__main__":
    main()
