"""Plot an entropy_profile.csv produced by bgan-eps-summary."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG, PDF or SVG output path")
    parser.add_argument("--error", choices=("std", "sem", "none"), default="std")
    parser.add_argument("--xlabel", default="Reaction coordinate")
    parser.add_argument("--reference-coordinate", type=float)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with args.profile.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: float(row["rc_mid"]))
    x = np.asarray([float(row["rc_mid"]) for row in rows])
    y = np.asarray([float(row["mean_minus_T_delta_S"]) for row in rows])
    errors = None if args.error == "none" else np.asarray([float(row[args.error]) for row in rows])
    if errors is not None and not np.isfinite(errors).all():
        raise ValueError("Error bars require >=2 loops with finite statistics; use --error none for a single loop")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=errors, fmt="o-", capsize=4, color="#00868b")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    if args.reference_coordinate is not None:
        ax.axvline(args.reference_coordinate, color="#D34A68", linestyle="--", linewidth=1)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(r"Relative $-T\Delta S$ (kcal/mol)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print("Saved %s" % args.output)


if __name__ == "__main__":
    main()
