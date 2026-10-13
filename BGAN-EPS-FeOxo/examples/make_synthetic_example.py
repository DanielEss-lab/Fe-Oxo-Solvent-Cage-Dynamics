"""Generate a tiny synthetic internal-coordinate dataset for installation checks.

These arrays are mathematical test inputs, not molecular trajectories.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("work/synthetic"))
    args = parser.parse_args()
    log = args.workdir / "log" / "synthetic"
    dataset = args.workdir / "dataset"
    log.mkdir(parents=True, exist_ok=True)
    dataset.mkdir(parents=True, exist_ok=True)
    if (log / "dof.npy").exists():
        raise FileExistsError("Choose a new work directory; synthetic inputs already exist")
    rng = np.random.RandomState(42)
    n = 256
    rc = rng.uniform(1.31, 3.89, n)
    # Two bonds, two angles and two torsions satisfy the average estimator.
    bonds = rng.uniform(1.2, 1.6, (n, 2))
    angles = rng.uniform(0.9, 2.2, (n, 2))
    torsions = rng.uniform(-1.5, 1.5, (n, 2))
    np.save(log / "dof.npy", np.column_stack((rc, bonds, angles, torsions)))
    np.save(dataset / "synthetic_rc.npy", rc)
    (log / "topology.txt").write_text("5\n1 2\n2 3\n1 2 3\n2 3 4\n1 2 3 4\n2 3 4 5\n")
    print("Synthetic inputs written to %s" % args.workdir)


if __name__ == "__main__":
    main()
