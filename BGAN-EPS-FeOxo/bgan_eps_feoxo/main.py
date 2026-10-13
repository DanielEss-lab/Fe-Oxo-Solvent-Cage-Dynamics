import os
import json
import platform
from pathlib import Path
from importlib.metadata import version
import random
import torch
import argparse
import numpy as np
from .bgan import BGANEPS
from sklearn.preprocessing import MinMaxScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reaction_coordinate(path, expected_length):
    if path.endswith(".npy"):
        rc = np.load(path)
    else:
        rc = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    rc = np.asarray(rc, dtype=float).reshape(-1)
    if rc.shape[0] != expected_length:
        raise ValueError(
            f"Reaction coordinate length mismatch: got {rc.shape[0]} values, "
            f"but dof.npy has {expected_length} snapshots."
        )
    return rc

def main(argv=None):
    parser = argparse.ArgumentParser(description="Modified BGAN-EPS: configurable architecture, custom coordinates, and arm-only entropy.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Directory containing dataset/ and log/ (default: current directory)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing bgan loop outputs for this reaction")
    parser.add_argument("--reaction", type=str, default="", help="Name of the reaction file without format tag")
    parser.add_argument(
        "--ensemble", type=int, default=10, help="Number of structural ensembles for entropic path sampling"
    )
    parser.add_argument(
        "--bondmax", type=float, default=2.897, help="Maximum bond length (i.e., bond length in the optimized TS structure)"
    )
    parser.add_argument("--rcmax", dest="bondmax", type=float, help="Alias for --bondmax when the first column is a custom reaction coordinate")
    parser.add_argument(
        "--bondmin", type=float, default=1.700, help="Minimum bond length (i.e., bond formation criterion)"
    )
    parser.add_argument("--rcmin", dest="bondmin", type=float, help="Alias for --bondmin when the first column is a custom reaction coordinate")
    parser.add_argument(
        "--rc_file",
        type=str,
        default="",
        help="Optional .npy/.txt/.csv file with one reaction-coordinate value per snapshot. Replaces dof.npy column 0.",
    )
    parser.add_argument(
        "--rc_direction",
        choices=["decreasing", "increasing"],
        default="decreasing",
        help="Direction used to build structural windows along the reaction coordinate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=298.15, help="Temperature in configurational entropy calculation"
    )
    parser.add_argument(
        "--eps_type",
        type=str,
        default="average",
        help="Type of entropic path sampling: average or max (average recommended)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for BGAN training (64 recommended)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for BGAN training (50 recommended)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of Adam Optimizer in BGAN training (1e-4 recommended)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum1 for Adam Optimizer (0.5 recommended)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Momentum2 for Adam Optimizer (0.999 recommended)")
    parser.add_argument("--hidden_nodes", type=int, default=0, help="Hidden nodes for generators. 0 preserves the original dim-wide model.")
    parser.add_argument("--latent_dim", type=int, default=0, help="Latent-space dimension. 0 preserves the original dim-dimensional latent space.")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins for 1D and 2D entropy estimates.")
    parser.add_argument("--window_points", type=int, default=30, help="Number of moving structural windows in the entropy profile.")
    parser.add_argument("--min_window_samples", type=int, default=100, help="Warn when an augmented window has fewer samples than this value.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Each BGAN-EPS loop adds the loop index.")
    parser.add_argument("--save_augmented", action="store_true", help="Save augmented internal-coordinate arrays for validation/debugging.")
    parser.add_argument(
        "--loop", type=int, default=20, help="Number of BGAN-EPS rounds (5-20 recommended based on available computation resources)"
    )

    args = parser.parse_args(argv)
    if args.reaction == "":
        raise ValueError("--reaction is required")
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.window_points < 1:
        raise ValueError("--window_points must be at least 1")

    if Path(args.reaction).name != args.reaction or args.reaction in (".", "..") or "/" in args.reaction or "\\" in args.reaction:
        raise ValueError("--reaction must be a simple dataset name, not a path")
    if args.loop < 1 or args.epochs < 1 or args.ensemble < 1:
        raise ValueError("--loop, --epochs, and --ensemble must be positive")
    if args.batch_size < 2 or args.lr <= 0 or args.temperature <= 0:
        raise ValueError("Batch size must be >=2; learning rate and temperature must be positive")
    if not np.isfinite([args.bondmin, args.bondmax, args.lr, args.temperature]).all():
        raise ValueError("Coordinate limits, learning rate, and temperature must be finite")
    if args.bondmin >= args.bondmax:
        raise ValueError("--rcmin must be smaller than --rcmax for either direction")
    if args.eps_type != "average":
        raise ValueError("Only --eps_type average is supported by this branch")
    workdir = args.workdir.expanduser().resolve()
    output_dir = str(workdir / "log" / args.reaction)
    if not args.overwrite and any(Path(output_dir).glob("bgan[0-9]*")):
        raise FileExistsError("Existing loop outputs found; choose a new reaction name or pass --overwrite")

    X = np.load(f"{output_dir}/dof.npy")
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("dof.npy must be a 2D array with at least two snapshots")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("dof.npy must contain floating-point coordinates")
    if args.rc_file:
        rc_path = Path(args.rc_file).expanduser()
        if not rc_path.is_absolute():
            rc_path = workdir / rc_path
        X[:, 0] = load_reaction_coordinate(str(rc_path), X.shape[0])
    if not np.isfinite(X).all():
        raise ValueError("Input coordinates contain NaN or infinity")
    topology = [line.split() for line in (Path(output_dir) / "topology.txt").read_text().splitlines() if line.strip()]
    if len(topology) != X.shape[1]:
        raise ValueError("Expected one topology entry per entropy column; column 0 is reserved for the reaction coordinate")
    lengths = [len(items) for items in topology[1:]]
    if any(n not in (2, 3, 4) for n in lengths) or lengths != sorted(lengths):
        raise ValueError("Topology entries must be bonds, then angles, then torsions")
    if int(topology[0][0]) < 4 or any(lengths.count(n) < 2 for n in (2, 3, 4)):
        raise ValueError("The average estimator requires >=4 entropy atoms and >=2 bonds, angles, and torsions")
    if not np.any((X[:, 0] > args.bondmin) & (X[:, 0] < args.bondmax)):
        raise ValueError("No real snapshots lie inside the reaction-coordinate range")
    metadata = {
        "package_version": version("bgan-eps-feoxo"),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "torch", "scikit-learn")},
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "input_shape": list(X.shape),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "loop_seeds": [args.seed + index for index in range(args.loop)],
    }
    (Path(output_dir) / "run_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    dim = X.shape[1]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)

    for loop_index in range(args.loop):
        set_seed(args.seed + loop_index)
        BGANEPS(output_dir).run(args, X, dim, scaler, loop_index)

    ep = np.load(f"{output_dir}/bgan1/eps.npy")
    for i in range(2, args.loop + 1):
        ep = np.vstack((ep, np.load(f"{output_dir}/bgan{i}/eps.npy")))

    np.save(f"{output_dir}/{args.reaction}_eps.npy", ep)


if __name__ == "__main__":
    main()
