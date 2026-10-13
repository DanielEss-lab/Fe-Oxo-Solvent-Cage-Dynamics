import os
from pathlib import Path
import numpy as np
import argparse


def get_connectivity(X, nb1, nb2):
    graph_dict = {}
    for atom_pair in X:
        a1, a2 = atom_pair
        index1 = a1.index
        index2 = a2.index

        # Add the indices to each other's neighbor list
        if index1 not in graph_dict:
            graph_dict[index1] = []
        if index2 not in graph_dict:
            graph_dict[index2] = []

        graph_dict[index1].append(index2)
        graph_dict[index2].append(index1)
    graph_dict[nb1 - 1].append(nb2 - 1)
    graph_dict[nb2 - 1].append(nb1 - 1)
    return graph_dict


def get_topology(graph, output_dir):
    import networkx as nx
    G = nx.Graph(graph)
    atom_num = len(graph)
    total_paths = []
    for i in range(atom_num):
        for j in range(atom_num):
            for path in nx.all_simple_paths(G, source=i, target=j):
                if len(path) in [2, 3, 4]:
                    total_paths.append(path)

    unique = []
    for items in total_paths:
        if tuple(items) not in unique and tuple(reversed(items)) not in unique:
            unique.append(tuple(items))
    unique.sort()
    unique.sort(key=len)

    with open(str(Path(output_dir) / "topology.txt"), "w") as fw:
        fw.write(str(atom_num) + "\n")
        for t in unique:
            fw.write(" ".join(str(s + 1) for s in t) + "\n")

    return atom_num, unique


def read_topology_file(topology_file, output_dir):
    with open(topology_file, "r") as fr:
        lines = [line.strip() for line in fr.readlines() if line.strip()]
    atom_num = int(lines[0])
    unique = [tuple(int(item) - 1 for item in line.split()) for line in lines[1:]]

    with open(str(Path(output_dir) / "topology.txt"), "w") as fw:
        fw.write(str(atom_num) + "\n")
        for t in unique:
            fw.write(" ".join(str(s + 1) for s in t) + "\n")

    return atom_num, unique


def xyz2bat(output_dir, args):
    try:
        import mdtraj as md
    except ImportError as exc:
        raise ImportError("XYZ conversion requires MDTraj; install the preprocess extra or a compatible Conda build") from exc
    traj = str(args.workdir / "dataset" / f"{args.reaction}.xyz")
    trajectory = md.load_xyz(traj, top=str(args.workdir / "dataset" / f"{args.ts}.pdb"))

    reference = md.load(str(args.workdir / "dataset" / f"{args.ts}.pdb"))
    if args.topology_file:
        atom_num, unique = read_topology_file(args.topology_file, output_dir)
    else:
        graph = get_connectivity(reference.top.bonds, args.nb1, args.nb2)
        atom_num, unique = get_topology(graph, output_dir)

    unique.insert(0, (args.atom1 - 1, args.atom2 - 1))

    bond_list = [items for items in unique if len(items) == 2]
    angle_list = [items for items in unique if len(items) == 3]
    torsion_list = [items for items in unique if len(items) == 4]

    arrays = []
    if bond_list:
        arrays.append(md.compute_distances(trajectory, bond_list) * 10.0)  # nm2ang
    if angle_list:
        arrays.append(md.compute_angles(trajectory, angle_list))
    if torsion_list:
        arrays.append(md.compute_dihedrals(trajectory, torsion_list))
    if not arrays:
        raise ValueError("No bond, angle, or torsion coordinates were generated.")

    snapshot = np.hstack(arrays)
    np.save(str(Path(output_dir) / "dof.npy"), snapshot)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert XYZ snapshots to internal coordinates using a PDB atom order and an optional manual topology.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Directory containing dataset/ and log/")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing dof.npy and topology.txt")
    parser.add_argument("--atom1", type=int, default=0, help="first atom number in reaction coordinate (e.g., bond 2 or bond 3)")
    parser.add_argument("--atom2", type=int, default=0, help="second atom number in reaction coordinate (e.g., bond 2 or bond 3)")
    parser.add_argument("--reaction", type=str, default="", help="Name of the reaction file without format tag")
    parser.add_argument("--ts", type=str, default="", help="Name of the optimized transition state structure file (pdb) without format tag")
    parser.add_argument("--nb1", type=int, default=0, help="first atom number in bond 1")
    parser.add_argument("--nb2", type=int, default=0, help="second atom number in bond 1")
    parser.add_argument(
        "--topology_file",
        type=str,
        default="",
        help="Optional topology file to use instead of PDB-inferred connectivity. First line is atom count; remaining lines are 1-based bond/angle/torsion atom indices.",
    )

    args = parser.parse_args(argv)
    args.workdir = args.workdir.expanduser().resolve()
    for name in (args.reaction, args.ts):
        if not name or Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("--reaction and --ts must be simple dataset names, not paths")
    if args.topology_file:
        topology_path = Path(args.topology_file).expanduser()
        if not topology_path.is_absolute():
            topology_path = args.workdir / topology_path
        args.topology_file = str(topology_path)
    if args.atom1 <= 0 or args.atom2 <= 0:
        raise ValueError(
            "--atom1 and --atom2 are required because dof.npy column 0 is the reaction coordinate. "
            "If you plan to use --rc_file in main.py, provide any valid placeholder atom pair here."
        )
    if not args.topology_file and (args.nb1 <= 0 or args.nb2 <= 0):
        raise ValueError("--nb1 and --nb2 are required unless --topology_file is provided.")
    if args.reaction == "":
        raise ValueError("--reaction is required.")

    output_dir = str(args.workdir / "log" / args.reaction)
    if not args.overwrite and any((Path(output_dir) / name).exists() for name in ("dof.npy", "topology.txt")):
        raise FileExistsError("Existing converted coordinates found; use a new reaction name or pass --overwrite")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xyz2bat(output_dir, args)


if __name__ == "__main__":
    main()
