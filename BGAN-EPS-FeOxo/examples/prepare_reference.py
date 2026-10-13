"""Copy system-specific reference files into a user-selected work directory."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("feoxo", "tqacl"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.workdir / "dataset"
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent / "reference" / args.system
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            raise FileExistsError("Reference file already exists: %s" % target)
        shutil.copyfile(item, target)
    print("Copied %s reference files to %s" % (args.system, destination))


if __name__ == "__main__":
    main()
