import argparse
import sys
from pathlib import Path

## command:
# python get_mri_sublist.py --dir /ibmgpfs/cuizaixu_lab/liyang/BrainProject25/Tsinghua_data/results/fmriprep_rest --out  /ibmgpfs/cuizaixu_lab/xuhaoshu/code/data_driven_EF/data/EFNY/table/sublist/mri_sublist.txt

def list_subject_dirs(root: Path) -> list[str]:
    names = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("sub-"):
            names.append(p.name[4:])
    names.sort()
    return names

def write_list(names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        for n in names:
            f.write(n + "\n")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", default="mri_sublist.txt")
    args = parser.parse_args()
    root = Path(args.dir)
    if not root.exists():
        print(f"Input directory not found: {root}", file=sys.stderr)
        return
    names = list_subject_dirs(root)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out.name
    write_list(names, out)
    print(f"count={len(names)}")
    print(f"saved={out}")

if __name__ == "__main__":
    main()