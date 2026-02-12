from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


CONFIRMED_MAPPING = {
    "visit1_confirmed.json": "visit1_merged.json",
    "visit3_confirmed.json": "visit3_from_groups.json",
    "visit3_v1_confirmed.json": "visit3_v1_from_groups.json",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export confirmed sequence JSONs from run_corrected_v2/sequence_library "
            "to run_corrected_v2/conformed with unified names."
        )
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", dest="paths_config", default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    behavioral_cfg = dataset_cfg.get("behavioral", {})
    corrected_rel = behavioral_cfg.get("corrected_app_excel_dir")
    if not corrected_rel:
        raise ValueError("Missing dataset.behavioral.corrected_app_excel_dir in configs/paths.yaml")

    run_root = roots["processed_root"] / corrected_rel / "run_corrected_v2"
    src_dir = run_root / "sequence_library"
    dst_dir = run_root / "conformed"
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for dst_name, src_name in CONFIRMED_MAPPING.items():
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)
        copied.append(
            {
                "conformed_file": dst_name,
                "source_file": src_name,
                "source_path": src.as_posix(),
            }
        )

    manifest = {
        "schema": "run_corrected_v2_conformed_manifest_v1",
        "description": "Confirmed sequence files with unified names.",
        "files": copied,
    }
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"conformed_dir={dst_dir}")
    print(f"copied_n={len(copied)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
