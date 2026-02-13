from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


TASK_SOURCE_MAP = {
    "CPT": "visit1",
    "ColorStroop": "visit3",
    "DCCS": "visit1",
    "DT": "visit1",
    "Emotion1Back": "visit3",
    "Emotion2Back": "visit3_v1",
    "EmotionStroop": "visit3",
    "EmotionSwitch": "visit1",
    "FLANKER": "visit1",
    "FZSS": "visit1",
    "GNG": "visit1",
    "KT": "visit1",
    "Number1Back": "visit1",
    "Number2Back": "visit1",
    "SST": "visit1",
    "Spatial1Back": "visit1",
    "Spatial2Back": "visit1",
    "ZYST": "visit1",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build visit1_v1_confirmed from conformed confirmed sources.")
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
        raise ValueError("Missing corrected_app_excel_dir in config.")

    conformed_dir = roots["processed_root"] / corrected_rel / "run_corrected_v2" / "conformed"
    conformed_dir.mkdir(parents=True, exist_ok=True)

    visit1 = json.loads((conformed_dir / "visit1_confirmed.json").read_text(encoding="utf-8"))
    visit3 = json.loads((conformed_dir / "visit3_confirmed.json").read_text(encoding="utf-8"))
    visit3_v1 = json.loads((conformed_dir / "visit3_v1_confirmed.json").read_text(encoding="utf-8"))
    src_map = {"visit1": visit1, "visit3": visit3, "visit3_v1": visit3_v1}

    out = dict(visit1)
    for task, src in TASK_SOURCE_MAP.items():
        if task not in src_map[src]:
            raise KeyError(f"Task {task} not found in {src}_confirmed")
        row = dict(src_map[src][task])
        row["items_source"] = f"{src}_confirmed"
        row["answers_source"] = f"{src}_confirmed"
        row["items_n_rows"] = len(row.get("items", []))
        row["answers_n_rows"] = len(row.get("answers", []))
        out[task] = row

    out_path = conformed_dir / "visit1_v1_confirmed.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "visit1_v1_confirmed_build_meta_v1",
        "task_source_map": TASK_SOURCE_MAP,
        "notes": [
            "SST follows visit1_confirmed source; comparison-time None/empty equivalence is a matching rule, not a file rewrite rule.",
            "Emotion2Back source is visit3_v1_confirmed.",
        ],
    }
    (conformed_dir / "visit1_v1_confirmed_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = conformed_dir / "manifest.json"
    manifest = {"schema": "run_corrected_v2_conformed_manifest_v1", "description": "Confirmed sequence files with unified names.", "files": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "files" not in manifest or not isinstance(manifest["files"], list):
            manifest["files"] = []

    files = [f for f in manifest["files"] if f.get("conformed_file") != "visit1_v1_confirmed.json"]
    files.append(
        {
            "conformed_file": "visit1_v1_confirmed.json",
            "source_file": "composed_from_visit1_visit3_visit3_v1_confirmed",
            "source_path": str(conformed_dir.as_posix()),
            "build_meta_file": "visit1_v1_confirmed_build_meta.json",
        }
    )
    manifest["files"] = sorted(files, key=lambda x: x.get("conformed_file", ""))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output={out_path}")
    print("tasks=18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
