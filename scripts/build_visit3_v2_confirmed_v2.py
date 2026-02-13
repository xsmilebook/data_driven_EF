from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


TASK_SOURCE_MAP = {
    "CPT": "visit3",
    "ColorStroop": "visit3",
    "DT": "visit3",
    "EmotionStroop": "visit3",
    "EmotionSwitch": "visit3",
    "FLANKER": "visit3",
    "FZSS": "visit3",
    "GNG": "visit3",
    "KT": "visit3",
    "Emotion1Back": "visit3",
    "Number2Back": "visit3",
    "Number1Back": "visit3",
    "Emotion2Back": "visit3_v1",
    "Spatial1Back": "visit3_v1",
    "SST": "visit2",
    "DCCS": "visit1",
    "Spatial2Back": "visit3_v2_raw",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build visit3_v2_confirmed from conformed confirmed sources.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", dest="paths_config", default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", default=None)
    return ap.parse_args()


def _norm_answer(v: object) -> object:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in ("left", "right", "none"):
        return s.capitalize()
    return s


def _canon_task(name: str) -> str:
    s = str(name).strip().strip("_- ")
    m = {
        "EmotionStoop": "EmotionStroop",
        "LG": "EmotionSwitch",
        "STROOP": "ColorStroop",
        "SpatialNBack": "Spatial2Back",
        "Emotion2Backformal": "Emotion2Back",
        "Number2Backformal": "Number2Back",
        "Spatial1Backformal": "Spatial1Back",
    }
    if s in m:
        return m[s]
    if "formal" in s.lower():
        s = re.sub("formal", "", s, flags=re.IGNORECASE).strip("_- ")
    return s


def _read_text_auto(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin1")


def _parse_spatial2back_visit3_v2(path: Path) -> tuple[list[object], list[object]]:
    obj = json.loads(_read_text_auto(path).strip())
    trials = None
    for _, v in obj.items():
        if isinstance(v, list):
            trials = v
            break
    if trials is None:
        return [], []

    items: list[object] = []
    answers: list[object] = []
    for tr in trials:
        if not isinstance(tr, dict):
            items.append(None)
            answers.append(None)
            continue
        item = tr.get("picData")
        if item is None:
            item = tr.get("step2_PicName")
        items.append(item)
        answers.append(_norm_answer(tr.get("buttonName")))
    return items, answers


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths_cfg = load_paths_config(args.paths_config, repo_root=repo_root)
    roots = resolve_dataset_roots(paths_cfg, dataset=args.dataset)
    dataset_cfg = load_dataset_config(paths_cfg, dataset_config_path=args.dataset_config, repo_root=repo_root)

    behavioral_cfg = dataset_cfg.get("behavioral", {})
    corrected_rel = behavioral_cfg.get("corrected_app_excel_dir")
    app_seq_rel = behavioral_cfg.get("app_sequence_dir")
    if not corrected_rel or not app_seq_rel:
        raise ValueError("Missing corrected_app_excel_dir or app_sequence_dir in config.")

    run_root = roots["processed_root"] / corrected_rel / "run_corrected_v2"
    conformed_dir = run_root / "conformed"
    conformed_dir.mkdir(parents=True, exist_ok=True)

    visit1 = json.loads((conformed_dir / "visit1_confirmed.json").read_text(encoding="utf-8"))
    visit2 = json.loads((conformed_dir / "visit2_confirmed.json").read_text(encoding="utf-8"))
    visit3 = json.loads((conformed_dir / "visit3_confirmed.json").read_text(encoding="utf-8"))
    visit3_v1 = json.loads((conformed_dir / "visit3_v1_confirmed.json").read_text(encoding="utf-8"))

    sp2_raw = roots["raw_root"] / app_seq_rel / "visit3_v2" / "spatialNback2.1.json"
    if not sp2_raw.exists():
        raise FileNotFoundError(f"Missing Spatial2Back raw source: {sp2_raw}")
    sp2_items, sp2_answers = _parse_spatial2back_visit3_v2(sp2_raw)

    source_objs = {"visit1": visit1, "visit2": visit2, "visit3": visit3, "visit3_v1": visit3_v1}
    out = dict(visit3)
    per_task_source: dict[str, str] = {}

    for task, src in TASK_SOURCE_MAP.items():
        per_task_source[task] = src
        if task not in out:
            out[task] = {}
        row = dict(out[task])
        if src == "visit3_v2_raw":
            row["items"] = sp2_items
            row["answers"] = sp2_answers
            row["items_n_rows"] = len(sp2_items)
            row["answers_n_rows"] = len(sp2_answers)
            row["items_source"] = "app_sequence/visit3_v2/spatialNback2.1.json"
            row["answers_source"] = "app_sequence/visit3_v2/spatialNback2.1.json"
        else:
            src_obj = source_objs[src]
            if task not in src_obj:
                raise KeyError(f"Task {task} not found in source {src}.")
            srow = src_obj[task]
            row["items"] = srow.get("items", [])
            row["answers"] = srow.get("answers", [])
            row["items_n_rows"] = len(row["items"])
            row["answers_n_rows"] = len(row["answers"])
            row["items_source"] = f"{src}_confirmed"
            row["answers_source"] = f"{src}_confirmed"
        out[task] = row

    out_path = conformed_dir / "visit3_v2_confirmed.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "visit3_v2_confirmed_build_meta_v1",
        "task_source_map": per_task_source,
        "spatial2back_raw_source": str(sp2_raw.as_posix()),
        "note": "N-back leading None equivalence is a comparison rule; this file stores canonical source answers directly.",
    }
    (conformed_dir / "visit3_v2_confirmed_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = conformed_dir / "manifest.json"
    manifest = {"schema": "run_corrected_v2_conformed_manifest_v1", "description": "Confirmed sequence files with unified names.", "files": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "files" not in manifest or not isinstance(manifest["files"], list):
            manifest["files"] = []

    files = [f for f in manifest["files"] if f.get("conformed_file") != "visit3_v2_confirmed.json"]
    files.append(
        {
            "conformed_file": "visit3_v2_confirmed.json",
            "source_file": "composed_from_confirmed_sources_and_visit3_v2_raw",
            "source_path": str(conformed_dir.as_posix()),
            "build_meta_file": "visit3_v2_confirmed_build_meta.json",
        }
    )
    manifest["files"] = sorted(files, key=lambda x: x.get("conformed_file", ""))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output={out_path}")
    print("tasks=18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
