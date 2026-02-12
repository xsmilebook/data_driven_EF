from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.path_config import load_dataset_config, load_paths_config, resolve_dataset_roots


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build visit2_confirmed.json by combining visit2-1125 tasks with "
            "visit3_confirmed fallback tasks in run_corrected_v2/conformed."
        )
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", dest="paths_config", default="configs/paths.yaml")
    ap.add_argument("--dataset-config", dest="dataset_config", default=None)
    return ap.parse_args()


def _canon_task(name: str) -> str:
    s = str(name).strip().strip("_- ")
    mapping = {
        "EmotionStoop": "EmotionStroop",
        "LG": "EmotionSwitch",
        "STROOP": "ColorStroop",
        "SpatialNBack": "Spatial2Back",
        "Emotion2Backformal": "Emotion2Back",
        "Number2Backformal": "Number2Back",
        "Spatial1Backformal": "Spatial1Back",
    }
    if s in mapping:
        return mapping[s]
    if "formal" in s.lower():
        s = re.sub("formal", "", s, flags=re.IGNORECASE).strip("_- ")
    return s


def _task_from_visit2_filename(name: str) -> str:
    stem = Path(name).stem
    special = {
        "DCCS_Formal20251125": "DCCS",
        "Number1Back_Formal_Shuffled2": "Number1Back",
        "stroop1128": "ColorStroop",
        "SpatialNBack_Formal": "Spatial2Back",
        "ZYST_Formal2": "ZYST",
    }
    return special.get(stem, _canon_task(stem))


def _read_text_auto(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin1")


def _load_json_auto(path: Path) -> dict[str, Any]:
    return json.loads(_read_text_auto(path).strip())


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


def _first_list_value(obj: dict[str, Any]) -> list[Any]:
    for _, v in obj.items():
        if isinstance(v, list):
            return v
    return []


def _parse_ordinary_items(trials: list[Any]) -> list[object]:
    out: list[object] = []
    for tr in trials:
        if not isinstance(tr, dict):
            out.append(None)
            continue
        item = None
        for key in ("step2_PicName", "picData", "picName", "itemName"):
            if key in tr:
                item = tr[key]
                break
        out.append(item)
    return out


def _parse_visit2_task(task: str, path: Path) -> tuple[list[object], list[object]]:
    obj = _load_json_auto(path)

    if task == "ZYST":
        # Prefer ZYST_Formal.txt; supports wrapped ZYST_Formal2.json as fallback.
        zyst_items = None
        if "ZYST_Formal_Items" in obj and isinstance(obj["ZYST_Formal_Items"], list):
            zyst_items = obj["ZYST_Formal_Items"]
        elif "" in obj and isinstance(obj[""], dict):
            inner = obj[""]
            if isinstance(inner.get("ZYST_Formal_Items"), list):
                zyst_items = inner["ZYST_Formal_Items"]
        if zyst_items is None:
            zyst_items = _first_list_value(obj)

        answers: list[object] = []
        for blk in zyst_items:
            if not isinstance(blk, dict):
                continue
            for ans in blk.get("answers", []):
                if isinstance(ans, dict):
                    answers.append(_norm_answer(ans.get("answerPicName")))
                else:
                    answers.append(None)
        items = [None] * len(answers)
        return items, answers

    if task == "KT":
        blocks = _first_list_value(obj)
        answers: list[object] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            for ans in blk.get("answers", []):
                if isinstance(ans, dict):
                    answers.append(_norm_answer(ans.get("answerPicName")))
                else:
                    answers.append(None)
        items: list[object] = [answers[0] if answers else None]
        if len(answers) > 1:
            items.extend([None] * (len(answers) - 1))
        return items, answers

    trials = _first_list_value(obj)
    items = _parse_ordinary_items(trials)
    answers: list[object] = []
    for tr in trials:
        if not isinstance(tr, dict):
            answers.append(None)
            continue
        if task == "CPT":
            answers.append(_norm_answer(tr.get("needClickButton")))
            continue
        if task == "DCCS":
            v = tr.get("buttonName")
            if v in (0, "0"):
                answers.append("Left")
            elif v in (1, "1"):
                answers.append("Right")
            else:
                answers.append(_norm_answer(v))
            continue

        v = None
        for key in ("buttonName", "ButtonName", "answer", "Answer", "correctAnswer", "CorrectAnswer"):
            if key in tr:
                v = tr[key]
                break
        answers.append(_norm_answer(v))

    # Rule: for SST compare/keep first 96 trials.
    if task == "SST":
        items = [None] * min(96, len(answers))
        answers = answers[:96]

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
        raise ValueError("Missing corrected_app_excel_dir or app_sequence_dir in dataset.behavioral config.")

    run_root = roots["processed_root"] / corrected_rel / "run_corrected_v2"
    conformed_dir = run_root / "conformed"
    conformed_dir.mkdir(parents=True, exist_ok=True)
    visit3_confirmed = conformed_dir / "visit3_confirmed.json"
    if not visit3_confirmed.exists():
        raise FileNotFoundError(f"Missing base file: {visit3_confirmed}")

    visit2_dir = roots["raw_root"] / app_seq_rel / "visit2-1125"
    if not visit2_dir.exists():
        raise FileNotFoundError(f"Missing visit2 directory: {visit2_dir}")

    base = json.loads(visit3_confirmed.read_text(encoding="utf-8"))

    visit2_templates: dict[str, dict[str, list[object]]] = {}
    for p in sorted(visit2_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in (".json", ".txt"):
            continue
        task = _task_from_visit2_filename(p.name)
        if task == "ZYST" and p.name != "ZYST_Formal.txt":
            # User-confirmed source preference.
            continue
        items, answers = _parse_visit2_task(task, p)
        visit2_templates[task] = {"items": items, "answers": answers, "source_file": p.name}

    out = dict(base)
    for task, rec in visit2_templates.items():
        if task not in out:
            out[task] = {}
        row = dict(out[task])
        row["items"] = rec["items"]
        row["answers"] = rec["answers"]
        row["items_n_rows"] = len(rec["items"])
        row["answers_n_rows"] = len(rec["answers"])
        row["items_source"] = f"visit2-1125:{rec['source_file']}"
        row["answers_source"] = f"visit2-1125:{rec['source_file']}"
        out[task] = row

    out_path = conformed_dir / "visit2_confirmed.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "visit2_confirmed_build_meta_v1",
        "base_file": str(visit3_confirmed.as_posix()),
        "visit2_dir": str(visit2_dir.as_posix()),
        "rules": {
            "visit2_tasks": "use visit2-1125 items+answers",
            "missing_tasks_in_visit2": "fallback to visit3_confirmed",
            "SST": "truncate to first 96 trials",
            "ZYST": "use ZYST_Formal.txt instead of ZYST_Formal2.json",
            "KT": "answers from answers[].answerPicName",
            "DCCS": "map buttonName 0/1 to Left/Right",
            "CPT": "answer from needClickButton",
        },
        "visit2_task_count": len(visit2_templates),
        "visit2_tasks": sorted(visit2_templates.keys()),
    }
    (conformed_dir / "visit2_confirmed_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = conformed_dir / "manifest.json"
    manifest = {"schema": "run_corrected_v2_conformed_manifest_v1", "description": "Confirmed sequence files with unified names.", "files": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "files" not in manifest or not isinstance(manifest["files"], list):
            manifest["files"] = []

    files = [f for f in manifest["files"] if f.get("conformed_file") != "visit2_confirmed.json"]
    files.append(
        {
            "conformed_file": "visit2_confirmed.json",
            "source_file": "composed_from_visit2-1125_and_visit3_confirmed",
            "source_path": str(conformed_dir.as_posix()),
            "build_meta_file": "visit2_confirmed_build_meta.json",
        }
    )
    manifest["files"] = sorted(files, key=lambda x: x.get("conformed_file", ""))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"output={out_path}")
    print(f"visit2_tasks={len(visit2_templates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
