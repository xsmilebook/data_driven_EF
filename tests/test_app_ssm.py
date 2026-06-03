from __future__ import annotations

import unittest

import pandas as pd

from src.behavior.app.ssm import build_ssm_trials
from src.common import load_yaml


def _row(
    task: str,
    trial_index: int,
    answer: object = "left",
    key: object = "left",
    rt: float | None = 0.5,
    item: object = "item",
    valid_for_rt: bool = True,
    ssrt_or_ssd: object = None,
) -> dict[str, object]:
    return {
        "dataset": "THU",
        "subject_code": "s1",
        "workbook": "s1.xlsx",
        "task": task,
        "trial_index": trial_index,
        "subject_id": "sub-1",
        "name": "name",
        "item": item,
        "answer": answer,
        "key": key,
        "rt": rt,
        "ssrt_or_ssd": ssrt_or_ssd,
        "blank_screen_duration": None,
        "total_duration": None,
        "cross_pic_duration": None,
        "correct_trial": answer == key,
        "valid_for_acc": True,
        "valid_for_rt": valid_for_rt,
        "exclusion_reason": "" if valid_for_rt else "rt_missing",
    }


class AppSSMTrialBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml("configs/behavioral_metrics.yaml")

    def test_dccs_block_uses_first_20_pure_then_mixed(self) -> None:
        trials = pd.DataFrame([_row("DCCS", index) for index in range(1, 41)])
        model_trials = build_ssm_trials(trials, self.config, "dccs_block_ddm")
        first_20 = model_trials.loc[model_trials["trial_index"].le(20), "block"]
        last_20 = model_trials.loc[model_trials["trial_index"].gt(20), "block"]
        self.assertEqual(set(first_20), {"pure"})
        self.assertEqual(set(last_20), {"mixed"})

    def test_nback_domain_split_omits_target_type(self) -> None:
        trials = pd.DataFrame(
            [
                _row("Number1Back", 1),
                _row("Number2Back", 2),
                _row("Spatial1Back", 1),
                _row("Emotion2Back", 1),
            ]
        )
        model_trials = build_ssm_trials(trials, self.config, "nback_number_ddm")
        self.assertEqual(set(model_trials["task"]), {"Number1Back", "Number2Back"})
        self.assertEqual(set(model_trials["domain"]), {"number"})
        self.assertNotIn("target_type", model_trials.columns)

    def test_cpt_keeps_only_rt_bearing_response_trials(self) -> None:
        trials = pd.DataFrame(
            [
                _row("CPT", 1, answer=True, key=True, rt=0.5, valid_for_rt=True),
                _row("CPT", 2, answer=False, key=None, rt=None, valid_for_rt=False),
                _row("CPT", 3, answer=False, key=True, rt=0.6, valid_for_rt=True),
                _row("CPT", 4, answer=True, key=None, rt=None, valid_for_rt=False),
            ]
        )
        model_trials = build_ssm_trials(trials, self.config, "cpt_response_ddm")
        self.assertEqual(len(model_trials), 2)
        self.assertEqual(set(model_trials["stimulus_type"]), {"target", "nontarget"})

    def test_dt_cross_axis_response_is_not_dropped(self) -> None:
        trials = pd.DataFrame(
            [
                _row("DT", 1, answer="left", key="up", rt=0.5),
                _row("DT", 2, answer="right", key="right", rt=0.6),
            ]
        )
        model_trials = build_ssm_trials(trials, self.config, "dt_mixing_ddm")
        self.assertEqual(len(model_trials), 2)
        self.assertIn(-1, set(model_trials["response"]))


if __name__ == "__main__":
    unittest.main()
