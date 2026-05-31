from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    acc_threshold: float | None
    n_back: int | None = None
    mixed_from: int | None = None


def build_task_registry(config: dict[str, Any]) -> dict[str, TaskSpec]:
    registry: dict[str, TaskSpec] = {}
    for name, values in config["tasks"].items():
        registry[name] = TaskSpec(
            name=name,
            family=values["family"],
            acc_threshold=values.get("acc_threshold"),
            n_back=values.get("n_back"),
            mixed_from=values.get("mixed_from"),
        )
    return registry
