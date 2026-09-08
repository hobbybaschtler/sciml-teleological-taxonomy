"""Loading of the case-study task descriptors of Section 9.4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .descriptor import Constraints, DataRegime, Kappa, Knowledge, Task, Unknown

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "cases" / "cases.yaml"


@dataclass(frozen=True)
class Case:
    """A task descriptor together with the outcome reported in Table 7."""

    id: str
    task: Task
    expected: dict[str, Any]


def _knowledge(raw: dict[str, Any]) -> Knowledge:
    raw = dict(raw or {})
    kappa = Kappa(raw.pop("kappa", "none"))
    return Knowledge(kappa=kappa, **raw)


def _data(raw: dict[str, Any]) -> DataRegime:
    return DataRegime(**dict(raw or {}))


def _constraints(raw: dict[str, Any]) -> Constraints:
    return Constraints(**dict(raw or {}))


def load_cases(path: str | Path | None = None) -> list[Case]:
    path = Path(path) if path is not None else DEFAULT_CASES
    with path.open(encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)

    cases: list[Case] = []
    for entry in entries:
        task = Task(
            name=entry["name"],
            objective=entry["objective"],
            unknowns=frozenset(Unknown(u) for u in entry["unknowns"]),
            knowledge=_knowledge(entry.get("knowledge", {})),
            data=_data(entry.get("data", {})),
            constraints=_constraints(entry.get("constraints", {})),
            objective_functional=entry.get("objective_functional", False),
            a_type=entry.get("a_type"),
            discrete_type=entry.get("discrete_type"),
            reference=entry.get("reference"),
            notes=entry.get("expected", {}).get("note"),
        )
        cases.append(
            Case(id=entry["id"], task=task, expected=dict(entry.get("expected", {})))
        )
    return cases
