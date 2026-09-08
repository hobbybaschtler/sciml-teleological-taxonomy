"""Loader and validator for taxonomy.yaml (Section 8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .descriptor import PRECEDENCE

DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "taxonomy.yaml"


@dataclass(frozen=True)
class Leaf:
    """A leaf of Table 4."""

    id: str
    name: str
    macro_class: str
    recovers: str
    discharges: tuple[str, ...]
    requires: dict[str, Any]
    known_failure_modes: tuple[str, ...]
    references: tuple[str, ...]

    def satisfies(self, constraint: str) -> bool:
        """`a |= c` of Algorithm 1, line 13."""
        return constraint in self.discharges

    def applicable(self, facts: dict[str, Any]) -> bool:
        """Whether the task conditions in `requires` hold.

        A condition whose key is absent from the facts is treated as unmet, so
        that an under-specified descriptor produces a visible empty candidate
        set rather than a silently permissive one.
        """
        for key, expected in self.requires.items():
            if key not in facts or facts[key] != expected:
                return False
        return True


class Taxonomy:
    """The serialised taxonomy, loaded and validated."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.schema_version: str = data["schema_version"]
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.root: dict[str, Any] = data["root"]
        self.leaves: dict[str, Leaf] = {
            entry["id"]: Leaf(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                macro_class=entry["macro_class"],
                recovers=entry["recovers"],
                discharges=tuple(entry.get("discharges") or ()),
                requires=dict(entry.get("requires") or {}),
                known_failure_modes=tuple(entry.get("known_failure_modes") or ()),
                references=tuple(entry.get("references") or ()),
            )
            for entry in data["leaves"]
        }
        self.validate()

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Taxonomy":
        path = Path(path) if path is not None else DEFAULT_SCHEMA
        with path.open(encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    # -- queries -----------------------------------------------------------

    def macro_class_node(self, macro_class: str) -> dict[str, Any]:
        for child in self.root["children"]:
            if child["id"] == macro_class:
                return child
        raise KeyError(f"unknown macro-class {macro_class!r}")

    def leaves_of(self, macro_class: str) -> list[Leaf]:
        """Leaves(X, m) of Algorithm 1, line 6."""
        return [leaf for leaf in self.leaves.values() if leaf.macro_class == macro_class]

    def decision_nodes(self) -> list[dict[str, Any]]:
        """Every node carrying a question, collected in traversal order."""
        found: list[dict[str, Any]] = []

        def walk(entry: Any) -> None:
            if isinstance(entry, dict):
                if "question" in entry and "id" in entry:
                    if not any(n["id"] == entry["id"] for n in found):
                        found.append(entry)
                for key, value in entry.items():
                    if key != "children" or True:
                        walk(value)
            elif isinstance(entry, list):
                for item in entry:
                    walk(item)

        walk(self.root)
        return found

    def observability_ratio(self) -> tuple[int, int]:
        """omega of Table 5: observable nodes over total nodes."""
        nodes = self.decision_nodes()
        observable = sum(1 for n in nodes if n.get("a_priori_observable") is True)
        return observable, len(nodes)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []

        expected_leaves = self.metadata.get("corpus_size")
        if expected_leaves is not None and len(self.leaves) != expected_leaves:
            errors.append(
                f"corpus size: metadata declares {expected_leaves}, "
                f"file defines {len(self.leaves)}"
            )

        expected_nodes = self.metadata.get("decision_nodes")
        nodes = self.decision_nodes()
        if expected_nodes is not None and len(nodes) != expected_nodes:
            errors.append(
                f"decision nodes: metadata declares {expected_nodes}, "
                f"file defines {len(nodes)} ({sorted(n['id'] for n in nodes)})"
            )

        # Every leaf referenced by the tree must exist, and every leaf defined
        # must be reachable. This is the "all objects classified" and "no object
        # duplicated" pair of ending conditions (Table 6).
        referenced: list[str] = []

        def collect(entry: Any) -> None:
            if isinstance(entry, dict):
                referenced.extend(entry.get("leaves", []) or [])
                for value in entry.values():
                    collect(value)
            elif isinstance(entry, list):
                for item in entry:
                    collect(item)

        collect(self.root)

        for leaf_id in referenced:
            if leaf_id not in self.leaves:
                errors.append(f"tree references undefined leaf {leaf_id!r}")
        for leaf_id in self.leaves:
            if leaf_id not in referenced:
                errors.append(f"leaf {leaf_id!r} is defined but unreachable in the tree")
        duplicates = {x for x in referenced if referenced.count(x) > 1}
        if duplicates:
            errors.append(f"leaves reachable by more than one path: {sorted(duplicates)}")

        # Discharged constraints must be ranked requirements (Remark 1).
        for leaf in self.leaves.values():
            for constraint in leaf.discharges:
                if constraint not in PRECEDENCE:
                    errors.append(
                        f"leaf {leaf.id!r} discharges {constraint!r}, which is not a "
                        "ranked requirement; environment facts act as filters and "
                        "belong in `requires`"
                    )

        # No node may be marked unobservable: that is the claim of Table 5.
        for node in nodes:
            if node.get("a_priori_observable") is not True:
                errors.append(f"node {node['id']!r} is not marked a priori observable")

        if errors:
            raise ValueError(
                "taxonomy.yaml failed validation:\n  - " + "\n  - ".join(errors)
            )
