"""Reference implementation of Algorithm 1 and Algorithm 2.

Algorithm 1 is single-class selection with two declared failure modes:
under-determination when several leaves survive, and a relaxation request when
none does. Algorithm 2 composes stages for tasks spanning several macro-classes,
back-propagating constraints before selecting and propagating knowledge forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .descriptor import (
    PRECEDENCE,
    SILENT,
    Constraints,
    Kappa,
    Knowledge,
    MacroClass,
    Task,
)
from .schema import Leaf, Taxonomy


class Outcome(str, Enum):
    """The three terminal states of Algorithm 1."""

    RESOLVED = "resolved"
    UNDER_DETERMINED = "under_determined"
    RELAXATION_REQUESTED = "relaxation_requested"


@dataclass
class Result:
    """Candidate set C with provenance trace pi (Algorithm 1, output)."""

    task: str
    macro_classes: list[MacroClass]
    candidates: list[Leaf] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    outcome: Outcome = Outcome.RESOLVED
    unresolved_criterion: str | None = None
    relaxation: dict[str, Any] | None = None

    @property
    def ids(self) -> list[str]:
        return [leaf.id for leaf in self.candidates]

    def __len__(self) -> int:
        return len(self.candidates)

    def render(self) -> str:
        lines = [f"Task: {self.task}"]
        lines.append(
            "Macro-class(es): " + ", ".join(m.value for m in self.macro_classes)
        )
        lines.append("Trace:")
        lines.extend(f"  {step}" for step in self.trace)
        if self.outcome is Outcome.RELAXATION_REQUESTED:
            assert self.relaxation is not None
            lines.append("Result: NO ADMISSIBLE LEAF - relaxation requested")
            lines.append(
                "  binding constraints: "
                + ", ".join(self.relaxation["binding"])
            )
            lines.append(
                "  weakest constraint proposed for relaxation: "
                + str(self.relaxation["drop"])
            )
            for constraint, ids in self.relaxation["partial"].items():
                lines.append(f"  satisfies {constraint} only: {', '.join(ids)}")
            lines.append("  flagged for expert review")
        elif self.outcome is Outcome.UNDER_DETERMINED:
            lines.append(
                f"Result: UNDER-DETERMINED - {len(self)} survivors: "
                + ", ".join(self.ids)
            )
            lines.append(
                f"  criterion that failed to discriminate: {self.unresolved_criterion}"
            )
        else:
            lines.append("Result: " + ", ".join(self.ids))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------


def select(task: Task, taxonomy: Taxonomy) -> Result:
    """Algorithm 1, extended to Algorithm 2 when several classes are induced."""
    classes = task.macro_classes()
    if len(classes) > 1:
        return compose(task, taxonomy, classes)
    return _select_single(task, taxonomy, classes[0])


def _select_single(
    task: Task, taxonomy: Taxonomy, macro_class: MacroClass
) -> Result:
    result = Result(task=task.name, macro_classes=[macro_class])
    candidates = taxonomy.leaves_of(macro_class.value)
    result.trace.append(
        f"N0: unknown set {{{', '.join(sorted(u.value for u in task.unknowns))}}} "
        f"-> {macro_class.value} ({len(candidates)} leaves)"
    )

    facts = task.facts()

    if macro_class is MacroClass.M1:
        candidates = _filter_by_completeness(task, candidates, result)
    elif macro_class is MacroClass.M2:
        candidates = _filter_m2(task, candidates, result)
    elif macro_class is MacroClass.M3:
        candidates = _filter_m3(task, candidates, result)
    elif macro_class is MacroClass.M4:
        candidates = _filter_m4(task, candidates, result)

    # Task conditions act as filters throughout (Remark 1).
    before = list(candidates)
    candidates = [leaf for leaf in candidates if leaf.applicable(facts)]
    if len(candidates) != len(before):
        dropped = [leaf.id for leaf in before if leaf not in candidates]
        result.trace.append(
            "requires: dropped " + ", ".join(dropped) + " (task conditions unmet)"
        )

    result.candidates = candidates

    if not candidates:
        _request_relaxation(task, taxonomy, macro_class, result)
    elif len(candidates) > 1:
        result.outcome = Outcome.UNDER_DETERMINED
        result.unresolved_criterion = result.trace[-1]

    return result


def _filter_by_completeness(
    task: Task, candidates: list[Leaf], result: Result
) -> list[Leaf]:
    """Algorithm 1, line 9. Node N1.1."""
    kappa = task.knowledge.kappa.value
    kept = [leaf for leaf in candidates if leaf.requires.get("kappa") == kappa]
    result.trace.append(
        f"N1.1: kappa = {kappa} -> " + (", ".join(l.id for l in kept) or "none")
    )

    # A differentiability requirement back-propagated from a downstream control
    # stage by rule (C2) restricts the discovery stage to families whose
    # deliverable is itself differentiable and simulatable.
    if task.constraints.c_diff is True:
        kept = [
            leaf for leaf in kept if leaf.requires.get("differentiable_skeleton") is True
        ]
        result.trace.append(
            "(C2) deliverable must be differentiable -> "
            + (", ".join(l.id for l in kept) or "none")
        )
    return kept


def _filter_m2(task: Task, candidates: list[Leaf], result: Result) -> list[Leaf]:
    """Algorithm 1, lines 12-14. Nodes N2.1-N2.5.

    Reading of line 13: the dominant constraint selects the branch, and every
    other asserted requirement is then applied as a filter. Filtering on the
    dominant constraint alone would return a candidate that violates a second
    stated requirement, which is precisely the situation Case 7 constructs; the
    article reports that Case 7 reaches the empty branch, so the remaining
    requirements must also filter. See README, "Reading of Algorithm 1 line 13".
    """
    active = task.constraints.active_requirements()
    dominant = task.constraints.dominant()

    if dominant is None:
        result.trace.append("N2.1: no requirement binds -> unconstrained branch")
        kept = [leaf for leaf in candidates if not leaf.discharges]
    else:
        silent = " (silent)" if dominant in SILENT else " (manifest)"
        result.trace.append(
            f"N2.1: binding {{{', '.join(active)}}}, "
            f"dominant by precedence {list(PRECEDENCE)} = {dominant}{silent}"
        )
        kept = [leaf for leaf in candidates if leaf.satisfies(dominant)]
        result.trace.append(
            f"N2.1: discharging {dominant} -> " + (", ".join(l.id for l in kept) or "none")
        )
        for constraint in active[1:]:
            kept = [leaf for leaf in kept if leaf.satisfies(constraint)]
            result.trace.append(
                f"N2.1: also requiring {constraint} -> "
                + (", ".join(l.id for l in kept) or "none")
            )

    if len(kept) > 1:
        kept = _refine_by_operator_and_regime(task, kept, result)
    return kept


def _refine_by_operator_and_regime(
    task: Task, candidates: list[Leaf], result: Result
) -> list[Leaf]:
    """Node N2.5, stated over operator availability and regime.

    Both are properties of the task. The driver attribute of Pateras et al. is
    not used, because Proposition 1 shows it is not a priori observable except
    when kappa = none.
    """
    operator = task.knowledge.operator
    regime = task.data.regime

    kept = []
    for leaf in candidates:
        required_operator = leaf.requires.get("operator")
        if required_operator is not None and required_operator != operator:
            continue
        required_regime = leaf.requires.get("regime")
        if required_regime is not None and required_regime != regime:
            continue
        kept.append(leaf)

    result.trace.append(
        f"N2.5: operator = {operator}, regime = {regime} -> "
        + (", ".join(l.id for l in kept) or "none")
    )
    return kept or candidates


def _filter_m3(task: Task, candidates: list[Leaf], result: Result) -> list[Leaf]:
    """Algorithm 1, line 17. Nodes N3.1 and N3.2."""
    kept = [
        leaf
        for leaf in candidates
        if leaf.requires.get("c_diff") in (None, task.constraints.c_diff)
    ]
    result.trace.append(
        f"N3.1: c_diff = {task.constraints.c_diff} -> "
        + (", ".join(l.id for l in kept) or "none")
    )
    kept = [leaf for leaf in kept if leaf.requires.get("a_type") == task.a_type]
    result.trace.append(
        f"N3.2: a is a {task.a_type} -> " + (", ".join(l.id for l in kept) or "none")
    )
    return kept


def _filter_m4(task: Task, candidates: list[Leaf], result: Result) -> list[Leaf]:
    """Algorithm 1, line 20. Nodes N4.1 and N4.2."""
    result.trace.append("N4.1: irreducibly discrete component present -> M4")
    kept = [
        leaf
        for leaf in candidates
        if leaf.requires.get("discrete_type") == task.discrete_type
    ]
    result.trace.append(
        f"N4.2: discrete type = {task.discrete_type} -> "
        + (", ".join(l.id for l in kept) or "none")
    )
    return kept


def _request_relaxation(
    task: Task, taxonomy: Taxonomy, macro_class: MacroClass, result: Result
) -> None:
    """Algorithm 1, line 24.

    Returns a localised statement of the gap: which constraints bind, which
    leaves satisfy each of them separately, and which constraint is weakest
    under the precedence order and therefore the candidate for relaxation.
    """
    binding = task.constraints.active_requirements()
    partial: dict[str, list[str]] = {}
    for constraint in binding:
        satisfying = [
            leaf.id
            for leaf in taxonomy.leaves_of(macro_class.value)
            if leaf.satisfies(constraint)
        ]
        if satisfying:
            partial[constraint] = satisfying

    result.outcome = Outcome.RELAXATION_REQUESTED
    result.relaxation = {
        "binding": binding,
        "drop": binding[-1] if binding else None,
        "partial": partial,
    }
    result.trace.append(
        "C = empty: no leaf satisfies the stated constraints; "
        "relaxation requested, flagged for expert review"
    )


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------


def compose(
    task: Task, taxonomy: Taxonomy, classes: list[MacroClass] | None = None
) -> Result:
    """Algorithm 2: selection for tasks spanning several macro-classes."""
    stages = classes if classes is not None else task.macro_classes()
    combined = Result(task=task.name, macro_classes=list(stages))
    combined.trace.append(
        "Compose: stages " + " -> ".join(m.value for m in stages)
    )

    # (C2) Constraint back-propagation, before selecting anything.
    stage_constraints: list[Constraints] = [task.constraints for _ in stages]
    for i in range(len(stages) - 1, 0, -1):
        upstream, downstream = stages[i - 1], stages[i]
        if downstream is MacroClass.M3 and task.constraints.c_diff is False:
            # The control stage cannot differentiate the plant, so the stage-i-1
            # deliverable must itself be differentiable and simulatable.
            stage_constraints[i - 1] = replace(stage_constraints[i - 1], c_diff=True)
            combined.trace.append(
                f"(C2) back-propagate c_diff from {downstream.value} to "
                f"{upstream.value}: the discovery deliverable must be differentiable"
            )
        if downstream is MacroClass.M3 and task.constraints.c_guar:
            stage_constraints[i - 1] = replace(stage_constraints[i - 1], c_guar=True)
            combined.trace.append(
                f"(C2) back-propagate c_guar from {downstream.value} to "
                f"{upstream.value}: the deliverable must be structure-preserving"
            )

    # Forward selection, propagating knowledge.
    knowledge = task.knowledge
    for index, macro_class in enumerate(stages):
        stage_task = _stage_task(task, macro_class, knowledge, stage_constraints[index])
        stage_result = _select_single(stage_task, taxonomy, macro_class)
        combined.trace.append(f"-- stage {index + 1} ({macro_class.value}) --")
        combined.trace.extend(f"   {step}" for step in stage_result.trace)

        if stage_result.outcome is not Outcome.RESOLVED:
            combined.outcome = stage_result.outcome
            combined.relaxation = stage_result.relaxation
            combined.unresolved_criterion = stage_result.unresolved_criterion
            combined.candidates = stage_result.candidates
            return combined

        combined.candidates.extend(stage_result.candidates)

        # (C1) Knowledge propagation: an identified operator raises kappa.
        if macro_class is MacroClass.M1:
            knowledge = replace(knowledge, kappa=Kappa.COMPLETE, operator_known=True)
            combined.trace.append(
                "(C1) knowledge propagation: kappa raised to complete; "
                "the environment for the next stage is the identified model"
            )

    return combined


def _stage_task(
    task: Task, macro_class: MacroClass, knowledge: Knowledge, constraints: Constraints
) -> Task:
    """Project the task onto the unknown set of one stage."""
    if macro_class is MacroClass.M1:
        unknowns = frozenset(
            u for u in task.unknowns if u.value in {"N", "theta"}
        )
        objective_functional = False
    elif macro_class is MacroClass.M3:
        unknowns = frozenset(u for u in task.unknowns if u.value == "a")
        objective_functional = True
        # After (C1) the plant is the identified model, which is differentiable.
        if knowledge.kappa is Kappa.COMPLETE and knowledge.operator_known:
            constraints = replace(constraints, c_diff=True)
    else:
        unknowns = frozenset(u for u in task.unknowns if u.value == "u")
        objective_functional = False

    return replace(
        task,
        unknowns=unknowns,
        knowledge=knowledge,
        constraints=constraints,
        objective_functional=objective_functional,
    )
