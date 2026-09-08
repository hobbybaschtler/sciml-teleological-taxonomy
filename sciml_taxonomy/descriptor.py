"""Task descriptor of Definition 1 and constraint classes of Definition 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Kappa(str, Enum):
    """Completeness level of the governing operator (Definition 1)."""

    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"


class Unknown(str, Enum):
    """Members of the unknown set U of Definition 4."""

    OPERATOR = "N"
    PARAMETERS = "theta"
    STATE = "u"
    CONTROL = "a"


class MacroClass(str, Enum):
    M1 = "M1"  # inverse discovery
    M2 = "M2"  # forward simulation
    M3 = "M3"  # control and design
    M4 = "M4"  # domain coupling


#: The four ranked requirements of Remark 1, in the precedence order (2)
#: of Section 7.1: silent constraints outrank manifest ones.
PRECEDENCE: tuple[str, ...] = ("c_guar", "c_unc", "c_lat", "c_interp")

#: The two environment facts of Remark 1. These are not negotiable and act as
#: filters rather than as ranked objectives, which is why PRECEDENCE has four
#: elements and not six.
ENVIRONMENT_FACTS: tuple[str, ...] = ("c_diff", "c_disc")

#: Silent constraints (Definition 5): violation produces plausible-looking
#: output that is nonetheless wrong.
SILENT: frozenset[str] = frozenset({"c_guar", "c_unc"})


@dataclass(frozen=True)
class Knowledge:
    """K = (N_known, Theta_known, kappa)."""

    kappa: Kappa = Kappa.NONE
    operator_known: bool = False
    parameters_known: bool = False
    candidate_dictionary: bool = False
    differentiable_skeleton: bool = False
    knowledge_based_core: bool = False

    @property
    def operator(self) -> str:
        """Operator availability: the observable proxy used at node N2.5.

        Explicit when a governing statement is available in the domain
        literature (kappa is partial or complete), implied otherwise. This is a
        property of the task, unlike the driver attribute of Pateras et al.,
        which Proposition 1 shows to be inadmissible.
        """
        return "explicit" if self.kappa is not Kappa.NONE else "implied"


@dataclass(frozen=True)
class DataRegime:
    """D = (n, sigma, rho, Omega), plus the qualitative regime label."""

    n: int | None = None
    noise: float | None = None
    density: str | None = None
    domain: str | None = None
    regime: str | None = None  # parametric_family | chaotic | linearisable |
    #                            broad_corpus | sequential
    n_training_solutions: str | None = None  # low | high
    parametric_family: bool = False
    broad_corpus_coverage: bool = False
    finite_linearising_embedding: bool = False
    sequential_observations: bool = False
    moderate_state_dimension: bool = False
    conservative_system: bool = False
    explicit_residual_form: bool = False


@dataclass(frozen=True)
class Constraints:
    """C, partitioned into the six classes of Definition 2."""

    c_guar: bool = False
    c_unc: bool = False
    c_lat: bool = False
    c_interp: bool = False
    c_diff: bool | None = None  # environment fact; None = not applicable
    c_disc: bool = False

    def active_requirements(self) -> list[str]:
        """The ranked requirements that are asserted, in precedence order."""
        return [c for c in PRECEDENCE if getattr(self, c)]

    def dominant(self) -> str | None:
        """Dominant(C) of Algorithm 1, line 12: highest-ranked active element."""
        active = self.active_requirements()
        return active[0] if active else None


@dataclass(frozen=True)
class Task:
    """T = (G, K, D, C) of Definition 1.

    Every component is a priori observable in the sense of Definition 3
    (Proposition 2).
    """

    name: str
    objective: str
    unknowns: frozenset[Unknown]
    knowledge: Knowledge = field(default_factory=Knowledge)
    data: DataRegime = field(default_factory=DataRegime)
    constraints: Constraints = field(default_factory=Constraints)
    objective_functional: bool = False
    a_type: str | None = None  # action | geometry
    discrete_type: str | None = None  # agents | network_fluxes | boolean_rules
    reference: str | None = None
    notes: str | None = None

    # -- Definition 4 ------------------------------------------------------

    def macro_classes(self) -> list[MacroClass]:
        """MacroClasses(U) of Algorithm 1, line 2.

        The partition of Definition 4 is exhaustive and mutually exclusive over
        a single unknown set, but a task may legitimately carry unknowns of more
        than one kind, in which case several classes are returned and
        Algorithm 2 applies.
        """
        classes: list[MacroClass] = []

        if self.constraints.c_disc:
            classes.append(MacroClass.M4)

        if Unknown.OPERATOR in self.unknowns or Unknown.PARAMETERS in self.unknowns:
            classes.append(MacroClass.M1)

        if Unknown.CONTROL in self.unknowns and self.objective_functional:
            classes.append(MacroClass.M3)

        # A state unknown induces M2 only when the operator is a single one of
        # its mathematical type. If c_disc holds, the governing operator is the
        # composite N_cont (+) N_disc of M4 and the difficulty is
        # representational rather than statistical, so M4 claims the task; the
        # partition of Definition 4 is mutually exclusive. If the state is an
        # intermediate quantity of a control task, M3 claims it (Example 2).
        if (
            Unknown.STATE in self.unknowns
            and not self.constraints.c_disc
            and not (Unknown.CONTROL in self.unknowns and self.objective_functional)
        ):
            classes.append(MacroClass.M2)

        if not classes:
            raise ValueError(
                f"task {self.name!r}: unknown set {sorted(u.value for u in self.unknowns)} "
                "induces no macro-class; check the deliverable statement"
            )

        # Data-dependency order (Definition 6): discovery precedes control.
        order = {MacroClass.M1: 0, MacroClass.M2: 1, MacroClass.M3: 2, MacroClass.M4: 3}
        return sorted(set(classes), key=lambda m: order[m])

    # -- serialisation -----------------------------------------------------

    def facts(self) -> dict[str, Any]:
        """Flat view of the task, used to evaluate leaf `requires` clauses."""
        out: dict[str, Any] = {
            "kappa": self.knowledge.kappa.value,
            "operator": self.knowledge.operator,
            "candidate_dictionary": self.knowledge.candidate_dictionary,
            "differentiable_skeleton": self.knowledge.differentiable_skeleton,
            "knowledge_based_core": self.knowledge.knowledge_based_core,
            "a_type": self.a_type,
            "discrete_type": self.discrete_type,
            "c_diff": self.constraints.c_diff,
            "c_disc": self.constraints.c_disc,
        }
        for key, value in vars(self.data).items():
            if value is not None:
                out[key] = value
        return out
