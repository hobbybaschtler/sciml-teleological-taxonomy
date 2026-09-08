"""Reproduction of the structural analysis of Section 9.2 and Table 5.

All values follow from the partitions induced by Table 2 and Figure 1 and are
computed without experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from .schema import Taxonomy

#: Cell sizes of the descriptive matrix of Table 2, crossing information source
#: with integration mechanism over the same corpus of 16 families.
MATRIX_CELLS: tuple[int, ...] = (3, 5, 2, 2, 3, 1)

#: Row sizes of the same matrix, i.e. the partition after one question on the
#: integration axis alone: observational 3+5, inductive 2+2, learning 3+1.
MATRIX_ROWS: tuple[int, ...] = (8, 4, 4)

#: Depth-1 partition of the teleological tree: |M1|, |M2|, |M3|, |M4|.
TREE_DEPTH1: tuple[int, ...] = (2, 8, 3, 3)

#: Depth-2 partition. M2 splits by dominant constraint into (1, 1, 3, 3);
#: M1 contributes (1, 1); M3 contributes (1, 1, 1); M4 is not yet split.
TREE_DEPTH2: tuple[int, ...] = (1, 1, 1, 1, 3, 3, 1, 1, 1, 3)

#: Depth-2 partition under the conference ordering, in which the simulation
#: branch was split by information source into two groups of four.
TREE_DEPTH2_CONFERENCE: tuple[int, ...] = (1, 1, 4, 4, 1, 1, 1, 3)

#: Depth-3 partition: every leaf is a singleton.
TREE_DEPTH3: tuple[int, ...] = tuple([1] * 16)


def fmt(value: float, places: int = 2) -> str:
    """Round half up, matching the convention used in the published table."""
    quant = Decimal(10) ** -places
    return str(Decimal(value).quantize(quant, rounding=ROUND_HALF_UP))


def expected_candidate_set(partition: tuple[int, ...], corpus: int = 16) -> float:
    """Equation (3): E[|C|]_d = (1/|A|) * sum_g |g|^2."""
    return sum(size**2 for size in partition) / corpus


@dataclass(frozen=True)
class StructuralReport:
    corpus: int
    matrix_depth1: float
    matrix_depth2: float
    tree_depth1: float
    tree_depth2: float
    tree_depth3: float
    tree_depth2_conference: float
    reordering_gain: float
    matrix_reduction: float
    tree_reduction: float
    observability_matrix: tuple[int, int]
    observability_tree: tuple[int, int]
    under_determination_matrix: tuple[int, int]
    under_determination_tree: tuple[int, int]

    def render(self) -> str:
        omega_m = self.observability_matrix
        omega_t = self.observability_tree
        upsilon_m = self.under_determination_matrix
        upsilon_t = self.under_determination_tree
        rows = [
            ("Measure", "Matrix", "Tree"),
            ("E[|C|] after 1 question", fmt(self.matrix_depth1), fmt(self.tree_depth1)),
            ("E[|C|] after 2 questions", fmt(self.matrix_depth2), fmt(self.tree_depth2)),
            ("E[|C|] after 3 questions", "-", fmt(self.tree_depth3)),
            ("Reduction factor per question", fmt(self.matrix_reduction) + "x", fmt(self.tree_reduction) + "x"),
            ("Observability ratio omega", f"{omega_m[0]}/{omega_m[1]} = " + fmt(omega_m[0] / omega_m[1]),
             f"{omega_t[0]}/{omega_t[1]} = " + fmt(omega_t[0] / omega_t[1])),
            ("Under-determination rate upsilon",
             f"{upsilon_m[0]}/{upsilon_m[1]} = " + fmt(upsilon_m[0] / upsilon_m[1]),
             f"{upsilon_t[0]}/{upsilon_t[1]} = " + fmt(upsilon_t[0] / upsilon_t[1])),
            ("Coverage of A", "1.00", "1.00"),
        ]
        width = [max(len(row[i]) for row in rows) for i in range(3)]
        lines = []
        for index, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(width[i]) for i, cell in enumerate(row)))
            if index == 0:
                lines.append("  ".join("-" * w for w in width))
        lines.append("")
        lines.append(
            f"Conference ordering of the simulation branch: "
            f"E[|C|]_2 = " + fmt(self.tree_depth2_conference) + "; the restructuring of "
            f"Figure 1 is worth a further {self.reordering_gain:.0%} reduction."
        )
        return "\n".join(lines)


def analyse(taxonomy: Taxonomy, corpus: int = 16) -> StructuralReport:
    # Depth 1 for the matrix is a split by its first axis, the integration
    # mechanism, giving the three rows of Table 2 with sizes (8, 4, 4).
    matrix_d1 = expected_candidate_set(MATRIX_ROWS, corpus)
    matrix_d2 = expected_candidate_set(MATRIX_CELLS, corpus)

    tree_d1 = expected_candidate_set(TREE_DEPTH1, corpus)
    tree_d2 = expected_candidate_set(TREE_DEPTH2, corpus)
    tree_d3 = expected_candidate_set(TREE_DEPTH3, corpus)
    tree_d2_conf = expected_candidate_set(TREE_DEPTH2_CONFERENCE, corpus)

    observable, total = taxonomy.observability_ratio()

    return StructuralReport(
        corpus=corpus,
        matrix_depth1=matrix_d1,
        matrix_depth2=matrix_d2,
        tree_depth1=tree_d1,
        tree_depth2=tree_d2,
        tree_depth3=tree_d3,
        tree_depth2_conference=tree_d2_conf,
        reordering_gain=(tree_d2_conf - tree_d2) / tree_d2_conf,
        matrix_reduction=(corpus / matrix_d2) ** (1 / 2),
        tree_reduction=(corpus / tree_d3) ** (1 / 3),
        observability_matrix=(0, 2),
        observability_tree=(observable, total),
        under_determination_matrix=(15, 16),
        under_determination_tree=(0, 16),
    )
