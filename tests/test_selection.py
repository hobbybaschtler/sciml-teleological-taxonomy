"""Regression tests: every traversal reported in the article must replay."""

from __future__ import annotations

import pytest

from sciml_taxonomy import Outcome, Taxonomy, load_cases, select
from sciml_taxonomy.descriptor import PRECEDENCE, SILENT
from sciml_taxonomy.structural import (
    MATRIX_CELLS,
    MATRIX_ROWS,
    TREE_DEPTH1,
    TREE_DEPTH2,
    TREE_DEPTH2_CONFERENCE,
    analyse,
    expected_candidate_set,
)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return Taxonomy.load()


@pytest.fixture(scope="module")
def cases():
    return load_cases()


# -- schema ----------------------------------------------------------------


def test_corpus_has_sixteen_leaves(taxonomy: Taxonomy) -> None:
    assert len(taxonomy.leaves) == 16


def test_eleven_decision_nodes(taxonomy: Taxonomy) -> None:
    nodes = taxonomy.decision_nodes()
    assert len(nodes) == 11
    assert sorted(n["id"] for n in nodes) == [
        "N0", "N1.1", "N2.1", "N2.2", "N2.3", "N2.4", "N2.5",
        "N3.1", "N3.2", "N4.1", "N4.2",
    ]


def test_observability_ratio_is_one(taxonomy: Taxonomy) -> None:
    """omega = 11/11 = 1.00 (Table 5). This is the paper's decisive claim."""
    observable, total = taxonomy.observability_ratio()
    assert observable == total == 11


def test_every_leaf_reachable_exactly_once(taxonomy: Taxonomy) -> None:
    """Ending conditions: all objects classified, no object duplicated."""
    taxonomy.validate()


def test_macro_class_sizes(taxonomy: Taxonomy) -> None:
    sizes = tuple(len(taxonomy.leaves_of(m)) for m in ("M1", "M2", "M3", "M4"))
    assert sizes == TREE_DEPTH1 == (2, 8, 3, 3)


def test_only_ranked_requirements_are_discharged(taxonomy: Taxonomy) -> None:
    """Remark 1: c_diff and c_disc are filters, not dischargeable requirements."""
    for leaf in taxonomy.leaves.values():
        for constraint in leaf.discharges:
            assert constraint in PRECEDENCE


def test_every_leaf_documents_failure_modes(taxonomy: Taxonomy) -> None:
    """Section 8: a selector that recommends without warning is worse than none."""
    for leaf in taxonomy.leaves.values():
        assert leaf.known_failure_modes, f"{leaf.id} documents no failure mode"


# -- case studies ----------------------------------------------------------


def test_all_cases_replay(taxonomy: Taxonomy, cases) -> None:
    for case in cases:
        result = select(case.task, taxonomy)
        assert result.outcome.value == case.expected.get("outcome", "resolved"), case.id
        assert result.ids == list(case.expected.get("candidates", [])), case.id


@pytest.mark.parametrize(
    "case_id,expected",
    [
        ("case1", ["PIDLSTM"]),
        ("case2", ["UDE"]),
        ("case3", ["HybridABM"]),
        ("case4", ["NeuralOperator"]),
        ("case4b", ["FoundationModel"]),
        ("case5", ["PIGP"]),
        ("case6", ["GenerativeHybrid"]),
    ],
)
def test_table7_rows(taxonomy: Taxonomy, cases, case_id: str, expected: list[str]) -> None:
    case = next(c for c in cases if c.id == case_id)
    assert select(case.task, taxonomy).ids == expected


def test_case7_terminates_without_recommendation(taxonomy: Taxonomy, cases) -> None:
    """Section 9.5: the adversarial case must reach the C = empty branch."""
    case = next(c for c in cases if c.id == "case7")
    result = select(case.task, taxonomy)

    assert result.outcome is Outcome.RELAXATION_REQUESTED
    assert result.ids == []
    assert result.relaxation is not None
    assert result.relaxation["binding"] == ["c_guar", "c_lat"]
    # The gap is localised: each constraint is satisfied by some leaf, but no
    # leaf satisfies both.
    assert result.relaxation["partial"]["c_guar"] == ["HNN_LNN"]
    assert "HybridRC" in result.relaxation["partial"]["c_lat"]


def test_case6_is_not_misled_by_its_name(taxonomy: Taxonomy, cases) -> None:
    """"Inverse design" is M3, not M1: the class follows the unknown set."""
    case = next(c for c in cases if c.id == "case6")
    assert [m.value for m in case.task.macro_classes()] == ["M3"]


def test_bioprocess_composition(taxonomy: Taxonomy, cases) -> None:
    """Section 7.3: back-propagation then knowledge propagation."""
    case = next(c for c in cases if c.id == "case_bioprocess")
    result = select(case.task, taxonomy)

    assert [m.value for m in result.macro_classes] == ["M1", "M3"]
    assert result.ids == ["UDE", "DifferentiableSolver"]
    trace = "\n".join(result.trace)
    assert "(C2) back-propagate c_diff" in trace
    assert "(C1) knowledge propagation" in trace


def test_composition_differs_from_either_branch_alone(taxonomy: Taxonomy, cases) -> None:
    """Without (C1), stage two would still face a non-differentiable plant."""
    from dataclasses import replace

    case = next(c for c in cases if c.id == "case_bioprocess")
    control_only = replace(
        case.task,
        unknowns=frozenset(u for u in case.task.unknowns if u.value == "a"),
    )
    assert select(control_only, taxonomy).ids == ["PIDLSTM"]
    assert "DifferentiableSolver" in select(case.task, taxonomy).ids


# -- precedence and Definition 5 -------------------------------------------


def test_silent_constraints_outrank_manifest_ones(taxonomy: Taxonomy) -> None:
    """Equation (2) is ordered so that every silent constraint precedes every
    manifest one, which is the justification given in Section 7.1."""
    ranks = {c: i for i, c in enumerate(PRECEDENCE)}
    assert max(ranks[c] for c in SILENT) < min(
        ranks[c] for c in PRECEDENCE if c not in SILENT
    )


def test_precedence_has_four_elements_not_six(taxonomy: Taxonomy) -> None:
    """Remark 1: environment facts are not ranked alongside requirements."""
    assert len(PRECEDENCE) == 4


# -- structural analysis ---------------------------------------------------


def test_partitions_cover_the_corpus() -> None:
    for partition in (MATRIX_CELLS, MATRIX_ROWS, TREE_DEPTH1, TREE_DEPTH2,
                      TREE_DEPTH2_CONFERENCE):
        assert sum(partition) == 16


def test_table5_values(taxonomy: Taxonomy) -> None:
    report = analyse(taxonomy)
    assert report.matrix_depth1 == pytest.approx(6.00)
    assert report.matrix_depth2 == pytest.approx(3.25)
    assert report.tree_depth1 == pytest.approx(5.375, abs=5e-3)
    assert report.tree_depth2 == pytest.approx(2.125, abs=5e-3)
    assert report.tree_depth3 == pytest.approx(1.00)
    assert report.matrix_reduction == pytest.approx(2.22, abs=5e-3)
    assert report.tree_reduction == pytest.approx(2.52, abs=5e-3)


def test_reordering_gain_is_26_percent(taxonomy: Taxonomy) -> None:
    """Section 9.2: the restructured simulation branch is worth 26%."""
    report = analyse(taxonomy)
    assert report.tree_depth2_conference == pytest.approx(2.875, abs=5e-3)
    assert report.reordering_gain == pytest.approx(0.26, abs=5e-3)


def test_equation_3(taxonomy: Taxonomy) -> None:
    assert expected_candidate_set((1, 1, 1, 1, 3, 3, 1, 1, 1, 3)) == 34 / 16
    assert expected_candidate_set((3, 5, 2, 2, 3, 1)) == 52 / 16
