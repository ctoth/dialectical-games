"""Tests for the game-agnostic crisp + graded argument graph builder.

These tests verify the **generic** mechanics of
``dialectical_games.arguments``:

* the crisp Dung layer admits only FACT-tier witnesses, computes the grounded
  extension correctly, and applies the empty-survivor fallback;
* the graded layer is opinion-valued, uses the policy's three quantities only,
  and never resurrects a crisply-eliminated move;
* FACT-tier evidence labels typed by ``dialectical_games.evidence`` are
  recognised in both layers.

Cartridge-specific witness rules (the checkers FACT taxonomy is part of
``dialectical_games.evidence``; the HEURISTIC labels in this file are typed by
that same parser when they parse as one of its enrichment witnesses) live
cartridge-side. Here a single stub policy stands in for any cartridge's
:class:`GradedPolicy`.
"""

from __future__ import annotations

from doxa import Opinion

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    build_graded_layer,
    build_root_argument_graph,
    obj_arg_id,
    reply_arg_id,
)


class _StubPolicy:
    """A minimal :class:`GradedPolicy` for the generic tests.

    Yields a centred base rate that responds monotonically to
    ``probe.child_eval`` (smaller-is-better-for-the-mover convention),
    a fixed-belief witness opinion, and constant dogmatic edge trust.
    """

    @property
    def edge_trust(self) -> Opinion:
        return Opinion.dogmatic_true(0.5)

    def move_base_rate(self, probe: MoveProbe) -> float:
        # Smaller ``child_eval`` -> larger ``a``. A small, monotone form.
        x = probe.child_eval
        if x <= -100:
            return 0.80
        if x >= 100:
            return 0.20
        return 0.50 - x * 0.003

    def witness_opinion(
        self, *, probe: MoveProbe, label: str, magnitude: int | None
    ) -> Opinion:
        # A fixed-band HEURISTIC witness: belief 0.55, uncertainty 0.30,
        # disbelief 0.15. ``magnitude`` is ignored — the generic tests
        # exercise the topology, not magnitude scaling.
        return Opinion(0.55, 0.15, 0.30, 0.5)


_POLICY: GradedPolicy = _StubPolicy()


# ---------------------------------------------------------------------------
# Crisp layer
# ---------------------------------------------------------------------------


def test_clean_move_is_in_grounded_extension() -> None:
    """A move with no FACT objection survives the crisp layer."""
    probes = [MoveProbe(move_id="m1", reasons=("pro:crown",))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension
    assert "m1" in graph.survivors


def test_undefeated_fact_objection_eliminates_move() -> None:
    """A move with an undefeated FACT objection is NOT in the grounded extension."""
    probes = [MoveProbe(move_id="m1", objections=("obj:terminal_loss",))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id not in graph.grounded_extension
    # The empty-survivor fallback fires — the only move stays.
    assert "m1" in graph.survivors


def test_keyed_defense_restores_move() -> None:
    """A keyed FACT defense defeats only the objection it answers, restoring the move."""
    probes = [
        MoveProbe(
            move_id="m1",
            reply_attacks=("reply:material:200",),
            defenses=("defense:holds_exchange@reply:material:200",),
        )
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension
    assert "m1" in graph.survivors


def test_empty_survivor_fallback_returns_all_moves() -> None:
    """When every move carries an undefeated FACT objection, all moves survive.

    The cartridge selector then ranks by the magnitude of the unavoidable
    loss — the core only guarantees that survivors is non-empty.
    """
    probes = [
        MoveProbe(move_id="m1", objections=("obj:terminal_loss",)),
        MoveProbe(move_id="m2", objections=("obj:terminal_loss",)),
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    assert graph.survivors == frozenset({"m1", "m2"})


def test_no_duplicated_arguments() -> None:
    """Every argument id is distinct — no copy / duplicate arguments."""
    probes = [
        MoveProbe(
            move_id="m1",
            objections=("obj:terminal_loss",),
            reply_attacks=("reply:terminal_loss",),
        )
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    # No two arguments share an id.
    assert len(graph.arguments) == len({a for a in graph.arguments})
    assert obj_arg_id("m1", "obj:terminal_loss") in graph.arguments
    assert reply_arg_id("m1", "reply:terminal_loss") in graph.arguments


def test_heuristic_objection_does_not_eliminate() -> None:
    """A HEURISTIC objection cannot defeat the move in the crisp layer."""
    probes = [MoveProbe(move_id="m1", objections=("obj:exposes_man",))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension


def test_unknown_label_is_silently_excluded() -> None:
    """A label the evidence parser rejects is ignored by the crisp layer."""
    probes = [MoveProbe(move_id="m1", objections=("garbled:not_a_label",))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension


# ---------------------------------------------------------------------------
# Graded layer
# ---------------------------------------------------------------------------


def test_graded_layer_vacuous_move_resolves_to_base_rate() -> None:
    """An unargued move's resolved expectation == its base rate exactly."""
    probes = [MoveProbe(move_id="m1", child_eval=0)]
    graph = build_root_argument_graph(probes, _POLICY)
    # _StubPolicy.move_base_rate(child_eval=0) == 0.50.
    assert abs(graph.ranking["move_scores"]["m1"] - 0.50) < 1e-9


def test_graded_layer_supports_lifts_expectation() -> None:
    """A HEURISTIC pro-reason raises the move's resolved expectation."""
    bland = build_root_argument_graph(
        [MoveProbe(move_id="m1", child_eval=0)], _POLICY
    )
    supported = build_root_argument_graph(
        [MoveProbe(move_id="m1", child_eval=0, reasons=("pro:opposition",))],
        _POLICY,
    )
    assert (
        supported.ranking["move_scores"]["m1"]
        > bland.ranking["move_scores"]["m1"]
    )


def test_graded_layer_attacks_lower_expectation() -> None:
    """A HEURISTIC objection lowers the move's resolved expectation."""
    bland = build_root_argument_graph(
        [MoveProbe(move_id="m1", child_eval=0)], _POLICY
    )
    attacked = build_root_argument_graph(
        [
            MoveProbe(
                move_id="m1", child_eval=0, objections=("obj:exposes_man",)
            )
        ],
        _POLICY,
    )
    assert (
        attacked.ranking["move_scores"]["m1"]
        < bland.ranking["move_scores"]["m1"]
    )


def test_graded_layer_subset_of_survivors() -> None:
    """The graded move-node set is a subset of the crisp survivors."""
    probes = [
        MoveProbe(move_id="m1"),
        MoveProbe(
            move_id="m2",
            objections=("obj:terminal_loss",),  # FACT, eliminates m2
        ),
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    # Only m1 survives crisply (m2 is eliminated and the fallback does NOT
    # fire — at least one move survives).
    assert graph.survivors == frozenset({"m1"})
    # The graded move-scores key only m1.
    assert set(graph.ranking["move_scores"]) == {"m1"}


def test_graded_layer_empty_for_no_probes() -> None:
    """No probes -> trivial empty graded layer."""
    graph = build_root_argument_graph([], _POLICY)
    assert graph.survivors == frozenset()
    assert graph.ranking["move_scores"] == {}
    assert graph.ranking["arguments"] == frozenset()


def test_graded_layer_none_policy_yields_empty_layer() -> None:
    """``policy=None`` yields the trivial empty graded layer for any probes.

    The crisp layer still runs unchanged — only the graded ranking is
    suppressed.
    """
    probes = [MoveProbe(move_id="m1", reasons=("pro:opposition",))]
    graph = build_root_argument_graph(probes, None)
    assert "m1" in graph.survivors
    assert graph.ranking["move_scores"] == {}


def test_build_graded_layer_direct_call() -> None:
    """``build_graded_layer`` is independently callable for inspection / tests."""
    probes = [MoveProbe(move_id="m1", child_eval=0)]
    ranking = build_graded_layer(probes, frozenset({"m1"}), _POLICY)
    assert "m1" in ranking["move_scores"]


# ---------------------------------------------------------------------------
# MoveProbe surface
# ---------------------------------------------------------------------------


def test_move_probe_pdn_alias() -> None:
    """The pre-extraction ``probe.pdn`` alias still maps to ``move_id``."""
    probe = MoveProbe(move_id="11-15")
    assert probe.pdn == "11-15"


def test_move_probe_new_fields_default_to_zero_and_false() -> None:
    """``child_eval`` defaults to 0 and ``contested`` defaults to False."""
    probe = MoveProbe(move_id="m1")
    assert probe.child_eval == 0
    assert probe.contested is False
