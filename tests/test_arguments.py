"""Tests for the game-agnostic crisp + graded argument graph builder.

These tests verify the **generic** mechanics of
``dialectical_games.arguments``:

* the crisp Dung layer admits only FACT-tier witnesses, computes the grounded
  extension correctly, and applies the empty-survivor fallback;
* the graded layer is opinion-valued, uses the policy's three quantities only,
  and never resurrects a crisply-eliminated move;
* every FACT / HEURISTIC dispatch is enum-typed
  (``ArgumentEvidence.role`` × ``ArgumentEvidence.tier``); no string parse.

Phase 5 chunk 1: probes carry a typed
:class:`~dialectical_games.evidence.ArgumentEvidence` tuple. No game-specific
labels appear here — every test builds its evidence from the core
:class:`Role` × :class:`Tier` enums directly.
"""

from __future__ import annotations

from doxa import Opinion
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    build_graded_layer,
    build_root_argument_graph,
    obj_arg_id,
    reply_arg_id,
)
from dialectical_games.evidence import ArgumentEvidence, Role
from dialectical_games.scheme import Tier


# ---------------------------------------------------------------------------
# Stub policy
# ---------------------------------------------------------------------------


class _StubPolicy:
    """A minimal :class:`GradedPolicy` for the generic tests."""

    def with_probes(self, probes: object) -> "_StubPolicy":
        return self

    @property
    def edge_trust(self) -> Opinion:
        return Opinion.dogmatic_true(0.5)

    def move_base_rate(self, probe: MoveProbe) -> float:
        x = probe.child_eval
        if x <= -100:
            return 0.80
        if x >= 100:
            return 0.20
        return 0.50 - x * 0.003

    def witness_opinion(
        self, *, probe: MoveProbe, evidence: ArgumentEvidence
    ) -> Opinion:
        return Opinion(0.55, 0.15, 0.30, 0.5)


_POLICY: GradedPolicy = _StubPolicy()


# ---------------------------------------------------------------------------
# Construction helpers (game-agnostic fixtures)
# ---------------------------------------------------------------------------


def _heuristic_pro(tag: str = "pro-h") -> ArgumentEvidence:
    return ArgumentEvidence(role=Role.PRO, tier=Tier.HEURISTIC, tag=tag)


def _heuristic_objection(tag: str = "obj-h") -> ArgumentEvidence:
    return ArgumentEvidence(role=Role.OBJECTION, tier=Tier.HEURISTIC, tag=tag)


def _fact_pro(magnitude: int, tag: str = "pro-f") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.PRO, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _fact_objection(magnitude: int, tag: str = "obj-f") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _fact_reply(magnitude: int, tag: str = "rep-f") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.REPLY_ATTACK, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _fact_defense(
    answered: ArgumentEvidence, tag: str = "def-f"
) -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.DEFENSE, tier=Tier.FACT, answered=answered, tag=tag
    )


def _heuristic_defense(
    answered: ArgumentEvidence, tag: str = "def-h"
) -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.DEFENSE, tier=Tier.HEURISTIC, answered=answered, tag=tag
    )


# ---------------------------------------------------------------------------
# Crisp layer
# ---------------------------------------------------------------------------


def test_clean_move_is_in_grounded_extension() -> None:
    """A move with no FACT objection survives the crisp layer."""
    probes = [MoveProbe(move_id="m1", evidence=(_fact_pro(0, tag="crown"),))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension
    assert "m1" in graph.survivors


def test_undefeated_fact_objection_eliminates_move() -> None:
    """A move with an undefeated FACT objection is NOT in the grounded extension."""
    probes = [
        MoveProbe(
            move_id="m1",
            evidence=(_fact_objection(0, tag="terminal_loss"),),
        )
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id not in graph.grounded_extension
    # The empty-survivor fallback fires — the only move stays.
    assert "m1" in graph.survivors


def test_keyed_defense_restores_move() -> None:
    """A FACT defense referencing the reply EVIDENCE OBJECT defeats only
    that reply and restores the move."""
    reply = _fact_reply(200, tag="material:200")
    defense = _fact_defense(answered=reply, tag="holds_exchange")
    probes = [MoveProbe(move_id="m1", evidence=(reply, defense))]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension
    assert "m1" in graph.survivors


def test_empty_survivor_fallback_returns_all_moves() -> None:
    """When every move carries an undefeated FACT objection, all moves survive."""
    probes = [
        MoveProbe(
            move_id="m1",
            evidence=(_fact_objection(0, tag="terminal_loss"),),
        ),
        MoveProbe(
            move_id="m2",
            evidence=(_fact_objection(0, tag="terminal_loss"),),
        ),
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    assert graph.survivors == frozenset({"m1", "m2"})


def test_no_duplicated_arguments() -> None:
    """Every argument id is distinct — no copy / duplicate arguments."""
    objection = _fact_objection(0, tag="terminal_loss-obj")
    reply = _fact_reply(0, tag="terminal_loss-rep")
    probes = [
        MoveProbe(move_id="m1", evidence=(objection, reply)),
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    assert len(graph.arguments) == len({a for a in graph.arguments})
    assert obj_arg_id("m1", objection) in graph.arguments
    assert reply_arg_id("m1", reply) in graph.arguments


def test_heuristic_objection_does_not_eliminate() -> None:
    """A HEURISTIC objection cannot defeat the move in the crisp layer."""
    probes = [
        MoveProbe(
            move_id="m1",
            evidence=(_heuristic_objection(tag="exposes_man"),),
        )
    ]
    graph = build_root_argument_graph(probes, _POLICY)
    move_id = graph.move_arguments["m1"]
    assert move_id in graph.grounded_extension


def test_empty_evidence_keeps_move_grounded() -> None:
    """A probe with no evidence at all is grounded (no attacker, no pro)."""
    probes = [MoveProbe(move_id="m1")]
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
        [
            MoveProbe(
                move_id="m1",
                child_eval=0,
                evidence=(_heuristic_pro(tag="opposition"),),
            )
        ],
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
                move_id="m1",
                child_eval=0,
                evidence=(_heuristic_objection(tag="exposes_man"),),
            )
        ],
        _POLICY,
    )
    assert (
        attacked.ranking["move_scores"]["m1"]
        < bland.ranking["move_scores"]["m1"]
    )


def test_graded_heuristic_defense_attacks_answered_objection_witness() -> None:
    """A HEURISTIC defense suppresses its answered objection inside the graph."""
    from dialectical_games.arguments import _witness_arg_id

    objection = _heuristic_objection(tag="exposes_man")
    defense = _heuristic_defense(answered=objection, tag="suppression")
    graph = build_root_argument_graph(
        [
            MoveProbe(
                move_id="m1", child_eval=0, evidence=(objection, defense)
            )
        ],
        _POLICY,
    )
    defense_witness = _witness_arg_id("m1", defense)
    objection_witness = _witness_arg_id("m1", objection)
    move_node = "move:m1"

    assert (defense_witness, objection_witness) in graph.ranking["attacks"]
    assert (defense_witness, move_node) not in graph.ranking["supports"]
    assert (defense_witness, move_node) not in graph.ranking["attacks"]


def test_graded_heuristic_defense_restores_toward_baseline_without_boost() -> None:
    """A HEURISTIC defense weakens an objection but does not add pro support."""
    bland = build_root_argument_graph(
        [MoveProbe(move_id="m1", child_eval=0)], _POLICY
    )
    attacked = build_root_argument_graph(
        [
            MoveProbe(
                move_id="m1",
                child_eval=0,
                evidence=(_heuristic_objection(tag="exposes_man"),),
            )
        ],
        _POLICY,
    )
    objection = _heuristic_objection(tag="exposes_man")
    defense = _heuristic_defense(answered=objection, tag="suppression")
    defended = build_root_argument_graph(
        [
            MoveProbe(
                move_id="m1",
                child_eval=0,
                evidence=(objection, defense),
            )
        ],
        _POLICY,
    )

    assert (
        defended.ranking["move_scores"]["m1"]
        > attacked.ranking["move_scores"]["m1"]
    )
    assert (
        defended.ranking["move_scores"]["m1"]
        <= bland.ranking["move_scores"]["m1"]
    )


def test_graded_layer_subset_of_survivors() -> None:
    """The graded move-node set is a subset of the crisp survivors."""
    probes = [
        MoveProbe(move_id="m1"),
        MoveProbe(
            move_id="m2",
            evidence=(_fact_objection(0, tag="terminal_loss"),),
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
    probes = [
        MoveProbe(
            move_id="m1", evidence=(_heuristic_pro(tag="opposition"),)
        )
    ]
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


def test_move_probe_evidence_defaults_to_empty_tuple() -> None:
    """The single ``evidence`` field replaces the four legacy fields."""
    probe = MoveProbe(move_id="m1")
    assert probe.evidence == ()


# ---------------------------------------------------------------------------
# Chunk H': GradedPolicy.with_probes
# ---------------------------------------------------------------------------


def test_with_probes_default_returns_equivalent_policy() -> None:
    """A policy with no per-position aggregate returns an equivalent policy."""
    probes = (MoveProbe(move_id="m1", child_eval=0),)
    bound = _POLICY.with_probes(probes)
    assert bound.edge_trust == _POLICY.edge_trust
    assert bound.move_base_rate(probes[0]) == _POLICY.move_base_rate(probes[0])
    sample_ev = _heuristic_pro(tag="sample")
    op = bound.witness_opinion(probe=probes[0], evidence=sample_ev)
    assert op.b + op.d + op.u == 1.0


def test_with_probes_called_by_builder_once_at_entry() -> None:
    """``_build_graded_graph_internal`` calls ``policy.with_probes`` once
    at entry with the survivor probes (chunk H' D1)."""

    class _CountingPolicy:
        def __init__(self) -> None:
            self.with_probes_calls: list[tuple[MoveProbe, ...]] = []

        def with_probes(self, probes: object) -> "_CountingPolicy":
            self.with_probes_calls.append(tuple(probes))  # type: ignore[arg-type]
            return self

        @property
        def edge_trust(self) -> Opinion:
            return Opinion.dogmatic_true(0.5)

        def move_base_rate(self, probe: MoveProbe) -> float:
            return 0.5

        def witness_opinion(
            self, *, probe: MoveProbe, evidence: ArgumentEvidence
        ) -> Opinion:
            return Opinion(0.55, 0.15, 0.30, 0.5)

    policy = _CountingPolicy()
    probes = [
        MoveProbe(
            move_id="m1",
            evidence=(_fact_pro(100, tag="material:100"),),
        ),
        MoveProbe(
            move_id="m2",
            evidence=(_fact_pro(200, tag="material:200"),),
        ),
    ]
    build_graded_layer(probes, frozenset({"m1", "m2"}), policy)
    assert len(policy.with_probes_calls) == 1
    assert {p.move_id for p in policy.with_probes_calls[0]} == {"m1", "m2"}


def test_with_probes_return_type_satisfies_protocol() -> None:
    """The object ``with_probes`` returns has the full Protocol surface."""
    probes: tuple[MoveProbe, ...] = (MoveProbe(move_id="m1"),)
    bound: GradedPolicy = _POLICY.with_probes(probes)
    assert isinstance(bound.edge_trust, Opinion)
    assert 0.0 < bound.move_base_rate(probes[0]) < 1.0
    op = bound.witness_opinion(
        probe=probes[0], evidence=_heuristic_pro(tag="sample")
    )
    assert isinstance(op, Opinion)


# --- hypothesis-generative property tests for `with_probes` ------------------

_PROBE_STRATEGY = st.builds(
    MoveProbe,
    move_id=st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
        min_size=1,
        max_size=4,
    ),
    child_eval=st.integers(min_value=-10000, max_value=10000),
)


@given(probes=st.lists(_PROBE_STRATEGY, min_size=0, max_size=8))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_with_probes_return_satisfies_protocol_property(
    probes: list[MoveProbe],
) -> None:
    """For any probe sequence, the policy returned by ``with_probes`` has
    the full ``GradedPolicy`` Protocol surface."""
    bound: GradedPolicy = _POLICY.with_probes(tuple(probes))
    assert isinstance(bound.edge_trust, Opinion)
    sample_probe = probes[0] if probes else MoveProbe(move_id="sample")
    rate = bound.move_base_rate(sample_probe)
    assert 0.0 < rate < 1.0
    op = bound.witness_opinion(
        probe=sample_probe, evidence=_heuristic_pro(tag="sample")
    )
    assert isinstance(op, Opinion)
    assert abs((op.b + op.d + op.u) - 1.0) < 1e-9
