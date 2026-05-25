"""Unit tests for the game-agnostic lexicographic decider (design §7).

Exercises the core :mod:`dialectical_games.decider` module:

* :func:`lexicographic_decide` — the full FACT-then-graded key.
* :func:`fact_only_key` — the FACT-only key (terms 1-2) the §7
  fact-preservation differential tests use as an independent reference.

Phase 5 chunk 1: probes carry a typed
:class:`~dialectical_games.evidence.ArgumentEvidence` tuple. The
``Value`` enum is gone — priority is magnitude-only, and the cartridge
controls the scale. The terminal sentinel
:data:`~dialectical_games.decider._TERMINAL_LOSS_MAGNITUDE` is the
convention for "this is a game-winning fact".
"""

from __future__ import annotations

from doxa import Opinion

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    build_root_argument_graph,
)
from dialectical_games.decider import (
    _TERMINAL_LOSS_MAGNITUDE,
    fact_only_key,
    lexicographic_decide,
)
from dialectical_games.evidence import ArgumentEvidence, Role
from dialectical_games.scheme import Tier


# ---------------------------------------------------------------------------
# Stub policy
# ---------------------------------------------------------------------------


class _StubPolicy:
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
# Evidence construction helpers
# ---------------------------------------------------------------------------


def _terminal_pro(tag: str = "terminal_win") -> ArgumentEvidence:
    """A FACT pro with the terminal-loss magnitude — a "winning fact"."""
    return ArgumentEvidence(
        role=Role.PRO,
        tier=Tier.FACT,
        magnitude=_TERMINAL_LOSS_MAGNITUDE,
        tag=tag,
    )


def _material_pro(magnitude: int, tag: str = "material") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.PRO, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _terminal_objection(tag: str = "terminal_loss-obj") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.OBJECTION,
        tier=Tier.FACT,
        magnitude=_TERMINAL_LOSS_MAGNITUDE,
        tag=tag,
    )


def _terminal_reply(tag: str = "terminal_loss-rep") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.REPLY_ATTACK,
        tier=Tier.FACT,
        magnitude=_TERMINAL_LOSS_MAGNITUDE,
        tag=tag,
    )


def _material_objection(
    magnitude: int, tag: str = "allows_shot"
) -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _material_reply(magnitude: int, tag: str = "rep-mat") -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.REPLY_ATTACK, tier=Tier.FACT, magnitude=magnitude, tag=tag
    )


def _fact_defense(
    answered: ArgumentEvidence, tag: str = "holds_exchange"
) -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.DEFENSE, tier=Tier.FACT, answered=answered, tag=tag
    )


def _heuristic_pro(tag: str = "opposition") -> ArgumentEvidence:
    return ArgumentEvidence(role=Role.PRO, tier=Tier.HEURISTIC, tag=tag)


def _heuristic_objection(
    tag: str = "loses_opposition",
) -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.HEURISTIC, tag=tag
    )


# ---------------------------------------------------------------------------
# fact_only_key
# ---------------------------------------------------------------------------


def test_fact_only_key_is_zero_for_clean_grounded_survivor() -> None:
    """A clean move (no FACT attacker) yields term 1 = 0 and no FACT pro."""
    probes = [MoveProbe(move_id="m1")]
    graph = build_root_argument_graph(probes, _POLICY)
    key = fact_only_key(probes[0], graph)
    assert key == (0, 0)


def test_fact_only_key_term1_zero_for_defended_grounded_survivor() -> None:
    """A grounded survivor whose only FACT reply is defeated still keys at 0."""
    reply = _material_reply(100)
    defense = _fact_defense(answered=reply)
    pro = _material_pro(100)
    probe = MoveProbe(move_id="m1", evidence=(pro, reply, defense))
    graph = build_root_argument_graph([probe], _POLICY)
    # The move's argument is grounded -> term 1 = 0.
    assert graph.move_arguments["m1"] in graph.grounded_extension
    key = fact_only_key(probe, graph)
    assert key[0] == 0


def test_fact_only_key_winning_outranks_finite_material() -> None:
    """A terminal-magnitude FACT pro outranks a finite material FACT pro."""
    winning = MoveProbe(move_id="w", evidence=(_terminal_pro(),))
    large = MoveProbe(move_id="l", evidence=(_material_pro(300),))
    graph = build_root_argument_graph([winning, large], _POLICY)
    assert fact_only_key(winning, graph) < fact_only_key(large, graph)


def test_fact_only_key_terminal_loss_outranks_finite_material_loss() -> None:
    """In the empty-survivor fallback, terminal loss dominates material loss."""
    terminal_probe = MoveProbe(
        move_id="t", evidence=(_terminal_reply(),)
    )
    material_probe = MoveProbe(
        move_id="m", evidence=(_material_objection(300),)
    )
    graph = build_root_argument_graph(
        [terminal_probe, material_probe], _POLICY
    )
    key_t = fact_only_key(terminal_probe, graph)
    key_m = fact_only_key(material_probe, graph)
    # Both moves are in the empty-survivor fallback.
    assert key_t[0] == _TERMINAL_LOSS_MAGNITUDE
    assert key_m[0] == 300
    # Larger magnitude (terminal) sorts later — material is the better key.
    assert key_m < key_t


def test_fact_only_key_undefeated_attacker_only_contributes() -> None:
    """A FACT objection defeated by a keyed defense does NOT contribute to
    term 1 — the defense puts the attacker out of the grounded extension."""
    objection = _material_objection(500)
    defense = _fact_defense(answered=objection)
    probe = MoveProbe(move_id="m1", evidence=(objection, defense))
    graph = build_root_argument_graph([probe], _POLICY)
    # The move is grounded; term 1 is 0 anyway. But the property is the
    # objection is not in the grounded extension; verify directly.
    from dialectical_games.arguments import obj_arg_id

    assert obj_arg_id("m1", objection) not in graph.grounded_extension
    assert fact_only_key(probe, graph)[0] == 0


# ---------------------------------------------------------------------------
# lexicographic_decide
# ---------------------------------------------------------------------------


def test_decider_picks_winning_move_over_other_clean_moves() -> None:
    """The decider follows the FACT pro priority: terminal magnitude dominates."""
    winning = MoveProbe(move_id="w", evidence=(_terminal_pro(),))
    large = MoveProbe(move_id="l", evidence=(_material_pro(300),))
    small = MoveProbe(move_id="s", evidence=(_material_pro(50),))
    probes = [winning, large, small]
    graph = build_root_argument_graph(probes, _POLICY)
    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == "w"


def test_decider_picks_larger_finite_fact_pro_over_smaller() -> None:
    """Within finite magnitudes, larger FACT pro magnitude wins."""
    large = MoveProbe(move_id="l", evidence=(_material_pro(300),))
    small = MoveProbe(move_id="s", evidence=(_material_pro(50),))
    probes = [large, small]
    graph = build_root_argument_graph(probes, _POLICY)
    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == "l"


def test_decider_never_resurrects_crisply_eliminated_move() -> None:
    """The decider never returns a move outside the crisp survivors."""
    survivor = MoveProbe(move_id="ok", evidence=(_material_pro(100),))
    eliminated = MoveProbe(
        move_id="bad", evidence=(_material_objection(200),)
    )
    probes = [survivor, eliminated]
    graph = build_root_argument_graph(probes, _POLICY)
    assert graph.survivors == frozenset({"ok"})
    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == "ok"


def test_decider_returns_none_for_empty_probe_set() -> None:
    """An empty probe set yields no decision (the orchestrator's terminal case)."""
    graph = build_root_argument_graph([], _POLICY)
    assert lexicographic_decide([], graph) is None


def test_decider_breaks_fact_tie_by_graded_layer() -> None:
    """When the FACT key ties, the graded layer (term 3) decides."""
    supported = MoveProbe(
        move_id="sup", child_eval=0, evidence=(_heuristic_pro(),)
    )
    plain = MoveProbe(move_id="pln", child_eval=0)
    graph = build_root_argument_graph([supported, plain], _POLICY)
    # FACT keys are equal (no FACT material on either).
    assert fact_only_key(supported, graph) == fact_only_key(plain, graph)
    chosen = lexicographic_decide([supported, plain], graph)
    assert chosen is not None
    assert chosen.move_id == "sup"


def test_decider_is_evaluator_free() -> None:
    """The decider reads ``probe.child_eval`` directly — no live evaluator."""
    better = MoveProbe(move_id="b", child_eval=-50)
    worse = MoveProbe(move_id="w", child_eval=50)
    graph = build_root_argument_graph([better, worse], _POLICY)
    chosen = lexicographic_decide([better, worse], graph)
    assert chosen is not None
    assert chosen.move_id == "b"


def test_decider_fact_decision_dominates_graded() -> None:
    """A FACT pro outranks even a much better graded strength (term 2 > term 3)."""
    fact_move = MoveProbe(
        move_id="f",
        evidence=(_material_pro(100), _heuristic_objection()),
    )
    graded_move = MoveProbe(
        move_id="g", evidence=(_heuristic_pro(),)
    )
    graph = build_root_argument_graph([fact_move, graded_move], _POLICY)
    chosen = lexicographic_decide([fact_move, graded_move], graph)
    assert chosen is not None
    # The FACT pro decides despite the lower graded strength.
    assert chosen.move_id == "f"
