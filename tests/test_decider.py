"""Unit tests for the game-agnostic lexicographic decider (design §7).

Exercises the core :mod:`dialectical_games.decider` module:

* :func:`lexicographic_decide` — the full FACT-then-graded key.
* :func:`fact_only_key` — the FACT-only key (terms 1-2) the §7
  fact-preservation differential tests use as an independent reference.

All probes are hand-built so the decider's semantics are tested in
isolation, without a board or a cartridge selector.
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


# ---------------------------------------------------------------------------
# A minimal stub policy — exactly the shape the generic tests already use.
# ---------------------------------------------------------------------------


class _StubPolicy:
    def with_probes(self, probes: object) -> "_StubPolicy":
        # Chunk H': no per-position aggregates; identity.
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
        self, *, probe: MoveProbe, label: str, magnitude: int | None
    ) -> Opinion:
        return Opinion(0.55, 0.15, 0.30, 0.5)


_POLICY: GradedPolicy = _StubPolicy()


# ---------------------------------------------------------------------------
# fact_only_key — the FACT layer's primitive
# ---------------------------------------------------------------------------


def test_fact_only_key_is_zero_for_clean_grounded_survivor() -> None:
    """A clean move (no FACT attacker) yields term 1 = 0 and no FACT pro."""
    probes = [MoveProbe(move_id="m1")]
    graph = build_root_argument_graph(probes, _POLICY)
    key = fact_only_key(probes[0], graph)
    assert key == (0, 0, 0, 0, 0)


def test_fact_only_key_term1_zero_for_defended_grounded_survivor() -> None:
    """A grounded survivor whose only FACT reply is defeated still keys at 0."""
    probe = MoveProbe(
        move_id="m1",
        reasons=("pro:material:100",),
        reply_attacks=("reply:material:100",),
        defenses=("defense:holds_exchange@reply:material:100",),
    )
    graph = build_root_argument_graph([probe], _POLICY)
    # The move's argument is grounded -> term 1 = 0.
    assert graph.move_arguments["m1"] in graph.grounded_extension
    key = fact_only_key(probe, graph)
    # Term 1 zero; term 2: net material = 100 - 100 = 0 -> no pro components.
    assert key[0] == 0


def test_fact_only_key_winning_outranks_large_material() -> None:
    """`winning` outranks `large material` in the FACT pro-value tuple."""
    winning = MoveProbe(move_id="w", reasons=("pro:terminal_win",))
    large = MoveProbe(move_id="l", reasons=("pro:material:300",))
    graph = build_root_argument_graph([winning, large], _POLICY)
    assert fact_only_key(winning, graph) < fact_only_key(large, graph)


def test_fact_only_key_terminal_loss_outranks_finite_material_loss() -> None:
    """In the empty-survivor fallback, terminal loss dominates material loss."""
    terminal = MoveProbe(move_id="t", reply_attacks=("reply:terminal_loss",))
    material = MoveProbe(move_id="m", objections=("obj:allows_shot:300",))
    graph = build_root_argument_graph([terminal, material], _POLICY)
    key_t = fact_only_key(terminal, graph)
    key_m = fact_only_key(material, graph)
    # Both moves are in the empty-survivor fallback.
    assert key_t[0] == _TERMINAL_LOSS_MAGNITUDE
    assert key_m[0] == 300
    # Larger magnitude (terminal) sorts later — material is the better key.
    assert key_m < key_t


def test_fact_only_key_net_material_is_priority_input() -> None:
    """A defended even exchange scores 0 material in the FACT key."""
    clean = MoveProbe(move_id="c", reasons=("pro:material:100",))
    held_even = MoveProbe(
        move_id="h",
        reasons=("pro:material:100",),
        reply_attacks=("reply:material:100",),
        defenses=("defense:holds_exchange@reply:material:100",),
    )
    graph = build_root_argument_graph([clean, held_even], _POLICY)
    # clean's FACT pro keeps 100 (small_material); held_even keeps 0.
    assert fact_only_key(clean, graph) < fact_only_key(held_even, graph)


# ---------------------------------------------------------------------------
# lexicographic_decide — the full key
# ---------------------------------------------------------------------------


def test_decider_picks_winning_move_over_other_clean_moves() -> None:
    """The decider follows the FACT pro priority: winning > large > crown > small."""
    winning = MoveProbe(move_id="w", reasons=("pro:terminal_win",))
    large = MoveProbe(move_id="l", reasons=("pro:material:300",))
    crown = MoveProbe(move_id="c", reasons=("pro:crown",))
    small = MoveProbe(move_id="s", reasons=("pro:material:50",))
    probes = [winning, large, crown, small]
    graph = build_root_argument_graph(probes, _POLICY)
    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == "w"


def test_decider_never_resurrects_crisply_eliminated_move() -> None:
    """The decider never returns a move outside the crisp survivors."""
    survivor = MoveProbe(move_id="ok", reasons=("pro:material:100",))
    eliminated = MoveProbe(
        move_id="bad", objections=("obj:allows_shot:200",)
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
    """When the FACT key ties, the graded layer (term 3) decides.

    Two clean survivors with no FACT pro: one carries a HEURISTIC pro-reason
    (raises its graded strength), one carries nothing. The graded term picks
    the one with the supporter.
    """
    supported = MoveProbe(
        move_id="sup", reasons=("pro:opposition",), child_eval=0
    )
    plain = MoveProbe(move_id="pln", child_eval=0)
    graph = build_root_argument_graph([supported, plain], _POLICY)
    # FACT keys are equal (no FACT material on either).
    assert fact_only_key(supported, graph) == fact_only_key(plain, graph)
    chosen = lexicographic_decide([supported, plain], graph)
    assert chosen is not None
    assert chosen.move_id == "sup"


def test_decider_is_evaluator_free() -> None:
    """The decider reads ``probe.child_eval`` directly — no live evaluator.

    Two clean survivors identical except for ``child_eval``: the smaller
    (better for the mover) wins term-5 tiebreak.
    """
    better = MoveProbe(move_id="b", child_eval=-50)
    worse = MoveProbe(move_id="w", child_eval=50)
    graph = build_root_argument_graph([better, worse], _POLICY)
    # FACT keys tie; the graded layer ties (no witnesses on either); term 5
    # (probe.child_eval) breaks it.
    chosen = lexicographic_decide([better, worse], graph)
    assert chosen is not None
    assert chosen.move_id == "b"


def test_decider_fact_decision_dominates_graded(
    # Helper assertion: the §7 fact-as-highest-value property.
) -> None:
    """A FACT pro outranks even a much better graded strength (term 2 > term 3)."""
    fact_move = MoveProbe(
        move_id="f",
        reasons=("pro:material:100",),
        objections=("obj:loses_opposition",),
    )
    graded_move = MoveProbe(
        move_id="g", reasons=("pro:opposition",)
    )
    graph = build_root_argument_graph([fact_move, graded_move], _POLICY)
    chosen = lexicographic_decide([fact_move, graded_move], graph)
    assert chosen is not None
    # The FACT pro decides despite the lower graded strength.
    assert chosen.move_id == "f"
