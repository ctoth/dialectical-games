"""Property tests for the lexicographic decider (design §7).

Three invariants the core decider must satisfy:

* **idempotence under input permutation** — :func:`lexicographic_decide`
  returns the same probe regardless of input order. The key ends in
  ``probe.move_id``, so the ordering is total and deterministic.

* **total ordering on FACT survivors** — two probes with the SAME FACT
  key are distinguished only by the graded / tiebreak terms (so the
  ordering of the pair is stable under permutation); two probes with
  DIFFERENT FACT keys are decided strictly by FACT (so the better FACT
  key always wins regardless of graded values).

* **fact_only_key lower-bound dominance** — for any pair ``p1``, ``p2``,
  if ``fact_only_key(p1) < fact_only_key(p2)`` then
  :func:`lexicographic_decide` over ``{p1, p2}`` always returns ``p1``.
  This is the §7 fact-as-highest-value guarantee phrased as a pairwise
  invariant on the primitive.

All probes are drawn by ``hypothesis`` strategies — the FACT properties
are exercised across the typed evidence taxonomy
(``dialectical_games.evidence``).
"""

from __future__ import annotations

import random

import pytest
from doxa import Opinion
from hypothesis import given, settings, strategies as st

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    build_root_argument_graph,
)
from dialectical_games.decider import fact_only_key, lexicographic_decide


# ---------------------------------------------------------------------------
# A stub policy reused from the unit tests — magnitude-insensitive HEURISTIC
# witnesses, monotone base rate, dogmatic edge trust.
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
# Probe strategies — drawn over the typed evidence labels the parser accepts.
# ---------------------------------------------------------------------------

# A small spread of FACT pro / objection / reply / defense labels typed by
# ``dialectical_games.evidence.to_argument_evidence``. Keeping the set small
# means the strategies cover a wide range of FACT-key tuples in few samples.
_FACT_PROS = (
    "pro:terminal_win",
    "pro:material:100",
    "pro:material:300",
    "pro:crown",
)
_FACT_OBJECTIONS = (
    "obj:terminal_loss",
    "obj:allows_shot:100",
    "obj:allows_shot:300",
)
_FACT_REPLIES = (
    "reply:terminal_loss",
    "reply:material:100",
    "reply:material:200",
)
# Defenses keyed to one of the reply labels above.
_FACT_DEFENSES = (
    "defense:holds_exchange@reply:material:100",
    "defense:holds_exchange@reply:material:200",
)
# A small HEURISTIC vocabulary the evidence parser recognises.
_HEURISTIC_PROS = ("pro:opposition", "pro:back_rank_hold")
_HEURISTIC_OBJECTIONS = ("obj:loses_opposition",)


def _label_subset(labels: tuple[str, ...]) -> st.SearchStrategy[tuple[str, ...]]:
    """Draw an ordered subset (no duplicates) from ``labels``."""
    return st.lists(
        st.sampled_from(labels), max_size=len(labels), unique=True
    ).map(tuple)


@st.composite
def _probes(draw: st.DrawFn, move_id: str) -> MoveProbe:
    reasons = draw(_label_subset(_FACT_PROS + _HEURISTIC_PROS))
    objections = draw(_label_subset(_FACT_OBJECTIONS + _HEURISTIC_OBJECTIONS))
    reply_attacks = draw(_label_subset(_FACT_REPLIES))
    defenses = draw(_label_subset(_FACT_DEFENSES))
    child_eval = draw(st.integers(min_value=-500, max_value=500))
    contested = draw(st.booleans())
    return MoveProbe(
        move_id=move_id,
        reasons=reasons,
        objections=objections,
        reply_attacks=reply_attacks,
        defenses=defenses,
        child_eval=child_eval,
        contested=contested,
    )


@st.composite
def _probe_set(draw: st.DrawFn) -> list[MoveProbe]:
    n = draw(st.integers(min_value=1, max_value=6))
    return [draw(_probes(f"m{i}")) for i in range(n)]


# ---------------------------------------------------------------------------
# Invariant: idempotence under input permutation
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(probes=_probe_set(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_decider_idempotent_under_permutation(
    probes: list[MoveProbe], seed: int
) -> None:
    """The decider returns the same move regardless of probe input order.

    Two calls on the same probe set, the second on a permutation, must
    agree — the key terminates in ``probe.move_id`` so the ordering is a
    total function of the probe set.
    """
    graph = build_root_argument_graph(probes, _POLICY)
    first = lexicographic_decide(probes, graph)
    permuted = probes[:]
    random.Random(seed).shuffle(permuted)
    second = lexicographic_decide(permuted, graph)
    assert (first is None) == (second is None)
    if first is not None and second is not None:
        assert first.move_id == second.move_id


# ---------------------------------------------------------------------------
# Invariant: total ordering on FACT survivors
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(probes=_probe_set(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_decider_total_ordering_under_permutation(
    probes: list[MoveProbe], seed: int
) -> None:
    """The decider's chosen move is invariant under any probe permutation.

    A stronger statement than idempotence: not only do the two calls agree
    on which probe was chosen, the chosen probe is the SAME object — the
    decider is a true total ordering, not merely a stable selection of
    *some* maximum.
    """
    graph = build_root_argument_graph(probes, _POLICY)
    first = lexicographic_decide(probes, graph)
    permuted = probes[:]
    random.Random(seed).shuffle(permuted)
    second = lexicographic_decide(permuted, graph)
    if first is None:
        assert second is None
        return
    assert second is not None
    assert first is second  # identity, not just equality


# ---------------------------------------------------------------------------
# Invariant: fact_only_key lower-bound dominance
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    p1=_probes("m1"),
    p2=_probes("m2"),
)
def test_strict_fact_key_winner_always_chosen(
    p1: MoveProbe, p2: MoveProbe
) -> None:
    """If ``fact_only_key(p1) < fact_only_key(p2)``, the decider picks ``p1``.

    The §7 fact-as-highest-value guarantee at the pair level: a strictly
    better FACT key dominates any graded outcome. Skips pairs whose FACT
    keys tie (the graded layer is then *meant* to break the tie — not a
    FACT-decided pair).

    Restricted to pairs where BOTH probes survive the crisp layer (so the
    decider's candidate set is the same as the input set). With one
    eliminated probe the survivor wins trivially, which still satisfies
    the guarantee but does not exercise the FACT-tier ordering.
    """
    probes = [p1, p2]
    graph = build_root_argument_graph(probes, _POLICY)
    survivors = graph.survivors
    if p1.move_id not in survivors or p2.move_id not in survivors:
        return
    key1 = fact_only_key(p1, graph)
    key2 = fact_only_key(p2, graph)
    if key1 == key2:
        return  # FACT-tied — the graded layer may decide either way.
    better, worse = (p1, p2) if key1 < key2 else (p2, p1)
    chosen = lexicographic_decide(probes, graph)
    assert chosen is not None
    assert chosen.move_id == better.move_id, (
        key1,
        key2,
        chosen.move_id,
        worse.move_id,
    )
