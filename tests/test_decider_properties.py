"""Property tests for the lexicographic decider (design §7).

Three invariants the core decider must satisfy:

* **idempotence under input permutation** — :func:`lexicographic_decide`
  returns the same probe regardless of input order. The key ends in
  ``probe.move_id``, so the ordering is total and deterministic.

* **total ordering on FACT survivors** — the decider returns the SAME
  probe object regardless of input order.

* **fact_only_key lower-bound dominance** — if
  ``fact_only_key(p1) < fact_only_key(p2)``, the decider over ``{p1, p2}``
  always returns ``p1`` (the §7 fact-as-highest-value guarantee at the
  pair level).

Phase 5 chunk 1: probes are built from typed
:class:`~dialectical_games.evidence.ArgumentEvidence` directly. The
strategies span all four :class:`Role` × both :class:`Tier` combinations
across cartridge-agnostic magnitude ranges.
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
# Evidence strategies — game-agnostic, generic Role × Tier × magnitude
# ---------------------------------------------------------------------------


# A small spread of cartridge-style magnitudes. The terminal-loss sentinel
# appears explicitly so the strategies exercise the "winning" boundary.
_FINITE_MAGNITUDES = st.sampled_from((50, 100, 200, 300))
_FACT_MAGNITUDES = st.one_of(_FINITE_MAGNITUDES, st.just(_TERMINAL_LOSS_MAGNITUDE))
_TAGS = st.sampled_from(("a", "b", "c", "d", "e", "f"))


def _fact_pro_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    return st.builds(
        ArgumentEvidence,
        role=st.just(Role.PRO),
        tier=st.just(Tier.FACT),
        magnitude=_FACT_MAGNITUDES,
        answered=st.none(),
        tag=_TAGS,
    )


def _fact_objection_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    return st.builds(
        ArgumentEvidence,
        role=st.just(Role.OBJECTION),
        tier=st.just(Tier.FACT),
        magnitude=_FACT_MAGNITUDES,
        answered=st.none(),
        tag=_TAGS,
    )


def _fact_reply_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    return st.builds(
        ArgumentEvidence,
        role=st.just(Role.REPLY_ATTACK),
        tier=st.just(Tier.FACT),
        magnitude=_FACT_MAGNITUDES,
        answered=st.none(),
        tag=_TAGS,
    )


def _heuristic_pro_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    return st.builds(
        ArgumentEvidence,
        role=st.just(Role.PRO),
        tier=st.just(Tier.HEURISTIC),
        magnitude=st.integers(min_value=0, max_value=10),
        answered=st.none(),
        tag=_TAGS,
    )


def _heuristic_objection_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    return st.builds(
        ArgumentEvidence,
        role=st.just(Role.OBJECTION),
        tier=st.just(Tier.HEURISTIC),
        magnitude=st.integers(min_value=0, max_value=10),
        answered=st.none(),
        tag=_TAGS,
    )


@st.composite
def _evidence_tuple(draw: st.DrawFn) -> tuple[ArgumentEvidence, ...]:
    """Draw a small tuple of evidence + optionally a FACT defense whose
    ``answered`` is one of the FACT attackers on the same probe.

    Uses object identity for the defense's ``answered`` (the same evidence
    object selected from the tuple) to exercise the identity-keyed defeat
    edge in the builder.
    """
    pros = draw(st.lists(_fact_pro_strategy(), max_size=2))
    objs = draw(st.lists(_fact_objection_strategy(), max_size=2))
    replies = draw(st.lists(_fact_reply_strategy(), max_size=2))
    h_pros = draw(st.lists(_heuristic_pro_strategy(), max_size=2))
    h_objs = draw(st.lists(_heuristic_objection_strategy(), max_size=1))

    attackers = objs + replies
    defenses: list[ArgumentEvidence] = []
    if attackers and draw(st.booleans()):
        target = draw(st.sampled_from(attackers))
        defenses.append(
            ArgumentEvidence(
                role=Role.DEFENSE,
                tier=Tier.FACT,
                magnitude=draw(_FINITE_MAGNITUDES),
                answered=target,
                tag=draw(_TAGS),
            )
        )

    return tuple(pros + objs + replies + h_pros + h_objs + defenses)


@st.composite
def _probe(draw: st.DrawFn, move_id: str) -> MoveProbe:
    return MoveProbe(
        move_id=move_id,
        evidence=draw(_evidence_tuple()),
        child_eval=draw(st.integers(min_value=-500, max_value=500)),
        contested=draw(st.booleans()),
    )


@st.composite
def _probe_set(draw: st.DrawFn) -> list[MoveProbe]:
    n = draw(st.integers(min_value=1, max_value=6))
    return [draw(_probe(f"m{i}")) for i in range(n)]


# ---------------------------------------------------------------------------
# Invariant: idempotence under input permutation
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=150, deadline=None)
@given(probes=_probe_set(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_decider_idempotent_under_permutation(
    probes: list[MoveProbe], seed: int
) -> None:
    """The decider returns the same move regardless of probe input order."""
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
    """The decider's chosen move is INVARIANT (identity) under permutation."""
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
    p1=_probe("m1"),
    p2=_probe("m2"),
)
def test_strict_fact_key_winner_always_chosen(
    p1: MoveProbe, p2: MoveProbe
) -> None:
    """If ``fact_only_key(p1) < fact_only_key(p2)``, the decider picks ``p1``."""
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


# ---------------------------------------------------------------------------
# Property: terminal-magnitude FACT pro outranks any finite FACT pro
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(finite_mag=_FINITE_MAGNITUDES)
def test_property_terminal_pro_dominates_any_finite_pro(
    finite_mag: int,
) -> None:
    """A terminal-magnitude FACT pro on one move ALWAYS beats any
    finite-magnitude FACT pro on a sibling.

    The §7 fact-as-highest-value property phrased as a magnitude
    invariant: the cartridge's "winning fact" convention
    (magnitude = ``_TERMINAL_LOSS_MAGNITUDE``) is strictly dominant.
    """
    winning = MoveProbe(
        move_id="w",
        evidence=(
            ArgumentEvidence(
                role=Role.PRO,
                tier=Tier.FACT,
                magnitude=_TERMINAL_LOSS_MAGNITUDE,
                tag="winning",
            ),
        ),
    )
    finite = MoveProbe(
        move_id="f",
        evidence=(
            ArgumentEvidence(
                role=Role.PRO,
                tier=Tier.FACT,
                magnitude=finite_mag,
                tag="finite",
            ),
        ),
    )
    graph = build_root_argument_graph([winning, finite], _POLICY)
    chosen = lexicographic_decide([winning, finite], graph)
    assert chosen is not None
    assert chosen.move_id == "w"
