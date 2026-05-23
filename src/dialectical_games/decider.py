"""Game-agnostic lexicographic FACT-then-graded decider (design §7).

The single canonical move decider for the core. Pure function: takes the
:class:`MoveProbe` set + the :class:`RootArgumentGraph` the probes were
evaluated against, returns the chosen :class:`MoveProbe` (or ``None`` for
an empty probe set — a terminal position).

The decider is **board-free** and **evaluator-free**: term 5's static-eval
tiebreak reads the cartridge-precomputed :attr:`MoveProbe.child_eval`
(populated cartridge-side at probe time — Phase-2 settlement 1) rather
than applying a move to a board and calling a live ``static_evaluation``.

The key (per surviving move, lexicographic — smaller is better; the key is
consumed by :func:`min`):

1. **minimise the worst unavoidable FACT-objection magnitude.** Zero for
   any grounded crisp survivor; non-zero only in the design §6
   empty-survivor fallback. A forced terminal game loss outranks any
   finite material loss via :data:`_TERMINAL_LOSS_MAGNITUDE`.
2. **maximise the FACT-tier pro value** by the priority tuple
   ``winning > large material > crown > small material`` (design §7
   term 2). The material component is **net** — the immediate FACT
   capture minus any defended reply that recaptures part of it.
3. **maximise the move's opinion-valued graded strength** (design
   V1.5-D7) — ``graph.ranking["move_scores"]`` from the graded layer.
4. **maximise the value-weighted count of accepted HEURISTIC
   pro-reasons** — the v1 support proxy, retained for redundancy.
5. **deterministic tiebreak**: the cartridge-precomputed
   :attr:`MoveProbe.child_eval` (smaller = better for the mover under the
   checkers convention; this term is the cartridge's own choice of
   evaluation), then the move id string.

The graded terms 3-4 come strictly **after** the FACT terms 1-2 in the
lexicographic ordering: a FACT decision always dominates a graded one
(design V1.5-D6 — fact-as-highest-value).

Two public names:

* :func:`lexicographic_decide` — the full-key decider.
* :func:`fact_only_key` — JUST the FACT terms (terms 1-2). A pure helper
  the §7 fact-preservation differential tests use as an independent
  reference for the FACT-only key.

This module imports only ``dialectical_games`` and the stdlib.
"""

from __future__ import annotations

from collections.abc import Iterable

from dialectical_games.arguments import (
    MoveProbe,
    RootArgumentGraph,
    obj_arg_id,
    reply_arg_id,
)
from dialectical_games.evidence import to_argument_evidence
from dialectical_games.scheme import Tier, Value


__all__ = [
    "fact_only_key",
    "lexicographic_decide",
]


# --- term-1 / term-2 magnitude scales ---------------------------------------

#: A forced terminal game loss outranks every finite material loss in term 1.
_TERMINAL_LOSS_MAGNITUDE = 10**9

#: Strictly more than one man (in centipawn-scale units) is "large material".
_LARGE_MATERIAL_THRESHOLD = 100

#: Graded-strength scale: opinion expectation in (0, 1) -> negated int.
_GRADED_SCALE = 10**9

#: A move absent from the graded layer scores the neutral 0.5 expectation.
_NEUTRAL_GRADED_STRENGTH = 0.5

#: Uniform weight per accepted HEURISTIC pro-reason — design §7 names no
#: per-value priority for the HEURISTIC values, so the only non-discretionary
#: reading is a uniform weight of 1. See the v1 selector-key derivation.
_HEURISTIC_PRO_WEIGHT = 1


# --- FACT terms (key terms 1 + 2) -------------------------------------------


def _worst_fact_objection_magnitude(
    probe: MoveProbe, graph: RootArgumentGraph
) -> int:
    """The worst unavoidable FACT-objection magnitude on ``probe`` (term 1).

    **0 for a grounded crisp survivor** — a move whose ``move:`` argument is
    in the grounded extension has no undefeated FACT attacker, so it carries
    no unavoidable loss. This is the normal case.

    Non-zero only in the design §6 empty-survivor fallback: with no move
    grounded, this term is the magnitude of the move's worst **undefeated**
    FACT attacker — a terminal game loss outranking any finite material loss
    via :data:`_TERMINAL_LOSS_MAGNITUDE`. A FACT attacker defeated by a
    keyed FACT defense never contributes — its argument is not in the
    grounded extension.
    """
    move_arg = graph.move_arguments.get(probe.move_id)
    if move_arg is not None and move_arg in graph.grounded_extension:
        return 0

    worst = 0
    attackers: list[tuple[str, str]] = []
    for label in probe.objections:
        attackers.append((label, obj_arg_id(probe.move_id, label)))
    for label in probe.reply_attacks:
        attackers.append((label, reply_arg_id(probe.move_id, label)))

    for label, attacker_id in attackers:
        try:
            evidence = to_argument_evidence(label)
        except ValueError:
            continue
        if evidence.tier is not Tier.FACT:
            continue
        # Only an UNDEFEATED attacker (its argument grounded) contributes.
        if attacker_id not in graph.grounded_extension:
            continue
        if evidence.value is Value.WINNING:
            worst = max(worst, _TERMINAL_LOSS_MAGNITUDE)
        elif evidence.magnitude is not None:
            worst = max(worst, evidence.magnitude)
    return worst


def _defended_reply_giveback(probe: MoveProbe) -> int:
    """The FACT material the move's keyed defenses concede back to the opponent.

    A ``defense:holds_exchange@{answered}`` on a grounded survivor proves the
    exchange is even / favourable, but the opponent's ``{answered}`` forcing
    reply still recaptures its ``reply:material:{n}`` magnitude. The move's
    net pro-material is its immediate capture minus this giveback.
    """
    giveback = 0
    seen: set[str] = set()
    for label in probe.defenses:
        try:
            evidence = to_argument_evidence(label)
        except ValueError:
            continue
        if evidence.tier is not Tier.FACT or evidence.answered is None:
            continue
        answered = evidence.answered
        if answered in seen:
            continue
        seen.add(answered)
        try:
            answered_evidence = to_argument_evidence(answered)
        except ValueError:
            continue
        if (
            answered_evidence.value is Value.MATERIAL
            and answered_evidence.magnitude is not None
        ):
            giveback += answered_evidence.magnitude
    return giveback


def _fact_pro_priority(probe: MoveProbe) -> tuple[int, int, int, int]:
    """The FACT pro-value priority tuple for ``probe`` (term 2).

    Returns ``(winning, large_material, crown, small_material)``; each
    component "bigger is better". The material component is the **net** —
    the move's biggest immediate FACT pro-material minus
    :func:`_defended_reply_giveback`. A defended even exchange scores 0
    material; a clean small gain scores in ``small_material``; a net gain
    of more than one man scores in ``large_material``.
    """
    winning = 0
    crown = 0
    gross_material = 0
    for label in probe.reasons:
        try:
            evidence = to_argument_evidence(label)
        except ValueError:
            continue
        if evidence.tier is not Tier.FACT:
            continue
        if evidence.value is Value.WINNING:
            winning = 1
        elif evidence.value is Value.KING_COUNT:
            crown = 1
        elif evidence.value is Value.MATERIAL and evidence.magnitude is not None:
            gross_material = max(gross_material, evidence.magnitude)

    net_material = gross_material - _defended_reply_giveback(probe)
    large_material = 0
    small_material = 0
    if net_material > _LARGE_MATERIAL_THRESHOLD:
        large_material = net_material
    elif net_material > 0:
        small_material = net_material
    return (winning, large_material, crown, small_material)


# --- graded terms (key terms 3 + 4) -----------------------------------------


def _graded_strength(probe: MoveProbe, graph: RootArgumentGraph) -> float:
    """The move's opinion-valued graded strength (term 3, design V1.5-D7).

    Reads ``graph.ranking["move_scores"]`` — each surviving move's
    ``Opinion.expectation()`` keyed by move id. A move absent from the
    graded layer (no ranking, or not in the crisp survivor set) scores the
    neutral :data:`_NEUTRAL_GRADED_STRENGTH`.
    """
    move_scores = graph.ranking.get("move_scores")
    if not move_scores:
        return _NEUTRAL_GRADED_STRENGTH
    return float(move_scores.get(probe.move_id, _NEUTRAL_GRADED_STRENGTH))


def _accepted_heuristic_pro_count(probe: MoveProbe) -> int:
    """The value-weighted accepted-HEURISTIC-pro count for ``probe`` (term 4).

    Counts every HEURISTIC-tier pro-reason on the probe at the uniform
    :data:`_HEURISTIC_PRO_WEIGHT`. FACT pros never contribute — they are
    term 2's business.
    """
    total = 0
    for label in probe.reasons:
        try:
            evidence = to_argument_evidence(label)
        except ValueError:
            continue
        if evidence.tier is Tier.HEURISTIC:
            total += _HEURISTIC_PRO_WEIGHT
    return total


# --- candidate set ----------------------------------------------------------


def _candidates(
    probes: Iterable[MoveProbe], graph: RootArgumentGraph
) -> list[MoveProbe]:
    """The crisp survivors among ``probes`` — the candidate set the key ranks.

    ``graph.survivors`` is the grounded ``move:`` set, or — under the design
    §6 empty-survivor fallback — all moves. Restricting to this set is the
    structural guarantee that the decider can never resurrect a
    crisply-eliminated move (design §7).
    """
    probe_list = list(probes)
    survivors = graph.survivors
    candidates = (
        [p for p in probe_list if p.move_id in survivors]
        if survivors
        else list(probe_list)
    )
    if not candidates:
        candidates = list(probe_list)
    return candidates


# --- public surface: the full key + the FACT-only key -----------------------


def fact_only_key(
    probe: MoveProbe, graph: RootArgumentGraph
) -> tuple[int, int, int, int, int]:
    """The FACT-only selector key for ``probe`` (terms 1-2 only).

    A pure helper exposing the FACT-bearing primitive the §7
    fact-preservation differential tests use as an independent reference
    for "what does the FACT layer alone say?". Smaller is better — the key
    is consumed by :func:`min`.

    The five components:

    1. the worst unavoidable FACT-objection magnitude
       (:func:`_worst_fact_objection_magnitude`);
    2. the negated FACT pro-value priority tuple (``-winning``,
       ``-large_material``, ``-crown``, ``-small_material``) — four
       components, each "more pro" sorting first.

    Two probes with an equal FACT-only key are not distinguished by the
    FACT layer; any difference in the engine's choice between them is the
    graded layer's doing. By design V1.5-D6 (fact-as-highest-value), the
    move with the strictly best FACT-only key is the move
    :func:`lexicographic_decide` returns when that FACT key is unique among
    survivors — the property the §7 differential test pins.

    Pure function: reads only ``probe`` and ``graph``; no board, no
    evaluator, no cartridge dispatch.
    """
    magnitude = _worst_fact_objection_magnitude(probe, graph)
    winning, large_material, crown, small_material = _fact_pro_priority(probe)
    return (magnitude, -winning, -large_material, -crown, -small_material)


def _selection_key(
    probe: MoveProbe, graph: RootArgumentGraph
) -> tuple[int, int, int, int, int, int, int, int, str]:
    """The full lexicographic selection key for ``probe`` (design §7).

    Smaller is better — consumed by :func:`min`. Five terms:

    1. worst unavoidable FACT-objection magnitude (term 1);
    2. FACT pro-value priority tuple, negated so more-pro sorts first
       (term 2, four components);
    3. graded-layer strength scaled to a negated int (term 3);
    4. value-weighted accepted-HEURISTIC-pro count, negated (term 4);
    5. cartridge-precomputed ``probe.child_eval`` (smaller = better for
       the mover under the checkers convention, but the cartridge owns the
       sign — the value is read verbatim), then ``probe.move_id`` for a
       total deterministic tiebreak (term 5, two components).

    Reads ``probe.child_eval`` directly — the decider never applies a move
    to a board, never calls a live evaluator (Phase-2 settlement 1).
    """
    objection_magnitude = _worst_fact_objection_magnitude(probe, graph)
    winning, large_material, crown, small_material = _fact_pro_priority(probe)
    graded_key = -round(_graded_strength(probe, graph) * _GRADED_SCALE)
    heuristic_pro_key = -_accepted_heuristic_pro_count(probe)
    return (
        objection_magnitude,
        -winning,
        -large_material,
        -crown,
        -small_material,
        graded_key,
        heuristic_pro_key,
        probe.child_eval,
        probe.move_id,
    )


def lexicographic_decide(
    probes: Iterable[MoveProbe],
    graph: RootArgumentGraph,
) -> MoveProbe | None:
    """Decide a move from the crisp survivors (design §7).

    Restricts ``probes`` to the crisp survivors (``graph.survivors`` — or,
    under the design §6 empty-survivor fallback, all probes), then returns
    the probe minimising :func:`_selection_key`. The graded terms 3-4 come
    strictly after the FACT terms 1-2, so a FACT-decided position is never
    overridden by the graded layer (design V1.5-D6).

    Returns ``None`` only for an empty ``probes`` set (a terminal position
    — no move to choose). The caller's orchestrator is responsible for the
    terminal-position case.

    Pure function: reads ``probe.child_eval`` / ``probe.contested`` /
    ``probe.reasons`` / ``probe.objections`` / ``probe.reply_attacks`` /
    ``probe.defenses`` / ``probe.move_id`` and ``graph``. No board, no
    cartridge callbacks, no live evaluator.
    """
    candidates = _candidates(probes, graph)
    if not candidates:
        return None
    return min(candidates, key=lambda p: _selection_key(p, graph))
