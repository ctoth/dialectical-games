"""Game-agnostic lexicographic FACT-then-graded decider (design §7).

The single canonical move decider for the core. Pure function: takes the
:class:`MoveProbe` set + the :class:`RootArgumentGraph` the probes were
evaluated against, returns the chosen :class:`MoveProbe` (or ``None`` for
an empty probe set — a terminal position).

The decider is **board-free** and **evaluator-free**: term 5's static-eval
tiebreak reads the cartridge-precomputed :attr:`MoveProbe.child_eval`
(populated cartridge-side at probe time — Phase-2 settlement 1) rather
than applying a move to a board and calling a live ``static_evaluation``.

Phase 5 chunk 1: every term in the key is computed from typed
:class:`ArgumentEvidence` (``role`` / ``tier`` / ``magnitude``). The
``Value`` enum is gone — cartridges encode importance via magnitude on
their own scale. The single sentinel :data:`_TERMINAL_LOSS_MAGNITUDE`
lets a cartridge express "this is a game-winning fact" as a magnitude
the core can naturally order above any finite cartridge magnitude.

The key (per surviving move, lexicographic — smaller is better; the key is
consumed by :func:`min`):

1. **minimise the worst unavoidable FACT-objection magnitude.** Zero for
   any grounded crisp survivor; non-zero only in the design §6
   empty-survivor fallback. A forced terminal game loss outranks any
   finite material loss via :data:`_TERMINAL_LOSS_MAGNITUDE`.
2. **maximise the strongest FACT pro magnitude** (term 2). The cartridge
   controls the scale — a "winning fact" is one whose magnitude equals
   :data:`_TERMINAL_LOSS_MAGNITUDE` (the cartridge sets it that way) and
   so dominates any finite pro.
3. **maximise the move's opinion-valued graded strength** (design
   V1.5-D7) — ``graph.ranking["move_scores"]`` from the graded layer.
4. **maximise the count of accepted HEURISTIC pro-reasons** — the v1
   support proxy, retained for redundancy.
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
from dialectical_games.evidence import Role
from dialectical_games.scheme import Tier


__all__ = [
    "fact_only_key",
    "lexicographic_decide",
]


#: A forced terminal game loss outranks every finite cartridge magnitude.
#: The single load-bearing literal in this module. Cartridges that want to
#: express "this fact is game-winning / game-losing" set the
#: :attr:`ArgumentEvidence.magnitude` of their terminal-fact evidence to
#: this value; the core then naturally orders it above any finite magnitude
#: a cartridge would assign to material / positional facts.
_TERMINAL_LOSS_MAGNITUDE = 10**9


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
    FACT attacker. A FACT attacker defeated by a keyed FACT defense never
    contributes — its argument is not in the grounded extension.
    """
    move_arg = graph.move_arguments.get(probe.move_id)
    if move_arg is not None and move_arg in graph.grounded_extension:
        return 0

    worst = 0
    for ev in probe.evidence:
        if ev.tier is not Tier.FACT:
            continue
        if ev.role is Role.OBJECTION:
            attacker_id = obj_arg_id(probe.move_id, ev)
        elif ev.role is Role.REPLY_ATTACK:
            attacker_id = reply_arg_id(probe.move_id, ev)
        else:
            continue
        # Only an UNDEFEATED attacker (its argument grounded) contributes.
        if attacker_id not in graph.grounded_extension:
            continue
        if ev.magnitude > worst:
            worst = ev.magnitude
    return worst


def _fact_pro_priority(probe: MoveProbe) -> int:
    """The strongest FACT-pro magnitude for ``probe`` (term 2).

    Magnitude-only: the cartridge encodes "how strong is this pro" entirely
    via :attr:`ArgumentEvidence.magnitude`. A cartridge that wants its
    "game-winning" pros to dominate any "material" pros sets the winning
    pros' magnitude to :data:`_TERMINAL_LOSS_MAGNITUDE`; finite material
    magnitudes then naturally sort below.

    Returns 0 when the probe has no FACT pro — the smallest magnitude, so
    the move sorts (in the negated lexicographic key) below any probe that
    carries a positive-magnitude FACT pro.
    """
    return max(
        (
            ev.magnitude
            for ev in probe.evidence
            if ev.tier is Tier.FACT and ev.role is Role.PRO
        ),
        default=0,
    )


# --- graded terms (key terms 3 + 4) -----------------------------------------


def _graded_strength(probe: MoveProbe, graph: RootArgumentGraph) -> float:
    """The move's opinion-valued graded strength (term 3, design V1.5-D7).

    Reads ``graph.ranking["move_scores"]`` — each surviving move's
    ``Opinion.expectation()`` keyed by move id. A move absent from the
    graded layer (no ranking, or not in the crisp survivor set) scores the
    neutral 0.5 expectation.
    """
    move_scores = graph.ranking.get("move_scores")
    if not move_scores:
        return 0.5
    return float(move_scores.get(probe.move_id, 0.5))


def _accepted_heuristic_pro_count(probe: MoveProbe) -> int:
    """The accepted-HEURISTIC-pro count for ``probe`` (term 4).

    Counts every HEURISTIC-tier pro-reason on the probe uniformly. FACT
    pros never contribute — they are term 2's business.
    """
    total = 0
    for ev in probe.evidence:
        if ev.tier is Tier.HEURISTIC and ev.role is Role.PRO:
            total += 1
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
) -> tuple[int, int]:
    """The FACT-only selector key for ``probe`` (terms 1-2 only).

    A pure helper exposing the FACT-bearing primitive the §7
    fact-preservation differential tests use as an independent reference
    for "what does the FACT layer alone say?". Smaller is better — the key
    is consumed by :func:`min`.

    The two components:

    1. the worst unavoidable FACT-objection magnitude
       (:func:`_worst_fact_objection_magnitude`);
    2. the negated strongest FACT pro magnitude (``-fact_pro_priority``)
       — more-pro sorts first.

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
    pro_priority = _fact_pro_priority(probe)
    return (magnitude, -pro_priority)


def _selection_key(
    probe: MoveProbe, graph: RootArgumentGraph
) -> tuple[int, int, int, int, int, str]:
    """The full lexicographic selection key for ``probe`` (design §7).

    Smaller is better — consumed by :func:`min`. Five terms:

    1. worst unavoidable FACT-objection magnitude (term 1);
    2. negated strongest FACT pro magnitude (term 2);
    3. graded-layer strength scaled to a negated int (term 3);
    4. accepted-HEURISTIC-pro count, negated (term 4);
    5. cartridge-precomputed ``probe.child_eval`` (smaller = better for
       the mover under the checkers convention, but the cartridge owns the
       sign — the value is read verbatim), then ``probe.move_id`` for a
       total deterministic tiebreak (term 5, two components).

    Reads ``probe.child_eval`` directly — the decider never applies a move
    to a board, never calls a live evaluator (Phase-2 settlement 1).
    """
    objection_magnitude = _worst_fact_objection_magnitude(probe, graph)
    pro_priority = _fact_pro_priority(probe)
    # The graded strength is a float in (0, 1); scale it to an integer key
    # by multiplying by the terminal sentinel and negating so larger
    # strength sorts first. ``_TERMINAL_LOSS_MAGNITUDE`` is reused as the
    # scale so the graded key occupies a comparable integer range.
    graded_key = -round(_graded_strength(probe, graph) * _TERMINAL_LOSS_MAGNITUDE)
    heuristic_pro_key = -_accepted_heuristic_pro_count(probe)
    return (
        objection_magnitude,
        -pro_priority,
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
    ``probe.evidence`` / ``probe.move_id`` and ``graph``. No board, no
    cartridge callbacks, no live evaluator.
    """
    candidates = _candidates(probes, graph)
    if not candidates:
        return None
    return min(candidates, key=lambda p: _selection_key(p, graph))
