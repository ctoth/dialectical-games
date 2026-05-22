"""Witness label -> typed ``ArgumentEvidence`` (design §5).

The comorphism that turns a stringly-typed witness label into typed evidence
carrying a ``Value``, a ``Tier`` and any parsed magnitude (design
``notes/checkers-design.md`` §4-5). dialectical-chess guessed evidence from
string prefixes; here every label is parsed once, in one place, into a closed
taxonomy — there is no prefix dispatch scattered through the codebase.

Phase 3a implemented the **FACT-tier** rows of the design §5 tables; Phase 4
adds the **HEURISTIC-tier** rows. The full §5 taxonomy this parser now knows:

    pro:terminal_win                  WINNING     FACT
    pro:material:{n}                  MATERIAL    FACT
    pro:crown                         KING_COUNT  FACT
    pro:shot_setup:{n}                MATERIAL    FACT
    obj:terminal_loss                 WINNING     FACT
    obj:allows_shot:{n}               MATERIAL    FACT
    obj:loses_exchange:{n}            MATERIAL    FACT
    reply:terminal_loss               WINNING     FACT
    reply:material:{n}                MATERIAL    FACT
    defense:holds_exchange@{answered} MATERIAL    FACT
    pro:opposition                    TEMPO       HEURISTIC
    pro:back_rank_hold                STRUCTURE   HEURISTIC
    pro:center:{n}                    STRUCTURE   HEURISTIC
    pro:mobility:{n}                  MOBILITY    HEURISTIC
    pro:formation:{kind}              STRUCTURE   HEURISTIC
    obj:loses_opposition              TEMPO       HEURISTIC
    obj:back_rank_break               STRUCTURE   HEURISTIC
    obj:single_corner_drift           STRUCTURE   HEURISTIC
    obj:exposes_man                   MATERIAL    HEURISTIC
    pro:frees_trapped_piece           STRUCTURE   HEURISTIC
    pro:safe_side_man                 STRUCTURE   HEURISTIC
    pro:strong_back_structure         STRUCTURE   HEURISTIC
    pro:king_centralised              KING_COUNT  HEURISTIC
    pro:cramps_opponent:{n}           MOBILITY    HEURISTIC
    obj:trapped_piece                 STRUCTURE   HEURISTIC
    obj:cedes_centre:{n}              STRUCTURE   HEURISTIC
    obj:weakens_back_structure        STRUCTURE   HEURISTIC

The witness-vocabulary enrichment added the last seven HEURISTIC rows (the
Tier A new witnesses plus the two cheap completions). ``pro:runaway`` is
deferred — see ``witnesses.py``.

A HEURISTIC label is a positional judgement, not a resolver/terminal proof; the
tier field is exactly the ``Bench-Capon_2003`` fact-as-highest-value bridge
(design §4) — a FACT label outranks every HEURISTIC one. ``pro:formation`` is
the one HEURISTIC label carrying a non-numeric tag: its ``{kind}`` is a closed
enum (``phalanx`` / ``bridge`` / ``echelon`` — design §5 "bridge / phalanx /
echelon") parsed into the ``magnitude``-less :class:`ArgumentEvidence` with the
kind kept in the ``label``.

A ``defense:`` label is **keyed to the specific objection / reply it answers**
(design §6 — "a defense defeats the objection/reply it answers, and only that
one"). The keyed form is ``defense:holds_exchange@{answered}`` where
``{answered}`` is itself a valid FACT objection / reply label (e.g.
``defense:holds_exchange@reply:material:100``). The ``answered`` field on the
parsed :class:`ArgumentEvidence` carries the target label so the crisp layer
(``arguments.py``) can wire the defense to *only* that attacker. The bare,
un-keyed ``defense:holds_exchange`` is still accepted by this parser (it is a
valid evidence type) but ``witnesses.py`` never emits it — every emitted
defense carries its target.

A FACT ``:{n}`` magnitude is the resolver's native **weighted material** unit
(man = 100, king = 150) — the same unit ``captures.ShotResult.material_net``
and ``ResolvedLine.material_swing`` report. ``reply:`` and ``defense:`` are
emitted by ``witnesses.py`` only when the resolver *proved* the line, so they
are FACT here; their HEURISTIC forms (a truncated resolver line) are simply
never produced and therefore never reach this parser.

A HEURISTIC ``:{n}`` magnitude is **not** a material unit: ``pro:center:{n}``
counts central-square occupation and ``pro:mobility:{n}`` counts a legal-move
gain. The magnitude is still a strictly positive base-10 integer, parsed into
the same ``magnitude`` field; its interpretation is the witness's own (a count,
not material). ``witnesses.py`` documents the exact count per HEURISTIC label.

This module imports only from within ``dialectical_games`` and the stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialectical_games.scheme import Tier, Value


@dataclass(frozen=True)
class ArgumentEvidence:
    """Typed evidence for one witness label (design §4-5).

    ``label`` is the original stringly-typed witness label; ``value`` is the
    AS2 value the witness promotes or demotes; ``tier`` is ``FACT`` for a
    resolver/terminal-proven witness and ``HEURISTIC`` for a positional one;
    ``magnitude`` is the parsed ``:{n}`` integer (weighted material) when the
    label carries one, else ``None``; ``answered`` is the objection / reply
    label a keyed ``defense:`` witness answers (design §6 "and only that one"),
    else ``None``.
    """

    label: str
    value: Value
    tier: Tier
    magnitude: int | None = None
    answered: str | None = None


# --- the §5 label taxonomy (FACT + HEURISTIC) -------------------------------
#
# Three tables. ``_FIXED`` — labels with no magnitude, mapped directly to their
# (Value, Tier). ``_MAGNITUDE`` — label prefixes that MUST carry a ``:{n}``
# integer magnitude, mapped to their (Value, Tier). ``_FORMATION_KINDS`` — the
# closed enum of ``pro:formation:{kind}`` suffixes (the one HEURISTIC label
# carrying a non-numeric tag). Splitting them keeps the parser dict lookups
# with no per-label branching.

_FIXED: dict[str, tuple[Value, Tier]] = {
    # FACT-tier (design §5, Phase 3a).
    "pro:terminal_win": (Value.WINNING, Tier.FACT),
    "pro:crown": (Value.KING_COUNT, Tier.FACT),
    "obj:terminal_loss": (Value.WINNING, Tier.FACT),
    "reply:terminal_loss": (Value.WINNING, Tier.FACT),
    "defense:holds_exchange": (Value.MATERIAL, Tier.FACT),
    # HEURISTIC-tier (design §5, Phase 4) — fixed, no magnitude.
    "pro:opposition": (Value.TEMPO, Tier.HEURISTIC),
    "pro:back_rank_hold": (Value.STRUCTURE, Tier.HEURISTIC),
    "obj:loses_opposition": (Value.TEMPO, Tier.HEURISTIC),
    "obj:back_rank_break": (Value.STRUCTURE, Tier.HEURISTIC),
    "obj:single_corner_drift": (Value.STRUCTURE, Tier.HEURISTIC),
    "obj:exposes_man": (Value.MATERIAL, Tier.HEURISTIC),
    # HEURISTIC-tier (witness-vocabulary enrichment) — fixed, no magnitude.
    "pro:frees_trapped_piece": (Value.STRUCTURE, Tier.HEURISTIC),
    "pro:safe_side_man": (Value.STRUCTURE, Tier.HEURISTIC),
    "pro:strong_back_structure": (Value.STRUCTURE, Tier.HEURISTIC),
    "pro:king_centralised": (Value.KING_COUNT, Tier.HEURISTIC),
    "obj:trapped_piece": (Value.STRUCTURE, Tier.HEURISTIC),
    "obj:weakens_back_structure": (Value.STRUCTURE, Tier.HEURISTIC),
}

_MAGNITUDE: dict[str, tuple[Value, Tier]] = {
    # FACT-tier (design §5, Phase 3a) — magnitude is weighted material.
    "pro:material": (Value.MATERIAL, Tier.FACT),
    "pro:shot_setup": (Value.MATERIAL, Tier.FACT),
    "obj:allows_shot": (Value.MATERIAL, Tier.FACT),
    "obj:loses_exchange": (Value.MATERIAL, Tier.FACT),
    "reply:material": (Value.MATERIAL, Tier.FACT),
    # HEURISTIC-tier (design §5, Phase 4) — magnitude is a positional COUNT,
    # not material (``pro:center`` central-square occupation gained,
    # ``pro:mobility`` legal-move-count gained).
    "pro:center": (Value.STRUCTURE, Tier.HEURISTIC),
    "pro:mobility": (Value.MOBILITY, Tier.HEURISTIC),
    # HEURISTIC-tier (witness-vocabulary enrichment) — magnitude is a
    # positional COUNT (``pro:cramps_opponent`` the opponent's legal-move-count
    # drop, ``obj:cedes_centre`` the central-square occupation lost).
    "pro:cramps_opponent": (Value.MOBILITY, Tier.HEURISTIC),
    "obj:cedes_centre": (Value.STRUCTURE, Tier.HEURISTIC),
}

# ``pro:formation:{kind}`` — the closed set of named formations (design §5
# "bridge / phalanx / echelon"). The kind is a non-numeric suffix; a label
# whose suffix is not one of these is rejected, never silently mistyped.
_FORMATION_PREFIX = "pro:formation"
_FORMATION_KINDS: frozenset[str] = frozenset({"phalanx", "bridge", "echelon"})
_FORMATION_TYPING: tuple[Value, Tier] = (Value.STRUCTURE, Tier.HEURISTIC)


# --- keyed defense labels (design §6) ---------------------------------------
#
# A defense is keyed to the objection / reply it answers with an ``@``
# separator: ``defense:holds_exchange@reply:material:100``. The part before the
# ``@`` is a bare defense type (it must be in ``_FIXED`` and be a ``defense:``
# label); the part after is the answered objection / reply label, itself parsed
# recursively so a malformed target is rejected.

_DEFENSE_KEY_SEP = "@"


def to_argument_evidence(label: str) -> ArgumentEvidence:
    """Map a witness label to typed :class:`ArgumentEvidence` (design §5).

    A fixed label (no magnitude) is looked up directly. A magnitude-carrying
    label has the form ``<prefix>:<n>`` where ``<n>`` is a base-10 integer;
    the prefix is looked up and ``<n>`` parsed into ``magnitude``. A keyed
    defense label has the form ``<defense-type>@<answered>`` where
    ``<answered>`` is itself a valid objection / reply label; the parsed
    ``answered`` field carries that target (design §6 "and only that one"). A
    ``pro:formation:{kind}`` label has a named-formation suffix from the closed
    ``_FORMATION_KINDS`` enum — not a magnitude.

    Both FACT and HEURISTIC §5 rows are recognised (Phase 4 added the
    HEURISTIC rows). Raises :class:`ValueError` for an empty, malformed, or
    unknown label, for a magnitude label whose ``:{n}`` part is missing or
    non-numeric, or for a ``pro:formation`` label with an unknown kind — a
    label is never silently mistyped.
    """
    if not label:
        raise ValueError("empty witness label")

    # A keyed defense label: ``<defense-type>@<answered-label>``.
    if _DEFENSE_KEY_SEP in label:
        defense_type, _, answered_label = label.partition(_DEFENSE_KEY_SEP)
        defense_fixed = _FIXED.get(defense_type)
        if defense_fixed is None or not defense_type.startswith("defense:"):
            raise ValueError(f"unknown keyed defense label {label!r}")
        if not answered_label:
            raise ValueError(
                f"keyed defense label {label!r} has an empty answered target"
            )
        # The answered target must itself be a valid objection / reply label.
        answered = to_argument_evidence(answered_label)
        if not (
            answered_label.startswith("obj:")
            or answered_label.startswith("reply:")
        ):
            raise ValueError(
                f"keyed defense label {label!r} answers a non-attack label "
                f"{answered_label!r}"
            )
        value, tier = defense_fixed
        return ArgumentEvidence(
            label=label, value=value, tier=tier, answered=answered_label
        )

    fixed = _FIXED.get(label)
    if fixed is not None:
        value, tier = fixed
        return ArgumentEvidence(label=label, value=value, tier=tier)

    # A magnitude label is ``<prefix>:<n>`` — split off the trailing ``:<n>``.
    head, sep, tail = label.rpartition(":")
    if not sep:
        raise ValueError(f"unknown witness label {label!r}")

    # ``pro:formation:{kind}`` — a HEURISTIC label whose tail is a named
    # formation kind, not a magnitude. The kind must be in the closed
    # ``_FORMATION_KINDS`` enum; an unknown kind is rejected, never mistyped.
    if head == _FORMATION_PREFIX:
        if tail not in _FORMATION_KINDS:
            raise ValueError(
                f"witness label {label!r} has an unknown formation kind "
                f"{tail!r} (known: {sorted(_FORMATION_KINDS)})"
            )
        value, tier = _FORMATION_TYPING
        return ArgumentEvidence(label=label, value=value, tier=tier)

    mag = _MAGNITUDE.get(head)
    if mag is None:
        raise ValueError(f"unknown witness label {label!r}")
    # ``{n}`` is a material gain/loss magnitude (design §5): a strictly
    # positive base-10 integer of bare ASCII digits. A signed (``-100``,
    # ``+100``) or zero magnitude is malformed — the witness producers only
    # ever emit positive magnitudes, and accepting a signed/zero one would
    # silently mistype a malformed label as valid FACT evidence (cf. the
    # malformed-label rejection above). ``str.isascii() and str.isdecimal()``
    # admits exactly a run of ASCII ``0``-``9`` and rejects empty strings,
    # signs, whitespace, and unicode-digit lookalikes (e.g. ``²``) that would
    # otherwise crash ``int()``.
    if not (tail.isascii() and tail.isdecimal()):
        raise ValueError(
            f"witness label {label!r} has a non-integer magnitude {tail!r}"
        )
    magnitude = int(tail)
    if magnitude <= 0:
        raise ValueError(
            f"witness label {label!r} has a non-positive magnitude {tail!r}"
        )
    value, tier = mag
    return ArgumentEvidence(
        label=label, value=value, tier=tier, magnitude=magnitude
    )
