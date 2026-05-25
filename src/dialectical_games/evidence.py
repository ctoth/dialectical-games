"""Typed evidence for one witness on one move (game-agnostic core).

Phase 5 chunk 1: the game-flavoured label parser is gone. The cartridge
constructs :class:`ArgumentEvidence` directly per witness and attaches the
tuple to :class:`~dialectical_games.arguments.MoveProbe.evidence`. The core
never inspects a label string, never dispatches on a prefix, never parses an
``@``-keyed defense — every dispatch is enum-typed.

The shape: one :class:`Role` enum (``PRO`` / ``OBJECTION`` / ``REPLY_ATTACK``
/ ``DEFENSE``) and one frozen :class:`ArgumentEvidence` dataclass carrying
``(role, tier, magnitude, answered, tag)``. ``answered`` is the EVIDENCE
OBJECT the defense answers (identity-typed, not a string label), so the
graph builders match the defense to its attacker via ``id(ev.answered)`` and
never through string equality. ``magnitude`` is an int on a scale the
cartridge controls — the core only does ``max``/comparison and treats the
default 0 as the smallest value. ``tag`` is opaque (typed ``Any``) and only
ever surfaces in argument ids and diagnostics; the core never inspects it.

This module imports only from within ``dialectical_games`` and the stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from dialectical_games.scheme import Tier


class Role(Enum):
    """What role one witness plays for its move (design §5).

    A FACT-tier ``OBJECTION`` / ``REPLY_ATTACK`` becomes a crisp Dung
    defeater of the move; a FACT-tier ``DEFENSE`` becomes a crisp defeater
    of whichever ``ArgumentEvidence`` it answers. HEURISTIC witnesses live
    in the graded layer with the same role-driven dispatch.
    """

    PRO = "pro"
    OBJECTION = "objection"
    REPLY_ATTACK = "reply_attack"
    DEFENSE = "defense"


@dataclass(frozen=True)
class ArgumentEvidence:
    """Typed evidence for one witness on one move (game-agnostic).

    Fields:

    * ``role`` — :class:`Role` enum: pro-reason, objection, reply attack, or
      defense. Drives every builder dispatch in :mod:`arguments` and every
      decider component in :mod:`decider`.
    * ``tier`` — :class:`Tier` enum: ``FACT`` (proven resolver / terminal)
      or ``HEURISTIC`` (positional judgement). The
      Bench-Capon-2003 fact-as-highest-value bridge: a FACT outranks any
      HEURISTIC.
    * ``magnitude`` — non-negative integer on a scale the cartridge
      controls. The core treats 0 as the smallest value (so boolean
      witnesses pass 0 / the default). Cartridges that want "this is a
      game-winning fact" use
      :data:`dialectical_games.decider._TERMINAL_LOSS_MAGNITUDE` as the
      magnitude convention; the core then naturally sorts it above any
      finite cartridge value.
    * ``answered`` — for a ``DEFENSE`` only: the exact
      :class:`ArgumentEvidence` object that this defense answers, by
      Python identity (``is``). The builders match by ``id(ev.answered)``
      to the attacker's argument id; a defense whose answered evidence is
      not on the same move's attacker set wires no edge.
    * ``tag`` — opaque cartridge identifier; the core never inspects it,
      only surfaces it in argument-id construction and diagnostics. A
      cartridge that wants stable / reproducible argument ids carries its
      own ``SupportKind`` / ``ObjectionKind`` / ``DefeaterKind`` /
      whatever in ``tag``.
    """

    role: Role
    tier: Tier
    magnitude: int = 0
    answered: "ArgumentEvidence | None" = None
    tag: Any = None
