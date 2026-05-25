"""AS2 scheme: ``Tier`` and ``CriticalQuestion`` (game-agnostic core).

Phase 5 chunk 1: the ``Value`` enum that previously named six game-flavoured
values (``WINNING`` / ``MATERIAL`` / ``KING_COUNT`` / ``TEMPO`` / ``MOBILITY``
/ ``STRUCTURE``) has been removed. The game-agnostic core does not name
values — cartridges encode importance via :attr:`ArgumentEvidence.magnitude`
on whatever scale the cartridge controls. The terminal-loss convention is
the single sentinel :data:`dialectical_games.decider._TERMINAL_LOSS_MAGNITUDE`
that lets cartridges express "this is a game-winning fact" magnitude-only.

What survives: :class:`Tier` (the fact-as-highest-value bridge,
Bench-Capon 2003) and :class:`CriticalQuestion` (AS2 scheme, design §4).
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Whether a witness is proven (FACT) or a positional judgement (design §4)."""

    FACT = "fact"
    HEURISTIC = "heuristic"


class CriticalQuestion(Enum):
    """The critical questions that generate objections to an AS1 argument (design §4)."""

    CQ2_3 = "cq2_3"
    CQ8_9 = "cq8_9"
    CQ17 = "cq17"
    CQ5_6_11 = "cq5_6_11"
