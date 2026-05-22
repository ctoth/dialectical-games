"""AS2 scheme: ``Value`` enum, ``CriticalQuestion`` enum, ``Tier``.

Phase 0 skeleton. The Atkinson AS2 practical-reasoning scheme over a
game-AATS (design §4) gives the witness vocabulary a closed taxonomy. The
three enums below are declared per design §4; the producers that consume them
live in ``witnesses.py`` and are built in Phases 3 and 5.
"""

from __future__ import annotations

from enum import Enum


class Value(Enum):
    """Values a move can promote or demote (design §4)."""

    WINNING = "winning"
    MATERIAL = "material"
    KING_COUNT = "king_count"
    TEMPO = "tempo"
    MOBILITY = "mobility"
    STRUCTURE = "structure"


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
