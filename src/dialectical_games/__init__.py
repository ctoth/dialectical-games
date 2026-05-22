"""dialectical-games — game-agnostic dialectical argumentation engine core.

Extracted from ``dialectical-checkers`` and ``dialectical-chess``. A game built
on this core supplies a thin cartridge (board substrate, move generation,
witness producers, value vocabulary, search backend, protocol harness); this
package supplies the argumentation machinery.

Current scope (Phase 2 initial extraction): the foundational typed-evidence
taxonomy — :class:`~dialectical_games.scheme.Tier`,
:class:`~dialectical_games.scheme.Value`,
:class:`~dialectical_games.scheme.CriticalQuestion`,
:class:`~dialectical_games.evidence.ArgumentEvidence`, and
:func:`~dialectical_games.evidence.to_argument_evidence`.
"""

from dialectical_games.evidence import (
    ArgumentEvidence,
    to_argument_evidence,
)
from dialectical_games.scheme import CriticalQuestion, Tier, Value

__all__ = [
    "ArgumentEvidence",
    "CriticalQuestion",
    "Tier",
    "Value",
    "to_argument_evidence",
]
