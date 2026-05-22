"""The :class:`ForcedLoss` dataclass and :class:`ForcedLossResolver` protocol.

The cartridge supplies a resolver that answers, for a position and a candidate
move, whether the move hands the opponent a proven forced loss. The core's
``loss_mining`` is the only consumer.

For checkers, the resolver wraps :func:`dialectical_checkers.captures.opponent_shot`
(forced-capture resolution). For chess, it wraps ``has_forced_mate``. Both
return the same generic :class:`ForcedLoss` summary when a loss is proven, or
``None`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dialectical_games.board import Board, Move


@dataclass(frozen=True)
class ForcedLoss:
    """A proven forced loss after a candidate move.

    Generalises checkers' ``ShotResult`` and chess's "has_forced_mate-yields-
    True" predicate. ``material_net`` is the weighted-material the opponent's
    forced reply nets (``0`` for mate-only resolvers); ``wins_game`` is
    ``True`` iff that forced sequence ends the game (mate, or shot leading to
    a terminal position).
    """

    material_net: int
    wins_game: bool


class ForcedLossResolver(Protocol):
    """Cartridge-supplied predicate: "does this move concede a proven forced
    loss?"

    The resolver is invoked on the position *before* the candidate move and
    the move itself; it is responsible for applying the move (or any
    equivalent operation) internally. The core does NOT call ``board.apply``;
    the resolver owns its own search/proof procedure.
    """

    def opponent_loss(self, board: Board, move: Move) -> ForcedLoss | None:
        """Return the proven loss summary, or ``None`` if no loss is proven."""
        ...
