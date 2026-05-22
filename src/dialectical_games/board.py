"""Minimal Board / Move protocols.

Both engines satisfy this surface: ``CheckersBoard`` /
``dialectical_chess.board.OwnedBoard`` for the board, ``CheckersMove`` /
``OwnedMove`` for the move. The protocols expose ONLY what the core consumes
(today: ``loss_mining``); they are intentionally tiny so future seams can
broaden them without conflict.

Settled per the Phase-2 continuation prompt (point 4): the FEN method is named
``to_fen()`` (checkers' name). Chess's cartridge supplies an alias or renames
its ``OwnedBoard.fen()`` to match.
"""

from __future__ import annotations

from typing import Protocol


class Move(Protocol):
    """The minimal move surface ``loss_mining`` needs.

    The cartridge's move type — for checkers ``CheckersMove`` (its ``pdn()``
    method satisfies ``move_id()``); for chess ``OwnedMove`` (its ``uci()``
    method satisfies it).

    The core never inspects internal fields of a move beyond passing it to a
    cartridge-supplied :class:`~dialectical_games.forced_loss.ForcedLossResolver`
    and reading ``move_id()`` for diagnostics.
    """

    def move_id(self) -> str:
        """A stable, human-readable identifier for the move."""
        ...


class Board(Protocol):
    """The minimal board surface ``loss_mining`` needs.

    The cartridge's board type — for checkers ``CheckersBoard``, for chess
    ``OwnedBoard``. Both engines already satisfy ``turn``, ``legal_moves``,
    and a FEN serialiser; this protocol mandates the latter under the name
    ``to_fen()`` (checkers' name; settled per the Phase-2 continuation
    prompt). The core does not assume any particular semantics for ``turn``
    beyond equality comparison with the engine's side identifier.
    """

    @property
    def turn(self) -> str:
        """A side identifier (e.g. ``'r'``/``'w'`` for checkers, ``'w'``/``'b'``
        for chess). The core compares this only by equality."""
        ...

    def legal_moves(self) -> tuple[Move, ...]:
        """The legal moves from this position. The core does not inspect the
        moves beyond passing each to the cartridge's
        :class:`~dialectical_games.forced_loss.ForcedLossResolver`."""
        ...

    def to_fen(self) -> str:
        """A round-trippable position string. Recorded on
        :class:`~dialectical_games.loss_mining.LossTurningPoint` as the
        position before the blunder."""
        ...
