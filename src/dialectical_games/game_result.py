"""The :class:`GameResult` protocol consumed by ``loss_mining``.

A game-agnostic surface: the result of one game between two players, exposing
the outcome (a cartridge-owned string), the move sequence played and every
position (`len(moves) + 1` entries — the starting position plus one after
each move). The core compares ``outcome`` only by equality against
cartridge-supplied outcome strings (e.g. ``"red"`` / ``"white"`` / ``"draw"``
for checkers).
"""

from __future__ import annotations

from typing import Protocol

from dialectical_games.board import Board, Move


class GameResult(Protocol):
    """The minimal game-result surface ``loss_mining`` needs.

    Settled per the Phase-2 continuation prompt (point 5): ``outcome`` is a
    cartridge-owned string; the core compares strings by equality, with
    ``engine_outcome`` / ``draw_outcome`` parameters passed to
    :func:`~dialectical_games.loss_mining.mine_turning_point` at call time.
    """

    @property
    def outcome(self) -> str:
        """The outcome string — one of the cartridge's "side-A wins" /
        "side-B wins" / "draw" constants."""
        ...

    @property
    def moves(self) -> tuple[Move, ...]:
        """The move sequence played."""
        ...

    @property
    def positions(self) -> tuple[Board, ...]:
        """Every board from start to terminal — ``len(moves) + 1`` entries."""
        ...
