"""Loss turning-point mining — the generic Phase-7 diagnostic.

Given a game the engine **lost**, walk the engine's plies in order asking the
cartridge's :class:`~dialectical_games.forced_loss.ForcedLossResolver` whether
the move played at each ply handed the opponent a proven forced loss. Report
the **first avoidable** conceding ply — a non-losing move was legal at that
ply and the engine did not play it. If no avoidable conceding ply exists, the
first **unavoidable** one (every legal move conceded) is reported, honestly
flagged ``was_avoidable=False`` — the position was already lost on an earlier
quiet move the resolver cannot see.

Algorithm taken unchanged from
``dialectical_checkers.loss_mining.mine_turning_point`` (Phase 7 diagnostic).
Generalised by parameterising the engine-side identifier and the
"engine won" / "draw" outcome strings (settled per the Phase-2 continuation
prompt, point 5).

This module is **evaluation tooling**; the core's runtime decider never
imports it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dialectical_games.board import Board
from dialectical_games.forced_loss import ForcedLoss, ForcedLossResolver
from dialectical_games.game_result import GameResult


@dataclass(frozen=True)
class LossTurningPoint:
    """The ply at which the engine's game turned from non-losing to losing.

    ``game_index`` identifies the game within a batch (1-based; ``1`` for a
    single game). ``ply`` is the 1-based half-move number of the blunder.
    ``fen_before`` is the position *before* the blunder move, ``played_move``
    the blunder's stable identifier (e.g. PDN for checkers, UCI for chess),
    ``side`` the engine's side identifier on that ply.
    ``shot_material_net`` is the weighted material the opponent's forced reply
    nets (the proven cost of the blunder); ``shot_wins_game`` is ``True`` iff
    that forced sequence wins the game outright. ``safe_alternatives`` lists
    the legal moves at ``fen_before`` that conceded no forced loss — moves the
    engine could have played instead.

    ``was_avoidable`` is the load-bearing honesty field. It is ``True`` iff at
    ``ply`` the engine had at least one legal move that conceded NO forced
    loss — so the loss at this ply was a genuine choice. It is ``False`` iff
    every legal move at ``ply`` conceded one (typically because all legal
    moves were mandatory captures, in checkers): the position was *already*
    lost before this ply, and this is merely the first ply where the loss
    became resolvable — the true blunder happened earlier, on a quiet move
    the resolver cannot see. The diagnostic reports this distinction rather
    than blaming a forced move.
    """

    game_index: int
    ply: int
    fen_before: str
    played_move: str
    side: str
    shot_material_net: int
    shot_wins_game: bool
    safe_alternatives: tuple[str, ...]
    was_avoidable: bool

    def describe(self) -> str:
        """A one-line human-readable description of the turning point."""
        kind = "loses the game" if self.shot_wins_game else (
            f"loses {self.shot_material_net} material"
        )
        if self.was_avoidable:
            safe = ", ".join(self.safe_alternatives)
            tail = f"avoidable; safe alternatives: {safe}"
        else:
            tail = (
                "unavoidable at this ply (every legal move concedes — the "
                "loss was locked in by an earlier quiet move)"
            )
        return (
            f"game {self.game_index} ply {self.ply} ({self.side}): "
            f"{self.played_move} {kind}; {tail}"
        )


def _engine_lost(
    result: GameResult, *, engine_outcome: str, draw_outcome: str
) -> bool:
    """True iff ``result`` is a game the engine lost.

    The engine lost iff the outcome is neither the engine's winning outcome
    nor a draw — string equality only; the outcome alphabet is the
    cartridge's.
    """
    if result.outcome == draw_outcome:
        return False
    return result.outcome != engine_outcome


def _turning_point_at(
    result: GameResult,
    ply: int,
    engine_side: str,
    game_index: int,
    loss: ForcedLoss,
    resolver: ForcedLossResolver,
) -> LossTurningPoint:
    """Build a :class:`LossTurningPoint` for a known conceding engine ``ply``."""
    board = result.positions[ply - 1]
    safe = [
        candidate.move_id()
        for candidate in board.legal_moves()
        if resolver.opponent_loss(board, candidate) is None
    ]
    return LossTurningPoint(
        game_index=game_index,
        ply=ply,
        fen_before=board.to_fen(),
        played_move=result.moves[ply - 1].move_id(),
        side=engine_side,
        shot_material_net=loss.material_net,
        shot_wins_game=loss.wins_game,
        safe_alternatives=tuple(safe),
        was_avoidable=bool(safe),
    )


def mine_turning_point(
    result: GameResult,
    resolver: ForcedLossResolver,
    *,
    engine_side: str,
    engine_outcome: str,
    draw_outcome: str,
    game_index: int = 1,
) -> LossTurningPoint | None:
    """Find the turning point of a game the engine lost.

    Walks the engine's plies of ``result`` in order, asking ``resolver`` whether
    — in the position *before* each engine move — the move handed the
    opponent a proven forced loss. Two kinds of conceding ply are
    distinguished, to point at the *cause* not the *symptom*:

    * an **avoidable** turning point — the engine conceded a forced loss but
      a legal move that conceded none was available. This is a genuine
      blunder: a non-losing move was there and the engine did not play it.
      The **first** avoidable conceding ply is the turning point.
    * an **unavoidable** conceding ply — every legal move conceded
      (typically all were mandatory captures, in checkers). The position was
      already lost; the real blunder was an earlier quiet move the resolver
      cannot see. Reported with ``was_avoidable=False``, only when *no*
      avoidable turning point exists in the game.

    Returns ``None`` when the engine did not lose, or when no engine ply
    conceded a resolvable forced loss at all (a loss from slow attrition) —
    the diagnostic never invents a turning point it did not measure.

    ``engine_side`` is the side identifier the engine plays (compared to
    ``board.turn`` for equality only). ``engine_outcome`` is the outcome
    string that means the engine WON; ``draw_outcome`` is the outcome string
    for a draw. The core never inspects the alphabet beyond these.
    """
    if not _engine_lost(
        result, engine_outcome=engine_outcome, draw_outcome=draw_outcome
    ):
        return None

    first_unavoidable: LossTurningPoint | None = None
    for ply, move in enumerate(result.moves, start=1):
        board = result.positions[ply - 1]
        if board.turn != engine_side:
            continue
        loss = resolver.opponent_loss(board, move)
        if loss is None:
            continue
        point = _turning_point_at(
            result, ply, engine_side, game_index, loss, resolver
        )
        if point.was_avoidable:
            # A genuine, avoidable blunder — the turning point we want.
            return point
        if first_unavoidable is None:
            first_unavoidable = point
    # No avoidable blunder: report the first unavoidable conceding ply (if
    # any), honestly flagged as already-lost.
    return first_unavoidable


def mine_losses(
    results: Iterable[tuple[GameResult, str]],
    resolver: ForcedLossResolver,
    *,
    engine_outcome_for_side: dict[str, str],
    draw_outcome: str,
) -> list[LossTurningPoint]:
    """Mine turning points across a batch of ``(game, engine_side)`` pairs.

    ``results`` is an iterable of ``(GameResult, engine_side)`` tuples — typically
    every game of a matchup, each tagged with the side the engine played.
    ``engine_outcome_for_side`` maps each side identifier to the outcome
    string that would mean the engine (playing that side) WON (e.g.
    ``{"r": "red", "w": "white"}``). ``draw_outcome`` is the draw outcome
    string. Returns one :class:`LossTurningPoint` per lost game in which a
    turning point was found, in game order, each tagged with its 1-based
    ``game_index``. Lost games with no resolvable turning point, and non-lost
    games, contribute nothing.
    """
    points: list[LossTurningPoint] = []
    for index, (result, engine_side) in enumerate(results, start=1):
        point = mine_turning_point(
            result,
            resolver,
            engine_side=engine_side,
            engine_outcome=engine_outcome_for_side[engine_side],
            draw_outcome=draw_outcome,
            game_index=index,
        )
        if point is not None:
            points.append(point)
    return points
