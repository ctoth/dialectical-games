"""Tests for the generic loss-mining algorithm.

The algorithm is ported unchanged from
``dialectical_checkers.loss_mining.mine_turning_point`` (Phase 7), only
generalised by parameterising the engine side identifier and outcome strings.
The cartridge-coupled tests (real CheckersBoard + opponent_shot) remain in
the checkers repo and exercise the same algorithm via the cartridge
``CheckersForcedLossResolver``; the tests here exercise the algorithm in
isolation against a hand-built ``Board`` / ``Move`` / ``GameResult`` /
``ForcedLossResolver`` substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from dialectical_games.board import Board, Move
from dialectical_games.forced_loss import ForcedLoss, ForcedLossResolver
from dialectical_games.game_result import GameResult
from dialectical_games.loss_mining import (
    LossTurningPoint,
    mine_losses,
    mine_turning_point,
)


# --- Test substrate ---------------------------------------------------------


@dataclass(frozen=True)
class _Move:
    """Tiny ``Move`` implementation: a stable ``move_id`` and nothing else."""

    move_id_str: str

    def move_id(self) -> str:
        return self.move_id_str


@dataclass(frozen=True)
class _Board:
    """Tiny ``Board`` implementation: explicit turn, legal-moves, fen string."""

    turn_: str
    legal_: tuple[_Move, ...]
    fen_: str

    @property
    def turn(self) -> str:
        return self.turn_

    def legal_moves(self) -> tuple[_Move, ...]:
        return self.legal_

    def to_fen(self) -> str:
        return self.fen_


@dataclass(frozen=True)
class _Result:
    """Tiny ``GameResult`` implementation."""

    outcome: str
    moves: tuple[_Move, ...]
    positions: tuple[_Board, ...]


@dataclass
class _Resolver:
    """A resolver driven by a hand-supplied ``(board_fen, move_id) -> ForcedLoss``
    table."""

    losses: dict[tuple[str, str], ForcedLoss] = field(default_factory=dict)
    seen: list[tuple[str, str]] = field(default_factory=list)

    def opponent_loss(self, board: Board, move: Move) -> ForcedLoss | None:
        key = (board.to_fen(), move.move_id())
        self.seen.append(key)
        return self.losses.get(key)


# --- engine-lost gate -------------------------------------------------------


@pytest.mark.unit
def test_returns_none_when_engine_won() -> None:
    """The engine winning yields no turning point — equality-only outcome check."""
    result = _Result(
        outcome="red",
        moves=(_Move("a"),),
        positions=(
            _Board("r", (_Move("a"),), "fen0"),
            _Board("w", (), "fen1"),
        ),
    )
    assert (
        mine_turning_point(
            result,
            _Resolver(),
            engine_side="r",
            engine_outcome="red",
            draw_outcome="draw",
        )
        is None
    )


@pytest.mark.unit
def test_returns_none_when_draw() -> None:
    """A drawn game yields no turning point."""
    result = _Result(
        outcome="draw",
        moves=(_Move("a"),),
        positions=(
            _Board("r", (_Move("a"),), "fen0"),
            _Board("w", (), "fen1"),
        ),
    )
    assert (
        mine_turning_point(
            result,
            _Resolver(),
            engine_side="r",
            engine_outcome="red",
            draw_outcome="draw",
        )
        is None
    )


# --- avoidable turning point -----------------------------------------------


@pytest.mark.unit
def test_first_avoidable_blunder_is_the_turning_point() -> None:
    """First conceding engine ply with a safe alternative is the turning point."""
    blunder = _Move("blunder")
    safe = _Move("safe")
    start = _Board("r", (blunder, safe), "fen0")
    mid = _Board("w", (), "fen1")
    result = _Result(outcome="white", moves=(blunder,), positions=(start, mid))
    resolver = _Resolver(
        losses={("fen0", "blunder"): ForcedLoss(material_net=2, wins_game=False)}
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="r",
        engine_outcome="red",
        draw_outcome="draw",
    )

    assert point is not None
    assert isinstance(point, LossTurningPoint)
    assert point.ply == 1
    assert point.played_move == "blunder"
    assert point.side == "r"
    assert point.fen_before == "fen0"
    assert point.was_avoidable is True
    assert point.safe_alternatives == ("safe",)
    assert point.shot_material_net == 2
    assert point.shot_wins_game is False


# --- unavoidable conceding ply ---------------------------------------------


@pytest.mark.unit
def test_unavoidable_concede_is_flagged() -> None:
    """A ply with no safe alternative is reported, flagged ``was_avoidable=False``."""
    only = _Move("only")
    start = _Board("r", (only,), "fen0")
    mid = _Board("w", (), "fen1")
    result = _Result(outcome="white", moves=(only,), positions=(start, mid))
    resolver = _Resolver(
        losses={("fen0", "only"): ForcedLoss(material_net=1, wins_game=True)}
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="r",
        engine_outcome="red",
        draw_outcome="draw",
    )

    assert point is not None
    assert point.ply == 1
    assert point.was_avoidable is False
    assert point.safe_alternatives == ()
    assert point.shot_wins_game is True


@pytest.mark.unit
def test_avoidable_preferred_over_earlier_unavoidable() -> None:
    """An earlier unavoidable ply yields to a later avoidable blunder.

    Engine plies 1 and 3. Ply 1: only one legal move, it concedes
    (unavoidable). Ply 3: two legal moves, the played one concedes
    (avoidable). The avoidable ply 3 is the turning point per the
    cause-not-symptom rule.
    """
    forced = _Move("forced")
    opp = _Move("opp")
    blunder = _Move("blunder")
    safe = _Move("safe")

    p0 = _Board("r", (forced,), "p0")
    p1 = _Board("w", (opp,), "p1")
    p2 = _Board("r", (blunder, safe), "p2")
    p3 = _Board("w", (), "p3")
    result = _Result(
        outcome="white",
        moves=(forced, opp, blunder),
        positions=(p0, p1, p2, p3),
    )
    resolver = _Resolver(
        losses={
            ("p0", "forced"): ForcedLoss(material_net=1, wins_game=False),
            ("p2", "blunder"): ForcedLoss(material_net=3, wins_game=False),
        }
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="r",
        engine_outcome="red",
        draw_outcome="draw",
    )

    assert point is not None
    assert point.ply == 3
    assert point.played_move == "blunder"
    assert point.was_avoidable is True


@pytest.mark.unit
def test_first_unavoidable_reported_when_no_avoidable_exists() -> None:
    """Without an avoidable blunder the FIRST unavoidable ply is reported."""
    forced0 = _Move("forced0")
    opp = _Move("opp")
    forced2 = _Move("forced2")

    p0 = _Board("r", (forced0,), "p0")
    p1 = _Board("w", (opp,), "p1")
    p2 = _Board("r", (forced2,), "p2")
    p3 = _Board("w", (), "p3")
    result = _Result(
        outcome="white",
        moves=(forced0, opp, forced2),
        positions=(p0, p1, p2, p3),
    )
    resolver = _Resolver(
        losses={
            ("p0", "forced0"): ForcedLoss(material_net=1, wins_game=False),
            ("p2", "forced2"): ForcedLoss(material_net=2, wins_game=False),
        }
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="r",
        engine_outcome="red",
        draw_outcome="draw",
    )

    assert point is not None
    assert point.ply == 1
    assert point.played_move == "forced0"
    assert point.was_avoidable is False


# --- engine side filtering --------------------------------------------------


@pytest.mark.unit
def test_only_engine_plies_inspected() -> None:
    """Opponent plies are never inspected; their conceding moves never count."""
    opp = _Move("opp_blunder")
    eng_safe = _Move("eng_safe")
    p0 = _Board("w", (opp,), "p0")
    p1 = _Board("r", (eng_safe,), "p1")
    p2 = _Board("w", (), "p2")
    result = _Result(outcome="white", moves=(opp, eng_safe), positions=(p0, p1, p2))
    resolver = _Resolver(
        losses={("p0", "opp_blunder"): ForcedLoss(material_net=99, wins_game=True)}
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="r",
        engine_outcome="red",
        draw_outcome="draw",
    )

    # The opponent's conceding move is irrelevant; the engine never blundered.
    assert point is None


# --- engine side parametric -------------------------------------------------


@pytest.mark.unit
def test_engine_side_white() -> None:
    """The diagnostic works when the engine is the second-to-move side."""
    blunder = _Move("w_blunder")
    safe = _Move("w_safe")
    p0 = _Board("w", (blunder, safe), "p0")
    p1 = _Board("r", (), "p1")
    result = _Result(outcome="red", moves=(blunder,), positions=(p0, p1))
    resolver = _Resolver(
        losses={("p0", "w_blunder"): ForcedLoss(material_net=2, wins_game=False)}
    )

    point = mine_turning_point(
        result,
        resolver,
        engine_side="w",
        engine_outcome="white",
        draw_outcome="draw",
    )

    assert point is not None
    assert point.side == "w"
    assert point.played_move == "w_blunder"


# --- mine_losses ------------------------------------------------------------


@pytest.mark.unit
def test_mine_losses_tags_game_indices() -> None:
    """A batch returns one turning point per lost game; non-losses contribute nothing."""

    def lost_game() -> _Result:
        blunder = _Move("b")
        p0 = _Board("r", (blunder, _Move("s")), "p0")
        p1 = _Board("w", (), "p1")
        return _Result(outcome="white", moves=(blunder,), positions=(p0, p1))

    def won_game() -> _Result:
        only = _Move("ok")
        p0 = _Board("r", (only,), "won_p0")
        p1 = _Board("w", (), "won_p1")
        return _Result(outcome="red", moves=(only,), positions=(p0, p1))

    resolver = _Resolver(
        losses={("p0", "b"): ForcedLoss(material_net=4, wins_game=False)}
    )

    points = mine_losses(
        [(lost_game(), "r"), (won_game(), "r"), (lost_game(), "r")],
        resolver,
        engine_outcome_for_side={"r": "red", "w": "white"},
        draw_outcome="draw",
    )

    assert [p.game_index for p in points] == [1, 3]
    assert all(p.played_move == "b" for p in points)
