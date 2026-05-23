"""dialectical-games — game-agnostic dialectical argumentation engine core.

Extracted from ``dialectical-checkers`` and ``dialectical-chess``. A game built
on this core supplies a thin cartridge (board substrate, move generation,
witness producers, value vocabulary, search backend, protocol harness); this
package supplies the argumentation machinery.

Phase 2 surface:
- Foundational typed-evidence taxonomy — :class:`~dialectical_games.scheme.Tier`,
  :class:`~dialectical_games.scheme.Value`,
  :class:`~dialectical_games.scheme.CriticalQuestion`,
  :class:`~dialectical_games.evidence.ArgumentEvidence`,
  :func:`~dialectical_games.evidence.to_argument_evidence`.
- Loss-mining diagnostic — :class:`~dialectical_games.board.Board`,
  :class:`~dialectical_games.board.Move`,
  :class:`~dialectical_games.game_result.GameResult`,
  :class:`~dialectical_games.forced_loss.ForcedLoss`,
  :class:`~dialectical_games.forced_loss.ForcedLossResolver`,
  :class:`~dialectical_games.loss_mining.LossTurningPoint`,
  :func:`~dialectical_games.loss_mining.mine_turning_point`,
  :func:`~dialectical_games.loss_mining.mine_losses`.
- Crisp + graded argument graph — :class:`~dialectical_games.arguments.MoveProbe`,
  :class:`~dialectical_games.arguments.RootArgumentGraph`,
  :class:`~dialectical_games.arguments.GradedPolicy`,
  :func:`~dialectical_games.arguments.build_root_argument_graph`,
  :func:`~dialectical_games.arguments.build_graded_layer`,
  :func:`~dialectical_games.arguments.obj_arg_id`,
  :func:`~dialectical_games.arguments.reply_arg_id`.
"""

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    RootArgumentGraph,
    build_graded_layer,
    build_root_argument_graph,
    obj_arg_id,
    reply_arg_id,
)
from dialectical_games.board import Board, Move
from dialectical_games.engine import (
    Cartridge,
    EngineAnalysis,
    EngineDecision,
    EngineSettings,
    PostDecisionContext,
    PostDecisionHook,
    PostDecisionResult,
    ReDecide,
    analyze,
)
from dialectical_games.evidence import (
    ArgumentEvidence,
    to_argument_evidence,
)
from dialectical_games.forced_loss import ForcedLoss, ForcedLossResolver
from dialectical_games.game_result import GameResult
from dialectical_games.loss_mining import (
    LossTurningPoint,
    mine_losses,
    mine_turning_point,
)
from dialectical_games.scheme import CriticalQuestion, Tier, Value
from dialectical_games.search_backend import (
    SearchBackend,
    SearchBackendRegistry,
)

__all__ = [
    "ArgumentEvidence",
    "Board",
    "Cartridge",
    "CriticalQuestion",
    "EngineAnalysis",
    "EngineDecision",
    "EngineSettings",
    "ForcedLoss",
    "ForcedLossResolver",
    "GameResult",
    "GradedPolicy",
    "LossTurningPoint",
    "Move",
    "MoveProbe",
    "PostDecisionContext",
    "PostDecisionHook",
    "PostDecisionResult",
    "ReDecide",
    "RootArgumentGraph",
    "SearchBackend",
    "SearchBackendRegistry",
    "Tier",
    "Value",
    "analyze",
    "build_graded_layer",
    "build_root_argument_graph",
    "mine_losses",
    "mine_turning_point",
    "obj_arg_id",
    "reply_arg_id",
    "to_argument_evidence",
]
