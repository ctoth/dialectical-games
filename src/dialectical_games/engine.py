"""Game-agnostic engine orchestrator: probe -> graph -> decide (-> hook).

A thin orchestrator. The pipeline:

1. ``cartridge.probe_moves(board)`` -> the depth-0 :class:`MoveProbe` set.
2. :func:`build_root_argument_graph` (the generic crisp + graded layer)
   using ``cartridge.make_graded_policy(board)``.
3. :func:`dialectical_games.decider.lexicographic_decide` -> the depth-0
   chosen probe. The single canonical lexicographic FACT-then-graded
   decider — game-agnostic, board-free, evaluator-free (it reads the
   cartridge-precomputed ``probe.child_eval`` / ``probe.contested``).
4. (Optional) ``post_decision(context, probes, selected)`` -> a possibly-
   revised ``(probes, selected)``.

The orchestrator owns the wiring; every game-specific decision lives
behind the :class:`Cartridge` Protocol or the :class:`PostDecisionHook`
callable. The decider itself is a single core function, not a cartridge
seam — Phase-2-continuation cycle 4 deleted the cartridge ``select``
callback as part of the core/cartridge boundary cleanup.

The orchestrator does NOT call a search backend directly — backends are a
cartridge concern, invoked from the post-decision hook (Seam 3 +
:class:`SearchBackend`).

The hook MUST tolerate ``selected is None`` only by **not being called** —
the orchestrator skips the hook when no probe was chosen (a terminal
position). This is the single load-bearing invariant the hook
implementations may rely on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    RootArgumentGraph,
    build_root_argument_graph,
)
from dialectical_games.decider import lexicographic_decide


@dataclass(frozen=True)
class EngineSettings:
    """Configuration for the core orchestrator.

    Holds only the load-bearing fields the orchestrator itself reads:

    * ``search_backend`` — the cartridge's chosen post-decision backend
      name (the cartridge's :class:`SearchBackendRegistry` key). ``""``
      means no backend (the hook may still run on its own).
    * ``deadline`` — a monotonic-clock cap (seconds; ``None`` for
      unlimited) the orchestrator threads into the post-decision context.
    * ``cartridge_settings`` — opaque cartridge-side settings carrier
      (the cartridge's own dataclass extending whatever fields its probe
      layer / hook needs). Threaded into the post-decision context.

    Cartridges that need richer per-engine configuration carry it on
    ``cartridge_settings`` rather than fattening this dataclass. (The
    pre-cycle-4 ``selector_mode`` field is gone — there is one canonical
    core decider, no mode dial.)
    """

    search_backend: str = ""
    deadline: float | None = None
    cartridge_settings: Any = None


@dataclass(frozen=True)
class EngineDecision:
    """The engine's chosen move and the probe it came from."""

    move_id: str
    selected: MoveProbe | None

    @property
    def move_pdn(self) -> str:
        """Backwards-compatible alias for ``move_id`` (checkers' name)."""
        return self.move_id

    @property
    def score(self) -> int | None:
        """The selected probe's cartridge-supplied integer score, or None.

        Read off :attr:`MoveProbe.score` when a probe was selected; the
        score's interpretation is cartridge-defined (typically a static
        evaluation in centipawn-scale units). ``None`` for a null
        decision (terminal position).
        """
        return None if self.selected is None else self.selected.score


@dataclass(frozen=True)
class EngineAnalysis:
    """The full per-position analysis: probes, graph, decision."""

    probes: tuple[MoveProbe, ...]
    graph: RootArgumentGraph
    decision: EngineDecision


# A redecide callable the post-decision hook may invoke to re-run the
# depth-0 selection against a (possibly mutated) probe tuple.
ReDecide = Callable[[tuple[MoveProbe, ...]], MoveProbe | None]


@dataclass(frozen=True)
class PostDecisionContext:
    """The context carrier a :class:`PostDecisionHook` receives.

    Carries the load-bearing inputs of the depth-0 path the hook may need:

    * ``board`` — the position the engine is deciding from.
    * ``deadline`` — the monotonic-clock cap threaded from
      :class:`EngineSettings`; a hook that runs lookahead respects this.
    * ``redecide`` — a callable the hook may invoke to re-run the depth-0
      selection against a (possibly mutated) probe tuple. Returns the
      newly selected probe (or ``None`` if no probes remain).
    * ``cartridge_settings`` — the cartridge's opaque settings carrier
      from :class:`EngineSettings`.
    """

    board: Any
    deadline: float | None
    redecide: ReDecide
    cartridge_settings: Any = None


@dataclass(frozen=True)
class PostDecisionResult:
    """The hook's return: possibly-revised probes and selection.

    A no-op hook returns ``PostDecisionResult(probes=probes,
    selected=selected)`` unchanged. A hook that wants to swap the
    selected probe returns a new ``selected``; a hook that mutates
    probes (e.g. appends a freshly-proven objection) returns a new
    ``probes`` tuple.
    """

    probes: tuple[MoveProbe, ...]
    selected: MoveProbe | None


PostDecisionHook = Callable[
    [PostDecisionContext, tuple[MoveProbe, ...], MoveProbe | None],
    PostDecisionResult,
]


class Cartridge(Protocol):
    """The cartridge surface the orchestrator consumes.

    The depth-0 game-specific behaviour the orchestrator needs lives
    behind these two methods — the cartridge produces probes and the
    cartridge supplies the graded policy. The orchestrator never imports
    a game module directly. The move decider itself is the core's own
    :func:`dialectical_games.decider.lexicographic_decide`; there is no
    cartridge ``select`` callback (Phase-2-continuation cycle 4 deleted
    it).
    """

    def probe_moves(self, board: Any) -> tuple[MoveProbe, ...]:
        """Produce one :class:`MoveProbe` per legal move on ``board``.

        Each probe must populate ``move_id``, the witness fields, and the
        cartridge-precomputed graded-policy inputs (``child_eval``,
        ``contested``) — the latter are what the core decider's term-5
        tiebreak reads (Phase-2 settlement 1).
        """
        ...

    def make_graded_policy(self, board: Any) -> GradedPolicy:
        """Construct the per-build :class:`GradedPolicy` bound to ``board``."""
        ...


def analyze(
    board: Any,
    *,
    cartridge: Cartridge,
    settings: EngineSettings | None = None,
    post_decision: PostDecisionHook | None = None,
) -> EngineAnalysis:
    """Run the depth-0 pipeline on ``board``, optionally with a hook.

    The pipeline is:

    1. ``cartridge.probe_moves(board)`` -> the depth-0 probes.
    2. :func:`build_root_argument_graph` with
       ``cartridge.make_graded_policy(board)``.
    3. :func:`dialectical_games.decider.lexicographic_decide` -> the
       initial chosen probe (skipped, with a null decision, on a
       terminal position).
    4. (Optional, only when a probe was selected)
       ``post_decision(context, probes, selected)`` -> revised
       ``(probes, selected)``.

    The hook is only invoked when ``selected is not None``. A hook that
    wants to skip a specific position simply returns the unchanged
    ``(probes, selected)``.
    """
    settings = settings or EngineSettings()
    probes = tuple(cartridge.probe_moves(board))
    policy = cartridge.make_graded_policy(board)
    graph = build_root_argument_graph(list(probes), policy)

    if not probes:
        return EngineAnalysis(
            probes=probes,
            graph=graph,
            decision=EngineDecision(move_id="", selected=None),
        )

    selected: MoveProbe | None = lexicographic_decide(probes, graph)

    if post_decision is not None and selected is not None:

        def _redecide(new_probes: tuple[MoveProbe, ...]) -> MoveProbe | None:
            """Re-run the core decider on a (possibly mutated) probe set.

            The redecide path rebuilds the argument graph against the
            updated probe tuple so the decider sees a consistent crisp
            layer. A hook that mutates probes (e.g. appends an objection)
            invokes this to obtain the new selection.
            """
            if not new_probes:
                return None
            new_graph = build_root_argument_graph(list(new_probes), policy)
            return lexicographic_decide(new_probes, new_graph)

        context = PostDecisionContext(
            board=board,
            deadline=settings.deadline,
            redecide=_redecide,
            cartridge_settings=settings.cartridge_settings,
        )
        result = post_decision(context, probes, selected)
        probes = result.probes
        selected = result.selected

    decision = EngineDecision(
        move_id="" if selected is None else selected.move_id,
        selected=selected,
    )
    return EngineAnalysis(probes=probes, graph=graph, decision=decision)


# Re-exports kept on the engine module for cartridge convenience.
__all__ = [
    "Cartridge",
    "EngineAnalysis",
    "EngineDecision",
    "EngineSettings",
    "PostDecisionContext",
    "PostDecisionHook",
    "PostDecisionResult",
    "ReDecide",
    "analyze",
]
