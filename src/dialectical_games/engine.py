"""Game-agnostic engine orchestrator: probe -> graph -> choose (-> hook).

A thin orchestrator. The pipeline:

1. ``cartridge.probe_moves(board)`` -> the depth-0 :class:`MoveProbe` set.
2. :func:`build_root_argument_graph` (the generic crisp + graded layer)
   using ``cartridge.make_graded_policy(board)``.
3. ``cartridge.select(probes, graph, board)`` -> the depth-0 chosen probe.
4. (Optional) ``post_decision(context, probes, selected)`` -> a possibly-
   revised ``(probes, selected)``.

The orchestrator owns the wiring; every game-specific decision lives behind
the :class:`Cartridge` Protocol or the :class:`PostDecisionHook` callable.
The orchestrator does NOT call a search backend directly — backends are a
cartridge concern, invoked from the post-decision hook (Seam 3 +
:class:`SearchBackend`).

The hook MUST tolerate ``selected is None`` only by **not being called** —
the orchestrator skips the hook when no probe was chosen (a terminal
position). This is the single load-bearing invariant the hook implementations
may rely on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    RootArgumentGraph,
    build_root_argument_graph,
)


@dataclass(frozen=True)
class EngineSettings:
    """Configuration for the core orchestrator.

    Holds only the load-bearing fields the orchestrator itself reads:

    * ``selector_mode`` — the cartridge selector's mode tag. The
      cartridge interprets the value; the core forwards it as opaque.
    * ``search_backend`` — the cartridge's chosen post-decision backend
      name (the cartridge's :class:`SearchBackendRegistry` key). ``""``
      means no backend (the hook may still run on its own).
    * ``deadline`` — a monotonic-clock cap (seconds; ``None`` for
      unlimited) the orchestrator threads into the post-decision context.
    * ``cartridge_settings`` — opaque cartridge-side settings carrier
      (the cartridge's own dataclass extending whatever fields its probe
      layer / selector / hook needs). Threaded into the post-decision
      context.

    Cartridges that need richer per-engine configuration carry it on
    ``cartridge_settings`` rather than fattening this dataclass.
    """

    selector_mode: str = ""
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

    Every game-specific behaviour the depth-0 path needs lives behind one
    of these callables / methods. The orchestrator never imports a game
    module directly.
    """

    def probe_moves(self, board: Any) -> tuple[MoveProbe, ...]:
        """Produce one :class:`MoveProbe` per legal move on ``board``.

        Each probe must populate ``move_id``, the witness fields, and the
        cartridge-precomputed graded-policy inputs (``child_eval``,
        ``contested``).
        """
        ...

    def make_graded_policy(self, board: Any) -> GradedPolicy:
        """Construct the per-build :class:`GradedPolicy` bound to ``board``."""
        ...

    def select(
        self,
        probes: list[MoveProbe],
        graph: RootArgumentGraph,
        *,
        board: Any,
        settings: EngineSettings,
    ) -> MoveProbe:
        """Run the cartridge selector over the crisp survivors.

        Returns the chosen probe; must be one of ``probes`` (the
        cartridge selector ranks only the crisp survivors).
        """
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
    3. ``cartridge.select(probes, graph, board=, settings=)`` ->
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

    selected: MoveProbe | None = cartridge.select(
        list(probes), graph, board=board, settings=settings
    )

    if post_decision is not None and selected is not None:

        def _redecide(new_probes: tuple[MoveProbe, ...]) -> MoveProbe | None:
            """Re-run the cartridge selector on a (possibly mutated) probe set.

            The redecide path rebuilds the argument graph against the
            updated probe tuple so the selector sees a consistent crisp
            layer. A hook that mutates probes (e.g. appends an objection)
            invokes this to obtain the new selection.
            """
            if not new_probes:
                return None
            new_graph = build_root_argument_graph(list(new_probes), policy)
            return cartridge.select(
                list(new_probes), new_graph, board=board, settings=settings
            )

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


# Suppress unused-name warnings — ``field`` is imported for cartridge
# convenience even though this module does not consume it itself.
_ = field
