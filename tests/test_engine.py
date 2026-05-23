"""Tests for the game-agnostic engine orchestrator."""

from __future__ import annotations

from dataclasses import replace

from doxa import Opinion

from dialectical_games.arguments import GradedPolicy, MoveProbe, RootArgumentGraph
from dialectical_games.engine import (
    Cartridge,
    EngineSettings,
    PostDecisionContext,
    PostDecisionResult,
    analyze,
)


class _StubPolicy:
    """A minimal :class:`GradedPolicy` for the orchestrator tests."""

    @property
    def edge_trust(self) -> Opinion:
        return Opinion.dogmatic_true(0.5)

    def move_base_rate(self, probe: MoveProbe) -> float:
        return 0.5

    def witness_opinion(
        self, *, probe: MoveProbe, label: str, magnitude: int | None
    ) -> Opinion:
        return Opinion(0.55, 0.15, 0.30, 0.5)


class _StubCartridge:
    """A minimal :class:`Cartridge` returning a fixed probe set.

    The cartridge no longer carries a selector callback — the core
    :func:`dialectical_games.decider.lexicographic_decide` is the
    decider. The cartridge supplies only the probe set and the graded
    policy.
    """

    def __init__(self, probes: tuple[MoveProbe, ...]) -> None:
        self._probes = probes

    def probe_moves(self, board: object) -> tuple[MoveProbe, ...]:
        return self._probes

    def make_graded_policy(self, board: object) -> GradedPolicy:
        return _StubPolicy()


def test_analyze_terminal_position() -> None:
    """Empty probe set -> null decision; hook NOT invoked."""
    invocations: list[object] = []

    def hook(
        ctx: PostDecisionContext,
        probes: tuple[MoveProbe, ...],
        selected: MoveProbe | None,
    ) -> PostDecisionResult:
        invocations.append((ctx, probes, selected))
        return PostDecisionResult(probes=probes, selected=selected)

    cartridge: Cartridge = _StubCartridge(probes=())
    analysis = analyze(object(), cartridge=cartridge, post_decision=hook)
    assert analysis.probes == ()
    assert analysis.decision.move_id == ""
    assert analysis.decision.selected is None
    # Hook must not have been called.
    assert invocations == []


def test_analyze_picks_core_decider_selection() -> None:
    """The orchestrator returns the core decider's choice on the probe set.

    Two clean probes with no FACT pro / objection and ``child_eval == 0``
    tie on FACT terms 1-4 and on the term-5 child_eval; the move-id
    tiebreak picks the lexicographically smaller id (``m1``).
    """
    probes = (MoveProbe(move_id="m1"), MoveProbe(move_id="m2"))
    cartridge: Cartridge = _StubCartridge(probes)
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.move_id == "m1"
    assert analysis.decision.selected is probes[0]
    assert analysis.probes == probes


def test_analyze_picks_fact_winner() -> None:
    """The core decider picks the move carrying the FACT terminal-win pro."""
    losing = MoveProbe(move_id="m1")
    winning = MoveProbe(move_id="m2", reasons=("pro:terminal_win",))
    cartridge: Cartridge = _StubCartridge((losing, winning))
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.move_id == "m2"


def test_post_decision_hook_can_replace_selection() -> None:
    """A hook returning a different ``selected`` overrides the depth-0 choice."""
    probes = (MoveProbe(move_id="m1"), MoveProbe(move_id="m2"))
    cartridge: Cartridge = _StubCartridge(probes)

    def hook(
        ctx: PostDecisionContext,
        probes: tuple[MoveProbe, ...],
        selected: MoveProbe | None,
    ) -> PostDecisionResult:
        # Override: pick m2.
        return PostDecisionResult(probes=probes, selected=probes[1])

    analysis = analyze(object(), cartridge=cartridge, post_decision=hook)
    assert analysis.decision.move_id == "m2"


def test_post_decision_hook_can_mutate_probes_and_redecide() -> None:
    """A hook can append an objection and re-run the core decider via ``redecide``."""
    probes = (
        MoveProbe(move_id="m1"),
        MoveProbe(move_id="m2"),
    )
    cartridge: Cartridge = _StubCartridge(probes)

    def hook(
        ctx: PostDecisionContext,
        probes: tuple[MoveProbe, ...],
        selected: MoveProbe | None,
    ) -> PostDecisionResult:
        # Add a FACT terminal-loss objection to m1 and re-decide. The
        # crisp layer now eliminates m1; the core decider picks m2.
        assert selected is not None
        attacked = replace(probes[0], objections=("obj:terminal_loss",))
        new_probes = (attacked, probes[1])
        new_selected = ctx.redecide(new_probes)
        return PostDecisionResult(probes=new_probes, selected=new_selected)

    analysis = analyze(object(), cartridge=cartridge, post_decision=hook)
    # m1 carries an undefeated FACT objection and is eliminated; the core
    # decider must return m2 — the only crisp survivor.
    assert analysis.decision.move_id == "m2"
    assert analysis.probes == (
        MoveProbe(move_id="m1", objections=("obj:terminal_loss",)),
        probes[1],
    )


def test_post_decision_context_carries_settings() -> None:
    probes = (MoveProbe(move_id="m1"),)
    cartridge: Cartridge = _StubCartridge(probes)
    cartridge_settings = {"foo": 42}
    settings = EngineSettings(
        deadline=1.5, cartridge_settings=cartridge_settings
    )

    captured: list[PostDecisionContext] = []

    def hook(
        ctx: PostDecisionContext,
        probes: tuple[MoveProbe, ...],
        selected: MoveProbe | None,
    ) -> PostDecisionResult:
        captured.append(ctx)
        return PostDecisionResult(probes=probes, selected=selected)

    analyze(
        object(), cartridge=cartridge, settings=settings, post_decision=hook
    )
    assert len(captured) == 1
    ctx = captured[0]
    assert ctx.deadline == 1.5
    assert ctx.cartridge_settings == cartridge_settings


def test_default_settings_no_hook() -> None:
    """``analyze`` works without explicit settings or a hook."""
    probes = (MoveProbe(move_id="m1"),)
    cartridge: Cartridge = _StubCartridge(probes)
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.move_id == "m1"


def test_move_pdn_alias_on_decision() -> None:
    probes = (MoveProbe(move_id="11-15"),)
    cartridge: Cartridge = _StubCartridge(probes)
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.move_pdn == "11-15"


def test_decision_score_reads_selected_probe_score() -> None:
    """``EngineDecision.score`` is the selected probe's ``MoveProbe.score``."""
    probes = (MoveProbe(move_id="m1", score=42),)
    cartridge: Cartridge = _StubCartridge(probes)
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.score == 42


def test_decision_score_none_for_terminal() -> None:
    cartridge: Cartridge = _StubCartridge(probes=())
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.score is None
