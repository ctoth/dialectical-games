"""Tests for the game-agnostic engine orchestrator.

Phase 5 chunk 1: probes carry typed
:class:`~dialectical_games.evidence.ArgumentEvidence`. The orchestrator
itself does not inspect evidence; these tests cover the same orchestrator
behaviours under the new probe shape.
"""

from __future__ import annotations

from dataclasses import replace

from doxa import Opinion

from dialectical_games.arguments import (
    ArgumentEvidence,
    GradedPolicy,
    MoveProbe,
)
from dialectical_games.decider import _TERMINAL_LOSS_MAGNITUDE
from dialectical_games.engine import (
    Cartridge,
    EngineSettings,
    PostDecisionContext,
    PostDecisionResult,
    analyze,
)
from dialectical_games.evidence import Role
from dialectical_games.scheme import Tier


class _StubPolicy:
    """A minimal :class:`GradedPolicy` for the orchestrator tests."""

    def with_probes(self, probes: object) -> "_StubPolicy":
        return self

    @property
    def edge_trust(self) -> Opinion:
        return Opinion.dogmatic_true(0.5)

    def move_base_rate(self, probe: MoveProbe) -> float:
        return 0.5

    def witness_opinion(
        self, *, probe: MoveProbe, evidence: ArgumentEvidence
    ) -> Opinion:
        return Opinion(0.55, 0.15, 0.30, 0.5)


class _StubCartridge:
    """A minimal :class:`Cartridge` returning a fixed probe set."""

    def __init__(self, probes: tuple[MoveProbe, ...]) -> None:
        self._probes = probes

    def probe_moves(self, board: object) -> tuple[MoveProbe, ...]:
        return self._probes

    def make_graded_policy(self, board: object) -> GradedPolicy:
        return _StubPolicy()


def _terminal_pro() -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.PRO,
        tier=Tier.FACT,
        magnitude=_TERMINAL_LOSS_MAGNITUDE,
        tag="terminal_win",
    )


def _terminal_objection() -> ArgumentEvidence:
    return ArgumentEvidence(
        role=Role.OBJECTION,
        tier=Tier.FACT,
        magnitude=_TERMINAL_LOSS_MAGNITUDE,
        tag="terminal_loss",
    )


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
    """The orchestrator returns the core decider's choice on the probe set."""
    probes = (MoveProbe(move_id="m1"), MoveProbe(move_id="m2"))
    cartridge: Cartridge = _StubCartridge(probes)
    analysis = analyze(object(), cartridge=cartridge)
    assert analysis.decision.move_id == "m1"
    assert analysis.decision.selected is probes[0]
    assert analysis.probes == probes


def test_analyze_picks_fact_winner() -> None:
    """The core decider picks the move carrying the FACT terminal-win pro."""
    losing = MoveProbe(move_id="m1")
    winning = MoveProbe(move_id="m2", evidence=(_terminal_pro(),))
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
        return PostDecisionResult(probes=probes, selected=probes[1])

    analysis = analyze(object(), cartridge=cartridge, post_decision=hook)
    assert analysis.decision.move_id == "m2"


def test_post_decision_hook_can_mutate_probes_and_redecide() -> None:
    """A hook can add an objection and re-run the core decider via ``redecide``."""
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
        attacked = replace(probes[0], evidence=(_terminal_objection(),))
        new_probes = (attacked, probes[1])
        new_selected = ctx.redecide(new_probes)
        return PostDecisionResult(probes=new_probes, selected=new_selected)

    analysis = analyze(object(), cartridge=cartridge, post_decision=hook)
    assert analysis.decision.move_id == "m2"
    assert analysis.probes[0].move_id == "m1"
    assert len(analysis.probes[0].evidence) == 1
    assert analysis.probes[0].evidence[0].role is Role.OBJECTION
    assert analysis.probes[1] == probes[1]


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
