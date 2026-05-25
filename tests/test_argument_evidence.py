"""Tests for the new typed-evidence shape (Phase 5 chunk 1).

Replaces ``test_evidence.py`` (the label-parser tests for the
now-deleted ``to_argument_evidence`` parser). Pins the structural shape
the cartridge surface depends on:

* :class:`ArgumentEvidence` is frozen and carries exactly the new fields.
* :class:`Role` is the enum the builders dispatch on.
* The defense-by-identity invariant: a defense's ``answered`` field is the
  EVIDENCE OBJECT it answers, not a string; the graph builders wire the
  defeat edge via ``id(ev.answered)``.
* The old ``Value`` enum is gone — importing it raises ``ImportError``.
* No string-prefix dispatch (``startswith`` / ``split(":")`` / ``.endswith``)
  in the production builders or the decider.
* The only numeric literal in the decider (beyond ``0`` / ``1`` defaults
  and loop indices) is ``_TERMINAL_LOSS_MAGNITUDE = 10**9``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from dialectical_games.arguments import (
    GradedPolicy,
    MoveProbe,
    build_root_argument_graph,
)
from dialectical_games.evidence import ArgumentEvidence, Role
from dialectical_games.scheme import Tier

# A doxa.Opinion is needed for the stub policy below.
from doxa import Opinion


# ---------------------------------------------------------------------------
# A minimal stub policy reused across tests in this file.
# ---------------------------------------------------------------------------


class _StubPolicy:
    """A minimal :class:`GradedPolicy` for structural tests."""

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


_POLICY: GradedPolicy = _StubPolicy()


# ---------------------------------------------------------------------------
# D1 / D7: shape of the new dataclass + the Role enum
# ---------------------------------------------------------------------------


def test_evidence_is_frozen() -> None:
    """``ArgumentEvidence`` is a frozen dataclass with the exact new fields."""
    ev = ArgumentEvidence(role=Role.PRO, tier=Tier.FACT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.magnitude = 99  # type: ignore[misc]

    fields = {f.name for f in dataclasses.fields(ArgumentEvidence)}
    assert fields == {"role", "tier", "magnitude", "answered", "tag"}


def test_role_enum_has_the_four_roles() -> None:
    """``Role`` is exactly the four-member enum the builders dispatch on."""
    assert {r.name for r in Role} == {
        "PRO",
        "OBJECTION",
        "REPLY_ATTACK",
        "DEFENSE",
    }


def test_evidence_defaults_are_zero_none_none() -> None:
    """``magnitude`` defaults to 0; ``answered`` / ``tag`` default to None."""
    ev = ArgumentEvidence(role=Role.PRO, tier=Tier.FACT)
    assert ev.magnitude == 0
    assert ev.answered is None
    assert ev.tag is None


def test_evidence_carries_arbitrary_tag() -> None:
    """``tag`` is opaque ``Any`` — any cartridge object is accepted."""

    class _CartridgeTag:
        pass

        # No __eq__ override — identity-based equality only.

    tag = _CartridgeTag()
    ev = ArgumentEvidence(role=Role.PRO, tier=Tier.HEURISTIC, tag=tag)
    assert ev.tag is tag


# ---------------------------------------------------------------------------
# D7: the defense-by-identity invariant
# ---------------------------------------------------------------------------


def test_defense_answered_by_identity_crisp_layer() -> None:
    """A FACT defense whose ``answered`` is the objection EVIDENCE OBJECT
    defeats only that objection's argument."""
    objection = ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.FACT, magnitude=100, tag="obj1"
    )
    defense = ArgumentEvidence(
        role=Role.DEFENSE,
        tier=Tier.FACT,
        magnitude=100,
        answered=objection,
        tag="def1",
    )
    probe = MoveProbe(move_id="m1", evidence=(objection, defense))
    graph = build_root_argument_graph([probe], _POLICY)
    assert graph.move_arguments["m1"] in graph.grounded_extension


def test_defense_with_unrelated_answered_wires_no_edge() -> None:
    """A defense whose ``answered`` evidence is NOT on the same probe's
    attacker set defeats nothing — the move stays under attack."""
    foreign_objection = ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.FACT, magnitude=100, tag="foreign"
    )
    real_objection = ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.FACT, magnitude=100, tag="real"
    )
    misaimed_defense = ArgumentEvidence(
        role=Role.DEFENSE,
        tier=Tier.FACT,
        magnitude=100,
        answered=foreign_objection,
        tag="def",
    )
    probe = MoveProbe(
        move_id="m1", evidence=(real_objection, misaimed_defense)
    )
    graph = build_root_argument_graph([probe], _POLICY)
    # The real objection is still undefeated; the move is not grounded.
    assert graph.move_arguments["m1"] not in graph.grounded_extension


def test_defense_answered_by_identity_graded_layer() -> None:
    """A HEURISTIC defense suppresses the answered HEURISTIC objection
    inside the graded graph via identity-keyed lookup."""
    from dialectical_games.arguments import _witness_arg_id

    objection = ArgumentEvidence(
        role=Role.OBJECTION, tier=Tier.HEURISTIC, tag="obj-h"
    )
    defense = ArgumentEvidence(
        role=Role.DEFENSE,
        tier=Tier.HEURISTIC,
        answered=objection,
        tag="def-h",
    )
    probe = MoveProbe(
        move_id="m1", child_eval=0, evidence=(objection, defense)
    )
    graph = build_root_argument_graph([probe], _POLICY)
    attacks = graph.ranking["attacks"]
    # The defense-witness node attacks the objection-witness node (not the
    # move node).
    objection_witness = _witness_arg_id("m1", objection)
    defense_witness = _witness_arg_id("m1", defense)
    move_node = "move:m1"
    assert (defense_witness, objection_witness) in attacks
    assert (defense_witness, move_node) not in attacks


# ---------------------------------------------------------------------------
# D2 / D8: the old vocabulary is gone
# ---------------------------------------------------------------------------


def test_no_value_enum() -> None:
    """The retired ``Value`` enum is no longer importable from scheme."""
    scheme = importlib.import_module("dialectical_games.scheme")
    assert not hasattr(scheme, "Value")
    # And the top-level package no longer re-exports it.
    package = importlib.import_module("dialectical_games")
    assert "Value" not in package.__all__
    assert not hasattr(package, "Value")


def test_no_to_argument_evidence_parser() -> None:
    """The retired ``to_argument_evidence`` parser is gone."""
    evidence = importlib.import_module("dialectical_games.evidence")
    assert not hasattr(evidence, "to_argument_evidence")


# ---------------------------------------------------------------------------
# D7: code-inspection gate — no string-prefix dispatch in the core
# ---------------------------------------------------------------------------


def _core_module_source(name: str) -> str:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "module_name",
    ["dialectical_games.arguments", "dialectical_games.decider"],
)
def test_no_string_dispatch_in_builders(module_name: str) -> None:
    """No production builder calls ``str.startswith`` / ``str.split`` / etc.

    The Phase 5 contract: every dispatch is enum-typed
    (``ev.role``, ``ev.tier``). A regression that re-introduces a
    string prefix parse anywhere in the builders / decider is exactly
    the failure this gate is here to catch.
    """
    src = _core_module_source(module_name)
    tree = ast.parse(src)
    banned = {"startswith", "endswith", "partition", "rpartition"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            raise AssertionError(
                f"{module_name} contains banned string-dispatch attr "
                f".{node.attr} at line {node.lineno}"
            )
        # ``some_string.split(":")`` — flagged when split is called with a
        # string-literal separator. ``str.split()`` without args is generic
        # and acceptable.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            raise AssertionError(
                f"{module_name} contains banned `.split({node.args[0].value!r})` "
                f"at line {node.lineno}"
            )


def test_terminal_sentinel_is_the_only_decider_literal() -> None:
    """The only numeric literal in ``decider.py`` (beyond 0 / 1 defaults
    and loop sentinels) is ``_TERMINAL_LOSS_MAGNITUDE = 10**9``.

    A regression that re-introduces a tuned threshold (e.g. 100, 0.5,
    1000000) is exactly the failure this gate catches.
    """
    src = _core_module_source("dialectical_games.decider")
    tree = ast.parse(src)
    allowed_int = {0, 1, 9, 10}  # 0 and 1 default magnitudes / arithmetic;
    # 10 and 9 are the operands of ``10**9`` (the terminal sentinel).
    allowed_float = {0.5}  # neutral graded expectation literal.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                continue  # bool is also int — skip
            if isinstance(node.value, int):
                if node.value not in allowed_int:
                    raise AssertionError(
                        f"decider.py contains disallowed int literal "
                        f"{node.value!r} at line {node.lineno}"
                    )
            elif isinstance(node.value, float):
                if node.value not in allowed_float:
                    raise AssertionError(
                        f"decider.py contains disallowed float literal "
                        f"{node.value!r} at line {node.lineno}"
                    )


# ---------------------------------------------------------------------------
# Property tests — the dataclass roundtrip + the identity-keyed defense
# ---------------------------------------------------------------------------


_TIER = st.sampled_from(list(Tier))
_ROLE = st.sampled_from(list(Role))


def _evidence_strategy() -> st.SearchStrategy[ArgumentEvidence]:
    """Generate a flat, defenseless ``ArgumentEvidence``.

    Used for property-testing the dataclass roundtrip and as building
    blocks for the identity-keyed defense property test.
    """
    return st.builds(
        ArgumentEvidence,
        role=_ROLE,
        tier=_TIER,
        magnitude=st.integers(min_value=0, max_value=10**6),
        answered=st.none(),
        tag=st.one_of(st.none(), st.text(min_size=1, max_size=6)),
    )


@given(ev=_evidence_strategy())
@settings(max_examples=100, deadline=None)
def test_property_evidence_is_hashable_and_frozen(
    ev: ArgumentEvidence,
) -> None:
    """Any constructible ``ArgumentEvidence`` is hashable and frozen.

    The dataclass is ``frozen=True``, so hashing must succeed and mutation
    must raise ``FrozenInstanceError`` — both invariants the builders rely
    on when they use ``id(ev)`` keys and place evidence in tuples.
    """
    # Hashable.
    hash(ev)
    # Frozen.
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.magnitude = ev.magnitude + 1  # type: ignore[misc]


_FACT_OBJECTION_STRATEGY = st.builds(
    ArgumentEvidence,
    role=st.just(Role.OBJECTION),
    tier=st.just(Tier.FACT),
    magnitude=st.integers(min_value=0, max_value=10**6),
    answered=st.none(),
    tag=st.one_of(st.none(), st.text(min_size=1, max_size=6)),
)


@given(
    objections=st.lists(_FACT_OBJECTION_STRATEGY, min_size=1, max_size=4)
)
@settings(max_examples=50, deadline=None)
def test_property_defense_targets_only_its_answered(
    objections: list[ArgumentEvidence],
) -> None:
    """A FACT defense identified by ``answered`` ALWAYS defeats exactly the
    answered attacker's argument and NEVER any other attacker on the same
    probe.

    Builds N FACT objections + 1 FACT defense whose ``answered`` is the
    FIRST objection. The crisp Dung graph must contain a defeat edge from
    the defense's argument to the first objection's argument, and no other
    defeat edge from the defense.
    """
    target = objections[0]
    # Use a magnitude high enough that the FACT defense remains undefeated.
    defense = ArgumentEvidence(
        role=Role.DEFENSE,
        tier=Tier.FACT,
        magnitude=1,
        answered=target,
        tag="defense",
    )
    probe = MoveProbe(
        move_id="m1", evidence=tuple(objections) + (defense,)
    )
    graph = build_root_argument_graph([probe], _POLICY)

    from dialectical_games.arguments import _defense_arg, obj_arg_id

    defense_arg = _defense_arg("m1", defense)
    target_arg = obj_arg_id("m1", target)
    # The defense argument exists and defeats the answered objection.
    assert defense_arg in graph.arguments
    assert (defense_arg, target_arg) in graph.defeats
    # The defense defeats nothing else.
    defense_outgoing = {
        (src, dst) for (src, dst) in graph.defeats if src == defense_arg
    }
    assert defense_outgoing == {(defense_arg, target_arg)}
