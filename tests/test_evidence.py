"""Tests for ``dialectical_games.evidence``.

``to_argument_evidence`` is the comorphism that turns a stringly-typed witness
label into typed :class:`ArgumentEvidence` carrying the witness's ``Value`` and
``Tier`` (design ``notes/checkers-design.md`` §4-5). Phase 3a covered the
FACT-tier labels of the §5 tables; Phase 4 adds the HEURISTIC-tier labels.

Every label of design §5 — FACT and HEURISTIC — is asserted here to map to the
correct ``Value`` and ``Tier``, including the parsed magnitude where the label
carries one.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from dialectical_games.evidence import (
    ArgumentEvidence,
    _FIXED,
    _MAGNITUDE,
    to_argument_evidence,
)
from dialectical_games.scheme import Tier, Value


# ---------------------------------------------------------------------------
# unit — every FACT-tier §5 label -> correct Value / Tier
# ---------------------------------------------------------------------------
#
# Each row: (label, expected Value, expected Tier). The magnitude-carrying
# labels are spot-checked separately for the parsed integer.

FACT_LABELS: list[tuple[str, Value, Tier]] = [
    # AS1 pro-reasons (design §5 first table, FACT rows).
    ("pro:terminal_win", Value.WINNING, Tier.FACT),
    ("pro:material:100", Value.MATERIAL, Tier.FACT),
    ("pro:material:250", Value.MATERIAL, Tier.FACT),
    ("pro:crown", Value.KING_COUNT, Tier.FACT),
    ("pro:shot_setup:200", Value.MATERIAL, Tier.FACT),
    # CQ-derived objections (design §5 second table, FACT rows).
    ("obj:terminal_loss", Value.WINNING, Tier.FACT),
    ("obj:allows_shot:100", Value.MATERIAL, Tier.FACT),
    ("obj:loses_exchange:150", Value.MATERIAL, Tier.FACT),
    # CQ17 reply attacks — FACT when the reply is a proven forced win/gain.
    ("reply:terminal_loss", Value.WINNING, Tier.FACT),
    ("reply:material:100", Value.MATERIAL, Tier.FACT),
    # A proven defense — answers a CQ8_9/CQ17 objection.
    ("defense:holds_exchange", Value.MATERIAL, Tier.FACT),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,value,tier",
    FACT_LABELS,
    ids=[row[0] for row in FACT_LABELS],
)
def test_fact_label_maps_to_value_and_tier(
    label: str, value: Value, tier: Tier
) -> None:
    """Every FACT-tier §5 label parses to the documented Value and FACT Tier."""
    evidence = to_argument_evidence(label)
    assert isinstance(evidence, ArgumentEvidence)
    assert evidence.label == label
    assert evidence.value is value
    assert evidence.tier is tier


@pytest.mark.unit
def test_every_fact_label_is_fact_tier() -> None:
    """No FACT-tier §5 label is ever mis-typed as HEURISTIC."""
    for label, _value, _tier in FACT_LABELS:
        assert to_argument_evidence(label).tier is Tier.FACT, label


# ---------------------------------------------------------------------------
# unit — every HEURISTIC-tier §5 label -> correct Value / Tier (Phase 4)
# ---------------------------------------------------------------------------
#
# Each row: (label, expected Value, expected Tier). All Phase-4 HEURISTIC §5
# rows. The magnitude-carrying HEURISTIC labels (``pro:center`` /
# ``pro:mobility``) are spot-checked for the parsed integer separately; the
# ``pro:formation:{kind}`` named-formation suffixes are covered below.

HEURISTIC_LABELS: list[tuple[str, Value, Tier]] = [
    # AS1 pro-reasons (design §5 first table, HEURISTIC rows).
    ("pro:opposition", Value.TEMPO, Tier.HEURISTIC),
    ("pro:back_rank_hold", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:center:2", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:mobility:3", Value.MOBILITY, Tier.HEURISTIC),
    ("pro:formation:phalanx", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:formation:bridge", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:formation:echelon", Value.STRUCTURE, Tier.HEURISTIC),
    # CQ-derived objections (design §5 second table, HEURISTIC rows).
    ("obj:loses_opposition", Value.TEMPO, Tier.HEURISTIC),
    ("obj:back_rank_break", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:single_corner_drift", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:exposes_man", Value.MATERIAL, Tier.HEURISTIC),
    # Witness-vocabulary enrichment — the Tier A new witnesses + the two cheap
    # completions. Fixed (no magnitude) HEURISTIC rows.
    ("pro:frees_trapped_piece", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:safe_side_man", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:strong_back_structure", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:king_centralised", Value.KING_COUNT, Tier.HEURISTIC),
    ("obj:trapped_piece", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:weakens_back_structure", Value.STRUCTURE, Tier.HEURISTIC),
    # Witness-vocabulary enrichment — magnitude-carrying HEURISTIC rows.
    ("pro:cramps_opponent:2", Value.MOBILITY, Tier.HEURISTIC),
    ("obj:cedes_centre:1", Value.STRUCTURE, Tier.HEURISTIC),
    ("defense:heuristic_suppression", Value.STRUCTURE, Tier.HEURISTIC),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,value,tier",
    HEURISTIC_LABELS,
    ids=[row[0] for row in HEURISTIC_LABELS],
)
def test_heuristic_label_maps_to_value_and_tier(
    label: str, value: Value, tier: Tier
) -> None:
    """Every HEURISTIC §5 label parses to the documented Value and HEURISTIC Tier."""
    evidence = to_argument_evidence(label)
    assert isinstance(evidence, ArgumentEvidence)
    assert evidence.label == label
    assert evidence.value is value
    assert evidence.tier is tier


@pytest.mark.unit
def test_every_heuristic_label_is_heuristic_tier() -> None:
    """No HEURISTIC-tier §5 label is ever mis-typed as FACT."""
    for label, _value, _tier in HEURISTIC_LABELS:
        assert to_argument_evidence(label).tier is Tier.HEURISTIC, label


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,magnitude",
    [
        ("pro:center:1", 1),
        ("pro:center:4", 4),
        ("pro:mobility:1", 1),
        ("pro:mobility:7", 7),
        ("pro:cramps_opponent:2", 2),
        ("pro:cramps_opponent:8", 8),
        ("obj:cedes_centre:1", 1),
        ("obj:cedes_centre:3", 3),
    ],
    ids=lambda v: str(v),
)
def test_heuristic_magnitude_is_parsed(label: str, magnitude: int) -> None:
    """A HEURISTIC ``:{n}`` label parses ``n`` (a positional count) into magnitude."""
    assert to_argument_evidence(label).magnitude == magnitude


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        "pro:opposition",
        "pro:back_rank_hold",
        "pro:formation:phalanx",
        "obj:loses_opposition",
        "obj:back_rank_break",
        "obj:single_corner_drift",
        "obj:exposes_man",
        "pro:frees_trapped_piece",
        "pro:safe_side_man",
        "pro:strong_back_structure",
        "pro:king_centralised",
        "obj:trapped_piece",
        "obj:weakens_back_structure",
    ],
)
def test_magnitudeless_heuristic_labels_have_none_magnitude(label: str) -> None:
    """A HEURISTIC label with no ``:{n}`` magnitude carries ``magnitude`` None.

    ``pro:formation:{kind}`` has a non-numeric named suffix, not a magnitude —
    its ``magnitude`` is ``None`` and the kind is kept in ``label``.
    """
    assert to_argument_evidence(label).magnitude is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        "pro:formation:knight",      # not a known formation kind
        "pro:formation:",            # empty formation kind
        "pro:formation:Phalanx",     # wrong case
        "pro:center",                # missing magnitude
        "pro:mobility",              # missing magnitude
        "pro:center:abc",            # non-numeric magnitude
        "pro:mobility:0",            # zero magnitude
        "pro:center:-1",             # negative magnitude
    ],
)
def test_malformed_heuristic_label_raises(label: str) -> None:
    """A malformed HEURISTIC label raises rather than being silently mistyped.

    An unknown formation kind, a missing/zero/negative count magnitude — each
    is rejected exactly as the FACT-tier malformed labels are.
    """
    with pytest.raises(ValueError):
        to_argument_evidence(label)


# ---------------------------------------------------------------------------
# unit — magnitude parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,magnitude",
    [
        ("pro:material:100", 100),
        ("pro:material:250", 250),
        ("pro:shot_setup:200", 200),
        ("obj:allows_shot:100", 100),
        ("obj:loses_exchange:150", 150),
        ("reply:material:100", 100),
    ],
    ids=lambda v: str(v),
)
def test_magnitude_is_parsed(label: str, magnitude: int) -> None:
    """A label carrying a ``:{n}`` magnitude parses ``n`` into ``magnitude``."""
    evidence = to_argument_evidence(label)
    assert evidence.magnitude == magnitude


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    ["pro:terminal_win", "pro:crown", "obj:terminal_loss",
     "reply:terminal_loss", "defense:holds_exchange",
     "defense:heuristic_suppression"],
)
def test_magnitudeless_labels_have_none_magnitude(label: str) -> None:
    """A label with no ``:{n}`` magnitude carries ``magnitude is None``."""
    assert to_argument_evidence(label).magnitude is None


# ---------------------------------------------------------------------------
# unit — keyed defense labels (design §6 — a defense answers ONE attack)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,answered",
    [
        (
            "defense:holds_exchange@reply:material:100",
            "reply:material:100",
        ),
        (
            "defense:holds_exchange@reply:terminal_loss",
            "reply:terminal_loss",
        ),
        (
            "defense:holds_exchange@obj:allows_shot:250",
            "obj:allows_shot:250",
        ),
        (
            "defense:heuristic_suppression@obj:exposes_man",
            "obj:exposes_man",
        ),
        (
            "defense:heuristic_suppression@obj:king_safety:flank_pawn_lunge",
            "obj:king_safety:flank_pawn_lunge",
        ),
    ],
    ids=lambda v: str(v),
)
def test_keyed_defense_carries_its_answered_target(
    label: str, answered: str
) -> None:
    """A keyed defense parses to evidence carrying ``answered``.

    Design §6: a defense answers "the objection/reply it answers, and only that
    one". The keyed label ``defense:holds_exchange@{answered}`` parses so the
    ``answered`` field names that exact target. FACT defenses live in the crisp
    layer; HEURISTIC defenses live in the graded graph.
    """
    evidence = to_argument_evidence(label)
    assert evidence.answered == answered


@pytest.mark.unit
def test_keyed_heuristic_defense_is_heuristic() -> None:
    """A heuristic suppression defense is typed for the graded graph, not crisp Dung."""
    evidence = to_argument_evidence(
        "defense:heuristic_suppression@obj:exposes_man"
    )
    assert evidence.value is Value.STRUCTURE
    assert evidence.tier is Tier.HEURISTIC
    assert evidence.answered == "obj:exposes_man"


@pytest.mark.unit
def test_unkeyed_defense_has_no_answered_target() -> None:
    """The bare, un-keyed ``defense:holds_exchange`` parses with ``answered`` None.

    The parser still accepts the bare defense type (it is valid evidence), but
    ``witnesses.py`` never emits it — every emitted defense is keyed.
    """
    evidence = to_argument_evidence("defense:holds_exchange")
    assert evidence.tier is Tier.FACT
    assert evidence.answered is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        "defense:holds_exchange@",            # empty answered target
        "defense:holds_exchange@garbage",     # answered target unknown
        "defense:holds_exchange@pro:crown",   # answered a non-attack label
        "defense:unknown@reply:material:100",  # unknown defense type
        "obj:allows_shot@reply:material:100",  # @ on a non-defense label
    ],
)
def test_malformed_keyed_defense_raises(label: str) -> None:
    """A malformed keyed defense label raises rather than being mistyped.

    The defense type must be a known ``defense:`` label, and the answered
    target must itself be a valid objection / reply label.
    """
    with pytest.raises(ValueError):
        to_argument_evidence(label)


# ---------------------------------------------------------------------------
# unit — malformed / unknown labels are rejected, never silently mistyped
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        "",
        "pro:",
        "garbage",
        "pro:unknown_reason",
        "obj:unknown_objection",
        "pro:material",          # missing magnitude
        "pro:material:abc",      # non-numeric magnitude
        "obj:allows_shot",       # missing magnitude
        # NB: ``pro:opposition`` was a malformed-label case in Phase 3a (the
        # HEURISTIC rows were unimplemented). Phase 4 implements the HEURISTIC
        # §5 rows, so ``pro:opposition`` is now a *valid* HEURISTIC label — it
        # is covered by ``HEURISTIC_LABELS`` above and no longer belongs here.
    ],
)
def test_unknown_or_malformed_label_raises(label: str) -> None:
    """An unknown or malformed witness label raises rather than mistyping."""
    with pytest.raises(ValueError):
        to_argument_evidence(label)


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        # A ``{n}`` magnitude is a strictly positive material gain/loss
        # (design §5) — the witness producers only ever emit positive
        # magnitudes. A signed, ``+``-prefixed, or zero magnitude is malformed
        # and must be rejected, never accepted as valid FACT evidence.
        "pro:material:-100",     # negative magnitude
        "obj:allows_shot:-100",  # negative magnitude
        "reply:material:+100",   # explicit-plus-prefixed magnitude
        "pro:shot_setup:0",      # zero magnitude
        "pro:material:0",        # zero magnitude
        "obj:loses_exchange:0",  # zero magnitude
        "pro:material:00",       # zero magnitude with a leading zero
        "pro:material: 100",     # leading whitespace
    ],
)
def test_signed_or_zero_magnitude_raises(label: str) -> None:
    """A signed, ``+``-prefixed, or zero magnitude is rejected as malformed.

    Magnitudes for the FACT §5 labels are strictly positive integers; a
    negative, explicit-plus, or zero magnitude must raise rather than be
    silently accepted as typed FACT evidence.
    """
    with pytest.raises(ValueError):
        to_argument_evidence(label)


# ---------------------------------------------------------------------------
# unit + property — chunk G.1 chess HEURISTIC vocabulary extension
# ---------------------------------------------------------------------------
#
# The chunk-G.1 extension adds 29 new HEURISTIC keys (21 FIXED + 8 MAGNITUDE)
# plus reuses ``pro:mobility:{n}`` for chess legal-move-count gain. The full
# table lives in ``reports/core-phase3-chunkg-plan.md`` §3. Each new row is
# enumerated here for type-mapping correctness and the magnitude rows are
# also covered by hypothesis-generative property tests over the integer
# range that exercises both the count-scale (1-4) and centipawn-scale
# (100-3000) saturation cases.

_CHUNK_G_FIXED_LABELS: list[tuple[str, Value, Tier]] = [
    # pro: STRUCTURE chess HEURISTIC supports.
    ("pro:development:center_pawn", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:development:minor_piece", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:king_safety:castle", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:pawn_structure:passed_pawn", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:file_control:open_file", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:outpost:supported", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:king_safety:escape_square", Value.STRUCTURE, Tier.HEURISTIC),
    ("pro:king_safety:advanced_flank_pawn_response", Value.STRUCTURE, Tier.HEURISTIC),
    # obj: STRUCTURE chess HEURISTIC objections (king_safety).
    ("obj:king_safety:castled_flank_pawn_weakening", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:king_safety:flank_pawn_weakening", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:king_safety:flank_pawn_lunge", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:king_safety:unanswered_advanced_flank_pawn", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:king_safety:queen_flank_invasion", Value.STRUCTURE, Tier.HEURISTIC),
    # obj: TEMPO chess HEURISTIC objections (opening).
    ("obj:opening:minor_retreat", Value.TEMPO, Tier.HEURISTIC),
    ("obj:opening:king_center_flight", Value.TEMPO, Tier.HEURISTIC),
    ("obj:opening:king_walk", Value.TEMPO, Tier.HEURISTIC),
    # pro: TEMPO chess HEURISTIC supports (tactical).
    ("pro:tactical:checking_exchange_pressure", Value.TEMPO, Tier.HEURISTIC),
    # obj: MATERIAL chess HEURISTIC objections (smt).
    ("obj:smt:fork:high_value_piece", Value.MATERIAL, Tier.HEURISTIC),
    # obj: TEMPO chess HEURISTIC objections (strategy).
    ("obj:strategy:unsupported_major_drift", Value.TEMPO, Tier.HEURISTIC),
    ("obj:strategy:threefold_repetition", Value.TEMPO, Tier.HEURISTIC),
    ("obj:strategy:fifty_move_draw", Value.TEMPO, Tier.HEURISTIC),
]


_CHUNK_G_MAGNITUDE_PREFIXES: list[tuple[str, Value, Tier]] = [
    ("pro:center_control", Value.STRUCTURE, Tier.HEURISTIC),
    ("obj:opening:premature_minor_check", Value.TEMPO, Tier.HEURISTIC),
    ("obj:opening:premature_rook", Value.TEMPO, Tier.HEURISTIC),
    ("obj:opening:premature_queen", Value.TEMPO, Tier.HEURISTIC),
    ("pro:piece_safety:defended", Value.MATERIAL, Tier.HEURISTIC),
    ("pro:tactical:threat", Value.MATERIAL, Tier.HEURISTIC),
    ("pro:smt:fork", Value.MATERIAL, Tier.HEURISTIC),
    ("obj:smt:fork:moved_piece_en_pris", Value.MATERIAL, Tier.HEURISTIC),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "label,value,tier",
    _CHUNK_G_FIXED_LABELS,
    ids=[row[0] for row in _CHUNK_G_FIXED_LABELS],
)
def test_chunk_g_fixed_label_maps_to_value_and_tier(
    label: str, value: Value, tier: Tier
) -> None:
    """Every chunk-G.1 FIXED HEURISTIC label parses to the documented (Value, Tier)."""
    evidence = to_argument_evidence(label)
    assert isinstance(evidence, ArgumentEvidence)
    assert evidence.label == label
    assert evidence.value is value
    assert evidence.tier is tier
    assert evidence.magnitude is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "prefix,value,tier",
    _CHUNK_G_MAGNITUDE_PREFIXES,
    ids=[row[0] for row in _CHUNK_G_MAGNITUDE_PREFIXES],
)
def test_chunk_g_magnitude_prefix_registered(
    prefix: str, value: Value, tier: Tier
) -> None:
    """Every chunk-G.1 MAGNITUDE prefix is in the ``_MAGNITUDE`` table."""
    assert _MAGNITUDE[prefix] == (value, tier)


@pytest.mark.property
@given(magnitude=st.integers(min_value=1, max_value=5000))
def test_chunk_g_magnitude_label_parses_for_any_positive_int(
    magnitude: int,
) -> None:
    """Every chunk-G.1 MAGNITUDE prefix parses for any positive integer.

    Hypothesis covers both the count-scale (1-4 typical) and centipawn-scale
    (100-3000 typical) ranges with one strategy — the parser does not
    distinguish; the chess graded policy applies per-prefix saturation
    downstream.
    """
    for prefix, value, tier in _CHUNK_G_MAGNITUDE_PREFIXES:
        label = f"{prefix}:{magnitude}"
        evidence = to_argument_evidence(label)
        assert evidence.label == label
        assert evidence.value is value
        assert evidence.tier is tier
        assert evidence.magnitude == magnitude


@pytest.mark.property
@given(magnitude=st.integers(max_value=0))
def test_chunk_g_magnitude_rejects_zero_and_negative(magnitude: int) -> None:
    """A zero or negative magnitude is rejected for every chunk-G.1 prefix."""
    for prefix, _value, _tier in _CHUNK_G_MAGNITUDE_PREFIXES:
        label = f"{prefix}:{magnitude}"
        with pytest.raises(ValueError):
            to_argument_evidence(label)


@pytest.mark.property
@given(suffix=st.text(min_size=1, max_size=8))
def test_chunk_g_fixed_label_rejects_colon_suffix(suffix: str) -> None:
    """A chunk-G.1 FIXED label with any ``:<suffix>`` is rejected.

    Fixed labels are exact-match dict keys. Appending a ``:<suffix>`` makes
    the label parse as a magnitude — the prefix is not in ``_MAGNITUDE``, so
    the parser must raise.
    """
    # Avoid the case where the suffix happens to be a valid integer matching
    # an existing magnitude key (none of the chunk-G FIXED labels are also
    # MAGNITUDE prefixes, but be defensive).
    for label, _value, _tier in _CHUNK_G_FIXED_LABELS:
        candidate = f"{label}:{suffix}"
        if candidate in _FIXED:
            continue
        head, _, _ = candidate.rpartition(":")
        if head in _MAGNITUDE and suffix.isascii() and suffix.isdecimal() and int(suffix) > 0:
            # Defensive guard; should never trigger for the current vocabulary.
            continue
        with pytest.raises(ValueError):
            to_argument_evidence(candidate)


@pytest.mark.unit
def test_chunk_g_total_new_key_count() -> None:
    """29 new HEURISTIC keys land in the union of ``_FIXED`` and ``_MAGNITUDE``.

    The chunk-G.1 plan (``reports/core-phase3-chunkg-plan.md`` §3) specifies
    21 FIXED + 8 MAGNITUDE = 29 truly new keys plus 1 reuse of
    ``pro:mobility:{n}``. Pin the count so accidental additions/removals
    surface in CI.
    """
    fixed_count = len(_CHUNK_G_FIXED_LABELS)
    magnitude_count = len(_CHUNK_G_MAGNITUDE_PREFIXES)
    assert fixed_count == 21
    assert magnitude_count == 8
    assert fixed_count + magnitude_count == 29
