"""Crisp Dung layer + graded Categoriser layer (game-agnostic).

Two layers, both **game-agnostic**, both consumed by every cartridge that wants
the dialectical argumentation pipeline:

* the **crisp** layer — a plain Dung :class:`ArgumentationFramework` of
  FACT-tier defeaters, evaluated with ``formal-argumentation``'s
  ``grounded_extension``. A ``move:`` argument is grounded iff no undefeated
  FACT objection / reply attacks it. The surviving move set is its grounded
  ``move:`` arguments (or — the empty-survivor fallback — all moves). The
  crisp layer admits only FACT-tier witnesses.

* the **opinion-valued graded** layer — a :class:`doxa.BipolarOpinionGraph`
  whose nodes are the surviving ``move:`` arguments plus one leaf node per
  HEURISTIC witness on those survivors. ``doxa.evaluate`` resolves it bottom-up
  to a per-argument Jøsang :class:`doxa.Opinion`. Each move's resolved opinion
  accrues its HEURISTIC supporters (pro-reasons) and attackers (objections /
  heuristic reply attacks) under doxa's CCF operator. HEURISTIC defenses are
  witness nodes that attack only the answered objection / reply witness, so
  suppression is represented in the graph instead of being rechannelled as
  pro support. The per-move ``Opinion`` and its
  ``expectation()`` strength are exposed on :attr:`RootArgumentGraph.ranking`.

The graded layer **only ranks** — it never resurrects a crisply-eliminated
move (its move-node set is a subset of the crisp survivors), and never
overrides a FACT decision.

Phase 5 chunk 1: every witness on a :class:`MoveProbe` is a typed
:class:`ArgumentEvidence` (role / tier / magnitude / answered / tag); the
builders dispatch on :class:`Role` and :class:`Tier` enums only. A
``DEFENSE`` evidence's ``answered`` field is the EVIDENCE OBJECT it
answers — identity-typed — and the builders match by ``id(ev.answered)``
against the per-probe attacker map. No string parsing, no ``@``-keyed
labels, no prefix dispatch anywhere.

The crisp argument families (one Dung argument per row):

* ``move:{move_id}`` — one per legal move. The thing being attacked.
* ``obj:{move_id}:{ev_id}`` — one per FACT-tier objection. ``ev_id`` is the
  cartridge's ``tag`` when present, else the Python ``id(evidence)``.
* ``reply:{move_id}:{ev_id}`` — one per FACT-tier reply attack.
* ``defense:{move_id}:{ev_id}`` — one per FACT-tier proven defense. Defeats
  *only* the one objection / reply argument its ``answered`` evidence
  identifies on the same move.

There is no ``doubt`` node — soft reasoning lives in the graded layer. There
are no duplicated arguments — every argument id is distinct. HEURISTIC
witnesses never enter the crisp layer; the construction filters by
``ev.tier is Tier.FACT`` for crisp and ``ev.tier is Tier.HEURISTIC`` for
graded.

The graded layer is parameterised by a cartridge-supplied
:class:`GradedPolicy` — the only place game-specific tuning enters the
builder. Cartridges supply the move base rate, the per-witness leaf opinion,
and the (witness -> move) edge trust. The builder reads only those three
quantities through the Protocol.

This module imports only ``dialectical_games``, the stdlib, ``doxa`` and
``formal-argumentation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from argumentation.dung import ArgumentationFramework, grounded_extension
from doxa import BipolarOpinionGraph, Opinion, evaluate

from dialectical_games.evidence import ArgumentEvidence, Role
from dialectical_games.scheme import Tier


@dataclass(frozen=True)
class MoveProbe:
    """One AS1 argument for a legal move (design §5).

    Generic, cartridge-agnostic. ``move_id`` is the stable cartridge-supplied
    move identifier (checkers' PDN, chess's UCI, ...); ``evidence`` is the
    tuple of typed :class:`ArgumentEvidence` witnesses the cartridge probe
    layer emitted for this move.

    Phase 5 chunk 1: the four legacy string-tuple fields (``reasons`` /
    ``objections`` / ``reply_attacks`` / ``defenses``) collapsed into one
    typed ``evidence`` field. The builders partition by :attr:`Role`.

    ``child_eval`` and ``contested`` are **pre-computed cartridge-side** —
    they are the per-probe inputs the graded layer's policy reads. Pre-
    computation removes the need for the core builder to apply moves or
    inspect boards itself, and removes the need for a live ``StaticEvaluation``
    seam: the cartridge has already evaluated the child and decided whether
    the move is contested at probe time.

    * ``child_eval`` — the cartridge's chosen integer evaluation of the
      position the move reaches (centipawn-scale; opponent-relative by the
      checkers convention — smaller is better for the mover). The graded
      policy's ``move_base_rate`` typically squashes this into ``(0, 1)``.

    * ``contested`` — true iff the move carries both a HEURISTIC pro-reason
      and a HEURISTIC objection. The graded policy may use this to condition
      witness uncertainty.

    The ``score`` / ``search_score`` / ``search_line`` fields are cartridge-
    diagnostic carriers (probe-time integer evaluation, search-reported
    score, principal-variation line); the core ranking and decider do not
    read them.
    """

    move_id: str
    score: int = 0
    evidence: tuple[ArgumentEvidence, ...] = ()
    search_score: int | None = None
    search_line: tuple[str, ...] = ()
    child_eval: int = 0
    contested: bool = False

    @property
    def pdn(self) -> str:
        """Backwards-compatible alias for ``move_id`` (checkers' PDN name).

        Pre-Phase-2-continuation callers used ``probe.pdn``; the generic field
        is ``move_id``. The alias keeps the existing checkers call sites
        working while the cartridge migrates its own labels.
        """
        return self.move_id


@dataclass(frozen=True)
class RootArgumentGraph:
    """The crisp + graded argument graph output.

    ``arguments`` / ``defeats`` are the **crisp** Dung AF of FACT-tier
    defeaters; ``grounded_extension`` is its grounded extension;
    ``move_arguments`` maps each move's id to its ``move:`` argument id;
    ``survivors`` is the set of *move ids* that survive the crisp layer
    (the grounded ``move:`` arguments, or — under the empty-survivor
    fallback — all move ids).

    ``ranking`` carries the **opinion-valued graded** layer, built by
    :func:`build_graded_layer` over the crisp survivors. Its keys:

    * ``"move_opinions"`` — ``dict[str, Opinion]``: each surviving move's
      resolved Jøsang ``Opinion`` from ``doxa.evaluate``, keyed by move id.
    * ``"move_scores"`` — ``dict[str, float]``: each surviving move's scalar
      strength ``Opinion.expectation()`` (``b + a*u``), keyed by move id.
    * ``"opinions"`` — ``dict[str, Opinion]``: the resolved ``Opinion`` of
      every node of the graded graph (move nodes and HEURISTIC witness leaf
      nodes), for inspection / tests.
    * ``"arguments"`` / ``"supports"`` / ``"attacks"`` — the graded graph's
      node and edge sets, for inspection / tests.

    ``ranking`` is ``{}`` only for an empty graph (a terminal position, no
    probes); for any non-empty graph it carries the graded layer.
    """

    arguments: frozenset[str] = frozenset()
    defeats: frozenset[tuple[str, str]] = frozenset()
    move_arguments: dict[str, str] = field(default_factory=dict)
    grounded_extension: frozenset[str] = frozenset()
    survivors: frozenset[str] = frozenset()
    ranking: dict[str, Any] = field(default_factory=dict)


class GradedPolicy(Protocol):
    """Cartridge-supplied policy for the graded opinion-graph layer.

    The generic :func:`build_graded_layer` reads ONLY these methods/properties
    — it never reaches into a cartridge module, never imports tuning
    constants, never applies moves to a board. A cartridge implements the
    Protocol; the core never sees a tuning literal, a search function, or a
    board type.

    A policy is **per-build**: a cartridge typically constructs one policy
    object bound to the root board (so position-level features such as the
    game phase can be cached once), then passes it into the builder. The
    policy reads what it needs from each :class:`MoveProbe` (``child_eval``,
    ``contested``, the typed :class:`ArgumentEvidence` itself).

    The builder calls :meth:`with_probes` once at entry with the survivor
    probe sequence; a cartridge that needs per-position aggregates (a CDF
    over sibling magnitudes, a histogram of child evaluations) builds them
    there and returns a new policy carrying the cache. The default
    implementation returns ``self`` — a cartridge with no per-position
    aggregate needs nothing.
    """

    def with_probes(self, probes: Sequence[MoveProbe]) -> "GradedPolicy":
        """Return a policy bound to ``probes`` (chunk H').

        Called once by :func:`_build_graded_graph_internal` at entry, before
        iterating. A cartridge that needs to build per-position aggregates
        (e.g. a per-label-prefix CDF over sibling magnitudes, a per-position
        CDF over sibling child evaluations) does so here and returns a NEW
        policy carrying the cache (Protocol immutability — never mutate
        ``self``). The default implementation returns ``self``.

        The return value must still satisfy :class:`GradedPolicy`.
        """
        return self

    def move_base_rate(self, probe: MoveProbe) -> float:
        """The move node's vacuous-opinion base rate ``a``.

        Must lie strictly in the open interval ``(0, 1)`` (``doxa.Opinion``
        requires non-dogmatic). A typical implementation squashes
        ``probe.child_eval`` through a logistic so a better child evaluation
        for the mover maps to a larger ``a``.
        """
        ...

    def witness_opinion(
        self,
        *,
        probe: MoveProbe,
        evidence: ArgumentEvidence,
    ) -> Opinion:
        """The intrinsic ``Opinion(b, d, u, a)`` of one HEURISTIC witness leaf.

        Encodes the cartridge's belief band for ``evidence`` (its
        ``role`` / ``magnitude`` / ``tag``) plus any position-conditioned
        uncertainty. The same opinion shape encodes a pro-reason and an
        objection — the graph's ``supports`` vs ``attacks`` edge decides the
        sign, so a witness opinion is always a positive belief in *the
        witness's own claim*.
        """
        ...

    @property
    def edge_trust(self) -> Opinion:
        """The (witness -> move) edge trust opinion.

        v1: a constant per cartridge. Per-edge witness reliability is a
        later tuning knob.
        """
        ...


def _move_arg(move_id: str) -> str:
    """The ``move:`` argument id for the move identified by ``move_id``."""
    return f"move:{move_id}"


def _evidence_id(evidence: ArgumentEvidence) -> str:
    """A unique per-evidence id fragment for argument-id construction.

    Always includes ``id(evidence)`` so two evidence objects with the
    same cartridge ``tag`` on the same probe become distinct arguments
    (no collisions in the Dung framework's argument set). When ``tag``
    is set, it is concatenated as a diagnostic-friendly prefix; tests
    and logs see the cartridge-supplied name as well as the uniqueness
    suffix. The core never inspects the tag's structure; it only
    stringifies it.
    """
    if evidence.tag is None:
        return str(id(evidence))
    return f"{evidence.tag}:{id(evidence)}"


def obj_arg_id(move_id: str, evidence: ArgumentEvidence) -> str:
    """The objection argument id for FACT objection ``evidence`` on ``move_id``.

    Public so a cartridge selector can reconstruct the same objection
    argument id and ask whether that attacker is in the grounded extension
    — i.e. still undefeated.
    """
    return f"obj:{move_id}:{_evidence_id(evidence)}"


def reply_arg_id(move_id: str, evidence: ArgumentEvidence) -> str:
    """The reply-attack argument id for FACT reply ``evidence`` on ``move_id``.

    Public for the same reason as :func:`obj_arg_id`.
    """
    return f"reply:{move_id}:{_evidence_id(evidence)}"


def _defense_arg(move_id: str, evidence: ArgumentEvidence) -> str:
    """The defense argument id for FACT defense ``evidence`` on ``move_id``."""
    return f"defense:{move_id}:{_evidence_id(evidence)}"


def _witness_arg_id(move_id: str, evidence: ArgumentEvidence) -> str:
    """The graded-graph leaf-node id for HEURISTIC witness ``evidence``.

    A ``wit:`` prefix (distinct from the crisp layer's ``obj:`` / ``reply:``
    / ``defense:`` families) so a graded witness node can never collide with
    a crisp argument id. The move id is embedded so the same HEURISTIC tag
    on two different moves gets two distinct leaf nodes.
    """
    return f"wit:{move_id}:{_evidence_id(evidence)}"


def _build_graded_graph_internal(
    probes: list[MoveProbe],
    survivors: frozenset[str],
    policy: GradedPolicy,
) -> tuple[BipolarOpinionGraph, dict[str, str]]:
    """Construct the graded :class:`BipolarOpinionGraph` and the move-node map.

    Shared core of :func:`build_graded_layer` (returns the resolved
    ranking dict) and :func:`build_graded_graph` (returns the
    :class:`BipolarOpinionGraph` itself). Builds one ``move:`` node per
    surviving move and one ``wit:`` leaf per HEURISTIC witness, with
    support / attack edges keyed by edge-trust.

    The caller filters by ``survivors`` and screens ``policy is None`` /
    empty cases before invoking this helper.
    """
    arguments: set[str] = set()
    intrinsic: dict[str, Opinion] = {}
    supports: set[tuple[str, str]] = set()
    attacks: set[tuple[str, str]] = set()
    edge_opinions: dict[tuple[str, str], Opinion] = {}
    move_node_by_id: dict[str, str] = {}

    # Chunk H': bind the policy to the survivor probes ONCE before iterating.
    # A cartridge that needs per-position aggregates (e.g. a per-label-prefix
    # CDF over sibling magnitudes, a per-position CDF over sibling child
    # evaluations) builds them here and returns a NEW policy carrying the
    # cache. The default Protocol implementation returns ``self``.
    survivor_probes_seq = [p for p in probes if p.move_id in survivors]
    policy = policy.with_probes(survivor_probes_seq)

    edge_trust = policy.edge_trust

    for probe in probes:
        if probe.move_id not in survivors:
            continue
        move_id = _move_arg(probe.move_id)
        move_node_by_id[probe.move_id] = move_id
        arguments.add(move_id)
        intrinsic[move_id] = Opinion.vacuous(policy.move_base_rate(probe))

        # First pass: HEURISTIC pro / objection / reply-attack witnesses.
        # Defenses come in a second pass so an answered attacker is always
        # already in the per-probe attacker map by id(...) lookup.
        attacker_by_evidence: dict[int, str] = {}
        for ev in probe.evidence:
            if ev.tier is not Tier.HEURISTIC:
                continue
            if ev.role is Role.PRO:
                wit_id = _witness_arg_id(probe.move_id, ev)
                arguments.add(wit_id)
                intrinsic[wit_id] = policy.witness_opinion(
                    probe=probe, evidence=ev
                )
                edge = (wit_id, move_id)
                supports.add(edge)
                edge_opinions[edge] = edge_trust
            elif ev.role is Role.OBJECTION or ev.role is Role.REPLY_ATTACK:
                wit_id = _witness_arg_id(probe.move_id, ev)
                arguments.add(wit_id)
                intrinsic[wit_id] = policy.witness_opinion(
                    probe=probe, evidence=ev
                )
                edge = (wit_id, move_id)
                attacks.add(edge)
                edge_opinions[edge] = edge_trust
                attacker_by_evidence[id(ev)] = wit_id

        for ev in probe.evidence:
            if ev.tier is not Tier.HEURISTIC or ev.role is not Role.DEFENSE:
                continue
            if ev.answered is None:
                continue
            answered_wit_id = attacker_by_evidence.get(id(ev.answered))
            if answered_wit_id is None:
                continue
            wit_id = _witness_arg_id(probe.move_id, ev)
            arguments.add(wit_id)
            intrinsic[wit_id] = policy.witness_opinion(
                probe=probe, evidence=ev
            )
            edge = (wit_id, answered_wit_id)
            attacks.add(edge)
            edge_opinions[edge] = edge_trust

    graph = BipolarOpinionGraph(
        arguments=frozenset(arguments),
        intrinsic=intrinsic,
        supports=frozenset(supports),
        attacks=frozenset(attacks),
        edge_opinions=edge_opinions,
    )
    return graph, move_node_by_id


def build_graded_graph(
    probes: list[MoveProbe],
    survivors: frozenset[str],
    policy: GradedPolicy | None,
) -> BipolarOpinionGraph:
    """Build the opinion-valued graded :class:`BipolarOpinionGraph` directly.

    The graded layer's underlying :class:`doxa.BipolarOpinionGraph` —
    sibling to :func:`build_graded_layer`, which returns the *resolved*
    ranking dict. Callers that need the graph itself (sensitivity analysis,
    perturbation studies — e.g. the bstar search's S-D4 reply ordering)
    use this entry point; callers that need the resolved per-move
    expectations use :func:`build_graded_layer`.

    Same structure as :func:`build_graded_layer`: one ``move:{move_id}``
    node per surviving move, one ``wit:{move_id}:{ev_id}`` leaf per
    HEURISTIC witness, support / attack edges keyed by edge-trust.

    Returns the empty :class:`BipolarOpinionGraph` when ``policy is None``
    or when no probe survives — symmetric to :func:`build_graded_layer`.
    """
    if policy is None:
        return BipolarOpinionGraph(
            arguments=frozenset(),
            intrinsic={},
            supports=frozenset(),
            attacks=frozenset(),
            edge_opinions={},
        )
    survivor_probes = [p for p in probes if p.move_id in survivors]
    if not survivor_probes:
        return BipolarOpinionGraph(
            arguments=frozenset(),
            intrinsic={},
            supports=frozenset(),
            attacks=frozenset(),
            edge_opinions={},
        )
    graph, _ = _build_graded_graph_internal(survivor_probes, survivors, policy)
    return graph


def build_graded_layer(
    probes: list[MoveProbe],
    survivors: frozenset[str],
    policy: GradedPolicy | None,
) -> dict[str, Any]:
    """Build the opinion-valued graded layer over the crisp survivors.

    Returns the resolved ranking — a dict keyed by ``"move_opinions"``,
    ``"move_scores"``, ``"opinions"``, ``"arguments"``, ``"supports"``,
    ``"attacks"``. The graph itself (the :class:`doxa.BipolarOpinionGraph`
    before :func:`doxa.evaluate`) is the sibling :func:`build_graded_graph`.

    Each surviving move becomes a ``move:{move_id}`` node with intrinsic
    ``Opinion.vacuous(policy.move_base_rate(probe))``. Each HEURISTIC
    witness on a surviving move becomes a ``wit:{move_id}:{ev_id}`` leaf
    with the policy's ``witness_opinion(...)`` as intrinsic — support edge
    for a pro-reason, attack edge for an objection — every edge carrying
    ``policy.edge_trust``. ``doxa.evaluate`` resolves the graph; each
    move's resolved opinion is its ``Opinion.expectation()`` strength.

    ``policy`` may be ``None`` only when the caller has decided the graded
    layer is unused (a board-free unit test or a degenerate empty-survivor
    set). With ``policy is None`` and a non-empty survivor set the trivial
    empty layer is returned — no graded ranking is produced.

    The graded graph can never resurrect a crisply-eliminated move: its
    move-node set is exactly ``survivors``. An empty ``survivors`` yields
    the trivial empty layer.
    """
    survivor_probes = [p for p in probes if p.move_id in survivors]
    if not survivor_probes or policy is None:
        return {
            "move_opinions": {},
            "move_scores": {},
            "opinions": {},
            "arguments": frozenset(),
            "supports": frozenset(),
            "attacks": frozenset(),
        }

    graph, move_node_by_id = _build_graded_graph_internal(
        survivor_probes, survivors, policy
    )
    opinions = evaluate(graph)

    move_opinions = {
        m_id: opinions[node] for m_id, node in move_node_by_id.items()
    }
    move_scores = {
        m_id: opinion.expectation() for m_id, opinion in move_opinions.items()
    }

    return {
        "move_opinions": move_opinions,
        "move_scores": move_scores,
        "opinions": dict(opinions),
        "arguments": graph.arguments,
        "supports": graph.supports,
        "attacks": graph.attacks,
    }


def build_root_argument_graph(
    probes: list[MoveProbe],
    policy: GradedPolicy | None = None,
) -> RootArgumentGraph:
    """Build the crisp Dung argument graph + opinion-valued graded layer.

    For each probe (one per legal move):

    * a ``move:{move_id}`` argument;
    * for every FACT-tier objection on the probe, an ``obj:`` argument that
      defeats the move;
    * for every FACT-tier reply attack on the probe, a ``reply:`` argument
      that defeats the move;
    * for every FACT-tier defense on the probe, a ``defense:`` argument
      that defeats only the one objection / reply argument its
      :attr:`ArgumentEvidence.answered` identifies. A defense whose
      answered evidence is not in the move's attacker set defeats nothing.

    No ``doubt`` argument, no duplicated arguments — every id is distinct.
    HEURISTIC witnesses are filtered out of the crisp layer. The grounded
    extension is computed with ``formal-argumentation``; the surviving move
    set is the moves whose ``move:`` argument is grounded, or — when none is
    (the empty-survivor fallback) — all moves.

    The opinion-valued graded layer is then built by
    :func:`build_graded_layer` over those crisp survivors and stored on
    :attr:`RootArgumentGraph.ranking`. The graded layer is purely additive.

    ``policy`` is the cartridge-supplied :class:`GradedPolicy`. When
    ``None`` the graded layer is the trivial empty result.
    """
    arguments: set[str] = set()
    defeats: set[tuple[str, str]] = set()
    move_arguments: dict[str, str] = {}

    for probe in probes:
        move_id = _move_arg(probe.move_id)
        move_arguments[probe.move_id] = move_id
        arguments.add(move_id)

        # First pass: FACT attackers (objections + reply attacks). Each
        # attacker registers under ``id(evidence)`` so a defense built in
        # the same per-probe pass can locate it by identity.
        attacker_by_evidence: dict[int, str] = {}
        for ev in probe.evidence:
            if ev.tier is not Tier.FACT:
                continue
            if ev.role is Role.OBJECTION:
                arg_id = obj_arg_id(probe.move_id, ev)
                arguments.add(arg_id)
                defeats.add((arg_id, move_id))
                attacker_by_evidence[id(ev)] = arg_id
            elif ev.role is Role.REPLY_ATTACK:
                arg_id = reply_arg_id(probe.move_id, ev)
                arguments.add(arg_id)
                defeats.add((arg_id, move_id))
                attacker_by_evidence[id(ev)] = arg_id

        # Second pass: FACT defenses. ``answered`` is the EVIDENCE OBJECT
        # the defense answers; identity-keyed lookup wires the defeat edge
        # to the corresponding attacker argument.
        for ev in probe.evidence:
            if ev.tier is not Tier.FACT or ev.role is not Role.DEFENSE:
                continue
            arg_id = _defense_arg(probe.move_id, ev)
            arguments.add(arg_id)
            if ev.answered is not None:
                target = attacker_by_evidence.get(id(ev.answered))
                if target is not None:
                    defeats.add((arg_id, target))

    framework = ArgumentationFramework(
        arguments=frozenset(arguments),
        defeats=frozenset(defeats),
    )
    grounded = grounded_extension(framework)

    grounded_moves = frozenset(
        m_id for m_id, arg_id in move_arguments.items() if arg_id in grounded
    )
    if grounded_moves:
        survivors = grounded_moves
    else:
        survivors = frozenset(move_arguments)

    ranking = build_graded_layer(probes, survivors, policy)

    return RootArgumentGraph(
        arguments=frozenset(arguments),
        defeats=frozenset(defeats),
        move_arguments=move_arguments,
        grounded_extension=grounded,
        survivors=survivors,
        ranking=ranking,
    )
