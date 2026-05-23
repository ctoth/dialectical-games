"""Cartridge-supplied lookahead backend, dispatched by name.

The core does NOT call a backend directly — backends are a *cartridge*
concern. The core owns only the **shape** of a backend (the
:class:`SearchBackend` Protocol) and the **registry** by which cartridges
declare and resolve backends by name (:class:`SearchBackendRegistry`).

The orchestrator's only consumer is the post-decision hook (Seam 3). The
orchestrator passes the registry's selected backend through the hook's
:class:`PostDecisionContext`; the hook does the actual dispatch.

A cartridge typically:

1. Implements one :class:`SearchBackend` per concrete search algorithm
   (checkers: ``negamax``, ``bstar``).
2. Constructs a :class:`SearchBackendRegistry` and registers each backend
   on package import.
3. Validates its ``EngineSettings.search_backend`` against
   ``registry.names`` rather than against a hard-coded frozenset.

A cartridge that needs no post-decision lookahead simply leaves the
registry empty.
"""

from __future__ import annotations

from typing import Any, Protocol

from dialectical_games.arguments import MoveProbe


class SearchBackend(Protocol):
    """A named, cartridge-supplied search routine.

    A backend takes the depth-0 probes for a position and returns a chosen
    probe (must be one of ``probes``). The ``settings`` carrier is the
    cartridge's own engine-settings object — the core never inspects it.
    ``deadline`` is a monotonic-clock cap (seconds, or ``None`` for
    unlimited); a backend that does not respect a deadline may ignore it.
    """

    @property
    def name(self) -> str:
        ...

    def run(
        self,
        *,
        board: object,
        probes: tuple[MoveProbe, ...],
        settings: Any,
        deadline: float | None,
    ) -> MoveProbe:
        ...


class SearchBackendRegistry:
    """A cartridge's by-name registry of available :class:`SearchBackend`.

    Cartridges construct one instance, register their backends on package
    import, and validate ``EngineSettings.search_backend`` against
    :attr:`names`. The post-decision hook (Seam 3) resolves the named
    backend via :meth:`get` and invokes its ``run`` method.

    An empty registry is allowed — a cartridge with no post-decision
    lookahead need not register anything.
    """

    def __init__(self) -> None:
        self._backends: dict[str, SearchBackend] = {}

    def register(self, backend: SearchBackend) -> None:
        """Register ``backend`` under its ``name`` attribute.

        Re-registering the same name replaces the prior backend — a
        cartridge that wants to forbid re-registration must check
        :attr:`names` itself before calling.
        """
        self._backends[backend.name] = backend

    def get(self, name: str) -> SearchBackend:
        """Return the backend registered under ``name``.

        Raises :class:`KeyError` if ``name`` is not registered. The
        cartridge typically validates ``settings.search_backend`` against
        :attr:`names` before calling so the error never reaches the user
        — but propagating ``KeyError`` is the documented contract.
        """
        try:
            return self._backends[name]
        except KeyError as exc:
            raise KeyError(f"unknown search_backend: {name!r}") from exc

    @property
    def names(self) -> frozenset[str]:
        """The set of currently-registered backend names."""
        return frozenset(self._backends)

    def __contains__(self, name: object) -> bool:
        return name in self._backends
