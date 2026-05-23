"""Tests for the SearchBackend Protocol + SearchBackendRegistry."""

from __future__ import annotations

import pytest

from dialectical_games.arguments import MoveProbe
from dialectical_games.search_backend import (
    SearchBackend,
    SearchBackendRegistry,
)


class _RecordingBackend:
    """A trivial backend that records its inputs and returns probes[0]."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_call: dict[str, object] | None = None

    def run(
        self,
        *,
        board: object,
        probes: tuple[MoveProbe, ...],
        settings: object,
        deadline: float | None,
    ) -> MoveProbe:
        self.last_call = {
            "board": board,
            "probes": probes,
            "settings": settings,
            "deadline": deadline,
        }
        return probes[0]


def test_registry_starts_empty() -> None:
    reg = SearchBackendRegistry()
    assert reg.names == frozenset()


def test_register_and_get() -> None:
    reg = SearchBackendRegistry()
    backend = _RecordingBackend("alpha")
    reg.register(backend)
    assert reg.get("alpha") is backend
    assert reg.names == frozenset({"alpha"})
    assert "alpha" in reg
    assert "beta" not in reg


def test_get_unknown_raises_key_error() -> None:
    reg = SearchBackendRegistry()
    with pytest.raises(KeyError) as exc_info:
        reg.get("missing")
    assert "missing" in str(exc_info.value)


def test_register_replaces() -> None:
    """Re-registering the same name overwrites the prior backend."""
    reg = SearchBackendRegistry()
    a = _RecordingBackend("dup")
    b = _RecordingBackend("dup")
    reg.register(a)
    reg.register(b)
    assert reg.get("dup") is b
    assert reg.names == frozenset({"dup"})


def test_backend_protocol_call() -> None:
    """A registered backend runs with the documented keyword arguments."""
    reg = SearchBackendRegistry()
    backend = _RecordingBackend("rec")
    reg.register(backend)

    probes = (MoveProbe(move_id="m1"), MoveProbe(move_id="m2"))
    settings_obj = object()
    chosen = reg.get("rec").run(
        board=None, probes=probes, settings=settings_obj, deadline=None
    )
    assert chosen is probes[0]
    assert backend.last_call == {
        "board": None,
        "probes": probes,
        "settings": settings_obj,
        "deadline": None,
    }


def test_multiple_backends() -> None:
    reg = SearchBackendRegistry()
    a = _RecordingBackend("alpha")
    b = _RecordingBackend("beta")
    reg.register(a)
    reg.register(b)
    assert reg.names == frozenset({"alpha", "beta"})
    assert reg.get("alpha") is a
    assert reg.get("beta") is b


def test_protocol_structural_check() -> None:
    """``_RecordingBackend`` is structurally a :class:`SearchBackend`."""
    backend: SearchBackend = _RecordingBackend("p")
    assert backend.name == "p"
