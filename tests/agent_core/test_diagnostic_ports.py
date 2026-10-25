import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("synthetic_diagnostic_ports", Path(__file__).resolve().parents[2] / "scripts/run_agent_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("system", ["posix", "nt"])
def test_preflight_has_platform_binding_policy_without_port_sharing(monkeypatch, system):
    calls = []
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def setsockopt(self, *args): calls.append(("option", args))
        def bind(self, address): calls.append(("bind", address))
        def listen(self, backlog): calls.append(("listen", backlog))
    monkeypatch.setattr(module.socket, "socket", FakeSocket)
    monkeypatch.setattr(module.os, "name", system)
    monkeypatch.setattr(module.socket, "SO_EXCLUSIVEADDRUSE", 99999, raising=False)
    module.port_preflight(SimpleNamespace(api_port=18000, logistics_port=18101, aftersales_port=18102))
    options = [value[1] for value in calls if value[0] == "option"]
    assert len(options) == 3
    assert all(value[1] == (module.socket.SO_REUSEADDR if system == "posix" else 99999) for value in options)
    if hasattr(module.socket, "SO_REUSEPORT"):
        assert all(value[1] != module.socket.SO_REUSEPORT for value in options)


def test_preflight_keeps_live_port_error_as_environment_failure(monkeypatch):
    class BusySocket:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def setsockopt(self, *_): pass
        def bind(self, _): raise OSError("synthetic occupied port")
    monkeypatch.setattr(module.socket, "socket", BusySocket)
    with pytest.raises(module.DiagnosticError, match="loopback_port_occupied"):
        module.port_preflight(SimpleNamespace(api_port=18000, logistics_port=18101, aftersales_port=18102))
