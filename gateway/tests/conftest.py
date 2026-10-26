"""Pytest configuration."""

import os

import pytest

# Tests spin up loopback WebSocket servers. websockets 14+ honors proxy env
# vars (HTTP_PROXY etc.) for ws:// clients, so a system proxy would hijack
# test connections to 127.0.0.1 and they'd time out. Bypass proxies entirely.
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "defaults: run with the shipped default environment (no opt-in tool flags)",
    )


# Use asyncio mode for all async tests
def pytest_collection_modifyitems(config, items):
    """Auto-mark all async tests."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            if asyncio_test(item):
                item.add_marker(pytest.mark.asyncio)


def asyncio_test(item):
    """Check if test is async."""
    return hasattr(item, "function") and hasattr(item.function, "__wrapped__")


@pytest.fixture(autouse=True)
def _no_real_face_tracker(monkeypatch):
    """Gateway tests never launch the real Mac camera face tracker."""
    monkeypatch.setenv("STACKCHAN_FACE_TRACKER_AUTOSTART", "0")
    monkeypatch.delenv("STACKCHAN_FACE_TRACKER_CAMERA", raising=False)


@pytest.fixture(autouse=True)
def _isolated_console_settings(tmp_path):
    """Never read or write the real data/console-settings.json from tests."""
    from stackchan_mcp.console_settings import reset_console_settings_for_tests

    reset_console_settings_for_tests(tmp_path / "console-settings.json")
    yield
    reset_console_settings_for_tests(None)


@pytest.fixture(autouse=True)
def _reset_debug_status():
    """Keep the process-wide debug snapshot from leaking across tests."""
    from stackchan_mcp.debug_status import reset_debug_status

    reset_debug_status()
    yield
    reset_debug_status()
