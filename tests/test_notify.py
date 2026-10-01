"""ntfy notification tests: payload shape, config, and failure tolerance."""

from __future__ import annotations

import json
import urllib.error

import pytest

from movement.utils import notify as notify_mod  # the submodule, not a shadowing re-export


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep tests independent of the repo's real .env (which may have a topic)."""
    monkeypatch.setattr(notify_mod, "load_env", lambda: None)


@pytest.fixture
def capture(monkeypatch):
    """Capture the Request handed to urlopen, without any network."""
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_noop_without_topic(monkeypatch, capture):
    monkeypatch.delenv(notify_mod.TOPIC_VAR, raising=False)
    assert notify_mod.notify("title", "body") is False
    assert "request" not in capture  # urlopen was never called


def test_posts_json_to_topic(monkeypatch, capture):
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "gpu-runs")
    monkeypatch.delenv(notify_mod.URL_VAR, raising=False)
    monkeypatch.delenv(notify_mod.TOKEN_VAR, raising=False)

    assert notify_mod.notify("done", "ADE 1.0 m", tags=["warning"], priority="urgent") is True

    request = capture["request"]
    assert request.full_url == "https://ntfy.sh/"  # JSON publish posts to the root
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json"
    body = json.loads(request.data.decode("utf-8"))
    assert body == {"topic": "gpu-runs", "title": "done", "message": "ADE 1.0 m",
                    "tags": ["warning"], "priority": 5}


def test_custom_url_and_token(monkeypatch, capture):
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "t")
    monkeypatch.setenv(notify_mod.URL_VAR, "https://ntfy.example.com/")
    monkeypatch.setenv(notify_mod.TOKEN_VAR, "secret")

    notify_mod.notify("t", "m")

    request = capture["request"]
    assert request.full_url == "https://ntfy.example.com/"
    assert request.get_header("Authorization") == "Bearer secret"


def test_numeric_priority_passthrough(monkeypatch, capture):
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "t")
    notify_mod.notify("t", "m", priority=2)
    assert json.loads(capture["request"].data.decode("utf-8"))["priority"] == 2


def test_delivery_failure_is_swallowed(monkeypatch):
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "t")

    def boom(request, timeout=None):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", boom)
    assert notify_mod.notify("t", "m") is False  # must not raise


def test_configured(monkeypatch):
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "topic")
    assert notify_mod.configured() is True
    monkeypatch.setenv(notify_mod.TOPIC_VAR, "  ")
    assert notify_mod.configured() is False
