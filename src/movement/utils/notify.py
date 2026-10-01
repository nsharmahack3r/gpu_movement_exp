"""ntfy.sh push notifications for long-running experiments.

Set ``NTFY_NOTIFICATION_TOPIC`` in ``.env`` (the topic name, e.g.
``gpu-movement-runs``) and subscribe to ``https://ntfy.sh/<topic>`` in the ntfy
app or web UI. Optional: ``NTFY_NOTIFICATION_URL`` for a self-hosted server,
``NTFY_NOTIFICATION_TOKEN`` for a protected topic.

Delivery is best-effort by design: a study that runs for hours on a GPU must
never fail because a notification could not be sent, so every error is logged
and swallowed, and ``notify`` simply returns ``False``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

from .env import load_env

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://ntfy.sh"
TOPIC_VAR = "NTFY_NOTIFICATION_TOPIC"
URL_VAR = "NTFY_NOTIFICATION_URL"
TOKEN_VAR = "NTFY_NOTIFICATION_TOKEN"
# ntfy's named priorities, mapped to the numeric levels it accepts
# ("urgent" is its alias for the top level).
PRIORITIES = {"min": 1, "low": 2, "default": 3, "high": 4, "max": 5, "urgent": 5}


def configured() -> bool:
    """True when a notification topic is set."""
    load_env()
    return bool(os.environ.get(TOPIC_VAR, "").strip())


def notify(title: str, message: str, *, tags: list[str] | None = None,
           priority: str | int | None = None, timeout: float = 10.0) -> bool:
    """Publish one notification and return whether the server accepted it.

    No-op (returns ``False``) when ``NTFY_NOTIFICATION_TOPIC`` is unset, so it is
    safe to call unconditionally. Never raises; failures are logged and reported
    as ``False``. ``tags`` are ntfy's emoji short names (e.g. ``"warning"``);
    ``priority`` is a named level (``"low"`` … ``"urgent"``) or its number.
    """
    load_env()
    topic = os.environ.get(TOPIC_VAR, "").strip()
    if not topic:
        return False
    url = os.environ.get(URL_VAR, "").strip() or DEFAULT_URL
    payload: dict[str, object] = {"topic": topic, "title": title, "message": message}
    if tags:
        payload["tags"] = list(tags)
    if priority is not None:
        payload["priority"] = PRIORITIES.get(str(priority).lower(), priority)
    # ntfy's JSON publish format posts to the server root, with the topic in the
    # body. Posting the JSON to /<topic> instead makes the server treat it as the
    # message text, so the raw JSON is what shows up on the phone.
    request = urllib.request.Request(
        f"{url.rstrip('/')}/",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    token = os.environ.get(TOKEN_VAR, "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001 - notifications must never break a run
        logger.warning("ntfy notification failed (%s): %s", title, exc)
        return False
