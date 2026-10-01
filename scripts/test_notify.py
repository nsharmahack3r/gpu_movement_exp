"""Send a test ntfy notification (uses ``NTFY_NOTIFICATION_TOPIC`` from ``.env``).

    uv run python scripts/test_notify.py

Edit the constants below to change what the test sends. Exit code is 0 when the
server accepted it and 1 otherwise (e.g. no topic set, or delivery failed).
"""

from __future__ import annotations

import logging

from movement.utils.env import load_env
from movement.utils.notify import configured, notify

TITLE = "Test"
MESSAGE = "gpu_movement_exp notification test"
TAGS = ["test_tube"]
PRIORITY = "default"


def main() -> None:
    load_env()
    if not configured():
        raise SystemExit("NTFY_NOTIFICATION_TOPIC is not set in .env; nothing sent.")
    sent = notify(TITLE, MESSAGE, tags=TAGS, priority=PRIORITY)
    print("sent" if sent else "delivery failed (see warning above)")
    raise SystemExit(0 if sent else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    main()
