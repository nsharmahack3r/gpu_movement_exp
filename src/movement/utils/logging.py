"""Logging helpers: root logger setup that plays nicely with tqdm."""

from __future__ import annotations

import logging
import sys

import tqdm


class _TqdmHandler(logging.Handler):
    """Route log records through ``tqdm.write`` so bars don't interleave badly."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=sys.stderr)
        except Exception:  # noqa: BLE001 - logging must never crash
            self.handleError(record)


def setup_logging(level: str = "INFO", *, use_tqdm: bool = True) -> None:
    """Configure the root logger once.

    Logs go to stderr through ``tqdm.write`` (when active) so progress bars and
    log lines don't fight for the terminal.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = _TqdmHandler() if use_tqdm else logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    # Don't stack handlers on repeated calls (tests call this several times).
    root.handlers.clear()
    root.addHandler(handler)
    # Keep third-party loggers from being noisy below our level.
    for name in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
