"""CLI entrypoints: ``uv run train`` and ``uv run eval``."""

from .eval import main as eval_main
from .train import main as train_main

__all__ = ["eval_main", "train_main"]
