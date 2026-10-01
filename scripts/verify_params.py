"""Verify per-arm parameter counts on CPU, and find the mystery 16,512-param layer.

The spec's §2 formula was reverse-engineered from recorded parameter_count values.
This script instantiates each model from its config and counts parameters directly,
then decomposes each model's parameters by named module so the 128x128+128 linear
layer can be identified.

Usage:
    uv run python scripts/verify_params.py [--input-len 24] [--horizon 12]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from movement.config import load_config
from movement.models import build_model


def count_by_module(model: torch.nn.Module, prefix: str = "") -> dict[str, int]:
    """Recursively count parameters per named submodule."""
    counts: dict[str, int] = defaultdict(int)
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        if child_params:
            counts[f"{prefix}{name}"] = child_params
        counts.update(count_by_module(child, prefix=f"{prefix}{name}."))
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=12)
    args = p.parse_args(argv)

    print(f"input_len={args.input_len}, horizon={args.horizon}, n_features=2\n")
    for arm, cfg_path in [
        ("tcn", "configs/model/tcn.yaml"),
        ("lstm", "configs/model/lstm.yaml"),
        ("transformer", "configs/model/transformer.yaml"),
    ]:
        cfg = load_config(cfg_path)
        cfg.windowing.input_len = args.input_len
        cfg.windowing.horizon = args.horizon
        model = build_model(cfg.model, cfg.windowing, cfg.transforms)
        total = model.count_parameters()
        print(f"=== {arm}: {total:,} parameters ===")
        for name, count in sorted(count_by_module(model).items(), key=lambda kv: -kv[1]):
            print(f"  {name:<45} {count:>10,}")


if __name__ == "__main__":
    main()
