"""Inspect the real deer dataset: sampling intervals, individuals, gaps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from movement.utils.env import load_env

load_env()
raw = Path("F:/dev/organisations/penn_state_univ/data_bank/raw")
csvs = sorted(raw.glob("*.csv"))
print(f"CSVs: {[c.name for c in csvs]}")

df = pd.read_csv(csvs[0])
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Individuals: {df['individual_id'].nunique()}")
print(f"Studies: {df['study_id'].nunique()}")
print(f"Species: {df['species'].unique()}")
print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

ts = pd.to_datetime(df["timestamp"])
dt = ts.diff().dt.total_seconds()
dt_h = dt.dropna() / 3600.0
print(f"\nInterval stats (hours):\n{dt_h.describe()}")
print(f"\nFixes with gap > 3h: {(dt_h > 3).sum():,} ({(dt_h > 3).mean()*100:.2f}%)")
print(f"Fixes with gap > 6h: {(dt_h > 6).sum():,}")
print(f"Fixes with gap > 24h: {(dt_h > 24).sum():,}")
print(f"Duplicate timestamps (per individual): {df.duplicated(['individual_id','timestamp']).sum():,}")

# Per-individual fix counts.
counts = df.groupby("individual_id").size()
print(f"\nFixes per individual: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}")
print(f"Individuals with >= 36 fixes (input_len+horizon=24+12): {(counts >= 36).sum()} / {len(counts)}")
print(f"Individuals with >= 100 fixes: {(counts >= 100).sum()}")
