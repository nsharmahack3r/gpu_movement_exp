# Scan elephant data for extreme single-step displacements (outlier candidates)
import numpy as np

from movement.data.loading import load_raw_csv
from movement.data.transforms import project_to_local

raw = "F:/dev/organisations/penn_state_univ/data_bank/raw/african_elephant_reshaped.csv"
df = load_raw_csv(raw)
print(f"Rows: {len(df):,}, individuals: {df['individual_id'].nunique()}")

# Per-individual consecutive step distances (metres).
all_steps = []
for (study, indiv), g in df.groupby(["study_id", "individual_id"], sort=False):
    g = g.sort_values("timestamp")
    lat = g["lat"].to_numpy(float)
    lon = g["lon"].to_numpy(float)
    x, y, _, _ = project_to_local(lat, lon)
    d = np.hypot(np.diff(x), np.diff(y))
    all_steps.append((indiv, d))

big = []
for indiv, d in all_steps:
    if len(d) and d.max() > 1000:  # >1 km in one step at 10s sampling
        big.append((indiv, len(d), float(d.max()), float(np.percentile(d, 99.9))))
        print(f"  {indiv}: n={len(d)}, max_step={d.max():.0f} m, p99.9={np.percentile(d, 99.9):.0f} m")

print(f"\nIndividuals with >1km steps: {len(big)}")
d_all = np.concatenate([d for _, d in all_steps])
print(f"Global step dist: p50={np.median(d_all):.1f} m, p99={np.percentile(d_all, 99):.1f} m, "
      f"p99.9={np.percentile(d_all, 99.9):.1f} m, max={d_all.max():.0f} m")
