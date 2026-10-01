# For a given run dir, report whether the run completed (manifest exists) and
# the checkpoint epoch stored in best.pt.
from pathlib import Path

import torch

for run in sorted(Path("runs").glob("white_tail_deer_reshaped/*")):
    if not run.is_dir():
        continue
    manifest = (run / "manifest.json").exists()
    best = run / "checkpoints" / "best.pt"
    epoch = "?"
    if best.exists():
        try:
            state = torch.load(best, map_location="cpu", weights_only=False)
            epoch = state.get("epoch")
        except Exception:
            epoch = "corrupt"
    print(f"{run.name}: manifest={manifest}, best.pt_epoch={epoch}")
