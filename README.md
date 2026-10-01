```bash

uv run pytest tests/test_memory_hex.py -q
uv run python scripts/run_covariate_study.py --dataset boar_reshaped --arms pnocov,pmem,pmem_hex --folds 9 --val tail

# Test ntfy notifications (uses NTFY_NOTIFICATION_TOPIC from .env)
uv run python scripts/test_notify.py

```
