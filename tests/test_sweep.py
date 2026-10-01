"""Sweep tooling tests (all CPU-only).

Covers the parts of scripts/sweep.py that can be verified without training:
space loading, trial-list construction, config resolution, capacity matching,
resume logic, VRAM guard, and the test-split guard (the search path must never
evaluate test).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ is not a package (no __init__.py); load sweep.py by file path so the
# test does not require a repo-layout change. Register it in sys.modules first
# (dataclasses and other stdlib machinery look up the module by name).
_SWEEP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep.py"
_spec = importlib.util.spec_from_file_location("movement_sweep", _SWEEP_PATH)
assert _spec and _spec.loader
sweep = importlib.util.module_from_spec(_spec)
sys.modules["movement_sweep"] = sweep
_spec.loader.exec_module(sweep)

ARMS = sweep.ARMS
Trial = sweep.Trial
build_trials = sweep.build_trials
capacity_match_overrides = sweep.capacity_match_overrides
config_hash = sweep.config_hash
load_space = sweep.load_space
resolve_config = sweep.resolve_config
sample_search_overrides = sweep.sample_search_overrides
_load_completed_hashes = sweep._load_completed_hashes
_append_trial_row = sweep._append_trial_row
estimate_vram_mb = sweep.estimate_vram_mb
_row_for_trial = sweep._row_for_trial
count_params_for_cfg = sweep.count_params_for_cfg
studies_path = sweep.studies_path
_recover_failed_trial = sweep._recover_failed_trial

from movement.config import Config  # noqa: E402

SPACE = load_space()


def test_space_loads_required_keys():
    for key in ("study_name", "datasets", "shared", "capacity", "seeds"):
        assert key in SPACE
    assert len(SPACE["datasets"]) == 3  # wolf, boar, cougars
    assert SPACE["seeds"] == [42, 7, 123]


def test_sensitivity_trial_count():
    trials = build_trials(SPACE, "sensitivity")
    assert len(trials) == 3 * 3 * 5  # 3 arms x 3 datasets x 5 lrs = 45
    # lr values match the spec (log-spaced 3e-5 .. 3e-3).
    lrs = sorted({t.overrides["trainer.lr"] for t in trials})
    assert lrs == pytest.approx([3e-5, 1e-4, 3e-4, 1e-3, 3e-3])


def test_search_trial_count_and_seed():
    trials = build_trials(SPACE, "search")
    assert len(trials) == 3 * 3 * 20  # 180
    # Seeded: rebuilding gives the identical schedule.
    trials2 = build_trials(SPACE, "search")
    assert [t.trial_id for t in trials] == [t.trial_id for t in trials2]
    # Capacity axis: each trial carries a label from its arm's options.
    labels: dict[str, set[str]] = {}
    for arm in ARMS:
        labels[arm] = {t.capacity for t in trials if t.arm == arm}
    assert labels["tcn"] == {"c32", "c48", "c64", "c96"}
    assert labels["lstm"] == {"h64", "h96", "h128", "h192"}
    assert labels["transformer"] == {"d64", "d96", "d128", "d192"}


def test_search_overrides_resolve_to_valid_config():
    import random

    rng = random.Random(0)
    for arm in ARMS:
        overrides, _ = sample_search_overrides(SPACE, arm, rng)
        trial = Trial(study="search", dataset="wolf_reshaped", arm=arm, seed=42, overrides=overrides)
        cfg = resolve_config(SPACE, trial)
        Config.model_validate(cfg)  # must not raise
        # Shared axes sampled from the declared ranges.
        assert SPACE["shared"]["lr"]["min"] <= cfg["trainer"]["lr"] <= SPACE["shared"]["lr"]["max"]
        assert cfg["model"]["dropout"] in SPACE["shared"]["dropout"]["options"]


def test_capacity_trial_count():
    trials = build_trials(SPACE, "capacity")
    assert len(trials) == 3 * 2  # 3 datasets x 2 arms (no TCN in the capacity arm)
    for t in trials:
        assert t.arm in ("lstm", "transformer")
        assert t.overrides  # a matched config was found


def test_capacity_match_within_tolerance():
    """The matched config's parameter count is within ±10% of the TCN's."""
    for ds in SPACE["datasets"]:
        tcn_cfg = resolve_config(SPACE, Trial(study="capacity", dataset=ds, arm="tcn", seed=42, overrides={}))
        tcn_params = count_params_for_cfg(tcn_cfg)
        for arm in ("lstm", "transformer"):
            matched = capacity_match_overrides(SPACE, ds, arm)
            assert matched is not None, f"no match for {arm}/{ds}"
            trial = Trial(study="capacity", dataset=ds, arm=arm, seed=42, overrides=matched)
            p = count_params_for_cfg(resolve_config(SPACE, trial))
            assert abs(p - tcn_params) / tcn_params <= 0.10, f"{arm}/{ds}: {p} vs {tcn_params}"


def test_config_hash_deterministic_and_distinct():
    a = {"trainer": {"lr": 1e-3}, "model": {"name": "tcn"}}
    b = {"trainer": {"lr": 1e-3}, "model": {"name": "tcn"}}
    c = {"trainer": {"lr": 3e-4}, "model": {"name": "tcn"}}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_resume_skips_completed(tmp_path):
    """A trial whose config_hash is recorded complete is skipped on re-run."""
    path = tmp_path / "trials.csv"
    trial = Trial(study="search", dataset="wolf_reshaped", arm="tcn", seed=42,
                  overrides={"trainer.lr": 1e-3})
    cfg = resolve_config(SPACE, trial)
    h = config_hash(cfg)
    _append_trial_row(path, _row_for_trial(
        SPACE, trial, cfg, status="complete", metrics={"ade": 123.4, "fde": 200.0},
    ))
    completed = _load_completed_hashes(path)
    assert h in completed
    assert len(completed) == 1


def test_failed_trial_not_skipped(tmp_path):
    """Failed trials are NOT in the resume set — only complete ones are."""
    path = tmp_path / "trials.csv"
    trial = Trial(study="search", dataset="wolf_reshaped", arm="tcn", seed=42,
                  overrides={"trainer.lr": 1e-3})
    cfg = resolve_config(SPACE, trial)
    _append_trial_row(path, _row_for_trial(
        SPACE, trial, cfg, status="failed", failure_reason="boom",
    ))
    completed = _load_completed_hashes(path)
    assert config_hash(cfg) not in completed


def test_interrupt_records_trial_and_stops(monkeypatch, tmp_path, capsys):
    """Ctrl-C mid-trial records the trial as failed/interrupted (retried on
    restart) and stops the sweep cleanly instead of tracebacking."""
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep, "_run_train_cli", _boom)
    # Redirect the trials.csv into tmp so the real results/ is untouched.
    monkeypatch.setattr(sweep, "studies_path", lambda space, study: tmp_path)

    with pytest.raises(SystemExit) as exc:
        sweep.run(SPACE, "sensitivity", limit=1)
    assert "Interrupted during trial" in str(exc.value)

    rows = list(csv.DictReader((tmp_path / "trials.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_reason"] == "interrupted"


def test_vram_guard_estimates():
    """Largest TCN configs estimate under the 8 GB ceiling on the sweep datasets."""
    for ds in SPACE["datasets"]:
        trial = Trial(study="search", dataset=ds, arm="tcn", seed=42,
                      overrides={"model.channels": [96, 96, 96, 96]})
        est = estimate_vram_mb(SPACE, trial)
        assert est < SPACE["vram_ceiling_mb"], f"{ds}: {est:.0f} MB exceeds ceiling"


def test_recover_failed_trial_promotes_complete(tmp_path, monkeypatch):
    """A trial recorded failed whose run dir already finished (manifest +
    best.pt + val metrics.json) is promoted to complete without retraining."""
    trial = Trial(study="sensitivity", dataset="wolf_reshaped", arm="tcn", seed=42,
                  overrides={"trainer.lr": 1e-3})
    cfg = resolve_config(SPACE, trial)

    # Point run_root into tmp so we control the artifacts.
    run_root = tmp_path / trial.trial_id
    monkeypatch.setattr(Trial, "run_root", property(lambda self: run_root))

    # A completed run dir: timestamped child with manifest + best.pt + val metrics.json.
    child = run_root / "20260101-000000-tcn-s42"
    (child / "checkpoints").mkdir(parents=True)
    (child / "checkpoints" / "best.pt").write_bytes(b"ckpt")
    (child / "config.yaml").write_text(
        "trainer:\n  checkpoint_dir: checkpoints\n", encoding="utf-8"
    )
    (child / "manifest.json").write_text(
        '{"parameter_count": 102104, "wall_clock_seconds": 100.0, "peak_vram_mb": 300.0}',
        encoding="utf-8",
    )
    (child / "metrics.json").write_text(
        '{"ade": 123.4, "fde": 200.0, "split": "val"}', encoding="utf-8"
    )

    # Record the trial as failed in a trials.csv inside tmp.
    trials_path = tmp_path / "trials.csv"
    _append_trial_row(trials_path, _row_for_trial(
        SPACE, trial, cfg, status="failed", failure_reason="train exited 1",
    ))

    assert _recover_failed_trial(SPACE, trial, trials_path) is True

    rows = list(csv.DictReader(trials_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["failure_reason"] == ""
    assert rows[0]["val_ade"] == "123.4"
    assert rows[0]["parameter_count"] == "102104"


def test_recover_failed_trial_false_when_incomplete(tmp_path, monkeypatch):
    """A failed trial with NO run dir is not recovered — it stays failed."""
    trial = Trial(study="sensitivity", dataset="wolf_reshaped", arm="tcn", seed=42,
                  overrides={"trainer.lr": 1e-3})
    cfg = resolve_config(SPACE, trial)
    monkeypatch.setattr(Trial, "run_root", property(lambda self: tmp_path / "nonexistent"))

    trials_path = tmp_path / "trials.csv"
    _append_trial_row(trials_path, _row_for_trial(
        SPACE, trial, cfg, status="failed", failure_reason="train exited 1",
    ))

    assert _recover_failed_trial(SPACE, trial, trials_path) is False
    rows = list(csv.DictReader(trials_path.open(encoding="utf-8")))
    assert rows[0]["status"] == "failed"


def test_search_path_never_reads_test():
    """The test split is off limits during search — enforced in code.

    The run loop evaluates the val split only; `finalise-test` is the single
    place eval runs with --split test. This greps the source for the guard so
    a future edit that adds test reads to the search path fails the test.
    """
    import inspect

    src = inspect.getsource(sweep)
    # The run path's eval call targets the val split only.
    assert "_evaluate_split(run_dir, \"val\")" in src
    # finalise-test is the only command that evaluates test.
    assert "finalise-test" in src
    # The search run loop must not contain a --split test eval.
    run_src = inspect.getsource(sweep.run)
    assert "--split\", \"test\"" not in run_src
    assert "split=\"test\"" not in run_src
    # And the only --split test construction lives in finalise_test.
    finalise_src = inspect.getsource(sweep.finalise_test)
    assert "_evaluate_split(run_dir, \"test\")" in finalise_src


def test_seeds_plan_includes_families(tmp_path, monkeypatch):
    """Arm D builds default + capacity-matched families; search-selected appears
    once search trials exist. Uses a tmp studies_path so real on-disk sweep data
    (if any) is never read or clobbered."""
    # Point studies_path (and RESULTS_ROOT) at tmp so _search_selected_overrides
    # reads from an isolated, controlled location.
    tmp_results = tmp_path / "results" / "sweeps"
    monkeypatch.setattr(sweep, "RESULTS_ROOT", tmp_results)
    monkeypatch.setattr(sweep, "studies_path", lambda space, study: tmp_results / space["study_name"] / study)

    # Without search trials: default (3 arms) + capacity-matched (2 arms) only.
    trials = build_trials(SPACE, "seeds")
    assert len(trials) == 45
    assert {t.capacity for t in trials} == {"default", "capacity-matched"}

    # Now write a fake search trials.csv (one complete row per arm/dataset).
    search_path = studies_path(SPACE, "search") / "trials.csv"
    search_path.parent.mkdir(parents=True, exist_ok=True)
    with search_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep.TRIAL_COLUMNS)
        writer.writeheader()
        for ds in SPACE["datasets"]:
            for arm in ARMS:
                writer.writerow({
                    "study": "search", "trial_id": f"fake-search-{ds}-{arm}",
                    "dataset": ds, "arm": arm, "seed": 42, "config_hash": "x" * 16,
                    "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.2,
                    "clip_grad_norm": 1.0, "warmup_fraction": 0.05,
                    "capacity": "c64", "parameter_count": 100000,
                    "val_ade": "100.0", "val_fde": "200.0",
                    "status": "complete", "failure_reason": "",
                })
    trials2 = build_trials(SPACE, "seeds")
    # default + search-selected (3 arms) + capacity-matched (2 arms) =
    # 8 families x 3 seeds x 3 datasets = 72.
    assert len(trials2) == 72
    assert {t.capacity for t in trials2} == {"default", "search-selected", "capacity-matched"}
