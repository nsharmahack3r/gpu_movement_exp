"""Pre-registered analysis of the green-wave experiment (reports/preregistration_green_wave.md).

Reads ``per_window.csv`` of the latest run of each arm in every fold of a study directory, assigns each
forecast to a season by the date of its origin (last observed fix, UTC) and applies the decision rules:

* H1 (primary): in spring migration (1 Apr – 30 Jun) ES(covariate model) − ES(reference) has a 95% animal-bootstrap
  CI entirely below 0 **and** is negative in >= 4 of 5 folds.
* H2 (control): (spring difference) − (winter difference, 1 Dec – 15 Mar) has a 95% animal-bootstrap CI below 0.
* H3 (secondary): all seasons pooled, same rule as H1.

The first arm in ``--arms`` is the primary covariate model; any others are reported as secondary.
Defaults follow the amended pre-registration: reference ``pnocov``, primary ``faunaformer`` (both with the
9-day imagery embargo passed at training time).

Usage:
    uv run python scripts/green_wave_analysis.py --dataset mule_deer_6h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 4000
SEASONS = ["spring migration", "winter", "summer/autumn"]


def season(t: pd.Series) -> pd.Series:
    t = pd.to_datetime(t)
    md = t.dt.month * 100 + t.dt.day
    out = np.where((md >= 401) & (md <= 630), "spring migration",
                   np.where((md >= 1201) | (md <= 315), "winter", "summer/autumn"))
    return pd.Series(out, index=t.index)


def latest(run_dir: Path) -> Path | None:
    runs = sorted(p for p in run_dir.glob("2*") if (p / "per_window.csv").exists())
    return runs[-1] if runs else None


def load(study: Path, arms: list[str]) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(study.glob("fold*")):
        per_arm = {}
        for arm in arms:
            run = latest(fold_dir / arm)
            if run is not None:
                per_arm[arm] = pd.read_csv(run / "per_window.csv", parse_dates=["t_origin"])
        if len(per_arm) < len(arms):
            continue
        base = per_arm[arms[0]][["individual_id", "t_origin", "ade_cp"]].copy()
        for arm, df in per_arm.items():
            assert (df["individual_id"].to_numpy() == base["individual_id"].to_numpy()).all(), "window order differs"
            base[f"{arm}:es"] = df["es"].to_numpy()
            base[f"{arm}:ade"] = df["ade"].to_numpy()
            base[f"{arm}:fde"] = df["fde"].to_numpy()
        base["fold"] = fold_dir.name
        frames.append(base)
    if not frames:
        raise SystemExit(f"No fold in {study} has per_window.csv for all of {arms}. Evaluate first.")
    df = pd.concat(frames, ignore_index=True)
    df["season"] = season(df["t_origin"])
    return df


def animal_boot(d: pd.Series, animals: pd.Series, rng: np.random.Generator) -> tuple[float, float, float]:
    g = d.groupby(animals).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def contrast_boot(df: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """(spring diff) − (winter diff), bootstrapping animals jointly across both seasons."""
    d = df[f"{a}:es"] - df[f"{b}:es"]
    tab = pd.DataFrame({"animal": df["individual_id"], "season": df["season"], "d": d})
    agg = tab[tab.season.isin(["spring migration", "winter"])].groupby(["animal", "season"])["d"].agg(["sum", "count"])
    agg = agg.unstack("season").fillna(0.0)
    ss, sc = agg[("sum", "spring migration")].to_numpy(), agg[("count", "spring migration")].to_numpy()
    ws, wc = agg[("sum", "winter")].to_numpy(), agg[("count", "winter")].to_numpy()

    def stat(ix):
        return ss[ix].sum(-1) / np.maximum(sc[ix].sum(-1), 1) - ws[ix].sum(-1) / np.maximum(wc[ix].sum(-1), 1)

    point = float(stat(np.arange(len(ss))))
    bs = stat(rng.integers(0, len(ss), size=(N_BOOT, len(ss))))
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="mule_deer_6h")
    p.add_argument("--tag", default="kfold5_seed42_anchor_last_v2")
    p.add_argument("--reference", default="pnocov")
    p.add_argument("--arms", default="faunaformer")
    p.add_argument("--exploratory", default=None, metavar="LABEL",
                   help="Analysis added after the pre-registered results were seen: written to "
                        "<dataset>_green_wave_<LABEL>.md and headed EXPLORATORY.")
    args = p.parse_args(argv)
    study = REPO_ROOT / "runs" / "covariate_study" / args.dataset / args.tag
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    df = load(study, [args.reference, *arms])
    rng = np.random.default_rng(0)
    ref = args.reference
    title = f"# Green-wave experiment — {args.dataset}" + (f" — EXPLORATORY ({args.exploratory})" if args.exploratory else "")
    L = [title, "",
         *(["> **Exploratory.** Added after the pre-registered results were known (see the deviations in "
            "`reports/preregistration_green_wave.md`). The decision rules below are applied as written, but a "
            "positive result here is hypothesis-generating, not confirmatory.", ""] if args.exploratory else []),
         "Pre-registered in `reports/preregistration_green_wave.md`. Energy score (ES), ADE, FDE in metres; "
         "lower is better. Seasons by forecast-origin date (UTC). CIs: 95% bootstrap over animals "
         f"({N_BOOT} resamples).", "",
         f"Test windows: {len(df):,} from {df['individual_id'].nunique()} animals in {df['fold'].nunique()} folds.", "",
         "## 1. Scores by season", "",
         "| Season | windows | stay put | " + " | ".join(f"`{a}` ES" for a in [ref, *arms]) + " | "
         + " | ".join(f"`{a}` ADE" for a in [ref, *arms]) + " |",
         "|---|---|---|" + "---|" * (2 * (len(arms) + 1))]
    for s in [*SEASONS, "all"]:
        sub = df if s == "all" else df[df.season == s]
        if sub.empty:
            continue
        L.append(f"| {s} | {len(sub):,} | {sub['ade_cp'].mean():.1f} | "
                 + " | ".join(f"{sub[f'{a}:es'].mean():.1f}" for a in [ref, *arms]) + " | "
                 + " | ".join(f"{sub[f'{a}:ade'].mean():.1f}" for a in [ref, *arms]) + " |")
    verdicts = {}
    for a in arms:
        L += ["", f"## 2. `{a}` − `{ref}` (energy score)", "",
              "| Season | Δ ES (m) | 95% CI | Δ % | folds with Δ < 0 |", "|---|---|---|---|---|"]
        res = {}
        for s in [*SEASONS, "all"]:
            sub = df if s == "all" else df[df.season == s]
            if sub.empty:
                continue
            d = sub[f"{a}:es"] - sub[f"{ref}:es"]
            m, lo, hi = animal_boot(d, sub["individual_id"], rng)
            per_fold = d.groupby(sub["fold"]).mean()
            neg = int((per_fold < 0).sum())
            res[s] = (m, lo, hi, neg, len(per_fold))
            L.append(f"| {s} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {100 * m / sub[f'{ref}:es'].mean():+.2f}% | "
                     f"{neg}/{len(per_fold)} |")
        c, clo, chi = contrast_boot(df, a, ref, rng)
        L += ["", f"Spring − winter difference of Δ: {c:+.2f} m [{clo:+.2f}, {chi:+.2f}]"]
        sp, al = res.get("spring migration"), res.get("all")
        h1 = bool(sp and sp[2] < 0 and sp[3] >= 4)
        h2 = bool(chi < 0)
        h3 = bool(al and al[2] < 0 and al[3] >= 4)
        verdicts[a] = {"H1_spring": h1, "H2_spring_vs_winter": h2, "H3_all": h3}
        L += ["", f"**Decision rules:** H1 (spring) **{'supported' if h1 else 'not supported'}**; "
                  f"H2 (spring benefit > winter) **{'supported' if h2 else 'not supported'}**; "
                  f"H3 (all seasons) **{'supported' if h3 else 'not supported'}**."]
    L += ["", f"Only `{arms[0]}` is the primary test"
          + (f"; {', '.join(f'`{a}`' for a in arms[1:])} are secondary." if len(arms) > 1 else ".")]
    suffix = f"_{args.exploratory}" if args.exploratory else ""
    out = REPO_ROOT / "reports" / "covariate_study" / f"{args.dataset}_green_wave{suffix}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    main()
