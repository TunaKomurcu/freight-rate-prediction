"""Phase 3: validation schemes, baselines, ablations and residual analysis.

Run:  python -m src.validate --scheme time|city|random   (fits and caches predictions)
      python -m src.validate --report                    (metrics tables + figures)

Leakage guard: every fold gets a fresh WeightImputer, corrupted-label filter and RateModel
fitted on that fold's training rows only. The only thing computed once on all development
rows is `eval_corrupted`, a label mask used purely to REPORT metrics on clean vs raw labels.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .baselines import GlobalRPM, LaneMedian, RidgeLogRPM
from .clean import WeightImputer, find_corrupted_rates, fix_features
from .config import ARTIFACTS, INK2, REPORTS, SEED, SERIES
from .data import load_raw
from .features import DIST_BANDS, daily_market
from .model import RateModel

TIME_FOLDS = [  # rolling origin, 2-month horizon, mirrors train(Jan-Oct) -> validation(Nov-Dec)
    ("T1: Jan-Apr > May-Jun", "2025-05-01", "2025-07-01"),
    ("T2: Jan-Jun > Jul-Aug", "2025-07-01", "2025-09-01"),
    ("T3: Jan-Aug > Sep-Oct", "2025-09-01", "2025-11-01"),
]
N_CITY_GROUPS = 8  # 64 cities -> groups of 8, like the 8 unseen cities in validation.csv

BASELINES = {"B1 global median rpm x distance": GlobalRPM, "B2 lane median rpm": LaneMedian,
             "B3 ridge (log rpm)": RidgeLogRPM}
MAIN = "M0 LightGBM log-rpm"
FINAL = "A40 A39 + recency half-life 60d"  # == model.FINAL_CONFIG
EXPERIMENTS = {
    MAIN: {},
    "A1 - quote_signal": {"quote_signal": False},
    "A2 - holiday features": {"holidays": False},
    "A3 + linear time trend": {"trend": "linear"},
    "A4 target = rpm": {"target": "rpm"},
    "A5 target = rate ($)": {"target": "rate"},
    "A6 raw labels + Huber": {"clean_labels": False, "objective": "huber"},
    "A7 raw labels + L2": {"clean_labels": False},
    "A8 + smearing": {"smearing": True},
    "A9 no unseen-city blanking": {"unseen_rate": 0.0},
    "A10 - holidays - quote_signal": {"holidays": False, "quote_signal": False},
    "A11 quote_signal as deviation from daily mean": {"quote_signal": "dev"},
    "A12 market: weekly cycle only (no slow level)": {"market_features": "cycle"},
    "A13 + linear trend (quarter-ramp controlled)": {"trend": "linear_q"},
    "A16 cycle + qs dev": {"market_features": "cycle", "quote_signal": "dev"},
    "A18 cycle + qs dev + trend_q": {"market_features": "cycle", "quote_signal": "dev", "trend": "linear_q"},
    "A20 cycle - quote_signal": {"market_features": "cycle", "quote_signal": False},
    "A21 cycle - holiday features": {"market_features": "cycle", "holidays": False},
    "A22 cycle + trend_q": {"market_features": "cycle", "trend": "linear_q"},
    "A23 cycle - holidays + smearing": {"market_features": "cycle", "holidays": False, "smearing": True},
    "A24 cycle - holidays + additive calendar": {"market_features": "cycle", "holidays": False, "calendar": "additive"},
    "A25 all market - holidays + additive calendar": {"holidays": False, "calendar": "additive"},
    "A26 cycle - holidays + additive calendar with market": {"market_features": "cycle", "holidays": False,
                                                            "calendar": "additive", "calendar_market": True},
    "A27 A24 + qs_day (quote regime)": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_day"},
    "A28 A24 + recency half-life 30d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "half_life": 30},
    "A29 A24 + recency half-life 60d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "half_life": 60},
    "A30 A24 + recency half-life 90d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "half_life": 90},
    "A31 A24 + level offset 28d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "level_window": 28},
    "A32 A24 + level offset 56d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "level_window": 56},
    "A33 A27 + level offset 28d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_day", "level_window": 28},
    "A34 A27 + recency half-life 60d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_day", "half_life": 60},
    "A35 A27 + recency half-life 90d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_day", "half_life": 90},
    "A36 A27 + level offset 42d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_day", "level_window": 42},
    "A37 A24 + qs_7d (smoothed quote regime)": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_7d"},
    "A38 A37 + recency half-life 60d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_7d", "half_life": 60},
    "A39 A24 + coarse quote regime": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_regime"},
    "A40 A39 + recency half-life 60d": {"market_features": "cycle", "holidays": False, "calendar": "additive", "quote_signal": "with_regime", "half_life": 60},
    "T1 leaves=63, min_child=20": {"params": {"num_leaves": 63, "min_child_samples": 20}},
    "T2 leaves=15, lr=0.05, 800 trees": {"params": {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 800}},
}
SCHEME_EXPERIMENTS = {
    "time": list(EXPERIMENTS),
    "city": [MAIN, "A1 - quote_signal", "A2 - holiday features", "A9 no unseen-city blanking",
             "A12 market: weekly cycle only (no slow level)", "A21 cycle - holiday features",
             "A24 cycle - holidays + additive calendar", "A27 A24 + qs_day (quote regime)",
             "A34 A27 + recency half-life 60d", "A39 A24 + coarse quote regime", "A40 A39 + recency half-life 60d"],
    "random": [MAIN, "A21 cycle - holiday features", "A24 cycle - holidays + additive calendar",
               "A27 A24 + qs_day (quote regime)", "A34 A27 + recency half-life 60d",
               "A39 A24 + coarse quote regime", "A40 A39 + recency half-life 60d"],
}


def prepare_dev():
    train_raw, valid_raw = load_raw()
    train, _ = fix_features(train_raw)
    valid, _ = fix_features(valid_raw)
    market = daily_market(train, valid)  # train+validation concatenated & sorted: Nov rolling uses late Oct
    imputed = WeightImputer().fit(train).transform(train)
    eval_corrupted = find_corrupted_rates(imputed)[0].to_numpy()
    return train, valid, market, eval_corrupted


def splits(train, scheme):
    if scheme == "time":
        for name, start, end in TIME_FOLDS:
            yield name, np.flatnonzero(train.date < start), np.flatnonzero((train.date >= start) & (train.date < end))
    elif scheme == "city":
        cities = np.array(sorted(set(train.pickup) | set(train.delivery)))
        np.random.default_rng(SEED).shuffle(cities)
        for g, group in enumerate(np.array_split(cities, N_CITY_GROUPS)):
            touches = train.pickup.isin(group) | train.delivery.isin(group)  # pickup OR delivery
            yield f"C{g + 1}", np.flatnonzero(~touches), np.flatnonzero(touches)
    else:
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(train)):
            yield f"R{k + 1}", a, b


def run(scheme, only=None):
    """Fit every configured model on every fold. `only`: subset of model names; their results
    replace those rows in the cached file, other models' cached results are kept."""
    train, _, market, eval_corrupted = prepare_dev()
    names = [n for n in SCHEME_EXPERIMENTS[scheme] if only is None or n in only]
    baselines = BASELINES if only is None else {}
    rows = []
    for fold, fit_idx, test_idx in splits(train, scheme):
        fit, test = train.iloc[fit_idx].reset_index(drop=True), train.iloc[test_idx].reset_index(drop=True)
        imputer = WeightImputer().fit(fit)
        fit_imp = imputer.transform(fit)
        bad = find_corrupted_rates(fit_imp)[0].to_numpy()  # fold-local label filter
        clean_fit, test_imp = fit_imp[~bad], imputer.transform(test)
        preds = {name: cls().fit(clean_fit).predict(test_imp) for name, cls in baselines.items()}
        for name in names:
            preds[name] = RateModel(market, **EXPERIMENTS[name]).fit(fit, corrupted=bad).predict(test)
            print(f"{scheme} {fold} {name}: MAE {np.abs(preds[name] - test.posted_rate).mean():.1f}", flush=True)
        base = pd.DataFrame({"scheme": scheme, "fold": fold, "load_id": test.load_id, "date": test.date,
                             "equipment": test.equipment, "distance": test.distance,
                             "y": test.posted_rate, "corrupted": eval_corrupted[test_idx],
                             "n_train": len(fit), "n_train_dropped": int(bad.sum())})
        rows += [base.assign(model=name, pred=p) for name, p in preds.items()]
    out = pd.concat(rows, ignore_index=True)
    path = ARTIFACTS / f"cv_{scheme}.csv"
    if only is not None and path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        out = pd.concat([old[~old.model.isin(names)], out], ignore_index=True)
    ARTIFACTS.mkdir(exist_ok=True)
    out.to_csv(path, index=False)


# ------------------------------------------------------------------ reporting
def metrics(g):
    err = g.pred - g.y
    return pd.Series({"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
                      "MAPE %": 100 * (err.abs() / g.y).mean(), "n": len(g)})


def table(df, labels):
    d = df if labels == "raw" else df[~df.corrupted]
    per_fold = d.groupby(["model", "fold"]).apply(metrics, include_groups=False).reset_index()
    mean = per_fold.groupby("model")[["MAE", "RMSE", "MAPE %"]].mean()
    return per_fold, mean


def fmt(mean_clean, mean_raw, order):
    t = mean_clean.join(mean_raw, lsuffix=" (clean)", rsuffix=" (raw)").reindex([o for o in order if o in mean_clean.index])
    return t.round(2)


KEY_MODELS = list(BASELINES) + [MAIN, "A21 cycle - holiday features", "A24 cycle - holidays + additive calendar",
                                 "A27 A24 + qs_day (quote regime)", "A34 A27 + recency half-life 60d",
                                 "A39 A24 + coarse quote regime", "A40 A39 + recency half-life 60d"]
NOTES = {
    "A3 + linear time trend": "M0 + linear trend removed from the log target (slope from daily residuals, controlling for log market index)",
    "A13 + linear trend (quarter-ramp controlled)": "as A3, but the slope regression also controls for the quarter-end ramp",
    "A12 market: weekly cycle only (no slow level)": "drops market_index, mi_day, mi_7d, mi_28d; keeps mi_cycle (= day - 28d mean) and per-load deviation",
    "A24 cycle - holidays + additive calendar": "tree without date features + ridge stage 2 (weekday + quarter-end ramp) on out-of-fold residuals",
    "A27 A24 + qs_day (quote regime)": "adds the daily mean quote_signal, which identifies the regime of the quote-price relation",
    "A31 A24 + level offset 28d": "level offset = mean out-of-time residual of the last 28 training days",
    "A34 A27 + recency half-life 60d": "most accurate in CV, but the continuous daily quote mean acts as a date ID: erratic December curve",
    "A37 A24 + qs_7d (smoothed quote regime)": "trailing 7-day mean quote instead of the daily mean",
    "A39 A24 + coarse quote regime": "quote regime as 3 levels (7-day mean quote < 2.0 / 2.0-2.1 / > 2.1)",
    "A40 A39 + recency half-life 60d": "FINAL: lowest worst-fold MAE among models with a smooth December curve",
}


def summary_table():
    """One complete row per key model: time folds, mean, worst, MAPE, monthly bias, city, random."""
    rows = {}
    for scheme in ("time", "city", "random"):
        df = pd.read_csv(ARTIFACTS / f"cv_{scheme}.csv", parse_dates=["date"])
        c = df[~df.corrupted & df.model.isin(KEY_MODELS)].copy()
        c["ae"], c["pe"] = (c.pred - c.y).abs(), 100 * (c.pred - c.y) / c.y
        per_fold = c.groupby(["model", "fold"]).ae.mean().unstack()
        for m in per_fold.index:
            r = rows.setdefault(m, {})
            if scheme == "time":
                r.update({f.split(":")[0]: per_fold.loc[m, f] for f in per_fold.columns})
                r["time mean"], r["time worst"] = per_fold.loc[m].mean(), per_fold.loc[m].max()
                g = c[c.model == m]
                r["time MAPE %"] = 100 * (g.ae / g.y).mean()
                r.update({f"bias {k} %": v for k, v in g.groupby(g.date.dt.strftime("%b")).pe.mean().items()})
            else:
                r[f"{scheme} MAE"] = per_fold.loc[m].mean()
    t = pd.DataFrame(rows).T.reindex(KEY_MODELS)
    bias_cols = [f"bias {m} %" for m in ("May", "Jun", "Jul", "Aug", "Sep", "Oct")]
    return t[["T1", "T2", "T3", "time mean", "time worst", "time MAPE %", "city MAE", "random MAE"] + bias_cols].round(2)


def report():
    from .plotting import plt, save, setup
    setup()
    order = list(BASELINES) + list(EXPERIMENTS)
    legend = pd.DataFrame([{"id": n, "config (changes vs defaults)": str(EXPERIMENTS.get(n, "baseline")),
                            "note": NOTES.get(n, "")} for n in order])
    md = ["# Validation results", "",
          "Dollar-scale metrics. *clean* = test rows whose label is not flagged as corrupted; "
          "*raw* = all test rows (Spotter's labels likely contain the same ~1.4% corruption).",
          "Time folds: T1 Jan-Apr > May-Jun, T2 Jan-Jun > Jul-Aug, T3 Jan-Aug > Sep-Oct. "
          "Final model: " + FINAL, "",
          "## Key models (MAE $, clean labels; bias = mean signed % error, negative = under-prediction)", "",
          summary_table().to_markdown(), "",
          "## Experiment legend", "", legend.to_markdown(index=False), ""]
    per_fold_all = {}
    for scheme in ("time", "city", "random"):
        df = pd.read_csv(ARTIFACTS / f"cv_{scheme}.csv", parse_dates=["date"])
        pf_c, m_c = table(df, "clean")
        pf_r, m_r = table(df, "raw")
        per_fold_all[scheme] = df
        md += [f"## Scheme: {scheme} (mean over folds)", "", fmt(m_c, m_r, order).to_markdown(), ""]
        wide = pf_c.pivot(index="model", columns="fold", values="MAE").reindex([o for o in order if o in m_c.index])
        md += [f"Per-fold MAE, clean labels ({scheme}):", "", wide.round(1).to_markdown(), ""]
        if scheme == "time":
            wide = pf_r.pivot(index="model", columns="fold", values="RMSE").reindex([o for o in order if o in m_c.index])
            md += ["Per-fold RMSE, raw labels (time):", "", wide.round(1).to_markdown(), ""]
        folds = df.groupby("fold")[["n_train", "n_train_dropped"]].first()
        folds["n_test"] = df[df.model == MAIN].groupby("fold").size()
        c = df[~df.corrupted].assign(pe=100 * (df.pred - df.y) / df.y, month=df.date.dt.strftime("%Y-%m"))
        if scheme == "time":
            bias = c.pivot_table(index="model", columns="month", values="pe", aggfunc="mean")
            md += ["Mean signed % error by test month (clean labels; negative = under-prediction):", "",
                   bias.reindex([o for o in order if o in bias.index]).round(2).to_markdown(), ""]
        md += ["Fold sizes:", "", folds.to_markdown(), ""]
    (REPORTS / "validation_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    # breakdowns + residual plots for the main model on the time folds (clean labels)
    t = per_fold_all["time"]
    t = t[(t.model == FINAL) & ~t.corrupted].copy()
    t["distance band"] = pd.cut(t.distance, DIST_BANDS[:-1] + [4000]).astype(str)
    t["month"] = t.date.dt.strftime("%Y-%m")
    parts = []
    for col in ("equipment", "distance band", "month"):
        b = t.groupby(col).apply(metrics, include_groups=False).round(2)
        parts += [f"### By {col}", "", b.to_markdown(), ""]
    (REPORTS / "validation_breakdown.md").write_text(f"# Final model ({FINAL}), time folds, clean labels\n\n" + "\n".join(parts), encoding="utf-8")
    print("\n".join(parts))

    t["pct_err"] = 100 * (t.pred - t.y) / t.y
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    s = t.sample(min(6000, len(t)), random_state=0)
    ax[0].scatter(s.y, s.pred, s=3, alpha=.3, color=SERIES[0])
    lim = [0, s.y.quantile(.995)]
    ax[0].plot(lim, lim, color=INK2, lw=1, ls="--")
    ax[0].set(title="Predicted vs actual (time folds)", xlabel="actual $", ylabel="predicted $", xlim=lim, ylim=lim)
    b = t.groupby(pd.cut(t.distance, np.arange(0, 3601, 200)), observed=True).pct_err
    mid = [i.mid for i in b.median().index]
    ax[1].plot(mid, b.median().values, color=SERIES[0], marker="o", ms=4, label="median")
    ax[1].fill_between(mid, b.quantile(.1).values, b.quantile(.9).values, color=SERIES[0], alpha=.15, label="p10-p90")
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].set(title="% error vs distance", xlabel="distance (mi)", ylabel="% error")
    ax[1].legend()
    daily = t.groupby("date").pct_err.mean()
    ax[2].plot(daily.index, daily.values, color=SERIES[0], lw=1)
    ax[2].axhline(0, color=INK2, lw=1)
    ax[2].set(title="Daily mean % error over the test months", ylabel="% error")
    ax[2].tick_params(axis="x", rotation=30)
    save(fig, "10_residuals.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["time", "city", "random"])
    ap.add_argument("--models", help="comma-separated subset of model names (merged into the cache)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.scheme:
        run(a.scheme, a.models.split(",") if a.models else None)
    if a.report:
        report()


if __name__ == "__main__":
    main()
