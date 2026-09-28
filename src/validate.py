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
FINAL = "A24 cycle - holidays + additive calendar"  # == model.FINAL_CONFIG
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
    "A14 + 28-day level anchor": {"level_window": 28},
    "A15 cycle + level anchor": {"market_features": "cycle", "level_window": 28},
    "A16 cycle + qs dev": {"market_features": "cycle", "quote_signal": "dev"},
    "A17 cycle + qs dev + level anchor": {"market_features": "cycle", "quote_signal": "dev", "level_window": 28},
    "A18 cycle + qs dev + trend_q": {"market_features": "cycle", "quote_signal": "dev", "trend": "linear_q"},
    "A19 cycle + qs dev + trend_q + level anchor": {"market_features": "cycle", "quote_signal": "dev", "trend": "linear_q", "level_window": 28},
    "A20 cycle - quote_signal": {"market_features": "cycle", "quote_signal": False},
    "A21 cycle - holiday features": {"market_features": "cycle", "holidays": False},
    "A22 cycle + trend_q": {"market_features": "cycle", "trend": "linear_q"},
    "A23 cycle - holidays + smearing": {"market_features": "cycle", "holidays": False, "smearing": True},
    "A24 cycle - holidays + additive calendar": {"market_features": "cycle", "holidays": False, "calendar": "additive"},
    "A25 all market - holidays + additive calendar": {"holidays": False, "calendar": "additive"},
    "A26 cycle - holidays + additive calendar with market": {"market_features": "cycle", "holidays": False,
                                                            "calendar": "additive", "calendar_market": True},
    "T1 leaves=63, min_child=20": {"params": {"num_leaves": 63, "min_child_samples": 20}},
    "T2 leaves=15, lr=0.05, 800 trees": {"params": {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 800}},
}
SCHEME_EXPERIMENTS = {
    "time": list(EXPERIMENTS),
    "city": [MAIN, "A1 - quote_signal", "A2 - holiday features", "A9 no unseen-city blanking",
             "A12 market: weekly cycle only (no slow level)", "A21 cycle - holiday features",
             "A24 cycle - holidays + additive calendar"],
    "random": [MAIN, "A21 cycle - holiday features", "A24 cycle - holidays + additive calendar"],
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


def report():
    from .plotting import plt, save, setup
    setup()
    order = list(BASELINES) + list(EXPERIMENTS)
    md = ["# Validation results", "",
          "Dollar-scale metrics. *clean* = test rows whose label is not flagged as corrupted; "
          "*raw* = all test rows (Spotter's labels likely contain the same ~1.4% corruption).", ""]
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
