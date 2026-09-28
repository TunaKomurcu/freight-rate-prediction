"""December chart preview + driver decomposition (run after src.train / src.predict).

Run: python -m src.december -> reports/december_drivers.md, reports/figures/11_december_preview.png
The day-to-day movement is split into additive log-space contributions (SHAP) and shown
as % of the month's average rate.
"""
import pickle

import numpy as np
import pandas as pd

from .clean import fix_features
from .config import ARTIFACTS, INK2, METRICS, REPORTS, SERIES
from .data import load_december, load_raw
from .features import city_coordinates, daily_market, prepare_december
from .predict import december_quote_fill


def drivers(model, dates):
    """Stage-2 date effects in % (the tree part is constant here: every December row has the same
    load features, and market_index equals the day mean so the per-load deviation is zero)."""
    D = model._date_design(dates)
    coef = model.calendar_coef_
    dow = D[[c for c in D if c.startswith("dow_")]] @ coef[[c for c in D if c.startswith("dow_")]]
    ramp = D["quarter_end_ramp"] * coef["quarter_end_ramp"]
    return pd.DataFrame({"weekday %": 100 * (np.exp(dow) - 1), "quarter-end ramp %": 100 * (np.exp(ramp) - 1)})


def main():
    from .plotting import plt, save, setup
    with open(ARTIFACTS / "model.pkl", "rb") as f:
        model = pickle.load(f)
    train_raw, valid_raw = load_raw()
    train, _ = fix_features(train_raw)
    dec = load_december()
    rows = prepare_december(dec, city_coordinates(train_raw, valid_raw), model.market, december_quote_fill(train))
    pred = model.predict(rows)
    tree = model.contributions(rows).drop(columns=["calendar (stage 2)", "trend", "level"]).sum(axis=1)
    out = pd.DataFrame({"date": rows.date.dt.date, "weekday": rows.date.dt.day_name().str[:3],
                        "market_index (day mean)": rows.market_index.round(3), "predicted_rate": pred.round(2)})
    out = pd.concat([out, drivers(model, rows.date).round(2)], axis=1)
    # tree part in $: load-level price before the stage-2 weekday/ramp effects. The load features are
    # identical on all 31 days, so any movement here comes from date-level inputs (daily mean quote).
    out["tree part $"] = (np.exp(tree) * 360).round(2)
    coef = (100 * (np.exp(model.calendar_coef_) - 1)).round(2)
    md = ["# December predictions and drivers", "",
          f"Range ${pred.min():.0f}..${pred.max():.0f} (mean ${pred.mean():.0f}, max/min {pred.max() / pred.min():.3f}, "
          f"day-to-day std ${np.diff(pred).std():.2f}).",
          f"Tree (load-level) part: ${out['tree part $'].min():.2f}..${out['tree part $'].max():.2f} across days "
          "(moves only through the daily mean quote_signal).",
          "Stage-2 effects, relative to a Monday outside the quarter-end window:", "",
          coef.to_frame("effect %").to_markdown(), "", out.to_markdown(index=False), ""]
    out.to_csv(METRICS / "december_drivers.csv", index=False)
    import json
    tree_dollars = np.exp(tree) * 360
    (METRICS / "december_summary.json").write_text(json.dumps({
        "min": float(pred.min()), "max": float(pred.max()), "mean": float(pred.mean()),
        "day_to_day_std": float(np.diff(pred).std()),
        "tree_min": float(tree_dollars.min()), "tree_max": float(tree_dollars.max()),
        "stage2_pct": {k: float(v) for k, v in coef.items()},
        "quote_fill": float(rows.quote_signal.iloc[0]),
        "qs_day_min": float(model.market.qs_day.reindex(rows.date).min()),
        "qs_day_max": float(model.market.qs_day.reindex(rows.date).max()),
    }, indent=2), encoding="utf-8")
    (REPORTS / "december_drivers.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    setup()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(rows.date, pred, color=SERIES[0], marker="o", ms=4)
    ax[0].set(title="December 2025: Lexington > Fort Wayne, 360 mi, Dry Van, 32,000 lb", ylabel="predicted rate ($)")
    for i, g in enumerate(["weekday %", "quarter-end ramp %"]):
        ax[1].plot(rows.date, out[g], color=SERIES[i], lw=1.8, label=g.replace(" %", ""))
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].set(title="Drivers: % effect vs a Monday outside the quarter-end window", ylabel="%")
    ax[1].legend(loc="upper left", ncol=2)
    ax[1].tick_params(axis="x", rotation=30)
    save(fig, "11_december_preview.png")
    return out


SMOOTH_VARIANTS = ["A34 A27 + recency half-life 60d", "A40 A39 + recency half-life 60d",
                   "A37 A24 + qs_7d (smoothed quote regime)", "A41 A24 + qs_14d (continuous)",
                   "A42 A24 + qs_7d + quote x regime", "A43 A24 + qs_14d + quote x regime",
                   "A44 A42 + recency half-life 60d", "A45 A43 + recency half-life 60d",
                   "A46 A41 + recency half-life 60d"]


def variant_stats():
    """Fit each quote-regime variant on all development rows and describe its December curve,
    next to its CV accuracy. Writes reports/smooth_variants.md. (~5 min)"""
    from .model import RateModel
    from .validate import EXPERIMENTS
    train_raw, valid_raw = load_raw()
    train, _ = fix_features(train_raw)
    valid, _ = fix_features(valid_raw)
    market = daily_market(train, valid)
    rows = prepare_december(load_december(), city_coordinates(train_raw, valid_raw), market, december_quote_fill(train))
    cv = {}
    for scheme in ("time", "city", "random"):
        d = pd.read_csv(ARTIFACTS / f"cv_{scheme}.csv")
        d = d[~d.corrupted & d.model.isin(SMOOTH_VARIANTS)]
        per_fold = d.assign(ae=(d.pred - d.y).abs()).groupby(["model", "fold"]).ae.mean().unstack()
        cv[scheme] = per_fold
    out = []
    for name in SMOOTH_VARIANTS:
        m = RateModel(market, **EXPERIMENTS[name]).fit(train)
        p = m.predict(rows)
        tree = np.exp(m.contributions(rows).drop(columns=["calendar (stage 2)", "trend", "level"]).sum(axis=1)) * 360
        t = cv["time"].loc[name]
        out.append({"model": name, "T1": t.iloc[0], "T2": t.iloc[1], "T3": t.iloc[2], "time mean": t.mean(),
                    "time worst": t.max(), "city": cv["city"].loc[name].mean(), "random": cv["random"].loc[name].mean(),
                    "Dec min $": p.min(), "Dec max $": p.max(), "Dec day-to-day std $": np.diff(p).std(),
                    "tree part range $": tree.max() - tree.min(),
                    "stage-2 Thu vs Mon %": 100 * (np.exp(m.calendar_coef_["dow_Thu"]) - 1),
                    "stage-2 quarter-end ramp %": 100 * (np.exp(m.calendar_coef_["quarter_end_ramp"]) - 1)})
    t = pd.DataFrame(out).round(2)
    METRICS.mkdir(parents=True, exist_ok=True)
    t.to_csv(METRICS / "smooth_variants.csv", index=False)
    md = ["# Quote-regime variants: accuracy vs December smoothness", "",
          "Time/city/random = MAE ($, clean labels). 'tree part range' = how much the load-level tree moves "
          "across the 31 December days (0 = all movement comes from the stage-2 weekday/ramp model).", "",
          t.to_markdown(index=False), ""]
    (REPORTS / "smooth_variants.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    import sys
    variant_stats() if "--variants" in sys.argv else main()
