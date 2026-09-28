"""December chart preview + driver decomposition (run after src.train / src.predict).

Run: python -m src.december -> reports/december_drivers.md, reports/figures/11_december_preview.png
The day-to-day movement is split into additive log-space contributions (SHAP) and shown
as % of the month's average rate.
"""
import pickle

import numpy as np
import pandas as pd

from .clean import fix_features
from .config import ARTIFACTS, INK2, REPORTS, SERIES
from .data import load_december, load_raw
from .features import city_coordinates, prepare_december
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
    coef = (100 * (np.exp(model.calendar_coef_) - 1)).round(2)
    md = ["# December predictions and drivers", "",
          f"Range ${pred.min():.0f}..${pred.max():.0f} (mean ${pred.mean():.0f}, max/min {pred.max() / pred.min():.3f}).",
          f"Tree (load-level) part is identical for all 31 days: exp(sum SHAP) x 360 mi = "
          f"${np.exp(tree.iloc[0]) * 360:.2f} (spread across days: {tree.max() - tree.min():.2e}).",
          "Day-to-day movement = stage-2 effects, relative to a Monday outside the quarter-end window:", "",
          coef.to_frame("effect %").to_markdown(), "", out.to_markdown(index=False), ""]
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


if __name__ == "__main__":
    main()
