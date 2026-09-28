"""Phase 1: exploratory analysis + data-quality audit on the RAW files.

Writes figures to reports/figures/ and a numeric summary to reports/eda_summary.md.
Run: python -m src.eda
"""
import numpy as np
import pandas as pd

from .config import COLORS, EQUIPMENT, INK2, REPORTS, SERIES
from .data import haversine_miles, lane_relative_rate, load_december, load_raw
from .plotting import plt, save, setup

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def audit(train, valid):
    """Collect data-quality facts as (section, text) lines."""
    out = []
    add = lambda s: out.append(s)

    for name, df in (("train_test", train), ("validation", valid)):
        add(f"### {name} ({len(df):,} rows)")
        na = df.isna().sum()
        add(f"- Missing values: {', '.join(f'{c}={n}' for c, n in na[na > 0].items()) or 'none'}")
        add(f"- Duplicate rows: {df.duplicated().sum()}, duplicate load_id: {df.load_id.duplicated().sum()}, "
            f"duplicate rows ignoring load_id: {df.drop(columns='load_id').duplicated().sum()}")
        add(f"- Date range: {df.date.min().date()} .. {df.date.max().date()} ({df.date.nunique()} days, none unparseable)")
        add(f"- Equipment values: {df.equipment.value_counts().to_dict()}")
        cities = pd.concat([df.pickup, df.delivery])
        stripped = cities.str.strip().str.title()
        add(f"- Cities: {cities.nunique()} distinct; whitespace/case variants: {(stripped != cities).sum()}; "
            f"pickup == delivery: {(df.pickup == df.delivery).sum()}")
        w = df.weight
        add(f"- Weight: negative={int((w < 0).sum())}, zero={int((w == 0).sum())}, "
            f"at cap 47,500={int((w.abs() == 47500).sum())}, at floor 5,000={int((w.abs() == 5000).sum())}, "
            f"range {w.min():,.0f}..{w.max():,.0f}")
        hav = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
        ratio = df.distance / hav
        add(f"- distance / haversine: median {ratio.median():.3f}, p1 {ratio.quantile(.01):.3f}, "
            f"p99 {ratio.quantile(.99):.3f}, max {ratio.max():.1f}; rows with ratio > 2: {(ratio > 2).sum()}")
        add(f"- Coordinates: lat {df[['pickup_lat','delivery_lat']].min().min():.2f}..{df[['pickup_lat','delivery_lat']].max().max():.2f}, "
            f"lon {df[['pickup_lon','delivery_lon']].min().min():.2f}..{df[['pickup_lon','delivery_lon']].max().max():.2f}; "
            f"lat/lon swapped (lat<0 or lon>0): {int(((df.pickup_lat < 0) | (df.pickup_lon > 0)).sum())}")
        add(f"- Distance range {df.distance.min():.0f}..{df.distance.max():.0f} mi; non-positive: {(df.distance <= 0).sum()}")
        add(f"- market_index range {df.market_index.min():.3f}..{df.market_index.max():.3f}; "
            f"quote_signal range {df.quote_signal.min():.3f}..{df.quote_signal.max():.3f}")

    # city -> coordinate consistency (one coordinate pair per city across both files)
    pts = pd.concat([
        pd.DataFrame({"city": d[f"{s}"], "lat": d[f"{s}_lat"], "lon": d[f"{s}_lon"]})
        for d in (train, valid) for s in ("pickup", "delivery")])
    per_city = pts.groupby("city")[["lat", "lon"]].nunique()
    add("### Cross-file checks")
    add(f"- Cities with >1 coordinate pair: {int((per_city > 1).any(axis=1).sum())}")
    clipped = pts.loc[(pts.lon == -69.5) | (pts.lat == 25.5), "city"].unique()
    add(f"- Coordinates sitting exactly on a round bound (lon=-69.5 / lat=25.5, likely clipped): {sorted(clipped)}")
    new = sorted((set(valid.pickup) | set(valid.delivery)) - (set(train.pickup) | set(train.delivery)))
    has_new = valid.pickup.isin(new) | valid.delivery.isin(new)
    add(f"- Cities only in validation: {new} -> {has_new.sum():,} validation rows ({has_new.mean():.1%})")
    tl = set(zip(train.pickup, train.delivery))
    seen = pd.Series([l in tl for l in zip(valid.pickup, valid.delivery)])
    add(f"- Validation rows whose lane appears in train: {seen.mean():.1%}")
    add(f"- Lanes in train: {len(tl):,}, median loads per lane: {train.groupby(['pickup','delivery']).size().median():.0f}")

    rel = lane_relative_rate(train)
    add("### Target")
    add(f"- posted_rate: min {train.posted_rate.min():.2f}, median {train.posted_rate.median():.2f}, max {train.posted_rate.max():,.2f}; non-positive: {(train.posted_rate <= 0).sum()}")
    rpm = train.posted_rate / train.distance
    add(f"- rate per mile: p1 {rpm.quantile(.01):.2f}, median {rpm.median():.2f}, p99 {rpm.quantile(.99):.2f}, max {rpm.max():.2f}")
    add(f"- Lane-relative rate (rpm / lane-equipment median): p1 {rel.quantile(.01):.3f}, p99 {rel.quantile(.99):.3f}; "
        f"< 0.6: {(rel < 0.6).sum()}, > 1.7: {(rel > 1.7).sum()}, in (0.6,0.8)U(1.3,1.7): {((rel.between(.6,.8)) | (rel.between(1.3,1.7))).sum()}")

    ok = (rel > 0.6) & (rel < 1.7)
    add("### Leakage check: quote_signal")
    add(f"- corr(quote_signal, posted_rate) = {train.posted_rate.corr(train.quote_signal):.3f}; "
        f"corr(quote_signal, rate per mile) = {rpm.corr(train.quote_signal):.3f}; "
        f"corr(quote_signal, lane-relative rate, clean rows) = {rel[ok].corr(train.quote_signal[ok]):.3f}")
    add(f"- corr(quote_signal * distance, posted_rate) = {(train.quote_signal*train.distance).corr(train.posted_rate):.3f} "
        f"vs corr(distance, posted_rate) = {train.distance.corr(train.posted_rate):.3f}")
    add("### market_index")
    day = train.groupby("date").market_index
    add(f"- Within-day std {day.std().mean():.3f} vs total std {train.market_index.std():.3f}: "
        f"market_index is a daily market series plus small per-load noise")
    d = pd.DataFrame({"mi": day.mean(), "rel": rel[ok].groupby(train.date[ok]).mean()})
    add(f"- corr(daily mean market_index, daily mean lane-relative rate) = {d.mi.corr(d.rel):.3f}; "
        f"row-level corr = {train.market_index[ok].corr(rel[ok]):.3f}")
    return "\n".join(out), rel


def figures(train, valid, rel):
    setup()
    rpm = train.posted_rate / train.distance
    ok = (rel > 0.6) & (rel < 1.7)

    # 1. Target distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(train.posted_rate, bins=120, color=SERIES[0])
    ax[0].set(title="posted_rate", xlabel="$ per load", ylabel="loads")
    ax[1].hist(np.log10(rpm), bins=150, color=SERIES[0])
    ax[1].set(title="Rate per mile (log10 scale)", xlabel="log10($ / mile)")
    ax[1].set_yscale("log")
    save(fig, "01_target_distribution.png")

    # 2. Rate per mile vs distance by equipment (binned medians)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 3600, 150)
    for eq in EQUIPMENT:
        m = train.equipment.eq(eq) & ok
        g = rpm[m].groupby(pd.cut(train.distance[m], bins), observed=True).median()
        ax.plot([b.mid for b in g.index], g.values, color=COLORS[eq], marker="o", ms=4, label=eq)
    ax.set(title="Rate per mile falls with distance; Reefer > Flatbed > Dry Van",
           xlabel="distance (mi)", ylabel="median $ / mile")
    ax.legend()
    save(fig, "02_rpm_vs_distance.png")

    # 3. Time: daily market index (train + validation) and daily rate index (train) — two panels, one axis each
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for df, c, lab in ((train, SERIES[0], "train_test"), (valid, SERIES[1], "validation")):
        s = df.groupby("date").market_index.mean()
        ax[0].plot(s.index, s.values, color=c, lw=1.2, label=lab)
    ax[0].set(title="Daily mean market_index (strong weekly cycle + seasonal swell)", ylabel="market_index")
    ax[0].legend(loc="upper right")
    r = rel[ok].groupby(train.date[ok]).mean()
    ax[1].plot(r.index, r.values, color=SERIES[0], lw=1)
    ax[1].plot(r.index, r.rolling(14, center=True).mean(), color=INK2, lw=2, label="14-day mean")
    ax[1].set(title="Daily mean lane-relative rate (train only)", ylabel="rate / lane median")
    ax[1].legend(loc="upper left")
    save(fig, "03_time_series.png")

    # 4. Day of week
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    dow = train.date.dt.dayofweek
    ax[0].bar(DOW, rel[ok].groupby(dow[ok]).mean().values, color=SERIES[0], width=0.6)
    ax[0].set(title="Lane-relative rate by weekday", ylim=(0.98, 1.02))
    ax[1].bar(DOW, train.groupby(dow).market_index.mean().values, color=SERIES[0], width=0.6)
    ax[1].set(title="market_index by weekday", ylim=(0.9, 1.2))
    save(fig, "04_day_of_week.png")

    # 5. Month by equipment
    fig, ax = plt.subplots(figsize=(7, 3.4))
    mon = train.date.dt.month
    for eq in EQUIPMENT:
        m = train.equipment.eq(eq) & ok
        g = rel[m].groupby(mon[m]).mean()
        ax.plot(g.index, g.values, color=COLORS[eq], marker="o", ms=4, label=eq)
    ax.set(title="Lane-relative rate by month", xlabel="month (2025)", ylabel="rate / lane median")
    ax.legend()
    save(fig, "05_month_equipment.png")

    # 6. Weight
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(train.weight.dropna(), bins=120, color=SERIES[0])
    ax[0].set(title="Weight: negatives and a spike at the 47,500 cap", xlabel="lb")
    wb = pd.cut(train.weight.abs(), np.arange(0, 50001, 5000))
    g = rel[ok].groupby(wb[ok], observed=True).mean()
    ax[1].plot([b.mid for b in g.index], g.values, color=SERIES[0], marker="o", ms=4)
    ax[1].set(title="Lane-relative rate vs |weight|", xlabel="|weight| (lb)")
    save(fig, "06_weight.png")

    # 7. quote_signal leakage check
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    s = train.sample(6000, random_state=0).index
    ax[0].scatter(train.quote_signal[s], rpm[s], s=4, alpha=.3, color=SERIES[0])
    ax[0].set(title="quote_signal vs rate per mile", xlabel="quote_signal", ylabel="$ / mile", ylim=(1, 4))
    s = s[ok[s]]
    ax[1].scatter(train.quote_signal[s], rel[s], s=4, alpha=.3, color=SERIES[0])
    ax[1].set(title="quote_signal vs lane-relative rate", xlabel="quote_signal", ylabel="rate / lane median")
    save(fig, "07_quote_signal.png")

    # 8. Data quality: target corruption + distance consistency
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(np.log2(rel), bins=150, color=SERIES[0])
    ax[0].set_yscale("log")
    ax[0].set(title="log2(rate / lane median): corrupted tails", xlabel="log2 ratio")
    for x in (np.log2(0.6), np.log2(1.7)):
        ax[0].axvline(x, color=INK2, ls="--", lw=1)
    hav = haversine_miles(train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon)
    ax[1].scatter(hav[::5], train.distance[::5], s=3, alpha=.3, color=SERIES[0])
    ax[1].set(title="distance vs haversine (road factor ~1.18)", xlabel="haversine (mi)", ylabel="distance (mi)")
    save(fig, "08_data_quality.png")

    # 9. Coverage: loads per day in each file
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for df, c, lab in ((train, SERIES[0], "train_test"), (valid, SERIES[1], "validation")):
        s = df.groupby("date").size()
        ax.plot(s.index, s.values, color=c, lw=1.2, label=lab)
    ax.set(title="Loads per day: validation is the two months after training", ylabel="loads")
    ax.legend()
    save(fig, "09_coverage.png")


def main():
    train, valid = load_raw()
    text, rel = audit(train, valid)
    dec = load_december()
    text += (f"\n### December chart inputs\n- Columns: {list(dec.columns)}; dates {dec.date.min()}..{dec.date.max()}; "
             f"no lat/lon, market_index or quote_signal\n")
    figures(train, valid, rel)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eda_summary.md").write_text("# EDA and data-quality summary (raw data)\n\n" + text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
