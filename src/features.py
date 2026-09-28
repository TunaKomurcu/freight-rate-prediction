"""Phase 2: feature engineering.

Two kinds of features:
  * `base_features(df, market)`: deterministic, target-free (geometry, calendar, weight,
    market index). Safe to compute on any row, including validation and December.
  * `TargetEncoder`: smoothed mean of the log rate-per-mile residual per lane / city /
    coarse region. Fit on training rows only; training rows get out-of-fold values so a
    row never sees its own target.

Unknown lanes/cities fall back gracefully: every encoding is shrunk toward 0 (= the
equipment x distance-band prior), so an unseen key simply gets the prior, while the
coarse-region encodings and raw coordinates still carry location information.
"""
import holidays
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import EQUIPMENT, SEED
from .data import haversine_miles

DIST_BANDS = [0, 250, 500, 750, 1000, 1500, 2000, 2500, np.inf]
REGION_DEG = 5.0  # coarse grid cell size (degrees) for region fallback
QS_REGIME_CUTS = [2.0, 2.1]  # weekly quote means cluster at ~1.93 / ~2.05 / ~2.2-2.3 (see EDA)
US_HOLIDAYS = holidays.US(years=[2024, 2025, 2026])


# ---------------------------------------------------------------- market index
def daily_market(*frames):
    """Daily feature aggregates across all supplied frames (never uses the target).

    Frames are concatenated and sorted by date (train + validation), so early-November
    rolling values use late-October days exactly as they would at prediction time.
    market_index is one market value per day plus per-load noise, so the daily mean is a
    de-noised version. December comes from validation.csv, where every day is present.
    qs_day (daily mean quote_signal) lets the model use a load's quote relative to its day.
    """
    rows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    g = rows.groupby("date")
    daily = g.market_index.mean().sort_index().asfreq("D").interpolate(limit_direction="both")
    qs = g.quote_signal.mean().sort_index().asfreq("D").interpolate(limit_direction="both")
    out = pd.DataFrame({"mi_day": daily, "mi_7d": daily.rolling(7, min_periods=1).mean(),
                        "mi_28d": daily.rolling(28, min_periods=1).mean(), "qs_day": qs,
                        # trailing 7-day mean: the quote regime lasts weeks, the daily mean is noisy
                        "qs_7d": qs.rolling(7, min_periods=1).mean(),
                        "qs_14d": qs.rolling(14, min_periods=1).mean()})
    # Coarse quote regime (0 low / 1 mid / 2 high) from the trailing 7-day mean. Coarse on purpose:
    # a continuous daily value is unique per date and lets the tree memorise day-level prices.
    out["qs_regime"] = np.digitize(out.qs_7d, QS_REGIME_CUTS)
    out["mi_cycle"] = out.mi_day - out.mi_28d  # weekly cycle around the slow market level
    return out


# ---------------------------------------------------------------- calendar
def calendar_features(dates):
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    q_end = d.dt.to_period("Q").dt.end_time.dt.normalize()
    q_start = d.dt.to_period("Q").dt.start_time
    hol = pd.to_datetime(sorted(US_HOLIDAYS.keys()))
    # signed distance (days) to the nearest US federal holiday, clipped to +-7
    diff = (d.values[:, None] - hol.values[None, :]).astype("timedelta64[D]").astype(int)
    nearest = diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]
    return pd.DataFrame({
        "dow": d.dt.dayofweek,
        "is_weekend": (d.dt.dayofweek >= 5).astype(int),
        "day_of_month": d.dt.day,
        "days_to_month_end": (d.dt.days_in_month - d.dt.day),
        "days_to_quarter_end": (q_end - d).dt.days,
        "days_since_quarter_start": (d - q_start).dt.days,
        "is_holiday": (nearest == 0).astype(int),
        "days_from_holiday": np.clip(nearest, -7, 7),
    })


# ---------------------------------------------------------------- base features
def region(lat, lon):
    return (np.floor(lat / REGION_DEG).astype(int).astype(str) + "_"
            + np.floor(lon / REGION_DEG).astype(int).astype(str))


def add_keys(df):
    """Categorical keys used by the target encoder (not model inputs themselves)."""
    out = df.copy()
    out["lane"] = out.pickup + ">" + out.delivery
    out["pickup_region"] = region(out.pickup_lat, out.pickup_lon)
    out["delivery_region"] = region(out.delivery_lat, out.delivery_lon)
    out["region_lane"] = out.pickup_region + ">" + out.delivery_region
    out["dist_band"] = pd.cut(out.distance, DIST_BANDS, labels=False)
    out["prior_key"] = out.equipment + "|" + out.dist_band.astype(str)
    return out


def base_features(df, market):
    hav = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    m = market.reindex(pd.to_datetime(df.date)).reset_index(drop=True)
    X = pd.DataFrame({
        "distance": df.distance.to_numpy(),
        "log_distance": np.log(df.distance.to_numpy()),
        "haversine": hav.to_numpy(),
        # road-to-crow ratio; huge values come from clipped/overlapping coordinates, so clip
        "circuity": np.clip(df.distance.to_numpy() / np.maximum(hav.to_numpy(), 1), 1, 3),
        "pickup_lat": df.pickup_lat.to_numpy(), "pickup_lon": df.pickup_lon.to_numpy(),
        "delivery_lat": df.delivery_lat.to_numpy(), "delivery_lon": df.delivery_lon.to_numpy(),
        "equipment": pd.Categorical(df.equipment, categories=EQUIPMENT).codes,
        "weight": df.weight.to_numpy(),
        "weight_capped": df.weight_capped.to_numpy(), "weight_floored": df.weight_floored.to_numpy(),
        "weight_missing": df.weight_missing.to_numpy(),
        "market_index": df.market_index.to_numpy(),
        "mi_day": m.mi_day.to_numpy(), "mi_7d": m.mi_7d.to_numpy(), "mi_28d": m.mi_28d.to_numpy(),
        "mi_cycle": m.mi_cycle.to_numpy(),
        "quote_signal": df.quote_signal.to_numpy(),
        "qs_dev": df.quote_signal.to_numpy() - m.qs_day.to_numpy(),
        # daily mean quote: identifies the regime in which quote_signal relates positively /
        # negatively / not at all to price (see EDA); known on the day like market_index
        "qs_day": m.qs_day.to_numpy(),
        "qs_7d": m.qs_7d.to_numpy(),
        "qs_14d": m.qs_14d.to_numpy(),
        "qs_regime": m.qs_regime.to_numpy(),
    })
    X["mi_dev"] = X.market_index - X.mi_day  # per-load deviation from the day's market
    return pd.concat([X, calendar_features(df.date)], axis=1)


# ---------------------------------------------------------------- target encoding
class TargetEncoder:
    """Smoothed residual encodings of log(rate per mile).

    prior(equipment, distance band) = mean log rpm of that cell.
    For each key k: enc_k = sum(residual) / (count + SMOOTH)  -> 0 (the prior) when unseen.
    `unseen_rate` simulates unseen cities while encoding training rows (see fit_transform).
    """

    KEYS = ["lane", "pickup", "delivery", "pickup_region", "delivery_region", "region_lane"]
    SMOOTH = 5.0

    def __init__(self, n_folds=5, unseen_rate=0.12, seed=SEED):
        self.n_folds, self.unseen_rate, self.seed = n_folds, unseen_rate, seed

    def _fit_stats(self, keys, y):
        prior = y.groupby(keys.prior_key).mean()
        resid = y - keys.prior_key.map(prior)
        stats = {k: resid.groupby(keys[k]).agg(["sum", "count"]) for k in self.KEYS}
        return prior, stats, y.mean()

    def _apply(self, keys, fitted):
        prior, stats, global_mean = fitted
        out = pd.DataFrame(index=keys.index)
        out["te_prior"] = keys.prior_key.map(prior).fillna(global_mean)
        for k in self.KEYS:
            s = stats[k].reindex(keys[k])
            out[f"te_{k}"] = (s["sum"] / (s["count"] + self.SMOOTH)).fillna(0).to_numpy()
            if k in ("lane", "pickup", "delivery"):
                out[f"n_{k}"] = s["count"].fillna(0).to_numpy()
        return out

    def fit(self, keys, y):
        self.fitted_ = self._fit_stats(keys, y)
        return self

    def transform(self, keys):
        return self._apply(keys, self.fitted_)

    def fit_transform(self, keys, y):
        """Out-of-fold encodings for the training rows, then fit on all of them.

        With ~12 loads per lane, plain OOF almost never produces an unseen lane/city, so a
        model trained on it would never learn what to do with one. We therefore blank one
        endpoint city (and the lane) for a random `unseen_rate` share of rows, matching the
        ~12% of validation loads that touch a city absent from training.
        """
        keys = keys.reset_index(drop=True)
        y = pd.Series(np.asarray(y), index=keys.index)
        out = []
        for fit_idx, enc_idx in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(keys):
            fitted = self._fit_stats(keys.iloc[fit_idx], y.iloc[fit_idx])
            out.append(self._apply(keys.iloc[enc_idx], fitted))
        enc = pd.concat(out).sort_index()
        if self.unseen_rate > 0:
            rng = np.random.default_rng(self.seed)
            mask = rng.random(len(enc)) < self.unseen_rate
            side = rng.random(len(enc)) < 0.5
            for city, rows in (("pickup", mask & side), ("delivery", mask & ~side)):
                enc.loc[rows, [f"te_{city}", f"n_{city}"]] = 0
            enc.loc[mask, ["te_lane", "n_lane"]] = 0
        self.fit(keys, y)
        return enc


def encode_pair(fit_df, apply_df, y_fit, unseen_rate=0.12):
    """Encodings for a (training, scoring) pair: OOF on training rows, full-fit on scoring rows."""
    te = TargetEncoder(unseen_rate=unseen_rate)
    fit_keys, apply_keys = add_keys(fit_df), add_keys(apply_df)
    return te.fit_transform(fit_keys, y_fit), te.transform(apply_keys).reset_index(drop=True)


# ---------------------------------------------------------------- December inputs
def city_coordinates(*frames):
    """One (lat, lon) per city, taken from the feature columns of the supplied frames."""
    pts = pd.concat([
        f[[f"{s}", f"{s}_lat", f"{s}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for f in frames for s in ("pickup", "delivery")])
    return pts.groupby("city")[["lat", "lon"]].median()


def prepare_december(dec, coords, market, quote_fill):
    """Turn the 7-column December chart file into model-ready rows.

    Missing columns are reconstructed (see report):
      * lat/lon: looked up per city from the delivered data (identical for every load of a city)
      * market_index: that day's mean market_index from validation.csv (all 31 days present)
      * quote_signal: median of comparable training loads (Dry Van, 300-420 mi), see predict.py
      * weight flags: 32,000 lb is neither capped, floored nor missing
    """
    out = dec.drop(columns="predicted_rate").copy()
    out["date"] = pd.to_datetime(out.date)
    for s in ("pickup", "delivery"):
        out[f"{s}_lat"] = out[s].map(coords.lat)
        out[f"{s}_lon"] = out[s].map(coords.lon)
    out["market_index"] = market.mi_day.reindex(out.date).to_numpy()
    out["quote_signal"] = quote_fill
    for c in ("weight_capped", "weight_floored", "weight_missing", "market_index_missing"):
        out[c] = 0
    if out[["pickup_lat", "delivery_lat", "market_index"]].isna().any().any():
        raise ValueError("December inputs could not be completed (unknown city or date)")
    return out
