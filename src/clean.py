"""Data cleaning, split into stateless fixes and fitted steps.

Stateless (`fix_features`): row-level rules that need no statistics learned from the target
or from other loads' labels. Safe to apply once to every file.

Fitted (must be refit inside every CV fold, on that fold's training rows only):
  * `WeightImputer`         - medians used to fill missing weight
  * `find_corrupted_rates`  - out-of-fold robust model that flags corrupted targets

Validation rows are never dropped; only training rows with a corrupted target are.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import SEED

# Plausible band for posted rate / expected rate. Clean rows sit within ~0.9-1.1;
# corrupted ones are off by x2 or more, so any cut inside the empty gap works.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.7
WEIGHT_CAP, WEIGHT_FLOOR = 47_500, 5_000
MIN_LANE_LOADS_FOR_WEIGHT = 5


# ---------------------------------------------------------------- stateless rules
def fix_features(df):
    """R1-R3, R5, R6. Returns (fixed_df, {rule: rows_affected})."""
    df = df.copy()
    counts = {}

    before = df[["pickup", "delivery", "equipment"]].copy()
    for c in ("pickup", "delivery"):
        df[c] = df[c].str.strip().str.title()
    df["equipment"] = df["equipment"].str.strip().str.title()
    counts["R1 city/equipment spelling"] = int((before != df[["pickup", "delivery", "equipment"]]).any(axis=1).sum())

    # Negative weights are sign flips: |weight| has the same rate relationship as positive weights.
    counts["R2 negative weight"] = int((df.weight < 0).sum())
    df["weight"] = df.weight.abs()

    # Values at the cap/floor are censored: kept, but flagged so the model can treat them apart.
    df["weight_capped"] = (df.weight >= WEIGHT_CAP).astype(int)
    df["weight_floored"] = (df.weight <= WEIGHT_FLOOR).astype(int)
    counts["R3 weight at cap/floor"] = int((df.weight_capped + df.weight_floored).sum())

    # market_index = daily market value + small per-load noise, so the same-date mean of the
    # *feature* is a near-exact fill. Uses no target information.
    df["market_index_missing"] = df.market_index.isna().astype(int)
    df["market_index"] = df.market_index.fillna(df.groupby("date").market_index.transform("mean"))
    counts["R5 missing market_index"] = int(df.market_index_missing.sum())

    clipped = ((df[["pickup_lon", "delivery_lon"]] == -69.5).any(axis=1)
               | (df[["pickup_lat", "delivery_lat"]] == 25.5).any(axis=1))
    counts["R6 clipped coordinates (kept)"] = int(clipped.sum())

    df["weight_missing"] = df.weight.isna().astype(int)
    counts["R4 missing weight"] = int(df.weight_missing.sum())
    return df, counts


# ---------------------------------------------------------------- fitted steps
class WeightImputer:
    """Missing weight -> lane x equipment median (if >= 5 loads) else equipment median."""

    def fit(self, train):
        g = train.groupby(["pickup", "delivery", "equipment"]).weight
        self.lane_eq_ = g.median()[g.count() >= MIN_LANE_LOADS_FOR_WEIGHT]
        self.eq_ = train.groupby("equipment").weight.median()
        return self

    def transform(self, df):
        df = df.copy()
        key = pd.MultiIndex.from_frame(df[["pickup", "delivery", "equipment"]])
        by_lane = pd.Series(self.lane_eq_.reindex(key).to_numpy(), index=df.index)
        df["weight"] = df.weight.fillna(by_lane).fillna(df.equipment.map(self.eq_))
        return df


def expected_rate_oof(train, seed=SEED):
    """Out-of-fold expected rate per row from a robust (L1) GBM on log rate-per-mile.

    Deliberately uses no lane/city identifiers: the expectation comes from distance,
    coordinates, equipment, date and market index, so a lane with 1-2 loads (whose own
    median is meaningless) still gets a sensible reference value.
    """
    X = pd.DataFrame({
        "log_distance": np.log(train.distance),
        "equipment": train.equipment.astype("category").cat.codes,
        "pickup_lat": train.pickup_lat, "pickup_lon": train.pickup_lon,
        "delivery_lat": train.delivery_lat, "delivery_lon": train.delivery_lon,
        "day": (train.date - pd.Timestamp("2025-01-01")).dt.days,
        "market_index": train.market_index, "weight": train.weight,
    }).reset_index(drop=True)
    y = np.log(train.posted_rate.to_numpy() / train.distance.to_numpy())
    oof = np.zeros(len(train))
    params = dict(objective="l1", n_estimators=250, learning_rate=0.1, num_leaves=63,
                  min_child_samples=20, random_state=seed, verbose=-1, n_jobs=8)
    for fit_idx, pred_idx in KFold(5, shuffle=True, random_state=seed).split(X):
        m = lgb.LGBMRegressor(**params).fit(X.iloc[fit_idx], y[fit_idx])
        oof[pred_idx] = m.predict(X.iloc[pred_idx])
    return np.exp(oof) * train.distance.to_numpy()


def find_corrupted_rates(train):
    """Boolean mask (aligned to `train`) of rows whose posted_rate is x2+ away from expectation."""
    ratio = train.posted_rate.to_numpy() / expected_rate_oof(train)
    return pd.Series((ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH), index=train.index), ratio
