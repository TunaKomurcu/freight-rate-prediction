"""Phase 2: reproducible cleaning.

Every rule is applied by `clean()` and logged with the number of rows it touched,
so the report can quote the exact numbers. Validation rows are never dropped:
feature problems are fixed or imputed and flagged; only training rows with a
corrupted *target* are removed.

Statistics used for imputation (weight medians) come from the training file only.
The market_index fill uses same-date means of the *feature* in whichever file
the row belongs to (no target involved).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import SEED
from .data import lane_relative_rate

# Plausible band for posted rate / expected rate. Clean rows sit within ~0.85-1.15;
# corrupted ones are off by x2 or more, so any cut inside the empty gap works.
OUTLIER_LOW, OUTLIER_HIGH = 0.6, 1.7
WEIGHT_CAP, WEIGHT_FLOOR = 47_500, 5_000
MIN_LANE_LOADS_FOR_WEIGHT = 5


def expected_rate_oof(train):
    """Out-of-fold expected rate per row from a robust (L1) GBM on log rate-per-mile.

    Deliberately uses no lane/city identifiers: the expectation comes from distance,
    coordinates, equipment, date and market index, so a lane with 1-2 loads (whose
    own median is meaningless) still gets a sensible reference value.
    """
    X = pd.DataFrame({
        "log_distance": np.log(train.distance),
        "equipment": train.equipment.astype("category").cat.codes,
        "pickup_lat": train.pickup_lat, "pickup_lon": train.pickup_lon,
        "delivery_lat": train.delivery_lat, "delivery_lon": train.delivery_lon,
        "day": (train.date - train.date.min()).dt.days,
        "market_index": train.market_index, "weight": train.weight.abs(),
    })
    y = np.log(train.posted_rate / train.distance)
    oof = np.zeros(len(train))
    params = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=63,
                  min_child_samples=20, random_state=SEED, verbose=-1)
    for fit_idx, pred_idx in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = lgb.LGBMRegressor(**params).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[pred_idx] = m.predict(X.iloc[pred_idx])
    return np.exp(oof) * train.distance


def fill_weight(df, weight_lookup):
    """Impute missing weight: lane x equipment median (if >= 5 loads) else equipment median."""
    lane_eq, eq = weight_lookup
    key = pd.MultiIndex.from_frame(df[["pickup", "delivery", "equipment"]])
    by_lane = pd.Series(lane_eq.reindex(key).to_numpy(), index=df.index)
    by_eq = df.equipment.map(eq)
    return df.weight.fillna(by_lane).fillna(by_eq)


def weight_lookup_from(train):
    w = train.assign(weight=train.weight.abs())
    g = w.groupby(["pickup", "delivery", "equipment"]).weight
    lane_eq = g.median()[g.count() >= MIN_LANE_LOADS_FOR_WEIGHT]
    return lane_eq, w.groupby("equipment").weight.median()


def clean(train, valid):
    """Return (train_clean, valid_clean, log) where log is a list of rule dicts."""
    train, valid = train.copy(), valid.copy()
    log = []

    def record(rule, action, n_train, n_valid):
        log.append({"rule": rule, "action": action, "train_rows": int(n_train), "valid_rows": int(n_valid)})

    # R1 text normalisation (defensive; the delivered files are already clean)
    n = {}
    for name, df in (("t", train), ("v", valid)):
        before = df[["pickup", "delivery", "equipment"]].copy()
        for c in ("pickup", "delivery"):
            df[c] = df[c].str.strip().str.title()
        df["equipment"] = df["equipment"].str.strip().str.title().replace({"Dryvan": "Dry Van"})
        n[name] = (before != df[["pickup", "delivery", "equipment"]]).any(axis=1).sum()
    record("R1 city/equipment spelling", "strip + title-case", n["t"], n["v"])

    # R2 negative weights are sign flips: |weight| has the same rate relationship as positive weights
    record("R2 negative weight", "take absolute value", (train.weight < 0).sum(), (valid.weight < 0).sum())
    for df in (train, valid):
        df["weight"] = df["weight"].abs()

    # R3 weight at the 47,500 cap / 5,000 floor: censored values, kept but flagged
    for df in (train, valid):
        df["weight_capped"] = (df.weight >= WEIGHT_CAP).astype(int)
        df["weight_floored"] = (df.weight <= WEIGHT_FLOOR).astype(int)
    record("R3 weight at cap (47,500) or floor (5,000)", "keep value, add flag column",
           (train.weight_capped + train.weight_floored).sum(), (valid.weight_capped + valid.weight_floored).sum())

    # R4 missing weight: impute from TRAINING medians
    lookup = weight_lookup_from(train)
    for df in (train, valid):
        df["weight_missing"] = df.weight.isna().astype(int)
        df["weight"] = fill_weight(df, lookup)
    record("R4 missing weight", "impute lane x equipment median (>=5 loads) else equipment median; flag",
           train.weight_missing.sum(), valid.weight_missing.sum())

    # R5 missing market_index: it is a daily series + small noise, so the same-date mean is near exact
    for df in (train, valid):
        df["market_index_missing"] = df.market_index.isna().astype(int)
        df["market_index"] = df.market_index.fillna(df.groupby("date").market_index.transform("mean"))
    record("R5 missing market_index", "fill with same-date mean of the feature; flag",
           train.market_index_missing.sum(), valid.market_index_missing.sum())

    # R6 clipped coordinates (Boston/Providence lon=-69.5, Laredo lat=25.5): cannot be recovered,
    # `distance` is used as the authoritative length, coordinates only as coarse location
    clip = lambda df: ((df[["pickup_lon", "delivery_lon"]] == -69.5).any(axis=1)
                       | (df[["pickup_lat", "delivery_lat"]] == 25.5).any(axis=1))
    record("R6 clipped coordinates", "keep; distance is authoritative", clip(train).sum(), clip(valid).sum())

    # R7 corrupted targets (train only): posted rate off by x2-x6 from the robust expectation
    expected = expected_rate_oof(train)
    ratio = train.posted_rate / expected
    bad = (ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH)
    lane_rel = lane_relative_rate(train)
    lane_bad = (lane_rel < OUTLIER_LOW) | (lane_rel > OUTLIER_HIGH)
    train["rate_ratio_expected"] = ratio
    record("R7 corrupted posted_rate (x2-x6 off)", "drop from training", bad.sum(), 0)
    agreement = {"model_flags": int(bad.sum()), "lane_median_flags": int(lane_bad.sum()),
                 "both": int((bad & lane_bad).sum())}
    train = train[~bad].reset_index(drop=True)
    return train, valid, log, agreement


def log_table(log):
    return pd.DataFrame(log)
