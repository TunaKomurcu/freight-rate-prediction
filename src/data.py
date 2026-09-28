"""Raw data loading and small geometry helpers."""
import numpy as np
import pandas as pd

from .config import DECEMBER_CSV, TRAIN_CSV, VALID_CSV

EARTH_RADIUS_MI = 3958.8


def load_raw():
    """Return (train, validation) exactly as delivered, with `date` parsed."""
    train = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    valid = pd.read_csv(VALID_CSV, parse_dates=["date"])
    return train, valid


def load_december():
    return pd.read_csv(DECEMBER_CSV)


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(h))


def lane_relative_rate(df):
    """Rate-per-mile divided by the median of its (lane, equipment) group.

    Used for EDA and for flagging corrupted targets: a clean row sits near 1.0.
    Groups with fewer than 3 loads fall back to the lane median across equipment,
    rescaled by the global equipment ratio.
    """
    rpm = df["posted_rate"] / df["distance"]
    lane = [df["pickup"], df["delivery"]]
    eq_ratio = rpm.groupby(df["equipment"]).transform("median") / rpm.median()
    grp = rpm.groupby(lane + [df["equipment"]])
    med = grp.transform("median")
    fallback = (rpm / eq_ratio).groupby(lane).transform("median") * eq_ratio
    med = med.where(grp.transform("size") >= 3, fallback)
    return rpm / med
