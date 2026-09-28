"""Simple reference models. Same fit(train)/predict(df) contract as RateModel.

They receive training rows that were already cleaned (corrupted labels removed,
weight imputed) with the fold's own statistics.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import DIST_BANDS


def _rpm(df):
    return df.posted_rate / df.distance


class GlobalRPM:
    """Global median rate-per-mile x distance."""
    def fit(self, train):
        self.rpm_ = _rpm(train).median()
        return self

    def predict(self, df):
        return self.rpm_ * df.distance.to_numpy()


class LaneMedian:
    """Lane x equipment median rpm -> lane median (x equipment ratio) -> equipment x distance band."""
    MIN = 3

    def fit(self, train):
        t = train.assign(rpm=_rpm(train), band=pd.cut(train.distance, DIST_BANDS, labels=False))
        self.eq_ratio_ = t.groupby("equipment").rpm.median() / t.rpm.median()
        t["rpm_norm"] = t.rpm / t.equipment.map(self.eq_ratio_)
        g = t.groupby(["pickup", "delivery", "equipment"]).rpm
        self.lane_eq_ = g.median()[g.size() >= self.MIN]
        g = t.groupby(["pickup", "delivery"]).rpm_norm
        self.lane_ = g.median()[g.size() >= self.MIN]
        self.band_ = t.groupby(["equipment", "band"]).rpm.median()
        return self

    def predict(self, df):
        d = df.assign(band=pd.cut(df.distance, DIST_BANDS, labels=False))
        k3 = pd.MultiIndex.from_frame(d[["pickup", "delivery", "equipment"]])
        k2 = pd.MultiIndex.from_frame(d[["pickup", "delivery"]])
        kb = pd.MultiIndex.from_frame(d[["equipment", "band"]])
        rpm = self.lane_eq_.reindex(k3).to_numpy()
        lane = self.lane_.reindex(k2).to_numpy() * d.equipment.map(self.eq_ratio_).to_numpy()
        rpm = np.where(np.isnan(rpm), lane, rpm)
        rpm = np.where(np.isnan(rpm), self.band_.reindex(kb).to_numpy(), rpm)
        return rpm * d.distance.to_numpy()


class RidgeLogRPM:
    """Ridge regression on log rate-per-mile with one-hot cities/equipment/weekday."""
    CAT = ["pickup", "delivery", "equipment", "dow", "band"]
    NUM = ["log_distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
           "market_index", "weight"]

    def _frame(self, df):
        return pd.DataFrame({
            "pickup": df.pickup.to_numpy(), "delivery": df.delivery.to_numpy(),
            "equipment": df.equipment.to_numpy(), "dow": pd.to_datetime(df.date).dt.dayofweek.to_numpy(),
            "band": pd.cut(df.distance, DIST_BANDS, labels=False).to_numpy(),
            "log_distance": np.log(df.distance.to_numpy()),
            **{c: df[c].to_numpy() for c in self.NUM if c != "log_distance"},
        })

    def fit(self, train):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), self.CAT),
                                 ("num", StandardScaler(), self.NUM)])
        self.model_ = make_pipeline(pre, Ridge(alpha=1.0)).fit(self._frame(train), np.log(_rpm(train)))
        return self

    def predict(self, df):
        return np.exp(self.model_.predict(self._frame(df))) * df.distance.to_numpy()
