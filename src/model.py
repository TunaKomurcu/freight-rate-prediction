"""The rate model as a single fit/predict object.

`RateModel.fit(train_rows)` learns EVERYTHING that depends on data from `train_rows` only:
weight-imputation medians, corrupted-label filter, target encodings (+ their priors and
the unseen-city blanking), the optional time trend, the GBM and the optional smearing
factor. `predict(rows)` only applies what was learned. The CV loop builds a fresh
RateModel per fold, so no statistic can leak from a test period or held-out city.

Inputs to both methods are frames that already went through the stateless
`clean.fix_features`. `market` is the daily market table (feature-only, see features.py).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .clean import WeightImputer, find_corrupted_rates
from .config import SEED
from .features import TargetEncoder, add_keys, base_features

T0 = pd.Timestamp("2025-01-01")

BASE_PARAMS = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=SEED, verbose=-1, n_jobs=8)  # >8 threads oversubscribes on ~40k rows

HOLIDAY_COLS = ["is_holiday", "days_from_holiday"]
SLOW_MARKET_COLS = ["market_index", "mi_day", "mi_7d", "mi_28d"]
# date-level features the tree does NOT see when calendar="additive" (handled by stage 2)
DATE_COLS = ["dow", "is_weekend", "day_of_month", "days_to_month_end", "days_to_quarter_end",
             "days_since_quarter_start", "is_holiday", "days_from_holiday", "mi_cycle"]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
QUARTER_RAMP_DAYS = 21

DEFAULT_CONFIG = dict(
    target="log_rpm",        # log_rpm | rpm | rate
    objective="l2",          # l2 | huber | l1
    clean_labels=True,       # drop corrupted training labels (refit per fold)
    quote_signal=True,       # True (raw) | "with_regime" (raw + coarse quote regime) | "with_day" / "with_7d"
                             #   (raw + daily / trailing 7-day mean quote) | "dev" | False
    market_features="all",   # "all" | "cycle": drop slow market-level features, keep weekly cycle + per-load deviation
    holidays=True,
    trend=False,             # False | "linear" | "linear_q": linear time trend removed from the target
                             #   ("linear_q" controls for the quarter-end ramp when estimating the slope)
    level_window=0,          # >0: shift predictions by the mean out-of-time residual of the last N training days
    half_life=0,             # >0: recency sample weights, halving every N days back
    calendar_market=False,   # additive only: add log(daily market index) to the stage-2 date model
    calendar="gbm",          # "gbm": date features inside the tree | "additive": tree without date features +
                             #   a ridge model of weekday + quarter-end ramp fitted on the tree's daily residuals
    unseen_rate=0.12,        # share of training rows with one city blanked in the encodings
    smearing=False,          # Duan smearing factor on the exp back-transform
    params={},
)


# Chosen in Phase 3 on the rolling-origin time folds (reports/validation_results.md) by worst-fold
# MAE among configurations whose December curve is smooth:
# * slow market-level features dropped: their link to rates was confounded with the low
#   Jan-Feb level and caused -3..-5% bias two months out;
# * date effects moved out of the tree into an additive stage-2 model (weekday + quarter-end
#   ramp), fitted on date-grouped out-of-fold residuals -> smooth, explainable daily movement;
# * coarse quote regime added: the quote-price relation flips sign between regimes; a continuous
#   daily quote mean is more accurate in CV but acts as a date ID and makes December erratic;
# * recency weights (60-day half-life): lower worst-fold MAE than without;
# * holiday features, time trend, level offsets and a ridge blend were tested and rejected.
FINAL_CONFIG = dict(market_features="cycle", holidays=False, calendar="additive",
                    quote_signal="with_regime", half_life=60)


def days(dates):
    return (pd.to_datetime(dates) - T0).dt.days.to_numpy().astype(float)


class RateModel:
    def __init__(self, market, **config):
        self.market = market
        self.cfg = {**DEFAULT_CONFIG, **config}

    # ------------------------------------------------------------ helpers
    def _columns(self, X):
        qs = self.cfg["quote_signal"]
        extra = ["qs_dev", "qs_day", "qs_7d", "qs_regime"]
        keep = {True: ["quote_signal"], "with_day": ["quote_signal", "qs_day"], "with_7d": ["quote_signal", "qs_7d"],
                "with_regime": ["quote_signal", "qs_regime"], "dev": ["qs_dev"], False: []}[qs]
        drop = [c for c in ["quote_signal"] + extra if c not in keep]
        if self.cfg["market_features"] == "cycle":
            drop += SLOW_MARKET_COLS
        else:
            drop += ["mi_cycle"]
        if not self.cfg["holidays"]:
            drop += HOLIDAY_COLS
        if self.cfg["calendar"] == "additive":
            drop += DATE_COLS
        return [c for c in X.columns if c not in drop]

    def _fit_trend(self, df, y, enc):
        """Slope of the daily mean residual (after lane/equipment/distance priors) on time,
        controlling for the de-noised market index so the slope does not absorb market moves."""
        resid = y - enc.te_prior.to_numpy() - enc.te_lane.to_numpy()
        dates = pd.to_datetime(df.date)
        to_q_end = (dates.dt.to_period("Q").dt.end_time.dt.normalize() - dates).dt.days.to_numpy()
        d = pd.DataFrame({"t": days(df.date), "r": resid,
                          "mi": np.log(self.market.mi_day.reindex(dates).to_numpy()),
                          # 0 most of the quarter, rising linearly to 1 on its last day
                          "ramp": np.clip(1 - to_q_end / QUARTER_RAMP_DAYS, 0, 1)})
        daily = d.groupby("t").mean().reset_index()
        cols = [np.ones(len(daily)), daily.t, daily.mi]
        if self.cfg["trend"] == "linear_q":
            cols.append(daily.ramp)
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, daily.r.to_numpy(), rcond=None)
        return coef[1]

    def _date_design(self, dates):
        """Stage-2 design per calendar day: weekday dummies (Mon = reference) + quarter-end ramp
        (+ log daily market index if calendar_market)."""
        dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        to_q_end = (dates.dt.to_period("Q").dt.end_time.dt.normalize() - dates).dt.days.to_numpy()
        D = pd.DataFrame({f"dow_{n}": (dates.dt.dayofweek == i).astype(float) for i, n in enumerate(DOW_NAMES) if i})
        D["quarter_end_ramp"] = np.clip(1 - to_q_end / QUARTER_RAMP_DAYS, 0, 1)
        if self.cfg["calendar_market"]:
            D["log_market_index"] = np.log(self.market.mi_day.reindex(dates).to_numpy())
        return D

    def _fit_calendar(self, dates, resid, weights=None):
        """Daily mean residual ~ weekday + quarter-end ramp (weighted by loads per day x recency)."""
        d = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "r": resid,
                          "w": 1.0 if weights is None else weights})
        daily = d.groupby("date").agg(mean=("r", "mean"), size=("w", "sum")).reset_index()
        D = self._date_design(daily.date)
        self.calendar_ = Ridge(alpha=1.0).fit(D, daily["mean"], sample_weight=daily["size"])
        self.calendar_coef_ = pd.Series(self.calendar_.coef_, index=D.columns)
        # centre on the average training day so stage 2 moves prices around the tree's level
        self.calendar_center_ = float(np.average(self.calendar_.predict(D), weights=daily["size"]))

    def _calendar_effect(self, dates):
        if self.calendar_ is None:
            return 0.0
        return self.calendar_.predict(self._date_design(dates)) - self.calendar_center_

    def _target(self, df):
        rpm = df.posted_rate.to_numpy() / df.distance.to_numpy()
        return np.log(rpm)

    def _to_rate(self, pred, df):
        if self.cfg["target"] == "log_rpm":
            return np.exp(pred) * df.distance.to_numpy()
        if self.cfg["target"] == "rpm":
            return pred * df.distance.to_numpy()
        return pred

    def _gbm(self):
        params = {**BASE_PARAMS, **self.cfg["params"]}
        obj = self.cfg["objective"]
        params["objective"] = {"l2": "regression", "huber": "huber", "l1": "regression_l1"}[obj]
        if obj == "huber":
            params.setdefault("alpha", 0.1 if self.cfg["target"] == "log_rpm" else 1.0)
        return lgb.LGBMRegressor(**params)

    def _design(self, df, enc):
        return pd.concat([base_features(df, self.market), enc.reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------ fit / predict
    def fit(self, train, corrupted=None):
        """`corrupted`: optional precomputed mask from find_corrupted_rates(train) (same rows),
        passed by the CV loop so several configs on one fold share the computation."""
        train = train.reset_index(drop=True)
        self.imputer_ = WeightImputer().fit(train)
        train = self.imputer_.transform(train)
        if self.cfg["clean_labels"]:
            bad = find_corrupted_rates(train)[0] if corrupted is None else pd.Series(np.asarray(corrupted))
            train = train[~bad.to_numpy()].reset_index(drop=True)
        self.n_train_ = len(train)

        y = self._target(train)
        keys = add_keys(train)
        self.trend_ = None
        if self.cfg["trend"]:
            # encodings on the raw target only to estimate the slope, then refit on detrended y
            probe = TargetEncoder(unseen_rate=0).fit(keys, pd.Series(y)).transform(keys)
            self.trend_ = self._fit_trend(train, y, probe)
            y = y - self.trend_ * days(train.date)

        self.encoder_ = TargetEncoder(unseen_rate=self.cfg["unseen_rate"])
        enc = self.encoder_.fit_transform(keys, pd.Series(y))  # OOF + blanking: training rows only
        X = self._design(train, enc)
        self.columns_ = self._columns(X)

        fit_y = y
        if self.cfg["target"] == "rpm":
            fit_y = np.exp(y)
        elif self.cfg["target"] == "rate":
            fit_y = np.exp(y) * train.distance.to_numpy()
        t = days(train.date)
        w = None
        if self.cfg["half_life"]:
            # recency weighting: a load's weight halves every `half_life` days back from the fold's last day
            w = 0.5 ** ((t.max() - t) / self.cfg["half_life"])
        Xc = X[self.columns_]
        self.gbm_ = self._gbm().fit(Xc, fit_y, sample_weight=w)

        def oof_by_date():
            # folds grouped by DATE: held-out days are unseen, so any day-level signal the tree could
            # only get by memorising dates stays in the residual for stage 2 to estimate
            oof = np.zeros(len(Xc))
            for a, b in GroupKFold(3, shuffle=True, random_state=SEED).split(Xc, groups=t):
                oof[b] = self._gbm().fit(Xc.iloc[a], fit_y[a], sample_weight=None if w is None else w[a]).predict(Xc.iloc[b])
            return oof

        self.calendar_ = None
        oof = None
        if self.cfg["calendar"] == "additive":
            # Stage 2 is fitted on OUT-OF-FOLD tree residuals: in-sample residuals are shrunk
            # because the tree partly fits the day effects through other features/noise.
            oof = oof_by_date()
            self._fit_calendar(train.date, fit_y - oof, None if w is None else w)

        self.level_ = 0.0
        if self.cfg["level_window"] and self.cfg["target"] == "log_rpm":
            # Local-level offset = mean out-of-time residual of the last N days: a tree fitted on the
            # rows BEFORE the window predicts the window (the same situation as forecasting ahead);
            # its mean error, net of the stage-2 calendar effect, is the recent level shift.
            recent = t > t.max() - self.cfg["level_window"]
            past = self._gbm().fit(Xc[~recent], fit_y[~recent], sample_weight=None if w is None else w[~recent])
            resid = fit_y[recent] - past.predict(Xc[recent])
            if self.calendar_ is not None:
                resid = resid - self._calendar_effect(train.date[recent])
            self.level_ = float(np.mean(resid))

        self.smear_ = 1.0
        if self.cfg["smearing"] and self.cfg["target"] == "log_rpm":
            # Duan smearing from out-of-fold residuals (in-sample GBM residuals are too small)
            oof = oof_by_date() if oof is None else oof
            self.smear_ = float(np.mean(np.exp(fit_y - oof)))
        return self

    def predict(self, df):
        df = self.imputer_.transform(df.reset_index(drop=True))
        enc = self.encoder_.transform(add_keys(df))  # full-fit encodings, no blanking
        X = self._design(df, enc)
        pred = self.gbm_.predict(X[self.columns_])
        if self.cfg["target"] == "log_rpm":
            if self.trend_ is not None:
                pred = pred + self.trend_ * days(df.date)
            pred = pred + self.level_ + self._calendar_effect(df.date)
            return np.exp(pred) * df.distance.to_numpy() * self.smear_
        if self.trend_ is not None:
            pred = pred * np.exp(self.trend_ * days(df.date))
        return self._to_rate(pred, df)

    def feature_importance(self):
        return pd.Series(self.gbm_.booster_.feature_importance("gain"), index=self.columns_).sort_values(ascending=False)

    def contributions(self, df):
        """Per-feature additive contributions to the log-rpm prediction (LightGBM SHAP values).
        Only meaningful for the log_rpm target. Adds the trend term as its own column."""
        df = self.imputer_.transform(df.reset_index(drop=True))
        X = self._design(df, self.encoder_.transform(add_keys(df)))[self.columns_]
        c = pd.DataFrame(self.gbm_.booster_.predict(X, pred_contrib=True), columns=self.columns_ + ["bias"])
        c["trend"] = self.trend_ * days(df.date) if self.trend_ is not None else 0.0
        c["level"] = self.level_
        c["calendar (stage 2)"] = self._calendar_effect(df.date)
        return c
