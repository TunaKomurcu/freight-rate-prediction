"""Phase 2 entry point: clean the data, build December inputs, document every rule.

Run: python -m src.prepare
Writes artifacts/{train,valid,december}_clean.csv and reports/cleaning_log.md.
"""
import numpy as np
import pandas as pd

from .clean import clean, log_table
from .config import ARTIFACTS, REPORTS
from .data import load_december, load_raw
from .features import add_keys, base_features, city_coordinates, daily_market, prepare_december


def december_market_options(train, valid):
    """How well could December's daily market index be *forecast* from training alone?

    Compares the real December daily mean (from validation.csv) with two forecasts that only
    use data up to 2025-10-31. If forecasts are poor, using the real series is clearly better.
    """
    hist = train.groupby("date").market_index.mean()
    actual = valid.groupby("date").market_index.mean().loc["2025-12-01":"2025-12-31"]
    last4 = hist.loc["2025-10-04":"2025-10-31"]
    dow_profile = last4.groupby(last4.index.dayofweek).mean()
    f_dow = pd.Series(actual.index.dayofweek.map(dow_profile), index=actual.index)
    f_flat = pd.Series(hist.loc["2025-10-01":].mean(), index=actual.index)
    return pd.DataFrame({
        "method": ["Oct weekday profile (seasonal naive)", "Oct flat mean"],
        "MAE vs real Dec index": [np.abs(f_dow - actual).mean(), np.abs(f_flat - actual).mean()],
        "corr with real Dec index": [f_dow.corr(actual), np.nan],
    }), actual


def main():
    train_raw, valid_raw = load_raw()
    train, valid, log, agreement = clean(train_raw, valid_raw)
    market = daily_market(train_raw, valid_raw)
    coords = city_coordinates(train_raw, valid_raw)
    dec = prepare_december(load_december(), coords, market, train.quote_signal.median())

    ARTIFACTS.mkdir(exist_ok=True)
    train.to_csv(ARTIFACTS / "train_clean.csv", index=False)
    valid.to_csv(ARTIFACTS / "valid_clean.csv", index=False)
    dec.to_csv(ARTIFACTS / "december_clean.csv", index=False)

    # sanity: features build without NaN on every frame
    for name, df in (("train", train), ("valid", valid), ("december", dec)):
        X = base_features(df, market)
        add_keys(df)
        assert not X.isna().any().any(), f"NaN features in {name}"

    options, actual = december_market_options(train_raw, valid_raw)
    lines = ["# Cleaning log", "", log_table(log).to_markdown(index=False), "",
             "## Outlier detector agreement (rule R7)", "",
             f"- Model-based (OOF robust GBM, no lane identity): {agreement['model_flags']} flags",
             f"- Lane-median based: {agreement['lane_median_flags']} flags; both agree on {agreement['both']}",
             "- Every disagreement is on a lane x equipment cell with 1-2 loads, where a corrupted row "
             "distorts the lane median; the model-based detector is used.", "",
             f"Rows after cleaning: train {len(train):,} (from {len(train_raw):,}), validation {len(valid):,}.", "",
             "## December market_index options", "",
             options.to_markdown(index=False), "",
             f"Real December daily mean ranges {actual.min():.3f}..{actual.max():.3f}. "
             "The real series from validation.csv is used (it is an input feature, not the target).", ""]
    (REPORTS / "cleaning_log.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(dec.head(8).to_string())


if __name__ == "__main__":
    main()
