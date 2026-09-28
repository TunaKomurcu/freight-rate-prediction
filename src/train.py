"""Fit the final model on ALL development rows with the configuration chosen in Phase 3.

Run: python -m src.train   -> artifacts/model.pkl
"""
import pickle

from .clean import fix_features
from .config import ARTIFACTS
from .data import load_raw
from .features import daily_market
from .model import FINAL_CONFIG, RateModel


def main():
    train_raw, valid_raw = load_raw()
    train, _ = fix_features(train_raw)
    valid, _ = fix_features(valid_raw)
    market = daily_market(train, valid)  # feature-only daily series, train+validation sorted by date
    model = RateModel(market, **FINAL_CONFIG).fit(train)
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Trained on {model.n_train_:,} rows (after dropping corrupted labels); config={FINAL_CONFIG}")
    print(model.feature_importance().head(15).round(1).to_string())
    return model


if __name__ == "__main__":
    main()
