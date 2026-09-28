"""Score validation.csv and fill the December chart file.

Run: python -m src.predict  (needs artifacts/model.pkl from src.train)
Writes validation_predictions.csv (repo root) and fills data/december_chart_inputs.csv in place.
"""
import pickle

import pandas as pd

from .clean import fix_features
from .config import ARTIFACTS, DECEMBER_CSV, PREDICTIONS_CSV, TEMPLATE_CSV
from .data import load_december, load_raw
from .features import city_coordinates, prepare_december


def december_quote_fill(train):
    """quote_signal is absent from the December file: use the median of comparable training loads
    (Dry Van, 300-420 mi) rather than the global median, because its spread depends on distance."""
    similar = train[(train.equipment == "Dry Van") & train.distance.between(300, 420)]
    return float(similar.quote_signal.median())


def main(model=None):
    if model is None:
        with open(ARTIFACTS / "model.pkl", "rb") as f:
            model = pickle.load(f)
    train_raw, valid_raw = load_raw()
    train, _ = fix_features(train_raw)
    valid, _ = fix_features(valid_raw)

    # validation predictions, in template order
    pred = pd.Series(model.predict(valid), index=valid.load_id.to_numpy())
    template = pd.read_csv(TEMPLATE_CSV)
    template["predicted_rate"] = template.load_id.map(pred).round(2)
    assert template.predicted_rate.notna().all() and (template.predicted_rate > 0).all()
    template[["load_id", "predicted_rate"]].to_csv(PREDICTIONS_CSV, index=False)

    # December chart: reconstruct missing inputs, keep the 7 original columns in order
    dec = load_december()
    coords = city_coordinates(train_raw, valid_raw)
    rows = prepare_december(dec, coords, model.market, december_quote_fill(train))
    dec["predicted_rate"] = model.predict(rows).round(2)
    dec.to_csv(DECEMBER_CSV, index=False)
    print(f"Wrote {PREDICTIONS_CSV.name} ({len(template):,} rows) and filled {DECEMBER_CSV.name}")
    return template, dec, rows


if __name__ == "__main__":
    main()
