# Freight Rate Prediction (Spotter ML Engineer assessment)

Predicts `posted_rate` for 12,000 loads in Nov–Dec 2025 from 48,000 labelled loads (Jan–Oct 2025),
and a daily December 2025 rate curve for one fixed lane. See `Freight_Rate_ML_Assessment.pdf` for the
original instructions and `reports/Freight_Rate_Report.docx` for the write-up.

## Deliverables

| File | What |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | the December file with `predicted_rate` filled (7 original columns) |
| `scorer_results/candidate_december.png` | chart produced by `score.py` |
| `reports/Freight_Rate_Report.docx` / `.pdf` | report: data, data quality, validation/split, model, metrics, December chart, limitations (PDF exported from the DOCX with Word) |
| `reports/loom_talking_points.md` | script for the 2–3 minute Loom |

## Setup

Python 3.12 (tested on Windows; nothing is OS-specific).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
# everything end to end (~1 min): EDA, cleaning log, final model, predictions,
# December drivers, score.py, DOCX report
python -m src.pipeline

# also re-run every validation experiment and regenerate the metric tables (~1.5 h)
python -m src.pipeline --cv

# scorer only
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Individual steps: `python -m src.eda`, `src.prepare`, `src.validate --scheme time|city|random [--models ...]`,
`src.validate --report`, `src.ensemble`, `src.train`, `src.predict`, `src.december [--variants]`, `src.report`.
All randomness is seeded (`src/config.py: SEED = 42`).

## Results (final model A34, MAE in $ on clean labels)

| Time folds T1 / T2 / T3 | Mean | Worst | MAPE | City holdout | Random 5-fold |
|---|---|---|---|---|---|
| 30.2 / 64.1 / 26.8 | 40.4 | 64.1 | 1.85% | 31.3 | 28.6 |

Baselines on the same time folds: ridge 71.8 (worst 110.7), lane median 102.9, global median $/mile 197.6.
Full tables: `reports/validation_results.md`.

## Approach in one paragraph

Validation is the two months after training, so model choice uses **rolling-origin time folds**
(Jan–Apr→May–Jun, Jan–Jun→Jul–Aug, Jan–Aug→Sep–Oct), judged on the worst fold. A **city-holdout** CV
(8 groups of cities held out as pickup or delivery) measures the unseen-city risk; a random split is shown only
for contrast. The model predicts log rate-per-mile in two stages: a **LightGBM tree** for the load-level price
(lane/city/region target encodings with fallbacks, distance, equipment, weight, `quote_signal` and its daily
**regime**) and a small **ridge model for day effects** (weekday, quarter-end ramp). Every learned statistic
(imputation medians, corrupted-label filter, encodings, stage 2, the tree) is refitted inside each fold.

## Code map

| Module | Role |
|---|---|
| `src/config.py` | paths, seed, chart colours |
| `src/data.py` | raw loading, haversine, lane-relative rate |
| `src/clean.py` | stateless fixes (`fix_features`) + fitted steps (`WeightImputer`, `find_corrupted_rates`) |
| `src/features.py` | base features, daily market/quote series, `TargetEncoder`, December input reconstruction |
| `src/model.py` | `RateModel` (fit/predict, two-stage) and `FINAL_CONFIG` |
| `src/baselines.py` | global median, lane median, ridge |
| `src/validate.py` | CV schemes, experiments, metric tables |
| `src/ensemble.py` | tree + ridge blend check |
| `src/train.py`, `src/predict.py` | final fit on all development data; prediction files |
| `src/december.py` | December driver decomposition; quote-regime variant comparison |
| `src/eda.py`, `src/prepare.py` | EDA figures/summary; cleaning log |
| `src/report.py`, `src/pipeline.py` | DOCX report; end-to-end entry point |

Outputs: `reports/figures/` (plots), `reports/metrics/` (CSV tables used by the report), `reports/*.md`
(EDA summary, cleaning log, validation results, December drivers, variant comparison).
`artifacts/` (cached CV predictions, model pickle) is git-ignored and rebuilt by the pipeline.

`score.py` is the provided scorer and is unchanged.
