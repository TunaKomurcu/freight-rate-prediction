# Loom script (2–3 minutes)

Target ~380 words spoken. Screen cues are in *italics*.

## 0:00 – Intro (15 s)
*README.md*
"This is my freight rate model for Spotter. One command, `python -m src.pipeline`, rebuilds everything: EDA,
cleaning, the final model, both prediction files, the scorer chart and this report."

## 0:15 – Key EDA findings (40 s)
*reports/figures/03_time_series.png, then 12_quote_regime.png*
- "Validation is the two months after training: Jan–Oct in, Nov–Dec out. So this is a forecast, and 12% of
  validation loads touch a city never seen in training."
- "Rate per mile falls with distance and depends on equipment. market_index is a daily series with a strong weekly
  cycle, but its slow level misleads: it was low in Jan–Feb when prices were low, and low again in Aug–Oct when they
  weren't."
- "The main finding: quote_signal has almost zero correlation with price overall, so it looked useless. But week by
  week the correlation swings from −0.3 to +0.5, and the weekly mean quote predicts that swing with r = 0.87. Three
  regimes that cancel out when pooled. Adding the regime was the largest single gain."

## 0:55 – Data quality (25 s)
*reports/cleaning_log.md, figure 08_data_quality.png*
"1.4% of rates are off by 2–6×, randomly. I flag them with an out-of-fold robust model rather than lane medians,
because thin lanes fool the median. Negative weights are sign flips; capped and missing weights and a missing market
index are fixed and flagged. No validation row is dropped."

## 1:20 – Validation and split (30 s)
*src/validate.py (TIME_FOLDS, splits), reports/validation_results.md key table*
"I pick models on rolling-origin time folds with a 2-month horizon — Jan–Apr to May–Jun, up to Jan–Aug to Sep–Oct —
using the worst fold. The random split is 1.4 to 3 times too optimistic because it shares dates. A city holdout shows
the unseen-city penalty is only about $3. Everything learned from data — encodings, imputation, outlier filter — is
refitted inside each fold."

## 1:50 – Model reasoning (30 s)
*src/model.py: RateModel.fit, FINAL_CONFIG*
"Two stages on log rate-per-mile: LightGBM for the load-level price with leak-free lane, city and region encodings
and an unseen-city fallback, plus a small ridge model for weekday and quarter-end effects. I removed the slow market
level and holiday features and tested trends, level offsets and a ridge blend; they didn't survive the time folds.
The result: MAE $40 and 1.85% MAPE on the time folds, versus $72 for ridge."

## 2:20 – December chart and limits (25 s)
*scorer_results/candidate_december.png*
"December uses real daily market and quote inputs from the validation file. The curve jitters because December's
daily quote sits right on the regime boundary. Smoother versions cost 8 to 21% accuracy, so I kept the accurate
model. Limits: no Christmas period in training, and the quarter-end effect comes from only three quarters."

## Code to show (in order)
1. `src/pipeline.py`: single entry point
2. `src/clean.py`: `fix_features`, `find_corrupted_rates`
3. `src/features.py`: `TargetEncoder.fit_transform` (OOF + unseen-city blanking), `daily_market`
4. `src/model.py`: `RateModel.fit` (fit-inside-fold, stage 2 on date-grouped OOF residuals)
5. `src/validate.py`: `TIME_FOLDS`, `splits`, city holdout
