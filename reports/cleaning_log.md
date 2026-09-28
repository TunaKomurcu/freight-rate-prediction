# Cleaning log

| rule                          | action                                                                                          |   train_rows |   valid_rows |
|:------------------------------|:------------------------------------------------------------------------------------------------|-------------:|-------------:|
| R1 city/equipment spelling    | strip + title-case (defensive)                                                                  |            0 |            0 |
| R2 negative weight            | absolute value (sign flip)                                                                      |          292 |          145 |
| R3 weight at cap/floor        | keep value, add flag column                                                                     |         1236 |          308 |
| R4 missing weight             | impute lane x equipment median (>=5 loads) else equipment median, fitted on training rows; flag |          300 |          165 |
| R5 missing market_index       | same-date mean of the feature; flag                                                             |          374 |          249 |
| R6 clipped coordinates (kept) | keep; distance is authoritative                                                                 |         2052 |          669 |
| R7 corrupted posted_rate      | drop from training (rate/expected outside 0.6-1.7)                                              |          677 |            0 |

## Corrupted-label detector (R7)

- Model-based (out-of-fold robust GBM, no lane identity): 677 rows
- Lane-median based: 675 rows; both agree on 673
- Every disagreement sits in a lane x equipment cell with 1-2 loads, where the corrupted row itself distorts the lane median. The model-based detector is used.
- Remaining rows: rate/expected in 0.917..1.103
- No pattern: ~1.4% in every equipment, month, weekday, distance band and pickup city (chi-square p > 0.29); factors spread continuously over x2..x5.8 up and down.

## December market_index

| December market_index source                 |   MAE vs real daily index |
|:---------------------------------------------|--------------------------:|
| Real daily mean from validation.csv (chosen) |                    0      |
| Forecast: October weekday profile            |                    0.02   |
| Forecast: October flat mean                  |                    0.0657 |

Real December daily mean ranges 0.831..1.045.
