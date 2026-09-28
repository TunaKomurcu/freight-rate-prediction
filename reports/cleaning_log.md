# Cleaning log

| rule                                       | action                                                                 |   train_rows |   valid_rows |
|:-------------------------------------------|:-----------------------------------------------------------------------|-------------:|-------------:|
| R1 city/equipment spelling                 | strip + title-case                                                     |            0 |            0 |
| R2 negative weight                         | take absolute value                                                    |          292 |          145 |
| R3 weight at cap (47,500) or floor (5,000) | keep value, add flag column                                            |         1236 |          308 |
| R4 missing weight                          | impute lane x equipment median (>=5 loads) else equipment median; flag |          300 |          165 |
| R5 missing market_index                    | fill with same-date mean of the feature; flag                          |          374 |          249 |
| R6 clipped coordinates                     | keep; distance is authoritative                                        |         2052 |          669 |
| R7 corrupted posted_rate (x2-x6 off)       | drop from training                                                     |          677 |            0 |

## Outlier detector agreement (rule R7)

- Model-based (OOF robust GBM, no lane identity): 677 flags
- Lane-median based: 675 flags; both agree on 673
- Every disagreement is on a lane x equipment cell with 1-2 loads, where a corrupted row distorts the lane median; the model-based detector is used.

Rows after cleaning: train 47,323 (from 48,000), validation 12,000.

## December market_index options

| method                               |   MAE vs real Dec index |   corr with real Dec index |
|:-------------------------------------|------------------------:|---------------------------:|
| Oct weekday profile (seasonal naive) |               0.0200112 |                   0.988528 |
| Oct flat mean                        |               0.0656958 |                 nan        |

Real December daily mean ranges 0.831..1.045. The real series from validation.csv is used (it is an input feature, not the target).
