# Validation results

Dollar-scale metrics. *clean* = test rows whose label is not flagged as corrupted; *raw* = all test rows (Spotter's labels likely contain the same ~1.4% corruption).
Time folds: T1 Jan-Apr > May-Jun, T2 Jan-Jun > Jul-Aug, T3 Jan-Aug > Sep-Oct. Final model: A34 A27 + recency half-life 60d

## Key models (MAE $, clean labels; bias = mean signed % error, negative = under-prediction)

|                                          |     T1 |     T2 |     T3 |   time mean |   time worst |   time MAPE % |   time RMSE clean |   time MAE raw |   time RMSE raw |   city MAE |   random MAE |   bias May % |   bias Jun % |   bias Jul % |   bias Aug % |   bias Sep % |   bias Oct % |
|:-----------------------------------------|-------:|-------:|-------:|------------:|-------------:|--------------:|------------------:|---------------:|----------------:|-----------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| B1 global median rpm x distance          | 189.83 | 199.15 | 203.69 |      197.56 |       203.69 |          9.37 |            276.89 |         249.25 |          676.94 |     213.04 |       213.48 |        -5.11 |        -7.46 |        -2.77 |         0.2  |        -1.23 |        -1.67 |
| B2 lane median rpm                       | 148.75 |  83.85 |  76.18 |      102.92 |       148.75 |          4.25 |            143.61 |         156.17 |          645.11 |      94.84 |        90.88 |        -4.55 |        -6.77 |        -2.15 |         0.91 |        -0.6  |        -0.92 |
| B3 ridge (log rpm)                       |  52.55 |  52.33 | 110.67 |       71.85 |       110.67 |          3.01 |             98.68 |         125.49 |          636.22 |      59.62 |        54.76 |         0.45 |        -2.25 |        -0.75 |        -2.64 |        -5.1  |        -4.28 |
| M0 LightGBM log-rpm                      |  59.2  |  69.97 |  81.16 |       70.11 |        81.16 |          3.11 |            106.66 |         123.62 |          633.51 |      26.63 |        23.78 |        -0.13 |        -1.95 |        -0.59 |        -3    |        -2.3  |        -4.98 |
| A21 cycle - holiday features             |  68.4  |  85.45 |  32.88 |       62.24 |        85.45 |          2.7  |             91.82 |         115.89 |          631.92 |      29.21 |        26.57 |        -3.1  |        -2.06 |        -2.58 |        -0.88 |         1.13 |        -0.5  |
| A24 cycle - holidays + additive calendar |  55.92 |  80.23 |  30.88 |       55.68 |        80.23 |          2.38 |             87.68 |         109.39 |          630.57 |      42.9  |        40.95 |        -2.02 |        -1.87 |        -0.86 |        -0.84 |         0.41 |        -0.29 |
| A27 A24 + qs_day (quote regime)          |  28.34 |  67.95 |  23.17 |       39.82 |        67.95 |          1.83 |             62.5  |          93.77 |          627.34 |      27.13 |        24.91 |        -0.34 |        -0.88 |        -0.08 |        -3.27 |         0.21 |         0.17 |
| A34 A27 + recency half-life 60d          |  30.21 |  64.11 |  26.78 |       40.37 |        64.11 |          1.85 |             63.29 |          94.29 |          626.98 |      31.3  |        28.63 |        -0.35 |        -1.03 |        -0.12 |        -3.24 |         0.19 |         0.27 |
| A39 A24 + coarse quote regime            |  41.24 |  73.67 |  27.26 |       47.39 |        73.67 |          2.07 |             77.05 |         101.21 |          628.75 |      33.23 |        31.77 |        -0.74 |        -1.74 |        -0.37 |        -2.29 |         0.34 |         0.19 |
| A40 A39 + recency half-life 60d          |  42.51 |  71.43 |  32.37 |       48.77 |        71.43 |          2.12 |             78.06 |         102.56 |          628.43 |      38.4  |        36.66 |        -0.78 |        -1.89 |        -0.36 |        -1.92 |         0.96 |         0.45 |
| A42 A24 + qs_7d + quote x regime         |  32.76 |  68.19 |  30.21 |       43.72 |        68.19 |          1.95 |             70.22 |          97.58 |          627.7  |      28.43 |        26.26 |         0.08 |        -0.96 |        -0.29 |        -2.18 |        -0.2  |        -0.46 |
| A44 A42 + recency half-life 60d          |  33.48 |  66.51 |  35.35 |       45.12 |        66.51 |          2    |             70.78 |          98.95 |          627.58 |      31.54 |        28.84 |         0.02 |        -1.07 |        -0.28 |        -2.03 |        -0.39 |        -0.6  |

## Experiment legend

| id                                                   | config (changes vs defaults)                                                                                             | note                                                                                                         |
|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| B1 global median rpm x distance                      | baseline                                                                                                                 |                                                                                                              |
| B2 lane median rpm                                   | baseline                                                                                                                 |                                                                                                              |
| B3 ridge (log rpm)                                   | baseline                                                                                                                 |                                                                                                              |
| M0 LightGBM log-rpm                                  | {}                                                                                                                       |                                                                                                              |
| A1 - quote_signal                                    | {'quote_signal': False}                                                                                                  |                                                                                                              |
| A2 - holiday features                                | {'holidays': False}                                                                                                      |                                                                                                              |
| A3 + linear time trend                               | {'trend': 'linear'}                                                                                                      | M0 + linear trend removed from the log target (slope from daily residuals, controlling for log market index) |
| A4 target = rpm                                      | {'target': 'rpm'}                                                                                                        |                                                                                                              |
| A5 target = rate ($)                                 | {'target': 'rate'}                                                                                                       |                                                                                                              |
| A6 raw labels + Huber                                | {'clean_labels': False, 'objective': 'huber'}                                                                            |                                                                                                              |
| A7 raw labels + L2                                   | {'clean_labels': False}                                                                                                  |                                                                                                              |
| A8 + smearing                                        | {'smearing': True}                                                                                                       |                                                                                                              |
| A9 no unseen-city blanking                           | {'unseen_rate': 0.0}                                                                                                     |                                                                                                              |
| A10 - holidays - quote_signal                        | {'holidays': False, 'quote_signal': False}                                                                               |                                                                                                              |
| A11 quote_signal as deviation from daily mean        | {'quote_signal': 'dev'}                                                                                                  |                                                                                                              |
| A12 market: weekly cycle only (no slow level)        | {'market_features': 'cycle'}                                                                                             | drops market_index, mi_day, mi_7d, mi_28d; keeps mi_cycle (= day - 28d mean) and per-load deviation          |
| A13 + linear trend (quarter-ramp controlled)         | {'trend': 'linear_q'}                                                                                                    | as A3, but the slope regression also controls for the quarter-end ramp                                       |
| A16 cycle + qs dev                                   | {'market_features': 'cycle', 'quote_signal': 'dev'}                                                                      |                                                                                                              |
| A18 cycle + qs dev + trend_q                         | {'market_features': 'cycle', 'quote_signal': 'dev', 'trend': 'linear_q'}                                                 |                                                                                                              |
| A20 cycle - quote_signal                             | {'market_features': 'cycle', 'quote_signal': False}                                                                      |                                                                                                              |
| A21 cycle - holiday features                         | {'market_features': 'cycle', 'holidays': False}                                                                          |                                                                                                              |
| A22 cycle + trend_q                                  | {'market_features': 'cycle', 'trend': 'linear_q'}                                                                        |                                                                                                              |
| A23 cycle - holidays + smearing                      | {'market_features': 'cycle', 'holidays': False, 'smearing': True}                                                        |                                                                                                              |
| A24 cycle - holidays + additive calendar             | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive'}                                                  | tree without date features + ridge stage 2 (weekday + quarter-end ramp) on out-of-fold residuals             |
| A25 all market - holidays + additive calendar        | {'holidays': False, 'calendar': 'additive'}                                                                              |                                                                                                              |
| A26 cycle - holidays + additive calendar with market | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'calendar_market': True}                         |                                                                                                              |
| A27 A24 + qs_day (quote regime)                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_day'}                      | adds the daily mean quote_signal, which identifies the regime of the quote-price relation                    |
| A28 A24 + recency half-life 30d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'half_life': 30}                                 |                                                                                                              |
| A29 A24 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'half_life': 60}                                 |                                                                                                              |
| A30 A24 + recency half-life 90d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'half_life': 90}                                 |                                                                                                              |
| A31 A24 + level offset 28d                           | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'level_window': 28}                              | level offset = mean out-of-time residual of the last 28 training days                                        |
| A32 A24 + level offset 56d                           | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'level_window': 56}                              |                                                                                                              |
| A33 A27 + level offset 28d                           | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_day', 'level_window': 28}  |                                                                                                              |
| A34 A27 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_day', 'half_life': 60}     | FINAL: lowest worst-fold and mean MAE; December curve jitters (daily quote mean around the regime boundary)  |
| A35 A27 + recency half-life 90d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_day', 'half_life': 90}     |                                                                                                              |
| A36 A27 + level offset 42d                           | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_day', 'level_window': 42}  |                                                                                                              |
| A37 A24 + qs_7d (smoothed quote regime)              | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_7d'}                       | trailing 7-day mean quote instead of the daily mean                                                          |
| A38 A37 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_7d', 'half_life': 60}      |                                                                                                              |
| A39 A24 + coarse quote regime                        | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_regime'}                   | quote regime as 3 levels (7-day mean quote < 2.0 / 2.0-2.1 / > 2.1)                                          |
| A40 A39 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_regime', 'half_life': 60}  | smoothest December curve, but +11% worst-fold / +21% mean MAE vs A34                                         |
| A41 A24 + qs_14d (continuous)                        | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_14d'}                      |                                                                                                              |
| A42 A24 + qs_7d + quote x regime                     | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'interact_7d'}                   | quote_signal x (trailing 7-day mean quote - c), c = training mean quote of the fold                          |
| A43 A24 + qs_14d + quote x regime                    | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'interact_14d'}                  |                                                                                                              |
| A44 A42 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'interact_7d', 'half_life': 60}  |                                                                                                              |
| A45 A43 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'interact_14d', 'half_life': 60} |                                                                                                              |
| A46 A41 + recency half-life 60d                      | {'market_features': 'cycle', 'holidays': False, 'calendar': 'additive', 'quote_signal': 'with_14d', 'half_life': 60}     |                                                                                                              |
| T1 leaves=63, min_child=20                           | {'params': {'num_leaves': 63, 'min_child_samples': 20}}                                                                  |                                                                                                              |
| T2 leaves=15, lr=0.05, 800 trees                     | {'params': {'num_leaves': 15, 'learning_rate': 0.05, 'n_estimators': 800}}                                               |                                                                                                              |

## Scheme: time (mean over folds)

| model                                                |   MAE (clean) |   RMSE (clean) |   MAPE % (clean) |   MAE (raw) |   RMSE (raw) |   MAPE % (raw) |
|:-----------------------------------------------------|--------------:|---------------:|-----------------:|------------:|-------------:|---------------:|
| B1 global median rpm x distance                      |        197.56 |         276.89 |             9.37 |      249.25 |       676.94 |          11.56 |
| B2 lane median rpm                                   |        102.92 |         143.61 |             4.24 |      156.17 |       645.11 |           6.53 |
| B3 ridge (log rpm)                                   |         71.85 |          98.68 |             3.02 |      125.49 |       636.22 |           5.32 |
| M0 LightGBM log-rpm                                  |         70.11 |         106.66 |             3.11 |      123.62 |       633.51 |           5.39 |
| A1 - quote_signal                                    |         74.07 |          98.07 |             3.09 |      127.63 |       636.6  |           5.37 |
| A2 - holiday features                                |         69.71 |         106    |             3.09 |      123.25 |       633.69 |           5.38 |
| A3 + linear time trend                               |         70.12 |         108.92 |             2.83 |      123.57 |       630.67 |           5.19 |
| A4 target = rpm                                      |         70.91 |         106.41 |             3.11 |      124.42 |       633.43 |           5.4  |
| A5 target = rate ($)                                 |         71.06 |          97.47 |             3.47 |      124.58 |       633.35 |           5.74 |
| A6 raw labels + Huber                                |         70.06 |         104.7  |             3.15 |      123.6  |       633.89 |           5.43 |
| A7 raw labels + L2                                   |         97.17 |         137.4  |             4.41 |      150.37 |       640.99 |           6.67 |
| A8 + smearing                                        |         69.93 |         106.49 |             3.1  |      123.44 |       633.46 |           5.39 |
| A9 no unseen-city blanking                           |         69.8  |         105.14 |             3.09 |      123.32 |       633.34 |           5.38 |
| A10 - holidays - quote_signal                        |         73.65 |          97.66 |             3.08 |      127.21 |       636.46 |           5.36 |
| A11 quote_signal as deviation from daily mean        |         71.38 |          95.78 |             3.3  |      124.95 |       634.4  |           5.57 |
| A12 market: weekly cycle only (no slow level)        |         64.85 |          94.59 |             2.79 |      118.46 |       632.32 |           5.1  |
| A13 + linear trend (quarter-ramp controlled)         |         59.26 |          95.7  |             2.42 |      112.87 |       628.46 |           4.78 |
| A16 cycle + qs dev                                   |         89.11 |         120.12 |             3.84 |      142.43 |       637.44 |           6.13 |
| A18 cycle + qs dev + trend_q                         |         82.28 |         118.34 |             3.42 |      135.61 |       632.93 |           5.77 |
| A20 cycle - quote_signal                             |        109.56 |         143.82 |             4.47 |      162.61 |       644.51 |           6.73 |
| A21 cycle - holiday features                         |         62.24 |          91.82 |             2.69 |      115.89 |       631.92 |           5.01 |
| A22 cycle + trend_q                                  |         64.67 |          97.22 |             2.66 |      118.21 |       628.44 |           5.03 |
| A23 cycle - holidays + smearing                      |         62.16 |          91.75 |             2.69 |      115.81 |       631.88 |           5    |
| A24 cycle - holidays + additive calendar             |         55.68 |          87.68 |             2.38 |      109.39 |       630.57 |           4.7  |
| A25 all market - holidays + additive calendar        |         74.05 |         111.82 |             3.26 |      127.5  |       634.53 |           5.54 |
| A26 cycle - holidays + additive calendar with market |         55.1  |          86.33 |             2.39 |      108.83 |       630.56 |           4.71 |
| A27 A24 + qs_day (quote regime)                      |         39.82 |          62.5  |             1.83 |       93.77 |       627.34 |           4.17 |
| A28 A24 + recency half-life 30d                      |         58.29 |          91.3  |             2.5  |      111.93 |       629.53 |           4.84 |
| A29 A24 + recency half-life 60d                      |         55.69 |          87.67 |             2.38 |      109.38 |       629.92 |           4.72 |
| A30 A24 + recency half-life 90d                      |         55.62 |          87.4  |             2.37 |      109.32 |       630.14 |           4.7  |
| A31 A24 + level offset 28d                           |         55.56 |          86.27 |             2.36 |      109.24 |       628.53 |           4.72 |
| A32 A24 + level offset 56d                           |         68.44 |          99.17 |             2.91 |      121.94 |       628.52 |           5.29 |
| A33 A27 + level offset 28d                           |         54.95 |          79.4  |             2.43 |      108.64 |       626.39 |           4.8  |
| A34 A27 + recency half-life 60d                      |         40.37 |          63.29 |             1.84 |       94.29 |       626.98 |           4.18 |
| A35 A27 + recency half-life 90d                      |         40.44 |          63.36 |             1.85 |       94.37 |       627.21 |           4.19 |
| A36 A27 + level offset 42d                           |         90.64 |         119.01 |             3.92 |      143.85 |       630.58 |           6.3  |
| A37 A24 + qs_7d (smoothed quote regime)              |         44.94 |          72.13 |             1.99 |       98.81 |       628.34 |           4.32 |
| A38 A37 + recency half-life 60d                      |         46.52 |          73.11 |             2.05 |      100.35 |       628.24 |           4.38 |
| A39 A24 + coarse quote regime                        |         47.39 |          77.05 |             2.07 |      101.21 |       628.75 |           4.4  |
| A40 A39 + recency half-life 60d                      |         48.77 |          78.06 |             2.12 |      102.56 |       628.43 |           4.46 |
| A41 A24 + qs_14d (continuous)                        |         47.47 |          76.01 |             2.05 |      101.33 |       628.5  |           4.39 |
| A42 A24 + qs_7d + quote x regime                     |         43.72 |          70.22 |             1.94 |       97.58 |       627.7  |           4.28 |
| A43 A24 + qs_14d + quote x regime                    |         47.24 |          75.74 |             2.05 |      101.1  |       628.39 |           4.39 |
| A44 A42 + recency half-life 60d                      |         45.12 |          70.78 |             2    |       98.95 |       627.58 |           4.34 |
| A45 A43 + recency half-life 60d                      |         48.66 |          76.48 |             2.12 |      102.49 |       628.28 |           4.46 |
| A46 A41 + recency half-life 60d                      |         49.59 |          78.28 |             2.14 |      103.41 |       628.6  |           4.48 |
| T1 leaves=63, min_child=20                           |         73.12 |         113.69 |             3.22 |      126.59 |       634.79 |           5.5  |
| T2 leaves=15, lr=0.05, 800 trees                     |         70.7  |         105.28 |             3.14 |      124.22 |       633.59 |           5.42 |

Per-fold MAE, clean labels (time):

| model                                                |   T1: Jan-Apr > May-Jun |   T2: Jan-Jun > Jul-Aug |   T3: Jan-Aug > Sep-Oct |
|:-----------------------------------------------------|------------------------:|------------------------:|------------------------:|
| B1 global median rpm x distance                      |                   189.8 |                   199.2 |                   203.7 |
| B2 lane median rpm                                   |                   148.7 |                    83.8 |                    76.2 |
| B3 ridge (log rpm)                                   |                    52.6 |                    52.3 |                   110.7 |
| M0 LightGBM log-rpm                                  |                    59.2 |                    70   |                    81.2 |
| A1 - quote_signal                                    |                    51.4 |                    52.7 |                   118.1 |
| A2 - holiday features                                |                    61.2 |                    68.4 |                    79.5 |
| A3 + linear time trend                               |                    55.1 |                   116.3 |                    39   |
| A4 target = rpm                                      |                    59.5 |                    71.2 |                    82   |
| A5 target = rate ($)                                 |                    59.6 |                    59.5 |                    94   |
| A6 raw labels + Huber                                |                    53.6 |                    73.2 |                    83.4 |
| A7 raw labels + L2                                   |                   105.8 |                    87.1 |                    98.6 |
| A8 + smearing                                        |                    59.2 |                    69.9 |                    80.7 |
| A9 no unseen-city blanking                           |                    57.5 |                    70   |                    81.9 |
| A10 - holidays - quote_signal                        |                    51.5 |                    52.3 |                   117.2 |
| A11 quote_signal as deviation from daily mean        |                    43.9 |                    65.9 |                   104.4 |
| A12 market: weekly cycle only (no slow level)        |                    67.7 |                    93.6 |                    33.3 |
| A13 + linear trend (quarter-ramp controlled)         |                    52.9 |                    86.3 |                    38.6 |
| A16 cycle + qs dev                                   |                   111.4 |                   106.2 |                    49.7 |
| A18 cycle + qs dev + trend_q                         |                    73.8 |                    93.8 |                    79.3 |
| A20 cycle - quote_signal                             |                   148.9 |                   118.1 |                    61.7 |
| A21 cycle - holiday features                         |                    68.4 |                    85.5 |                    32.9 |
| A22 cycle + trend_q                                  |                    48   |                    83.6 |                    62.4 |
| A23 cycle - holidays + smearing                      |                    68.1 |                    85.4 |                    33   |
| A24 cycle - holidays + additive calendar             |                    55.9 |                    80.2 |                    30.9 |
| A25 all market - holidays + additive calendar        |                    60.9 |                    72.5 |                    88.7 |
| A26 cycle - holidays + additive calendar with market |                    42.8 |                    77   |                    45.4 |
| A27 A24 + qs_day (quote regime)                      |                    28.3 |                    68   |                    23.2 |
| A28 A24 + recency half-life 30d                      |                    54.3 |                    75.5 |                    45.1 |
| A29 A24 + recency half-life 60d                      |                    55   |                    77.4 |                    34.7 |
| A30 A24 + recency half-life 90d                      |                    55.8 |                    78.3 |                    32.7 |
| A31 A24 + level offset 28d                           |                    45.1 |                    89.9 |                    31.7 |
| A32 A24 + level offset 56d                           |                    77.3 |                    90   |                    38   |
| A33 A27 + level offset 28d                           |                    29.7 |                    70.6 |                    64.5 |
| A34 A27 + recency half-life 60d                      |                    30.2 |                    64.1 |                    26.8 |
| A35 A27 + recency half-life 90d                      |                    29.7 |                    66.3 |                    25.2 |
| A36 A27 + level offset 42d                           |                   138.9 |                    69.1 |                    63.9 |
| A37 A24 + qs_7d (smoothed quote regime)              |                    36.2 |                    69.2 |                    29.4 |
| A38 A37 + recency half-life 60d                      |                    37   |                    67   |                    35.6 |
| A39 A24 + coarse quote regime                        |                    41.2 |                    73.7 |                    27.3 |
| A40 A39 + recency half-life 60d                      |                    42.5 |                    71.4 |                    32.4 |
| A41 A24 + qs_14d (continuous)                        |                    41.2 |                    70.8 |                    30.4 |
| A42 A24 + qs_7d + quote x regime                     |                    32.8 |                    68.2 |                    30.2 |
| A43 A24 + qs_14d + quote x regime                    |                    39.6 |                    71.3 |                    30.8 |
| A44 A42 + recency half-life 60d                      |                    33.5 |                    66.5 |                    35.4 |
| A45 A43 + recency half-life 60d                      |                    40.3 |                    70.1 |                    35.6 |
| A46 A41 + recency half-life 60d                      |                    42   |                    71.1 |                    35.6 |
| T1 leaves=63, min_child=20                           |                    65.6 |                    73.7 |                    80.1 |
| T2 leaves=15, lr=0.05, 800 trees                     |                    62.4 |                    65   |                    84.7 |

Per-fold RMSE, raw labels (time):

| model                                                |   T1: Jan-Apr > May-Jun |   T2: Jan-Jun > Jul-Aug |   T3: Jan-Aug > Sep-Oct |
|:-----------------------------------------------------|------------------------:|------------------------:|------------------------:|
| B1 global median rpm x distance                      |                   671.8 |                   674.7 |                   684.2 |
| B2 lane median rpm                                   |                   666.4 |                   631.9 |                   637   |
| B3 ridge (log rpm)                                   |                   633.1 |                   625.2 |                   650.4 |
| M0 LightGBM log-rpm                                  |                   632.3 |                   627.2 |                   641   |
| A1 - quote_signal                                    |                   633.3 |                   624.9 |                   651.6 |
| A2 - holiday features                                |                   633.4 |                   626.7 |                   641   |
| A3 + linear time trend                               |                   629.6 |                   633.3 |                   629.2 |
| A4 target = rpm                                      |                   631.7 |                   627   |                   641.5 |
| A5 target = rate ($)                                 |                   632.6 |                   624.2 |                   643.2 |
| A6 raw labels + Huber                                |                   631.2 |                   628.3 |                   642.2 |
| A7 raw labels + L2                                   |                   645.6 |                   630.9 |                   646.5 |
| A8 + smearing                                        |                   632.3 |                   627.1 |                   640.9 |
| A9 no unseen-city blanking                           |                   631.3 |                   627.2 |                   641.5 |
| A10 - holidays - quote_signal                        |                   633.2 |                   624.8 |                   651.3 |
| A11 quote_signal as deviation from daily mean        |                   631.7 |                   625.1 |                   646.4 |
| A12 market: weekly cycle only (no slow level)        |                   637.8 |                   631.7 |                   627.5 |
| A13 + linear trend (quarter-ramp controlled)         |                   629.5 |                   626.8 |                   629.1 |
| A16 cycle + qs dev                                   |                   650   |                   633.8 |                   628.6 |
| A18 cycle + qs dev + trend_q                         |                   638.8 |                   629.8 |                   630.2 |
| A20 cycle - quote_signal                             |                   660.2 |                   641.8 |                   631.5 |
| A21 cycle - holiday features                         |                   638.6 |                   629.9 |                   627.3 |
| A22 cycle + trend_q                                  |                   631.3 |                   626.6 |                   627.3 |
| A23 cycle - holidays + smearing                      |                   638.5 |                   629.8 |                   627.3 |
| A24 cycle - holidays + additive calendar             |                   635.5 |                   628.7 |                   627.6 |
| A25 all market - holidays + additive calendar        |                   633.4 |                   627   |                   643.2 |
| A26 cycle - holidays + additive calendar with market |                   631.5 |                   628.1 |                   632   |
| A27 A24 + qs_day (quote regime)                      |                   628.6 |                   626.6 |                   626.8 |
| A28 A24 + recency half-life 30d                      |                   634.6 |                   626.8 |                   627.2 |
| A29 A24 + recency half-life 60d                      |                   635   |                   627.9 |                   626.9 |
| A30 A24 + recency half-life 90d                      |                   635.3 |                   628.2 |                   626.9 |
| A31 A24 + level offset 28d                           |                   629.4 |                   629   |                   627.2 |
| A32 A24 + level offset 56d                           |                   629.9 |                   629   |                   626.6 |
| A33 A27 + level offset 28d                           |                   626.6 |                   625.3 |                   627.2 |
| A34 A27 + recency half-life 60d                      |                   628.7 |                   625.5 |                   626.7 |
| A35 A27 + recency half-life 90d                      |                   628.6 |                   626.1 |                   626.9 |
| A36 A27 + level offset 42d                           |                   639   |                   625.5 |                   627.2 |
| A37 A24 + qs_7d (smoothed quote regime)              |                   630.5 |                   626   |                   628.5 |
| A38 A37 + recency half-life 60d                      |                   630.6 |                   624.8 |                   629.4 |
| A39 A24 + coarse quote regime                        |                   631.3 |                   627.7 |                   627.3 |
| A40 A39 + recency half-life 60d                      |                   631.5 |                   626.8 |                   626.9 |
| A41 A24 + qs_14d (continuous)                        |                   631.1 |                   625.9 |                   628.5 |
| A42 A24 + qs_7d + quote x regime                     |                   629   |                   625.6 |                   628.5 |
| A43 A24 + qs_14d + quote x regime                    |                   630.6 |                   626   |                   628.6 |
| A44 A42 + recency half-life 60d                      |                   628.8 |                   624.7 |                   629.2 |
| A45 A43 + recency half-life 60d                      |                   630.6 |                   625.4 |                   628.9 |
| A46 A41 + recency half-life 60d                      |                   631.4 |                   625.6 |                   628.8 |
| T1 leaves=63, min_child=20                           |                   634.6 |                   628.3 |                   641.4 |
| T2 leaves=15, lr=0.05, 800 trees                     |                   633.7 |                   625.1 |                   641.9 |

Mean signed % error by test month (clean labels; negative = under-prediction):

| model                                                |   2025-05 |   2025-06 |   2025-07 |   2025-08 |   2025-09 |   2025-10 |
|:-----------------------------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|
| B1 global median rpm x distance                      |     -5.11 |     -7.46 |     -2.77 |      0.2  |     -1.23 |     -1.67 |
| B2 lane median rpm                                   |     -4.55 |     -6.77 |     -2.15 |      0.91 |     -0.6  |     -0.92 |
| B3 ridge (log rpm)                                   |      0.45 |     -2.25 |     -0.75 |     -2.64 |     -5.1  |     -4.28 |
| M0 LightGBM log-rpm                                  |     -0.13 |     -1.95 |     -0.59 |     -3    |     -2.3  |     -4.98 |
| A1 - quote_signal                                    |     -0.64 |     -2.62 |     -1.24 |     -2.62 |     -4.19 |     -5.66 |
| A2 - holiday features                                |     -0.11 |     -2.19 |     -0.57 |     -2.96 |     -2.11 |     -5.06 |
| A3 + linear time trend                               |      0.27 |     -0.14 |      3.04 |      4.6  |      0.63 |     -1.01 |
| A4 target = rpm                                      |     -0.03 |     -1.74 |     -0.59 |     -2.73 |     -2.33 |     -4.99 |
| A5 target = rate ($)                                 |     -0.5  |     -3.88 |     -0.87 |     -2.64 |     -2.89 |     -6.25 |
| A6 raw labels + Huber                                |     -0.32 |     -1.92 |     -0.73 |     -3.08 |     -2.2  |     -5.22 |
| A7 raw labels + L2                                   |     -2.16 |     -2.85 |     -0.52 |     -2.04 |     -2.11 |     -4.74 |
| A8 + smearing                                        |     -0.12 |     -1.94 |     -0.58 |     -2.99 |     -2.27 |     -4.96 |
| A9 no unseen-city blanking                           |     -0.01 |     -1.73 |     -0.58 |     -3.01 |     -2.31 |     -5    |
| A10 - holidays - quote_signal                        |     -0.67 |     -2.64 |     -1.16 |     -2.61 |     -4.09 |     -5.66 |
| A11 quote_signal as deviation from daily mean        |     -0.26 |     -2.15 |     -1.15 |     -2.84 |     -4.27 |     -5.08 |
| A12 market: weekly cycle only (no slow level)        |     -2.95 |     -2.08 |     -3.39 |     -0.47 |      0.98 |     -0.59 |
| A13 + linear trend (quarter-ramp controlled)         |      0.14 |     -0.52 |      1.86 |      2.27 |      0.65 |     -0.93 |
| A16 cycle + qs dev                                   |     -3.89 |     -5.61 |     -4.7  |      0.79 |      2.16 |     -0.48 |
| A18 cycle + qs dev + trend_q                         |     -2.63 |     -3.39 |     -0.99 |      4.21 |      4.02 |      2.29 |
| A20 cycle - quote_signal                             |     -5.83 |     -6.1  |     -6.88 |      1.3  |      1.88 |     -1.5  |
| A21 cycle - holiday features                         |     -3.1  |     -2.06 |     -2.58 |     -0.88 |      1.13 |     -0.5  |
| A22 cycle + trend_q                                  |     -1.8  |     -0.08 |      0.11 |      3.01 |      2.8  |      2.18 |
| A23 cycle - holidays + smearing                      |     -3.09 |     -2.04 |     -2.57 |     -0.87 |      1.15 |     -0.48 |
| A24 cycle - holidays + additive calendar             |     -2.02 |     -1.87 |     -0.86 |     -0.84 |      0.41 |     -0.29 |
| A25 all market - holidays + additive calendar        |     -0.29 |     -2.22 |     -0.43 |     -2.72 |     -2.92 |     -4.98 |
| A26 cycle - holidays + additive calendar with market |     -0.59 |     -0.7  |     -0.43 |     -1.57 |     -1.82 |     -1.61 |
| A27 A24 + qs_day (quote regime)                      |     -0.34 |     -0.88 |     -0.08 |     -3.27 |      0.21 |      0.17 |
| A28 A24 + recency half-life 30d                      |     -1.65 |     -2.15 |     -0.37 |      0.01 |      2.37 |      0.14 |
| A29 A24 + recency half-life 60d                      |     -1.8  |     -2.03 |     -0.6  |     -0.47 |      1.24 |     -0.04 |
| A30 A24 + recency half-life 90d                      |     -1.89 |     -2.01 |     -0.67 |     -0.56 |      0.9  |     -0.1  |
| A31 A24 + level offset 28d                           |      0.26 |      0.42 |      1.1  |      1.12 |      0.67 |     -0.03 |
| A32 A24 + level offset 56d                           |      2.66 |      2.82 |      1.11 |      1.13 |      1.41 |      0.7  |
| A33 A27 + level offset 28d                           |      0.96 |      0.41 |      0.99 |     -2.23 |      2.69 |      2.64 |
| A34 A27 + recency half-life 60d                      |     -0.35 |     -1.03 |     -0.12 |     -3.24 |      0.19 |      0.27 |
| A35 A27 + recency half-life 90d                      |     -0.35 |     -1.01 |     -0.1  |     -3.22 |      0.16 |      0.25 |
| A36 A27 + level offset 42d                           |      6.02 |      5.44 |      0.73 |     -2.49 |      2.66 |      2.62 |
| A37 A24 + qs_7d (smoothed quote regime)              |      0.07 |     -1.45 |     -0.28 |     -2.22 |     -0.24 |     -0.45 |
| A38 A37 + recency half-life 60d                      |      0.03 |     -1.6  |     -0.25 |     -2.03 |     -0.46 |     -0.63 |
| A39 A24 + coarse quote regime                        |     -0.74 |     -1.74 |     -0.37 |     -2.29 |      0.34 |      0.19 |
| A40 A39 + recency half-life 60d                      |     -0.78 |     -1.89 |     -0.36 |     -1.92 |      0.96 |      0.45 |
| A41 A24 + qs_14d (continuous)                        |      0.08 |     -1.78 |     -0.24 |     -0.82 |     -0.15 |     -0.23 |
| A42 A24 + qs_7d + quote x regime                     |      0.08 |     -0.96 |     -0.29 |     -2.18 |     -0.2  |     -0.46 |
| A43 A24 + qs_14d + quote x regime                    |      0.06 |     -1.65 |     -0.26 |     -0.95 |     -0.17 |     -0.24 |
| A44 A42 + recency half-life 60d                      |      0.02 |     -1.07 |     -0.28 |     -2.03 |     -0.39 |     -0.6  |
| A45 A43 + recency half-life 60d                      |      0    |     -1.82 |     -0.2  |     -0.77 |     -0.09 |     -0.3  |
| A46 A41 + recency half-life 60d                      |      0.01 |     -1.94 |     -0.18 |     -0.7  |     -0.1  |     -0.35 |
| T1 leaves=63, min_child=20                           |     -0.14 |     -2.41 |     -0.56 |     -2.98 |     -2.12 |     -4.96 |
| T2 leaves=15, lr=0.05, 800 trees                     |     -0.08 |     -2.12 |     -0.72 |     -3.14 |     -2.55 |     -5.12 |

Fold sizes:

| fold                  |   n_train |   n_train_dropped |   n_test |
|:----------------------|----------:|------------------:|---------:|
| T1: Jan-Apr > May-Jun |     19110 |               275 |     9696 |
| T2: Jan-Jun > Jul-Aug |     28806 |               400 |     9671 |
| T3: Jan-Aug > Sep-Oct |     38477 |               533 |     9523 |

## Scheme: city (mean over folds)

| model                                         |   MAE (clean) |   RMSE (clean) |   MAPE % (clean) |   MAE (raw) |   RMSE (raw) |   MAPE % (raw) |
|:----------------------------------------------|--------------:|---------------:|-----------------:|------------:|-------------:|---------------:|
| B1 global median rpm x distance               |        213.04 |         310.48 |             9.45 |      260.94 |       659.74 |          11.67 |
| B2 lane median rpm                            |         94.84 |         131.38 |             4.31 |      144.6  |       605.06 |           6.66 |
| B3 ridge (log rpm)                            |         59.62 |          86.46 |             2.53 |      109.79 |       596.24 |           4.89 |
| M0 LightGBM log-rpm                           |         26.63 |          39.67 |             1.16 |       77.2  |       591.11 |           3.53 |
| A1 - quote_signal                             |         37.07 |          53.23 |             1.62 |       87.51 |       592.27 |           3.99 |
| A2 - holiday features                         |         26.66 |          39.61 |             1.16 |       77.23 |       591.06 |           3.53 |
| A9 no unseen-city blanking                    |         28.04 |          41.24 |             1.25 |       78.6  |       591.2  |           3.62 |
| A12 market: weekly cycle only (no slow level) |         28.75 |          43.86 |             1.25 |       79.29 |       591.36 |           3.62 |
| A21 cycle - holiday features                  |         29.21 |          44.71 |             1.27 |       79.74 |       591.34 |           3.64 |
| A24 cycle - holidays + additive calendar      |         42.9  |          67.55 |             1.79 |       93.22 |       593.37 |           4.16 |
| A27 A24 + qs_day (quote regime)               |         27.13 |          40.81 |             1.19 |       77.7  |       591.25 |           3.56 |
| A34 A27 + recency half-life 60d               |         31.3  |          46.89 |             1.37 |       81.81 |       591.4  |           3.74 |
| A39 A24 + coarse quote regime                 |         33.23 |          52.47 |             1.43 |       83.69 |       592.11 |           3.79 |
| A40 A39 + recency half-life 60d               |         38.4  |          58.14 |             1.67 |       88.8  |       592.29 |           4.04 |
| A41 A24 + qs_14d (continuous)                 |         30.69 |          47.3  |             1.32 |       81.2  |       591.41 |           3.69 |
| A42 A24 + qs_7d + quote x regime              |         28.43 |          43.1  |             1.24 |       78.97 |       591.2  |           3.61 |
| A43 A24 + qs_14d + quote x regime             |         30.56 |          46.95 |             1.32 |       81.07 |       591.41 |           3.68 |
| A44 A42 + recency half-life 60d               |         31.54 |          47.26 |             1.38 |       82.04 |       591.37 |           3.75 |
| A45 A43 + recency half-life 60d               |         34.09 |          51.53 |             1.47 |       84.55 |       591.64 |           3.84 |
| A46 A41 + recency half-life 60d               |         34.29 |          51.81 |             1.48 |       84.75 |       591.68 |           3.85 |

Per-fold MAE, clean labels (city):

| model                                         |    C1 |    C2 |    C3 |    C4 |    C5 |    C6 |    C7 |    C8 |
|:----------------------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|
| B1 global median rpm x distance               | 186.9 | 283.5 | 214.9 | 198.4 | 221.3 | 196.4 | 227.1 | 175.8 |
| B2 lane median rpm                            |  87.9 | 107.7 |  96.9 |  92.4 |  93.8 |  90.5 | 100.5 |  89.1 |
| B3 ridge (log rpm)                            |  50.3 |  69.7 |  58.2 |  55.5 |  65.2 |  54.9 |  67.7 |  55.5 |
| M0 LightGBM log-rpm                           |  23.2 |  31.4 |  27.3 |  25.3 |  25.7 |  25.8 |  29.2 |  25.1 |
| A1 - quote_signal                             |  32.3 |  42.9 |  37.7 |  34.5 |  35.9 |  36.6 |  41.4 |  35.3 |
| A2 - holiday features                         |  23.4 |  31.4 |  27.1 |  25.4 |  25.6 |  25.9 |  29.3 |  25.3 |
| A9 no unseen-city blanking                    |  24.8 |  31.9 |  29.7 |  26   |  26.9 |  26.7 |  32   |  26.4 |
| A12 market: weekly cycle only (no slow level) |  25   |  33.8 |  29.1 |  27.5 |  27.8 |  27.7 |  31.8 |  27.3 |
| A21 cycle - holiday features                  |  25.5 |  34.3 |  29.6 |  27.9 |  28   |  28.2 |  32.3 |  27.8 |
| A24 cycle - holidays + additive calendar      |  37.8 |  50.7 |  42.9 |  41.5 |  41.8 |  41.4 |  46.6 |  40.4 |
| A27 A24 + qs_day (quote regime)               |  24.1 |  31.7 |  27.6 |  26.2 |  26.3 |  26.5 |  29.6 |  25.1 |
| A34 A27 + recency half-life 60d               |  28.1 |  36.5 |  32.3 |  29.9 |  30.4 |  30.6 |  33.7 |  28.9 |
| A39 A24 + coarse quote regime                 |  29.8 |  39.7 |  33.8 |  32.3 |  32.2 |  31.6 |  35.9 |  30.5 |
| A40 A39 + recency half-life 60d               |  34.5 |  45.5 |  39.5 |  37.2 |  37.6 |  37   |  41.1 |  34.9 |
| A41 A24 + qs_14d (continuous)                 |  26.9 |  36.5 |  31.3 |  29.6 |  29.7 |  29.4 |  33.8 |  28.3 |
| A42 A24 + qs_7d + quote x regime              |  25.2 |  33.3 |  28.9 |  27.4 |  27.4 |  27.4 |  31.3 |  26.5 |
| A43 A24 + qs_14d + quote x regime             |  26.8 |  36.1 |  31.1 |  29.3 |  29.7 |  29.5 |  33.6 |  28.4 |
| A44 A42 + recency half-life 60d               |  28.1 |  36.8 |  32.3 |  30.3 |  31   |  30.7 |  34.3 |  28.8 |
| A45 A43 + recency half-life 60d               |  29.8 |  40.2 |  34.8 |  33.2 |  33.4 |  32.9 |  37.2 |  31.2 |
| A46 A41 + recency half-life 60d               |  29.9 |  40.8 |  35.1 |  33.2 |  33.5 |  32.9 |  37.5 |  31.4 |

Fold sizes:

| fold   |   n_train |   n_train_dropped |   n_test |
|:-------|----------:|------------------:|---------:|
| C1     |     36930 |               512 |    11070 |
| C2     |     35237 |               513 |    12763 |
| C3     |     35729 |               503 |    12271 |
| C4     |     37239 |               517 |    10761 |
| C5     |     37257 |               517 |    10743 |
| C6     |     37588 |               534 |    10412 |
| C7     |     36366 |               514 |    11634 |
| C8     |     37013 |               532 |    10987 |

## Scheme: random (mean over folds)

| model                                    |   MAE (clean) |   RMSE (clean) |   MAPE % (clean) |   MAE (raw) |   RMSE (raw) |   MAPE % (raw) |
|:-----------------------------------------|--------------:|---------------:|-----------------:|------------:|-------------:|---------------:|
| B1 global median rpm x distance          |        213.48 |         315.08 |             9.41 |      261.11 |       658.81 |          11.64 |
| B2 lane median rpm                       |         90.88 |         131.12 |             3.85 |      140.36 |       602.08 |           6.21 |
| B3 ridge (log rpm)                       |         54.76 |          79.51 |             2.33 |      104.69 |       592.61 |           4.7  |
| M0 LightGBM log-rpm                      |         23.78 |          35.72 |             1.01 |       74.1  |       587.85 |           3.4  |
| A21 cycle - holiday features             |         26.57 |          40.92 |             1.13 |       76.85 |       588.23 |           3.51 |
| A24 cycle - holidays + additive calendar |         40.95 |          64.6  |             1.68 |       91.01 |       590.16 |           4.06 |
| A27 A24 + qs_day (quote regime)          |         24.91 |          37.66 |             1.07 |       75.22 |       588.04 |           3.45 |
| A34 A27 + recency half-life 60d          |         28.63 |          43.35 |             1.23 |       78.9  |       588.25 |           3.62 |
| A39 A24 + coarse quote regime            |         31.77 |          50.25 |             1.34 |       81.97 |       589.03 |           3.72 |
| A40 A39 + recency half-life 60d          |         36.66 |          55.68 |             1.56 |       86.81 |       589.33 |           3.95 |
| A41 A24 + qs_14d (continuous)            |         28.35 |          43.99 |             1.19 |       78.59 |       588.25 |           3.57 |
| A42 A24 + qs_7d + quote x regime         |         26.26 |          40.21 |             1.11 |       76.54 |       588.05 |           3.5  |
| A43 A24 + qs_14d + quote x regime        |         28.29 |          43.73 |             1.19 |       78.54 |       588.27 |           3.57 |
| A44 A42 + recency half-life 60d          |         28.84 |          43.72 |             1.24 |       79.09 |       588.18 |           3.62 |
| A45 A43 + recency half-life 60d          |         31.31 |          47.76 |             1.33 |       81.51 |       588.38 |           3.71 |
| A46 A41 + recency half-life 60d          |         31.44 |          47.96 |             1.33 |       81.64 |       588.43 |           3.72 |

Per-fold MAE, clean labels (random):

| model                                    |    R1 |    R2 |    R3 |    R4 |    R5 |
|:-----------------------------------------|------:|------:|------:|------:|------:|
| B1 global median rpm x distance          | 214.3 | 213.2 | 213.2 | 214.3 | 212.4 |
| B2 lane median rpm                       |  92.1 |  90.6 |  89.4 |  91.7 |  90.6 |
| B3 ridge (log rpm)                       |  54.8 |  55   |  54.3 |  55.1 |  54.6 |
| M0 LightGBM log-rpm                      |  24.1 |  23.9 |  23.3 |  23.8 |  23.8 |
| A21 cycle - holiday features             |  26.9 |  26.4 |  26.1 |  26.5 |  26.9 |
| A24 cycle - holidays + additive calendar |  41.2 |  40.9 |  40.5 |  40.5 |  41.7 |
| A27 A24 + qs_day (quote regime)          |  24.9 |  24.9 |  24.6 |  25   |  25.1 |
| A34 A27 + recency half-life 60d          |  28.5 |  28.7 |  28.8 |  28.6 |  28.5 |
| A39 A24 + coarse quote regime            |  31.9 |  31.8 |  31.3 |  31.7 |  32.2 |
| A40 A39 + recency half-life 60d          |  36.7 |  36.9 |  36.2 |  36.4 |  37.1 |
| A41 A24 + qs_14d (continuous)            |  28.5 |  28.1 |  27.9 |  28.6 |  28.6 |
| A42 A24 + qs_7d + quote x regime         |  26.4 |  26.2 |  25.8 |  26.1 |  26.7 |
| A43 A24 + qs_14d + quote x regime        |  28.6 |  27.9 |  28   |  28.5 |  28.5 |
| A44 A42 + recency half-life 60d          |  29   |  28.6 |  28.5 |  28.6 |  29.4 |
| A45 A43 + recency half-life 60d          |  31.4 |  31.1 |  31.2 |  31.5 |  31.4 |
| A46 A41 + recency half-life 60d          |  31.5 |  31.2 |  31.2 |  31.6 |  31.6 |

Fold sizes:

| fold   |   n_train |   n_train_dropped |   n_test |
|:-------|----------:|------------------:|---------:|
| R1     |     38400 |               548 |     9600 |
| R2     |     38400 |               545 |     9600 |
| R3     |     38400 |               537 |     9600 |
| R4     |     38400 |               541 |     9600 |
| R5     |     38400 |               537 |     9600 |
