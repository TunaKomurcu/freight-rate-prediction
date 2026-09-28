# Validation results

Dollar-scale metrics. *clean* = test rows whose label is not flagged as corrupted; *raw* = all test rows (Spotter's labels likely contain the same ~1.4% corruption).

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
| A14 + 28-day level anchor                            |         70    |         106.56 |             3.11 |      123.51 |       633.48 |           5.39 |
| A15 cycle + level anchor                             |         64.48 |          94.24 |             2.78 |      118.09 |       632.29 |           5.09 |
| A16 cycle + qs dev                                   |         89.11 |         120.12 |             3.84 |      142.43 |       637.44 |           6.13 |
| A17 cycle + qs dev + level anchor                    |         88.69 |         119.58 |             3.82 |      142.02 |       637.38 |           6.11 |
| A18 cycle + qs dev + trend_q                         |         82.28 |         118.34 |             3.42 |      135.61 |       632.93 |           5.77 |
| A19 cycle + qs dev + trend_q + level anchor          |         81.39 |         117.45 |             3.38 |      134.74 |       632.83 |           5.73 |
| A20 cycle - quote_signal                             |        109.56 |         143.82 |             4.47 |      162.61 |       644.51 |           6.73 |
| A21 cycle - holiday features                         |         62.24 |          91.82 |             2.69 |      115.89 |       631.92 |           5.01 |
| A22 cycle + trend_q                                  |         64.67 |          97.22 |             2.66 |      118.21 |       628.44 |           5.03 |
| A23 cycle - holidays + smearing                      |         62.16 |          91.75 |             2.69 |      115.81 |       631.88 |           5    |
| A24 cycle - holidays + additive calendar             |         55.68 |          87.7  |             2.38 |      109.39 |       630.58 |           4.7  |
| A25 all market - holidays + additive calendar        |         74.64 |         112.74 |             3.28 |      128.08 |       634.75 |           5.56 |
| A26 cycle - holidays + additive calendar with market |         55.06 |          87.38 |             2.36 |      108.8  |       630.9  |           4.68 |
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
| A14 + 28-day level anchor                            |                    59.2 |                    69.9 |                    80.8 |
| A15 cycle + level anchor                             |                    67.3 |                    93.3 |                    32.9 |
| A16 cycle + qs dev                                   |                   111.4 |                   106.2 |                    49.7 |
| A17 cycle + qs dev + level anchor                    |                   111   |                   105.9 |                    49.2 |
| A18 cycle + qs dev + trend_q                         |                    73.8 |                    93.8 |                    79.3 |
| A19 cycle + qs dev + trend_q + level anchor          |                    73.6 |                    94   |                    76.6 |
| A20 cycle - quote_signal                             |                   148.9 |                   118.1 |                    61.7 |
| A21 cycle - holiday features                         |                    68.4 |                    85.5 |                    32.9 |
| A22 cycle + trend_q                                  |                    48   |                    83.6 |                    62.4 |
| A23 cycle - holidays + smearing                      |                    68.1 |                    85.4 |                    33   |
| A24 cycle - holidays + additive calendar             |                    56   |                    80.2 |                    30.8 |
| A25 all market - holidays + additive calendar        |                    61.4 |                    72.3 |                    90.2 |
| A26 cycle - holidays + additive calendar with market |                    47.4 |                    77.9 |                    39.9 |
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
| A14 + 28-day level anchor                            |                   632.3 |                   627.2 |                   641   |
| A15 cycle + level anchor                             |                   637.6 |                   631.6 |                   627.6 |
| A16 cycle + qs dev                                   |                   650   |                   633.8 |                   628.6 |
| A17 cycle + qs dev + level anchor                    |                   649.9 |                   633.7 |                   628.6 |
| A18 cycle + qs dev + trend_q                         |                   638.8 |                   629.8 |                   630.2 |
| A19 cycle + qs dev + trend_q + level anchor          |                   638.7 |                   629.9 |                   629.9 |
| A20 cycle - quote_signal                             |                   660.2 |                   641.8 |                   631.5 |
| A21 cycle - holiday features                         |                   638.6 |                   629.9 |                   627.3 |
| A22 cycle + trend_q                                  |                   631.3 |                   626.6 |                   627.3 |
| A23 cycle - holidays + smearing                      |                   638.5 |                   629.8 |                   627.3 |
| A24 cycle - holidays + additive calendar             |                   635.5 |                   628.7 |                   627.6 |
| A25 all market - holidays + additive calendar        |                   633.8 |                   626.9 |                   643.6 |
| A26 cycle - holidays + additive calendar with market |                   633.5 |                   628.3 |                   630.9 |
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
| A14 + 28-day level anchor                            |     -0.13 |     -1.95 |     -0.59 |     -2.99 |     -2.28 |     -4.97 |
| A15 cycle + level anchor                             |     -2.93 |     -2.06 |     -3.35 |     -0.43 |      0.88 |     -0.69 |
| A16 cycle + qs dev                                   |     -3.89 |     -5.61 |     -4.7  |      0.79 |      2.16 |     -0.48 |
| A17 cycle + qs dev + level anchor                    |     -3.87 |     -5.6  |     -4.66 |      0.84 |      2.07 |     -0.57 |
| A18 cycle + qs dev + trend_q                         |     -2.63 |     -3.39 |     -0.99 |      4.21 |      4.02 |      2.29 |
| A19 cycle + qs dev + trend_q + level anchor          |     -2.62 |     -3.38 |     -0.96 |      4.25 |      3.89 |      2.17 |
| A20 cycle - quote_signal                             |     -5.83 |     -6.1  |     -6.88 |      1.3  |      1.88 |     -1.5  |
| A21 cycle - holiday features                         |     -3.1  |     -2.06 |     -2.58 |     -0.88 |      1.13 |     -0.5  |
| A22 cycle + trend_q                                  |     -1.8  |     -0.08 |      0.11 |      3.01 |      2.8  |      2.18 |
| A23 cycle - holidays + smearing                      |     -3.09 |     -2.04 |     -2.57 |     -0.87 |      1.15 |     -0.48 |
| A24 cycle - holidays + additive calendar             |     -2.02 |     -1.88 |     -0.85 |     -0.84 |      0.41 |     -0.28 |
| A25 all market - holidays + additive calendar        |     -0.25 |     -2.32 |     -0.35 |     -2.64 |     -3.1  |     -4.92 |
| A26 cycle - holidays + additive calendar with market |     -1.17 |     -1.39 |     -0.52 |     -1.26 |     -1.47 |     -1.27 |
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
| A24 cycle - holidays + additive calendar |         40.95 |          64.61 |             1.68 |       91.01 |       590.16 |           4.06 |

Per-fold MAE, clean labels (random):

| model                                    |    R1 |    R2 |    R3 |    R4 |    R5 |
|:-----------------------------------------|------:|------:|------:|------:|------:|
| B1 global median rpm x distance          | 214.3 | 213.2 | 213.2 | 214.3 | 212.4 |
| B2 lane median rpm                       |  92.1 |  90.6 |  89.4 |  91.7 |  90.6 |
| B3 ridge (log rpm)                       |  54.8 |  55   |  54.3 |  55.1 |  54.6 |
| M0 LightGBM log-rpm                      |  24.1 |  23.9 |  23.3 |  23.8 |  23.8 |
| A21 cycle - holiday features             |  26.9 |  26.4 |  26.1 |  26.5 |  26.9 |
| A24 cycle - holidays + additive calendar |  41.2 |  40.9 |  40.5 |  40.5 |  41.7 |

Fold sizes:

| fold   |   n_train |   n_train_dropped |   n_test |
|:-------|----------:|------------------:|---------:|
| R1     |     38400 |               548 |     9600 |
| R2     |     38400 |               545 |     9600 |
| R3     |     38400 |               537 |     9600 |
| R4     |     38400 |               541 |     9600 |
| R5     |     38400 |               537 |     9600 |
