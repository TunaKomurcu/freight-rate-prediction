# Ensemble check: A40 A39 + recency half-life 60d + ridge

Weight on the tree tuned on T1-T2: 0.40

| model                           |   T1: Jan-Apr > May-Jun |   T2: Jan-Jun > Jul-Aug |   T3: Jan-Aug > Sep-Oct |   mean |   worst |
|:--------------------------------|------------------------:|------------------------:|------------------------:|-------:|--------:|
| A40 A39 + recency half-life 60d |                    42.5 |                    71.4 |                    32.4 |   48.8 |    71.4 |
| B3 ridge (log rpm)              |                    52.6 |                    52.3 |                   110.7 |   71.8 |   110.7 |
| blend w=0.40 (tuned on T1-T2)   |                    43.8 |                    46.7 |                    61.5 |   50.7 |    61.5 |
| blend w=0.50                    |                    42.5 |                    48.7 |                    50.4 |   47.2 |    50.4 |
