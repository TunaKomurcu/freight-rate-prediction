# Ensemble check: A34 A27 + recency half-life 60d + ridge

Weight on the tree tuned on T1-T2: 0.60

| model                           |   T1: Jan-Apr > May-Jun |   T2: Jan-Jun > Jul-Aug |   T3: Jan-Aug > Sep-Oct |   mean |   worst |
|:--------------------------------|------------------------:|------------------------:|------------------------:|-------:|--------:|
| A34 A27 + recency half-life 60d |                    30.2 |                    64.1 |                    26.8 |   40.4 |    64.1 |
| B3 ridge (log rpm)              |                    52.6 |                    52.3 |                   110.7 |   71.8 |   110.7 |
| blend w=0.60 (tuned on T1-T2)   |                    35.1 |                    50.7 |                    44.5 |   43.4 |    50.7 |
| blend w=0.50                    |                    37.4 |                    48.9 |                    54.5 |   47   |    54.5 |
