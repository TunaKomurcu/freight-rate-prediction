# EDA and data-quality summary (raw data)

### train_test (48,000 rows)
- Missing values: weight=300, market_index=374
- Duplicate rows: 0, duplicate load_id: 0, duplicate rows ignoring load_id: 0
- Date range: 2025-01-01 .. 2025-10-31 (304 days, none unparseable)
- Equipment values: {'Dry Van': 27202, 'Reefer': 12045, 'Flatbed': 8753}
- Cities: 64 distinct; whitespace/case variants: 0; pickup == delivery: 0
- Weight: negative=292, zero=0, at cap 47,500=1204, at floor 5,000=32, range -47,500..47,500
- distance / haversine: median 1.182, p1 1.130, p99 1.404, max 9.6; rows with ratio > 2: 22
- Coordinates: lat 28.36..44.30, lon -121.70..-69.50; lat/lon swapped (lat<0 or lon>0): 0
- Distance range 70..3440 mi; non-positive: 0
- market_index range 0.676..1.468; quote_signal range 0.692..3.610
### validation (12,000 rows)
- Missing values: weight=165, market_index=249
- Duplicate rows: 0, duplicate load_id: 0, duplicate rows ignoring load_id: 0
- Date range: 2025-11-01 .. 2025-12-31 (61 days, none unparseable)
- Equipment values: {'Dry Van': 6780, 'Reefer': 3051, 'Flatbed': 2169}
- Cities: 72 distinct; whitespace/case variants: 0; pickup == delivery: 0
- Weight: negative=145, zero=0, at cap 47,500=301, at floor 5,000=7, range -47,500..47,500
- distance / haversine: median 1.182, p1 1.128, p99 1.414, max 78.5; rows with ratio > 2: 16
- Coordinates: lat 25.50..44.30, lon -121.70..-69.50; lat/lon swapped (lat<0 or lon>0): 0
- Distance range 70..3326 mi; non-positive: 0
- market_index range 0.724..1.099; quote_signal range 1.235..2.896
### Cross-file checks
- Cities with >1 coordinate pair: 0
- Coordinates sitting exactly on a round bound (lon=-69.5 / lat=25.5, likely clipped): ['Boston', 'Laredo', 'Providence']
- Cities only in validation: ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego'] -> 1,447 validation rows (12.1%)
- Validation rows whose lane appears in train: 87.8%
- Lanes in train: 4,014, median loads per lane: 10
### Target
- posted_rate: min 57.22, median 2030.76, max 25,533.00; non-positive: 0
- rate per mile: p1 1.67, median 2.15, p99 3.18, max 14.13
- Lane-relative rate (rpm / lane-equipment median): p1 0.886, p99 1.120; < 0.6: 338, > 1.7: 337, in (0.6,0.8)U(1.3,1.7): 5
### Leakage check: quote_signal
- corr(quote_signal, posted_rate) = -0.040; corr(quote_signal, rate per mile) = 0.049; corr(quote_signal, lane-relative rate, clean rows) = 0.070
- corr(quote_signal * distance, posted_rate) = 0.899 vs corr(distance, posted_rate) = 0.909
### market_index
- Within-day std 0.025 vs total std 0.168: market_index is a daily market series plus small per-load noise
- corr(daily mean market_index, daily mean lane-relative rate) = 0.706; row-level corr = 0.452
### December chart inputs
- Columns: ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']; dates 2025-12-01..2025-12-31; no lat/lon, market_index or quote_signal

### quote_signal regimes (key finding)
- Weekly corr(quote_signal, lane-relative rate) on loads > 800 mi ranges -0.32..0.49; it tracks the weekly mean quote (r = 0.87): positive when the mean quote is high (> 2.1: late Feb, Mar, Jun, Sep), negative when low (< 2.0: Apr, May, Jul, Oct), ~0 in between (Aug). Pooled over all weeks these cancel out, which is why the raw correlation looked ~0.

