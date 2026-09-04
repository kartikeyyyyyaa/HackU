"""
PHASE 1 - Baseline demand forecast, validated on real held-out data.
No dashboard, no DISCOM split, no stress score here by design -- this
script's only job is to prove a credible forecast is possible from
information that would genuinely be available in advance.

Data provenance: CONFIRMED real historical data (Delhi SLDC, via Kaggle
dataset prash4nt/delhi-sldc-load-data-5-min-resolution), 1 Apr 2023 -
12 Jan 2026, 5-minute resolution.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "load_data.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

report = []
report.append("PHASE 1 REPORT -- baseline demand forecast (real data, held-out validation)")
report.append("=" * 78)

# ---------- 1. Load + data-quality pass (CONFIRMED facts about the real file) ----------
raw = pd.read_csv(DATA, parse_dates=["timestamp"])
n_raw = len(raw)
dupes = raw.duplicated(subset="timestamp").sum()
raw = raw.sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp")

full_5min = pd.date_range(raw.index.min(), raw.index.max(), freq="5min")
missing_5min = full_5min.difference(raw.index)
missing_pct = 100 * len(missing_5min) / len(full_5min)

bad = raw[(raw["load_MW"] <= 0) | (raw["load_MW"] > 15000)]
n_bad = len(bad)
clean = raw.drop(index=bad.index)

report.append(f"[CONFIRMED] Raw rows: {n_raw:,} | duplicate timestamps dropped: {dupes:,}")
report.append(f"[CONFIRMED] Date range: {raw.index.min()} to {raw.index.max()}")
report.append(f"[CONFIRMED] Missing 5-min timestamps vs a complete calendar: {len(missing_5min):,} ({missing_pct:.2f}%)")
report.append(f"[CONFIRMED] Rows dropped as physically implausible (<=0 MW or >15,000 MW): {n_bad:,}")

# ---------- 2. Resample to 15-min ABT/DSM blocks ----------
s15 = clean["load_MW"].resample("15min").mean()
n_before_interp = s15.isna().sum()
s15 = s15.interpolate(limit=4)  # fill gaps up to 1 hour; longer gaps stay NaN and get dropped later
report.append(f"[CONFIRMED->INFERENCE] Resampled to 15-min blocks: {len(s15):,} blocks; "
              f"{n_before_interp:,} empty blocks, gaps up to 1hr linearly interpolated (labeled, not hidden)")

# ---------- 3. Feature engineering (time-pattern only -- no weather in Phase 1) ----------
feat = pd.DataFrame(index=s15.index)
feat["load_MW"] = s15
feat["hour"] = feat.index.hour + feat.index.minute / 60
feat["dow"] = feat.index.dayofweek
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
feat["sin_hod"] = np.sin(2 * np.pi * feat["hour"] / 24)
feat["cos_hod"] = np.cos(2 * np.pi * feat["hour"] / 24)
feat["sin_doy"] = np.sin(2 * np.pi * feat.index.dayofyear / 365.25)
feat["cos_doy"] = np.cos(2 * np.pi * feat.index.dayofyear / 365.25)
feat["lag_1block"] = feat["load_MW"].shift(1)          # 15 min ago
feat["lag_1day"] = feat["load_MW"].shift(96)            # same block, yesterday
feat["lag_1week"] = feat["load_MW"].shift(96 * 7)        # same block, last week
feat = feat.dropna()

report.append(f"[CONFIRMED] Feature set: calendar (hour-of-day, day-of-year Fourier terms, weekend flag) "
              f"+ lag features (15min/1day/1week ago). NOTE: weather is NOT yet included -- Phase 1 tests "
              f"whether load pattern alone is predictive; weather is a planned Phase 1b addition.")

# ---------- 4. Chronological train/test split (no leakage) ----------
test_start = feat.index.max() - pd.Timedelta(days=45)
train = feat[feat.index < test_start]
test = feat[feat.index >= test_start]
report.append(f"[CONFIRMED] Train: {train.index.min()} to {train.index.max()} ({len(train):,} blocks)")
report.append(f"[CONFIRMED] Test (held out, never seen by the model): {test.index.min()} to {test.index.max()} ({len(test):,} blocks)")

# ---------- 5. Naive baseline: "same block, last week" ----------
naive_pred = test["lag_1week"]
naive_mape = mape(test["load_MW"], naive_pred)
naive_rmse = rmse(test["load_MW"], naive_pred)
report.append("")
report.append(f"[CONFIRMED] Naive baseline (persistence: same 15-min block last week)")
report.append(f"            MAPE = {naive_mape:.2f}%   RMSE = {naive_rmse:.1f} MW")

# ---------- 6. Real model: OLS linear regression (closed-form, numpy only) ----------
feature_cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "is_weekend",
                 "lag_1block", "lag_1day", "lag_1week"]
Xtr = np.column_stack([np.ones(len(train)), train[feature_cols].values])
ytr = train["load_MW"].values
coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

Xte = np.column_stack([np.ones(len(test)), test[feature_cols].values])
model_pred = Xte @ coef
model_mape = mape(test["load_MW"], model_pred)
model_rmse = rmse(test["load_MW"], model_pred)

report.append("")
report.append(f"[CONFIRMED] Phase-1 model: OLS linear regression on calendar + lag features (numpy, no external ML lib)")
report.append(f"            MAPE = {model_mape:.2f}%   RMSE = {model_rmse:.1f} MW")
report.append(f"            Improvement over naive baseline: {naive_mape - model_mape:+.2f} pp MAPE")

# ---------- 7. Save test predictions for later use (dashboard / proof mode) ----------
out = test[["load_MW"]].copy()
out["naive_pred_MW"] = naive_pred
out["model_pred_MW"] = model_pred
out.to_csv(OUT / "phase1_test_predictions.csv")
report.append("")
report.append(f"[CONFIRMED] Test-period predictions saved to outputs/phase1_test_predictions.csv ({len(out):,} rows)")

report_text = "\n".join(report)
(OUT / "phase1_report.txt").write_text(report_text, encoding="utf-8")
print(report_text)
