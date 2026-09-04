"""
PHASE 1d -- CORRECTED methodology: genuine 24-hour-ahead forecast.

Phase 1/1b/1c used lag_1block (demand 15 min before the target) as a
feature. That's only valid for a 15-minute nowcast -- for a next-day
forecast, "15 minutes before target" does not exist yet at the moment
you'd issue the forecast. This script removes that leak and keeps only
features genuinely available >=24h before the target time:
  - same block, 24h before target (i.e. "now", at forecast-issue time)
  - same block, 7 days before target
  - calendar (always known in advance)
  - temperature at target time -- ASSUMPTION: treated as if a perfect
    24h-ahead weather forecast were available. Real weather-forecast
    error is a real, separate source of uncertainty not modeled here.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOAD = BASE / "data" / "load_data.csv"
WEATHER = BASE / "data" / "delhi_weather_hourly.csv"
OUT = BASE / "outputs"

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

report = []
report.append("PHASE 1d -- CORRECTED: genuine 24h-ahead forecast (no near-term leakage)")
report.append("=" * 78)

raw = pd.read_csv(LOAD, parse_dates=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp")
s15 = raw["load_MW"].resample("15min").mean().interpolate(limit=4)

wx = pd.read_csv(WEATHER, parse_dates=["timestamp"]).drop_duplicates(subset="timestamp").set_index("timestamp")
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")

feat = pd.DataFrame(index=s15.index)
feat["load_MW"] = s15
feat["temp_C"] = wx15.reindex(feat.index)
feat = feat.dropna(subset=["temp_C"])

feat["hour"] = feat.index.hour + feat.index.minute / 60
feat["dow"] = feat.index.dayofweek
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
feat["sin_hod"] = np.sin(2 * np.pi * feat["hour"] / 24)
feat["cos_hod"] = np.cos(2 * np.pi * feat["hour"] / 24)
feat["sin_doy"] = np.sin(2 * np.pi * feat.index.dayofyear / 365.25)
feat["cos_doy"] = np.cos(2 * np.pi * feat.index.dayofyear / 365.25)

# ONLY features available >=24h before the target -- no lag_1block this time
feat["same_block_24h_ago"] = feat["load_MW"].shift(96)        # = "now" at a 24h-ahead forecast issue time
feat["same_block_7d_ago"] = feat["load_MW"].shift(96 * 7)
feat["cdh"] = np.clip(feat["temp_C"] - 24, 0, None)
feat["cdh_sq"] = np.clip(feat["temp_C"] - 38, 0, None) ** 2
feat = feat.dropna()

feature_cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "is_weekend",
                 "same_block_24h_ago", "same_block_7d_ago", "temp_C", "cdh", "cdh_sq"]

def fit_eval(train, test, cols):
    Xtr = np.column_stack([np.ones(len(train)), train[cols].values])
    ytr = train["load_MW"].values
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte = np.column_stack([np.ones(len(test)), test[cols].values])
    pred = Xte @ coef
    return mape(test["load_MW"], pred), pred

train = feat[feat.index < "2025-05-01"]
test = feat[(feat.index >= "2025-05-01") & (feat.index < "2025-07-01")]
report.append(f"[CONFIRMED] Train: before 2025-05-01 ({len(train):,} blocks) | Test: 2025-05-01 to 2025-06-30 ({len(test):,} blocks, summer peak season)")

naive_pred = test["same_block_7d_ago"]
naive_mape = mape(test["load_MW"], naive_pred)

model_mape, model_pred = fit_eval(train, test, feature_cols)

report.append("")
report.append(f"[CONFIRMED] Naive baseline (same block, 7 days ago -- also a legitimate >=24h-ahead-available baseline): MAPE = {naive_mape:.2f}%")
report.append(f"[CONFIRMED] Corrected 24h-ahead model (calendar + 24h-ago + 7d-ago + temperature, NO near-term leak): MAPE = {model_mape:.2f}%")
report.append(f"[CONFIRMED] Improvement over naive: {naive_mape - model_mape:+.2f} pp MAPE")
report.append("")
report.append(f"[ASSUMPTION] Temperature at target time is treated as a perfect 24h-ahead forecast (we used real observed")
report.append(f"             temperature, not a real weather forecast with its own error). This is standard practice for a")
report.append(f"             prototype but should be disclosed as such, not presented as production-grade weather-forecast accuracy.")

peak_row = test["load_MW"].idxmax()
peak_actual = test.loc[peak_row, "load_MW"]
peak_pred = model_pred[test.index.get_loc(peak_row)]
report.append("")
report.append(f"[CONFIRMED] Highest actual demand block: {peak_row} -> {peak_actual:.1f} MW")
report.append(f"[CONFIRMED] Corrected model's 24h-ahead prediction for that block: {peak_pred:.1f} MW "
              f"(error: {peak_pred - peak_actual:+.1f} MW, {abs(peak_pred-peak_actual)/peak_actual*100:.2f}%)")

out = test[["load_MW", "temp_C"]].copy()
out["naive_pred_MW"] = naive_pred
out["model_pred_MW"] = model_pred
out.to_csv(OUT / "phase1d_test_predictions.csv")

text = "\n".join(report)
(OUT / "phase1d_report.txt").write_text(text, encoding="utf-8")
print(text)
