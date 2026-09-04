"""
PHASE 1c -- same OLS approach as Phase 1/1b, now with real temperature
added as a feature. Still Phase 1 scope: proving the forecast, nothing
about DISCOMs/stress/dashboard here.
Weather source: CONFIRMED real data, Open-Meteo archive API (ERA5-based
reanalysis), Delhi coordinates (28.6139N, 77.2090E), hourly, fetched live.
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
report.append("PHASE 1c -- baseline + real temperature (real data, held-out validation)")
report.append("=" * 78)

# ---- load demand, resample to 15-min ----
raw = pd.read_csv(LOAD, parse_dates=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp")
s15 = raw["load_MW"].resample("15min").mean().interpolate(limit=4)

# ---- load weather (hourly, CONFIRMED real), upsample to 15-min via interpolation ----
wx = pd.read_csv(WEATHER, parse_dates=["timestamp"]).drop_duplicates(subset="timestamp").set_index("timestamp")
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")
report.append(f"[CONFIRMED] Weather range: {wx.index.min()} to {wx.index.max()} ({len(wx):,} hourly readings, Open-Meteo)")
report.append(f"[CONFIRMED->INFERENCE] Temperature upsampled hourly->15min via time-interpolation (weather itself changes slowly, this is a standard and defensible step)")

feat = pd.DataFrame(index=s15.index)
feat["load_MW"] = s15
feat["temp_C"] = wx15.reindex(feat.index)
feat = feat.dropna(subset=["temp_C"])  # keep only rows where we have real weather overlap
report.append(f"[CONFIRMED] Rows with both real demand and real temperature available: {len(feat):,}")

feat["hour"] = feat.index.hour + feat.index.minute / 60
feat["dow"] = feat.index.dayofweek
feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
feat["sin_hod"] = np.sin(2 * np.pi * feat["hour"] / 24)
feat["cos_hod"] = np.cos(2 * np.pi * feat["hour"] / 24)
feat["sin_doy"] = np.sin(2 * np.pi * feat.index.dayofyear / 365.25)
feat["cos_doy"] = np.cos(2 * np.pi * feat.index.dayofyear / 365.25)
feat["lag_1block"] = feat["load_MW"].shift(1)
feat["lag_1day"] = feat["load_MW"].shift(96)
feat["lag_1week"] = feat["load_MW"].shift(96 * 7)
# cooling-degree feature: AC saturation kicks in above ~24C, and accelerates further above ~38C
feat["cdh"] = np.clip(feat["temp_C"] - 24, 0, None)
feat["cdh_sq"] = np.clip(feat["temp_C"] - 38, 0, None) ** 2   # nonlinear AC-saturation term
feat = feat.dropna()

feature_cols_base = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "is_weekend",
                      "lag_1block", "lag_1day", "lag_1week"]
feature_cols_weather = feature_cols_base + ["temp_C", "cdh", "cdh_sq"]

def fit_eval(train, test, cols):
    Xtr = np.column_stack([np.ones(len(train)), train[cols].values])
    ytr = train["load_MW"].values
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte = np.column_stack([np.ones(len(test)), test[cols].values])
    pred = Xte @ coef
    return mape(test["load_MW"], pred), pred, coef

# Same summer-2025 holdout as Phase 1b, for a direct apples-to-apples comparison
train = feat[feat.index < "2025-05-01"]
test = feat[(feat.index >= "2025-05-01") & (feat.index < "2025-07-01")]
report.append(f"[CONFIRMED] Train: before 2025-05-01 ({len(train):,} blocks) | Test: 2025-05-01 to 2025-06-30 ({len(test):,} blocks)")

mape_no_weather, _, _ = fit_eval(train, test, feature_cols_base)
mape_weather, pred_weather, coef = fit_eval(train, test, feature_cols_weather)

report.append("")
report.append(f"[CONFIRMED] Same model, same split, calendar+lag only (no weather): MAPE = {mape_no_weather:.2f}%")
report.append(f"[CONFIRMED] Same model + real temperature + cooling-degree terms:  MAPE = {mape_weather:.2f}%")
report.append(f"[CONFIRMED] Weather's effect on this summer window: {mape_no_weather - mape_weather:+.2f} pp MAPE")

peak_row = test["load_MW"].idxmax()
peak_actual = test.loc[peak_row, "load_MW"]
peak_pred = pred_weather[test.index.get_loc(peak_row)]
peak_temp = test.loc[peak_row, "temp_C"]
report.append("")
report.append(f"[CONFIRMED] Highest actual demand block: {peak_row} -> {peak_actual:.1f} MW (temp: {peak_temp:.1f}C)")
report.append(f"[CONFIRMED] Weather-aware model's prediction for that block: {peak_pred:.1f} MW "
              f"(error: {peak_pred - peak_actual:+.1f} MW, {abs(peak_pred-peak_actual)/peak_actual*100:.2f}%)")

text = "\n".join(report)
(OUT / "phase1c_weather_report.txt").write_text(text, encoding="utf-8")
print(text)
