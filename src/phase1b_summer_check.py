"""
PHASE 1b -- same model, tested specifically against a real summer peak
window instead of the winter holdout. This directly matters because the
problem statement is about summer AC-driven demand, not winter load.
No new features, no dashboard -- still Phase 1 scope.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "load_data.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs"

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

raw = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp")
s15 = raw["load_MW"].resample("15min").mean().interpolate(limit=4)

feat = pd.DataFrame(index=s15.index)
feat["load_MW"] = s15
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
feat = feat.dropna()

feature_cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy", "is_weekend",
                 "lag_1block", "lag_1day", "lag_1week"]

report = []
report.append("PHASE 1b -- summer-specific validation (real peak-demand season)")
report.append("=" * 78)

# Train on everything before 1 May 2025, test on the summer peak May-Jun 2025
train = feat[feat.index < "2025-05-01"]
test = feat[(feat.index >= "2025-05-01") & (feat.index < "2025-07-01")]

Xtr = np.column_stack([np.ones(len(train)), train[feature_cols].values])
ytr = train["load_MW"].values
coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

Xte = np.column_stack([np.ones(len(test)), test[feature_cols].values])
model_pred = Xte @ coef
naive_pred = test["lag_1week"]

model_mape = mape(test["load_MW"], model_pred)
naive_mape = mape(test["load_MW"], naive_pred)

peak_row = test["load_MW"].idxmax()
peak_actual = test.loc[peak_row, "load_MW"]
peak_pred = model_pred[test.index.get_loc(peak_row)]

report.append(f"[CONFIRMED] Train: before 2025-05-01 ({len(train):,} blocks) | Test: 2025-05-01 to 2025-06-30 ({len(test):,} blocks, real summer peak season)")
report.append(f"[CONFIRMED] Naive baseline (same block last week): MAPE = {naive_mape:.2f}%")
report.append(f"[CONFIRMED] Phase-1 OLS model (no weather):        MAPE = {model_mape:.2f}%")
report.append(f"[CONFIRMED] Highest actual demand block in this summer window: {peak_row} -> {peak_actual:.1f} MW")
report.append(f"[CONFIRMED] Model's prediction for that exact block: {peak_pred:.1f} MW  (error: {peak_pred - peak_actual:+.1f} MW, {abs(peak_pred-peak_actual)/peak_actual*100:.2f}%)")

text = "\n".join(report)
(OUT / "phase1b_summer_report.txt").write_text(text, encoding="utf-8")
print(text)
