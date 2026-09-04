"""
PHASE 2 (FINAL) -- APPROVED CONTROLLED MODEL EXPERIMENT
========================================================
MODEL A = OLS linear regression  (the Phase 1 validated methodology)
MODEL B = XGBoost                (tree-based nonlinear)

Runs on the APPROVED corrected pipeline. No preprocessing or feature-
engineering changes have been made to that pipeline. Nothing is tuned
against the test set. Nothing is shuffled.

PRIMARY EXPERIMENT ("genuine forecast"): TIER 1 features only.
  Every feature is available at forecast-issue time = target - 24h.
  Target-time (future) temperature is EXCLUDED, per the approved rule that
  observed future weather must not enter the genuine-forecast experiment.

SUPPLEMENTARY (clearly separated, NOT the headline): TIER 2 adds target-time
  observed temperature. It is an ASSUMPTION standing in for a 24h weather
  forecast we do not possess, and only measures how much weather COULD add.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACKU_DATA", BASE / "data"))
OUT_DIR = Path(os.environ.get("HACKU_OUT", BASE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- FIXED, APPROVED EVALUATION SETTINGS -- do not change ----
H = 96                                    # 24h at 15-min resolution
TRAIN_END = "2025-05-01"
TEST_START, TEST_END = "2025-05-01", "2025-07-01"
PEAK_QUANTILE = 0.90
KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
KNOWN_PEAK_MW = 8392.6
SEED = 42

XGB_PARAMS = dict(                        # ONE fixed configuration. No search.
    n_estimators=800, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
    objective="reg:squarederror", tree_method="hist",
    random_state=SEED, n_jobs=-1,
)

INK, COL_A, COL_B = "#2f3437", "#0d76b8", "#c9701a"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 2 (FINAL) -- APPROVED MODEL EXPERIMENT: OLS vs XGBoost")
say("=" * 92)

# ============================================================================
# APPROVED PIPELINE -- reproduced verbatim, unchanged
# ============================================================================
load = (pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp"))
s15 = load["load_MW"].resample("15min").mean().interpolate(limit=4)   # TARGET: never imputed further

s15_feat = s15.copy()                                                 # FEATURE INPUTS only
for _ in range(6):
    s15_feat = s15_feat.fillna(s15_feat.shift(H * 7))
s15_feat = s15_feat.interpolate(limit_direction="both")

wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")

g = pd.DataFrame({"load_MW": s15, "load_feat": s15_feat})
g["temp_C"] = wx15.reindex(g.index)
g = g.dropna(subset=["temp_C", "load_feat"])
lf = g["load_feat"]

g["block_of_day"] = (g.index.hour * 4 + g.index.minute // 15).astype(int)
g["hour"] = g.index.hour + g.index.minute / 60
g["sin_hod"] = np.sin(2 * np.pi * g["hour"] / 24)
g["cos_hod"] = np.cos(2 * np.pi * g["hour"] / 24)
g["dow"] = g.index.dayofweek
g["is_weekend"] = (g["dow"] >= 5).astype(int)
g["month"] = g.index.month
g["doy"] = g.index.dayofyear
g["sin_doy"] = np.sin(2 * np.pi * g["doy"] / 365.25)
g["cos_doy"] = np.cos(2 * np.pi * g["doy"] / 365.25)

g["lag_24h"] = lf.shift(H)
g["lag_48h"] = lf.shift(H * 2)
g["lag_7d"] = lf.shift(H * 7)
g["lag_14d"] = lf.shift(H * 14)
past = lf.shift(H)
g["roll24_mean"] = past.rolling(H, min_periods=int(H * .75)).mean()
g["roll24_max"] = past.rolling(H, min_periods=int(H * .75)).max()
g["roll24_min"] = past.rolling(H, min_periods=int(H * .75)).min()
g["roll7d_mean"] = past.rolling(H * 7, min_periods=int(H * 7 * .75)).mean()
tpast = g["temp_C"].shift(H)
g["temp_at_issue"] = tpast
g["temp_prevday_max"] = tpast.rolling(H, min_periods=int(H * .75)).max()
g["temp_prevday_mean"] = tpast.rolling(H, min_periods=int(H * .75)).mean()

# TIER 2 columns (supplementary only)
g["temp_target"] = g["temp_C"]
g["cdh_target"] = np.clip(g["temp_C"] - 24, 0, None)
g["temp_target_sq"] = g["temp_C"] ** 2
g["heat_extreme"] = np.clip(g["temp_C"] - 38, 0, None) ** 2

f = g.dropna()

TIER1 = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
         "doy", "sin_doy", "cos_doy",
         "lag_24h", "lag_48h", "lag_7d", "lag_14d",
         "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
         "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]
TIER2 = TIER1 + ["temp_target", "cdh_target", "temp_target_sq", "heat_extreme"]

train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]

# ============================================================================
# PRE-FLIGHT: CONFIRM THE PEAK IS PRESENT *BEFORE* ANY METRIC IS COMPUTED
# ============================================================================
say()
say("PRE-FLIGHT CHECKS (run before any model metric is calculated)")
say("-" * 92)
peak_present = KNOWN_PEAK_TS in test.index
peak_value = float(test.loc[KNOWN_PEAK_TS, "load_MW"]) if peak_present else float("nan")
say(f"  known peak {KNOWN_PEAK_TS} present in test set : {'YES' if peak_present else 'NO'}")
say(f"  its value in the test set                        : {peak_value:,.1f} MW "
    f"(expected {KNOWN_PEAK_MW}) -> {'MATCH' if abs(peak_value - KNOWN_PEAK_MW) < 1 else 'MISMATCH'}")
say(f"  test max demand                                  : {test['load_MW'].max():,.1f} MW")
if not peak_present or abs(peak_value - KNOWN_PEAK_MW) >= 1:
    raise SystemExit("ABORT: the approved peak is not intact in the test set. No metrics computed.")

say(f"  training rows                                    : {len(train):,}")
say(f"  testing rows                                     : {len(test):,}")
say(f"  test date range                                  : {test.index.min()} -> {test.index.max()}")
say(f"  train date range                                 : {train.index.min()} -> {train.index.max()}")
say(f"  train max demand                                 : {train['load_MW'].max():,.1f} MW "
    f"(test peak is {'INSIDE' if test['load_MW'].max() <= train['load_MW'].max() else 'OUTSIDE'} the trained range)")

# ============================================================================
# PEAK METHODOLOGY -- fixed before any model is fitted
# ============================================================================
peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()
say()
say("PEAK METHODOLOGY (fixed before fitting; identical for both models)")
say("-" * 92)
say(f"  peak period  = actual demand >= P{int(PEAK_QUANTILE*100)} of ACTUAL in the test window "
    f"= {peak_threshold:,.1f} MW  ({int(is_peak.sum()):,} of {len(test):,} blocks)")
say(f"  major peaks  = highest-actual block of each calendar day ({len(daily_peak_idx)} days)")
say(f"  underpredict = (actual - predicted)/actual, positive means forecast too LOW")

# ============================================================================
# MODELS
# ============================================================================
def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

def fit_xgb(tr, cols):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(tr[cols], tr["load_MW"])          # no early stopping -> test never seen
    return (lambda fr: m.predict(fr[cols])), m

y = test["load_MW"].to_numpy(float)
primary, supp = {}, {}

primary["Naive: same time yesterday"] = test["lag_24h"].to_numpy(float)
primary["Naive: same time last week"] = test["lag_7d"].to_numpy(float)
primary["A. OLS (Tier 1)"] = fit_ols(train, TIER1)(test)
xgb_pred_fn, xgb_model = fit_xgb(train, TIER1)
primary["B. XGBoost (Tier 1)"] = xgb_pred_fn(test)

supp["A. OLS (Tier 2, assumed perfect temp fcst)"] = fit_ols(train, TIER2)(test)
supp["B. XGBoost (Tier 2, assumed perfect temp fcst)"] = fit_xgb(train, TIER2)[0](test)

# ============================================================================
# METRICS
# ============================================================================
def metrics(p):
    p = np.asarray(p, float)
    err = y - p                                     # positive = underprediction
    ser = pd.Series(p, index=test.index)
    dp_a = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
    dp_p = ser.loc[daily_peak_idx].to_numpy(float)
    dp_u = (dp_a - dp_p) / dp_a * 100
    pk_pred = float(ser.loc[KNOWN_PEAK_TS])
    return {
        "MAPE": float(np.mean(np.abs(err / y)) * 100),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "PeakMAPE": float(np.mean(np.abs(err[is_peak] / y[is_peak])) * 100),
        "PeakMAE": float(np.mean(np.abs(err[is_peak]))),
        "OffPeakMAPE": float(np.mean(np.abs(err[~is_peak] / y[~is_peak])) * 100),
        "MaxUnder_MW": float(err.max()),
        "KnownPeak_pred": pk_pred,
        "KnownPeak_err_MW": pk_pred - KNOWN_PEAK_MW,
        "KnownPeak_err_pct": (pk_pred - KNOWN_PEAK_MW) / KNOWN_PEAK_MW * 100,
        "DailyPeakMAPE": float(np.mean(np.abs(dp_u))),
        "n_under_5": int((dp_u > 5).sum()),
        "n_under_10": int((dp_u > 10).sum()),
        "n_days": int(len(dp_u)),
    }

res = {k: metrics(v) for k, v in primary.items()}
res_supp = {k: metrics(v) for k, v in supp.items()}

say()
say("PRIMARY RESULTS -- genuine 24h-ahead forecast, NO future weather")
say("-" * 92)
say(f"{'Model':<32}{'MAPE':>7}{'MAE':>8}{'RMSE':>8}{'PeakMAPE':>10}{'Err@8392.6':>12}{'MaxUnder':>10}")
say("-" * 92)
for k, m in res.items():
    say(f"{k:<32}{m['MAPE']:>6.2f}%{m['MAE']:>8.1f}{m['RMSE']:>8.1f}{m['PeakMAPE']:>9.2f}%"
        f"{m['KnownPeak_err_MW']:>+11.1f}{m['MaxUnder_MW']:>10.1f}")
say("-" * 92)

say()
say("EXTREME-PEAK DETAIL (primary)")
say("-" * 92)
say(f"{'Model':<32}{'PeakMAE':>10}{'DailyPkMAPE':>13}{'>5% under':>12}{'>10% under':>12}{'Err@peak %':>12}")
say("-" * 92)
for k, m in res.items():
    say(f"{k:<32}{m['PeakMAE']:>10.1f}{m['DailyPeakMAPE']:>12.2f}%"
        f"{m['n_under_5']:>8}/{m['n_days']:<3}{m['n_under_10']:>8}/{m['n_days']:<3}{m['KnownPeak_err_pct']:>+11.2f}%")
say("-" * 92)

say()
say("GENERALIZATION CHECK -- ordinary days vs peak days (primary)")
say("-" * 92)
for k in ["A. OLS (Tier 1)", "B. XGBoost (Tier 1)"]:
    m = res[k]
    say(f"  {k:<24} off-peak MAPE {m['OffPeakMAPE']:.2f}%   peak MAPE {m['PeakMAPE']:.2f}%   "
        f"overall {m['MAPE']:.2f}%")

say()
say("SUPPLEMENTARY ONLY -- adds target-time OBSERVED temperature.")
say("This is an ASSUMPTION (a perfect 24h weather forecast we do not have), NOT the genuine")
say("forecast result, and must not be quoted as project accuracy.")
say("-" * 92)
for k, m in res_supp.items():
    say(f"  {k:<46} MAPE {m['MAPE']:.2f}%   PeakMAPE {m['PeakMAPE']:.2f}%   "
        f"Err@peak {m['KnownPeak_err_pct']:+.2f}%")

# ============================================================================
# FIGURES
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

pa, pb = primary["A. OLS (Tier 1)"], primary["B. XGBoost (Tier 1)"]

fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.plot(test.index, y, color=INK, linewidth=.9, label="Actual demand")
ax.plot(test.index, pa, color=COL_A, linewidth=.9, alpha=.9, label="Model A - OLS")
ax.plot(test.index, pb, color=COL_B, linewidth=.9, alpha=.9, label="Model B - XGBoost")
ax.axhline(peak_threshold, color="#8a8f88", linewidth=1, linestyle=(0, (4, 4)))
ax.text(test.index[10], peak_threshold + 70, f"peak threshold P90 = {peak_threshold:,.0f} MW",
        color="#5c655e", fontsize=8.5)
style(ax)
ax.set_title("Genuine 24h-ahead forecast vs actual, 1 May - 30 Jun 2025 (held out, no future weather)",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase2_final_fig1_full_test.png", dpi=150, facecolor=SURFACE); plt.close(fig)

lo, hi = KNOWN_PEAK_TS - pd.Timedelta(days=3), KNOWN_PEAK_TS + pd.Timedelta(days=3)
z = test.loc[lo:hi]; zi = (test.index >= lo) & (test.index <= hi)
fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.plot(z.index, z["load_MW"], color=INK, linewidth=1.6, label="Actual demand")
ax.plot(z.index, pa[zi], color=COL_A, linewidth=1.6, label="Model A - OLS")
ax.plot(z.index, pb[zi], color=COL_B, linewidth=1.6, label="Model B - XGBoost")
ax.scatter([KNOWN_PEAK_TS], [KNOWN_PEAK_MW], s=70, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
ax.annotate(f"{KNOWN_PEAK_MW:,.0f} MW", xy=(KNOWN_PEAK_TS, KNOWN_PEAK_MW),
            xytext=(-10, -22), textcoords="offset points", fontsize=9.5, color=INK, ha="right")
style(ax)
ax.set_title(f"Zoom: six days around the season peak ({KNOWN_PEAK_TS:%d %b %Y})",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase2_final_fig2_peak_zoom.png", dpi=150, facecolor=SURFACE); plt.close(fig)

dp_a = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
ua = (dp_a - pd.Series(pa, index=test.index).loc[daily_peak_idx].to_numpy(float)) / dp_a * 100
ub = (dp_a - pd.Series(pb, index=test.index).loc[daily_peak_idx].to_numpy(float)) / dp_a * 100
days = pd.DatetimeIndex(daily_peak_idx.values)
fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
ax.axhline(0, color="#8a8f88", linewidth=1)
ax.axhline(5, color="#c04530", linewidth=1, linestyle=(0, (4, 4)), alpha=.35)
ax.plot(days, ua, color=COL_A, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Model A - OLS")
ax.plot(days, ub, color=COL_B, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Model B - XGBoost")
style(ax)
ax.set_title("Daily peak error -- positive means the forecast was too LOW",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Underprediction of daily peak (%)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase2_final_fig3_peak_error.png", dpi=150, facecolor=SURFACE); plt.close(fig)

# ============================================================================
# SAVE
# ============================================================================
out = test[["load_MW", "temp_C"]].copy()
for k, v in primary.items():
    out[k] = v
out.to_csv(OUT_DIR / "phase2_final_test_predictions.csv")
(OUT_DIR / "phase2_final_results.json").write_text(json.dumps(
    {"primary": res, "supplementary_assumption_only": res_supp,
     "train_rows": len(train), "test_rows": len(test),
     "test_range": [str(test.index.min()), str(test.index.max())],
     "test_max_MW": float(test["load_MW"].max()),
     "peak_threshold_MW": peak_threshold}, indent=2))
(OUT_DIR / "phase2_final_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say(f"[SAVED] phase2_final_report.txt, phase2_final_results.json, "
    f"phase2_final_test_predictions.csv, 3 figures")
