"""
PHASE 2 -- ONE CONTROLLED MODEL IMPROVEMENT EXPERIMENT
=======================================================
Question: does a tree-based nonlinear model genuinely fix the extreme-peak
underprediction seen in the Phase 1d linear model?

MODEL A = current validated model (OLS linear regression)
MODEL B = LightGBM (tree-based, nonlinear)

Both models receive an IDENTICAL feature matrix, identical rows, identical
train/test split, identical target and identical 24h-ahead leakage rules.
No shuffling. No test-set-driven tuning. No early stopping on test data.

WEATHER METHODOLOGY (important):
  TIER 1 = only features unquestionably available at forecast-issue time
           (calendar + demand lags >=24h old + OBSERVED PAST weather).
           No future weather whatsoever.
  TIER 2 = TIER 1 + temperature AT TARGET TIME. This is an ASSUMPTION:
           we substitute observed temperature for a weather forecast we do
           not have. It measures how much weather COULD help, and is an
           upper bound, not operational performance.
  TIER 2N = sensitivity: same as TIER 2 but the target-time temperature is
           perturbed with Gaussian noise to imitate real 24h forecast error.
           Models are trained on true weather (as you would) and only the
           inference-time weather is degraded.

Run:  python phase2_model_experiment.py
Env:  HACKU_DATA / HACKU_OUT may override the data and output directories.
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

import lightgbm as lgb

# ----------------------------------------------------------------------------
# CONFIG -- every choice below is fixed BEFORE any result is seen
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACKU_DATA", BASE / "data"))
OUT_DIR = Path(os.environ.get("HACKU_OUT", BASE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2025-05-01"      # train strictly before this
TEST_START = "2025-05-01"     # real summer peak season, unchanged from Phase 1d
TEST_END = "2025-07-01"       # exclusive

HORIZON_BLOCKS = 96           # 24h at 15-min resolution
PEAK_QUANTILE = 0.90          # "peak period" = actual demand >= this quantile of ACTUAL in test
WEATHER_NOISE_SIGMA = 1.5     # degC, illustrative 24h forecast error
WEATHER_NOISE_SEEDS = 20
SEED = 42

LGB_PARAMS = dict(            # ONE fixed configuration. No search, no tuning loop.
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

INK = "#2f3437"       # actual demand (reference truth, neutral -- not a category)
COL_A = "#0d76b8"     # Model A (OLS)      -- CVD-validated pair
COL_B = "#c9701a"     # Model B (LightGBM) -- CVD-validated pair
GRID = "#dfe2df"
SURFACE = "#fcfcfb"

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say("PHASE 2 -- CONTROLLED MODEL IMPROVEMENT EXPERIMENT")
say("=" * 78)

# ----------------------------------------------------------------------------
# 1. LOAD + PREPROCESS (identical logic for every model)
# ----------------------------------------------------------------------------
load = (pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates(subset="timestamp")
          .set_index("timestamp"))
s15_raw = load["load_MW"].resample("15min").mean()
s15 = s15_raw.interpolate(limit=4)          # TARGET: real values + gaps <=1h only
n_target_missing = int(s15.isna().sum())

# FEATURE-INPUT series only. 1,027 blocks remain missing after the 1h fill, spread
# over 161 gaps (longest 48h). A 7-day rolling window needs 672 consecutive values,
# so leaving them NaN silently deletes tens of thousands of rows -- including the
# season peak. We therefore impute the FEATURE-INPUT copy (seasonal-naive from the
# same block a week earlier, then linear), and NEVER impute the target, and NEVER
# score on an imputed target.
s15_feat = s15.copy()
for _ in range(6):
    s15_feat = s15_feat.fillna(s15_feat.shift(96 * 7))
s15_feat = s15_feat.interpolate(limit_direction="both")

wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")

df = pd.DataFrame({"load_MW": s15, "load_feat": s15_feat})
df["temp_C"] = wx15.reindex(df.index)
df = df.dropna(subset=["temp_C", "load_feat"])

say(f"[CONFIRMED] Demand series: {len(s15):,} 15-min blocks, {s15.index.min()} -> {s15.index.max()}")
say(f"[CONFIRMED] Weather series: {len(wx):,} hourly readings (Open-Meteo, observed/reanalysis)")
say(f"[CONFIRMED] Blocks with no real observed demand after <=1h interpolation: {n_target_missing:,} "
    f"({n_target_missing/len(s15)*100:.2f}%) -- these are EXCLUDED from training targets and from all scoring")
say(f"[INFERENCE]  Those same blocks are imputed (seasonal-naive, same block previous week) ONLY where they "
    f"appear inside lag/rolling FEATURE INPUTS, so that isolated gaps do not delete whole windows")

# ----------------------------------------------------------------------------
# 2. FEATURES -- everything strictly computable at issue time = target - 24h
# ----------------------------------------------------------------------------
H = HORIZON_BLOCKS
f = pd.DataFrame(index=df.index)
f["load_MW"] = df["load_MW"]      # real target only (NaN where unobserved)
lf = df["load_feat"]              # gap-filled copy, used ONLY to build features

# -- calendar: deterministic, known arbitrarily far in advance
f["block_of_day"] = (df.index.hour * 4 + df.index.minute // 15).astype(int)
f["hour"] = df.index.hour + df.index.minute / 60
f["sin_hod"] = np.sin(2 * np.pi * f["hour"] / 24)
f["cos_hod"] = np.cos(2 * np.pi * f["hour"] / 24)
f["dow"] = df.index.dayofweek
f["is_weekend"] = (f["dow"] >= 5).astype(int)
f["month"] = df.index.month
f["doy"] = df.index.dayofyear
f["sin_doy"] = np.sin(2 * np.pi * f["doy"] / 365.25)
f["cos_doy"] = np.cos(2 * np.pi * f["doy"] / 365.25)

# -- demand history: every term shifted by >= H blocks, so nothing after issue time
f["lag_24h"] = lf.shift(H)                      # value AT issue time
f["lag_48h"] = lf.shift(H * 2)
f["lag_7d"] = lf.shift(H * 7)
f["lag_14d"] = lf.shift(H * 14)
past = lf.shift(H)                              # everything below is built only from <= issue time
f["roll24_mean"] = past.rolling(H, min_periods=int(H * 0.75)).mean()
f["roll24_max"] = past.rolling(H, min_periods=int(H * 0.75)).max()
f["roll24_min"] = past.rolling(H, min_periods=int(H * 0.75)).min()
f["roll7d_mean"] = past.rolling(H * 7, min_periods=int(H * 7 * 0.75)).mean()

# -- OBSERVED PAST weather: real, available at issue time (not a forecast)
tpast = df["temp_C"].shift(H)
f["temp_at_issue"] = tpast
f["temp_prevday_max"] = tpast.rolling(H, min_periods=int(H * 0.75)).max()
f["temp_prevday_mean"] = tpast.rolling(H, min_periods=int(H * 0.75)).mean()

# -- FUTURE weather (TIER 2 ONLY -- assumption, see header)
f["temp_target"] = df["temp_C"]
f["cdh_target"] = np.clip(df["temp_C"] - 24, 0, None)
f["temp_target_sq"] = df["temp_C"] ** 2
f["heat_extreme"] = np.clip(df["temp_C"] - 38, 0, None) ** 2

# Single dropna on the union of ALL columns so every model sees identical rows
f = f.dropna()

TIER1 = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend",
         "month", "doy", "sin_doy", "cos_doy",
         "lag_24h", "lag_48h", "lag_7d", "lag_14d",
         "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
         "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]
TIER2 = TIER1 + ["temp_target", "cdh_target", "temp_target_sq", "heat_extreme"]

train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]

say(f"[CONFIRMED] Train: {train.index.min()} -> {train.index.max()}  ({len(train):,} blocks)")
say(f"[CONFIRMED] Test:  {test.index.min()} -> {test.index.max()}  ({len(test):,} blocks)")
say(f"[CONFIRMED] Train max demand {train['load_MW'].max():,.1f} MW | Test max demand {test['load_MW'].max():,.1f} MW")
say(f"[CONFIRMED] Test peak is {'INSIDE' if test['load_MW'].max() <= train['load_MW'].max() else 'OUTSIDE'} "
    f"the training range -- trees can only interpolate, so this matters.")

# ----------------------------------------------------------------------------
# 3. PEAK DEFINITIONS -- fixed here, before any model is fitted
# ----------------------------------------------------------------------------
peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = test["load_MW"] >= peak_threshold
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()

say()
say("PEAK METHODOLOGY (defined before computing any result):")
say(f"  peak period    = blocks where ACTUAL demand >= P{int(PEAK_QUANTILE*100)} of actual in the test window "
    f"= {peak_threshold:,.1f} MW ({int(is_peak.sum()):,} of {len(test):,} blocks)")
say(f"  major peaks    = the single highest-actual block of each calendar day ({len(daily_peak_idx)} days)")
say(f"  underpredict %  = (actual - predicted) / actual * 100, positive = model too low")

# ----------------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------------
def fit_ols(train_df, cols):
    X = train_df[cols].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, train_df["load_MW"].to_numpy(dtype=float), rcond=None)
    def predict(frame):
        Xt = frame[cols].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(Xt)), (Xt - mu) / sd]) @ coef
    return predict

def fit_lgbm(train_df, cols):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(train_df[cols], train_df["load_MW"])   # no early stopping -> no test peeking
    return lambda frame: m.predict(frame[cols]), m

y_test = test["load_MW"].to_numpy(dtype=float)
preds = {}

preds["Baseline: same time yesterday"] = test["lag_24h"].to_numpy(dtype=float)
preds["Baseline: same time last week"] = test["lag_7d"].to_numpy(dtype=float)

ols_t1 = fit_ols(train, TIER1)
preds["A. OLS (T1, no future weather)"] = ols_t1(test)
lgbm_t1, model_t1 = fit_lgbm(train, TIER1)
preds["B. LightGBM (T1, no future weather)"] = lgbm_t1(test)

ols_t2 = fit_ols(train, TIER2)
preds["A. OLS (T2, +perfect temp fcst)"] = ols_t2(test)
lgbm_t2, model_t2 = fit_lgbm(train, TIER2)
preds["B. LightGBM (T2, +perfect temp fcst)"] = lgbm_t2(test)

# ----------------------------------------------------------------------------
# 5. METRICS
# ----------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    err = y - p                                   # positive = underprediction
    out = {
        "MAPE": float(np.mean(np.abs(err / y)) * 100),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
    }
    pk = is_peak.to_numpy()
    out["PeakMAPE"] = float(np.mean(np.abs(err[pk] / y[pk])) * 100)
    out["PeakMAE"] = float(np.mean(np.abs(err[pk])))
    out["OffPeakMAPE"] = float(np.mean(np.abs(err[~pk] / y[~pk])) * 100)
    out["MaxUnder_MW"] = float(err.max())
    out["MaxUnder_pct"] = float((err / y).max() * 100)

    ser = pd.Series(p, index=test.index)
    dp_actual = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
    dp_pred = ser.loc[daily_peak_idx].to_numpy(float)
    dp_under = (dp_actual - dp_pred) / dp_actual * 100
    out["DailyPeakMAPE"] = float(np.mean(np.abs(dp_under)))
    out["n_peaks_under_5"] = int((dp_under > 5).sum())
    out["n_peaks_under_10"] = int((dp_under > 10).sum())
    out["n_daily_peaks"] = int(len(dp_under))

    top_idx = test["load_MW"].idxmax()
    out["TopBlock_actual"] = float(test.loc[top_idx, "load_MW"])
    out["TopBlock_pred"] = float(ser.loc[top_idx])
    out["TopBlock_err_MW"] = out["TopBlock_pred"] - out["TopBlock_actual"]
    out["TopBlock_err_pct"] = out["TopBlock_err_MW"] / out["TopBlock_actual"] * 100
    return out

results = {name: metrics(y_test, p) for name, p in preds.items()}

# ----------------------------------------------------------------------------
# 6. WEATHER-FORECAST-ERROR SENSITIVITY (TIER 2N)
# ----------------------------------------------------------------------------
def noisy_weather_scores(predict_fn, sigma, seeds):
    ms, pms = [], []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        t = test.copy()
        noisy = t["temp_target"] + rng.normal(0, sigma, len(t))
        t["temp_target"] = noisy
        t["cdh_target"] = np.clip(noisy - 24, 0, None)
        t["temp_target_sq"] = noisy ** 2
        t["heat_extreme"] = np.clip(noisy - 38, 0, None) ** 2
        p = predict_fn(t)
        m = metrics(y_test, p)
        ms.append(m["MAPE"]); pms.append(m["PeakMAPE"])
    return float(np.mean(ms)), float(np.mean(pms))

ols_t2n = noisy_weather_scores(ols_t2, WEATHER_NOISE_SIGMA, WEATHER_NOISE_SEEDS)
lgb_t2n = noisy_weather_scores(lgbm_t2, WEATHER_NOISE_SIGMA, WEATHER_NOISE_SEEDS)

# ----------------------------------------------------------------------------
# 7. TABLES
# ----------------------------------------------------------------------------
say()
say("RESULTS -- identical test window, identical rows, identical target")
say("-" * 78)
hdr = f"{'Model':<38}{'MAPE':>7}{'MAE':>8}{'RMSE':>8}{'PeakMAPE':>10}{'MaxUnder':>10}"
say(hdr)
say("-" * 78)
for name, m in results.items():
    say(f"{name:<38}{m['MAPE']:>6.2f}%{m['MAE']:>7.1f}{m['RMSE']:>8.1f}"
        f"{m['PeakMAPE']:>9.2f}%{m['MaxUnder_MW']:>9.1f}")
say("-" * 78)

say()
say("EXTREME-PEAK ANALYSIS")
say("-" * 78)
say(f"{'Model':<38}{'DailyPkMAPE':>12}{'>5% under':>11}{'>10% under':>12}{'TopBlockErr':>13}")
say("-" * 78)
for name, m in results.items():
    say(f"{name:<38}{m['DailyPeakMAPE']:>11.2f}%{m['n_peaks_under_5']:>8}/{m['n_daily_peaks']:<2}"
        f"{m['n_peaks_under_10']:>9}/{m['n_daily_peaks']:<2}{m['TopBlock_err_pct']:>+12.2f}%")
say("-" * 78)

say()
say("GENERALIZATION CHECK -- does the tree model damage ordinary days?")
for name in ["A. OLS (T1, no future weather)", "B. LightGBM (T1, no future weather)",
             "A. OLS (T2, +perfect temp fcst)", "B. LightGBM (T2, +perfect temp fcst)"]:
    m = results[name]
    say(f"  {name:<38} off-peak MAPE {m['OffPeakMAPE']:.2f}%   peak MAPE {m['PeakMAPE']:.2f}%")

say()
say(f"WEATHER-FORECAST-ERROR SENSITIVITY (sigma={WEATHER_NOISE_SIGMA} degC, {WEATHER_NOISE_SEEDS} seeds,")
say("trained on true weather, degraded only at inference -- ILLUSTRATIVE, not a measured forecast distribution)")
say(f"  A. OLS      T2 perfect {results['A. OLS (T2, +perfect temp fcst)']['MAPE']:.2f}% -> noisy {ols_t2n[0]:.2f}%"
    f"   (peak {results['A. OLS (T2, +perfect temp fcst)']['PeakMAPE']:.2f}% -> {ols_t2n[1]:.2f}%)")
say(f"  B. LightGBM T2 perfect {results['B. LightGBM (T2, +perfect temp fcst)']['MAPE']:.2f}% -> noisy {lgb_t2n[0]:.2f}%"
    f"   (peak {results['B. LightGBM (T2, +perfect temp fcst)']['PeakMAPE']:.2f}% -> {lgb_t2n[1]:.2f}%)")

# ----------------------------------------------------------------------------
# 8. PLOTS
# ----------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

pa = preds["A. OLS (T2, +perfect temp fcst)"]
pb = preds["B. LightGBM (T2, +perfect temp fcst)"]

# Figure 1 -- full test period
fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.plot(test.index, y_test, color=INK, linewidth=0.9, label="Actual demand")
ax.plot(test.index, pa, color=COL_A, linewidth=0.9, alpha=0.9, label="Model A - OLS")
ax.plot(test.index, pb, color=COL_B, linewidth=0.9, alpha=0.9, label="Model B - LightGBM")
ax.axhline(peak_threshold, color="#8a8f88", linewidth=1, linestyle=(0, (4, 4)))
ax.text(test.index[10], peak_threshold + 60, f"peak threshold P90 = {peak_threshold:,.0f} MW",
        color="#5c655e", fontsize=8.5)
style(ax)
ax.set_title("24h-ahead forecast vs actual demand, May-Jun 2025 (held-out summer peak season)",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "phase2_fig1_full_test_period.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# Figure 2 -- zoom on the largest peak
top_idx = test["load_MW"].idxmax()
lo, hi = top_idx - pd.Timedelta(days=3), top_idx + pd.Timedelta(days=3)
z = test.loc[lo:hi]
zi = (test.index >= lo) & (test.index <= hi)
fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.plot(z.index, z["load_MW"], color=INK, linewidth=1.6, label="Actual demand")
ax.plot(z.index, pa[zi], color=COL_A, linewidth=1.6, label="Model A - OLS")
ax.plot(z.index, pb[zi], color=COL_B, linewidth=1.6, label="Model B - LightGBM")
ax.scatter([top_idx], [test.loc[top_idx, "load_MW"]], s=70, color=INK, zorder=5,
           edgecolor=SURFACE, linewidth=2)
ax.annotate(f"season peak {test.loc[top_idx,'load_MW']:,.0f} MW",
            xy=(top_idx, test.loc[top_idx, "load_MW"]),
            xytext=(-12, 16), textcoords="offset points",
            fontsize=9.5, color=INK, ha="right")
style(ax)
ax.set_title(f"Zoom: six days around the season peak ({top_idx:%d %b %Y})",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_DIR / "phase2_fig2_peak_zoom.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# Figure 3 -- daily-peak underprediction by day
dp_actual = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
sa = pd.Series(pa, index=test.index).loc[daily_peak_idx].to_numpy(float)
sb = pd.Series(pb, index=test.index).loc[daily_peak_idx].to_numpy(float)
ua = (dp_actual - sa) / dp_actual * 100
ub = (dp_actual - sb) / dp_actual * 100
days = pd.DatetimeIndex(daily_peak_idx.values)
fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
ax.axhline(0, color="#8a8f88", linewidth=1)
ax.axhline(5, color="#c0453040", linewidth=1, linestyle=(0, (4, 4)))
ax.plot(days, ua, color=COL_A, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Model A - OLS")
ax.plot(days, ub, color=COL_B, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Model B - LightGBM")
style(ax)
ax.set_title("Daily peak error -- positive means the model predicted too LOW",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Underprediction of daily peak (%)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "phase2_fig3_daily_peak_error.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ----------------------------------------------------------------------------
# 9. SAVE
# ----------------------------------------------------------------------------
pred_out = test[["load_MW", "temp_C"]].copy() if "temp_C" in test.columns else test[["load_MW"]].copy()
for name, p in preds.items():
    pred_out[name] = p
pred_out.to_csv(OUT_DIR / "phase2_test_predictions.csv")

payload = {"results": results,
           "sensitivity_noisy_weather": {"sigma_degC": WEATHER_NOISE_SIGMA,
                                         "seeds": WEATHER_NOISE_SEEDS,
                                         "ols_t2_mape_peak": ols_t2n,
                                         "lgbm_t2_mape_peak": lgb_t2n},
           "peak_threshold_MW": peak_threshold,
           "train_max_MW": float(train["load_MW"].max()),
           "test_max_MW": float(test["load_MW"].max())}
(OUT_DIR / "phase2_results.json").write_text(json.dumps(payload, indent=2))
(OUT_DIR / "phase2_report.txt").write_text("\n".join(lines), encoding="utf-8")

say()
say(f"[SAVED] {OUT_DIR}/phase2_report.txt, phase2_results.json, phase2_test_predictions.csv, 3 figures")
