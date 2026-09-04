"""
PHASE 5 -- WEATHER FORECAST BIAS CORRECTION
============================================
Question: does correcting bias in the 24h temperature forecast improve the
DEMAND forecast, especially under temperature-driven grid stress?

Demand model, pipeline, target, resolution, split and leakage rules are the
LOCKED ones. The only thing that varies is the temperature input:

  A. no forward-looking weather              (locked baseline)
  B. raw 24h forecast temperature            (Phase 4 result)
  C. bias-corrected 24h forecast temperature (this phase)
  D. observed future temperature             (REFERENCE ONLY, not operational)

LEAKAGE RULE FOR THE CORRECTION ITSELF
--------------------------------------
For a target block T, the forecast is issued at T-24h. At that moment the
operator knows forecast/observation pairs for every time <= T-24h, and
nothing after. Every rolling correction below is therefore computed from
`error.shift(96)` -- the identical shift-by-one-day construction already
leakage-verified in Phase 2 -- so no test-period observation is ever used to
correct a test-period forecast.

The static variants are fitted on TRAINING DATA ONLY (< 2025-05-01).

METHOD SELECTION is made on TRAINING-PERIOD weather evidence, never by
looking at test-period demand metrics.
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

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACKU_DATA", BASE / "data"))
OUT_DIR = Path(os.environ.get("HACKU_OUT", BASE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 96
TRAIN_END, TEST_START, TEST_END = "2025-05-01", "2025-05-01", "2025-07-01"
PEAK_QUANTILE = 0.90
KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
KNOWN_PEAK_MW = 8392.6
HEATWAVE_TMAX = 40.0
ROLL_DAYS = 30

INK, COL_A, COL_B, COL_C = "#2f3437", "#0d76b8", "#c9701a", "#2f7d4f"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 5 -- WEATHER FORECAST BIAS CORRECTION")
say("=" * 96)

# ---------------- locked pipeline ----------------
load = (pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp"))
s15 = load["load_MW"].resample("15min").mean().interpolate(limit=4)
s15_feat = s15.copy()
for _ in range(6):
    s15_feat = s15_feat.fillna(s15_feat.shift(H * 7))
s15_feat = s15_feat.interpolate(limit_direction="both")

wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")
fc = (pd.read_csv(DATA_DIR / "delhi_weather_forecast_day1.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
fc15 = fc["temp_fcst_C"].resample("15min").interpolate(method="time", limit=4)

g = pd.DataFrame({"load_MW": s15, "load_feat": s15_feat})
g["temp_C"] = wx15.reindex(g.index)
g["temp_fcst_C"] = fc15.reindex(g.index)
g = g.dropna(subset=["temp_C", "load_feat"])

# ================= BIAS-CORRECTION CANDIDATES =================
g["block_of_day"] = (g.index.hour * 4 + g.index.minute // 15).astype(int)
err = g["temp_C"] - g["temp_fcst_C"]                 # + means forecast was too COLD
err_avail = err.shift(H)                              # only known at/before issue time

train_mask = g.index < TRAIN_END

# 1. constant bias, fitted on TRAINING ONLY (static)
const_bias = float(err[train_mask].mean())

# 2. month-specific bias, TRAINING ONLY (static)
month_bias = err[train_mask].groupby(g.index[train_mask].month).mean()

# 3. rolling global bias -- trailing 30 days ending at issue time (adaptive, leakage-safe)
roll_global = err_avail.rolling(H * ROLL_DAYS, min_periods=H * 7).mean()

# 4. rolling hour-of-day bias -- trailing 30 days at the SAME time of day (adaptive)
roll_hod = err_avail.groupby(g["block_of_day"]).transform(
    lambda s: s.rolling(ROLL_DAYS, min_periods=7).mean())

# 5. static linear regression obs ~ fcst + diurnal terms, TRAINING ONLY
sin_h = np.sin(2 * np.pi * (g.index.hour + g.index.minute / 60) / 24)
cos_h = np.cos(2 * np.pi * (g.index.hour + g.index.minute / 60) / 24)
Xlr = np.column_stack([np.ones(len(g)), g["temp_fcst_C"].to_numpy(float), sin_h, cos_h])
tm = np.asarray(train_mask) & g["temp_fcst_C"].notna().to_numpy() & g["temp_C"].notna().to_numpy()
beta, *_ = np.linalg.lstsq(Xlr[tm], g.loc[tm, "temp_C"].to_numpy(float), rcond=None)
lin_pred = Xlr @ beta

cands = {
    "raw forecast": g["temp_fcst_C"],
    "C1 constant bias (static, train-fitted)": g["temp_fcst_C"] + const_bias,
    "C2 month bias (static, train-fitted)": g["temp_fcst_C"] + g.index.month.map(month_bias).to_numpy(),
    "C3 rolling 30d bias (adaptive)": g["temp_fcst_C"] + roll_global,
    "C4 rolling 30d bias by hour (adaptive)": g["temp_fcst_C"] + roll_hod,
    "C5 linear regression (static, train-fitted)": pd.Series(lin_pred, index=g.index),
}
for k, v in cands.items():
    g[k] = v

# ---------------- assemble locked feature frame ----------------
lf = g["load_feat"]
g["hour"] = g.index.hour + g.index.minute / 60
g["sin_hod"] = sin_h; g["cos_hod"] = cos_h
g["dow"] = g.index.dayofweek
g["is_weekend"] = (g["dow"] >= 5).astype(int)
g["month"] = g.index.month
g["doy"] = g.index.dayofyear
g["sin_doy"] = np.sin(2 * np.pi * g["doy"] / 365.25)
g["cos_doy"] = np.cos(2 * np.pi * g["doy"] / 365.25)
g["lag_24h"] = lf.shift(H); g["lag_48h"] = lf.shift(H * 2)
g["lag_7d"] = lf.shift(H * 7); g["lag_14d"] = lf.shift(H * 14)
past = lf.shift(H)
g["roll24_mean"] = past.rolling(H, min_periods=int(H * .75)).mean()
g["roll24_max"] = past.rolling(H, min_periods=int(H * .75)).max()
g["roll24_min"] = past.rolling(H, min_periods=int(H * .75)).min()
g["roll7d_mean"] = past.rolling(H * 7, min_periods=int(H * 7 * .75)).mean()
tpast = g["temp_C"].shift(H)
g["temp_at_issue"] = tpast
g["temp_prevday_max"] = tpast.rolling(H, min_periods=int(H * .75)).max()
g["temp_prevday_mean"] = tpast.rolling(H, min_periods=int(H * .75)).mean()

TIER1 = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
         "doy", "sin_doy", "cos_doy", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
         "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
         "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]

def add_temp_features(frame, src, tag):
    frame[f"{tag}_t"] = frame[src]
    frame[f"{tag}_cdh"] = np.clip(frame[src] - 24, 0, None)
    frame[f"{tag}_sq"] = frame[src] ** 2
    frame[f"{tag}_hx"] = np.clip(frame[src] - 38, 0, None) ** 2
    return [f"{tag}_t", f"{tag}_cdh", f"{tag}_sq", f"{tag}_hx"]

FB = add_temp_features(g, "temp_fcst_C", "fb")
FD = add_temp_features(g, "temp_C", "fd")

need = TIER1 + FB + FD + ["load_MW"] + list(cands.keys())
f_all = g.dropna(subset=list(dict.fromkeys(need)))

# ================= WEATHER-SIDE SELECTION (training evidence only) =================
def wx_metrics(frame, col):
    e = frame[col] - frame["temp_C"]
    hot_days = frame.groupby(frame.index.normalize())["temp_C"].max()
    hot = frame.index.normalize().isin(hot_days[hot_days >= HEATWAVE_TMAX].index)
    return {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "worst": float(e.abs().max()),
            "hot_MAE": float(e[hot].abs().mean()) if hot.any() else float("nan"),
            "hot_bias": float(e[hot].mean()) if hot.any() else float("nan")}

tr_all = f_all[f_all.index < TRAIN_END]
te_all = f_all[(f_all.index >= TEST_START) & (f_all.index < TEST_END)]

say()
say("WEATHER-SIDE: CANDIDATE SELECTION ON TRAINING-PERIOD EVIDENCE ONLY")
say("-" * 96)
say(f"{'candidate':<44}{'MAE':>7}{'RMSE':>8}{'bias':>8}{'worst':>8}{'hotMAE':>9}{'hotBias':>9}")
say("-" * 96)
tr_scores = {}
for k in cands:
    m = wx_metrics(tr_all, k)
    tr_scores[k] = m
    say(f"{k:<44}{m['MAE']:>7.2f}{m['RMSE']:>8.2f}{m['bias']:>+8.2f}{m['worst']:>8.2f}"
        f"{m['hot_MAE']:>9.2f}{m['hot_bias']:>+9.2f}")
say("-" * 96)
chosen = min([k for k in cands if k != "raw forecast"], key=lambda k: tr_scores[k]["MAE"])
say(f"SELECTED on training MAE (test never consulted): {chosen}")

say()
say("WEATHER-SIDE: HELD-OUT TEST PERIOD (1 May - 30 Jun 2025)")
say("-" * 96)
say(f"{'candidate':<44}{'MAE':>7}{'RMSE':>8}{'bias':>8}{'worst':>8}{'hotMAE':>9}{'hotBias':>9}")
say("-" * 96)
te_scores = {}
for k in cands:
    m = wx_metrics(te_all, k)
    te_scores[k] = m
    mark = "  <- selected" if k == chosen else ""
    say(f"{k:<44}{m['MAE']:>7.2f}{m['RMSE']:>8.2f}{m['bias']:>+8.2f}{m['worst']:>8.2f}"
        f"{m['hot_MAE']:>9.2f}{m['hot_bias']:>+9.2f}{mark}")
say("-" * 96)

# ================= DEMAND-SIDE EXPERIMENT =================
FC_COLS = add_temp_features(f_all, chosen, "fc")
f_all = f_all.dropna(subset=FC_COLS)
train = f_all[f_all.index < TRAIN_END]
test = f_all[(f_all.index >= TEST_START) & (f_all.index < TEST_END)]

say()
say("PRE-FLIGHT (before any demand metric)")
say("-" * 96)
assert KNOWN_PEAK_TS in test.index, "approved peak missing -- aborting"
say(f"  peak {KNOWN_PEAK_TS} present : YES, {test.loc[KNOWN_PEAK_TS,'load_MW']:,.1f} MW "
    f"({'MATCH' if abs(test.loc[KNOWN_PEAK_TS,'load_MW']-KNOWN_PEAK_MW)<1 else 'MISMATCH'})")
say(f"  test max demand              : {test['load_MW'].max():,.1f} MW")
say(f"  test range                   : {test.index.min()} -> {test.index.max()}")
say(f"  train rows / test rows       : {len(train):,} / {len(test):,}")

peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()
obs_tmax = test.groupby(test.index.normalize())["temp_C"].max()
heat_days = set(obs_tmax[obs_tmax >= HEATWAVE_TMAX].index)
is_heat = np.array([ts.normalize() in heat_days for ts in test.index])
say(f"  peak threshold P90           : {peak_threshold:,.1f} MW ({int(is_peak.sum()):,} blocks)")
say(f"  heatwave days (obs Tmax>=40C): {len(heat_days)} days, {int(is_heat.sum()):,} blocks (definition unchanged)")

def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

y = test["load_MW"].to_numpy(float)
configs = {
    "A. no forward weather": TIER1,
    "B. raw forecast temp": TIER1 + FB,
    "C. bias-corrected forecast temp": TIER1 + FC_COLS,
    "D. observed future temp (REFERENCE)": TIER1 + FD,
}
preds = {k: fit_ols(train, c)(test) for k, c in configs.items()}

def metrics(p):
    p = np.asarray(p, float); e = y - p
    ser = pd.Series(p, index=test.index)
    dpa = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
    dpp = ser.loc[daily_peak_idx].to_numpy(float)
    du = (dpa - dpp) / dpa * 100
    pk = float(ser.loc[KNOWN_PEAK_TS])
    return {"MAPE": float(np.mean(np.abs(e / y)) * 100), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "PeakMAPE": float(np.mean(np.abs(e[is_peak] / y[is_peak])) * 100),
            "HeatMAPE": float(np.mean(np.abs(e[is_heat] / y[is_heat])) * 100),
            "DailyPeakMAPE": float(np.mean(np.abs(du))),
            "KnownPeak_err_MW": pk - KNOWN_PEAK_MW,
            "KnownPeak_pred": pk,
            "n_under_5": int((du > 5).sum()), "n_under_10": int((du > 10).sum()),
            "n_days": int(len(du)), "MaxUnder_MW": float(e.max())}

res = {k: metrics(v) for k, v in preds.items()}

say()
say("DEMAND RESULTS -- identical rows, target, horizon and metric definitions")
say("-" * 96)
say(f"{'Model':<38}{'MAPE':>7}{'MAE':>8}{'RMSE':>8}{'PeakMAPE':>10}{'HeatMAPE':>10}{'DailyPk':>9}")
say("-" * 96)
for k, m in res.items():
    say(f"{k:<38}{m['MAPE']:>6.2f}%{m['MAE']:>8.1f}{m['RMSE']:>8.1f}"
        f"{m['PeakMAPE']:>9.2f}%{m['HeatMAPE']:>9.2f}%{m['DailyPeakMAPE']:>8.2f}%")
say("-" * 96)
say(f"{'Model':<38}{'Err@8392.6':>14}{'>5% under':>12}{'>10% under':>12}{'MaxUnder':>11}")
say("-" * 96)
for k, m in res.items():
    say(f"{k:<38}{m['KnownPeak_err_MW']:>+9.1f} MW{m['n_under_5']:>8}/{m['n_days']:<3}"
        f"{m['n_under_10']:>8}/{m['n_days']:<3}{m['MaxUnder_MW']:>11.1f}")
say("-" * 96)

A, B, C, D = (res["A. no forward weather"], res["B. raw forecast temp"],
              res["C. bias-corrected forecast temp"], res["D. observed future temp (REFERENCE)"])
say()
say("CHANGES vs the locked baseline A (positive = better)")
say("-" * 96)
for lbl, key in [("overall MAPE", "MAPE"), ("peak-period MAPE", "PeakMAPE"),
                 ("heatwave MAPE", "HeatMAPE"), ("daily-peak MAPE", "DailyPeakMAPE")]:
    say(f"  {lbl:<22} A {A[key]:>6.2f}%   B {B[key]:>6.2f}% ({A[key]-B[key]:+.2f})   "
        f"C {C[key]:>6.2f}% ({A[key]-C[key]:+.2f})   D(ref) {D[key]:>6.2f}% ({A[key]-D[key]:+.2f})")
say(f"  {'peaks missed >10%':<22} A {A['n_under_10']}/{A['n_days']}   B {B['n_under_10']}/{B['n_days']}   "
    f"C {C['n_under_10']}/{C['n_days']}   D(ref) {D['n_under_10']}/{D['n_days']}")
say(f"  {'error @ 8,392.6 MW':<22} A {A['KnownPeak_err_MW']:+,.1f}   B {B['KnownPeak_err_MW']:+,.1f}   "
    f"C {C['KnownPeak_err_MW']:+,.1f}   D(ref) {D['KnownPeak_err_MW']:+,.1f}")

# ================= HEATWAVE WINDOW DETAIL =================
say()
say("HEATWAVE DETAIL -- 11-13 June 2025, daily peak blocks")
say("-" * 96)
win = test.loc["2025-06-11":"2025-06-13"]
wpi = win.groupby(win.index.normalize())["load_MW"].idxmax()
say(f"{'timestamp':<21}{'obsT':>7}{'rawT':>7}{'corrT':>7}{'actual':>10}{'A':>9}{'B':>9}{'C':>9}")
say("-" * 96)
for ts in wpi:
    row = test.loc[ts]
    say(f"{str(ts):<21}{row['temp_C']:>7.1f}{row['temp_fcst_C']:>7.1f}{row[chosen]:>7.1f}"
        f"{row['load_MW']:>10,.0f}"
        f"{pd.Series(preds['A. no forward weather'], index=test.index).loc[ts]:>9,.0f}"
        f"{pd.Series(preds['B. raw forecast temp'], index=test.index).loc[ts]:>9,.0f}"
        f"{pd.Series(preds['C. bias-corrected forecast temp'], index=test.index).loc[ts]:>9,.0f}")
say("-" * 96)
say(f"the season peak {KNOWN_PEAK_TS}: observed {test.loc[KNOWN_PEAK_TS,'temp_C']:.1f}C, "
    f"raw forecast {test.loc[KNOWN_PEAK_TS,'temp_fcst_C']:.1f}C, corrected {test.loc[KNOWN_PEAK_TS,chosen]:.1f}C")

# ================= FIGURES =================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

lo_t, hi_t = KNOWN_PEAK_TS - pd.Timedelta(days=3), KNOWN_PEAK_TS + pd.Timedelta(days=3)
m = (test.index >= lo_t) & (test.index <= hi_t)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.2), facecolor=SURFACE, sharex=True,
                                gridspec_kw={"height_ratios": [1, 1.35]})
ax1.plot(test.index[m], test["temp_C"][m], color=INK, linewidth=1.6, label="Observed temp")
ax1.plot(test.index[m], test["temp_fcst_C"][m], color=COL_B, linewidth=1.6, label="Raw 24h forecast")
ax1.plot(test.index[m], test[chosen][m], color=COL_C, linewidth=1.6, label="Bias-corrected forecast")
style(ax1); ax1.set_ylabel("Temperature (C)", fontsize=9.5, color="#5c655e")
ax1.set_title("Heatwave window: temperature forecast quality, then its effect on demand",
              fontsize=12, color=INK, loc="left", pad=12)
ax1.legend(frameon=False, fontsize=9.5, ncols=3, loc="upper left")
ax2.plot(test.index[m], y[m], color=INK, linewidth=1.6, label="Actual demand")
ax2.plot(test.index[m], preds["A. no forward weather"][m], color=COL_A, linewidth=1.4, label="A - no weather")
ax2.plot(test.index[m], preds["B. raw forecast temp"][m], color=COL_B, linewidth=1.4, label="B - raw forecast")
ax2.plot(test.index[m], preds["C. bias-corrected forecast temp"][m], color=COL_C, linewidth=1.4, label="C - corrected")
ax2.scatter([KNOWN_PEAK_TS], [KNOWN_PEAK_MW], s=70, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
style(ax2); ax2.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax2.legend(frameon=False, fontsize=9.5, ncols=4, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase5_fig1_heatwave.png", dpi=150, facecolor=SURFACE); plt.close(fig)

dpa = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
days = pd.DatetimeIndex(daily_peak_idx.values)
fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
ax.axhline(0, color="#8a8f88", linewidth=1)
ax.axhline(5, color="#c04530", linewidth=1, linestyle=(0, (4, 4)), alpha=.35)
for k, col, lab in [("A. no forward weather", COL_A, "A - no weather"),
                    ("B. raw forecast temp", COL_B, "B - raw forecast"),
                    ("C. bias-corrected forecast temp", COL_C, "C - corrected")]:
    u = (dpa - pd.Series(preds[k], index=test.index).loc[daily_peak_idx].to_numpy(float)) / dpa * 100
    ax.plot(days, u, color=col, linewidth=1.5, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=.8, label=lab)
style(ax)
ax.set_title("Daily peak underprediction -- positive means forecast too LOW",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Underprediction (%)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase5_fig2_daily_peak.png", dpi=150, facecolor=SURFACE); plt.close(fig)

out = test[["load_MW", "temp_C", "temp_fcst_C", chosen]].copy()
for k, v in preds.items():
    out[k] = v
out.to_csv(OUT_DIR / "phase5_test_predictions.csv")
(OUT_DIR / "phase5_results.json").write_text(json.dumps(
    {"chosen_correction": chosen, "weather_train": tr_scores, "weather_test": te_scores,
     "demand": res, "train_rows": len(train), "test_rows": len(test)}, indent=2))
(OUT_DIR / "phase5_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say("[SAVED] phase5_report.txt, phase5_results.json, phase5_test_predictions.csv, 2 figures")
