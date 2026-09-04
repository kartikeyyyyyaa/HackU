"""
PHASE 4 -- FORWARD-LOOKING WEATHER: CONTROLLED A vs B EXPERIMENT
=================================================================
MODEL A = locked Tier-1 OLS, no forward-looking weather      (the approved baseline)
MODEL B = same OLS + GENUINE 24h-lead FORECAST temperature

The forecast temperature is NOT observed future weather. It comes from
Open-Meteo's Previous Runs API variable `temperature_2m_previous_day1`,
defined as "the prediction made 24 hours before the target time" -- i.e. the
forecast that would actually have been on an operator's desk at issue time.

REFERENCE ROW ONLY (not a legitimate operational model): Tier-2 "perfect
weather", which uses observed future temperature. It is included solely as
an upper bound to show how much of the achievable gain a real forecast
captures. It must never be quoted as operational accuracy.

Everything else is identical between A and B: same pipeline, same rows,
same target, same train/test split, same horizon, same metrics.
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
HEATWAVE_TMAX = 40.0          # objective definition, fixed in advance

INK, COL_A, COL_B = "#2f3437", "#0d76b8", "#c9701a"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 4 -- FORWARD-LOOKING WEATHER: A vs B")
say("=" * 94)

# ---------------- approved pipeline, verbatim ----------------
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
g["temp_C"] = wx15.reindex(g.index)            # observed (past use + reference only)
g["temp_fcst_C"] = fc15.reindex(g.index)       # 24h-lead FORECAST vintage
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

# forward-looking FORECAST features (Model B only)
g["fcst_temp"] = g["temp_fcst_C"]
g["fcst_cdh"] = np.clip(g["temp_fcst_C"] - 24, 0, None)
g["fcst_temp_sq"] = g["temp_fcst_C"] ** 2
g["fcst_heat_extreme"] = np.clip(g["temp_fcst_C"] - 38, 0, None) ** 2
# observed-future reference features (REFERENCE ROW ONLY -- not operational)
g["obs_temp"] = g["temp_C"]
g["obs_cdh"] = np.clip(g["temp_C"] - 24, 0, None)
g["obs_temp_sq"] = g["temp_C"] ** 2
g["obs_heat_extreme"] = np.clip(g["temp_C"] - 38, 0, None) ** 2

TIER1 = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
         "doy", "sin_doy", "cos_doy", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
         "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
         "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]
FCST = TIER1 + ["fcst_temp", "fcst_cdh", "fcst_temp_sq", "fcst_heat_extreme"]
OBSREF = TIER1 + ["obs_temp", "obs_cdh", "obs_temp_sq", "obs_heat_extreme"]

# Single dropna over the union so A, B and the reference share IDENTICAL rows
f = g.dropna(subset=list(dict.fromkeys(TIER1 + FCST + OBSREF + ["load_MW"])))
train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]

say()
say("PRE-FLIGHT")
say("-" * 94)
assert KNOWN_PEAK_TS in test.index, "approved peak missing from test set -- aborting"
say(f"  peak {KNOWN_PEAK_TS} present in test : YES, {test.loc[KNOWN_PEAK_TS,'load_MW']:,.1f} MW")
say(f"  train rows / test rows               : {len(train):,} / {len(test):,}")
say(f"  test range                           : {test.index.min()} -> {test.index.max()}")
say(f"  test max demand                      : {test['load_MW'].max():,.1f} MW")
say(f"  rows dropped for missing forecast    : {len(g.dropna(subset=TIER1+['load_MW'])) - len(f):,} "
    f"(forecast archive has gaps in 2023-24; identical rows used for A, B and reference)")

# ---- how good is the 24h temperature forecast itself? ----
te = test["temp_fcst_C"] - test["temp_C"]
tr_e = train["temp_fcst_C"] - train["temp_C"]
say()
say("QUALITY OF THE 24h-LEAD TEMPERATURE FORECAST ITSELF (this bounds what Model B can gain)")
say("-" * 94)
say(f"  test window : MAE {te.abs().mean():.2f} C, bias {te.mean():+.2f} C, "
    f"sd {te.std():.2f} C, worst {te.abs().max():.2f} C")
say(f"  train window: MAE {tr_e.abs().mean():.2f} C, bias {tr_e.mean():+.2f} C")
say("  (a non-zero error confirms this is a genuine forecast, not observed weather relabelled)")

# ---- metric definitions, fixed before fitting ----
peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()
obs_daily_tmax = test.groupby(test.index.normalize())["temp_C"].max()
heat_days = set(obs_daily_tmax[obs_daily_tmax >= HEATWAVE_TMAX].index)
is_heat = np.array([ts.normalize() in heat_days for ts in test.index])

say()
say("METRIC DEFINITIONS (fixed in advance)")
say("-" * 94)
say(f"  peak period      : actual >= P{int(PEAK_QUANTILE*100)} of actual in test = {peak_threshold:,.1f} MW "
    f"({int(is_peak.sum()):,} blocks)")
say(f"  daily peaks      : highest-actual block of each of the {len(daily_peak_idx)} test days")
say(f"  heatwave period  : all blocks on days whose OBSERVED daily max temp >= {HEATWAVE_TMAX} C "
    f"({len(heat_days)} days, {int(is_heat.sum()):,} blocks)")

def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

y = test["load_MW"].to_numpy(float)
preds = {
    "A. OLS Tier-1 (no forward weather)": fit_ols(train, TIER1)(test),
    "B. OLS + 24h FORECAST temperature": fit_ols(train, FCST)(test),
}
ref = {"REF. OLS + observed future temp (not operational)": fit_ols(train, OBSREF)(test)}

def metrics(p):
    p = np.asarray(p, float); err = y - p
    ser = pd.Series(p, index=test.index)
    dp_a = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
    dp_p = ser.loc[daily_peak_idx].to_numpy(float)
    dp_u = (dp_a - dp_p) / dp_a * 100
    pk = float(ser.loc[KNOWN_PEAK_TS])
    return {"MAPE": float(np.mean(np.abs(err / y)) * 100),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "PeakMAPE": float(np.mean(np.abs(err[is_peak] / y[is_peak])) * 100),
            "HeatMAPE": float(np.mean(np.abs(err[is_heat] / y[is_heat])) * 100),
            "DailyPeakMAPE": float(np.mean(np.abs(dp_u))),
            "KnownPeak_pred": pk,
            "KnownPeak_err_MW": pk - KNOWN_PEAK_MW,
            "KnownPeak_err_pct": (pk - KNOWN_PEAK_MW) / KNOWN_PEAK_MW * 100,
            "n_under_5": int((dp_u > 5).sum()), "n_under_10": int((dp_u > 10).sum()),
            "n_days": int(len(dp_u)), "MaxUnder_MW": float(err.max())}

res = {k: metrics(v) for k, v in preds.items()}
res_ref = {k: metrics(v) for k, v in ref.items()}

say()
say("RESULTS -- identical rows, identical target, identical horizon")
say("-" * 94)
say(f"{'Model':<38}{'MAPE':>7}{'MAE':>8}{'RMSE':>8}{'PeakMAPE':>10}{'HeatMAPE':>10}{'DailyPk':>9}")
say("-" * 94)
for k, m in res.items():
    say(f"{k:<38}{m['MAPE']:>6.2f}%{m['MAE']:>8.1f}{m['RMSE']:>8.1f}{m['PeakMAPE']:>9.2f}%"
        f"{m['HeatMAPE']:>9.2f}%{m['DailyPeakMAPE']:>8.2f}%")
for k, m in res_ref.items():
    say(f"{k:<38}{m['MAPE']:>6.2f}%{m['MAE']:>8.1f}{m['RMSE']:>8.1f}{m['PeakMAPE']:>9.2f}%"
        f"{m['HeatMAPE']:>9.2f}%{m['DailyPeakMAPE']:>8.2f}%")
say("-" * 94)

say()
say("EXTREME-PEAK DETAIL")
say("-" * 94)
say(f"{'Model':<38}{'Err@8392.6':>14}{'>5% under':>12}{'>10% under':>12}{'MaxUnder':>11}")
say("-" * 94)
for k, m in list(res.items()) + list(res_ref.items()):
    say(f"{k:<38}{m['KnownPeak_err_MW']:>+9.1f} MW{m['n_under_5']:>8}/{m['n_days']:<3}"
        f"{m['n_under_10']:>8}/{m['n_days']:<3}{m['MaxUnder_MW']:>11.1f}")
say("-" * 94)

a, b = res["A. OLS Tier-1 (no forward weather)"], res["B. OLS + 24h FORECAST temperature"]
r = res_ref["REF. OLS + observed future temp (not operational)"]
say()
say("A -> B CHANGE (positive = forecast weather helped)")
say("-" * 94)
for lbl, key in [("overall MAPE", "MAPE"), ("peak-period MAPE", "PeakMAPE"),
                 ("heatwave-period MAPE", "HeatMAPE"), ("daily-peak MAPE", "DailyPeakMAPE")]:
    gain = a[key] - b[key]
    ceiling = a[key] - r[key]
    frac = (gain / ceiling * 100) if abs(ceiling) > 1e-9 else float("nan")
    say(f"  {lbl:<24} {a[key]:>6.2f}% -> {b[key]:>6.2f}%  ({gain:+.2f} pp)   "
        f"perfect-weather ceiling {r[key]:.2f}% ({ceiling:+.2f} pp), captured {frac:.0f}%")
say(f"  error at the 8,392.6 MW peak  {a['KnownPeak_err_MW']:+,.1f} MW -> {b['KnownPeak_err_MW']:+,.1f} MW "
    f"(perfect-weather ref {r['KnownPeak_err_MW']:+,.1f} MW)")
say(f"  daily peaks missed >10%       {a['n_under_10']}/{a['n_days']} -> {b['n_under_10']}/{b['n_days']} "
    f"(ref {r['n_under_10']}/{r['n_days']})")

# ---------------- figures ----------------
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

pa, pb = preds["A. OLS Tier-1 (no forward weather)"], preds["B. OLS + 24h FORECAST temperature"]

lo_t, hi_t = KNOWN_PEAK_TS - pd.Timedelta(days=3), KNOWN_PEAK_TS + pd.Timedelta(days=3)
m = (test.index >= lo_t) & (test.index <= hi_t)
fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.plot(test.index[m], y[m], color=INK, linewidth=1.6, label="Actual demand")
ax.plot(test.index[m], pa[m], color=COL_A, linewidth=1.6, label="A - no forward weather")
ax.plot(test.index[m], pb[m], color=COL_B, linewidth=1.6, label="B - with 24h forecast temp")
ax.scatter([KNOWN_PEAK_TS], [KNOWN_PEAK_MW], s=70, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
style(ax)
ax.set_title(f"Zoom around the season peak ({KNOWN_PEAK_TS:%d %b %Y}): does forecast weather close the gap?",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase4_fig1_peak_zoom.png", dpi=150, facecolor=SURFACE); plt.close(fig)

dp_a = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
ua = (dp_a - pd.Series(pa, index=test.index).loc[daily_peak_idx].to_numpy(float)) / dp_a * 100
ub = (dp_a - pd.Series(pb, index=test.index).loc[daily_peak_idx].to_numpy(float)) / dp_a * 100
days = pd.DatetimeIndex(daily_peak_idx.values)
fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
ax.axhline(0, color="#8a8f88", linewidth=1)
ax.axhline(5, color="#c04530", linewidth=1, linestyle=(0, (4, 4)), alpha=.35)
ax.plot(days, ua, color=COL_A, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="A - no forward weather")
ax.plot(days, ub, color=COL_B, linewidth=1.6, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="B - with 24h forecast temp")
style(ax)
ax.set_title("Daily peak underprediction -- positive means the forecast was too LOW",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Underprediction (%)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase4_fig2_daily_peak_error.png", dpi=150, facecolor=SURFACE); plt.close(fig)

out = test[["load_MW", "temp_C", "temp_fcst_C"]].copy()
for k, v in {**preds, **ref}.items():
    out[k] = v
out.to_csv(OUT_DIR / "phase4_test_predictions.csv")
(OUT_DIR / "phase4_results.json").write_text(json.dumps(
    {"models": res, "reference_not_operational": res_ref,
     "temp_forecast_quality": {"test_MAE_C": float(te.abs().mean()), "test_bias_C": float(te.mean()),
                               "test_worst_C": float(te.abs().max())},
     "train_rows": len(train), "test_rows": len(test),
     "test_max_MW": float(test["load_MW"].max())}, indent=2))
(OUT_DIR / "phase4_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say("[SAVED] phase4_report.txt, phase4_results.json, phase4_test_predictions.csv, 2 figures")
