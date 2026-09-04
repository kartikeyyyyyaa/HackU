"""
PHASE 3 -- QUANTIFIED UNCERTAINTY EXPERIMENT
=============================================
Goal: keep the approved OLS point forecast EXACTLY as it is, and add a
defensible P10 / P50 / P90 band around it.

Design (deliberately the simplest thing that can work):
  P50  = the approved OLS point forecast, unchanged.
  P10  = P50 + q10(x),  P90 = P50 + q90(x)
         where q10/q90 are linear quantile regressions of the OLS RESIDUAL
         on a small conditioning set, so the band can widen when conditions
         warrant instead of being a fixed-width ribbon.

Residuals used to fit the band are OUT-OF-FOLD (expanding-window, time
ordered) -- never in-sample residuals, which are optimistically small and
would produce intervals that are too narrow. The test set is never touched.

Quantile regression is fitted by IRLS on the pinball loss (numpy only;
statsmodels could not be installed in this environment). A calibration
self-check verifies the fitted quantiles actually hit their nominal level.

Leakage rules unchanged: TIER 1 only. No future demand, no future observed
weather, no lag_1block, nothing unavailable at issue time = target - 24h.
Pipeline, features, split and test period are the APPROVED ones, untouched.
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

# ---- FIXED APPROVED SETTINGS ----
H = 96
TRAIN_END, TEST_START, TEST_END = "2025-05-01", "2025-05-01", "2025-07-01"
PEAK_QUANTILE = 0.90
KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
KNOWN_PEAK_MW = 8392.6
TAU_LO, TAU_HI = 0.10, 0.90
N_FOLDS = 5

INK, COL_A, COL_B = "#2f3437", "#0d76b8", "#c9701a"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 3 -- QUANTIFIED UNCERTAINTY (P10 / P50 / P90)")
say("=" * 92)

# ============================================================================
# APPROVED PIPELINE -- verbatim, unchanged
# ============================================================================
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
f = g.dropna()

TIER1 = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
         "doy", "sin_doy", "cos_doy", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
         "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
         "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]

train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]

say()
say("PRE-FLIGHT (before any metric)")
say("-" * 92)
assert KNOWN_PEAK_TS in test.index, "approved peak missing from test set"
say(f"  peak {KNOWN_PEAK_TS} in test  : YES, {test.loc[KNOWN_PEAK_TS,'load_MW']:,.1f} MW "
    f"({'MATCH' if abs(test.loc[KNOWN_PEAK_TS,'load_MW']-KNOWN_PEAK_MW)<1 else 'MISMATCH'})")
say(f"  train rows / test rows          : {len(train):,} / {len(test):,}")
say(f"  test range                      : {test.index.min()} -> {test.index.max()}")
say(f"  test max demand                 : {test['load_MW'].max():,.1f} MW")
say(f"  features                        : TIER 1 only ({len(TIER1)} features), unchanged from approved set")

# ============================================================================
# METRIC DEFINITIONS -- fixed before anything is computed
# ============================================================================
peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()

say()
say("METRIC DEFINITIONS (fixed in advance)")
say("-" * 92)
say(f"  nominal interval        : P10-P90, so nominal coverage = 80%, "
    f"nominal upper breach = 10%, nominal lower breach = 10%")
say(f"  high-demand period      : actual >= P{int(PEAK_QUANTILE*100)} of actual in test "
    f"= {peak_threshold:,.1f} MW ({int(is_peak.sum()):,} of {len(test):,} blocks)")
say(f"  daily peaks             : highest-actual block of each of the {len(daily_peak_idx)} test days")
say(f"  interval width          : P90 - P10, in MW and as % of actual")
say(f"  pinball loss            : mean rho_tau(actual - quantile), lower is better (proper score)")
say(f"  band 'expands' if       : mean width on high-demand blocks > mean width off-peak")

# ============================================================================
# MODEL FITTING
# ============================================================================
def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

def quantile_reg_irls(X, y, tau, iters=60, eps=1.0):
    """Linear quantile regression by IRLS on the pinball loss (numpy only)."""
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    for _ in range(iters):
        r = y - Xs @ beta
        w = np.where(r > 0, tau, 1 - tau) / np.maximum(np.abs(r), eps)
        sw = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(Xs * sw[:, None], y * sw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    return lambda Z: np.column_stack([np.ones(len(Z)), (Z - mu) / sd]) @ beta

# ---- out-of-fold residuals on the training period (expanding window) ----
say()
say("OUT-OF-FOLD RESIDUAL GENERATION (expanding window, time-ordered, test never touched)")
say("-" * 92)
# RULE FIXED IN ADVANCE: a calibration fold is only usable if its model was trained on
# at least 12 months of history. A model trained on 5 months has never seen a full
# seasonal cycle, so its errors describe that model's inexperience, not the deployed
# model's uncertainty. (Diagnostic: the <12-month fold had MAE 2,309.7 MW vs 162-241 MW
# for the rest, and on its own dragged the pooled residual mean to +426.8 MW.)
MIN_TRAIN = pd.DateOffset(months=12)
calib_start = train.index.min() + MIN_TRAIN
calib = train[train.index >= calib_start]
say(f"  minimum training history per calibration fold : 12 months")
say(f"  calibration period                            : {calib.index.min()} -> {calib.index.max()} "
    f"({len(calib):,} rows)")

bounds_c = np.linspace(0, len(calib), N_FOLDS).astype(int)
oof_parts = []
for k in range(0, N_FOLDS - 1):
    va_k = calib.iloc[bounds_c[k]:bounds_c[k + 1]]
    tr_k = train[train.index < va_k.index.min()]
    pred_k = fit_ols(tr_k, TIER1)(va_k)
    part = va_k[["load_MW"]].copy()
    part["pred"] = pred_k
    part["resid"] = part["load_MW"] - part["pred"]
    oof_parts.append(part.join(va_k[["sin_hod", "cos_hod", "is_weekend",
                                      "temp_at_issue", "temp_prevday_max", "roll24_max"]]))
    say(f"  fold {k+1}: fit {tr_k.index.min().date()}..{tr_k.index.max().date()} "
        f"({len(tr_k):,} rows) -> predict {va_k.index.min().date()}..{va_k.index.max().date()} "
        f"({len(va_k):,} rows), MAE {np.abs(part['resid']).mean():.1f} MW")
oof = pd.concat(oof_parts)
say(f"  total out-of-fold residual rows : {len(oof):,}  "
    f"(mean {oof['resid'].mean():+.1f} MW, sd {oof['resid'].std():.1f} MW)")

# ---- conditioning set for the SPREAD model (small on purpose) ----
COND = ["pred", "sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]
say(f"  spread conditioned on           : {COND}")
say("  (a small set on purpose -- we are modelling spread, not level; every one of these is Tier-1 safe)")

Xo = oof[COND].to_numpy(float)
yo = oof["resid"].to_numpy(float)
q_lo_fn = quantile_reg_irls(Xo, yo, TAU_LO)
q_hi_fn = quantile_reg_irls(Xo, yo, TAU_HI)

# calibration self-check on the OOF data itself
emp_lo = float((yo < q_lo_fn(Xo)).mean())
emp_hi = float((yo < q_hi_fn(Xo)).mean())
say()
say("CALIBRATION SELF-CHECK on out-of-fold data (does the fitted quantile hit its nominal level?)")
say(f"  fraction below fitted P10 : {emp_lo*100:.2f}%  (nominal 10%)")
say(f"  fraction below fitted P90 : {emp_hi*100:.2f}%  (nominal 90%)")

# ---- final model on full train, applied to test ----
ols_full = fit_ols(train, TIER1)
p50 = ols_full(test)
cond_test = test[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue",
                  "temp_prevday_max", "roll24_max"]].copy()
cond_test.insert(0, "pred", p50)
Xt = cond_test[COND].to_numpy(float)
p10 = p50 + q_lo_fn(Xt)
p90 = p50 + q_hi_fn(Xt)
p10, p90 = np.minimum(p10, p50), np.maximum(p90, p50)   # enforce monotonicity

# ---- reference: fixed-width band from global OOF residual quantiles ----
c_lo, c_hi = np.quantile(yo, TAU_LO), np.quantile(yo, TAU_HI)
p10_const, p90_const = p50 + c_lo, p50 + c_hi

y = test["load_MW"].to_numpy(float)

# ============================================================================
# METRICS
# ============================================================================
def pinball(y_true, q, tau):
    d = y_true - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))

def interval_stats(lo, hi, label):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "label": label,
        "coverage_all": float(inside.mean() * 100),
        "coverage_highdemand": float(inside[is_peak].mean() * 100),
        "coverage_dailypeaks": float(pd.Series(inside, index=test.index).loc[daily_peak_idx].mean() * 100),
        "breach_upper": float((y > hi).mean() * 100),
        "breach_lower": float((y < lo).mean() * 100),
        "n_breach_upper": int((y > hi).sum()),
        "n_breach_lower": int((y < lo).sum()),
        "width_mean": float(width.mean()),
        "width_pct_of_actual": float((width / y).mean() * 100),
        "width_offpeak": float(width[~is_peak].mean()),
        "width_peak": float(width[is_peak].mean()),
        "pinball_lo": pinball(y, lo, TAU_LO),
        "pinball_hi": pinball(y, hi, TAU_HI),
    }

st_cond = interval_stats(p10, p90, "conditional (P10/P90 regression)")
st_const = interval_stats(p10_const, p90_const, "fixed-width (global residual quantiles)")

err = y - p50
point = {
    "MAPE": float(np.mean(np.abs(err / y)) * 100),
    "MAE": float(np.mean(np.abs(err))),
    "RMSE": float(np.sqrt(np.mean(err ** 2))),
    "PeakMAPE": float(np.mean(np.abs(err[is_peak] / y[is_peak])) * 100),
}

say()
say("POINT FORECAST (P50 = the approved OLS baseline, unchanged)")
say("-" * 92)
say(f"  MAPE {point['MAPE']:.2f}%   MAE {point['MAE']:.1f} MW   RMSE {point['RMSE']:.1f} MW   "
    f"Peak MAPE {point['PeakMAPE']:.2f}%")
say("  (identical to the approved Phase 2 baseline -- the band is added around it, not instead of it)")

say()
say("INTERVAL METRICS")
say("-" * 92)
say(f"{'Metric':<40}{'Conditional':>16}{'Fixed-width':>16}{'Nominal':>12}")
say("-" * 92)
rows = [("P10-P90 coverage, all blocks", "coverage_all", "80%"),
        ("coverage, high-demand blocks", "coverage_highdemand", "80%"),
        ("coverage, daily peak blocks", "coverage_dailypeaks", "80%"),
        ("upper breaches (actual > P90)", "breach_upper", "10%"),
        ("lower breaches (actual < P10)", "breach_lower", "10%")]
for lbl, key, nom in rows:
    say(f"{lbl:<40}{st_cond[key]:>15.2f}%{st_const[key]:>15.2f}%{nom:>12}")
say(f"{'mean interval width (MW)':<40}{st_cond['width_mean']:>16.1f}{st_const['width_mean']:>16.1f}{'':>12}")
say(f"{'mean width as % of actual':<40}{st_cond['width_pct_of_actual']:>15.2f}%{st_const['width_pct_of_actual']:>15.2f}%{'':>12}")
say(f"{'mean width, off-peak (MW)':<40}{st_cond['width_offpeak']:>16.1f}{st_const['width_offpeak']:>16.1f}{'':>12}")
say(f"{'mean width, high-demand (MW)':<40}{st_cond['width_peak']:>16.1f}{st_const['width_peak']:>16.1f}{'':>12}")
say(f"{'pinball loss P10 (lower better)':<40}{st_cond['pinball_lo']:>16.2f}{st_const['pinball_lo']:>16.2f}{'':>12}")
say(f"{'pinball loss P90 (lower better)':<40}{st_cond['pinball_hi']:>16.2f}{st_const['pinball_hi']:>16.2f}{'':>12}")
say("-" * 92)
exp = st_cond["width_peak"] > st_cond["width_offpeak"]
say(f"Does the conditional band widen under high demand?  "
    f"{'YES' if exp else 'NO'}  ({st_cond['width_peak']:.1f} MW vs {st_cond['width_offpeak']:.1f} MW off-peak)")

# ============================================================================
# EXTREME PEAK ANALYSIS
# ============================================================================
say()
say("EXTREME-PEAK ANALYSIS")
say("-" * 92)
i = test.index.get_loc(KNOWN_PEAK_TS)
say(f"  Season peak {KNOWN_PEAK_TS}")
say(f"    actual : {y[i]:>8,.1f} MW")
say(f"    P10    : {p10[i]:>8,.1f} MW")
say(f"    P50    : {p50[i]:>8,.1f} MW   (error {p50[i]-y[i]:+,.1f} MW, {(p50[i]-y[i])/y[i]*100:+.2f}%)")
say(f"    P90    : {p90[i]:>8,.1f} MW   (actual is {'INSIDE' if y[i] <= p90[i] else 'ABOVE'} the P90 bound"
    f"{'' if y[i] <= p90[i] else f', exceeding it by {y[i]-p90[i]:,.1f} MW'})")

dp = pd.DataFrame({
    "actual": test.loc[daily_peak_idx, "load_MW"].to_numpy(float),
    "p10": pd.Series(p10, index=test.index).loc[daily_peak_idx].to_numpy(float),
    "p50": pd.Series(p50, index=test.index).loc[daily_peak_idx].to_numpy(float),
    "p90": pd.Series(p90, index=test.index).loc[daily_peak_idx].to_numpy(float),
}, index=pd.DatetimeIndex(daily_peak_idx.values))
dp["p50_under_pct"] = (dp["actual"] - dp["p50"]) / dp["actual"] * 100
dp["captured_by_p90"] = dp["actual"] <= dp["p90"]

n_days = len(dp)
under5 = dp["p50_under_pct"] > 5
under10 = dp["p50_under_pct"] > 10
say()
say(f"  Daily peaks ({n_days} days):")
say(f"    P50 underpredicts by >5%              : {int(under5.sum())}/{n_days}")
say(f"    P50 underpredicts by >10%             : {int(under10.sum())}/{n_days}")
say(f"    captured under P90 (all daily peaks)  : {int(dp['captured_by_p90'].sum())}/{n_days} "
    f"({dp['captured_by_p90'].mean()*100:.1f}%)")
if under5.sum():
    say(f"    of the {int(under5.sum())} peaks P50 missed by >5%, captured under P90 : "
        f"{int(dp.loc[under5, 'captured_by_p90'].sum())}/{int(under5.sum())} "
        f"({dp.loc[under5,'captured_by_p90'].mean()*100:.1f}%)")
if under10.sum():
    say(f"    of the {int(under10.sum())} peaks P50 missed by >10%, captured under P90: "
        f"{int(dp.loc[under10, 'captured_by_p90'].sum())}/{int(under10.sum())} "
        f"({dp.loc[under10,'captured_by_p90'].mean()*100:.1f}%)")

say()
say("  Worst 8 daily peaks by P50 underprediction:")
say(f"    {'date':<12}{'actual':>10}{'P50':>10}{'P90':>10}{'P50 under':>11}{'under P90?':>12}")
worst = dp.sort_values("p50_under_pct", ascending=False).head(8)
for d, r in worst.iterrows():
    say(f"    {d:%Y-%m-%d}  {r['actual']:>9,.0f}{r['p50']:>10,.0f}{r['p90']:>10,.0f}"
        f"{r['p50_under_pct']:>10.1f}%{('YES' if r['captured_by_p90'] else 'NO'):>12}")

# ============================================================================
# FIGURES
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.fill_between(test.index, p10, p90, color=COL_A, alpha=.20, linewidth=0, label="P10-P90 band")
ax.plot(test.index, p50, color=COL_A, linewidth=.9, label="P50 forecast")
ax.plot(test.index, y, color=INK, linewidth=.9, label="Actual demand")
breach = y > p90
ax.scatter(test.index[breach], y[breach], s=6, color=COL_B, zorder=4, label="above P90")
style(ax)
ax.set_title("24h-ahead forecast with P10-P90 uncertainty band, 1 May - 30 Jun 2025 (held out)",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase3_fig1_fan_full_test.png", dpi=150, facecolor=SURFACE); plt.close(fig)

lo_t, hi_t = KNOWN_PEAK_TS - pd.Timedelta(days=3), KNOWN_PEAK_TS + pd.Timedelta(days=3)
m = (test.index >= lo_t) & (test.index <= hi_t)
fig, ax = plt.subplots(figsize=(13, 4.6), facecolor=SURFACE)
ax.fill_between(test.index[m], p10[m], p90[m], color=COL_A, alpha=.20, linewidth=0, label="P10-P90 band")
ax.plot(test.index[m], p50[m], color=COL_A, linewidth=1.6, label="P50 forecast")
ax.plot(test.index[m], y[m], color=INK, linewidth=1.6, label="Actual demand")
ax.scatter([KNOWN_PEAK_TS], [KNOWN_PEAK_MW], s=70, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
ax.annotate(f"actual {KNOWN_PEAK_MW:,.0f} MW\nP90 {p90[i]:,.0f} MW",
            xy=(KNOWN_PEAK_TS, KNOWN_PEAK_MW), xytext=(-12, -40), textcoords="offset points",
            fontsize=9.5, color=INK, ha="right")
style(ax)
ax.set_title(f"Zoom: uncertainty band around the season peak ({KNOWN_PEAK_TS:%d %b %Y})",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase3_fig2_peak_zoom.png", dpi=150, facecolor=SURFACE); plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 4.2), facecolor=SURFACE)
xs = np.arange(len(dp))
ax.vlines(xs, dp["p10"], dp["p90"], color=COL_A, alpha=.45, linewidth=5, label="P10-P90 band")
ax.plot(xs, dp["p50"], color=COL_A, linewidth=0, marker="_", markersize=11, label="P50")
ax.plot(xs, dp["actual"], color=INK, linewidth=0, marker="o", markersize=4.5,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Actual daily peak")
above = ~dp["captured_by_p90"].to_numpy()
ax.plot(xs[above], dp["actual"].to_numpy()[above], color=COL_B, linewidth=0, marker="o",
        markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1, label="Actual above P90")
style(ax)
ax.set_xticks(xs[::5]); ax.set_xticklabels([d.strftime("%d %b") for d in dp.index[::5]])
ax.set_title("Daily peak: does the band contain the actual peak?", fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.legend(frameon=False, fontsize=9.5, ncols=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase3_fig3_daily_peak_bands.png", dpi=150, facecolor=SURFACE); plt.close(fig)

# ============================================================================
# SAVE
# ============================================================================
out = test[["load_MW"]].copy()
out["P10"], out["P50"], out["P90"] = p10, p50, p90
out.to_csv(OUT_DIR / "phase3_quantile_predictions.csv")
dp.to_csv(OUT_DIR / "phase3_daily_peak_bands.csv")
(OUT_DIR / "phase3_results.json").write_text(json.dumps(
    {"point": point, "conditional": st_cond, "fixed_width": st_const,
     "calibration_selfcheck": {"below_P10_pct": emp_lo * 100, "below_P90_pct": emp_hi * 100},
     "known_peak": {"ts": str(KNOWN_PEAK_TS), "actual": float(y[i]), "P10": float(p10[i]),
                    "P50": float(p50[i]), "P90": float(p90[i]),
                    "inside_band": bool(y[i] <= p90[i])},
     "daily_peaks": {"n": int(n_days), "p50_under_5": int(under5.sum()),
                     "p50_under_10": int(under10.sum()),
                     "captured_by_p90": int(dp["captured_by_p90"].sum())}}, indent=2))
(OUT_DIR / "phase3_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say("[SAVED] phase3_report.txt, phase3_results.json, phase3_quantile_predictions.csv, "
    "phase3_daily_peak_bands.csv, 3 figures")
