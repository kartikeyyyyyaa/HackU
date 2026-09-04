"""
PHASE 6 -- CRITICAL INPUT ROBUSTNESS
=====================================
Question: how much of the peak underprediction is caused by MISSING recent-
demand telemetry (imputed lag_24h), and what is the safest way to handle it?

Model, pipeline, target, resolution, split, leakage rules and the approved
adaptive weather correction are all LOCKED. The only thing that varies is
how missing FEATURE-INPUT demand blocks are filled, plus an optional
reliability flag.

STRATEGIES
  A. current  : seasonal-naive -- fill from the same block 7 days earlier,
                chained (7, 14, 21 ... days back), then linear interpolation.
  B. nearest  : fill from the same block on the NEAREST available earlier day
                (1 day back, then 2, then 3 ...), always copying an OBSERVED
                value, never a previously-imputed one.
  C. flag     : strategy B plus an explicit `lag24_imputed` indicator exposed
                to the model, so it can learn to behave differently when its
                critical input is not real telemetry.

Everything uses only information available at issue time (target - 24h).
The target series is NEVER imputed and imputed targets are never scored.
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
SEED = 42

INK, COL_A, COL_B, COL_C = "#2f3437", "#0d76b8", "#c9701a", "#2f7d4f"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 6 -- CRITICAL INPUT ROBUSTNESS")
say("=" * 100)

# ---------------- raw series ----------------
load = (pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp"))
s_raw = load["load_MW"].resample("15min").mean()
s15 = s_raw.interpolate(limit=4)              # TARGET: real + <=1h interpolation only
observed = s15.notna()

wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")
fcr = (pd.read_csv(DATA_DIR / "delhi_weather_forecast_day1.csv", parse_dates=["timestamp"])
         .drop_duplicates(subset="timestamp").set_index("timestamp"))
fc15 = fcr["temp_fcst_C"].resample("15min").interpolate(method="time", limit=4)

# ============================================================================
# 1. GAP CHARACTERISATION
# ============================================================================
say()
say("1. GAP CHARACTERISATION")
say("-" * 100)
miss = ~observed
runs = (miss != miss.shift()).cumsum()
gap_sizes = miss.groupby(runs).sum()
gap_sizes = gap_sizes[gap_sizes > 0]
say(f"  15-min blocks with no observed demand : {int(miss.sum()):,} of {len(s15):,} ({miss.mean()*100:.2f}%)")
say(f"  number of separate gaps               : {len(gap_sizes):,}")
say(f"  largest gap                           : {int(gap_sizes.max())} blocks "
    f"({gap_sizes.max()*15/60:.1f} h)")
say(f"  gaps >= 1 day                         : {int((gap_sizes >= H).sum())}")

# which TARGET blocks have an imputed lag_24h?
lag24_imputed_all = (~observed).shift(H).fillna(True)
test_mask_full = (s15.index >= TEST_START) & (s15.index < TEST_END)
say(f"  target blocks whose lag_24h is imputed (whole series) : "
    f"{int(lag24_imputed_all.sum()):,} ({lag24_imputed_all.mean()*100:.2f}%)")
say(f"  ... within the approved test window                   : "
    f"{int(lag24_imputed_all[test_mask_full].sum()):,} "
    f"({lag24_imputed_all[test_mask_full].mean()*100:.2f}%)")

say()
say("2. AFFECTED TIMESTAMPS IN THE TEST WINDOW")
say("-" * 100)
aff = s15.index[test_mask_full & lag24_imputed_all]
if len(aff):
    aff_days = pd.Series(1, index=aff).groupby(aff.normalize()).sum()
    for d, n in aff_days.items():
        say(f"  {d:%Y-%m-%d}: {int(n)} target blocks with imputed lag_24h "
            f"(source day {d - pd.Timedelta(days=1):%Y-%m-%d} had "
            f"{int(observed.loc[str((d - pd.Timedelta(days=1)).date())].sum())}/96 blocks observed)")

# ============================================================================
# 3. IMPUTATION STRATEGIES
# ============================================================================
def impute_seasonal7(s):
    """A -- current: chained 7-day-back fill, then linear interpolation."""
    out = s.copy()
    for _ in range(6):
        out = out.fillna(out.shift(H * 7))
    return out.interpolate(limit_direction="both")

def impute_nearest_day(s, max_days=14):
    """B -- nearest available earlier day, same block; only ever copies OBSERVED values."""
    out = s.copy()
    for k in range(1, max_days + 1):
        out = out.fillna(s.shift(H * k))          # note: s, not out -> never chains imputed values
    return out.interpolate(limit_direction="both")

IMPUTERS = {"A. seasonal-naive 7d (current)": impute_seasonal7,
            "B. nearest available day": impute_nearest_day}

# ---- imputation quality measured by SIMULATED gaps (real gaps are only 39 test blocks) ----
say()
say("3a. IMPUTATION QUALITY VIA SIMULATED GAPS")
say("-" * 100)
say("  Real gaps give only 39 test blocks -- far too few to judge a rule on. So we hide")
say("  OBSERVED blocks at random, impute them, and compare against the truth we hid.")
rng = np.random.default_rng(SEED)
obs_idx = s15.index[observed]
summer = obs_idx[(obs_idx.month.isin([4, 5, 6])) & (obs_idx >= "2024-01-01")]
# simulate realistic outages: whole-evening blocks, mirroring the 10 June pattern
sim_days = rng.choice(pd.unique(summer.normalize()), size=60, replace=False)
sim_mask = pd.Series(False, index=s15.index)
for d in sim_days:
    d = pd.Timestamp(d)
    sim_mask.loc[d + pd.Timedelta(hours=21): d + pd.Timedelta(hours=23, minutes=45)] = True
sim_mask &= observed
s_holed = s15.mask(sim_mask)
say(f"  simulated outage blocks: {int(sim_mask.sum()):,} (evenings of {len(sim_days)} summer days)")
say()
say(f"  {'strategy':<34}{'MAE':>10}{'RMSE':>10}{'bias':>10}{'worst':>10}{'MAPE':>9}")
say("  " + "-" * 82)
truth = s15[sim_mask]
imp_quality = {}
for name, fn in IMPUTERS.items():
    filled = fn(s_holed)[sim_mask]
    e = filled - truth
    imp_quality[name] = {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                          "bias": float(e.mean()), "worst": float(e.abs().max()),
                          "MAPE": float((e.abs() / truth).mean() * 100)}
    m = imp_quality[name]
    say(f"  {name:<34}{m['MAE']:>10.1f}{m['RMSE']:>10.1f}{m['bias']:>+10.1f}{m['worst']:>10.1f}{m['MAPE']:>8.2f}%")
say("  " + "-" * 82)

# ============================================================================
# 4. END-TO-END DEMAND EXPERIMENT
# ============================================================================
def build_frame(imputer, add_flag):
    s_feat = imputer(s15)
    g = pd.DataFrame({"load_MW": s15, "load_feat": s_feat})
    g["temp_C"] = wx15.reindex(g.index)
    g["temp_fcst_C"] = fc15.reindex(g.index)
    g = g.dropna(subset=["temp_C", "load_feat"])
    g["block_of_day"] = (g.index.hour * 4 + g.index.minute // 15).astype(int)

    # approved adaptive weather correction (rolling 30d by time of day, leakage-safe)
    err = g["temp_C"] - g["temp_fcst_C"]
    corr = err.shift(H).groupby(g["block_of_day"]).transform(
        lambda x: x.rolling(ROLL_DAYS, min_periods=7).mean())
    g["temp_corr"] = g["temp_fcst_C"] + corr

    lf = g["load_feat"]
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
    tp = g["temp_C"].shift(H)
    g["temp_at_issue"] = tp
    g["temp_prevday_max"] = tp.rolling(H, min_periods=int(H * .75)).max()
    g["temp_prevday_mean"] = tp.rolling(H, min_periods=int(H * .75)).mean()
    g["ct"] = g["temp_corr"]
    g["ccdh"] = np.clip(g["temp_corr"] - 24, 0, None)
    g["csq"] = g["temp_corr"] ** 2
    g["chx"] = np.clip(g["temp_corr"] - 38, 0, None) ** 2
    g["lag24_imputed"] = (~observed).shift(H).reindex(g.index).fillna(True).astype(int)

    cols = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
            "doy", "sin_doy", "cos_doy", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
            "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
            "temp_at_issue", "temp_prevday_max", "temp_prevday_mean",
            "ct", "ccdh", "csq", "chx"]
    if add_flag:
        cols = cols + ["lag24_imputed"]
    return g.dropna(subset=cols + ["load_MW"]), cols

def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

variants = {
    "A. current imputation": (impute_seasonal7, False),
    "B. nearest-day imputation": (impute_nearest_day, False),
    "C. nearest-day + imputed flag": (impute_nearest_day, True),
}

results, preds_store, frames = {}, {}, {}
ref_test_index = None
for name, (imp, flag) in variants.items():
    f, cols = build_frame(imp, flag)
    tr = f[f.index < TRAIN_END]
    te = f[(f.index >= TEST_START) & (f.index < TEST_END)]
    if ref_test_index is None:
        ref_test_index = te.index
    else:
        assert te.index.equals(ref_test_index), f"{name}: test rows differ -- comparison invalid"
    preds_store[name] = pd.Series(fit_ols(tr, cols)(te), index=te.index)
    frames[name] = (tr, te)

tr0, test = frames["A. current imputation"]
y = test["load_MW"].to_numpy(float)

say()
say("PRE-FLIGHT")
say("-" * 100)
assert KNOWN_PEAK_TS in test.index
say(f"  peak {KNOWN_PEAK_TS} present : YES, {test.loc[KNOWN_PEAK_TS,'load_MW']:,.1f} MW")
say(f"  test rows (identical across all 3 strategies) : {len(test):,}")
say(f"  test max demand : {test['load_MW'].max():,.1f} MW")

peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()
obs_tmax = test.groupby(test.index.normalize())["temp_C"].max()
heat_days = set(obs_tmax[obs_tmax >= HEATWAVE_TMAX].index)
is_heat = np.array([t.normalize() in heat_days for t in test.index])
imp_flag = test["lag24_imputed"].to_numpy(bool) if "lag24_imputed" in test.columns else \
    (~observed).shift(H).reindex(test.index).fillna(True).to_numpy(bool)

def metrics(p, mask=None):
    p = np.asarray(p, float)
    yy, pp = (y, p) if mask is None else (y[mask], p[mask])
    if len(yy) == 0:
        return None
    e = yy - pp
    out = {"n": int(len(yy)), "MAPE": float(np.mean(np.abs(e / yy)) * 100),
           "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
           "MaxUnder_MW": float(e.max())}
    if mask is None:
        ser = pd.Series(p, index=test.index)
        dpa = test.loc[daily_peak_idx, "load_MW"].to_numpy(float)
        du = (dpa - ser.loc[daily_peak_idx].to_numpy(float)) / dpa * 100
        out.update({"PeakMAPE": float(np.mean(np.abs(e[is_peak] / y[is_peak])) * 100),
                    "HeatMAPE": float(np.mean(np.abs(e[is_heat] / y[is_heat])) * 100),
                    "DailyPeakMAPE": float(np.mean(np.abs(du))),
                    "n_under_5": int((du > 5).sum()), "n_under_10": int((du > 10).sum()),
                    "n_days": int(len(du)),
                    "KnownPeak_err_MW": float(ser.loc[KNOWN_PEAK_TS] - KNOWN_PEAK_MW)})
    return out

say()
say("4. OVERALL METRICS")
say("-" * 100)
say(f"{'Strategy':<32}{'MAPE':>7}{'MAE':>8}{'RMSE':>8}{'PeakMAPE':>10}{'HeatMAPE':>10}"
    f"{'DailyPk':>9}{'>5%':>6}{'>10%':>6}{'MaxUnder':>10}")
say("-" * 100)
for name in variants:
    m = metrics(preds_store[name].to_numpy())
    results[name] = m
    say(f"{name:<32}{m['MAPE']:>6.2f}%{m['MAE']:>8.1f}{m['RMSE']:>8.1f}{m['PeakMAPE']:>9.2f}%"
        f"{m['HeatMAPE']:>9.2f}%{m['DailyPeakMAPE']:>8.2f}%{m['n_under_5']:>6}{m['n_under_10']:>6}"
        f"{m['MaxUnder_MW']:>10.1f}")
say("-" * 100)
say(f"{'':<32}{'error at the 8,392.6 MW peak:':>40}")
for name in variants:
    say(f"{name:<32}{results[name]['KnownPeak_err_MW']:>+40,.1f} MW")

say()
say("5. OBSERVED-INPUT vs IMPUTED-INPUT PERFORMANCE")
say("-" * 100)
say(f"  test blocks with OBSERVED lag_24h : {int((~imp_flag).sum()):,}")
say(f"  test blocks with IMPUTED  lag_24h : {int(imp_flag.sum()):,}  "
    f"<- SMALL SAMPLE, treat these numbers as indicative only")
say()
say(f"{'Strategy':<32}{'obs MAPE':>10}{'obs MAE':>10}{'imp MAPE':>11}{'imp MAE':>10}{'imp worst under':>18}")
say("-" * 100)
for name in variants:
    p = preds_store[name].to_numpy()
    mo, mi = metrics(p, ~imp_flag), metrics(p, imp_flag)
    results[name]["observed_subset"], results[name]["imputed_subset"] = mo, mi
    say(f"{name:<32}{mo['MAPE']:>9.2f}%{mo['MAE']:>10.1f}{mi['MAPE']:>10.2f}%{mi['MAE']:>10.1f}"
        f"{mi['MaxUnder_MW']:>18.1f}")
say("-" * 100)

# ============================================================================
# 6. PEAK-DAY ANALYSIS
# ============================================================================
say()
say("6. PEAK-DAY ANALYSIS -- 11, 12, 13 June 2025 daily peaks + the season peak")
say("-" * 100)
say(f"{'timestamp':<21}{'actual':>9}{'lag24 obs?':>12}{'lag24 used':>12}"
    f"{'predA':>9}{'predB':>9}{'predC':>9}{'errA':>9}{'errB':>9}")
say("-" * 100)
focus = [test.loc["2025-06-11"]["load_MW"].idxmax(),
         test.loc["2025-06-12"]["load_MW"].idxmax(),
         test.loc["2025-06-13"]["load_MW"].idxmax()]
for ts in focus:
    src = ts - pd.Timedelta(hours=24)
    was_obs = bool(observed.get(src, False))
    lagA = frames["A. current imputation"][1].loc[ts, "lag_24h"]
    lagB = frames["B. nearest-day imputation"][1].loc[ts, "lag_24h"]
    act = test.loc[ts, "load_MW"]
    pA = preds_store["A. current imputation"].loc[ts]
    pB = preds_store["B. nearest-day imputation"].loc[ts]
    pC = preds_store["C. nearest-day + imputed flag"].loc[ts]
    say(f"{str(ts):<21}{act:>9,.0f}{('YES' if was_obs else 'NO'):>12}"
        f"{lagA:>7,.0f}/{lagB:<5,.0f}{pA:>9,.0f}{pB:>9,.0f}{pC:>9,.0f}{pA-act:>+9,.0f}{pB-act:>+9,.0f}")
say("-" * 100)
say("  ('lag24 used' shows the value strategy A vs strategy B fed the model)")
for ts in focus:
    src = ts - pd.Timedelta(hours=24)
    if not bool(observed.get(src, False)):
        say(f"  {ts}: lag_24h source {src} was NOT observed.")
        say(f"      A used {frames['A. current imputation'][1].loc[ts,'lag_24h']:,.0f} MW "
            f"(same block 7 days earlier), B used {frames['B. nearest-day imputation'][1].loc[ts,'lag_24h']:,.0f} MW "
            f"(nearest observed day), actual demand at that time was ~{s15.get(src, float('nan')):,.0f} MW"
            if pd.notna(s15.get(src, np.nan)) else "")

# ============================================================================
# 7. RELIABILITY FLAG + UNCERTAINTY WIDENING
# ============================================================================
say()
say("7. FORECAST RELIABILITY FLAG AND UNCERTAINTY BEHAVIOUR")
say("-" * 100)
best = "B. nearest-day imputation"
p_best = preds_store[best].to_numpy()

# reuse the Phase 3 band construction on the chosen strategy
tr_b, te_b = frames[best]
_, cols_b = build_frame(impute_nearest_day, False)
calib_start = tr_b.index.min() + pd.DateOffset(months=12)
calib = tr_b[tr_b.index >= calib_start]
bounds = np.linspace(0, len(calib), 5).astype(int)
oof = []
for k in range(0, 4):
    va = calib.iloc[bounds[k]:bounds[k + 1]]
    trk = tr_b[tr_b.index < va.index.min()]
    pk = fit_ols(trk, cols_b)(va)
    part = va[["load_MW"]].copy(); part["pred"] = pk
    part["resid"] = part["load_MW"] - part["pred"]
    oof.append(part.join(va[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue",
                             "temp_prevday_max", "roll24_max"]]))
oof = pd.concat(oof)

def qreg(X, yv, tau, iters=60, eps=1.0):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    b, *_ = np.linalg.lstsq(Xs, yv, rcond=None)
    for _ in range(iters):
        r = yv - Xs @ b
        w = np.where(r > 0, tau, 1 - tau) / np.maximum(np.abs(r), eps)
        sw = np.sqrt(w)
        bn, *_ = np.linalg.lstsq(Xs * sw[:, None], yv * sw, rcond=None)
        if np.max(np.abs(bn - b)) < 1e-8:
            b = bn; break
        b = bn
    return lambda Z: np.column_stack([np.ones(len(Z)), (Z - mu) / sd]) @ b

COND = ["pred", "sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]
Xo, yo = oof[COND].to_numpy(float), oof["resid"].to_numpy(float)
qlo, qhi = qreg(Xo, yo, 0.10), qreg(Xo, yo, 0.90)
ct = te_b[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]].copy()
ct.insert(0, "pred", p_best)
Xt = ct[COND].to_numpy(float)
p10 = np.minimum(p_best + qlo(Xt), p_best)
p90 = np.maximum(p_best + qhi(Xt), p_best)

inside = (y >= p10) & (y <= p90)
width = p90 - p10
say(f"  {'group':<34}{'n':>7}{'coverage':>11}{'mean width':>13}{'MAE':>10}")
say("  " + "-" * 76)
for lbl, msk in [("HIGH reliability (lag observed)", ~imp_flag),
                 ("LOW reliability (lag imputed)", imp_flag)]:
    if msk.sum():
        say(f"  {lbl:<34}{int(msk.sum()):>7}{inside[msk].mean()*100:>10.1f}%"
            f"{width[msk].mean():>13.1f}{np.abs(y[msk]-p_best[msk]).mean():>10.1f}")
say("  " + "-" * 76)
need_factor = (np.abs(y[imp_flag] - p_best[imp_flag]).mean() /
               np.abs(y[~imp_flag] - p_best[~imp_flag]).mean()) if imp_flag.sum() else float("nan")
say(f"  error on LOW-reliability rows is {need_factor:.2f}x the error on HIGH-reliability rows,")
say(f"  but the band widens only {width[imp_flag].mean()/width[~imp_flag].mean():.2f}x on them "
    f"-> the band does NOT currently reflect input reliability.")

# ============================================================================
# FIGURES
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

lo_t, hi_t = pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-14")
m = (test.index >= lo_t) & (test.index <= hi_t)
fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
ax.plot(test.index[m], y[m], color=INK, linewidth=1.7, label="Actual demand")
ax.plot(test.index[m], preds_store["A. current imputation"].to_numpy()[m], color=COL_A,
        linewidth=1.5, label="A - current imputation")
ax.plot(test.index[m], preds_store["B. nearest-day imputation"].to_numpy()[m], color=COL_C,
        linewidth=1.5, label="B - nearest-day imputation")
if imp_flag.any():
    ax.scatter(test.index[m & imp_flag], y[m & imp_flag], s=14, color=COL_B, zorder=5,
               label="lag_24h imputed")
style(ax)
ax.set_title("The 10 June telemetry outage and its effect on the 11 June peak forecast",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=4, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase6_fig1_outage.png", dpi=150, facecolor=SURFACE); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
names = list(imp_quality.keys())
xs = np.arange(len(names))
ax.bar(xs - .2, [imp_quality[n]["MAE"] for n in names], width=.38, color=COL_A, label="MAE (MW)")
ax.bar(xs + .2, [imp_quality[n]["RMSE"] for n in names], width=.38, color=COL_C, label="RMSE (MW)")
for i, n in enumerate(names):
    ax.text(i - .2, imp_quality[n]["MAE"] + 12, f"{imp_quality[n]['MAE']:.0f}", ha="center",
            fontsize=9.5, color=INK)
    ax.text(i + .2, imp_quality[n]["RMSE"] + 12, f"{imp_quality[n]['RMSE']:.0f}", ha="center",
            fontsize=9.5, color=INK)
style(ax)
ax.set_xticks(xs); ax.set_xticklabels([n.split(". ")[1] for n in names], fontsize=10)
ax.set_title("Imputation error on simulated evening outages (lower is better)",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Error (MW)", fontsize=9.5, color="#5c655e")
ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper right")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase6_fig2_imputation.png", dpi=150, facecolor=SURFACE); plt.close(fig)

out = test[["load_MW", "temp_C"]].copy()
out["lag24_imputed"] = imp_flag
for k, v in preds_store.items():
    out[k] = v
out["P10"], out["P90"] = p10, p90
out.to_csv(OUT_DIR / "phase6_test_predictions.csv")
(OUT_DIR / "phase6_results.json").write_text(json.dumps(
    {"imputation_quality_simulated": imp_quality, "demand": results,
     "n_imputed_test_blocks": int(imp_flag.sum()), "n_test": int(len(test))}, indent=2, default=str))
(OUT_DIR / "phase6_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say("[SAVED] phase6_report.txt, phase6_results.json, phase6_test_predictions.csv, 2 figures")
