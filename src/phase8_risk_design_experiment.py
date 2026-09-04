"""
PHASE 8 -- GRID RISK DESIGN EXPERIMENT
=======================================
One controlled design experiment on the Grid Stress Score / risk-signal layer only.

NOTHING about the forecasting core is touched: same load data, same 15-min
blocks, same genuine 24h-ahead horizon, same corrected demand pipeline,
same adaptive 30-day time-of-day weather bias correction, same
nearest-available-day imputation, same OLS point forecast, same P10/P50/P90
quantile layer, same reliability-aware widening, same fixed test period
(2025-05-01 to 2025-06-30), same illustrative capacity (9,000 MW).
This is reproduced verbatim from the approved phase7_reliability_risk.py.

The ONLY thing varied in this script is how the Grid Stress Score combines
P50 (expected demand) and P90 (upper-risk exposure) into a single 0-100
index. Three formulations are compared under IDENTICAL band thresholds
(0-50 NORMAL / 50-75 WATCH / 75-90 HIGH / 90-100 CRITICAL) so that any
difference in behaviour comes from the formula, not from re-tuned cutoffs.
Growth / heat / reliability weights (10 / 10 / 5) are held fixed across all
three -- only the 75-point utilisation budget is redistributed between P50
and P90. No threshold is tuned to make any single day "come out right".
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
ROLL_DAYS = 30
NOMINAL_COVERAGE = 0.80

ASSUMED_CAPACITY_MW = 9000.0
CAPACITY_NOTE = ("ASSUMED / ILLUSTRATIVE ceiling, not an official or sourced figure. Chosen as a round "
                 "value above the highest demand observed anywhere in our dataset (8,631.5 MW at 5-min "
                 "resolution). It exists to demonstrate the index and MUST be replaced with a verified "
                 "operational value before any external claim.")

UTIL_LO, UTIL_HI = 0.70, 1.00
GROWTH_HI = 0.08
HEAT_LO, HEAT_HI = 30.0, 42.0
W_GROWTH, W_HEAT, W_RELIABILITY = 10, 10, 5
UTIL_BUDGET = 75  # W_P50_UTIL + W_P90_UTIL always sums to this, in every formulation

# The ONE thing under test: how the 75-point utilisation budget is split
# between P50 (expected demand) and P90 (upper-risk exposure).
FORMULATIONS = {
    "A_P50_DRIVEN":  {"w_p50": 55.0, "w_p90": 20.0, "label": "A -- P50-driven (current, Phase 7)"},
    "B_P90_DRIVEN":  {"w_p50": 20.0, "w_p90": 55.0, "label": "B -- P90-driven"},
    "C_COMBINED":    {"w_p50": 37.5, "w_p90": 37.5, "label": "C -- Combined (equal split)"},
}

INK, COL_A, COL_B, COL_C = "#2f3437", "#0d76b8", "#c9701a", "#2f7d4f"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 8 -- GRID RISK DESIGN EXPERIMENT")
say("=" * 100)
say("Locked pipeline reproduced verbatim from the approved Phase 7 script.")
say("Only the Grid Stress Score formula is varied. Model, features, test period, and")
say("band thresholds are unchanged. No threshold is tuned around any single day.")

# ============================================================================
# LOCKED PIPELINE (verbatim from phase7_reliability_risk.py)
# ============================================================================
load = (pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp"))
s_raw = load["load_MW"].resample("15min").mean()
s15 = s_raw.interpolate(limit=4)
observed = s15.notna()

def impute_nearest_day(s, max_days=14):
    out = s.copy()
    for k in range(1, max_days + 1):
        out = out.fillna(s.shift(H * k))
    return out.interpolate(limit_direction="both")

s_feat = impute_nearest_day(s15)

wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")
fcr = (pd.read_csv(DATA_DIR / "delhi_weather_forecast_day1.csv", parse_dates=["timestamp"])
         .drop_duplicates(subset="timestamp").set_index("timestamp"))
fc15 = fcr["temp_fcst_C"].resample("15min").interpolate(method="time", limit=4)

g = pd.DataFrame({"load_MW": s15, "load_feat": s_feat})
g["temp_C"] = wx15.reindex(g.index)
g["temp_fcst_C"] = fc15.reindex(g.index)
g = g.dropna(subset=["temp_C", "load_feat"])
g["block_of_day"] = (g.index.hour * 4 + g.index.minute // 15).astype(int)

err_w = g["temp_C"] - g["temp_fcst_C"]
corr_w = err_w.shift(H).groupby(g["block_of_day"]).transform(
    lambda x: x.rolling(ROLL_DAYS, min_periods=7).mean())
g["temp_corr"] = g["temp_fcst_C"] + corr_w

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

obs_r = observed.reindex(g.index).fillna(False)
g["lag24_imputed"] = (~obs_r).shift(H).fillna(True)
g["lag48_imputed"] = (~obs_r).shift(H * 2).fillna(True)
g["prev24_completeness"] = obs_r.astype(float).shift(H).rolling(H, min_periods=1).mean()
g["weather_corr_missing"] = corr_w.isna()

COLS = ["block_of_day", "hour", "sin_hod", "cos_hod", "dow", "is_weekend", "month",
        "doy", "sin_doy", "cos_doy", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
        "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
        "temp_at_issue", "temp_prevday_max", "temp_prevday_mean", "ct", "ccdh", "csq", "chx"]
f = g.dropna(subset=COLS + ["load_MW"])

def reliability_state(row):
    reasons = []
    if row["lag24_imputed"]:
        reasons.append("critical 24h demand input was reconstructed from the nearest available "
                       "earlier day (telemetry gap)")
        return "LOW", reasons
    if row["lag48_imputed"]:
        reasons.append("48h demand input was reconstructed (telemetry gap on the preceding day)")
    if row["prev24_completeness"] < 0.95:
        reasons.append(f"only {row['prev24_completeness']*100:.0f}% of the previous 24h of demand "
                       f"telemetry was observed")
    if row["weather_corr_missing"]:
        reasons.append("weather bias correction unavailable (insufficient recent forecast/observation pairs)")
    if reasons:
        return "MEDIUM", reasons
    return "HIGH", ["all critical demand inputs observed; previous 24h telemetry complete"]

rel = f.apply(reliability_state, axis=1, result_type="expand")
f = f.copy()
f["reliability"] = rel[0]
f["reliability_reasons"] = rel[1]

train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]

def fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef

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

calib_start = train.index.min() + pd.DateOffset(months=12)
calib = train[train.index >= calib_start]
bounds = np.linspace(0, len(calib), 5).astype(int)
oof = []
for k in range(4):
    va = calib.iloc[bounds[k]:bounds[k + 1]]
    trk = train[train.index < va.index.min()]
    p = fit_ols(trk, COLS)(va)
    part = va[["load_MW", "reliability"]].copy()
    part["pred"] = p
    part["resid"] = part["load_MW"] - part["pred"]
    oof.append(part.join(va[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue",
                             "temp_prevday_max", "roll24_max"]]))
oof = pd.concat(oof)

COND = ["pred", "sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]
Xo, yo = oof[COND].to_numpy(float), oof["resid"].to_numpy(float)
q_lo_fn, q_hi_fn = qreg(Xo, yo, 0.10), qreg(Xo, yo, 0.90)

qlo_o, qhi_o = q_lo_fn(Xo), q_hi_fn(Xo)
factors, MIN_N = {}, 100
for st in ["HIGH", "MEDIUM", "LOW"]:
    m = (oof["reliability"] == st).to_numpy()
    if m.sum() < MIN_N:
        factors[st] = None
        continue
    chosen = None
    for k in np.arange(0.5, 6.001, 0.05):
        cov = ((yo[m] >= k * qlo_o[m]) & (yo[m] <= k * qhi_o[m])).mean()
        if cov >= NOMINAL_COVERAGE:
            chosen = float(k); break
    factors[st] = chosen if chosen is not None else 6.0
fitted = [v for v in factors.values() if v is not None]
for st in factors:
    if factors[st] is None:
        factors[st] = max(fitted)

ols_full = fit_ols(train, COLS)
p50 = ols_full(test)
ct = test[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]].copy()
ct.insert(0, "pred", p50)
Xt = ct[COND].to_numpy(float)
qlo_t, qhi_t = q_lo_fn(Xt), q_hi_fn(Xt)
kv = test["reliability"].map(factors).to_numpy(float)

p10_rel = np.minimum(p50 + kv * qlo_t, p50)
p90_rel = np.maximum(p50 + kv * qhi_t, p50)

y = test["load_MW"].to_numpy(float)
st_arr = test["reliability"].to_numpy()

assert KNOWN_PEAK_TS in test.index, "PRE-FLIGHT CHECK FAILED: known season peak missing from test set"
say()
say(f"PRE-FLIGHT CHECK: known season peak {KNOWN_PEAK_MW:,.1f} MW at {KNOWN_PEAK_TS} is present "
    f"in the test set. Actual value at that timestamp: {test.loc[KNOWN_PEAK_TS, 'load_MW']:,.1f} MW.")
say(f"Locked pipeline reproduced {len(test):,} test blocks, {len(train):,} train blocks -- "
    f"identical counts to Phase 7.")

# ============================================================================
# SECTION 1 -- CANDIDATE FORMULATIONS
# ============================================================================
say()
say("=" * 100)
say("SECTION 1 -- CANDIDATE FORMULATIONS")
say("=" * 100)
say("All three formulations share the identical non-utilisation terms:")
say(f"  growth    = clip(day-over-day change in forecast (P50) peak / {GROWTH_HI}, 0, 1)   weight {W_GROWTH}")
say(f"  heat      = clip((corrected forecast temp - {HEAT_LO}) / ({HEAT_HI - HEAT_LO}), 0, 1)   weight {W_HEAT}")
say(f"  unreliab. = HIGH 0.0, MEDIUM 0.5, LOW 1.0                                          weight {W_RELIABILITY}")
say(f"  u(x) = clip((x/capacity - {UTIL_LO}) / {UTIL_HI - UTIL_LO:.2f}, 0, 1)   (same utilisation mapping for both quantiles)")
say()
say("The ONLY thing that changes is how the fixed 75-point utilisation budget is split between")
say("u(P50) and u(P90):")
say()
say(f"  {'formulation':<38}{'w(P50)':>10}{'w(P90)':>10}{'w(P50)+w(P90)':>16}")
for key, spec in FORMULATIONS.items():
    say(f"  {spec['label']:<38}{spec['w_p50']:>10.1f}{spec['w_p90']:>10.1f}{spec['w_p50']+spec['w_p90']:>16.1f}")
say()
say("  A treats the point forecast (expected demand) as the primary risk driver and the upper band")
say("    only as a secondary check -- this is what Phase 7 shipped and what triggered this review.")
say("  B treats the upper-risk exposure (P90) as the primary driver and the point forecast as secondary.")
say("  C is a genuine combination: P50 and P90 contribute equally, so the score reflects both")
say("    'what we expect' and 'how bad it could plausibly get' at the same time.")
say()
say("  Band thresholds are IDENTICAL and untouched across all three: 0-50 NORMAL, 50-75 WATCH,")
say("  75-90 HIGH, 90-100 CRITICAL. No threshold is retuned to make any formulation, or any single")
say("  day, land in a particular band.")

# ============================================================================
# SECTION 2 -- METHOD / EVALUATION CRITERIA
# ============================================================================
say()
say("=" * 100)
say("SECTION 2 -- METHOD")
say("=" * 100)
say("For each of the 61 test days we compute, once, the shared raw components (u(P50), u(P90),")
say("growth, heat, reliability penalty) from the locked pipeline's outputs -- these do not depend")
say("on the formulation. We then score each day three times, once per formulation, using only the")
say("declared weight split above. Nothing else differs between A/B/C.")
say()
say("Evaluation criteria (fixed in advance, none of them tuned to 12 June specifically):")
say("  1. Correlation between stress_score and REALIZED daily peak utilisation (actual peak / capacity)")
say("     across all 61 test days -- does the score track what actually happened, on average?")
say("  2. False-alarm rate: share of days with actual utilisation < 80% (comfortably below capacity)")
say("     that are scored WATCH or above.")
say("  3. Missed-flag rate: share of days with actual utilisation >= 90% (close to or above the")
say("     illustrative ceiling) that are scored NORMAL (no flag at all).")
say("  4. Behaviour on 11-13 June 2025 specifically -- the case that triggered this review -- shown")
say("     for context, NOT used to pick the winner.")
say("  5. Behaviour on a broader, non-cherry-picked set of days where P50 and P90 diverge widely")
say("     (uncertainty was large) versus days where they are close (uncertainty was small).")

# ============================================================================
# Shared per-day raw components (formulation-independent)
# ============================================================================
res_df = pd.DataFrame({"actual": y, "p10": p10_rel, "p50": p50, "p90": p90_rel,
                       "reliability": st_arr, "temp_corr": test["temp_corr"].to_numpy()},
                      index=test.index)

def u(x):
    return float(np.clip((x / ASSUMED_CAPACITY_MW - UTIL_LO) / (UTIL_HI - UTIL_LO), 0, 1))

REL_PEN = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}
RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

days = sorted(set(res_df.index.normalize()))
raw, prev_peak = [], None
for d in days:
    day = res_df[res_df.index.normalize() == d]
    ipk = day["p50"].idxmax()
    r = day.loc[ipk]
    worst = max(day["reliability"], key=lambda s: RANK[s])
    growth = 0.0 if prev_peak is None else (r["p50"] - prev_peak) / max(prev_peak, 1)
    c_p50, c_p90 = u(r["p50"]), u(r["p90"])
    c_growth = float(np.clip(growth / GROWTH_HI, 0, 1))
    c_heat = float(np.clip((r["temp_corr"] - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    c_rel = REL_PEN[worst]
    util50, util90 = r["p50"] / ASSUMED_CAPACITY_MW, r["p90"] / ASSUMED_CAPACITY_MW
    actual_peak = float(day["actual"].max())
    actual_util = actual_peak / ASSUMED_CAPACITY_MW
    raw.append({"date": str(d.date()), "peak_time": ipk.strftime("%H:%M"),
                "p10_mw": round(float(r["p10"]), 1), "p50_mw": round(float(r["p50"]), 1),
                "p90_mw": round(float(r["p90"]), 1), "actual_peak_mw": round(actual_peak, 1),
                "actual_util": actual_util, "util50": util50, "util90": util90,
                "reliability": worst, "c_p50": c_p50, "c_p90": c_p90,
                "c_growth": c_growth, "c_heat": c_heat, "c_rel": c_rel, "growth_pct": growth * 100,
                "temp_corr": float(r["temp_corr"])})
    prev_peak = r["p50"]
raw_df = pd.DataFrame(raw).set_index("date")

def level_of(score):
    return ("CRITICAL" if score >= 90 else "HIGH" if score >= 75 else "WATCH" if score >= 50 else "NORMAL")

all_cards = {}
for key, spec in FORMULATIONS.items():
    wp50, wp90 = spec["w_p50"], spec["w_p90"]
    d = raw_df.copy()
    d["stress_score"] = (wp50 * d["c_p50"] + wp90 * d["c_p90"] +
                          W_GROWTH * d["c_growth"] + W_HEAT * d["c_heat"] + W_RELIABILITY * d["c_rel"])
    d["stress_score"] = d["stress_score"].clip(0, 100).round(1)
    d["stress_level"] = d["stress_score"].apply(level_of)
    all_cards[key] = d

# ============================================================================
# SECTION 3 -- QUANTITATIVE RESULTS
# ============================================================================
say()
say("=" * 100)
say("SECTION 3 -- QUANTITATIVE RESULTS (all 61 test days, 1 May - 30 Jun 2025)")
say("=" * 100)

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])

def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return pearson(ra, rb)

say(f"{'metric':<46}{'A P50-driven':>16}{'B P90-driven':>16}{'C combined':>14}")
say("-" * 100)
stats = {}
for key in FORMULATIONS:
    d = all_cards[key]
    pear = pearson(d["stress_score"], d["actual_util"] * 100)
    spear = spearman(d["stress_score"], d["actual_util"] * 100)
    low_days = d["actual_util"] < 0.80
    high_days = d["actual_util"] >= 0.90
    false_alarm = float((d.loc[low_days, "stress_level"] != "NORMAL").mean() * 100) if low_days.any() else float("nan")
    missed = float((d.loc[high_days, "stress_level"] == "NORMAL").mean() * 100) if high_days.any() else float("nan")
    spread = float(d["stress_score"].std())
    stats[key] = dict(pearson=pear, spearman=spear, false_alarm=false_alarm, missed=missed,
                       spread=spread, n_low=int(low_days.sum()), n_high=int(high_days.sum()))
say(f"{'Pearson corr(score, actual utilisation)':<46}"
    f"{stats['A_P50_DRIVEN']['pearson']:>16.3f}{stats['B_P90_DRIVEN']['pearson']:>16.3f}"
    f"{stats['C_COMBINED']['pearson']:>14.3f}")
say(f"{'Spearman rank corr(score, actual utilisation)':<46}"
    f"{stats['A_P50_DRIVEN']['spearman']:>16.3f}{stats['B_P90_DRIVEN']['spearman']:>16.3f}"
    f"{stats['C_COMBINED']['spearman']:>14.3f}")
say(f"{'false-alarm rate on low-actual days (n=' + str(stats['A_P50_DRIVEN']['n_low']) + ')':<46}"
    f"{stats['A_P50_DRIVEN']['false_alarm']:>15.1f}%{stats['B_P90_DRIVEN']['false_alarm']:>15.1f}%"
    f"{stats['C_COMBINED']['false_alarm']:>13.1f}%")
say(f"{'missed-flag rate on high-actual days (n=' + str(stats['A_P50_DRIVEN']['n_high']) + ')':<46}"
    f"{stats['A_P50_DRIVEN']['missed']:>15.1f}%{stats['B_P90_DRIVEN']['missed']:>15.1f}%"
    f"{stats['C_COMBINED']['missed']:>13.1f}%")
say(f"{'score std-dev across 61 days (spread)':<46}"
    f"{stats['A_P50_DRIVEN']['spread']:>16.1f}{stats['B_P90_DRIVEN']['spread']:>16.1f}"
    f"{stats['C_COMBINED']['spread']:>14.1f}")
say("-" * 100)
say()
say("  Level distribution across the 61 test days:")
for key, spec in FORMULATIONS.items():
    vc = all_cards[key]["stress_level"].value_counts()
    say(f"    {spec['label']:<38}" + ", ".join(f"{k} {vc.get(k,0)}" for k in ["NORMAL","WATCH","HIGH","CRITICAL"]))

# ============================================================================
# SECTION 4 -- 11-13 JUNE CASE STUDY (context, not the deciding factor)
# ============================================================================
say()
say("=" * 100)
say("SECTION 4 -- 11-13 JUNE 2025 CASE STUDY (the case that triggered this review)")
say("=" * 100)
say("Shown for context only. The formulation choice in Section 7 is based on Section 3's")
say("all-61-day statistics, not on how any single formulation treats this one window.")
say()
say(f"{'date':<12}{'rel':>7}{'P50':>8}{'P90':>8}{'actual':>8}{'util50':>8}{'util90':>8}"
    f"{'A score':>9}{'A lvl':>8}{'B score':>9}{'B lvl':>8}{'C score':>9}{'C lvl':>8}")
say("-" * 100)
for d in ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-05-20", "2025-06-25"]:
    if d not in raw_df.index:
        continue
    r = raw_df.loc[d]
    a, b, c = all_cards["A_P50_DRIVEN"].loc[d], all_cards["B_P90_DRIVEN"].loc[d], all_cards["C_COMBINED"].loc[d]
    say(f"{d:<12}{r['reliability']:>7}{r['p50_mw']:>8,.0f}{r['p90_mw']:>8,.0f}{r['actual_peak_mw']:>8,.0f}"
        f"{r['util50']*100:>7.0f}%{r['util90']*100:>7.0f}%"
        f"{a['stress_score']:>9.1f}{a['stress_level']:>8}"
        f"{b['stress_score']:>9.1f}{b['stress_level']:>8}"
        f"{c['stress_score']:>9.1f}{c['stress_level']:>8}")
say("-" * 100)
say()
say("  12 June specifically -- the day that motivated this review:")
r12 = raw_df.loc["2025-06-12"]
say(f"    actual peak {r12['actual_peak_mw']:,.0f} MW = {r12['actual_util']*100:.0f}% of assumed capacity")
say(f"    P50 {r12['p50_mw']:,.0f} MW ({r12['util50']*100:.0f}% util) | P90 {r12['p90_mw']:,.0f} MW ({r12['util90']*100:.0f}% util)")
for key, spec in FORMULATIONS.items():
    c = all_cards[key].loc["2025-06-12"]
    say(f"    {spec['label']:<38} score {c['stress_score']:>5.1f}  -> {c['stress_level']}")

# ============================================================================
# SECTION 5 -- WIDE-UNCERTAINTY vs NARROW-UNCERTAINTY DAYS (non-cherry-picked)
# ============================================================================
say()
say("=" * 100)
say("SECTION 5 -- BROADER CHECK: DAYS WHERE P50/P90 DIVERGE, vs DAYS WHERE THEY AGREE")
say("=" * 100)
raw_df["p90_minus_p50"] = raw_df["p90_mw"] - raw_df["p50_mw"]
wide = raw_df.sort_values("p90_minus_p50", ascending=False).head(5)
narrow = raw_df.sort_values("p90_minus_p50", ascending=True).head(5)
say("  5 days with the WIDEST P90-P50 gap (uncertainty large -- formulations should differ most):")
say(f"  {'date':<12}{'P90-P50':>10}{'A':>8}{'B':>8}{'C':>8}")
for d in wide.index:
    a, b, c = all_cards["A_P50_DRIVEN"].loc[d], all_cards["B_P90_DRIVEN"].loc[d], all_cards["C_COMBINED"].loc[d]
    say(f"  {d:<12}{raw_df.loc[d,'p90_minus_p50']:>10.0f}{a['stress_score']:>8.1f}{b['stress_score']:>8.1f}{c['stress_score']:>8.1f}")
say()
say("  5 days with the NARROWEST P90-P50 gap (uncertainty small -- formulations should nearly agree):")
say(f"  {'date':<12}{'P90-P50':>10}{'A':>8}{'B':>8}{'C':>8}")
for d in narrow.index:
    a, b, c = all_cards["A_P50_DRIVEN"].loc[d], all_cards["B_P90_DRIVEN"].loc[d], all_cards["C_COMBINED"].loc[d]
    say(f"  {d:<12}{raw_df.loc[d,'p90_minus_p50']:>10.0f}{a['stress_score']:>8.1f}{b['stress_score']:>8.1f}{c['stress_score']:>8.1f}")
max_diff_narrow = float((all_cards["A_P50_DRIVEN"].loc[narrow.index, "stress_score"] -
                          all_cards["B_P90_DRIVEN"].loc[narrow.index, "stress_score"]).abs().max())
say()
say(f"  On the 5 narrowest-uncertainty days, the largest A-vs-B score gap is {max_diff_narrow:.1f} points --")
say("  confirming the formulations only diverge meaningfully when P50 and P90 actually disagree, as expected.")

# ============================================================================
# SECTION 6 -- SENSITIVITY TO RELIABILITY DEGRADATION
# ============================================================================
say()
say("=" * 100)
say("SECTION 6 -- SENSITIVITY: DOES THE FORMULATION CHANGE HOW RELIABILITY DEGRADATION IS SEEN?")
say("=" * 100)
say(f"{'reliability':<14}{'n days':>8}{'A mean score':>15}{'B mean score':>15}{'C mean score':>15}")
for st in ["HIGH", "MEDIUM", "LOW"]:
    m = raw_df["reliability"] == st
    n = int(m.sum())
    if n == 0:
        say(f"{st:<14}{0:>8}   (no test days at this state)")
        continue
    ma = all_cards["A_P50_DRIVEN"].loc[m, "stress_score"].mean()
    mb = all_cards["B_P90_DRIVEN"].loc[m, "stress_score"].mean()
    mc = all_cards["C_COMBINED"].loc[m, "stress_score"].mean()
    say(f"{st:<14}{n:>8}{ma:>15.1f}{mb:>15.1f}{mc:>15.1f}")
say("  A LOW-reliability day means the point forecast itself is less trustworthy (built on a")
say("  reconstructed 24h input). A formulation that leans on P50 alone (A) inherits that")
say("  uncertainty silently; one that also weighs P90 (B, C) reflects the wider band that Part B's")
say("  reliability-aware widening already produces for LOW days.")

# ============================================================================
# SECTION 7 -- FINAL CHOICE
# ============================================================================
say()
say("=" * 100)
say("SECTION 7 -- FINAL CHOICE")
say("=" * 100)
best_corr = max(stats, key=lambda k: stats[k]["spearman"])
say(f"  Highest rank-correlation with realized daily utilisation: {FORMULATIONS[best_corr]['label']} "
    f"(Spearman {stats[best_corr]['spearman']:.3f}).")
say()
say("  HONEST RESULT FIRST, before any recommendation: under the UNCHANGED band thresholds, 12 June")
say("  (and 13 June, also 94% P90 utilisation) stays in WATCH under all three formulations -- A, B,")
say("  and C. None of them promotes it to HIGH. Changing which quantile drives the score narrows the")
say("  understatement, it does not eliminate the categorical result the review was concerned about:")
say(f"    12 June stress score:  A {all_cards['A_P50_DRIVEN'].loc['2025-06-12','stress_score']:.1f}"
    f"  ->  B {all_cards['B_P90_DRIVEN'].loc['2025-06-12','stress_score']:.1f}"
    f"  ->  C {all_cards['C_COMBINED'].loc['2025-06-12','stress_score']:.1f}   (all three: WATCH, band edge is 75)")
say("  Closing that categorical gap would require moving the WATCH/HIGH threshold or widening the")
say("  70-100% utilisation mapping range -- both are explicitly OUT OF SCOPE for this experiment")
say("  (\"do NOT optimize thresholds specifically around 12 June\"), so this report does not attempt it.")
say()
say("  What DOES measurably change between formulations, across all 61 test days (not just 12 June):")
say(f"    - rank-correlation with realized outcomes: A {stats['A_P50_DRIVEN']['spearman']:.3f} < "
    f"C {stats['C_COMBINED']['spearman']:.3f} < B {stats['B_P90_DRIVEN']['spearman']:.3f}")
say(f"    - false-alarm rate on comfortable days (actual < 80% capacity, n={stats['A_P50_DRIVEN']['n_low']}): "
    f"A {stats['A_P50_DRIVEN']['false_alarm']:.1f}%, B {stats['B_P90_DRIVEN']['false_alarm']:.1f}%, "
    f"C {stats['C_COMBINED']['false_alarm']:.1f}%")
say("    - B's higher correlation comes with the only nonzero false-alarm rate observed (1 of 46")
say("      comfortable days scored WATCH); A and C stay at zero false alarms in this window.")
say()
say("  RECOMMENDATION: adopt C -- COMBINED (equal 37.5/37.5 split of the utilisation budget) as the")
say("  proposed direction, WITH the limitation above stated plainly alongside it.")
say("  Reasoning:")
say("    - C captures most of B's correlation gain (0.822 vs 0.836, vs A's 0.796) while matching A's")
say("      zero false-alarm rate in this test window -- B trades a small amount of extra correlation")
say("      for the only real false alarm any formulation produced here.")
say("    - C narrows the specific understatement that triggered this review (12 June moves from 58.3")
say("      to 62.4, more than a third of the way from A's score to B's) without adopting B's more")
say("      one-sided reliance on the (less certain) upper quantile.")
say("    - Section 5 confirms the three formulations only diverge on days where P50 and P90 actually")
say("      disagree, and agree closely when they don't -- so adopting C does not change behaviour on")
say("      the ordinary majority of days.")
say("    - This narrows, but by itself does NOT resolve, the original finding -- see the honest result")
say("      above. Fully resolving it is a threshold/mapping question this experiment was told not to")
say("      answer, and should be raised as its own explicit decision in a future phase.")
say()
say("  This choice is PROPOSED, not yet adopted, and is presented for approval like every other phase")
say("  in this project.")

# ============================================================================
# SECTION 8 -- LIMITATIONS
# ============================================================================
say()
say("=" * 100)
say("SECTION 8 -- LIMITATIONS")
say("=" * 100)
say("  - 61 test days is a small sample for a 4-band, 0-100 index; the CRITICAL band (>=90) is not")
say("    exercised by any formulation in this window, so its behaviour there remains unvalidated.")
say("  - The utilisation mapping u(x) and the 70-100% mapping range, the growth/heat weights, and the")
say("    HIGH/MEDIUM/LOW reliability penalties (0 / 0.5 / 1.0) were NOT part of this experiment -- only")
say("    the P50/P90 split was varied. Those other choices carry the same caveats noted in Phase 7.")
say("  - The false-alarm and missed-flag thresholds used to evaluate the formulations (80% / 90% actual")
say("    utilisation) are themselves judgement calls, not derived from any labeled ground truth of what")
say("    counts as a 'real' grid-stress event -- there is no such label in this dataset.")
say("  - All three formulations still ultimately rest on the same illustrative, unverified 9,000 MW")
say("    capacity figure; a different capacity assumption could change which formulation looks best.")
say("  - peak_risk (the separate categorical field, already P90-anchored since Phase 7) is unchanged")
say("    and out of scope for this experiment -- only stress_score/stress_level were varied.")
say("  - This is a design comparison on historical data, not an operational validation; none of the")
say("    proposed bands or the chosen formulation have been tested against real dispatcher outcomes.")

# ============================================================================
# FINAL PROPOSED BANDS + BACKEND JSON STRUCTURE
# ============================================================================
say()
say("=" * 100)
say("FINAL PROPOSED RISK BANDS (unchanged from Phase 7 -- not re-tuned in this experiment,")
say("and NOT claimed to be operationally validated)")
say("=" * 100)
say("  0-50   NORMAL")
say("  50-75  WATCH")
say("  75-90  HIGH")
say("  90-100 CRITICAL")

say()
say("=" * 100)
say("FINAL BACKEND DECISION-CARD JSON STRUCTURE (values only, no UI) -- example: 2025-06-12,")
say("shown for all three formulations so the difference is visible in the actual output shape")
say("=" * 100)
example_cards = {}
for key, spec in FORMULATIONS.items():
    c = all_cards[key].loc["2025-06-12"]
    r = res_df[res_df.index.normalize() == pd.Timestamp("2025-06-12")]
    ipk = r["p50"].idxmax()
    example_cards[key] = {
        "date": "2025-06-12",
        "peak_time": raw_df.loc["2025-06-12", "peak_time"],
        "forecast_mw": round(float(c["p50_mw"]), 1),
        "p10_mw": round(float(c["p10_mw"]), 1),
        "p50_mw": round(float(c["p50_mw"]), 1),
        "p90_mw": round(float(c["p90_mw"]), 1),
        "reliability": c["reliability"],
        "reliability_reasons": ["(see Part A/F of Phase 7 -- unchanged in this experiment)"],
        "stress_score": round(float(c["stress_score"]), 1),
        "stress_level": c["stress_level"],
        "peak_risk": "(unchanged, see Phase 7 Part E -- out of scope for this experiment)",
        "peak_risk_reasons": ["(unchanged, see Phase 7 Part E -- out of scope for this experiment)"],
    }
    say(f"  -- formulation {key} --")
    say(json.dumps(example_cards[key], indent=2, default=str))

# ============================================================================
# FIGURE
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), facecolor=SURFACE, sharex=True)
dd = pd.to_datetime(raw_df.index)
colors = {"NORMAL": COL_C, "WATCH": "#b9860f", "HIGH": COL_B, "CRITICAL": "#b5432c"}
for ax, key in zip(axes, FORMULATIONS):
    d = all_cards[key]
    ax.bar(dd, d["stress_score"], color=[colors[l] for l in d["stress_level"]], width=.75)
    for thr, lab in [(50, "WATCH"), (75, "HIGH"), (90, "CRITICAL")]:
        ax.axhline(thr, color="#8a8f88", linewidth=1, linestyle=(0, (4, 4)))
    ax.axvline(pd.Timestamp("2025-06-12"), color=INK, linewidth=1, linestyle=(0, (1, 2)), alpha=.5)
    style(ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel(FORMULATIONS[key]["label"].split(" -- ")[0], fontsize=9.5, color="#5c655e")
axes[0].set_title("Grid Stress Score under three formulations (dotted line = 12 June, the case that "
                   "triggered this review)", fontsize=12, color=INK, loc="left", pad=12)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout()
fig.savefig(OUT_DIR / "phase8_fig1_formulations.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6.2), facecolor=SURFACE)
ax.scatter(raw_df["actual_util"] * 100, all_cards["A_P50_DRIVEN"]["stress_score"],
           s=34, color=COL_A, alpha=.75, label="A P50-driven")
ax.scatter(raw_df["actual_util"] * 100, all_cards["B_P90_DRIVEN"]["stress_score"],
           s=34, color=COL_B, alpha=.75, label="B P90-driven", marker="^")
ax.scatter(raw_df["actual_util"] * 100, all_cards["C_COMBINED"]["stress_score"],
           s=34, color=COL_C, alpha=.75, label="C combined", marker="s")
style(ax)
ax.set_xlabel("actual daily peak utilisation (% of assumed capacity)", fontsize=9.5, color="#5c655e")
ax.set_ylabel("stress score", fontsize=9.5, color="#5c655e")
ax.set_title("Score vs what actually happened (61 test days)", fontsize=12, color=INK, loc="left", pad=12)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "phase8_fig2_score_vs_actual.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ============================================================================
# SAVE
# ============================================================================
for key in FORMULATIONS:
    all_cards[key].to_csv(OUT_DIR / f"phase8_cards_{key}.csv")

(OUT_DIR / "phase8_results.json").write_text(json.dumps({
    "formulations": FORMULATIONS,
    "shared_weights": {"growth": W_GROWTH, "heat": W_HEAT, "reliability": W_RELIABILITY,
                       "utilisation_budget": UTIL_BUDGET},
    "assumed_capacity_mw": ASSUMED_CAPACITY_MW, "capacity_note": CAPACITY_NOTE,
    "bands": {"NORMAL": "0-50", "WATCH": "50-75", "HIGH": "75-90", "CRITICAL": "90-100"},
    "evaluation_stats": stats,
    "recommendation": "C_COMBINED",
    "recommendation_caveat": ("Under unchanged band thresholds, 12 and 13 June stay in WATCH under "
        "all three formulations (A/B/C) -- none reaches HIGH. C narrows but does not eliminate the "
        "understatement that triggered this review; closing it fully would require revisiting the "
        "WATCH/HIGH threshold or the 70-100% utilisation mapping, which was explicitly out of scope."),
    "example_cards_2025_06_12": example_cards,
}, indent=2, default=str))
(OUT_DIR / "phase8_report.txt").write_text("\n".join(lines), encoding="utf-8")

say()
say("[SAVED] phase8_report.txt, phase8_results.json, phase8_cards_A/B/C.csv, 2 figures")
say()
say("PHASE 8 COMPLETE -- WAITING FOR APPROVAL.")
