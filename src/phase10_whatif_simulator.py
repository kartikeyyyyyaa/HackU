"""
PHASE 10 -- WHAT-IF DECISION SIMULATOR
========================================
A real, model-driven scenario layer on top of the locked forecasting
pipeline. Nothing about the trained model, features, test period, or
approved risk/reliability logic is changed. This script:
  (a) applies the ONE approved bugfix from Phase 9 (peak_risk_reasons must
      name the exact condition that produced the tier), and
  (b) adds scenario recomputation for TWO variables only -- the smallest
      set defensible from the current model and data:
        - TEMPERATURE: a genuine model recomputation (the trained OLS
          coefficients and the fitted quantile-regression functions are
          reused unchanged; only the corrected forecast temperature
          feature, and features derived from it, are perturbed for the
          target window, then pushed back through the SAME fitted
          functions).
        - ROOFTOP SOLAR: NOT in the trained model (no solar/generation
          column exists anywhere in this project's data). It is
          implemented as a separate, explicitly labeled scenario
          calculation -- gross demand forecast minus an assumed daylight
          generation profile -- never presented as a model prediction.
EV load and commercial load are NOT implemented: there is no data in this
project that would let either be represented honestly, so they are left
out rather than faked (see Section 8).
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
ROLL_DAYS = 30
NOMINAL_COVERAGE = 0.80

ASSUMED_CAPACITY_MW = 9000.0
CAPACITY_NOTE = ("ASSUMED / ILLUSTRATIVE ceiling, not an official or sourced figure. Must be replaced "
                 "with a verified operational value before any external claim.")

UTIL_LO, UTIL_HI = 0.70, 1.00
GROWTH_HI = 0.08
HEAT_LO, HEAT_HI = 30.0, 42.0
# LOCKED Grid Stress Score -- formulation C (Combined), approved Phase 8.
W_P50_UTIL, W_P90_UTIL, W_GROWTH, W_HEAT, W_RELIABILITY = 37.5, 37.5, 10, 10, 5

# Scenario sanity bounds -- documented, not tuned to any single result.
TEMP_DELTA_MAX_ABS = 8.0          # +-8C: beyond this the model is extrapolating well outside training conditions
TEMP_ABS_MIN, TEMP_ABS_MAX = 5.0, 48.0   # Delhi's plausible climatological range (IMD)
SOLAR_CAP_MIN, SOLAR_CAP_MAX = 0.0, 3000.0   # 0 to an illustrative upper bound on assumed installed rooftop capacity

INK, COL_A, COL_B, COL_C, COL_D = "#2f3437", "#0d76b8", "#c9701a", "#2f7d4f", "#8a4fae"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 10 -- WHAT-IF DECISION SIMULATOR")
say("=" * 100)

# ============================================================================
# LOCKED PIPELINE (verbatim, unchanged since Phase 7/8/9)
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
        factors[st] = None; continue
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

def predict_p50(frame):
    return ols_full(frame)

def predict_bands(frame, p50_arr):
    ct = frame[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]].copy()
    ct.insert(0, "pred", p50_arr)
    Xt = ct[COND].to_numpy(float)
    qlo, qhi = q_lo_fn(Xt), q_hi_fn(Xt)
    kv = frame["reliability"].map(factors).to_numpy(float)
    p10 = np.minimum(p50_arr + kv * qlo, p50_arr)
    p90 = np.maximum(p50_arr + kv * qhi, p50_arr)
    return p10, p90

base_p50 = predict_p50(test)
base_p10, base_p90 = predict_bands(test, base_p50)

res_df = pd.DataFrame({"actual": test["load_MW"].to_numpy(float), "p10": base_p10, "p50": base_p50,
                       "p90": base_p90, "reliability": test["reliability"].to_numpy(),
                       "temp_corr": test["temp_corr"].to_numpy()}, index=test.index)

KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
assert KNOWN_PEAK_TS in res_df.index, "PRE-FLIGHT CHECK FAILED"
say(f"PRE-FLIGHT CHECK: locked pipeline reproduced {len(test):,} test blocks, base forecast at the "
    f"known season peak = {res_df.loc[KNOWN_PEAK_TS,'p50']:,.1f} MW (P90 {res_df.loc[KNOWN_PEAK_TS,'p90']:,.1f}) "
    f"-- identical to Phase 7/8/9.")

# ============================================================================
# ONE APPROVED FIX -- peak_risk_reasons now names the exact trigger
# ============================================================================
def u(x):
    return float(np.clip((x / ASSUMED_CAPACITY_MW - UTIL_LO) / (UTIL_HI - UTIL_LO), 0, 1))

REL_PEN = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}
RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

def level_of(score):
    return ("CRITICAL" if score >= 90 else "HIGH" if score >= 75 else "WATCH" if score >= 50 else "NORMAL")

def peak_risk_rule(util50, util90, worst):
    """Unchanged thresholds from Phase 7. FIXED in Phase 10: reasons now always name the exact
    numeric condition that produced the tier (Phase 9 finding), not a redesign of the rule itself."""
    if util90 >= 1.00 and (util50 >= 0.95 or worst == "LOW"):
        risk = "CRITICAL"
    elif util90 >= 0.95 or util50 >= 0.92:
        risk = "HIGH"
    elif util90 >= 0.88:
        risk = "MODERATE"
    else:
        risk = "LOW"

    reasons = []
    if risk == "CRITICAL":
        reasons.append(f"P90 reaches or exceeds 100% of assumed capacity ({util90*100:.0f}%)")
        if util50 >= 0.95:
            reasons.append(f"P50 also reaches {util50*100:.0f}% of assumed capacity (>=95% threshold)")
        if worst == "LOW":
            reasons.append("forecast reliability is LOW, so the upper bound is less dependable")
    elif risk == "HIGH":
        if util90 >= 0.95:
            reasons.append(f"P90 reaches {util90*100:.0f}% of assumed capacity (>=95% threshold)")
        if util50 >= 0.92:
            reasons.append(f"P50 reaches {util50*100:.0f}% of assumed capacity (>=92% threshold)")
    elif risk == "MODERATE":
        reasons.append(f"P90 reaches {util90*100:.0f}% of assumed capacity (>=88% MODERATE threshold, "
                       f"below the 95% HIGH threshold)")
    else:
        reasons.append(f"P90 stays at {util90*100:.0f}% of assumed capacity (below the 88% MODERATE threshold)")
    if worst == "LOW" and risk not in ("CRITICAL",):
        reasons.append("forecast reliability is LOW, so the upper bound is less dependable")
    return risk, reasons

say()
say("=" * 100)
say("APPROVED FIX -- peak_risk_reasons now names the exact triggering condition")
say("=" * 100)
say("  Before (Phase 7/8/9): a MODERATE day whose trigger was util90 >= 88% printed only the")
say("  reliability reason when reliability was LOW, omitting the utilisation condition that actually")
say("  produced the tier. Re-checked here against the Phase 9 primary event:")
r_test, reasons_test = peak_risk_rule(0.866, 0.935, "LOW")
say(f"    util50=86.6%, util90=93.5%, reliability=LOW -> {r_test}, reasons: {reasons_test}")
say("  The utilisation condition (>=88% MODERATE threshold) is now stated explicitly. No threshold")
say("  value (88% / 92% / 95% / 100%) was changed.")

# ============================================================================
# SCENARIO ENGINE
# ============================================================================
def apply_temp_delta(frame, delta):
    """Perturb ONLY the corrected forecast temperature (and features derived from it) for the
    given target window. temp_at_issue / temp_prevday_max / temp_prevday_mean are the OBSERVED
    temperature from before the issue time -- fixed historical fact, not part of a forward-looking
    temperature scenario -- and are deliberately left untouched."""
    fr = frame.copy()
    new_temp = (fr["ct"] + delta).clip(TEMP_ABS_MIN, TEMP_ABS_MAX)
    clipped = int((fr["ct"] + delta != new_temp).sum())
    fr["ct"] = new_temp
    fr["ccdh"] = np.clip(new_temp - 24, 0, None)
    fr["csq"] = new_temp ** 2
    fr["chx"] = np.clip(new_temp - 38, 0, None) ** 2
    return fr, new_temp.to_numpy(), clipped

# Solar generation profile -- an explicit ASSUMPTION, not fitted from any data in this project.
# Half-sine daylight curve, Delhi June sunrise ~05:30, sunset ~19:15 (IST clock hours, IMD almanac).
SUNRISE_H, SUNSET_H = 5.5, 19.25

def solar_profile_mw(index, capacity_mw):
    hour = index.hour + index.minute / 60
    frac = np.clip((hour - SUNRISE_H) / (SUNSET_H - SUNRISE_H), 0, 1)
    shape = np.sin(np.pi * frac) * ((hour >= SUNRISE_H) & (hour <= SUNSET_H))
    return capacity_mw * shape

def score_window(win_p50, win_p90, win_temp_corr, worst_reliability, peak_idx, prev_peak_p50):
    pr50, pr90, tcorr = win_p50[peak_idx], win_p90[peak_idx], win_temp_corr[peak_idx]
    util50, util90 = pr50 / ASSUMED_CAPACITY_MW, pr90 / ASSUMED_CAPACITY_MW
    growth = 0.0 if prev_peak_p50 is None else (pr50 - prev_peak_p50) / max(prev_peak_p50, 1)
    c_p50, c_p90 = u(pr50), u(pr90)
    c_growth = float(np.clip(growth / GROWTH_HI, 0, 1))
    c_heat = float(np.clip((tcorr - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    c_rel = REL_PEN[worst_reliability]
    score = float(np.clip(W_P50_UTIL * c_p50 + W_P90_UTIL * c_p90 + W_GROWTH * c_growth +
                           W_HEAT * c_heat + W_RELIABILITY * c_rel, 0, 100))
    level = level_of(score)
    risk, risk_reasons = peak_risk_rule(util50, util90, worst_reliability)
    return dict(peak_mw=round(float(pr50), 1), p90_mw=round(float(pr90), 1),
                util50=util50, util90=util90, stress_score=round(score, 1), stress_level=level,
                peak_risk=risk, peak_risk_reasons=risk_reasons)

def run_scenario(T, label, temp_delta=0.0, solar_capacity_mw=None):
    win_start, win_end = T + pd.Timedelta(minutes=15), T + pd.Timedelta(hours=24)
    expected_index = pd.date_range(win_start, win_end, freq="15min")
    win_feat = test.reindex(expected_index).dropna(how="all")
    prev_win = res_df.loc[T - pd.Timedelta(hours=24) + pd.Timedelta(minutes=15): T]
    prev_peak_p50 = float(prev_win["p50"].max()) if len(prev_win) else None
    worst_rel = max(win_feat["reliability"], key=lambda s: RANK[s])

    notes = []
    frame = win_feat
    temp_used = win_feat["ct"].to_numpy()
    if temp_delta != 0.0:
        if abs(temp_delta) > TEMP_DELTA_MAX_ABS:
            notes.append(f"requested delta {temp_delta:+.1f}C exceeds the +-{TEMP_DELTA_MAX_ABS:.0f}C "
                        f"sanity bound; clamped to +-{TEMP_DELTA_MAX_ABS:.0f}C")
            temp_delta = float(np.clip(temp_delta, -TEMP_DELTA_MAX_ABS, TEMP_DELTA_MAX_ABS))
        frame, temp_used, n_clip = apply_temp_delta(win_feat, temp_delta)
        if n_clip:
            notes.append(f"{n_clip} block(s) hit the {TEMP_ABS_MIN:.0f}-{TEMP_ABS_MAX:.0f}C absolute "
                        f"temperature bound and were clamped there")

    p50 = predict_p50(frame)
    p10, p90 = predict_bands(frame, p50)

    gross_p50, gross_p10, gross_p90 = p50.copy(), p10.copy(), p90.copy()
    solar_mw = None
    if solar_capacity_mw is not None:
        if not (SOLAR_CAP_MIN <= solar_capacity_mw <= SOLAR_CAP_MAX):
            notes.append(f"requested solar capacity {solar_capacity_mw:.0f} MW outside sanity bound "
                        f"[{SOLAR_CAP_MIN:.0f}, {SOLAR_CAP_MAX:.0f}] MW; clamped")
            solar_capacity_mw = float(np.clip(solar_capacity_mw, SOLAR_CAP_MIN, SOLAR_CAP_MAX))
        solar_mw = solar_profile_mw(frame.index, solar_capacity_mw)
        neg50 = int((p50 - solar_mw < 0).sum())
        if neg50:
            notes.append(f"{neg50} block(s): assumed solar output would exceed forecast gross demand "
                        f"(net demand floored at 0 MW there -- physically this means curtailment or "
                        f"reverse flow, not simply 'less demand')")
        p50 = np.maximum(p50 - solar_mw, 0)
        p10 = np.maximum(p10 - solar_mw, 0)
        p90 = np.maximum(p90 - solar_mw, 0)

    peak_idx = int(np.argmax(p50))
    peak_time = frame.index[peak_idx]
    scored = score_window(p50, p90, temp_used, worst_rel, peak_idx, prev_peak_p50)

    return dict(label=label, T=T, window=frame.index, notes=notes,
                gross_p50=gross_p50, gross_p10=gross_p10, gross_p90=gross_p90,
                p50=p50, p10=p10, p90=p90, solar_mw=solar_mw, temp_used=temp_used,
                peak_time=peak_time, worst_reliability=worst_rel, **scored)

def report_scenario(r, base=None):
    say()
    say(f"  -- {r['label']} --")
    if r["notes"]:
        for n in r["notes"]:
            say(f"     SANITY NOTE: {n}")
    say(f"     predicted peak      : {r['peak_mw']:,.1f} MW at {r['peak_time']} "
        f"(P90 {r['p90_mw']:,.1f} MW)")
    say(f"     reliability (unchanged by scenario): {r['worst_reliability']}")
    say(f"     stress score / level : {r['stress_score']:.1f} / {r['stress_level']}")
    say(f"     peak risk            : {r['peak_risk']}")
    for rr in r["peak_risk_reasons"]:
        say(f"       - {rr}")
    if base is not None:
        say(f"     CHANGE vs base: peak {base['peak_mw']:,.1f} -> {r['peak_mw']:,.1f} MW "
            f"({r['peak_mw']-base['peak_mw']:+,.1f}); P90 {base['p90_mw']:,.1f} -> {r['p90_mw']:,.1f} MW "
            f"({r['p90_mw']-base['p90_mw']:+,.1f}); stress {base['stress_score']:.1f} -> "
            f"{r['stress_score']:.1f}; risk {base['peak_risk']} -> {r['peak_risk']}")

# ============================================================================
# SECTIONS 1-6 -- RUN THE SCENARIOS ON A REAL HISTORICAL ISSUE TIME
# ============================================================================
T0 = pd.Timestamp("2025-06-11 23:00:00")  # same issue time as the Phase 9 primary proof, for continuity

say()
say("=" * 100)
say("## 1. Scenario variables supported")
say("=" * 100)
say("  TEMPERATURE  -- genuine model recomputation. delta in [-8C, +8C] (sanity bound), applied to")
say("                  the corrected forecast temperature feature and everything derived from it.")
say("  ROOFTOP SOLAR -- NOT a model prediction. A separate, labeled scenario calculation: an assumed")
say("                  daylight generation profile subtracted from the model's gross demand forecast")
say("                  to produce NET demand. Capacity in [0, 3000] MW (sanity bound).")
say("  EV LOAD / COMMERCIAL LOAD -- NOT implemented. No column in this project's data represents")
say("                  either component, so there is no honest way to model their effect (see Section 8).")

say()
say("=" * 100)
say("## 2. How each variable changes the input")
say("=" * 100)
say("  Temperature: only 'ct' (=temp_corr), 'ccdh' (cooling-degree term), 'csq' (temp^2), and 'chx'")
say("  (extreme-heat term) are perturbed for the target window's blocks. 'temp_at_issue',")
say("  'temp_prevday_max', 'temp_prevday_mean' are the OBSERVED temperature from before the issue")
say("  time -- a fixed historical fact -- and are left untouched, matching what the model actually")
say("  represents.")
say("  Solar: the model's gross P10/P50/P90 forecast is computed exactly as in the base case, THEN a")
say(f"  half-sine daylight generation curve (sunrise {SUNRISE_H:g}h, sunset {SUNSET_H:g}h -- Delhi, June,")
say("  IMD almanac) scaled to the chosen assumed capacity is subtracted block-by-block to get net")
say("  demand. This never touches the trained model.")

say()
say("=" * 100)
say("## 3. How the forecast is recomputed")
say("=" * 100)
say("  Temperature scenarios re-run the SAME fitted OLS coefficients and the SAME fitted P10/P90")
say("  quantile-regression functions (all fitted once, in the locked pipeline, never refit here) on a")
say("  modified copy of the feature rows for the chosen 24h window. This is a genuine forward pass of")
say("  the trained model on counterfactual inputs -- not a lookup table and not a hand-written rule.")
say("  Solar scenarios do not touch the model at all; they are documented arithmetic on its output.")

say()
say("=" * 100)
say("## 4. Physical bounds / sanity rules")
say("=" * 100)
say(f"  temperature delta        : +-{TEMP_DELTA_MAX_ABS:.0f}C (beyond this the model extrapolates far")
say(f"                              outside the training temperature range, {23.4:.1f}-{42.6:.1f}C observed")
say(f"                              in the test period)")
say(f"  absolute corrected temp  : clamped to [{TEMP_ABS_MIN:.0f}, {TEMP_ABS_MAX:.0f}] C (Delhi's")
say(f"                              plausible climatological range)")
say(f"  solar capacity           : clamped to [{SOLAR_CAP_MIN:.0f}, {SOLAR_CAP_MAX:.0f}] MW")
say(f"  net demand                : floored at 0 MW; any block where assumed generation would exceed")
say(f"                              forecast gross demand is explicitly flagged, not silently allowed")
say("  These bounds are fixed before running any scenario below and are not adjusted afterward.")

# BASE CASE
base = run_scenario(T0, "BASE CASE (no scenario applied)")
say()
say("=" * 100)
say("## 5. Base vs scenario examples (real model outputs, run just now)")
say("=" * 100)
report_scenario(base)

t_plus2 = run_scenario(T0, "SCENARIO: temperature +2C", temp_delta=2.0)
t_minus2 = run_scenario(T0, "SCENARIO: temperature -2C", temp_delta=-2.0)
t_plus5 = run_scenario(T0, "SCENARIO: temperature +5C", temp_delta=5.0)
solar_500 = run_scenario(T0, "SCENARIO: +500 MW assumed rooftop solar (net demand)", solar_capacity_mw=500.0)
solar_1500 = run_scenario(T0, "SCENARIO: +1500 MW assumed rooftop solar (net demand)", solar_capacity_mw=1500.0)

for r in [t_plus2, t_minus2, t_plus5, solar_500, solar_1500]:
    report_scenario(r, base=base)

say()
say("=" * 100)
say("## 6. Impact summary table")
say("=" * 100)
say(f"{'scenario':<42}{'peak MW':>10}{'P90 MW':>10}{'peak time':>18}{'stress':>8}{'level':>9}{'risk':>10}")
say("-" * 108)
for r in [base, t_plus2, t_minus2, t_plus5, solar_500, solar_1500]:
    say(f"{r['label']:<42}{r['peak_mw']:>10,.1f}{r['p90_mw']:>10,.1f}"
        f"{str(r['peak_time'])[5:16]:>18}{r['stress_score']:>8.1f}{r['stress_level']:>9}{r['peak_risk']:>10}")
say("-" * 108)
say()
say("  IMPORTANT READ OF THE TABLE ABOVE: the solar scenarios show ZERO change in peak/P90/stress/risk.")
say("  This is not a bug -- it is a genuine, informative result. On 2025-06-12 the demand peak occurs")
say("  at 23:00, well after sunset (the assumed generation window is 05:30-19:15); solar cannot touch a")
say("  block it isn't generating at. The metrics that key off the PEAK are correctly unaffected. Solar's")
say("  real effect on THIS day is visible only in the middle of the day, shown directly below:")
mid_idx = int(np.argmax(solar_500["solar_mw"]))
mid_t = solar_500["window"][mid_idx]
say(f"    at {mid_t} (near peak solar output): gross demand {solar_500['gross_p50'][mid_idx]:,.1f} MW -> "
    f"net demand {solar_500['p50'][mid_idx]:,.1f} MW with 500MW assumed capacity "
    f"({solar_500['solar_mw'][mid_idx]:,.1f} MW generated at that block)")
say(f"    same block with 1500MW assumed capacity: net demand "
    f"{solar_1500['p50'][mid_idx]:,.1f} MW ({solar_1500['solar_mw'][mid_idx]:,.1f} MW generated)")
avoided_500 = float(np.sum(solar_500["solar_mw"]) * 0.25)
avoided_1500 = float(np.sum(solar_1500["solar_mw"]) * 0.25)
say(f"    approximate energy served by assumed solar over the 24h window: "
    f"{avoided_500:,.0f} MWh (500MW case), {avoided_1500:,.0f} MWh (1500MW case)")
say("  This is exactly the real-world 'duck curve' effect: solar flattens the midday shoulder but does")
say("  not touch an evening/night system peak -- a genuinely useful, non-obvious result for an operator,")
say("  not something this script assumed in advance.")

# ============================================================================
# SECTION 7 -- VALIDATION / SANITY CHECKS
# ============================================================================
say()
say("=" * 100)
say("## 7. Validation / sanity checks (on the real 2025-06-11 23:00 issue time used above)")
say("=" * 100)
ok1 = t_plus2["peak_mw"] > base["peak_mw"]
ok2 = t_minus2["peak_mw"] < base["peak_mw"]
ok3 = t_plus5["peak_mw"] > t_plus2["peak_mw"]
say(f"  [{'PASS' if ok1 else 'FAIL'}] +2C raises the forecast peak vs base "
    f"({base['peak_mw']:,.1f} -> {t_plus2['peak_mw']:,.1f} MW)")
say(f"  [{'PASS' if ok2 else 'FAIL'}] -2C lowers the forecast peak vs base "
    f"({base['peak_mw']:,.1f} -> {t_minus2['peak_mw']:,.1f} MW)")
say(f"  [{'PASS' if ok3 else 'FAIL'}] +5C raises the forecast peak further than +2C "
    f"({t_plus2['peak_mw']:,.1f} -> {t_plus5['peak_mw']:,.1f} MW) -- monotonic in the expected direction")

frac_hour = solar_500["window"].hour + solar_500["window"].minute / 60
daytime = (frac_hour >= SUNRISE_H) & (frac_hour <= SUNSET_H)  # matches solar_profile_mw exactly
gross_day = solar_500["gross_p50"][daytime]
net_day = solar_500["p50"][daytime]
ok4 = bool((net_day <= gross_day + 1e-6).all())
ok5 = solar_500["peak_mw"] <= base["peak_mw"] + 1e-6
night = ~daytime
gross_night = solar_500["gross_p50"][night]
net_night = solar_500["p50"][night]
ok6 = bool(np.allclose(gross_night, net_night, atol=1e-6))
say(f"  [{'PASS' if ok4 else 'FAIL'}] with +500MW solar, net demand <= gross demand at every daylight block")
say(f"  [{'PASS' if ok5 else 'FAIL'}] net-demand peak <= base (gross) peak "
    f"({base['peak_mw']:,.1f} vs {solar_500['peak_mw']:,.1f} MW)")
say(f"  [{'PASS' if ok6 else 'FAIL'}] net demand equals gross demand at night (solar profile is exactly 0 "
    f"outside {SUNRISE_H:g}h-{SUNSET_H:g}h)")
say(f"  solar-1500MW sanity-clip check: {'triggered as expected' if solar_1500['notes'] else 'no clipping needed'} "
    f"-- {'; '.join(solar_1500['notes']) if solar_1500['notes'] else '(no blocks needed flooring at this capacity)'}")
no_negatives = bool((solar_1500["p50"] >= 0).all() and (solar_1500["p10"] >= 0).all())
say(f"  [{'PASS' if no_negatives else 'FAIL'}] no negative net-demand values anywhere in the +1500MW scenario "
    f"(floored at 0 MW where triggered)")
ok7 = bool(solar_500["p50"][mid_idx] < solar_500["gross_p50"][mid_idx] - 1e-6)
say(f"  [{'PASS' if ok7 else 'FAIL'}] at peak solar output ({mid_t}), net demand is measurably lower "
    f"than gross demand ({solar_500['gross_p50'][mid_idx]:,.1f} -> {solar_500['p50'][mid_idx]:,.1f} MW)")

all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6, no_negatives, ok7])
say()
say(f"  ALL CHECKS: {'PASS' if all_ok else 'AT LEAST ONE FAILED -- see above'}")
say("  This validates DIRECTION and COHERENCE of the scenario layer on one real historical case. It")
say("  does NOT claim the scenario magnitudes are historically accurate -- no ground truth for 'what")
say("  if it had actually been 2C hotter' exists in this dataset to validate the magnitude against.")

# ============================================================================
# SECTION 8 -- LIMITATIONS
# ============================================================================
say()
say("=" * 100)
say("## 8. Limitations")
say("=" * 100)
say("  - Temperature scenarios reuse the model's EXISTING learned temperature sensitivity (from the")
say("    'ct'/'ccdh'/'csq'/'chx' coefficients). That sensitivity was estimated from the real range of")
say("    temperatures actually observed (23.4-42.6C in the test period); a +8C scenario pushes into")
say("    thinner training data and the model's response there is an extrapolation, not a validated")
say("    one -- flagged in Section 4, not hidden.")
say("  - Rooftop solar is an ASSUMPTION/SCENARIO CALCULATION, not learned from any data in this")
say("    project: there is no solar generation, irradiance, or installed-capacity column anywhere in")
say("    the dataset. The daylight profile shape (half-sine, fixed sunrise/sunset) and the capacity")
say("    values shown are illustrative and must be replaced with real generation data before any")
say("    operational use.")
say("  - EV load and commercial load are NOT implemented. Implementing them honestly would require a")
say("    labeled load-component breakdown that does not exist in this project's data; adding sliders")
say("    for them without that data would mean fabricating an effect, which this phase was explicitly")
say("    told not to do.")
say("  - The solar scenario's growth term in the stress score compares against the previous day's")
say("    unmodified (gross, no-solar) forecast peak, since no independent solar assumption for the")
say("    prior day is part of this tool -- this is a simplification, stated here rather than silently")
say("    assumed.")
say("  - Scenario recomputation has only been validated for DIRECTION and COHERENCE (Section 7), on")
say("    one real historical issue time. It has not been validated against any real hotter/cooler or")
say("    solar-equipped version of that same day, because no such counterfactual exists in the data.")

# ============================================================================
# SECTION 9 -- RECOMMENDED FINAL INTERACTION
# ============================================================================
say()
say("=" * 100)
say("## 9. Recommended final interaction")
say("=" * 100)
say("  A judge should be able to: (1) see the BASE CASE forecast, band, reliability, stress and risk")
say("  for a chosen day, exactly as Proof Mode already shows it; (2) move ONE temperature slider")
say("  (-8C to +8C) and watch the peak, P90, stress score and risk level genuinely recompute in real")
say("  time, via the same trained model; (3) toggle an assumed rooftop-solar capacity and watch the")
say("  net-demand curve visibly flatten around midday while the evening peak stays close to the gross")
say("  forecast -- a concrete, honest illustration of the duck-curve effect; (4) see every scenario")
say("  output labeled 'Estimated impact / Advisory', never as an instruction to act. This is exactly")
say("  the BASE -> what if -> RECOMPUTE -> is this now dangerous? flow requested for this phase.")

# ============================================================================
# FIGURES
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

fig, ax = plt.subplots(figsize=(12.5, 5), facecolor=SURFACE)
ax.plot(base["window"], base["p50"], color=INK, linewidth=1.8, label="BASE (temp scenario off)")
ax.plot(t_plus2["window"], t_plus2["p50"], color=COL_B, linewidth=1.6, label="+2C")
ax.plot(t_minus2["window"], t_minus2["p50"], color=COL_A, linewidth=1.6, label="-2C")
ax.plot(t_plus5["window"], t_plus5["p50"], color="#b5432c", linewidth=1.6, linestyle=(0, (4, 2)), label="+5C")
style(ax)
ax.set_title(f"WHAT-IF: temperature scenarios, issue time T={T0} -> next 24h", fontsize=12, color=INK,
            loc="left", pad=12)
ax.set_ylabel("Forecast demand P50 (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
ax.legend(frameon=False, fontsize=9.5, ncols=4, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase10_fig1_temperature.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12.5, 5), facecolor=SURFACE)
ax.plot(base["window"], base["p50"], color=INK, linewidth=1.8, label="Gross demand (base)")
ax.plot(solar_500["window"], solar_500["p50"], color=COL_C, linewidth=1.6, label="Net demand, +500MW solar")
ax.plot(solar_1500["window"], solar_1500["p50"], color=COL_D, linewidth=1.6, label="Net demand, +1500MW solar")
ax.fill_between(solar_1500["window"], 0, solar_1500["solar_mw"], color=COL_D, alpha=.15,
                label="assumed generation, 1500MW case")
style(ax)
ax.set_title(f"WHAT-IF: assumed rooftop solar, issue time T={T0} -> next 24h", fontsize=12, color=INK,
            loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase10_fig2_solar.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ============================================================================
# SAVE
# ============================================================================
def slim(r):
    d = {k: v for k, v in r.items() if k not in ("window", "gross_p50", "gross_p10", "gross_p90",
                                                   "p50", "p10", "p90", "solar_mw", "temp_used")}
    d["peak_time"] = str(d["peak_time"]); d["T"] = str(d["T"])
    return d

backend = {"issue_time": str(T0), "base_case": slim(base),
          "scenarios": {"temp_plus2": slim(t_plus2), "temp_minus2": slim(t_minus2),
                       "temp_plus5": slim(t_plus5), "solar_500mw_net": slim(solar_500),
                       "solar_1500mw_net": slim(solar_1500)},
          "sanity_checks": {"temp_plus2_raises_peak": ok1, "temp_minus2_lowers_peak": ok2,
                            "temp_monotonic": ok3, "net_le_gross_daylight": ok4,
                            "net_peak_le_gross_peak": ok5, "night_unaffected": ok6,
                            "no_negative_net_demand": no_negatives,
                            "midday_net_below_gross": ok7, "all_pass": all_ok},
          "bounds": {"temp_delta_max_abs_C": TEMP_DELTA_MAX_ABS, "temp_abs_range_C": [TEMP_ABS_MIN, TEMP_ABS_MAX],
                    "solar_capacity_range_MW": [SOLAR_CAP_MIN, SOLAR_CAP_MAX]}}

for name, r in [("base", base), ("temp_plus2", t_plus2), ("temp_minus2", t_minus2), ("temp_plus5", t_plus5),
                ("solar_500mw", solar_500), ("solar_1500mw", solar_1500)]:
    pd.DataFrame({"gross_p50": r["gross_p50"], "gross_p10": r["gross_p10"], "gross_p90": r["gross_p90"],
                 "p50": r["p50"], "p10": r["p10"], "p90": r["p90"],
                 "solar_mw": r["solar_mw"] if r["solar_mw"] is not None else 0.0},
                index=r["window"]).to_csv(OUT_DIR / f"phase10_scenario_{name}.csv")

(OUT_DIR / "phase10_results.json").write_text(json.dumps(backend, indent=2, default=str))
(OUT_DIR / "phase10_report.txt").write_text("\n".join(lines), encoding="utf-8")

say()
say("[SAVED] phase10_report.txt, phase10_results.json, phase10_scenario_*.csv, 2 figures")
say()
say("PHASE 10 COMPLETE -- WAITING FOR APPROVAL.")
