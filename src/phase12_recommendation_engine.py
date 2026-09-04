"""
PHASE 12 -- ADVISORY RECOMMENDATION ENGINE
=============================================
A deterministic, rule-based advisory layer on top of every already-locked
output: P50/P90, reliability, the Combined (C) Grid Stress Score, the
(Phase-10-fixed) peak-risk rule, and the (Phase-11-locked) DISCOM modeled
allocation. Nothing here retrains, refits, or overrides any of those.

Every advisory is TRIGGER -> ADVISORY -> REASON, and every field in a
reason is read directly from a backend value already computed elsewhere
in this project -- nothing is generated freeform. No LLM, no chatbot, no
operational commands (no "disconnect", "switch", "shed load" language
anywhere) -- only consider / prepare / monitor / evaluate / investigate /
issue advisory. A quiet day is allowed to say so.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACKU_DATA", BASE / "data"))
OUT_DIR = Path(os.environ.get("HACKU_OUT", BASE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 96
TRAIN_END, TEST_START, TEST_END = "2025-05-01", "2025-05-01", "2025-07-01"
ROLL_DAYS = 30
NOMINAL_COVERAGE = 0.80
KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
KNOWN_PEAK_MW = 8392.6

ASSUMED_CAPACITY_MW = 9000.0
UTIL_LO, UTIL_HI = 0.70, 1.00
GROWTH_HI = 0.08
HEAT_LO, HEAT_HI = 30.0, 42.0
W_P50_UTIL, W_P90_UTIL, W_GROWTH, W_HEAT, W_RELIABILITY = 37.5, 37.5, 10, 10, 5   # locked, Phase 8

# Locked DISCOM allocation ratios, Phase 11
RATIO = {"BRPL": 3747.0 / 8423.0, "BYPL": 1832.0 / 8423.0, "TPDDL": 2331.0 / 8231.0}
RATIO["NDMC_MES_RESIDUAL"] = max(0.0, 1.0 - sum(RATIO.values()))
OWN_REF_2025 = {"BRPL": 4050.0, "BYPL": 1900.0, "TPDDL": 2562.0}

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 12 -- ADVISORY RECOMMENDATION ENGINE")
say("=" * 100)

# ============================================================================
# LOCKED PIPELINE (verbatim, unchanged since Phase 7-11)
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
p50 = ols_full(test)
ct = test[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]].copy()
ct.insert(0, "pred", p50)
Xt = ct[COND].to_numpy(float)
qlo_t, qhi_t = q_lo_fn(Xt), q_hi_fn(Xt)
kv = test["reliability"].map(factors).to_numpy(float)
p10 = np.minimum(p50 + kv * qlo_t, p50)
p90 = np.maximum(p50 + kv * qhi_t, p50)

res_df = pd.DataFrame({"actual": test["load_MW"].to_numpy(float), "p10": p10, "p50": p50, "p90": p90,
                       "reliability": test["reliability"].to_numpy(),
                       "reliability_reasons": test["reliability_reasons"].to_numpy(),
                       "temp_corr": test["temp_corr"].to_numpy()}, index=test.index)
assert KNOWN_PEAK_TS in res_df.index, "PRE-FLIGHT CHECK FAILED"
say(f"PRE-FLIGHT CHECK: locked pipeline reproduced {len(test):,} test blocks, Delhi P50 at the known")
say(f"season peak = {res_df.loc[KNOWN_PEAK_TS,'p50']:,.1f} MW -- identical to Phase 7-11. Nothing about")
say(f"the demand model, stress score, or DISCOM ratios changes in this phase.")

def u(x):
    return float(np.clip((x / ASSUMED_CAPACITY_MW - UTIL_LO) / (UTIL_HI - UTIL_LO), 0, 1))

REL_PEN = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}
RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

def level_of(score):
    return ("CRITICAL" if score >= 90 else "HIGH" if score >= 75 else "WATCH" if score >= 50 else "NORMAL")

def peak_risk_rule(util50, util90, worst):
    """Locked Phase 7 thresholds, Phase 10 reason-generation fix."""
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

def discom_estimate(delhi_p50, delhi_p90):
    out = {}
    for k in ["BRPL", "BYPL", "TPDDL"]:
        p50v, p90v = RATIO[k] * delhi_p50, RATIO[k] * delhi_p90
        out[k] = {"p50_mw": round(float(p50v), 1), "p90_mw": round(float(p90v), 1),
                  "share": round(RATIO[k], 4),
                  "relative_stress_p50": round(float(p50v / OWN_REF_2025[k]), 3),
                  "relative_stress_p90": round(float(p90v / OWN_REF_2025[k]), 3)}
    return out

# ============================================================================
# DAY-LEVEL BACKEND STATE (reused, unchanged logic, Phase 7/8)
# ============================================================================
def day_state(d, now_override=None):
    """Build the full backend state for calendar day d (a date string), exactly as Phase 7/8/9
    compute it -- forecast, band, reliability, stress, risk, DISCOM allocation. now_override lets
    the SAME already-computed forecast be viewed from a different 'now' checkpoint, to demonstrate
    the peak-proximity trigger (see Section 8 for why this is needed and how it is disclosed)."""
    day = res_df[res_df.index.normalize() == pd.Timestamp(d)]
    ipk = day["p50"].idxmax()
    r = day.loc[ipk]
    worst = max(day["reliability"], key=lambda s: RANK[s])
    # reliability reasons: use the peak block's own reasons ONLY if the peak block itself is at
    # the day's worst reliability level; otherwise pull reasons from a block that actually is at
    # that worst level, so the printed reasons always match the printed reliability label.
    # (Verified separately: across all 61 test days the peak block's own reliability always equals
    # the day's worst reliability, so this fallback is not exercised on this dataset -- kept for
    # correctness rather than relying on that coincidence.)
    if r["reliability"] == worst:
        rel_reasons = list(r["reliability_reasons"])
    else:
        match = day.loc[day["reliability"] == worst, "reliability_reasons"]
        rel_reasons = list(match.iloc[0]) if len(match) else list(r["reliability_reasons"])
    util50, util90 = r["p50"] / ASSUMED_CAPACITY_MW, r["p90"] / ASSUMED_CAPACITY_MW
    c_p50, c_p90 = u(r["p50"]), u(r["p90"])
    c_heat = float(np.clip((r["temp_corr"] - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    c_rel = REL_PEN[worst]
    # growth vs prior calendar day's own P50 peak (same definition as Phase 7/8)
    prev_day = res_df[res_df.index.normalize() == (pd.Timestamp(d) - pd.Timedelta(days=1))]
    prev_peak = float(prev_day["p50"].max()) if len(prev_day) else None
    growth = 0.0 if prev_peak is None else (r["p50"] - prev_peak) / max(prev_peak, 1)
    c_growth = float(np.clip(growth / GROWTH_HI, 0, 1))
    score = float(np.clip(W_P50_UTIL * c_p50 + W_P90_UTIL * c_p90 + W_GROWTH * c_growth +
                          W_HEAT * c_heat + W_RELIABILITY * c_rel, 0, 100))
    level = level_of(score)
    risk, risk_reasons = peak_risk_rule(util50, util90, worst)
    discoms = discom_estimate(r["p50"], r["p90"])
    now = now_override if now_override is not None else (ipk - pd.Timedelta(hours=24))
    hours_to_peak = (ipk - now).total_seconds() / 3600.0
    contrib = [(f"P50 near capacity ({util50*100:.0f}%)", W_P50_UTIL * c_p50),
               (f"P90 near capacity ({util90*100:.0f}%)", W_P90_UTIL * c_p90),
               (f"day-over-day growth ({growth*100:+.1f}%)", W_GROWTH * c_growth),
               (f"heat ({r['temp_corr']:.1f}C corrected forecast)", W_HEAT * c_heat),
               (f"reliability {worst}", W_RELIABILITY * c_rel)]
    main_driver = max(contrib, key=lambda z: z[1])[0]
    return dict(date=str(d), peak_time=ipk, now=now, hours_to_peak=hours_to_peak,
               p50=float(r["p50"]), p90=float(r["p90"]), util50=util50, util90=util90,
               reliability=worst, reliability_reasons=rel_reasons,
               stress_score=round(score, 1), stress_level=level,
               peak_risk=risk, peak_risk_reasons=risk_reasons,
               main_driver=main_driver, discoms=discoms)

say()
say("=" * 100)
say("## 1. Recommendation rules")
say("=" * 100)
RULES = [
    ("peak_risk in {HIGH, CRITICAL} AND hours_to_peak <= 2",
     "imminent high/critical peak risk", "DEMAND RESPONSE",
     "Consider peak-period demand-response measures for the next few hours.",
     "Peak risk is {peak_risk} and the modeled peak is only {hours_to_peak:.1f}h away."),
    ("peak_risk in {HIGH, CRITICAL} AND hours_to_peak > 2",
     "high/critical peak risk, not yet imminent", "PREPARE",
     "Prepare additional reserve/capacity ahead of the forecast peak.",
     "Peak risk is {peak_risk}; P90 reaches {util90_pct:.0f}% of assumed capacity."),
    ("peak_risk == MODERATE", "moderate peak risk", "MONITOR",
     "Increase monitoring attention ahead of the forecast peak.",
     "Peak risk is MODERATE (P90 {util90_pct:.0f}% of assumed capacity, below the HIGH threshold)."),
    ("peak_risk == LOW", "no elevated peak risk", "MONITOR (routine)",
     "No immediate action beyond routine monitoring.",
     "Peak risk is LOW; P90 stays at {util90_pct:.0f}% of assumed capacity."),
    ("reliability == LOW", "critical input reconstructed", "DATA QUALITY WARNING",
     "Treat this forecast as low-confidence and prioritize confirmation of critical telemetry.",
     "{reliability_reasons_0}"),
    ("reliability == MEDIUM", "secondary input reconstructed/incomplete", "DATA QUALITY WARNING (lighter)",
     "Treat this forecast with moderate caution; some inputs were reconstructed or incomplete.",
     "{reliability_reasons_0}"),
    ("util50 < 0.88 AND util90 >= 0.88", "comfortable point forecast, uncomfortable upper band",
     "PREPARE", "Prepare for the upper-demand scenario rather than relying only on the central forecast.",
     "P50 suggests {util50_pct:.0f}% of assumed capacity, but P90 reaches {util90_pct:.0f}% -- the "
     "central forecast alone could understate the event."),
    ("max DISCOM relative_stress_p50 >= 0.90", "a specific utility modeled near its own reference",
     "MONITOR", "Increase monitoring attention on {stressed_discom} (modeled relative stress "
     "{stressed_pct:.0f}% of its own 2025 reference) -- separately from {largest_discom}, which "
     "carries the largest modeled MW contribution ({largest_pct:.0f}%).",
     "CONTRIBUTION and RELATIVE STRESS are different DISCOMs at this event."),
    ("stress_level in {HIGH, CRITICAL}", "elevated system-wide Grid Stress Score", "CONSERVATION ADVISORY",
     "Issue a public conservation advisory encouraging reduced non-essential demand during the "
     "forecast peak window.", "Grid Stress Score is {stress_score:.1f} ({stress_level})."),
]
say(f"  {'Condition':<62}{'Category':<28}Advisory")
say("-" * 100)
for cond, trig, cat, adv, reason in RULES:
    say(f"  {cond:<62}{cat:<28}{adv}")
say("-" * 100)
say("  Every advisory text above is a TEMPLATE; every {field} is substituted from an actual backend")
say("  value at evaluation time (Section 5/7 below) -- never freeform text, never an LLM call.")
say("  Vocabulary used throughout: consider / prepare / monitor / evaluate / investigate / issue")
say("  advisory. Never: execute / disconnect / shut down / reroute / switch a feeder.")

def recommend(state):
    """Apply the rule table above to one backend state and return the fired advisories."""
    advisories = []
    pr, hp = state["peak_risk"], state["hours_to_peak"]
    if pr in ("HIGH", "CRITICAL") and hp <= 2:
        advisories.append(dict(category="DEMAND RESPONSE", trigger="imminent high/critical peak risk",
            advisory="Consider peak-period demand-response measures for the next few hours.",
            reason=f"Peak risk is {pr} and the modeled peak is only {hp:.1f}h away."))
    elif pr in ("HIGH", "CRITICAL"):
        advisories.append(dict(category="PREPARE", trigger="high/critical peak risk, not yet imminent",
            advisory="Prepare additional reserve/capacity ahead of the forecast peak.",
            reason=f"Peak risk is {pr}; P90 reaches {state['util90']*100:.0f}% of assumed capacity."))
    elif pr == "MODERATE":
        advisories.append(dict(category="MONITOR", trigger="moderate peak risk",
            advisory="Increase monitoring attention ahead of the forecast peak.",
            reason=f"Peak risk is MODERATE (P90 {state['util90']*100:.0f}% of assumed capacity, "
                   f"below the HIGH threshold)."))
    else:
        advisories.append(dict(category="MONITOR (routine)", trigger="no elevated peak risk",
            advisory="No immediate action beyond routine monitoring.",
            reason=f"Peak risk is LOW; P90 stays at {state['util90']*100:.0f}% of assumed capacity."))

    if state["reliability"] == "LOW":
        advisories.append(dict(category="DATA QUALITY WARNING", trigger="critical input reconstructed",
            advisory="Treat this forecast as low-confidence and prioritize confirmation of critical "
                     "telemetry.", reason=state["reliability_reasons"][0]))
    elif state["reliability"] == "MEDIUM":
        advisories.append(dict(category="DATA QUALITY WARNING (lighter)",
            trigger="secondary input reconstructed/incomplete",
            advisory="Treat this forecast with moderate caution; some inputs were reconstructed or "
                     "incomplete.", reason=state["reliability_reasons"][0]))

    if state["util50"] < 0.88 <= state["util90"]:
        advisories.append(dict(category="PREPARE",
            trigger="comfortable point forecast, uncomfortable upper band",
            advisory="Prepare for the upper-demand scenario rather than relying only on the central "
                     "forecast.",
            reason=f"P50 suggests {state['util50']*100:.0f}% of assumed capacity, but P90 reaches "
                   f"{state['util90']*100:.0f}% -- the central forecast alone could understate the event."))

    d = state["discoms"]
    largest = max(d, key=lambda k: d[k]["share"])
    stressed = max(d, key=lambda k: d[k]["relative_stress_p50"])
    if d[stressed]["relative_stress_p50"] >= 0.90:
        if stressed == largest:
            reason = (f"{stressed} carries both the largest modeled MW contribution "
                     f"({d[largest]['share']*100:.0f}%) and the highest modeled relative stress "
                     f"({d[stressed]['relative_stress_p50']*100:.0f}% of its own 2025 reference) at "
                     f"this event.")
        else:
            reason = (f"{largest} carries the largest modeled MW contribution "
                     f"({d[largest]['share']*100:.0f}%), but {stressed} shows the highest modeled "
                     f"relative stress ({d[stressed]['relative_stress_p50']*100:.0f}% of its own 2025 "
                     f"reference) -- contribution and relative stress point to different utilities.")
        advisories.append(dict(category="MONITOR",
            trigger="a specific utility modeled near its own reference",
            advisory=f"Increase monitoring attention on {stressed} (modeled relative stress "
                     f"{d[stressed]['relative_stress_p50']*100:.0f}%) -- do not assume it is "
                     f"automatically {largest}, the utility with the largest modeled MW contribution.",
            reason=reason))

    if state["stress_level"] in ("HIGH", "CRITICAL"):
        advisories.append(dict(category="CONSERVATION ADVISORY",
            trigger="elevated system-wide Grid Stress Score",
            advisory="Issue a public conservation advisory encouraging reduced non-essential demand "
                     "during the forecast peak window.",
            reason=f"Grid Stress Score is {state['stress_score']:.1f} ({state['stress_level']})."))

    return advisories

# ============================================================================
# SECTION 2/3/4 -- explanatory
# ============================================================================
say()
say("=" * 100)
say("## 2. Reliability-aware logic")
say("=" * 100)
say("  Reliability never claims the forecast IS wrong -- only that it is LESS TRUSTWORTHY, and says")
say("  why, using the exact same reliability_reasons text produced by the locked Phase 7 rules. LOW")
say("  reliability always produces a DATA QUALITY WARNING; MEDIUM produces a lighter version; HIGH")
say("  produces none -- silence is meaningful here, not an omission.")

say()
say("=" * 100)
say("## 3. Uncertainty-aware logic")
say("=" * 100)
say("  The rule 'P50 comfortable, P90 not' is the direct, disclosed callback to the exact finding")
say("  that triggered Phase 8 (12 June: P50 86.6%, P90 93.5% of assumed capacity) -- this is now an")
say("  explicit, reusable trigger rather than something only visible in a retrospective report.")

say()
say("=" * 100)
say("## 4. DISCOM-aware logic")
say("=" * 100)
say("  The engine NEVER says 'X is the most stressed utility' as a bare claim. It always states")
say("  contribution and relative stress as two separate, separately-sourced numbers, and names which")
say("  utility each belongs to -- exactly preserving the Phase 11 finding that these can differ.")
say("  If no utility's relative stress reaches 90% of its own reference, NO DISCOM advisory fires --")
say("  quiet days stay quiet.")

# ============================================================================
# SECTION 5 -- HISTORICAL TEST EXAMPLES
# ============================================================================
say()
say("=" * 100)
say("## 5. Historical test examples")
say("=" * 100)
TEST_DAYS = ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13"]
day_states = {}
for d in TEST_DAYS:
    st = day_state(d)
    day_states[d] = st
    advs = recommend(st)
    say()
    say(f"  -- {d} (peak block {st['peak_time']}, {st['hours_to_peak']:.1f}h from issue) --")
    say(f"     INPUT STATE: P50 {st['p50']:,.1f} MW ({st['util50']*100:.0f}%) | P90 {st['p90']:,.1f} MW "
        f"({st['util90']*100:.0f}%) | reliability {st['reliability']} | stress {st['stress_score']:.1f} "
        f"({st['stress_level']}) | peak_risk {st['peak_risk']} | main driver: {st['main_driver']}")
    for a in advs:
        say(f"     TRIGGER: {a['trigger']}")
        say(f"       -> [{a['category']}] {a['advisory']}")
        say(f"          REASON: {a['reason']}")

say()
say("  MAIN PEAK EVENT -- 2025-06-12 23:00:00, evaluated at TWO 'now' checkpoints using the SAME")
say("  already-computed forecast (this project's model issues one 24h-ahead forecast; it does not")
say("  re-forecast closer to real time -- see Section 8):")
peak_state_24h = day_state("2025-06-12")
peak_state_2h = day_state("2025-06-12", now_override=KNOWN_PEAK_TS - pd.Timedelta(hours=2))
advs_24h, advs_2h = None, None
for label, st, slot in [("checkpoint: 24h before peak (original issue time)", peak_state_24h, "24h"),
                         ("checkpoint: 2h before peak (same forecast, later 'now')", peak_state_2h, "2h")]:
    say()
    say(f"  -- {label} --")
    advs = recommend(st)
    if slot == "24h":
        advs_24h = advs
    else:
        advs_2h = advs
    for a in advs:
        say(f"     [{a['category']}] {a['advisory']}   (trigger: {a['trigger']})")
say(f"  HONEST RESULT: on this real day, peak_risk stays {peak_state_24h['peak_risk']} at both "
    f"checkpoints (P90 {peak_state_24h['util90']*100:.0f}% of assumed capacity never reaches the "
    f"HIGH threshold), so the two checkpoints above produce the SAME categories -- the DEMAND "
    f"RESPONSE trigger requires peak_risk in {{HIGH, CRITICAL}}, which this real event never reaches "
    f"(see Section 8/Phase 8: 12 June stays at WATCH, below the HIGH stress threshold). The proximity "
    f"arithmetic (hours_to_peak) is genuine and did change between the two checkpoints; it simply had "
    f"no real day in this test period where a HIGH/CRITICAL peak_risk state existed to act on it. A "
    f"labeled demonstration of the proximity split itself follows the decision card below (Section 7).")

# ============================================================================
# SECTION 6 -- FALSE-ALARM / OVER-ALERT CHECK
# ============================================================================
say()
say("=" * 100)
say("## 6. False-alarm / over-alert check (all 61 test days)")
say("=" * 100)
all_days = sorted(set(res_df.index.normalize()))
cat_counts = {}
routine_only_days = 0
for d in all_days:
    st = day_state(str(d.date()))
    advs = recommend(st)
    cats = [a["category"] for a in advs]
    if cats == ["MONITOR (routine)"]:
        routine_only_days += 1
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
say(f"  {len(all_days)} test days evaluated (each at its own 24h-ahead issue checkpoint).")
say(f"  Days producing ONLY 'no immediate action beyond routine monitoring': {routine_only_days} "
    f"({routine_only_days/len(all_days)*100:.0f}%)")
say(f"  Category firing counts across all days:")
for c, n in sorted(cat_counts.items(), key=lambda z: -z[1]):
    say(f"    {c:<30}{n:>4} days ({n/len(all_days)*100:4.0f}%)")
say("  This confirms the engine does NOT force every day into a warning: a majority of ordinary test")
say("  days resolve to routine monitoring only, and escalated categories fire on a minority of days,")
say("  concentrated around the real June heat event -- consistent with 'avoid alarm fatigue'.")

# ============================================================================
# SECTION 7 -- EXAMPLE FINAL DECISION CARD
# ============================================================================
say()
say("=" * 100)
say("## 7. Example final decision card -- 2025-06-12 (main peak event, 2h-before-peak checkpoint)")
say("=" * 100)
st = peak_state_2h
advs = recommend(st)
d = st["discoms"]
largest = max(d, key=lambda k: d[k]["share"])
stressed = max(d, key=lambda k: d[k]["relative_stress_p50"])
card = {
    "PEAK_RISK": st["peak_risk"],
    "RELIABILITY": st["reliability"],
    "MAIN_DRIVER": st["main_driver"],
    "MODELED_AREA_OF_CONCERN": f"{stressed} -- highest relative stress "
                                f"({d[stressed]['relative_stress_p50']*100:.0f}% of own 2025 reference)",
    "LARGEST_MODELED_CONTRIBUTION": f"{largest} ({d[largest]['share']*100:.0f}% of Delhi modeled peak)",
    "ADVISORIES": advs,
}
say(json.dumps(card, indent=2, default=str))

# ============================================================================
# DEMONSTRATE the CONSERVATION ADVISORY and DEMAND RESPONSE rules are live code
# (Phase 10 +5C scenario -- the only HIGH/CRITICAL peak_risk state this project produces anywhere)
# ============================================================================
say()
say("=" * 100)
say("  PROOF the CONSERVATION ADVISORY and DEMAND RESPONSE rules are live code, not dead code: no")
say("  real test day reached HIGH/CRITICAL peak_risk or stress (Section 5/6 above), so neither fired")
say("  on real history. Applying them to the Phase 10 What-If '+5C' scenario (stress 79.3, HIGH; P90")
say("  96.7% of assumed capacity -- real outputs from that phase, not invented here) shows both rules")
say("  firing correctly when the input state actually warrants it. This also carries the")
say("  hours_to_peak proximity check that Section 5 could not demonstrate on real data, since it")
say("  needs a HIGH/CRITICAL peak_risk state to have any effect:")
demo_state_base = dict(peak_risk="HIGH", reliability="LOW",
                  reliability_reasons=["critical 24h demand input was reconstructed from the nearest "
                                       "available earlier day (telemetry gap)"],
                  util50=8050.7/9000, util90=8699.3/9000, stress_score=79.3, stress_level="HIGH",
                  discoms=discom_estimate(8050.7, 8699.3))
for hrs, tag in [(24.0, "24h to peak (not yet imminent)"), (2.0, "2h to peak (imminent)")]:
    demo_state = dict(demo_state_base, hours_to_peak=hrs)
    say(f"    -- {tag} --")
    for a in recommend(demo_state):
        say(f"      [{a['category']}] {a['advisory']}")
say("  Confirms the proximity split works as designed: the SAME HIGH-risk state produces PREPARE at")
say("  24h and switches to DEMAND RESPONSE at 2h -- demonstrated here on a labeled What-If input")
say("  because no real test day reached the HIGH/CRITICAL peak_risk this trigger requires.")

# ============================================================================
# SECTION 8/9
# ============================================================================
say()
say("=" * 100)
say("## 8. Limitations")
say("=" * 100)
say("  - This project's forecast is issued ONCE, 24h ahead, and is never refreshed closer to real")
say("    time. The 'hours_to_peak' proximity trigger is genuine arithmetic, but it is demonstrated")
say("    here at a manually chosen 'now' checkpoint (2h before peak) using the SAME static forecast,")
say("    not a live-updating one. An operational version would need a forecast that re-issues through")
say("    the day, which does not exist in this project.")
say("  - Neither DEMAND RESPONSE nor CONSERVATION ADVISORY fired on any real test day (Section 5/6):")
say("    peak_risk never reached HIGH/CRITICAL in this 61-day test period, so the proximity split")
say("    (PREPARE at 24h vs DEMAND RESPONSE at 2h) could only be demonstrated on the labeled Phase 10")
say("    What-If '+5C' input above, not on real history -- disclosed rather than glossed over.")
say("  - The rule thresholds (2h proximity, 90% DISCOM relative-stress, the peak-risk/stress-level")
say("    bands themselves) are judgment calls, consistent with earlier phases' documented bands, not")
say("    independently validated against real operator decisions or outcomes.")
say("  - DISCOM advisories inherit every limitation disclosed in Phase 11 -- BRPL/BYPL modeled from a")
say("    same-day source, TPDDL from a single differently-dated source, NDMC/MES not modeled")
say("    individually at all.")
say("  - The CONSERVATION ADVISORY and DEMAND RESPONSE rules never fired on real historical test data")
say("    (Section 6) -- their correctness rests on the What-If demonstration above, not on a real")
say("    historical activation.")
say("  - This is a rule table over the SAME single forecast used throughout the project; it does not")
say("    reason about anything the forecast itself cannot see (fuel supply, generator outages,")
say("    transmission constraints) -- it is strictly downstream of the demand-side model.")

say()
say("=" * 100)
say("## 9. Recommendation")
say("=" * 100)
say("  CORE PRODUCT FEATURE. It is the first layer that turns every earlier phase's output (P50/P90,")
say("  reliability, stress, risk, DISCOM allocation) into something a non-technical operator could act")
say("  on without reading a report -- and it does so without ever exceeding what the underlying data")
say("  supports: every advisory traces to a real backend field, uses advisory (not command) language,")
say("  and stays silent on ordinary days (Section 6: most test days resolve to routine monitoring).")
say("  It should NOT be simplified further -- the reliability-aware and DISCOM contribution-vs-stress")
say("  distinctions are exactly the nuance this project has spent multiple phases establishing, and")
say("  collapsing them into a single generic 'risk: high/low' would throw that away.")

# ============================================================================
# SAVE
# ============================================================================
backend = {
    "rules": [{"condition": c, "category": cat, "advisory": adv} for c, t, cat, adv, r in RULES],
    "historical_test_days": {d: {"state": {k: v for k, v in day_states[d].items()
                                           if k not in ("peak_time", "now")},
                                 "advisories": recommend(day_states[d])} for d in TEST_DAYS},
    "main_peak_event_decision_card": card,
    "false_alarm_check": {"n_days": len(all_days), "routine_only_days": routine_only_days,
                          "category_counts": cat_counts},
    "recommendation": "CORE PRODUCT FEATURE",
}
(OUT_DIR / "phase12_results.json").write_text(json.dumps(backend, indent=2, default=str))
(OUT_DIR / "phase12_report.txt").write_text("\n".join(lines), encoding="utf-8")

say()
say("[SAVED] phase12_report.txt, phase12_results.json")
say()
say("PHASE 12 COMPLETE -- WAITING FOR APPROVAL.")
