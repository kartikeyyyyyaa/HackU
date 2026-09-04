"""
PHASE 7 -- FORECAST RELIABILITY + GRID RISK LAYER
==================================================
Built on the LOCKED foundation: OLS point forecast, 15-min blocks, genuine
24h-ahead horizon, corrected demand pipeline, adaptive 30-day time-of-day
weather bias correction, nearest-available-day imputation, P10/P50/P90.

Nothing about the forecasting core is changed here. This phase adds:
  A. a transparent rule-based reliability state (HIGH / MEDIUM / LOW)
  B. reliability-aware widening of the uncertainty band
  D. a documented 0-100 Grid Stress Score
  E. a peak-risk category
  F. machine-derived explanations tied to actual computed conditions
  G. a backend decision-card structure (values only -- no UI)

Every calibration quantity is derived from out-of-fold TRAINING data.
The test window is never used to choose a rule, a weight or a factor.
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

# ---- ASSUMED / ILLUSTRATIVE capacity. NOT an official figure. ----
ASSUMED_CAPACITY_MW = 9000.0
CAPACITY_NOTE = ("ASSUMED / ILLUSTRATIVE ceiling, not an official or sourced figure. Chosen as a round "
                 "value above the highest demand observed anywhere in our dataset (8,631.5 MW at 5-min "
                 "resolution). It exists to demonstrate the index and MUST be replaced with a verified "
                 "operational value before any external claim.")

# Grid Stress Score weights -- documented, sum to 100
W_P50_UTIL, W_P90_UTIL, W_GROWTH, W_HEAT, W_RELIABILITY = 55, 20, 10, 10, 5
UTIL_LO, UTIL_HI = 0.70, 1.00      # utilisation mapped from this range onto 0..1
GROWTH_HI = 0.08                    # +8% day-over-day maps to full marks
HEAT_LO, HEAT_HI = 30.0, 42.0       # corrected forecast temp mapped onto 0..1

INK, COL_A, COL_B, COL_C = "#2f3437", "#0d76b8", "#c9701a", "#2f7d4f"
GRID, SURFACE = "#dfe2df", "#fcfcfb"

lines = []
def say(s=""):
    print(s); lines.append(s)

say("PHASE 7 -- FORECAST RELIABILITY + GRID RISK")
say("=" * 100)

# ============================================================================
# LOCKED PIPELINE
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

# ============================================================================
# PART A -- RELIABILITY STATE (transparent rules, each with a reason)
# ============================================================================
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

say()
say("PART A -- RELIABILITY RULES")
say("-" * 100)
say("  LOW    : lag_24h reconstructed (the model's strongest input is not telemetry)")
say("  MEDIUM : lag_24h observed, but lag_48h reconstructed OR <95% of the previous 24h observed")
say("           OR the weather bias correction could not be computed")
say("  HIGH   : all critical demand inputs observed and the previous 24h of telemetry is complete")
say("  Every rule maps to a stated reason; nothing is included merely because it was available.")

train = f[f.index < TRAIN_END]
test = f[(f.index >= TEST_START) & (f.index < TEST_END)]
say()
say(f"  state distribution   {'train':>18}{'test':>18}")
for st in ["HIGH", "MEDIUM", "LOW"]:
    say(f"    {st:<8}{int((train['reliability']==st).sum()):>18,}"
        f"{int((test['reliability']==st).sum()):>18,}")

# ============================================================================
# MODEL + PART B/C -- RELIABILITY-AWARE UNCERTAINTY
# ============================================================================
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

# out-of-fold residuals, folds require >= 12 months of training history (Phase 3 rule)
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

say()
say("PART B -- RELIABILITY-AWARE WIDENING (calibrated on out-of-fold TRAINING data only)")
say("-" * 100)
say(f"  out-of-fold calibration rows by state: " +
    ", ".join(f"{st} {int((oof['reliability']==st).sum()):,}" for st in ["HIGH", "MEDIUM", "LOW"]))

# smallest multiplicative factor k_state achieving nominal coverage on OOF rows of that state
qlo_o, qhi_o = q_lo_fn(Xo), q_hi_fn(Xo)
factors, MIN_N = {}, 100
for st in ["HIGH", "MEDIUM", "LOW"]:
    m = (oof["reliability"] == st).to_numpy()
    if m.sum() < MIN_N:
        factors[st] = None
        say(f"  {st:<7}: only {int(m.sum())} calibration rows (<{MIN_N}) -> no state-specific factor fitted")
        continue
    chosen = None
    for k in np.arange(0.5, 6.001, 0.05):
        cov = ((yo[m] >= k * qlo_o[m]) & (yo[m] <= k * qhi_o[m])).mean()
        if cov >= NOMINAL_COVERAGE:
            chosen = float(k); break
    factors[st] = chosen if chosen is not None else 6.0
    say(f"  {st:<7}: {int(m.sum()):>6,} rows -> width factor k = {factors[st]:.2f} "
        f"(smallest k reaching {NOMINAL_COVERAGE*100:.0f}% coverage in calibration)")
# any state without its own factor inherits the next-most-cautious fitted factor
fitted = [v for v in factors.values() if v is not None]
for st in factors:
    if factors[st] is None:
        factors[st] = max(fitted)
        say(f"  {st:<7}: inherits the most cautious fitted factor k = {factors[st]:.2f}")

# apply to test
ols_full = fit_ols(train, COLS)
p50 = ols_full(test)
ct = test[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]].copy()
ct.insert(0, "pred", p50)
Xt = ct[COND].to_numpy(float)
qlo_t, qhi_t = q_lo_fn(Xt), q_hi_fn(Xt)
kv = test["reliability"].map(factors).to_numpy(float)

p10_base = np.minimum(p50 + qlo_t, p50)
p90_base = np.maximum(p50 + qhi_t, p50)
p10_rel = np.minimum(p50 + kv * qlo_t, p50)
p90_rel = np.maximum(p50 + kv * qhi_t, p50)

y = test["load_MW"].to_numpy(float)
peak_threshold = float(test["load_MW"].quantile(PEAK_QUANTILE))
is_peak = (test["load_MW"] >= peak_threshold).to_numpy()
daily_peak_idx = test.groupby(test.index.normalize())["load_MW"].idxmax()
st_arr = test["reliability"].to_numpy()

def pinball(q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))

def band_stats(lo, hi):
    ins = (y >= lo) & (y <= hi)
    w = hi - lo
    out = {"coverage_all": float(ins.mean() * 100),
           "coverage_peak": float(ins[is_peak].mean() * 100),
           "coverage_dailypeak": float(pd.Series(ins, index=test.index).loc[daily_peak_idx].mean() * 100),
           "breach_upper": float((y > hi).mean() * 100),
           "breach_lower": float((y < lo).mean() * 100),
           "width_mean": float(w.mean()),
           "pinball_lo": pinball(lo, 0.10), "pinball_hi": pinball(hi, 0.90)}
    for st in ["HIGH", "MEDIUM", "LOW"]:
        m = st_arr == st
        out[f"coverage_{st}"] = float(ins[m].mean() * 100) if m.sum() else float("nan")
        out[f"width_{st}"] = float(w[m].mean()) if m.sum() else float("nan")
        out[f"n_{st}"] = int(m.sum())
    return out

b_base, b_rel = band_stats(p10_base, p90_base), band_stats(p10_rel, p90_rel)

say()
say("PART C -- BEFORE vs AFTER")
say("-" * 100)
say(f"{'metric':<34}{'current band':>16}{'reliability-aware':>20}{'nominal':>10}")
say("-" * 100)
rows = [("overall P10-P90 coverage", "coverage_all", "80%"),
        ("coverage, HIGH reliability", "coverage_HIGH", "80%"),
        ("coverage, MEDIUM reliability", "coverage_MEDIUM", "80%"),
        ("coverage, LOW reliability", "coverage_LOW", "80%"),
        ("coverage, peak-period blocks", "coverage_peak", "80%"),
        ("coverage, daily peaks", "coverage_dailypeak", "80%"),
        ("P90 breach rate", "breach_upper", "10%"),
        ("P10 breach rate", "breach_lower", "10%")]
for lbl, key, nom in rows:
    say(f"{lbl:<34}{b_base[key]:>15.1f}%{b_rel[key]:>19.1f}%{nom:>10}")
say(f"{'mean interval width (MW)':<34}{b_base['width_mean']:>16.1f}{b_rel['width_mean']:>20.1f}")
for st in ["HIGH", "MEDIUM", "LOW"]:
    say(f"{'  width, ' + st + ' (MW)':<34}{b_base['width_'+st]:>16.1f}{b_rel['width_'+st]:>20.1f}"
        f"   (n={b_rel['n_'+st]:,})")
say(f"{'pinball loss P10':<34}{b_base['pinball_lo']:>16.2f}{b_rel['pinball_lo']:>20.2f}")
say(f"{'pinball loss P90':<34}{b_base['pinball_hi']:>16.2f}{b_rel['pinball_hi']:>20.2f}")
say("-" * 100)

# ============================================================================
# PART D/E/F -- STRESS SCORE, PEAK RISK, EXPLANATIONS (per forecast day)
# ============================================================================
say()
say("PART D -- GRID STRESS SCORE (transparent, documented, NOT a learned target)")
say("-" * 100)
say(f"  assumed capacity : {ASSUMED_CAPACITY_MW:,.0f} MW")
say(f"  {CAPACITY_NOTE}")
say(f"  score = {W_P50_UTIL}*u(P50) + {W_P90_UTIL}*u(P90) + {W_GROWTH}*growth + "
    f"{W_HEAT}*heat + {W_RELIABILITY}*unreliability   (max 100)")
say(f"    u(x)      = clip((x/capacity - {UTIL_LO}) / ({UTIL_HI} - {UTIL_LO}), 0, 1)")
say(f"    growth    = clip(day-over-day change in forecast peak / {GROWTH_HI}, 0, 1)")
say(f"    heat      = clip((corrected forecast temp - {HEAT_LO}) / ({HEAT_HI} - {HEAT_LO}), 0, 1)")
say(f"    unreliab. = HIGH 0.0, MEDIUM 0.5, LOW 1.0")
say(f"  bands: 0-50 NORMAL | 50-75 WATCH | 75-90 HIGH | 90-100 CRITICAL (provisional)")

res_df = pd.DataFrame({"actual": y, "p10": p10_rel, "p50": p50, "p90": p90_rel,
                       "reliability": st_arr, "temp_corr": test["temp_corr"].to_numpy()},
                      index=test.index)
res_df["p10_base"], res_df["p90_base"] = p10_base, p90_base

def u(x):
    return float(np.clip((x / ASSUMED_CAPACITY_MW - UTIL_LO) / (UTIL_HI - UTIL_LO), 0, 1))

REL_PEN = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}
RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

days = sorted(set(res_df.index.normalize()))
cards, prev_peak = [], None
for d in days:
    day = res_df[res_df.index.normalize() == d]
    ipk = day["p50"].idxmax()
    r = day.loc[ipk]
    # the day's reliability is the worst state seen across that day's blocks
    worst = max(day["reliability"], key=lambda s: RANK[s])
    growth = 0.0 if prev_peak is None else (r["p50"] - prev_peak) / max(prev_peak, 1)
    c_p50, c_p90 = u(r["p50"]), u(r["p90"])
    c_growth = float(np.clip(growth / GROWTH_HI, 0, 1))
    c_heat = float(np.clip((r["temp_corr"] - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    c_rel = REL_PEN[worst]
    score = (W_P50_UTIL * c_p50 + W_P90_UTIL * c_p90 + W_GROWTH * c_growth +
             W_HEAT * c_heat + W_RELIABILITY * c_rel)
    score = float(np.clip(score, 0, 100))
    level = ("CRITICAL" if score >= 90 else "HIGH" if score >= 75
             else "WATCH" if score >= 50 else "NORMAL")

    # PART E -- peak risk rule
    util50, util90 = r["p50"] / ASSUMED_CAPACITY_MW, r["p90"] / ASSUMED_CAPACITY_MW
    if util90 >= 1.00 and (util50 >= 0.95 or worst == "LOW"):
        risk = "CRITICAL"
    elif util90 >= 0.95 or util50 >= 0.92:
        risk = "HIGH"
    elif util90 >= 0.88:
        risk = "MODERATE"
    else:
        risk = "LOW"
    risk_reasons = []
    if util90 >= 0.95:
        risk_reasons.append(f"P90 reaches {util90*100:.0f}% of assumed capacity")
    if util50 >= 0.92:
        risk_reasons.append(f"P50 reaches {util50*100:.0f}% of assumed capacity")
    if worst == "LOW":
        risk_reasons.append("forecast reliability is LOW, so the upper bound is less dependable")
    if not risk_reasons:
        risk_reasons.append(f"P90 stays at {util90*100:.0f}% of assumed capacity")

    # PART F -- explanation from the largest actual contributors
    contrib = [(f"forecast peak is {util50*100:.0f}% of assumed capacity", W_P50_UTIL * c_p50),
               (f"upper bound (P90) reaches {util90*100:.0f}% of assumed capacity", W_P90_UTIL * c_p90),
               (f"forecast peak is {growth*100:+.1f}% versus the previous day", W_GROWTH * c_growth),
               (f"corrected temperature forecast at peak is {r['temp_corr']:.1f}C", W_HEAT * c_heat),
               (f"forecast reliability is {worst}", W_RELIABILITY * c_rel)]
    reasons = [t for t, v in sorted(contrib, key=lambda z: -z[1]) if v > 0.5][:4]

    # day-level reliability reasons, built from day-level FACTS (not concatenated block strings,
    # which produced contradictory and near-duplicate entries)
    dslice = test.loc[day.index]
    n_lag24 = int(dslice["lag24_imputed"].sum())
    n_lag48 = int(dslice["lag48_imputed"].sum())
    min_comp = float(dslice["prev24_completeness"].min())
    n_wx = int(dslice["weather_corr_missing"].sum())
    rel_reasons = []
    if n_lag24:
        rel_reasons.append(f"critical 24h demand input reconstructed from the nearest available "
                           f"earlier day for {n_lag24} of {len(dslice)} blocks (telemetry gap)")
    if n_lag48:
        rel_reasons.append(f"48h demand input reconstructed for {n_lag48} of {len(dslice)} blocks")
    if min_comp < 0.95:
        rel_reasons.append(f"previous-24h demand telemetry only {min_comp*100:.0f}% complete at worst")
    if n_wx:
        rel_reasons.append(f"weather bias correction unavailable for {n_wx} blocks")
    if not rel_reasons:
        rel_reasons = ["all critical demand inputs observed; previous 24h telemetry complete"]
    cards.append({"date": str(d.date()), "peak_time": ipk.strftime("%H:%M"),
                  "p10_mw": round(float(r["p10"]), 1), "p50_mw": round(float(r["p50"]), 1),
                  "p90_mw": round(float(r["p90"]), 1), "actual_peak_mw": round(float(day["actual"].max()), 1),
                  "assumed_capacity_mw": ASSUMED_CAPACITY_MW,
                  "reliability": worst, "reliability_reasons": rel_reasons[:3],
                  "stress_score": round(score, 1), "stress_level": level,
                  "stress_reasons": reasons, "peak_risk": risk, "peak_risk_reasons": risk_reasons})
    prev_peak = r["p50"]

cards_df = pd.DataFrame(cards).set_index("date")

say()
say("  stress level distribution across the 61 test days: " +
    ", ".join(f"{k} {v}" for k, v in cards_df["stress_level"].value_counts().items()))
say("  peak risk distribution:                           " +
    ", ".join(f"{k} {v}" for k, v in cards_df["peak_risk"].value_counts().items()))

# ============================================================================
# PART H -- HISTORICAL EXAMPLES
# ============================================================================
say()
say("PART H -- BEHAVIOUR ON REAL DAYS")
say("-" * 100)
say(f"{'date':<12}{'rel':>8}{'P50':>9}{'P90':>9}{'actual':>9}{'in band?':>10}{'stress':>8}"
    f"{'level':>10}{'peak risk':>11}")
say("-" * 100)
for d in ["2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13", "2025-05-20", "2025-06-25"]:
    if d in cards_df.index:
        c = cards_df.loc[d]
        inb = "YES" if c["actual_peak_mw"] <= c["p90_mw"] else "NO"
        say(f"{d:<12}{c['reliability']:>8}{c['p50_mw']:>9,.0f}{c['p90_mw']:>9,.0f}"
            f"{c['actual_peak_mw']:>9,.0f}{inb:>10}{c['stress_score']:>8.1f}"
            f"{c['stress_level']:>10}{c['peak_risk']:>11}")
say("-" * 100)

say()
say("  The season peak block, 2025-06-12 23:00 (actual 8,392.6 MW):")
row = res_df.loc[KNOWN_PEAK_TS]
say(f"    reliability {row['reliability']} | P10 {row['p10']:,.0f} | P50 {row['p50']:,.0f} | "
    f"P90 {row['p90']:,.0f} (base band P90 was {row['p90_base']:,.0f})")
say(f"    actual is {'INSIDE' if row['actual'] <= row['p90'] else 'ABOVE'} the reliability-aware P90")

say()
say("  11 June 2025 (the known impaired day) reliability reasons:")
for rr in cards_df.loc["2025-06-11", "reliability_reasons"]:
    say(f"    - {rr}")
say("  11 June stress explanation:")
for rr in cards_df.loc["2025-06-11", "stress_reasons"]:
    say(f"    - {rr}")

# ============================================================================
# PART G -- BACKEND DECISION CARD
# ============================================================================
say()
say("PART G -- EXAMPLE BACKEND DECISION CARD (values only, no UI)")
say("-" * 100)
example = cards_df.loc["2025-06-12"].to_dict()
example["date"] = "2025-06-12"
say(json.dumps(example, indent=2, default=str))

# ============================================================================
# FIGURES
# ============================================================================
def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, linewidth=.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#5c655e", labelsize=9)

lo_t, hi_t = pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-14")
m = (res_df.index >= lo_t) & (res_df.index <= hi_t)
fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=SURFACE)
ax.fill_between(res_df.index[m], p10_rel[m], p90_rel[m], color=COL_A, alpha=.18, linewidth=0,
                label="P10-P90, reliability-aware")
ax.plot(res_df.index[m], p90_base[m], color=COL_A, linewidth=1, linestyle=(0, (4, 3)),
        label="P90, current band")
ax.plot(res_df.index[m], p50[m], color=COL_A, linewidth=1.4, label="P50")
ax.plot(res_df.index[m], y[m], color=INK, linewidth=1.6, label="Actual")
low_m = m & (st_arr != "HIGH")
if low_m.any():
    ax.scatter(res_df.index[low_m], y[low_m], s=14, color=COL_B, zorder=5,
               label="reliability below HIGH")
style(ax)
ax.set_title("Reliability-aware uncertainty: the band widens where telemetry was reconstructed",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Demand (MW)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(frameon=False, fontsize=9.5, ncols=5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT_DIR / "phase7_fig1_reliability_band.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 4.0), facecolor=SURFACE)
dd = pd.to_datetime(cards_df.index)
colors = {"NORMAL": COL_C, "WATCH": "#b9860f", "HIGH": COL_B, "CRITICAL": "#b5432c"}
ax.bar(dd, cards_df["stress_score"], color=[colors[l] for l in cards_df["stress_level"]], width=.75)
for thr, lab in [(50, "WATCH"), (75, "HIGH"), (90, "CRITICAL")]:
    ax.axhline(thr, color="#8a8f88", linewidth=1, linestyle=(0, (4, 4)))
    ax.text(dd[0], thr + 1, lab, fontsize=8.5, color="#5c655e")
style(ax)
ax.set_ylim(0, 100)
ax.set_title("Grid Stress Score by day (assumed illustrative capacity 9,000 MW)",
             fontsize=12, color=INK, loc="left", pad=12)
ax.set_ylabel("Stress score (0-100)", fontsize=9.5, color="#5c655e")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout(); fig.savefig(OUT_DIR / "phase7_fig2_stress.png", dpi=150, facecolor=SURFACE)
plt.close(fig)

# ============================================================================
# SAVE
# ============================================================================
res_df["stress_day"] = res_df.index.normalize().map(cards_df["stress_score"])
res_df.to_csv(OUT_DIR / "phase7_block_outputs.csv")
cards_df.to_csv(OUT_DIR / "phase7_decision_cards.csv")
(OUT_DIR / "phase7_decision_cards.json").write_text(json.dumps(cards, indent=2, default=str))
(OUT_DIR / "phase7_results.json").write_text(json.dumps(
    {"band_current": b_base, "band_reliability_aware": b_rel, "width_factors": factors,
     "assumed_capacity_mw": ASSUMED_CAPACITY_MW, "capacity_note": CAPACITY_NOTE,
     "weights": {"p50_util": W_P50_UTIL, "p90_util": W_P90_UTIL, "growth": W_GROWTH,
                 "heat": W_HEAT, "reliability": W_RELIABILITY},
     "state_counts_test": {st: int((test["reliability"] == st).sum())
                           for st in ["HIGH", "MEDIUM", "LOW"]}}, indent=2, default=str))
(OUT_DIR / "phase7_report.txt").write_text("\n".join(lines), encoding="utf-8")
say()
say("[SAVED] phase7_report.txt, phase7_results.json, phase7_decision_cards.json/.csv, "
    "phase7_block_outputs.csv, 2 figures")
