"""
PHASE 13 -- SHARED BACKEND MODULE
====================================
Consolidates the LOCKED forecasting pipeline and every downstream layer built
in Phases 7-12 into one importable module, so the dashboard (and anything
else) can call into a single source of truth instead of re-deriving it.

Nothing in this file changes any modeling logic, threshold, weight, or
formula that was approved in an earlier phase. Every function below is a
direct, line-for-line carry-over from the phase script that locked it:
  - data loading, feature engineering, reliability rule           -- Phase 7
  - OLS point forecast + IRLS quantile regression + calibration   -- Phase 7
  - Grid Stress Score (Combined / "C" formulation)                -- Phase 7/8 (locked in Phase 8)
  - peak_risk_rule() (thresholds Phase 7, reasons fixed Phase 10) -- Phase 10
  - Proof Mode replay() engine                                    -- Phase 9
  - What-If apply_temp_delta / solar_profile_mw / run_scenario    -- Phase 10
  - DISCOM modeled-allocation ratios and discom_estimate()        -- Phase 11
  - day_state() / recommend() advisory engine                     -- Phase 12

The one genuinely NEW piece of logic in this file is noted explicitly where
it appears (see discom_estimate(): reporting the NDMC+MES residual bucket's
MW contribution for display, which Phase 13's brief requires be shown).
It reuses the RATIO dict that was already locked in Phase 11/12 -- it does
not derive a new ratio, invent a capacity, or compute a relative-stress
figure for that bucket, because no individual reference exists for it
(that absence is itself a disclosed Phase 11 finding, not an oversight).

No network calls anywhere in this module. All inputs are local CSV files
already present in the data directory -- there is nothing "live" to fail
if the internet is unavailable.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ============================================================================
# LOCKED CONSTANTS
# ============================================================================
H = 96                                    # 15-min blocks per day
TRAIN_END, TEST_START, TEST_END = "2025-05-01", "2025-05-01", "2025-07-01"
ROLL_DAYS = 30
NOMINAL_COVERAGE = 0.80
KNOWN_PEAK_TS = pd.Timestamp("2025-06-12 23:00:00")
KNOWN_PEAK_MW = 8392.6

ASSUMED_CAPACITY_MW = 9000.0              # ILLUSTRATIVE / ASSUMED -- never presented as official
UTIL_LO, UTIL_HI = 0.70, 1.00
GROWTH_HI = 0.08
HEAT_LO, HEAT_HI = 30.0, 42.0
W_P50_UTIL, W_P90_UTIL, W_GROWTH, W_HEAT, W_RELIABILITY = 37.5, 37.5, 10, 10, 5   # Combined (C), locked Phase 8

# Locked DISCOM allocation ratios, Phase 11 (see phase11_discom_localization.py SOURCES for the
# dated, cited MW figures these ratios were derived from)
RATIO = {"BRPL": 3747.0 / 8423.0, "BYPL": 1832.0 / 8423.0, "TPDDL": 2331.0 / 8231.0}
RATIO["NDMC_MES_RESIDUAL"] = max(0.0, 1.0 - sum(RATIO.values()))
OWN_REF_2025 = {"BRPL": 4050.0, "BYPL": 1900.0, "TPDDL": 2562.0}
INDIVIDUALLY_MODELED_DISCOMS = ("BRPL", "BYPL", "TPDDL")   # NDMC+MES has no individual reference

TEMP_DELTA_MAX_ABS = 8.0                  # Phase 10 What-If sanity bound
TEMP_ABS_MIN, TEMP_ABS_MAX = 5.0, 48.0    # Delhi's plausible climatological range (IMD)
SOLAR_CAP_MIN, SOLAR_CAP_MAX = 0.0, 3000.0
SUNRISE_H, SUNSET_H = 5.5, 19.25          # Delhi June sunrise/sunset (IST clock hours, IMD almanac)

RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
REL_PEN = {"HIGH": 0.0, "MEDIUM": 0.5, "LOW": 1.0}

RULES = [
    ("peak_risk in {HIGH, CRITICAL} AND hours_to_peak <= 2",
     "imminent high/critical peak risk", "DEMAND RESPONSE",
     "Consider peak-period demand-response measures for the next few hours."),
    ("peak_risk in {HIGH, CRITICAL} AND hours_to_peak > 2",
     "high/critical peak risk, not yet imminent", "PREPARE",
     "Prepare additional reserve/capacity ahead of the forecast peak."),
    ("peak_risk == MODERATE", "moderate peak risk", "MONITOR",
     "Increase monitoring attention ahead of the forecast peak."),
    ("peak_risk == LOW", "no elevated peak risk", "MONITOR (routine)",
     "No immediate action beyond routine monitoring."),
    ("reliability == LOW", "critical input reconstructed", "DATA QUALITY WARNING",
     "Treat this forecast as low-confidence and prioritize confirmation of critical telemetry."),
    ("reliability == MEDIUM", "secondary input reconstructed/incomplete", "DATA QUALITY WARNING (lighter)",
     "Treat this forecast with moderate caution; some inputs were reconstructed or incomplete."),
    ("util50 < 0.88 AND util90 >= 0.88", "comfortable point forecast, uncomfortable upper band",
     "PREPARE", "Prepare for the upper-demand scenario rather than relying only on the central forecast."),
    ("max DISCOM relative_stress_p50 >= 0.90", "a specific utility modeled near its own reference",
     "MONITOR", "Increase monitoring attention on the stressed utility -- named separately from "
     "whichever utility carries the largest modeled MW contribution."),
    ("stress_level in {HIGH, CRITICAL}", "elevated system-wide Grid Stress Score", "CONSERVATION ADVISORY",
     "Issue a public conservation advisory encouraging reduced non-essential demand during the "
     "forecast peak window."),
]

DEFAULT_DATA_DIR = Path(os.environ.get("HACKU_DATA", Path(__file__).resolve().parent.parent / "data"))


class BackendError(Exception):
    """Raised when the backend cannot be built or a request cannot be honestly served (missing
    data files, corrupted cache, pre-flight check failure, requested date outside the test period).
    The dashboard catches this and shows a plain message -- never a Python traceback."""


# ============================================================================
# PIPELINE HELPERS (verbatim, Phase 7)
# ============================================================================
def _impute_nearest_day(s, max_days=14):
    out = s.copy()
    for k in range(1, max_days + 1):
        out = out.fillna(s.shift(H * k))
    return out.interpolate(limit_direction="both")


def _reliability_state(row):
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


def _fit_ols(tr, cols):
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs, tr["load_MW"].to_numpy(float), rcond=None)
    return lambda fr: np.column_stack([np.ones(len(fr)), (fr[cols].to_numpy(float) - mu) / sd]) @ coef


def _qreg(X, yv, tau, iters=60, eps=1.0):
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


def u(x):
    return float(np.clip((x / ASSUMED_CAPACITY_MW - UTIL_LO) / (UTIL_HI - UTIL_LO), 0, 1))


def level_of(score):
    return "CRITICAL" if score >= 90 else "HIGH" if score >= 75 else "WATCH" if score >= 50 else "NORMAL"


def prev_calendar_day_p50_peak(res_df, ref_time):
    """CANONICAL growth reference window (Phase 13 consistency fix; see
    phase13_consistency_fix_report.txt Section 3 for the evidence trail).

    Verified against phase8_risk_design_experiment.py -- the phase that actually locked the
    Combined ('C') Grid Stress Score formulation by evaluating it across all 61 test days: its
    own day-loop iterates `days = sorted(set(res_df.index.normalize()))` and tracks
    `prev_peak = r['p50']` from the immediately preceding iteration -- i.e. the previous
    CALENDAR day's P50 peak. That is the approved reference window. (Phase 9/10's replay() and
    run_scenario() previously used a different, 24h-window-ending-at-T reference instead; that
    was the source of the up-to-19-point discrepancy found during Phase 13 integration testing.
    This function is now the ONLY place either definition is computed.)

    ref_time may be any timestamp; the calendar day it falls on is used to look up the PRECEDING
    calendar day's max P50 in res_df. Returns None if that preceding day has no data (e.g. the
    first day of the test period)."""
    d = pd.Timestamp(ref_time).normalize()
    prev_day = res_df[res_df.index.normalize() == (d - pd.Timedelta(days=1))]
    return float(prev_day["p50"].max()) if len(prev_day) else None


def canonical_worst_reliability(res_df, ref_time):
    """CANONICAL reliability scope for the Grid Stress Score's reliability component ONLY
    (Phase 13 consistency fix). Same evidence trail as prev_calendar_day_p50_peak:
    phase8_risk_design_experiment.py's locked day-loop computes 'worst' as the worst reliability
    across ref_time's own CALENDAR DAY (`day = res_df[res_df.index.normalize()==d]`, then
    `max(day["reliability"], key=RANK)`) -- that calendar-day scope is reproduced here so
    day_state / replay / run_scenario always feed stress_score() an identically-scoped
    reliability value for a given peak timestamp.

    This does NOT change the 'reliability' field each function separately reports for its own
    badge/advisory/Proof-Mode-narrative purposes (window-scoped in replay/run_scenario, already
    calendar-day-scoped in day_state) -- those are unchanged, out-of-scope consumers per this
    fix's brief. It is used only for the score."""
    d = pd.Timestamp(ref_time).normalize()
    day = res_df[res_df.index.normalize() == d]
    if len(day) == 0:
        return None
    return max(day["reliability"], key=lambda s: RANK[s])


def stress_score(p50, p90, temp_corr, reliability, prev_peak_p50):
    """THE single canonical Grid Stress Score implementation (Combined / 'C' formulation, locked
    Phase 8). Every application path -- day_state, replay, run_scenario, and therefore every UI
    tab that displays a stress score -- calls THIS function. It does not look anything up itself;
    callers pass p50/p90/temp_corr/reliability for the state being scored, and prev_peak_p50 as
    the growth-reference peak (obtained from prev_calendar_day_p50_peak, the one canonical
    lookup). Identical inputs always produce an identical score -- there is no second
    implementation of this formula anywhere in this module."""
    util50, util90 = p50 / ASSUMED_CAPACITY_MW, p90 / ASSUMED_CAPACITY_MW
    c_p50, c_p90 = u(p50), u(p90)
    growth = 0.0 if prev_peak_p50 is None else (p50 - prev_peak_p50) / max(prev_peak_p50, 1)
    c_growth = float(np.clip(growth / GROWTH_HI, 0, 1))
    c_heat = float(np.clip((temp_corr - HEAT_LO) / (HEAT_HI - HEAT_LO), 0, 1))
    c_rel = REL_PEN[reliability]
    score = float(np.clip(W_P50_UTIL * c_p50 + W_P90_UTIL * c_p90 + W_GROWTH * c_growth +
                          W_HEAT * c_heat + W_RELIABILITY * c_rel, 0, 100))
    return dict(stress_score=round(score, 1), stress_level=level_of(score),
                util50=util50, util90=util90, growth=growth,
                c_p50=c_p50, c_p90=c_p90, c_growth=c_growth, c_heat=c_heat, c_rel=c_rel)


def peak_risk_rule(util50, util90, worst):
    """Locked Phase 7 thresholds, Phase 10 reason-generation fix (names the exact numeric
    condition that produced the tier)."""
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
    """MODELED ESTIMATE, not live feeder data (Phase 11). Returns BRPL/BYPL/TPDDL with both
    contribution (share of Delhi-wide modeled demand) and relative stress (vs each utility's own
    2025 reference peak). Also returns NDMC_MES_RESIDUAL's MW contribution ONLY, using the same
    locked residual ratio -- this bucket has no individual own-reference figure, so its
    relative_stress fields are explicitly None rather than invented."""
    out = {}
    for k in INDIVIDUALLY_MODELED_DISCOMS:
        p50v, p90v = RATIO[k] * delhi_p50, RATIO[k] * delhi_p90
        out[k] = {"p50_mw": round(float(p50v), 1), "p90_mw": round(float(p90v), 1),
                  "share": round(RATIO[k], 4),
                  "relative_stress_p50": round(float(p50v / OWN_REF_2025[k]), 3),
                  "relative_stress_p90": round(float(p90v / OWN_REF_2025[k]), 3)}
    rp50 = RATIO["NDMC_MES_RESIDUAL"] * delhi_p50
    rp90 = RATIO["NDMC_MES_RESIDUAL"] * delhi_p90
    out["NDMC_MES_RESIDUAL"] = {"p50_mw": round(float(rp50), 1), "p90_mw": round(float(rp90), 1),
                                "share": round(RATIO["NDMC_MES_RESIDUAL"], 4),
                                "relative_stress_p50": None, "relative_stress_p90": None}
    return out


def recommend(state):
    """Apply the locked Phase 12 rule table to one backend state; returns fired advisories.
    Only BRPL/BYPL/TPDDL are considered for the 'largest contribution' / 'highest relative
    stress' comparison -- identical scope to the already-approved Phase 12 behavior. The
    NDMC+MES residual bucket is display-only (see discom_estimate) and never drives an advisory,
    since it has no individual reference to compare against."""
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
    modeled = {k: v for k, v in d.items() if k in INDIVIDUALLY_MODELED_DISCOMS}
    largest = max(modeled, key=lambda k: modeled[k]["share"])
    stressed = max(modeled, key=lambda k: modeled[k]["relative_stress_p50"])
    if modeled[stressed]["relative_stress_p50"] >= 0.90:
        if stressed == largest:
            reason = (f"{stressed} carries both the largest modeled MW contribution "
                     f"({modeled[largest]['share']*100:.0f}%) and the highest modeled relative stress "
                     f"({modeled[stressed]['relative_stress_p50']*100:.0f}% of its own 2025 reference) at "
                     f"this event.")
        else:
            reason = (f"{largest} carries the largest modeled MW contribution "
                     f"({modeled[largest]['share']*100:.0f}%), but {stressed} shows the highest modeled "
                     f"relative stress ({modeled[stressed]['relative_stress_p50']*100:.0f}% of its own 2025 "
                     f"reference) -- contribution and relative stress point to different utilities.")
        advisories.append(dict(category="MONITOR",
            trigger="a specific utility modeled near its own reference",
            advisory=f"Increase monitoring attention on {stressed} (modeled relative stress "
                     f"{modeled[stressed]['relative_stress_p50']*100:.0f}%) -- do not assume it is "
                     f"automatically {largest}, the utility with the largest modeled MW contribution.",
            reason=reason))

    if state["stress_level"] in ("HIGH", "CRITICAL"):
        advisories.append(dict(category="CONSERVATION ADVISORY",
            trigger="elevated system-wide Grid Stress Score",
            advisory="Issue a public conservation advisory encouraging reduced non-essential demand "
                     "during the forecast peak window.",
            reason=f"Grid Stress Score is {state['stress_score']:.1f} ({state['stress_level']})."))

    return advisories


def window_reasons(win):
    """Phase 9: reliability reasons summarized across a whole 24h replay window."""
    n_lag24 = int(win["lag24_imputed"].sum())
    n_lag48 = int(win["lag48_imputed"].sum())
    min_comp = float(win["prev24_completeness"].min())
    n_wx = int(win["weather_corr_missing"].sum())
    reasons = []
    if n_lag24:
        reasons.append(f"critical 24h demand input reconstructed from the nearest available earlier "
                       f"day for {n_lag24} of {len(win)} blocks in this 24h window (telemetry gap)")
    if n_lag48:
        reasons.append(f"48h demand input reconstructed for {n_lag48} of {len(win)} blocks")
    if min_comp < 0.95:
        reasons.append(f"previous-24h demand telemetry only {min_comp*100:.0f}% complete at worst")
    if n_wx:
        reasons.append(f"weather bias correction unavailable for {n_wx} blocks")
    if not reasons:
        reasons = ["all critical demand inputs observed; previous 24h telemetry complete throughout"]
    return reasons[:3]


# ============================================================================
# BUILD
# ============================================================================
def build_backend(data_dir=None):
    """Load data, engineer features, fit the locked pipeline ONCE, and return a namespace
    exposing res_df (test-period P10/P50/P90/reliability), the fitted-model closures, and every
    downstream function (day_state, recommend, replay, run_scenario, discom_estimate, ...).
    Raises BackendError with a plain message on any missing/corrupted input or failed
    pre-flight check -- never lets a raw exception escape to the caller."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    required = ["load_data.csv", "delhi_weather_hourly.csv", "delhi_weather_forecast_day1.csv"]
    missing = [fn for fn in required if not (data_dir / fn).exists()]
    if missing:
        raise BackendError(f"Missing required data file(s) in {data_dir}: {', '.join(missing)}. "
                            f"The dashboard cannot build a forecast without these.")

    try:
        load = (pd.read_csv(data_dir / "load_data.csv", parse_dates=["timestamp"])
                  .sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp"))
        s_raw = load["load_MW"].resample("15min").mean()
        s15 = s_raw.interpolate(limit=4)
        observed = s15.notna()
        s_feat = _impute_nearest_day(s15)

        wx = (pd.read_csv(data_dir / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
                .drop_duplicates(subset="timestamp").set_index("timestamp"))
        wx15 = wx["temp_C"].resample("15min").interpolate(method="time")
        fcr = (pd.read_csv(data_dir / "delhi_weather_forecast_day1.csv", parse_dates=["timestamp"])
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

        rel = f.apply(_reliability_state, axis=1, result_type="expand")
        f = f.copy()
        f["reliability"] = rel[0]
        f["reliability_reasons"] = rel[1]

        train = f[f.index < TRAIN_END]
        test = f[(f.index >= TEST_START) & (f.index < TEST_END)]
        if len(test) == 0 or len(train) == 0:
            raise BackendError("Train/test split produced an empty partition -- check the data's date range.")

        calib_start = train.index.min() + pd.DateOffset(months=12)
        calib = train[train.index >= calib_start]
        bounds = np.linspace(0, len(calib), 5).astype(int)
        oof = []
        for k in range(4):
            va = calib.iloc[bounds[k]:bounds[k + 1]]
            trk = train[train.index < va.index.min()]
            p = _fit_ols(trk, COLS)(va)
            part = va[["load_MW", "reliability"]].copy()
            part["pred"] = p
            part["resid"] = part["load_MW"] - part["pred"]
            oof.append(part.join(va[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue",
                                     "temp_prevday_max", "roll24_max"]]))
        oof = pd.concat(oof)

        COND = ["pred", "sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max", "roll24_max"]
        Xo, yo = oof[COND].to_numpy(float), oof["resid"].to_numpy(float)
        q_lo_fn, q_hi_fn = _qreg(Xo, yo, 0.10), _qreg(Xo, yo, 0.90)

        qlo_o, qhi_o = q_lo_fn(Xo), q_hi_fn(Xo)
        factors, MIN_N = {}, 100
        for st_ in ["HIGH", "MEDIUM", "LOW"]:
            m = (oof["reliability"] == st_).to_numpy()
            if m.sum() < MIN_N:
                factors[st_] = None; continue
            chosen = None
            for k in np.arange(0.5, 6.001, 0.05):
                cov = ((yo[m] >= k * qlo_o[m]) & (yo[m] <= k * qhi_o[m])).mean()
                if cov >= NOMINAL_COVERAGE:
                    chosen = float(k); break
            factors[st_] = chosen if chosen is not None else 6.0
        fitted_vals = [v for v in factors.values() if v is not None]
        for st_ in factors:
            if factors[st_] is None:
                factors[st_] = max(fitted_vals)

        ols_full = _fit_ols(train, COLS)

        def predict_p50(frame):
            return ols_full(frame)

        def predict_bands(frame, p50_arr):
            ct = frame[["sin_hod", "cos_hod", "is_weekend", "temp_at_issue", "temp_prevday_max",
                       "roll24_max"]].copy()
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
                               "reliability_reasons": test["reliability_reasons"].to_numpy(),
                               "temp_corr": test["temp_corr"].to_numpy(),
                               "lag24_imputed": test["lag24_imputed"].to_numpy(),
                               "lag48_imputed": test["lag48_imputed"].to_numpy(),
                               "prev24_completeness": test["prev24_completeness"].to_numpy(),
                               "weather_corr_missing": test["weather_corr_missing"].to_numpy()},
                              index=test.index)

        if KNOWN_PEAK_TS not in res_df.index:
            raise BackendError("PRE-FLIGHT CHECK FAILED: known season peak timestamp is missing from "
                                "the rebuilt test set -- the data or pipeline has changed unexpectedly.")
    except BackendError:
        raise
    except Exception as exc:   # noqa: BLE001 -- deliberately broad: convert ANY pipeline failure
        raise BackendError(f"Backend pipeline failed to build: {exc}") from exc

    # ------------------------------------------------------------------
    # day_state / replay / run_scenario -- closures over the fitted pipeline above
    # ------------------------------------------------------------------
    def day_state(d, now_override=None):
        day = res_df[res_df.index.normalize() == pd.Timestamp(d)]
        if len(day) == 0:
            raise BackendError(f"No test-period data for {d} (test period is "
                                f"{TEST_START} to {TEST_END}, exclusive of the end date).")
        ipk = day["p50"].idxmax()
        r = day.loc[ipk]
        worst = max(day["reliability"], key=lambda s: RANK[s])
        if r["reliability"] == worst:
            rel_reasons = list(r["reliability_reasons"])
        else:
            match = day.loc[day["reliability"] == worst, "reliability_reasons"]
            rel_reasons = list(match.iloc[0]) if len(match) else list(r["reliability_reasons"])
        prev_peak = prev_calendar_day_p50_peak(res_df, ipk)
        sc = stress_score(r["p50"], r["p90"], r["temp_corr"], worst, prev_peak)
        util50, util90, growth = sc["util50"], sc["util90"], sc["growth"]
        score, level = sc["stress_score"], sc["stress_level"]
        risk, risk_reasons = peak_risk_rule(util50, util90, worst)
        discoms = discom_estimate(r["p50"], r["p90"])
        now = now_override if now_override is not None else (ipk - pd.Timedelta(hours=24))
        hours_to_peak = (ipk - now).total_seconds() / 3600.0
        contrib = [(f"P50 near capacity ({util50*100:.0f}%)", W_P50_UTIL * sc["c_p50"]),
                   (f"P90 near capacity ({util90*100:.0f}%)", W_P90_UTIL * sc["c_p90"]),
                   (f"day-over-day growth ({growth*100:+.1f}%)", W_GROWTH * sc["c_growth"]),
                   (f"heat ({r['temp_corr']:.1f}C corrected forecast)", W_HEAT * sc["c_heat"]),
                   (f"reliability {worst}", W_RELIABILITY * sc["c_rel"])]
        main_driver = max(contrib, key=lambda z: z[1])[0]
        return dict(date=str(d), peak_time=ipk, now=now, hours_to_peak=hours_to_peak,
                   p50=float(r["p50"]), p90=float(r["p90"]), util50=util50, util90=util90,
                   reliability=worst, reliability_reasons=rel_reasons,
                   stress_score=score, stress_level=level,
                   peak_risk=risk, peak_risk_reasons=risk_reasons,
                   main_driver=main_driver, discoms=discoms)

    def replay(issue_time_T, label=""):
        """Phase 9 Proof Mode: genuine historical replay. Issues the forecast at T using only
        information at-or-before T, then (separately) reveals the actual outcome for comparison.
        Formal no-leakage assertion included."""
        T = pd.Timestamp(issue_time_T)
        win_start, win_end = T + pd.Timedelta(minutes=15), T + pd.Timedelta(hours=24)
        expected_index = pd.date_range(win_start, win_end, freq="15min")
        win = res_df.reindex(expected_index).dropna(how="all")
        missing = [t for t in expected_index if t not in res_df.index]
        n_missing = len(missing)
        if n_missing > 6:
            raise BackendError(f"Replay window for T={T} is missing {n_missing} of {H} target blocks -- "
                                f"too incomplete for an honest replay.")
        if not (win.index - pd.Timedelta(hours=24) <= T).all():
            raise BackendError("LEAKAGE CHECK FAILED for this replay window -- refusing to serve it.")

        pred_ipk = win["p50"].idxmax()
        pr = win.loc[pred_ipk]
        act_ipk = win["actual"].idxmax()
        ar = win.loc[act_ipk]

        # 'worst' (window-scoped reliability across this replay's own 24h forecast window) is
        # UNCHANGED -- still used for the reported reliability field, peak_risk_rule, and
        # window_reasons below, exactly as Phase 9 approved. It is intentionally NOT touched by
        # this fix (Proof Mode's reliability narrative is out of scope; see the fix report).
        worst = max(win["reliability"], key=lambda s: RANK[s])
        # CANONICAL inputs to the Grid Stress Score ONLY (Phase 13 consistency fix): both the
        # growth reference and the reliability component are looked up via the SAME calendar-day
        # scope day_state() uses, keyed off the predicted peak's own timestamp -- so the same
        # forecast state always yields the same stress score regardless of which application
        # path (Overview/day_state vs Proof Mode/replay) produced it.
        prev_peak_p50 = prev_calendar_day_p50_peak(res_df, pred_ipk)
        worst_for_score = canonical_worst_reliability(res_df, pred_ipk)
        sc = stress_score(pr["p50"], pr["p90"], pr["temp_corr"], worst_for_score, prev_peak_p50)
        util50, util90 = sc["util50"], sc["util90"]
        score, level = sc["stress_score"], sc["stress_level"]
        risk, risk_reasons = peak_risk_rule(util50, util90, worst)
        rel_reasons = window_reasons(win)

        mape_full = float((np.abs(win["actual"] - win["p50"]) / win["actual"]).mean() * 100)
        peak_abs_err = float(pr["p50"] - ar["actual"])
        peak_pct_err = float(peak_abs_err / ar["actual"] * 100)
        inside_at_actual_peak = bool(ar["p10"] <= ar["actual"] <= ar["p90"])
        inside_at_pred_peak_time = bool(win.loc[pred_ipk, "p10"] <= win.loc[pred_ipk, "actual"] <= win.loc[pred_ipk, "p90"])
        p90_margin_at_actual_peak = float(ar["p90"] - ar["actual"])
        warning_hours = (act_ipk - T).total_seconds() / 3600.0

        return dict(label=label, T=T, window_start=win_start, window_end=win_end,
                    win=win, n_missing=n_missing, missing=missing,
                    pred_peak_mw=round(float(pr["p50"]), 1), pred_peak_time=pred_ipk,
                    p10_at_pred_peak=round(float(win.loc[pred_ipk, 'p10']), 1),
                    p90_at_pred_peak=round(float(win.loc[pred_ipk, 'p90']), 1),
                    util50=util50, util90=util90,
                    actual_peak_mw=round(float(ar["actual"]), 1), actual_peak_time=act_ipk,
                    peak_abs_err_mw=round(peak_abs_err, 1), peak_pct_err=round(peak_pct_err, 2),
                    mape_full_pct=round(mape_full, 2),
                    reliability=worst, reliability_reasons=rel_reasons,
                    stress_score=round(score, 1), stress_level=level,
                    peak_risk=risk, peak_risk_reasons=risk_reasons,
                    inside_p90_at_actual_peak=inside_at_actual_peak,
                    p90_margin_at_actual_peak_mw=round(p90_margin_at_actual_peak, 1),
                    warning_hours=round(warning_hours, 2))

    def apply_temp_delta(frame, delta):
        """Perturb ONLY the corrected forecast temperature (and features derived from it).
        temp_at_issue / temp_prevday_max / temp_prevday_mean are OBSERVED pre-issue-time
        temperature -- fixed historical fact -- and are deliberately left untouched."""
        fr = frame.copy()
        new_temp = (fr["ct"] + delta).clip(TEMP_ABS_MIN, TEMP_ABS_MAX)
        clipped = int((fr["ct"] + delta != new_temp).sum())
        fr["ct"] = new_temp
        fr["ccdh"] = np.clip(new_temp - 24, 0, None)
        fr["csq"] = new_temp ** 2
        fr["chx"] = np.clip(new_temp - 38, 0, None) ** 2
        return fr, new_temp.to_numpy(), clipped

    def solar_profile_mw(index, capacity_mw):
        """ASSUMPTION-BASED scenario arithmetic, not fitted from any data: half-sine daylight
        curve between SUNRISE_H and SUNSET_H."""
        hour = index.hour + index.minute / 60
        frac = np.clip((hour - SUNRISE_H) / (SUNSET_H - SUNRISE_H), 0, 1)
        shape = np.sin(np.pi * frac) * ((hour >= SUNRISE_H) & (hour <= SUNSET_H))
        return capacity_mw * shape

    def _score_window(win_p50, win_p90, win_temp_corr, worst_reliability, peak_idx, prev_peak_p50,
                       worst_reliability_for_score):
        """worst_reliability (window-scoped, Phase 10 approved) drives peak_risk_rule, exactly as
        before this fix. worst_reliability_for_score (canonical, calendar-day-scoped) drives ONLY
        the Grid Stress Score's reliability component, via the shared stress_score() function --
        see canonical_worst_reliability()."""
        pr50, pr90, tcorr = win_p50[peak_idx], win_p90[peak_idx], win_temp_corr[peak_idx]
        sc = stress_score(pr50, pr90, tcorr, worst_reliability_for_score, prev_peak_p50)
        util50, util90 = sc["util50"], sc["util90"]
        risk, risk_reasons = peak_risk_rule(util50, util90, worst_reliability)
        return dict(peak_mw=round(float(pr50), 1), p90_mw=round(float(pr90), 1),
                    util50=util50, util90=util90, stress_score=sc["stress_score"],
                    stress_level=sc["stress_level"],
                    peak_risk=risk, peak_risk_reasons=risk_reasons)

    def run_scenario(T, label="", temp_delta=0.0, solar_capacity_mw=None):
        """Phase 10 What-If engine: genuinely RECOMPUTES the forecast on perturbed feature
        copies using the SAME fitted model closures (ols_full / q_lo_fn / q_hi_fn) -- never
        retrains, never looks up a canned value. Solar is separate, disclosed scenario
        arithmetic subtracted from the recomputed gross forecast, not part of the ML model."""
        T = pd.Timestamp(T)
        win_start, win_end = T + pd.Timedelta(minutes=15), T + pd.Timedelta(hours=24)
        expected_index = pd.date_range(win_start, win_end, freq="15min")
        win_feat = test.reindex(expected_index).dropna(how="all")
        if len(win_feat) == 0:
            raise BackendError(f"No feature data available for a scenario window starting at T={T}.")
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
        # CANONICAL inputs to the Grid Stress Score ONLY (Phase 13 consistency fix): growth
        # reference and reliability-for-scoring both use the SAME calendar-day scope day_state()
        # uses, keyed off THIS scenario's own (possibly temperature-shifted) peak timestamp -- so
        # BASE and SCENARIO calls, and every other application path, always score an identical
        # forecast state identically. worst_rel (window-scoped) still drives peak_risk_rule only.
        prev_peak_p50 = prev_calendar_day_p50_peak(res_df, peak_time)
        worst_for_score = canonical_worst_reliability(res_df, peak_time)
        scored = _score_window(p50, p90, temp_used, worst_rel, peak_idx, prev_peak_p50, worst_for_score)
        discoms = discom_estimate(scored["peak_mw"], scored["p90_mw"])

        return dict(label=label, T=T, window=frame.index, notes=notes, temp_delta=temp_delta,
                    solar_capacity_mw=solar_capacity_mw,
                    gross_p50=gross_p50, gross_p10=gross_p10, gross_p90=gross_p90,
                    p50=p50, p10=p10, p90=p90, solar_mw=solar_mw, temp_used=temp_used,
                    peak_time=peak_time, worst_reliability=worst_rel, discoms=discoms, **scored)

    return SimpleNamespace(
        data_dir=data_dir, res_df=res_df, test=test, train=train, factors=factors,
        n_test_blocks=len(test),
        known_peak_p50=float(res_df.loc[KNOWN_PEAK_TS, "p50"]),
        known_peak_p90=float(res_df.loc[KNOWN_PEAK_TS, "p90"]),
        predict_p50=predict_p50, predict_bands=predict_bands,
        day_state=day_state, recommend=recommend, replay=replay,
        run_scenario=run_scenario, apply_temp_delta=apply_temp_delta,
        solar_profile_mw=solar_profile_mw, discom_estimate=discom_estimate,
        peak_risk_rule=peak_risk_rule, u=u, level_of=level_of,
        stress_score=stress_score, prev_calendar_day_p50_peak=prev_calendar_day_p50_peak,
        canonical_worst_reliability=canonical_worst_reliability,
        test_days=sorted(str(d.date()) for d in set(res_df.index.normalize())),
    )
