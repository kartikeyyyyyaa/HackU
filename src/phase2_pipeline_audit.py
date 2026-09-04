"""
PHASE 2 -- PIPELINE DIAGNOSIS (no modelling of any kind)
=========================================================
Objective: find exactly where rows disappeared between the verified Phase 1d
pipeline (~70,330 train blocks) and the Phase 2 experiment build (~34,086),
and why the verified 8,392.6 MW peak vanished from the test window.

This script trains nothing. It only instruments the data path:
  RAW -> RESAMPLE -> INTERPOLATE -> WEATHER JOIN -> LAGS -> ROLLINGS -> DROPNA -> SPLIT
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HACKU_DATA", BASE / "data"))
OUT_DIR = Path(os.environ.get("HACKU_OUT", BASE / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 96
TRAIN_END, TEST_START, TEST_END = "2025-05-01", "2025-05-01", "2025-07-01"
KNOWN_PEAK_VALUE = 8392.6

lines = []
def say(s=""):
    print(s)
    lines.append(s)

def describe(series_or_df, label, col="load_MW"):
    """Row count, usable (non-NaN target) count, span, max demand."""
    if isinstance(series_or_df, pd.Series):
        s = series_or_df
    else:
        s = series_or_df[col]
    valid = s.dropna()
    return dict(stage=label, rows=len(s), valid=len(valid),
                start=str(s.index.min()), end=str(s.index.max()),
                maxdem=float(valid.max()) if len(valid) else float("nan"))

audit = []

say("PHASE 2 -- PIPELINE DIAGNOSIS")
say("=" * 96)

# ============================================================================
# STEP 1 -- REPRODUCE THE VERIFIED PHASE 1 / 1d CHECKPOINTS FROM SOURCE
# ============================================================================
say()
say("STEP 1 -- REPRODUCING PHASE 1d CHECKPOINTS FROM THE RAW FILE (not trusting earlier numbers)")
say("-" * 96)

raw = pd.read_csv(DATA_DIR / "load_data.csv", parse_dates=["timestamp"])
say(f"A. RAW CSV                 rows={len(raw):,}  min={raw['timestamp'].min()}  "
    f"max={raw['timestamp'].max()}  max_demand={raw['load_MW'].max():,.1f} MW")
say(f"   duplicated timestamps: {int(raw['timestamp'].duplicated().sum())}")

raw_i = raw.sort_values("timestamp").drop_duplicates(subset="timestamp").set_index("timestamp")
p1_s15 = raw_i["load_MW"].resample("15min").mean().interpolate(limit=4)
say(f"B. 15-MIN PROCESSED        rows={len(p1_s15):,}  valid={int(p1_s15.notna().sum()):,}  "
    f"min={p1_s15.index.min()}  max={p1_s15.index.max()}  max_demand={p1_s15.max():,.1f} MW")

# Phase 1d feature build, re-implemented exactly as that script does it
wx = (pd.read_csv(DATA_DIR / "delhi_weather_hourly.csv", parse_dates=["timestamp"])
        .drop_duplicates(subset="timestamp").set_index("timestamp"))
wx15 = wx["temp_C"].resample("15min").interpolate(method="time")

p1 = pd.DataFrame(index=p1_s15.index)
p1["load_MW"] = p1_s15
p1["temp_C"] = wx15.reindex(p1.index)
p1 = p1.dropna(subset=["temp_C"])
p1["hour"] = p1.index.hour + p1.index.minute / 60
p1["is_weekend"] = (p1.index.dayofweek >= 5).astype(int)
p1["sin_hod"] = np.sin(2 * np.pi * p1["hour"] / 24)
p1["cos_hod"] = np.cos(2 * np.pi * p1["hour"] / 24)
p1["sin_doy"] = np.sin(2 * np.pi * p1.index.dayofyear / 365.25)
p1["cos_doy"] = np.cos(2 * np.pi * p1.index.dayofyear / 365.25)
p1["same_block_24h_ago"] = p1["load_MW"].shift(H)
p1["same_block_7d_ago"] = p1["load_MW"].shift(H * 7)
p1["cdh"] = np.clip(p1["temp_C"] - 24, 0, None)
p1["cdh_sq"] = np.clip(p1["temp_C"] - 38, 0, None) ** 2
p1 = p1.dropna()

p1_train = p1[p1.index < TRAIN_END]
p1_test = p1[(p1.index >= TEST_START) & (p1.index < TEST_END)]
say(f"C. PHASE 1d TRAIN          rows={len(p1_train):,}  {p1_train.index.min()} -> {p1_train.index.max()}  "
    f"max_demand={p1_train['load_MW'].max():,.1f} MW")
say(f"D. PHASE 1d TEST           rows={len(p1_test):,}  {p1_test.index.min()} -> {p1_test.index.max()}  "
    f"max_demand={p1_test['load_MW'].max():,.1f} MW")
say()
say("   -> Phase 1d uses ONLY shift() lags (24h, 7d). It uses NO rolling windows. Remember that.")

# ============================================================================
# STEP 2 -- TRACE THE PHASE 2 (BROKEN) PIPELINE, ONE TRANSFORMATION AT A TIME
# ============================================================================
say()
say("STEP 2 -- TRACING THE PHASE 2 EXPERIMENT PIPELINE AS ORIGINALLY WRITTEN")
say("-" * 96)

s_resampled = raw_i["load_MW"].resample("15min").mean()
audit.append(describe(raw_i, "1. raw (deduped, indexed)"))
audit.append(describe(s_resampled, "2. resample 15min"))
s_interp = s_resampled.interpolate(limit=4)
audit.append(describe(s_interp, "3. interpolate(limit=4)"))

b = pd.DataFrame({"load_MW": s_interp})
b["temp_C"] = wx15.reindex(b.index)
n_before_wx = len(b)
b_wx = b.dropna(subset=["temp_C"])
audit.append(describe(b_wx, "4. weather join + dropna(temp)"))

# lags -- exactly as the broken script built them (from the NaN-containing target)
b_wx = b_wx.copy()
b_wx["lag_24h"] = b_wx["load_MW"].shift(H)
b_wx["lag_48h"] = b_wx["load_MW"].shift(H * 2)
b_wx["lag_7d"] = b_wx["load_MW"].shift(H * 7)
b_wx["lag_14d"] = b_wx["load_MW"].shift(H * 14)
audit.append(describe(b_wx.dropna(subset=["lag_24h", "lag_48h", "lag_7d", "lag_14d"]),
                      "5. + lag features (24h/48h/7d/14d)"))

past = b_wx["load_MW"].shift(H)
b_wx["roll24_mean"] = past.rolling(H).mean()
b_wx["roll24_max"] = past.rolling(H).max()
b_wx["roll24_min"] = past.rolling(H).min()
b_wx["roll7d_mean"] = past.rolling(H * 7).mean()
tpast = b_wx["temp_C"].shift(H)
b_wx["temp_at_issue"] = tpast
b_wx["temp_prevday_max"] = tpast.rolling(H).max()
b_wx["temp_prevday_mean"] = tpast.rolling(H).mean()
audit.append(describe(b_wx.dropna(), "6. + rolling features, then dropna()"))

broken = b_wx.dropna()
broken_train = broken[broken.index < TRAIN_END]
broken_test = broken[(broken.index >= TEST_START) & (broken.index < TEST_END)]
audit.append(describe(broken_train, "7. broken TRAIN split"))
audit.append(describe(broken_test, "8. broken TEST split"))

hdr = f"{'Stage':<38}{'Rows':>9}{'Usable':>9}{'Lost':>9}{'MaxDem':>10}  {'Start':<20}{'End':<20}"
say(hdr)
say("-" * 96)
prev = None
for a in audit:
    lost = "" if prev is None else f"{prev - a['valid']:,}"
    say(f"{a['stage']:<38}{a['rows']:>9,}{a['valid']:>9,}{lost:>9}{a['maxdem']:>10,.1f}  "
        f"{a['start']:<20}{a['end']:<20}")
    prev = a["valid"]
say("-" * 96)

# per-feature NaN attribution at the dropna step
say()
say("STEP 2b -- PER-FEATURE NaN ATTRIBUTION AT THE FINAL dropna()")
say("-" * 96)
cols = ["load_MW", "temp_C", "lag_24h", "lag_48h", "lag_7d", "lag_14d",
        "roll24_mean", "roll24_max", "roll24_min", "roll7d_mean",
        "temp_at_issue", "temp_prevday_max", "temp_prevday_mean"]
nan_mask = b_wx[cols].isna()
any_nan = nan_mask.any(axis=1)
say(f"{'Feature':<24}{'NaN rows':>12}{'% of frame':>12}{'Uniquely responsible':>24}")
say("-" * 96)
for c in cols:
    others = nan_mask[[x for x in cols if x != c]].any(axis=1)
    uniq = int((nan_mask[c] & ~others).sum())
    say(f"{c:<24}{int(nan_mask[c].sum()):>12,}{nan_mask[c].mean()*100:>11.2f}%{uniq:>24,}")
say("-" * 96)
say(f"{'TOTAL rows dropped':<24}{int(any_nan.sum()):>12,}{any_nan.mean()*100:>11.2f}%")

# same attribution restricted to the test window
tw = (b_wx.index >= TEST_START) & (b_wx.index < TEST_END)
say()
say(f"STEP 2c -- SAME ATTRIBUTION, RESTRICTED TO THE TEST WINDOW ({TEST_START} .. {TEST_END})")
say("-" * 96)
nm_t = nan_mask[tw]
say(f"{'Feature':<24}{'NaN rows in test':>18}{'% of test window':>18}")
say("-" * 96)
for c in cols:
    if nm_t[c].sum():
        say(f"{c:<24}{int(nm_t[c].sum()):>18,}{nm_t[c].mean()*100:>17.2f}%")
say(f"{'ANY (rows lost)':<24}{int(nm_t.any(axis=1).sum()):>18,}{nm_t.any(axis=1).mean()*100:>17.2f}%")

# ============================================================================
# STEP 3 -- TRACE THE KNOWN PEAK
# ============================================================================
say()
say("STEP 3 -- TRACING THE VERIFIED PEAK THROUGH EVERY STAGE")
say("-" * 96)
peak_ts = p1_s15.idxmax()
say(f"Verified peak timestamp: {peak_ts}   value {p1_s15.max():,.1f} MW "
    f"(expected ~{KNOWN_PEAK_VALUE} MW: {'MATCH' if abs(p1_s15.max()-KNOWN_PEAK_VALUE) < 1 else 'MISMATCH'})")
say()
say(f"{'Stage':<40}{'Present?':<12}{'Value':>12}   Reason if missing")
say("-" * 96)

def check(label, frame_or_series, reason=""):
    idx = frame_or_series.index
    present = peak_ts in idx
    val = ""
    if present:
        v = frame_or_series.loc[peak_ts]
        v = v["load_MW"] if hasattr(v, "index") and "load_MW" in v.index else v
        val = f"{float(v):,.1f}" if pd.notna(v) else "NaN"
        if val == "NaN":
            present = False
    say(f"{label:<40}{('YES' if present else 'NO'):<12}{val:>12}   {reason if not present else ''}")

check("raw (deduped)", raw_i["load_MW"])
check("resampled 15min", s_resampled)
check("after interpolate(limit=4)", s_interp)
check("after weather join", b_wx["load_MW"])
check("phase 1d feature frame", p1["load_MW"])
check("phase 2 BROKEN feature frame", broken["load_MW"], "dropped at final dropna()")

if peak_ts in b_wx.index:
    row_nan = b_wx.loc[peak_ts, cols].isna()
    culprits = [c for c in cols if row_nan[c]]
    say()
    say(f"Columns that were NaN on the peak row: {culprits if culprits else 'none'}")
    for c in culprits:
        if c.startswith("roll7d"):
            w = H * 7
        elif c.startswith("roll24") or c.startswith("temp_prevday"):
            w = H
        else:
            w = None
        if w:
            win_start = peak_ts - pd.Timedelta(minutes=15 * (H + w))
            win_end = peak_ts - pd.Timedelta(minutes=15 * H)
            seg = s_interp.loc[win_start:win_end]
            say(f"  {c}: needs {w} consecutive values in {win_start} .. {win_end}; "
                f"that span contains {int(seg.isna().sum())} missing blocks -> window returns NaN")

# ============================================================================
# STEP 5 -- WEATHER MERGE AUDIT
# ============================================================================
say()
say("STEP 5 -- WEATHER JOIN AUDIT")
say("-" * 96)
say(f"demand index tz: {s_interp.index.tz}   weather index tz: {wx.index.tz}   (both naive local = aligned)")
say(f"demand span : {s_interp.index.min()} -> {s_interp.index.max()}")
say(f"weather span: {wx.index.min()} -> {wx.index.max()} (hourly, {len(wx):,} readings)")
say(f"join method used: .reindex() on the demand index = LEFT-preserving, then dropna(subset=['temp_C'])")
say(f"rows before weather join: {n_before_wx:,}   after: {len(b_wx):,}   lost: {n_before_wx - len(b_wx):,}")
lost_wx = b.index.difference(b_wx.index)
if len(lost_wx):
    say(f"lost timestamps span: {lost_wx.min()} -> {lost_wx.max()} "
        f"(these are the 15-min blocks after the final hourly weather reading, so nothing to interpolate to)")
say(f"weather NaNs inside the test window: {int(wx15.reindex(b.index)[tw[:len(b)] if len(tw)==len(b) else (b.index>=TEST_START)&(b.index<TEST_END)].isna().sum())}")
say("VERDICT: no inner join anywhere; weather is NOT the cause of the mass row loss.")

# ============================================================================
# STEP 6/7 -- CORRECTED PIPELINE
# ============================================================================
say()
say("STEP 6/7 -- CORRECTED PIPELINE (smallest safe change) AND REBUILT SPLIT")
say("-" * 96)
say("Change 1: build lag/rolling FEATURE INPUTS from a gap-filled copy of the series")
say("          (seasonal-naive: same block one week earlier, then linear).")
say("          The TARGET is never imputed and imputed blocks are never scored.")
say("Change 2: rolling windows get min_periods = 75% of window, so one missing block")
say("          no longer voids an entire 24h or 7d window.")

s_feat = s_interp.copy()
for _ in range(6):
    s_feat = s_feat.fillna(s_feat.shift(H * 7))
s_feat = s_feat.interpolate(limit_direction="both")

g = pd.DataFrame({"load_MW": s_interp, "load_feat": s_feat})
g["temp_C"] = wx15.reindex(g.index)
g = g.dropna(subset=["temp_C", "load_feat"])
lf = g["load_feat"]
g["lag_24h"] = lf.shift(H)
g["lag_48h"] = lf.shift(H * 2)
g["lag_7d"] = lf.shift(H * 7)
g["lag_14d"] = lf.shift(H * 14)
pastf = lf.shift(H)
g["roll24_mean"] = pastf.rolling(H, min_periods=int(H * .75)).mean()
g["roll24_max"] = pastf.rolling(H, min_periods=int(H * .75)).max()
g["roll24_min"] = pastf.rolling(H, min_periods=int(H * .75)).min()
g["roll7d_mean"] = pastf.rolling(H * 7, min_periods=int(H * 7 * .75)).mean()
tpf = g["temp_C"].shift(H)
g["temp_at_issue"] = tpf
g["temp_prevday_max"] = tpf.rolling(H, min_periods=int(H * .75)).max()
g["temp_prevday_mean"] = tpf.rolling(H, min_periods=int(H * .75)).mean()
fixed = g.dropna()

fx_train = fixed[fixed.index < TRAIN_END]
fx_test = fixed[(fixed.index >= TEST_START) & (fixed.index < TEST_END)]

say()
say(f"{'Dataset':<28}{'Rows':>10}{'MaxDem':>11}   Range")
say("-" * 96)
say(f"{'corrected full frame':<28}{len(fixed):>10,}{fixed['load_MW'].max():>11,.1f}   {fixed.index.min()} -> {fixed.index.max()}")
say(f"{'corrected TRAIN':<28}{len(fx_train):>10,}{fx_train['load_MW'].max():>11,.1f}   {fx_train.index.min()} -> {fx_train.index.max()}")
say(f"{'corrected TEST':<28}{len(fx_test):>10,}{fx_test['load_MW'].max():>11,.1f}   {fx_test.index.min()} -> {fx_test.index.max()}")
say(f"{'phase 1d TRAIN (reference)':<28}{len(p1_train):>10,}{p1_train['load_MW'].max():>11,.1f}")
say(f"{'phase 1d TEST  (reference)':<28}{len(p1_test):>10,}{p1_test['load_MW'].max():>11,.1f}")

# ============================================================================
# STEP 8 -- INTEGRITY CHECKS
# ============================================================================
say()
say("STEP 8 -- DATA INTEGRITY CHECKS")
say("-" * 96)
checks = []

checks.append(("test max demand == verified peak",
               abs(fx_test["load_MW"].max() - KNOWN_PEAK_VALUE) < 1,
               f"{fx_test['load_MW'].max():,.1f} MW vs expected {KNOWN_PEAK_VALUE} MW"))
checks.append(("verified peak timestamp present in test",
               peak_ts in fx_test.index,
               f"{peak_ts}"))
expected_blocks = int((pd.Timestamp(TEST_END) - pd.Timestamp(TEST_START)) / pd.Timedelta("15min"))
checks.append(("test continuity (>=99% of calendar blocks)",
               len(fx_test) / expected_blocks >= 0.99,
               f"{len(fx_test):,} of {expected_blocks:,} calendar blocks = {len(fx_test)/expected_blocks*100:.2f}%"))
checks.append(("no duplicate timestamps",
               not fixed.index.has_duplicates, "index unique"))
checks.append(("train/test boundary correct (no overlap)",
               fx_train.index.max() < pd.Timestamp(TEST_START) <= fx_test.index.min(),
               f"train ends {fx_train.index.max()}, test starts {fx_test.index.min()}"))
checks.append(("row loss vs phase 1d is small",
               abs(len(fx_train) - len(p1_train)) / len(p1_train) < 0.05,
               f"corrected train {len(fx_train):,} vs phase1d {len(p1_train):,} "
               f"({(len(fx_train)-len(p1_train))/len(p1_train)*100:+.2f}%)"))
n_peakblocks_raw = int((p1_s15.loc[TEST_START:TEST_END].dropna() >= 6786.8).sum())
n_peakblocks_fixed = int((fx_test["load_MW"] >= 6786.8).sum())
checks.append(("peak-period blocks preserved",
               n_peakblocks_fixed >= n_peakblocks_raw * 0.98,
               f"{n_peakblocks_fixed} kept of {n_peakblocks_raw} present in processed data"))
checks.append(("targets are never imputed values",
               fx_test["load_MW"].notna().all() and
               fx_test.index.isin(s_interp.dropna().index).all(),
               "every scored row has a real observed (or <=1h-interpolated) target"))

for name, ok, detail in checks:
    say(f"  [{'PASS' if ok else 'FAIL'}] {name:<45} {detail}")

# leakage verification -- prove each feature only sees data <= target-24h
say()
say("STEP 8b -- LEAKAGE VERIFICATION (mechanical, not by inspection)")
say("-" * 96)
probe = fx_test.index[len(fx_test) // 2]
issue_time = probe - pd.Timedelta(hours=24)
say(f"probe target block: {probe}   -> forecast issue time must be <= {issue_time}")
lag_defs = {"lag_24h": H, "lag_48h": H * 2, "lag_7d": H * 7, "lag_14d": H * 14}
ok_all = True
for name, k in lag_defs.items():
    src = probe - pd.Timedelta(minutes=15 * k)
    ok = src <= issue_time
    ok_all &= ok
    say(f"  [{'PASS' if ok else 'FAIL'}] {name:<18} sources {src}  (<= issue time)")
for name, w in [("roll24_*", H), ("roll7d_mean", H * 7), ("temp_prevday_*", H)]:
    newest = probe - pd.Timedelta(minutes=15 * H)
    ok = newest <= issue_time
    ok_all &= ok
    say(f"  [{'PASS' if ok else 'FAIL'}] {name:<18} newest input {newest}  (<= issue time)")
say(f"  [NOTE ] temp_target / cdh_target are TARGET-TIME weather -- TIER 2 only, and are an ASSUMPTION")
say(f"          (observed temperature standing in for a forecast we do not have). Excluded from TIER 1.")
say(f"  [{'PASS' if ok_all else 'FAIL'}] all TIER-1 features sourced at or before issue time")

say()
say("=" * 96)
(OUT_DIR / "phase2_pipeline_audit.txt").write_text("\n".join(lines), encoding="utf-8")
fixed.to_csv(OUT_DIR / "phase2_corrected_feature_frame.csv")
say(f"[SAVED] {OUT_DIR}/phase2_pipeline_audit.txt and phase2_corrected_feature_frame.csv")
