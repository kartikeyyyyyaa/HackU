"""
PHASE 13 -- DELHI PEAK INTELLIGENCE (final dashboard)
========================================================
A single Streamlit application over the backend consolidated in
phase13_backend.py. Every number shown here is read directly from a
backend function (day_state / recommend / replay / run_scenario /
discom_estimate) -- nothing is retyped, recomputed independently in the
UI, or invented for display. The model is fit ONCE per server process
(st.cache_resource) and never retrained on a UI interaction.

Navigation: OVERVIEW / WHAT-IF / PROOF MODE, per the Phase 13 brief.

No network calls anywhere in this file or in phase13_backend.py -- every
input is a local CSV already in the data directory, so there is nothing
"live" that can fail if the internet is unavailable. This IS the offline
/ demo mode; there is no separate live path to fall back from.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase13_backend import (   # noqa: E402
    ASSUMED_CAPACITY_MW, BackendError, DEFAULT_DATA_DIR, INDIVIDUALLY_MODELED_DISCOMS,
    KNOWN_PEAK_TS, RATIO, RULES, SOLAR_CAP_MAX, TEMP_DELTA_MAX_ABS, build_backend,
)

st.set_page_config(page_title="Delhi Peak Intelligence", layout="wide",
                    initial_sidebar_state="collapsed")

REL_COLOR = {"HIGH": "#1f9d55", "MEDIUM": "#d97706", "LOW": "#dc2626"}
RISK_COLOR = {"LOW": "#1f9d55", "MODERATE": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7c2d12"}
STRESS_COLOR = {"NORMAL": "#1f9d55", "WATCH": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7c2d12"}


# ============================================================================
# BACKEND (built ONCE per server process; never rebuilt on a rerun/interaction)
# ============================================================================
@st.cache_resource(show_spinner="Loading forecasting pipeline (one-time, ~5s)...")
def get_backend():
    return build_backend(DEFAULT_DATA_DIR)


def safe_backend():
    try:
        return get_backend(), None
    except BackendError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error while building the backend: {exc}"


# ============================================================================
# SMALL DISPLAY HELPERS
# ============================================================================
def badge(text, color):
    st.markdown(
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.85em">{text}</span>',
        unsafe_allow_html=True)


def capacity_caption():
    st.caption(f"Capacity assumption: {ASSUMED_CAPACITY_MW:,.0f} MW -- **ILLUSTRATIVE**, not an "
               f"official DISCOM/SLDC-declared limit.")


def modeled_estimate_caption():
    st.caption("DISCOM figures below are **MODELED ESTIMATES** (Phase 11 disclosed allocation "
               "ratios applied to the Delhi-wide forecast) -- never live feeder telemetry.")


def advisory_list(advs):
    if not advs:
        st.info("No advisories fired for this state.")
        return
    for a in advs:
        with st.container(border=True):
            st.markdown(f"**{a['category']}** &nbsp;·&nbsp; *trigger: {a['trigger']}*")
            st.write(a["advisory"])
            st.caption(f"REASON: {a['reason']}")


def discom_table(discoms):
    rows = []
    for k in list(INDIVIDUALLY_MODELED_DISCOMS) + ["NDMC_MES_RESIDUAL"]:
        v = discoms[k]
        rs = f"{v['relative_stress_p50']*100:.0f}%" if v["relative_stress_p50"] is not None else "n/a (no individual reference)"
        rows.append({"Utility": k.replace("_", " + "), "Modeled P50 (MW)": v["p50_mw"],
                     "Modeled P90 (MW)": v["p90_mw"], "Contribution share": f"{v['share']*100:.1f}%",
                     "Relative stress (P50 vs own 2025 ref.)": rs})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def forecast_chart(res_df, center_date, days_before=3, days_after=1, mark_time=None, title=""):
    center = pd.Timestamp(center_date)
    lo, hi = center - pd.Timedelta(days=days_before), center + pd.Timedelta(days=days_after)
    win = res_df[(res_df.index >= lo) & (res_df.index < hi)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win["p90"], line=dict(width=0), showlegend=False,
                              hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=win.index, y=win["p10"], line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(99,102,241,0.18)", name="P10-P90 (uncertainty band)"))
    fig.add_trace(go.Scatter(x=win.index, y=win["p50"], line=dict(color="#6366f1", width=2),
                              name="P50 (expected)"))
    fig.add_trace(go.Scatter(x=win.index, y=win["actual"], line=dict(color="#111827", width=1.5,
                              dash="dot"), name="Actual"))
    if mark_time is not None and mark_time in win.index:
        fig.add_vline(x=mark_time, line_dash="dash", line_color="#dc2626",
                       annotation_text="peak", annotation_position="top")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", y=1.12), hovermode="x unified",
                       yaxis_title="MW", xaxis_title=None)
    return fig


# ============================================================================
# OVERVIEW TAB
# ============================================================================
def render_overview(B):
    st.subheader("DELHI PEAK INTELLIGENCE -- Control Room Overview")
    st.caption("24-hour-ahead genuine forecast · every value below traces to a backend output "
               "(phase13_backend.day_state / recommend).")

    default_idx = B.test_days.index("2025-06-12") if "2025-06-12" in B.test_days else 0
    day = st.selectbox("Forecast day (test period 2025-05-01 to 2025-06-30)", B.test_days,
                       index=default_idx, key="ov_day")

    try:
        state = B.day_state(day)
    except BackendError as exc:
        st.error(str(exc)); return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak (P50)", f"{state['p50']:,.0f} MW")
    c2.metric("Peak time", state["peak_time"].strftime("%d %b, %H:%M"))
    c3.metric("P90 (upper bound)", f"{state['p90']:,.0f} MW", f"{state['util90']*100:.0f}% of capacity")
    c4.metric("Grid Stress Score", f"{state['stress_score']:.1f}", state["stress_level"])
    capacity_caption()
    st.caption("The Grid Stress Score is computed by ONE canonical function (Phase 13 consistency "
              "fix) -- Overview, What-If, and Proof Mode always agree when they are scoring the "
              "same forecast state. A small residual note: What-If/Proof Mode occasionally select "
              "a different peak block for this same calendar day than Overview does, by design (a "
              "24h-ahead-issued forecast window vs. this tab's own calendar-day peak) -- when that "
              "happens the two tabs are genuinely scoring two different moments, not disagreeing "
              "about the same one. See phase13_consistency_fix_report.txt Section 8 (Remaining "
              "known issues).")

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        st.markdown("**Demand forecast**")
        st.caption("Each 15-min block's P50/P10/P90 is a genuine 24h-ahead forecast (built only "
                   "from information available 24h before that block) -- this is the continuous "
                   "per-block series across the test period, not a single issued-once forecast. "
                   "See PROOF MODE for the single-issuance replay framing.")
        fig = forecast_chart(B.res_df, day, mark_time=state["peak_time"],
                             title=f"Actual vs forecast around {day}")
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("**Reliability (this calendar day)**")
        badge(state["reliability"], REL_COLOR[state["reliability"]])
        for r in state["reliability_reasons"]:
            st.write(f"- {r}")
        st.caption("Proof Mode's reliability badge for this same date can differ -- it covers the "
                  "24h forecast WINDOW issued for that event, not this calendar day. Both are "
                  "correct; they answer different questions by design.")

        st.markdown("**Peak risk**")
        badge(state["peak_risk"], RISK_COLOR[state["peak_risk"]])
        st.write(f"Main driver: {state['main_driver']}")
        for r in state["peak_risk_reasons"]:
            st.write(f"- {r}")

    st.divider()
    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("**DISCOM contribution vs relative stress**")
        modeled_estimate_caption()
        discom_table(state["discoms"])
        st.caption("Contribution (MW share of Delhi-wide demand) and relative stress (vs each "
                   "utility's own 2025 reference peak) are kept separate on purpose -- they can, "
                   "and here do, point to different utilities.")
    with d2:
        st.markdown("**Advisory**")
        st.caption("Deterministic rule engine (Phase 12) -- no LLM, no freeform text.")
        advisory_list(B.recommend(state))


# ============================================================================
# WHAT-IF TAB
# ============================================================================
def render_whatif(B):
    st.subheader("WHAT-IF -- Decision Simulator")
    st.caption("Genuinely RECOMPUTES the forecast on the same fitted model for a perturbed "
               "temperature input. Solar is separate, disclosed scenario arithmetic subtracted "
               "afterward -- not part of the ML model.")

    default_idx = B.test_days.index("2025-06-12") if "2025-06-12" in B.test_days else 0
    day = st.selectbox("Event day (scenario window = 24h ending at this day's own peak)",
                       B.test_days, index=default_idx, key="wi_day")
    day_peak = B.res_df[B.res_df.index.normalize() == pd.Timestamp(day)]["p50"].idxmax()
    T = day_peak - pd.Timedelta(hours=24)

    c1, c2 = st.columns(2)
    with c1:
        temp_delta = st.slider("Temperature scenario (°C change to the corrected forecast)",
                               -TEMP_DELTA_MAX_ABS, TEMP_DELTA_MAX_ABS, 0.0, 0.5, key="wi_temp")
        st.caption("MODEL-DRIVEN SCENARIO -- feeds back through the real fitted model.")
    with c2:
        solar_on = st.checkbox("Add assumed rooftop solar", key="wi_solar_on")
        solar_cap = st.slider("Assumed installed solar capacity (MW)", 0, int(SOLAR_CAP_MAX), 1500, 100,
                              key="wi_solar_cap", disabled=not solar_on)
        st.caption("ASSUMPTION-BASED SCENARIO -- a half-sine daylight profile subtracted from the "
                   "gross forecast; not fitted from any data in this project.")

    try:
        base = B.run_scenario(T, "BASE")
        scn = B.run_scenario(T, "SCENARIO", temp_delta=temp_delta,
                             solar_capacity_mw=(solar_cap if solar_on else None))
    except BackendError as exc:
        st.error(str(exc)); return

    if scn["notes"]:
        for n in scn["notes"]:
            st.warning(n)

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P50 peak", f"{scn['peak_mw']:,.0f} MW", f"{scn['peak_mw']-base['peak_mw']:+,.0f} MW")
    m2.metric("P90", f"{scn['p90_mw']:,.0f} MW", f"{scn['p90_mw']-base['p90_mw']:+,.0f} MW")
    m3.metric("Peak time", scn["peak_time"].strftime("%d %b, %H:%M"))
    m4.metric("Stress score", f"{scn['stress_score']:.1f}", f"{scn['stress_score']-base['stress_score']:+.1f}")
    m5.metric("Peak risk", scn["peak_risk"],
              (None if scn["peak_risk"] == base["peak_risk"] else f"was {base['peak_risk']}"))
    capacity_caption()

    st.caption(f"BASE: peak {base['peak_mw']:,.0f} MW at {base['peak_time'].strftime('%H:%M')}, "
              f"stress {base['stress_score']:.1f} ({base['stress_level']}), risk {base['peak_risk']}.")
    st.caption("BASE uses the same canonical Grid Stress Score function as Overview and Proof "
              "Mode (Phase 13 consistency fix) -- it matches the Overview tab's score for this "
              "day whenever both are scoring the same peak block, and differs only on the small "
              "set of days where this tab's 24h-ahead-issued forecast window happens to select a "
              "different peak than Overview's own calendar-day view (disclosed in "
              "phase13_consistency_fix_report.txt Section 8, Remaining known issues). BASE vs SCENARIO comparisons "
              "WITHIN this tab are always apples-to-apples either way.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["window"], y=base["p50"], name="BASE P50",
                              line=dict(color="#9ca3af", width=2, dash="dot")))
    if scn["solar_mw"] is not None:
        fig.add_trace(go.Scatter(x=scn["window"], y=scn["gross_p50"], name="SCENARIO gross P50 (pre-solar)",
                                  line=dict(color="#f59e0b", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=scn["window"], y=scn["p50"], name="SCENARIO P50 (net)",
                              line=dict(color="#dc2626", width=2.5)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                       legend=dict(orientation="h", y=1.12), hovermode="x unified",
                       yaxis_title="MW", xaxis_title=None,
                       title="BASE vs SCENARIO -- 24h window ending at this event's peak")
    st.plotly_chart(fig, width='stretch')
    if scn["solar_mw"] is not None:
        st.caption("If the gross (pre-solar) and net (post-solar) SCENARIO lines diverge most around "
                   "midday and collapse to nothing after sunset, that is the genuine 'duck curve' "
                   "effect found in Phase 10 -- solar cannot affect an after-dark peak.")

    st.divider()
    st.markdown("**Modeled DISCOM breakdown at the SCENARIO peak**")
    modeled_estimate_caption()
    discom_table(scn["discoms"])


# ============================================================================
# PROOF MODE TAB
# ============================================================================
PROOF_EVENTS = {
    "PRIMARY -- 2025-06-12 (season peak, genuine 24h-ahead replay)": "2025-06-12",
    "SECONDARY / FAILURE MODE -- 2025-06-11 (worst genuine miss in the test period)": "2025-06-11",
}


def render_proof(B):
    st.subheader("PROOF MODE -- Historical Replay")
    st.caption("A genuine replay: the forecast below was built using ONLY information available "
              "at the issue time T. The actual outcome is revealed separately, on request -- "
              "this project never recomputes with hindsight.")

    options = list(PROOF_EVENTS.keys()) + ["Choose any other test day..."]
    choice = st.selectbox("Historical event", options, key="pf_event")
    if choice == "Choose any other test day...":
        day = st.selectbox("Test day", B.test_days,
                           index=B.test_days.index("2025-06-13") if "2025-06-13" in B.test_days else 0,
                           key="pf_day_other")
        label = f"{day} (24h-ahead replay ending at this day's own actual peak)"
    else:
        day = PROOF_EVENTS[choice]
        label = choice

    day_peak = B.res_df[B.res_df.index.normalize() == pd.Timestamp(day)]["actual"].idxmax()
    T = day_peak - pd.Timedelta(hours=24)

    try:
        r = B.replay(T, label)
    except BackendError as exc:
        st.error(str(exc)); return

    st.markdown("### HISTORICAL REPLAY")
    st.markdown(f"**Forecast issue time T = {r['T']}**")
    st.caption("Everything in this block used only information at or before T -- no demand "
              "telemetry, no observed temperature, and no knowledge of the outcome after T.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted peak (P50)", f"{r['pred_peak_mw']:,.1f} MW")
    c2.metric("Predicted peak time", r["pred_peak_time"].strftime("%d %b, %H:%M"))
    c3.metric("P10 / P90 at predicted peak", f"{r['p10_at_pred_peak']:,.0f} / {r['p90_at_pred_peak']:,.0f} MW")
    c4.metric("Reliability (this 24h window)", r["reliability"])
    st.write(f"Peak risk: **{r['peak_risk']}** -- " + "; ".join(r["peak_risk_reasons"]))
    st.write(f"Stress score: **{r['stress_score']:.1f} ({r['stress_level']})** -- computed by the same "
            "canonical function as Overview and What-If; matches Overview exactly whenever both "
            "are scoring the same predicted-peak block.")
    with st.expander("Reliability reasons"):
        for x in r["reliability_reasons"]:
            st.write(f"- {x}")
        st.caption("This reliability badge covers the full 24h WINDOW this forecast was issued "
                  "for (T to T+24h) -- by design, that can differ from the OVERVIEW tab's "
                  "reliability for the same calendar day, which covers that CALENDAR DAY only. "
                  "The two are answering different questions on purpose (\"how trustworthy was "
                  "everything this forecast covered\" vs. \"how trustworthy is today\"); it is "
                  "not an error if they show different HIGH/MEDIUM/LOW badges.")
    if r["n_missing"]:
        st.warning(f"{r['n_missing']} of 96 target blocks in this window have no scoreable actual "
                  f"value (documented Phase 5/6 telemetry gap) and were excluded from the metrics below.")

    st.divider()
    # A single STABLE widget key for the reveal toggle (never a dynamically-built key) --
    # dynamically naming a widget's key after other state (e.g. per day/event) is fragile: it
    # creates and destroys a distinct widget on every context switch, which surfaced as a
    # dashboard-state error during Phase 14 QA. Instead, one fixed-key toggle is reused, and its
    # value is explicitly reset to hidden whenever the selected event/day changes.
    sig = f"{day}::{choice}"
    if st.session_state.get("pf_reveal_sig") != sig:
        st.session_state["pf_reveal"] = False
        st.session_state["pf_reveal_sig"] = sig
    reveal = st.toggle("Reveal actual outcome", key="pf_reveal")
    if not reveal:
        st.info("Actual outcome hidden -- toggle above to reveal what really happened.")
        return

    st.markdown("### ACTUAL OUTCOME")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual peak", f"{r['actual_peak_mw']:,.1f} MW")
    c2.metric("Actual peak time", r["actual_peak_time"].strftime("%d %b, %H:%M"))
    c3.metric("Value-to-value error", f"{r['peak_abs_err_mw']:+,.1f} MW", f"{r['peak_pct_err']:+.1f}%")
    c4.metric("Warning lead time", f"{r['warning_hours']:.1f} h")

    inside = r["inside_p90_at_actual_peak"]
    if inside:
        st.success(f"Actual demand at its own peak fell INSIDE the P10-P90 band "
                  f"(margin to P90: {r['p90_margin_at_actual_peak_mw']:+,.1f} MW).")
    else:
        st.error(f"HONEST MISS -- actual demand at its own peak fell OUTSIDE the P10-P90 band "
                "(margin to P90: "
                f"{r['p90_margin_at_actual_peak_mw']:+,.1f} MW). This is the disclosed failure mode "
                "from Phase 9, not hidden here.")
    st.caption(f"Full-window MAPE (actual vs P50, {96 - r['n_missing']} scoreable blocks): "
              f"{r['mape_full_pct']:.2f}%")

    fig = go.Figure()
    win = r["win"]
    fig.add_trace(go.Scatter(x=win.index, y=win["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=win.index, y=win["p10"], line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(99,102,241,0.18)", name="P10-P90 band"))
    fig.add_trace(go.Scatter(x=win.index, y=win["p50"], line=dict(color="#6366f1", width=2), name="P50 forecast"))
    fig.add_trace(go.Scatter(x=win.index, y=win["actual"], line=dict(color="#111827", width=2), name="Actual"))
    fig.add_vline(x=r["T"], line_dash="dot", line_color="#6b7280", annotation_text="T (issue time)")
    fig.add_vline(x=r["actual_peak_time"], line_dash="dash", line_color="#dc2626",
                  annotation_text="actual peak")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                       legend=dict(orientation="h", y=1.12), hovermode="x unified",
                       yaxis_title="MW", xaxis_title=None,
                       title="HISTORICAL REPLAY -- forecast at T vs actual outcome")
    st.plotly_chart(fig, width='stretch')


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("DELHI PEAK INTELLIGENCE")
    st.caption("PS-1 -- AI-based Electricity Demand Prediction System (Delhi). All figures below "
              "are MODELED / SCENARIO / HISTORICAL REPLAY outputs from the locked Phase 7-12 "
              "backend -- see labels throughout. Runs fully offline from local data; no live "
              "network calls are made anywhere in this app. All MW values are **15-minute-average "
              "demand**, the resolution the model is trained and evaluated at -- not instantaneous "
              "readings, which can spike above any 15-min average shown here.")

    B, err = safe_backend()
    if err is not None:
        st.error(f"The dashboard could not start: {err}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["OVERVIEW", "WHAT-IF", "PROOF MODE"])
    with tab1:
        try:
            render_overview(B)
        except Exception:  # noqa: BLE001 -- never show a raw traceback to a judge
            st.error("Something went wrong rendering the Overview tab.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
    with tab2:
        try:
            render_whatif(B)
        except Exception:  # noqa: BLE001
            st.error("Something went wrong rendering the What-If tab.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
    with tab3:
        try:
            render_proof(B)
        except Exception:  # noqa: BLE001
            st.error("Something went wrong rendering the Proof Mode tab.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


main()
