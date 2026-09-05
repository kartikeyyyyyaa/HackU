"""
POWERWATCH (Delhi DEMO) -- operator dashboard (V2 presentation layer)
=====================================================================
This is a PRESENTATION-ONLY layer over phase13_backend.py.

Nothing in the forecasting, uncertainty, reliability, stress-scoring, DISCOM
allocation or advisory logic is changed, re-implemented or re-tuned here. Every
number on every screen is read directly from a phase13_backend function
(day_state / recommend / replay / run_scenario / discom_estimate). There are no
backend edits behind this file at all -- phase13_backend.py is byte-identical to
the version approved in Phase 13/14, and phase13_dashboard_app.py (V1) is left
untouched and still runnable as a fallback.

THEMING: the dashboard is theme-AGNOSTIC. Body text inherits Streamlit's own
text colour and surfaces are neutral grey tints, so switching Light/Dark from the
app menu recolours everything instantly -- no rerun, no refresh. Nothing here
reads st.context.theme, because Streamlit repaints its chrome without re-running
the script, which would leave any server-baked colour one theme behind.

Honesty rules preserved verbatim:
  - Runs on HISTORICAL data (approved test period 1 May - 30 Jun 2025). Never
    described as live telemetry, and no date is ever called "today".
  - Capacity (9,000 MW) is ILLUSTRATIVE, never an official declared limit.
  - DISCOM figures are MODELED ESTIMATES, never feeder telemetry.
  - What-If temperature is a genuine model recomputation; solar is disclosed,
    non-fitted scenario arithmetic. The two are always labelled differently.
  - Proof Mode never reveals the outcome before the user asks for it.
  - All MW values are 15-minute-average demand, not instantaneous readings.

No network calls anywhere in this file or in phase13_backend.py.
"""
import sys
import traceback
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase13_backend import (   # noqa: E402
    ASSUMED_CAPACITY_MW, BackendError, DEFAULT_DATA_DIR, INDIVIDUALLY_MODELED_DISCOMS,
    SOLAR_CAP_MAX, TEMP_DELTA_MAX_ABS, build_backend,
    canonical_worst_reliability, prev_calendar_day_p50_peak, stress_score,
    W_P50_UTIL, W_P90_UTIL, W_GROWTH, W_HEAT, W_RELIABILITY,
)

APP_NAME = "PowerWatch"
APP_TAG = "DELHI DEMO"

st.set_page_config(page_title=f"{APP_NAME} (Delhi DEMO)", layout="wide",
                   initial_sidebar_state="collapsed")


# ============================================================================
# THEME-AGNOSTIC PALETTE
#
# An earlier version resolved the palette per render from st.context.theme. That
# was wrong: Streamlit repaints its own chrome the instant a viewer switches
# Light/Dark from the app menu, but it does NOT re-run the script, so every
# colour we had baked into inline HTML stayed on the previous theme until a
# manual refresh -- dark text on a dark page, light text on a light page.
# Streamlit 1.63 exposes no CSS custom properties and no data-theme attribute to
# key off either, so there is nothing to react to client-side.
#
# The fix is to need no theme knowledge at all:
#   - body text INHERITS Streamlit's own text colour, so it always follows the
#     live theme with no rerun;
#   - surfaces are neutral grey tints that read as a raised card on either
#     background;
#   - semantic colours are mid-tones chosen to stay legible on both.
# ============================================================================
INK   = "inherit"                     # follow Streamlit's live theme text colour
SURF  = "rgba(128,128,128,0.10)"      # raised card, neutral on light and dark
SURF2 = "rgba(128,128,128,0.16)"      # secondary surface
LINE  = "rgba(128,128,128,0.22)"      # hairline border
MUTE_OPACITY = "0.68"                 # secondary text = inherited colour, dimmed

OK    = "#16A34A"   # NORMAL
WARN  = "#D97706"   # WATCH
BAD   = "#DC2626"   # HIGH
WORST = "#991B1B"   # CRITICAL
FC    = "#6366F1"   # forecast / P50
HIST  = "#6B7280"   # observed history: deliberately neutral -- context, not subject
BAND  = "rgba(99,102,241,0.20)"
CHART_TEXT = "#8B93A5"                # axis labels, legible on both themes
CHART_GRID = "rgba(128,128,128,0.25)"
TINT  = "1A"                          # alpha suffix for tinted status panels

PAL = dict(ink=INK, mute=INK, line=LINE, surf=SURF, ok=OK, warn=WARN, bad=BAD,
           worst=WORST, fc=FC, hist=HIST, band=BAND, tint=TINT)


def muted(text, size="0.88rem", extra=""):
    """Secondary text: inherits the live theme colour, just dimmed."""
    return (f'<span style="opacity:{MUTE_OPACITY};font-size:{size};{extra}">{text}</span>')


STRESS_DOT = {"NORMAL": "🟢", "WATCH": "🟡", "HIGH": "🔴", "CRITICAL": "🔴"}


def stress_c(level):
    return {"NORMAL": PAL["ok"], "WATCH": PAL["warn"],
            "HIGH": PAL["bad"], "CRITICAL": PAL["worst"]}[level]


def risk_c(risk):
    return {"LOW": PAL["ok"], "MODERATE": PAL["warn"],
            "HIGH": PAL["bad"], "CRITICAL": PAL["worst"]}[risk]


def rel_c(rel):
    return {"HIGH": PAL["ok"], "MEDIUM": PAL["warn"], "LOW": PAL["bad"]}[rel]


# Plain-English meaning of each reliability state. These restate the EXISTING
# Phase 7 reliability rule in operator language -- the rule itself is unchanged.
REL_PLAIN = {
    "HIGH": "The recent demand data behind this forecast is complete and directly measured.",
    "MEDIUM": "Some supporting data was reconstructed or incomplete. Treat with moderate caution.",
    "LOW": "A key recent demand input had to be reconstructed. Treat this forecast with extra caution.",
}

# ============================================================================
# GLOSSARY -- one short plain-English line per technical term, shown on hover.
# ============================================================================
G = {
    "mw": "Megawatts — how much electricity is being used at one moment. "
          "All values here are 15-minute averages, not instantaneous readings.",
    "expected_peak": "The highest electricity demand the model expects during this forecast. "
                     "Technical term: P50, the central estimate.",
    "peak_time": "The time of day when demand is expected to be highest.",
    "planning_level": "A higher demand level used for safer planning — demand is expected to stay "
                      "below it most of the time. Technical term: P90.",
    "low_level": "A lower demand level; demand is expected to stay above it most of the time. "
                 "Technical term: P10.",
    "condition": "An overall 0–100 score of how much pressure the forecast puts on modeled grid "
                 "capacity, combining expected demand, the high-demand planning level, "
                 "day-on-day growth, heat and data reliability.",
    "risk": "How likely this forecast is to strain modeled capacity, based on how close demand "
            "comes to it and how trustworthy the input data was.",
    "reliability": "How complete and directly-measured the input data behind this forecast was.",
    "capacity": "9,000 MW — an ILLUSTRATIVE planning reference used by this project. "
                "It is not an official DISCOM or SLDC declared limit.",
    "forecast": "Built 24 hours ahead using only information that existed at the issue time — "
                "no data from after that moment is used.",
    "issued": "The moment this forecast was made. Everything in it uses only data available "
              "at or before this time.",
    "modeled": "Calculated by this system from Delhi-wide demand, because direct measurements at "
               "this level are not available.",
    "discom": "Delhi's electricity distribution companies. Each one is the licensed distributor for "
              "a defined part of the city, so this doubles as an area-wise view. These are modeled "
              "estimates of each one's share, not measured feeder data.",
    "area": "The part of Delhi each distribution company is licensed to serve. This is the finest "
            "geographic split this project can support honestly — feeder-level data is not public.",
    "rel_stress": "Modeled demand for a utility compared with that utility's own stated 2025 peak "
                  "— not a measure of its physical capacity.",
    "replay": "A real past event replayed using only the information that existed before it "
              "happened, so the forecast can be checked honestly.",
    "whatif": "A test of what could happen if conditions changed. It is a scenario, not a "
              "prediction of a real event.",
    "solar": "An estimate of how much rooftop solar could reduce demand on the grid during "
             "daylight. It is an assumption, not something this model learned from data.",
    "temp_scn": "Changes the forecast temperature and genuinely re-runs the real model with it.",
    "band": "The range the model expects demand to fall within most of the time.",
    "historical": "This system runs on real recorded data from 1 May – 30 Jun 2025. "
                  "It is not connected to live grid telemetry.",
}


def tip(text, key):
    """Label with a hover explanation. Native browser tooltip -- no framework."""
    return (f'<span title="{G[key]}" style="border-bottom:1px dotted currentColor;cursor:help">'
            f'{text}</span> <span style="color:inherit;opacity:0.68;font-size:0.8em" title="{G[key]}">ⓘ</span>')


# ---------------------------------------------------------------- typography
def page_title():
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">'
        f'<span style="font-size:2.05rem;font-weight:800;letter-spacing:-.02em;'
        f'color:{PAL["ink"]};line-height:1.1">{APP_NAME}</span>'
        f'<span style="background:{PAL["fc"]}{PAL["tint"]};color:{PAL["fc"]};'
        f'border:1px solid {PAL["fc"]}66;padding:2px 9px;border-radius:11px;'
        f'font-size:0.7rem;font-weight:800;letter-spacing:.09em">{APP_TAG}</span></div>'
        f'<div style="font-size:0.95rem;color:inherit;opacity:0.68;margin-top:1px">'
        f'Day-ahead electricity demand and grid-risk decision support</div>',
        unsafe_allow_html=True)


def section(title, key=None, sub=None):
    """One consistent section heading style, noticeably larger than body text."""
    label = tip(title, key) if key else title
    st.markdown(
        f'<div style="font-size:1.22rem;font-weight:750;color:{PAL["ink"]};'
        f'letter-spacing:-.01em;margin:20px 0 0 0">{label}</div>'
        + (f'<div style="font-size:0.88rem;color:inherit;opacity:0.68;margin:2px 0 10px 0">{sub}</div>'
           if sub else '<div style="height:9px"></div>'),
        unsafe_allow_html=True)


def kicker(text, key=None):
    body = tip(text, key) if key else text
    st.markdown(f'<div style="font-size:0.73rem;letter-spacing:.07em;text-transform:uppercase;'
                f'color:inherit;opacity:0.68;font-weight:700;margin-bottom:3px">{body}</div>',
                unsafe_allow_html=True)


def big(value, sub=None, color=None):
    c = color or PAL["ink"]
    st.markdown(f'<div style="font-size:1.95rem;font-weight:750;color:{c};line-height:1.12">'
                f'{value}</div>' +
                (f'<div style="font-size:0.81rem;color:inherit;opacity:0.68;margin-top:2px">{sub}</div>'
                 if sub else ""), unsafe_allow_html=True)


def chip(text, color):
    st.markdown(f'<span style="background:{color}{PAL["tint"]};color:{color};'
                f'border:1px solid {color}66;padding:2px 10px;border-radius:12px;'
                f'font-weight:700;font-size:0.8rem">{text}</span>', unsafe_allow_html=True)


def body(text, size="0.93rem"):
    st.markdown(f'<div style="font-size:{size};color:{PAL["ink"]}">{text}</div>',
                unsafe_allow_html=True)


# ============================================================================
# BACKEND -- built ONCE per server process, never rebuilt on an interaction
# ============================================================================
@st.cache_resource(show_spinner="Loading forecasting model (one-time, ~5s)…")
def get_backend():
    return build_backend(DEFAULT_DATA_DIR)


def safe_backend():
    try:
        return get_backend(), None
    except BackendError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error while starting the forecasting model: {exc}"


@st.cache_resource(show_spinner=False)
def actual_series(_B):
    """Continuous observed-demand series across the train/test boundary, so the chart can
    show what had actually happened up to the forecast issue time even for the first test
    day. Read-only reuse of existing backend frames -- no new modelling."""
    s = pd.concat([_B.train["load_MW"], _B.test["load_MW"]]).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ============================================================================
# PLAIN-ENGLISH SUMMARY -- composed from real backend fields, never hardcoded
# ============================================================================
def headline_sentence(state):
    pk, t, u90 = state["p50"], state["peak_time"], state["util90"]
    lead = (f"Demand is expected to peak near {pk:,.0f} MW at {t:%H:%M}, with the "
            f"high-demand planning level reaching {u90 * 100:.0f}% of modeled capacity.")
    tail = {
        "NORMAL": "This looks like a normal day for the grid.",
        "WATCH": "That is tighter than a comfortable margin — worth watching through the peak window.",
        "HIGH": "That puts significant pressure on modeled capacity — prepare ahead of the peak.",
        "CRITICAL": "That is very close to modeled capacity — immediate preparation is advised.",
    }[state["stress_level"]]
    return f"{lead} {tail}"


def stress_components(B, state):
    """Recompute the score's parts for the technical drill-down, using the SAME canonical
    backend function the headline number came from (so it can never disagree with it)."""
    ipk = state["peak_time"]
    prev = prev_calendar_day_p50_peak(B.res_df, ipk)
    rel = canonical_worst_reliability(B.res_df, ipk)
    tc = float(B.res_df.loc[ipk, "temp_corr"])
    sc = stress_score(state["p50"], state["p90"], tc, rel, prev)
    return [
        ("Expected demand vs capacity", W_P50_UTIL * sc["c_p50"], W_P50_UTIL,
         f"{sc['util50'] * 100:.0f}% of modeled capacity"),
        ("Planning level vs capacity", W_P90_UTIL * sc["c_p90"], W_P90_UTIL,
         f"{sc['util90'] * 100:.0f}% of modeled capacity"),
        ("Day-on-day growth", W_GROWTH * sc["c_growth"], W_GROWTH, f"{sc['growth'] * 100:+.1f}% vs previous day"),
        ("Heat", W_HEAT * sc["c_heat"], W_HEAT, f"{tc:.1f}°C corrected forecast temperature"),
        ("Data reliability", W_RELIABILITY * sc["c_rel"], W_RELIABILITY, f"{rel.title()} reliability"),
    ], sc


# ============================================================================
# SHARED PIECES
# ============================================================================
def context_strip(line):
    st.markdown(
        f'<div style="background:{PAL["surf"]};border:1px solid {PAL["line"]};border-radius:7px;'
        f'padding:7px 13px;font-size:0.83rem;color:inherit;opacity:0.68;margin-bottom:14px">{line}</div>',
        unsafe_allow_html=True)


def chart_layout(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34, b=8), hovermode="x unified",
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=11, color=CHART_TEXT)),
        yaxis_title="MW", xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHART_TEXT),
        yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT),
        xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT))
    return fig


def data_notes(extra=None):
    with st.expander("About this data"):
        st.markdown(
            f"- **Historical, not live.** Every figure comes from real recorded Delhi demand and "
            f"weather for the approved test period **1 May – 30 June 2025**. Nothing here is a "
            f"live telemetry feed, and no date is presented as today.\n"
            f"- **Forecasts are genuinely 24 hours ahead.** Each forecast is built only from "
            f"information that existed at its issue time — never from what happened afterwards.\n"
            f"- **All MW values are 15-minute averages**, the resolution the model is trained and "
            f"measured at. A true instantaneous peak can sit above any 15-minute average shown here.\n"
            f"- **Capacity of {ASSUMED_CAPACITY_MW:,.0f} MW is ILLUSTRATIVE** — a planning reference "
            f"chosen by this project, not an official DISCOM/SLDC declared limit.\n"
            f"- **DISCOM figures are modeled estimates** derived from Delhi-wide demand using "
            f"disclosed allocation ratios. This project has no feeder-level telemetry.\n"
            + (f"- {extra}\n" if extra else ""))


# Each Delhi DISCOM is the licensed distributor for a defined part of the city, so the utility
# breakdown IS the area breakdown. These service areas are public licensing facts, not data this
# project derived or estimated -- only the MW figures beside them are modeled.
DISCOM_AREA = {
    "BRPL": "South & West Delhi",
    "BYPL": "Central & East Delhi",
    "TPDDL": "North & North-West Delhi",
    "NDMC_MES_RESIDUAL": "New Delhi (NDMC) & Delhi Cantonment",
}


def discom_block(discoms):
    rows = []
    for k in list(INDIVIDUALLY_MODELED_DISCOMS) + ["NDMC_MES_RESIDUAL"]:
        v = discoms[k]
        rs = (f"{v['relative_stress_p50'] * 100:.0f}%"
              if v["relative_stress_p50"] is not None else "not available")
        rows.append({"Area of Delhi": DISCOM_AREA[k],
                     "Distributor": k.replace("NDMC_MES_RESIDUAL", "NDMC + MES (combined)"),
                     "Expected demand (MW)": f"{v['p50_mw']:,.0f}",
                     "Planning level (MW)": f"{v['p90_mw']:,.0f}",
                     "Share of Delhi demand": f"{v['share'] * 100:.1f}%",
                     "Vs own 2025 peak": rs})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Each distributor is the licensed supplier for the area shown, so this is the "
               "area-wise view of where demand sits. The MW figures are **modeled estimates** split "
               "from the Delhi-wide forecast using disclosed ratios — not measured feeder data. "
               "'Share of Delhi demand' is how much of the city total each area carries; "
               "'vs own 2025 peak' compares it with that distributor's own stated 2025 peak. The two "
               "can point at different areas — the biggest area is not always the most stretched. "
               "NDMC + MES has no published individual reference, so it is shown only as a combined "
               "remainder. Individual feeder-level breakdown is not possible here: feeder telemetry "
               "is not publicly available, and this project does not invent it.")


# ============================================================================
# TAB 1 -- OVERVIEW
# ============================================================================
def render_overview(B):
    days = B.test_days
    c_sel, c_note = st.columns([1, 2.4])
    with c_sel:
        day = st.selectbox("Operating day", days,
                           index=days.index("2025-06-12") if "2025-06-12" in days else 0,
                           key="ov_day",
                           help="Choose any day in the recorded test period (1 May – 30 Jun 2025).")
    try:
        state = B.day_state(day)
    except BackendError as exc:
        st.error(str(exc))
        return

    d = pd.Timestamp(day)
    with c_note:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        context_strip(
            f"Forecast for <b>{d:%A, %d %B %Y}</b> &nbsp;·&nbsp; issued "
            f"<b>{state['now']:%d %b, %H:%M}</b>, 24 hours ahead &nbsp;·&nbsp; "
            f"<span title=\"{G['historical']}\">recorded historical day, not live telemetry ⓘ</span>")

    lvl, risk = state["stress_level"], state["peak_risk"]
    sc_col = stress_c(lvl)

    # ---- 1. THE ANSWER ----------------------------------------------------
    st.markdown(
        f'<div style="border-left:6px solid {sc_col};background:{sc_col}{PAL["tint"]};'
        f'border-radius:7px;padding:13px 17px;margin-bottom:15px">'
        f'<div style="font-size:1.32rem;font-weight:800;color:{sc_col};letter-spacing:.01em">'
        f'{STRESS_DOT[lvl]} GRID CONDITION: {lvl}</div>'
        f'<div style="font-size:0.97rem;color:{PAL["ink"]};margin-top:5px">'
        f'{headline_sentence(state)}</div></div>',
        unsafe_allow_html=True)

    # ---- 2. HEADLINE NUMBERS (no duplicates) ------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kicker("Expected peak demand", "expected_peak")
        big(f"{state['p50']:,.0f} MW", "central forecast (P50)")
    with k2:
        kicker("Expected peak time", "peak_time")
        big(f"{state['peak_time']:%H:%M}", f"{state['peak_time']:%I %p}".lstrip("0").lower())
    with k3:
        kicker("High-demand planning level", "planning_level")
        big(f"{state['p90']:,.0f} MW", f"{state['util90'] * 100:.0f}% of modeled capacity (P90)")
    with k4:
        kicker("Grid condition", "condition")
        big(f"{state['stress_score']:.0f}"
            f"<span style='font-size:1rem;color:{PAL['mute']}'>/100</span>", None, sc_col)
        chip(f"{lvl} · peak risk {risk}", sc_col)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ---- 3. THE PICTURE ---------------------------------------------------
    now, peak = state["now"], state["peak_time"]
    hist_from, fwd_to = now - pd.Timedelta(hours=24), now + pd.Timedelta(hours=24)
    obs = actual_series(B)
    hist = obs[(obs.index >= hist_from) & (obs.index <= now)]
    fwd = B.res_df[(B.res_df.index >= now) & (B.res_df.index <= fwd_to)]

    reveal = st.session_state.get("ov_reveal", False)
    fig = go.Figure()
    # Band drawn FIRST so it sits underneath the lines; legendrank sets reading order.
    fig.add_trace(go.Scatter(x=fwd.index, y=fwd["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fwd.index, y=fwd["p10"], line=dict(width=0), fill="tonexty",
                             fillcolor=PAL["band"], name="Expected range", legendrank=3,
                             hovertemplate="%{y:,.0f} MW<extra>range</extra>"))
    if len(hist):
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Already happened", legendrank=1,
                                 line=dict(color=PAL["hist"], width=2),
                                 hovertemplate="%{y:,.0f} MW<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=fwd.index, y=fwd["p50"], name="Forecast", legendrank=2,
                             line=dict(color=PAL["fc"], width=2.6),
                             hovertemplate="%{y:,.0f} MW<extra>forecast</extra>"))
    if reveal:
        fig.add_trace(go.Scatter(x=fwd.index, y=fwd["actual"], name="What actually happened",
                                 legendrank=4, line=dict(color=PAL["hist"], width=1.6, dash="dot"),
                                 hovertemplate="%{y:,.0f} MW<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=[peak], y=[state["p50"]], mode="markers+text",
                             marker=dict(color=sc_col, size=11,
                                         line=dict(color=CHART_GRID, width=2)),
                             text=[f"  peak {state['p50']:,.0f} MW"], textposition="top center",
                             textfont=dict(color=sc_col, size=12), showlegend=False,
                             hoverinfo="skip"))
    fig.add_vline(x=now.to_pydatetime(), line_dash="dot", line_color=CHART_TEXT,
                  annotation_text="forecast issued", annotation_position="top left",
                  annotation_font=dict(size=11, color=CHART_TEXT))
    fig.add_hline(y=ASSUMED_CAPACITY_MW, line_dash="dash", line_color=PAL["bad"], opacity=0.5,
                  annotation_text=f"modeled capacity {ASSUMED_CAPACITY_MW:,.0f} MW (illustrative)",
                  annotation_position="top right", annotation_font=dict(size=10, color=PAL["bad"]))
    st.plotly_chart(chart_layout(fig), width="stretch")

    cc1, cc2 = st.columns([3, 1])
    with cc1:
        st.caption("Solid line: demand that had already happened when the forecast was issued. "
                   "Coloured line and shaded band: what the model expected for the following 24 hours.")
    with cc2:
        st.checkbox("Show actual outcome", key="ov_reveal",
                    help="Overlays what demand actually turned out to be. This is hindsight — the "
                         "forecast above never used it.")

    st.divider()

    # ---- 4. WHAT TO PAY ATTENTION TO + RELIABILITY ------------------------
    a, b = st.columns([1.55, 1])
    with a:
        section("What to pay attention to")
        for adv in B.recommend(state):
            st.markdown(
                f'<div style="border:1px solid {PAL["line"]};border-left:4px solid rgba(128,128,128,0.45);'
                f'background:{PAL["surf"]};border-radius:6px;padding:9px 13px;margin-bottom:8px">'
                f'<div style="font-weight:800;font-size:0.82rem;letter-spacing:.04em;'
                f'color:{PAL["ink"]}">{adv["category"]}</div>'
                f'<div style="font-size:0.93rem;color:{PAL["ink"]};margin:3px 0">{adv["advisory"]}</div>'
                f'<div style="font-size:0.79rem;color:inherit;opacity:0.68">Why: {adv["reason"]}</div></div>',
                unsafe_allow_html=True)
        st.caption("Generated by a fixed rule set from the numbers above — not free-form AI text.")
    with b:
        section("Can we trust this forecast?")
        rel = state["reliability"]
        chip(f"{rel} reliability", rel_c(rel))
        st.markdown(f'<div style="font-size:0.9rem;color:{PAL["ink"]};margin-top:7px">'
                    f'{REL_PLAIN[rel]}</div>', unsafe_allow_html=True)
        with st.expander("What affected it"):
            for r in state["reliability_reasons"]:
                st.write(f"- {r}")
            st.caption("This covers the selected calendar day. Proof Mode reports reliability for the "
                       "24-hour window a forecast was issued for, so the two can differ for the same "
                       "date — by design.")

    # ---- 5. WHERE THE PRESSURE IS ----------------------------------------
    section("Where the pressure is — by area of Delhi", "area")
    discom_block(state["discoms"])

    # ---- 6. TECHNICAL DRILL-DOWN -----------------------------------------
    with st.expander("Technical detail"):
        p10 = float(B.res_df.loc[peak, "p10"])
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Forecast at the expected peak block**")
            st.write(f"- P10 (lower): **{p10:,.0f} MW**")
            st.write(f"- P50 (central): **{state['p50']:,.0f} MW** — {state['util50'] * 100:.1f}% of capacity")
            st.write(f"- P90 (upper): **{state['p90']:,.0f} MW** — {state['util90'] * 100:.1f}% of capacity")
            st.write(f"- Issue time T: **{state['now']}** ({state['hours_to_peak']:.1f} h before peak)")
            st.caption("P10/P50/P90 come from quantile regression on out-of-fold residuals, widened "
                       "according to the reliability state. The point forecast is an OLS model on "
                       "features built strictly ≥24 h before each target block.")
        with t2:
            st.markdown("**Grid condition score, by component**")
            comps, sc = stress_components(B, state)
            st.dataframe(pd.DataFrame(
                [{"Component": n, "Points": f"{p:.1f}", "Max": f"{m:.1f}", "Basis": w}
                 for n, p, m, w in comps]), hide_index=True, width="stretch")
            st.write(f"Total: **{sc['stress_score']:.1f} / 100 → {sc['stress_level']}**")
            st.caption(f"Largest single driver: {state['main_driver']}. Weights were fixed by a "
                       f"documented design experiment and are identical across every view.")
        st.markdown("**Peak risk reasoning**")
        for r in state["peak_risk_reasons"]:
            st.write(f"- {r}")
    data_notes()


# ============================================================================
# TAB 2 -- WHAT-IF
# ============================================================================
def render_whatif(B):
    days = B.test_days
    c_sel, c_note = st.columns([1, 2.4])
    with c_sel:
        day = st.selectbox("Day to test", days,
                           index=days.index("2025-06-12") if "2025-06-12" in days else 0,
                           key="wi_day", help="The real day the scenario is applied to.")
    peak_idx = B.res_df[B.res_df.index.normalize() == pd.Timestamp(day)]["p50"].idxmax()
    T = peak_idx - pd.Timedelta(hours=24)
    with c_note:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        context_strip(f"Scenario applied to the 24 hours from <b>{T:%d %b, %H:%M}</b> &nbsp;·&nbsp; "
                      f"<b>this is a hypothetical test, not something that happened</b>")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="font-size:1.02rem;font-weight:700;color:{PAL["ink"]}">'
                    f'{tip("If temperature were different", "temp_scn")}</div>',
                    unsafe_allow_html=True)
        temp_delta = st.slider("Change vs the forecast temperature (°C)",
                               -TEMP_DELTA_MAX_ABS, TEMP_DELTA_MAX_ABS, 0.0, 0.5, key="wi_temp",
                               help="Re-runs the real forecasting model with this temperature change.")
    with c2:
        st.markdown(f'<div style="font-size:1.02rem;font-weight:700;color:{PAL["ink"]}">'
                    f'{tip("If rooftop solar existed", "solar")}</div>', unsafe_allow_html=True)
        solar_on = st.checkbox("Include rooftop solar", key="wi_solar_on",
                               help="An assumed daylight generation curve subtracted from demand. "
                                    "Not something the model learned from data.")
        solar_cap = st.slider("Assumed installed capacity (MW)", 0, int(SOLAR_CAP_MAX), 1500, 100,
                              key="wi_solar_cap", disabled=not solar_on)

    try:
        base = B.run_scenario(T, "BASELINE")
        scn = B.run_scenario(T, "SCENARIO", temp_delta=temp_delta,
                             solar_capacity_mw=(solar_cap if solar_on else None))
    except BackendError as exc:
        st.error(str(exc))
        return

    for n in scn["notes"]:
        st.warning(n)

    changed = (temp_delta != 0.0) or (solar_on and solar_cap > 0)
    dpk = scn["peak_mw"] - base["peak_mw"]
    dsc = scn["stress_score"] - base["stress_score"]
    scn_col = stress_c(scn["stress_level"])

    section("What changes")
    m1, m2, m3 = st.columns(3)
    with m1:
        kicker("Expected peak demand", "expected_peak")
        big(f"{base['peak_mw']:,.0f} → {scn['peak_mw']:,.0f} MW" if changed
            else f"{base['peak_mw']:,.0f} MW",
            f"{dpk:+,.0f} MW" if changed else "no change applied yet")
    with m2:
        kicker("High-demand planning level", "planning_level")
        big(f"{base['p90_mw']:,.0f} → {scn['p90_mw']:,.0f} MW" if changed
            else f"{base['p90_mw']:,.0f} MW",
            f"{scn['p90_mw'] - base['p90_mw']:+,.0f} MW" if changed else None)
    with m3:
        kicker("Grid condition", "condition")
        if changed and scn["stress_level"] != base["stress_level"]:
            big(f"{base['stress_level']} → {scn['stress_level']}", f"score {dsc:+.1f}", scn_col)
        else:
            big(scn["stress_level"], f"score {scn['stress_score']:.0f}/100"
                + (f" ({dsc:+.1f})" if changed else ""), scn_col)
        chip(f"peak risk {base['peak_risk']} → {scn['peak_risk']}" if
             (changed and scn["peak_risk"] != base["peak_risk"]) else f"peak risk {scn['peak_risk']}",
             risk_c(scn["peak_risk"]))

    if changed and scn["peak_risk"] != base["peak_risk"]:
        rc = risk_c(scn["peak_risk"])
        st.markdown(
            f'<div style="border-left:5px solid {rc};background:{rc}{PAL["tint"]};border-radius:6px;'
            f'padding:10px 15px;margin:10px 0;font-size:0.92rem;color:{PAL["ink"]}">'
            f'In this scenario the peak risk changes from <b>{base["peak_risk"]}</b> to '
            f'<b>{scn["peak_risk"]}</b>. {"; ".join(scn["peak_risk_reasons"])}.</div>',
            unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["window"], y=base["p50"], name="Baseline",
                             line=dict(color=CHART_TEXT, width=2, dash="dot"),
                             hovertemplate="%{y:,.0f} MW<extra>baseline</extra>"))
    if scn["solar_mw"] is not None:
        fig.add_trace(go.Scatter(x=scn["window"], y=scn["gross_p50"], name="Scenario before solar",
                                 line=dict(color=PAL["warn"], width=1.6, dash="dash"),
                                 hovertemplate="%{y:,.0f} MW<extra>before solar</extra>"))
    fig.add_trace(go.Scatter(x=scn["window"], y=scn["p50"], name="Scenario",
                             line=dict(color=PAL["fc"], width=2.6),
                             hovertemplate="%{y:,.0f} MW<extra>scenario</extra>"))
    fig.add_hline(y=ASSUMED_CAPACITY_MW, line_dash="dash", line_color=PAL["bad"], opacity=0.45,
                  annotation_text="modeled capacity (illustrative)", annotation_position="top right",
                  annotation_font=dict(size=10, color=PAL["bad"]))
    st.plotly_chart(chart_layout(fig, 310), width="stretch")

    if scn["solar_mw"] is not None:
        st.caption("If the two scenario lines separate around midday and meet again after sunset, "
                   "that is the real effect being shown: solar can flatten daytime demand but cannot "
                   "reduce a peak that happens after dark.")

    with st.expander("Modeled area-level breakdown for this scenario"):
        discom_block(scn["discoms"])
    with st.expander("Technical detail"):
        st.write(f"- Baseline peak **{base['peak_mw']:,.0f} MW** at {base['peak_time']:%H:%M}, "
                 f"P90 {base['p90_mw']:,.0f} MW, score {base['stress_score']:.1f} "
                 f"({base['stress_level']}), risk {base['peak_risk']}")
        st.write(f"- Scenario peak **{scn['peak_mw']:,.0f} MW** at {scn['peak_time']:%H:%M}, "
                 f"P90 {scn['p90_mw']:,.0f} MW, score {scn['stress_score']:.1f} "
                 f"({scn['stress_level']}), risk {scn['peak_risk']}")
        st.write(f"- Utilisation: P50 {scn['util50'] * 100:.1f}%, P90 {scn['util90'] * 100:.1f}% "
                 f"of the illustrative {ASSUMED_CAPACITY_MW:,.0f} MW reference")
        st.caption("Temperature is fed back through the same fitted model that produces every other "
                   "forecast in this app; only the forecast temperature and terms derived from it "
                   "change. Solar is separate, disclosed arithmetic applied after the model, using a "
                   "half-sine daylight curve — it is never blended into the model itself. Baseline and "
                   "scenario are always scored by the same function, so the comparison is like-for-like.")
    data_notes("**What-If results are scenarios, not forecasts of real events.** They show how this "
               "model responds to a changed input.")


# ============================================================================
# TAB 3 -- PROOF MODE
# ============================================================================
EVENTS = {
    "The season's highest-demand day — 12 June 2025": "2025-06-12",
    "The forecast's worst miss — 11 June 2025": "2025-06-11",
}


def render_proof(B):
    section("Replay a real past event", "replay",
            "Pick a real day. The system shows the forecast it would have issued 24 hours "
            "beforehand, using only what was known at that moment. What actually happened stays "
            "hidden until you ask for it.")

    opts = list(EVENTS) + ["Any other day in the test period…"]
    choice = st.selectbox("Event", opts, key="pf_event")
    if choice == opts[-1]:
        day = st.selectbox("Day", B.test_days,
                           index=B.test_days.index("2025-06-13") if "2025-06-13" in B.test_days else 0,
                           key="pf_day_other")
    else:
        day = EVENTS[choice]

    act_peak = B.res_df[B.res_df.index.normalize() == pd.Timestamp(day)]["actual"].idxmax()
    T = act_peak - pd.Timedelta(hours=24)
    try:
        r = B.replay(T, choice)
    except BackendError as exc:
        st.error(str(exc))
        return

    context_strip(f"Forecast issued <b>{r['T']:%d %B %Y, %H:%M}</b> &nbsp;·&nbsp; covering the next "
                  f"24 hours &nbsp;·&nbsp; built only from information available at that moment")

    section("Step 1 — what the system predicted, 24 hours ahead")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        kicker("Predicted peak", "expected_peak")
        big(f"{r['pred_peak_mw']:,.0f} MW")
    with p2:
        kicker("Predicted peak time", "peak_time")
        big(f"{r['pred_peak_time']:%H:%M}")
    with p3:
        kicker("Expected range", "band")
        big(f"{r['p10_at_pred_peak']:,.0f}–{r['p90_at_pred_peak']:,.0f}", "MW (P10–P90)")
    with p4:
        kicker("Forecast reliability", "reliability")
        big(r["reliability"], None, rel_c(r["reliability"]))
    body(REL_PLAIN[r["reliability"]], "0.9rem")

    if r["n_missing"]:
        st.warning(f"{r['n_missing']} of 96 blocks in this window have no recorded actual value "
                   f"(a known telemetry gap) and are excluded from the accuracy figures below.")

    st.divider()
    sig = f"{day}::{choice}"
    if st.session_state.get("pf_reveal_sig") != sig:
        st.session_state["pf_reveal"] = False
        st.session_state["pf_reveal_sig"] = sig
    reveal = st.toggle("Step 2 — reveal what actually happened", key="pf_reveal")
    if not reveal:
        st.info("The outcome is hidden. Nothing above used it.")
        data_notes()
        return

    section("Step 2 — what actually happened")
    inside = r["inside_p90_at_actual_peak"]
    vc = PAL["ok"] if inside else PAL["bad"]
    # peak_abs_err_mw is FORECAST MINUS ACTUAL, so a negative error means the real peak came in
    # ABOVE what was forecast. Stating this backwards would misdescribe the system's own miss.
    err_word = "above" if r["peak_abs_err_mw"] < 0 else "below"
    st.markdown(
        f'<div style="border-left:6px solid {vc};background:{vc}{PAL["tint"]};'
        f'border-radius:7px;padding:13px 17px;margin-bottom:13px">'
        # Precise claim: measured AT THE ACTUAL PEAK BLOCK, not across the whole window. The chart
        # below often shows the actual outside the band at other hours, so an unqualified "stayed
        # inside the range" would overstate what was actually verified.
        f'<div style="font-size:1.1rem;font-weight:750;color:{vc}">'
        f'{"At the peak, actual demand stayed inside the predicted range" if inside else "At the peak, actual demand broke through the predicted range"}</div>'
        f'<div style="font-size:0.95rem;color:{PAL["ink"]};margin-top:5px">'
        f'Real peak was <b>{r["actual_peak_mw"]:,.0f} MW</b> at {r["actual_peak_time"]:%H:%M} — '
        f'{abs(r["peak_abs_err_mw"]):,.0f} MW ({abs(r["peak_pct_err"]):.1f}%) {err_word} the central '
        f'forecast, with <b>{abs(r["p90_margin_at_actual_peak_mw"]):,.0f} MW '
        f'{"to spare against" if inside else "beyond"}</b> the high-demand planning level. '
        f'The warning came <b>{r["warning_hours"]:.0f} hours</b> in advance.</div></div>',
        unsafe_allow_html=True)

    win = r["win"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=win.index, y=win["p10"], line=dict(width=0), fill="tonexty",
                             fillcolor=PAL["band"], name="Predicted range", legendrank=3))
    fig.add_trace(go.Scatter(x=win.index, y=win["p50"], name="Forecast", legendrank=2,
                             line=dict(color=PAL["fc"], width=2.4)))
    fig.add_trace(go.Scatter(x=win.index, y=win["actual"], name="Actual", legendrank=1,
                             line=dict(color=PAL["hist"], width=2.2)))
    fig.add_vline(x=r["T"].to_pydatetime(), line_dash="dot", line_color=CHART_TEXT,
                  annotation_text="forecast issued", annotation_font=dict(size=11, color=CHART_TEXT))
    fig.add_vline(x=r["actual_peak_time"].to_pydatetime(), line_dash="dash", line_color=PAL["bad"],
                  annotation_text="actual peak", annotation_font=dict(size=11, color=PAL["bad"]))
    st.plotly_chart(chart_layout(fig, 320), width="stretch")
    inside_pct = float(((win["actual"] >= win["p10"]) & (win["actual"] <= win["p90"])).mean() * 100)
    st.caption(f"The verdict above is measured at the peak — the moment that matters most for "
               f"planning. Across the whole 24-hour window, actual demand fell inside the predicted "
               f"range for {inside_pct:.0f}% of blocks; where the actual line sits above the band, "
               f"the forecast was running low at that hour.")

    with st.expander("Technical detail"):
        st.write(f"- Issue time T: **{r['T']}** — leakage assertion passed "
                 f"(no block in this window draws on information after T)")
        st.write(f"- Predicted peak **{r['pred_peak_mw']:,.1f} MW** at {r['pred_peak_time']}; "
                 f"P10 {r['p10_at_pred_peak']:,.1f} / P90 {r['p90_at_pred_peak']:,.1f} MW")
        st.write(f"- Actual peak **{r['actual_peak_mw']:,.1f} MW** at {r['actual_peak_time']}")
        st.write(f"- Error at peak: **{r['peak_abs_err_mw']:+,.1f} MW ({r['peak_pct_err']:+.2f}%)**")
        st.write(f"- Actual inside P10–P90 at its own peak: **{inside}**, margin to P90 "
                 f"**{r['p90_margin_at_actual_peak_mw']:+,.1f} MW**")
        st.write(f"- Mean absolute percentage error across this window: **{r['mape_full_pct']:.2f}%** "
                 f"({96 - r['n_missing']} scoreable blocks)")
        st.write(f"- Grid condition score: **{r['stress_score']:.1f} ({r['stress_level']})**, "
                 f"peak risk **{r['peak_risk']}**")
        st.markdown("**Reliability reasons for this window**")
        for x in r["reliability_reasons"]:
            st.write(f"- {x}")
        st.caption("Reliability here covers the full 24-hour window this forecast was issued for. The "
                   "Overview tab reports reliability for a calendar day, so the same date can carry a "
                   "different label in the two views — they answer different questions by design.")
    data_notes("This window's error figures describe **this single event**, not the model's overall "
               "accuracy across the test period.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    h1, h2 = st.columns([2.2, 1])
    with h1:
        page_title()
    with h2:
        st.markdown(
            f'<div style="text-align:right;font-size:0.78rem;color:inherit;opacity:0.68;padding-top:14px">'
            f'<span title="{G["historical"]}">Recorded data · 1 May – 30 Jun 2025 ⓘ</span><br>'
            f'<span title="{G["forecast"]}">Forecasts issued 24 h ahead ⓘ</span></div>',
            unsafe_allow_html=True)

    B, err = safe_backend()
    if err is not None:
        st.error(f"The dashboard could not start: {err}")
        st.stop()

    t1, t2, t3 = st.tabs(["  Overview  ", "  What-If  ", "  Proof Mode  "])
    for tab, fn, name in ((t1, render_overview, "Overview"),
                          (t2, render_whatif, "What-If"),
                          (t3, render_proof, "Proof Mode")):
        with tab:
            try:
                fn(B)
            except Exception:  # noqa: BLE001 -- never show a raw traceback to a viewer
                st.error(f"Something went wrong loading the {name} view.")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
