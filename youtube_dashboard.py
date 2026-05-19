"""
CoreAI — YouTube Predictive Virality Intelligence Platform
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="CoreAI — Predictive Virality",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================
# GLOBAL THEME CSS
# ======================================

st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────── */
.stApp { background-color: #060b18; color: #d0dae8; }
.block-container { padding: 1rem 1.5rem 2rem 1.5rem !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0b1525;
    border-right: 1px solid #192840;
}
[data-testid="stSidebar"] .stRadio label {
    color: #7a90a8 !important;
    font-size: 13px !important;
    padding: 6px 10px;
    border-radius: 6px;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0e1c30 !important;
    border-color: #1d3158 !important;
    color: #d0dae8 !important;
}

/* ── KPI Cards ───────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(145deg, #0e1c30, #162440);
    border: 1px solid #1d3158;
    border-radius: 10px;
    padding: 16px 20px 14px 20px;
    height: 100%;
}
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5a7090;
    margin-bottom: 8px;
}
.kpi-value-orange { font-size: 44px; font-weight: 800; color: #ff6a2f; line-height: 1; }
.kpi-value-blue   { font-size: 44px; font-weight: 800; color: #3bbdff; line-height: 1; }
.kpi-value-red    { font-size: 44px; font-weight: 800; color: #ff3c3c; line-height: 1; }
.kpi-sub-orange   { font-size: 11px; color: #ff6a2f; margin-top: 5px; }
.kpi-sub-blue     { font-size: 11px; color: #3bbdff; margin-top: 5px; }

/* ── Section Headers ─────────────────────────────────────── */
.sec-hdr {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #3bbdff;
    border-bottom: 1px solid #192840;
    padding-bottom: 6px;
    margin: 14px 0 10px 0;
}

/* ── RAVS bars ───────────────────────────────────────────── */
.ravs-row        { margin-bottom: 13px; }
.ravs-row-hdr    { display:flex; justify-content:space-between; margin-bottom:4px; }
.ravs-lbl        { font-size: 12px; color: #b0c4d8; }
.ravs-track      { background: #0a1422; border-radius: 4px; height: 9px; overflow: hidden; }
.ravs-fill       { height: 100%; border-radius: 4px; }

/* ── Narrative table ─────────────────────────────────────── */
.nar-table  { background:#0e1c30; border-radius:8px; border:1px solid #1d3158; overflow:hidden; }
.nar-thead  { display:grid; grid-template-columns:1fr 72px 90px; padding:8px 14px;
              background:#152035; font-size:10px; font-weight:700; color:#3bbdff;
              letter-spacing:1.5px; text-transform:uppercase; }
.nar-row    { display:grid; grid-template-columns:1fr 72px 90px; padding:8px 14px;
              border-top:1px solid #192840; font-size:11px; align-items:center; }
.nar-pill   { display:inline-block; padding:2px 9px; border-radius:10px;
              font-weight:700; font-size:11px; text-align:center; }

/* ── Tier cards ──────────────────────────────────────────── */
.tier-card  { background:#0e1c30; border-radius:8px; padding:12px 10px; min-height:118px; }
.tier-name  { font-size:11px; font-weight:700; letter-spacing:1px; margin-bottom:2px; }
.tier-range { font-size:10px; color:#506070; margin-bottom:8px; }
.tier-item  { font-size:10px; color:#b0c4d8; margin:3px 0; }

/* ── Streamlit overrides ─────────────────────────────────── */
.stButton > button {
    background:#162440 !important; color:#3bbdff !important;
    border:1px solid #1d3158 !important; border-radius:6px !important;
    font-size:12px !important;
}
.stButton > button:hover { background:#1d3158 !important; }
div[data-testid="stDataFrame"] { background:#0e1c30; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#060b18; }
::-webkit-scrollbar-thumb { background:#1d3158; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ======================================
# CONSTANTS
# ======================================

PLOT_BG    = "#0a1422"
PLOT_PAPER = "#060b18"
PLOT_GRID  = "#152035"
PLOT_TEXT  = "#b0c4d8"
ORANGE     = "#ff6a2f"
BLUE       = "#3bbdff"
RED        = "#ff3c3c"
GREEN      = "#3dbb6a"

# Indication groups — drives sidebar keyword chips and data filtering
INDICATION_GROUPS = {
    "All Products": [],
    "Contraception": [
        "nuvaring", "nexplanon", "implanon", "mercilon", "marvelon",
        "kyleena", "mirena", "skyla", "liletta", "paragard", "depo-provera",
        "xulane", "annovera", "opill", "twirla", "phexxi",
        "contraception", "birth control", "iud", "larc",
    ],
    "Fertility / IVF": [
        "puregon", "follistim", "elonva", "orgalutran", "pregnyl", "prometrium",
        "menopur", "rekovelle", "gonal", "fyremadel", "decapeptyl",
        "ivf", "fertility", "infertility", "ovarian stimulation",
        "ovulation induction", "gonadotropin", "embryo transfer",
    ],
    "Dermatology": [
        "vtama", "tapinarof",
        "dupixent", "dupilumab", "rinvoq", "upadacitinib",
        "tremfya", "guselkumab", "adbry", "tralokinumab",
        "cibinqo", "abrocitinib", "eucrisa", "crisaborole",
        "psoriasis", "atopic dermatitis", "eczema", "pruritus",
    ],
    "Immunology": [
        "renflexis", "hadlima", "brenzys", "bildyos", "bilprevda",
        "humira", "adalimumab", "amjevita", "imraldi", "yusimry",
        "biosimilar", "rheumatoid arthritis", "crohn disease",
        "ulcerative colitis", "ankylosing spondylitis",
    ],
    "Respiratory": [
        "singulair", "dulera", "asthmanex", "nasonex", "montelukast",
        "asthma", "allergic rhinitis", "eosinophilic asthma", "allergy",
        "inhaled corticosteroid", "bronchodilator",
    ],
    "Migraine": [
        "maxalt", "rizatriptan", "emgality", "rayvow",
        "migraine", "acute migraine", "cgrp", "headache",
    ],
    "Women's Health": [
        "xaciato", "duavive", "prometrium", "solosec",
        "bacterial vaginosis", "vaginal infection", "menopause",
        "hormone replacement therapy", "hrt", "hot flashes",
    ],
    "Oncology": [
        "ontruzant", "aybintio", "poherdy",
        "trastuzumab", "her2", "breast cancer biosimilar",
    ],
    "Cardiovascular": [
        "zetia", "atozet", "rosuzet", "coza", "ezetimibe",
        "hyperlipidemia", "cholesterol", "ldl", "statin", "dyslipidemia",
    ],
    "Hair Loss": [
        "propecia", "finasteride",
        "androgenetic alopecia", "male pattern baldness", "hair loss",
    ],
}

# ======================================
# DATA LOADING
# ======================================

@st.cache_data(ttl=300)
def load_all_data():
    def safe_csv(name):
        p = Path(name)
        return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()

    narrative_df  = safe_csv("youtube_narrative_detection.csv")
    prediction_df = safe_csv("youtube_virality_predictions.csv")
    alert_df      = safe_csv("youtube_alerts.csv")
    features_df   = safe_csv("youtube_video_data_cleaned_features.csv")

    if len(prediction_df) > 0 and len(narrative_df) > 0 and "video_id" in narrative_df.columns:
        merge_cols = [c for c in ["video_id", "viral_probability", "predicted_viral"] if c in prediction_df.columns]
        preds = prediction_df[merge_cols].drop_duplicates("video_id")
        narrative_df = narrative_df.merge(preds, on="video_id", how="left")

    return narrative_df, alert_df, features_df


# ======================================
# ANALYTICS HELPERS
# ======================================

def n(val, default=0.0):
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except Exception:
        return default


def time_to_virality(vp: float):
    if vp >= 0.88: return "< 1 hr",   "Already surging"
    if vp >= 0.80: return "~1–2 hrs",  "Critical threshold"
    if vp >= 0.65: return "~3.2 hrs",  "Estimated soon"
    if vp >= 0.50: return "~6–12 hrs", "Moderate trajectory"
    return "> 24 hrs", "Low momentum"


def response_window(ttv: str):
    return {
        "< 1 hr":    ("< 1 hr",   "Immediate action required"),
        "~1–2 hrs":  ("1–2 hrs",  "Act within the hour"),
        "~3.2 hrs":  ("6–8 hrs",  "Before peak reach"),
        "~6–12 hrs": ("6–8 hrs",  "Before peak reach"),
        "> 24 hrs":  ("12–24 hrs","Monitor and prepare"),
    }.get(ttv, ("6–8 hrs", "Before peak reach"))


def risk_tier(score: int):
    if score >= 70: return "High Risk",   RED
    if score >= 35: return "Medium Risk", "#ff9900"
    return "Low Risk", GREEN


def engagement_trajectory(views_now: float, vph: float, accel: float, hours: int = 24):
    t    = np.linspace(0, hours, 300)
    k    = max(views_now * 80, 50_000)
    r    = min(max(vph / max(views_now, 1), 0.008), 0.45)
    traj = k / (1 + ((k - views_now) / max(views_now, 1)) * np.exp(-r * t))
    if accel > 0:
        traj += accel * t * np.exp(-t / 5) * views_now * 0.08
    return t, traj


def signal_contributions(row: dict) -> dict:
    vp    = n(row.get("viral_probability",  0))
    vph   = n(row.get("views_per_hour",     0))
    subs  = n(row.get("channel_subscriber_count", row.get("subscriber_count", 0)))
    pharm = n(row.get("pharma_risk_score",  0))
    tox   = n(row.get("toxicity_score",     0))
    vol   = n(row.get("cluster_volume",     0))

    return {
        "Engagement Velocity":    min(100, max(40, int(vp * 95))),
        "Influencer Reach":       min(100, max(28, int(subs / 500_000 * 75))) if subs > 0 else 38,
        "Regulatory Sensitivity": min(100, max(25, int(pharm / 5 * 68)))      if pharm > 0 else 42,
        "Cross-Platform Spread":  min(100, max(18, int(vol / 50 * 55)))        if vol > 0  else 33,
        "Safety Amplification":   min(100, max(15, int(tox * 100))),
    }


def ravs_components(row: dict) -> dict:
    vp    = n(row.get("viral_probability",  0))
    pharm = n(row.get("pharma_risk_score",  0))
    tox   = n(row.get("toxicity_score",     0))
    subs  = n(row.get("channel_subscriber_count", row.get("subscriber_count", 0)))
    vol   = n(row.get("cluster_volume",     0))

    speed  = min(100, max(40, int(vp * 100)))
    reg    = min(100, max(35, int(pharm / 5 * 82) if pharm > 0 else int(vp * 75)))
    safety = min(100, max(20, int(tox * 100)))
    inf    = min(100, max(25, int(subs / 500_000 * 61))) if subs > 0 else 35
    topic  = min(100, max(25, int(vol / 50 * 59) if vol > 0 else int(vp * 60)))

    # Use pre-computed columns if available
    for attr, key in [
        ("speed",  "ravs_speed_of_engagement"),
        ("reg",    "ravs_regulatory_relevance"),
        ("safety", "ravs_safety_implications"),
        ("inf",    "ravs_influencer_reach"),
        ("topic",  "ravs_topic_sensitivity"),
    ]:
        if key in row and n(row[key]) > 0:
            locals_val = int(n(row[key]))
            if   attr == "speed":  speed  = locals_val
            elif attr == "reg":    reg    = locals_val
            elif attr == "safety": safety = locals_val
            elif attr == "inf":    inf    = locals_val
            elif attr == "topic":  topic  = locals_val

    total = int(speed * 0.30 + reg * 0.25 + safety * 0.20 + inf * 0.15 + topic * 0.10)
    return {
        "Speed of Engagement":  speed,
        "Regulatory Relevance": reg,
        "Safety Implications":  safety,
        "Influencer Reach":     inf,
        "Topic Sensitivity":    topic,
        "total":                total,
    }


def pick_top_row(df: pd.DataFrame, scenario: str) -> dict:
    """
    Each scenario surfaces a different risk dimension:
      A — Off-label Claims  → highest pharma_risk_score (regulatory/safety)
      B — Clinical Trial    → highest viral_probability  (spreading fast)
      C — Brand Sentiment   → highest toxicity_score     (reputation risk)
    Falls back to viral_probability if the primary column is missing.
    """
    if len(df) == 0:
        return {}

    scenario_sort = {
        "Highest Regulatory Risk":  ["pharma_risk_score", "toxicity_score",    "viral_probability"],
        "Highest Viral Threat":     ["viral_probability",  "trend_score",       "growth_acceleration"],
        "Highest Toxicity Signal":  ["toxicity_score",     "pharma_risk_score", "viral_probability"],
    }

    # pick the sort priority list for this scenario
    sort_priority = next(
        (cols for key, cols in scenario_sort.items() if key in scenario),
        ["viral_probability", "ravs_score"],
    )

    # use the first available column in priority order
    sort_col = next((c for c in sort_priority if c in df.columns), None)

    if sort_col is None:
        return df.iloc[0].to_dict()

    base = df.copy()
    base[sort_col] = pd.to_numeric(base[sort_col], errors="coerce").fillna(0)
    base = base.sort_values(sort_col, ascending=False)
    return base.iloc[0].to_dict()


# ======================================
# CHART BUILDERS
# ======================================

def _dark(fig, height=280):
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, font_color=PLOT_TEXT,
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(gridcolor=PLOT_GRID, showline=False, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=PLOT_GRID, showline=False, tickfont=dict(size=9)),
    )
    return fig


def chart_trajectory(views, vph, accel):
    t, traj = engagement_trajectory(views, vph, accel)
    viral_thresh = traj.max() * 0.82
    peak_idx     = int(np.argmax(traj))

    tick_vals = [0, 1, 3, 6, 12, 18, 24]
    tick_text = ["Now", "1h", "3h", "6h", "12h", "18h", "24h"]

    fig = go.Figure()

    # Shaded "safe zone" below threshold
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([np.full_like(t, viral_thresh), np.zeros_like(t)]),
        fill="toself", fillcolor="rgba(59,189,255,0.04)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Main trajectory
    fig.add_trace(go.Scatter(
        x=t, y=traj,
        fill="tozeroy", fillcolor="rgba(255,106,47,0.13)",
        line=dict(color=ORANGE, width=2.5),
        mode="lines", name="Projected Views",
        hovertemplate="t=%{x:.1f}h · %{y:,.0f} views<extra></extra>",
    ))

    # Viral threshold
    fig.add_hline(
        y=viral_thresh,
        line=dict(color=RED, dash="dash", width=1.2),
        annotation_text="Viral Threshold",
        annotation_font=dict(color=RED, size=9),
        annotation_position="top right",
    )

    # Now line
    fig.add_vline(
        x=0, line=dict(color=BLUE, dash="dot", width=1.5),
        annotation_text="Now",
        annotation_font=dict(color=BLUE, size=9),
        annotation_position="top left",
    )

    # Peak star
    fig.add_trace(go.Scatter(
        x=[t[peak_idx]], y=[traj[peak_idx]],
        mode="markers",
        marker=dict(color=ORANGE, size=12, symbol="star", line=dict(color="white", width=1.5)),
        name="Predicted Peak",
        hovertemplate="Peak at %{x:.1f}h<br>%{y:,.0f} views<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, font_color=PLOT_TEXT,
        height=265, margin=dict(l=8, r=10, t=10, b=32),
        xaxis=dict(
            gridcolor=PLOT_GRID, showline=False,
            tickvals=tick_vals, ticktext=tick_text, tickfont=dict(size=9),
        ),
        yaxis=dict(
            gridcolor=PLOT_GRID, showline=False, tickfont=dict(size=9),
            title=dict(text="Projected Views", font=dict(size=9, color=PLOT_TEXT)),
        ),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
    )
    return fig


def chart_signal_bars(signals: dict):
    labels = list(signals.keys())
    values = list(signals.values())
    colors = [ORANGE if v >= 65 else BLUE if v >= 40 else "#2a4a6a" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=PLOT_TEXT),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG, font_color=PLOT_TEXT,
        height=205, margin=dict(l=8, r=44, t=6, b=6),
        xaxis=dict(range=[0, 118], showticklabels=False, gridcolor=PLOT_GRID),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", showline=False, tickfont=dict(size=11)),
    )
    return fig


def chart_gauge(score: int, tier: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=54, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=8, color=PLOT_TEXT), tickcolor=PLOT_GRID),
            bar=dict(color=color, thickness=0.28),
            bgcolor=PLOT_BG,
            borderwidth=0,
            steps=[
                dict(range=[0,  35], color="#0a1a0d"),
                dict(range=[35, 70], color="#1a1200"),
                dict(range=[70, 100],color="#1a0505"),
            ],
        ),
        title=dict(text=f"RAVS Score — <b>{tier}</b>", font=dict(size=12, color=color)),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, font_color=PLOT_TEXT,
        height=215, margin=dict(l=20, r=20, t=34, b=5),
    )
    return fig


def chart_event_timing():
    stages = ["Original", "Trend",   "Viral",   "Media",   "Crisis"]
    colors = [BLUE,       "#ff9900", ORANGE,    "#ff5555", "#aa0000"]
    x_vals = [0, 6, 12, 18, 24]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=[0] * 5, mode="lines",
        line=dict(color=PLOT_GRID, width=2, dash="dot"), showlegend=False,
    ))
    for xv, stage, col in zip(x_vals, stages, colors):
        fig.add_trace(go.Scatter(
            x=[xv], y=[0], mode="markers+text",
            marker=dict(size=14, color=col, line=dict(color="white", width=1.5)),
            text=[stage], textposition="bottom center",
            textfont=dict(size=9, color=col), showlegend=False,
        ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        height=88, margin=dict(l=8, r=8, t=6, b=36),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-2, 26]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.6, 0.4]),
    )
    return fig


# ======================================
# SIDEBAR
# ======================================

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style="padding:14px 4px 18px 4px;">
            <span style="font-size:24px;font-weight:900;color:#3bbdff;letter-spacing:.5px;">CORE</span><span style="font-size:24px;font-weight:900;color:#ff6a2f;">AI</span>
            <div style="font-size:9px;color:#3a5a7a;letter-spacing:2px;margin-top:2px;">PREDICTIVE INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:9px;color:#3a5a7a;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:6px;'>Intelligence</div>", unsafe_allow_html=True)

        page = st.radio("nav", [
            "Dashboard",
            "Narrative Monitor",
            "Alerts",
            "Virality Predictor",
            "RAVS Engine",
            "Pharmacovigilance",
        ], label_visibility="collapsed")

        st.markdown("<hr style='border-color:#192840;margin:14px 0;'>", unsafe_allow_html=True)

        # ── Indication / Keyword Filter ──────────────────────
        st.markdown(
            "<div style='font-size:9px;color:#3a5a7a;letter-spacing:2.5px;"
            "text-transform:uppercase;margin-bottom:8px;'>Indication Area</div>",
            unsafe_allow_html=True,
        )

        indication = st.selectbox(
            "indication",
            list(INDICATION_GROUPS.keys()),
            label_visibility="collapsed",
        )

        # Show keyword chips for the selected indication
        kw_list = INDICATION_GROUPS.get(indication, [])
        if kw_list:
            chips_html = " ".join(
                f"<span style='display:inline-block;background:#0e1c30;"
                f"border:1px solid #1d3158;border-radius:10px;padding:2px 8px;"
                f"font-size:9px;color:#3bbdff;margin:2px 1px;'>{k}</span>"
                for k in kw_list[:18]  # cap at 18 to avoid overflow
            )
            st.markdown(
                f"<div style='line-height:1.8;margin-bottom:10px;'>{chips_html}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#192840;margin:12px 0;'>", unsafe_allow_html=True)

        if st.button("⟳  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            f"<div style='font-size:9px;color:#3a5a7a;margin-top:6px;'>Auto-refresh 5 min · {pd.Timestamp.utcnow().strftime('%H:%M')} UTC</div>",
            unsafe_allow_html=True,
        )

    return page, indication


# ======================================
# PAGE: VIRALITY PREDICTOR
# ======================================

def page_virality_predictor(df: pd.DataFrame):
    scenarios = [
        "Highest Regulatory Risk",
        "Highest Viral Threat",
        "Highest Toxicity Signal",
    ]
    scenario = st.radio("scenario", scenarios, horizontal=True, label_visibility="collapsed")
    st.markdown("<hr style='border-color:#192840;margin:4px 0 16px 0;'>", unsafe_allow_html=True)

    row    = pick_top_row(df, scenario)
    vp     = n(row.get("viral_probability",  0.91))
    vph    = n(row.get("views_per_hour",      800))
    views  = n(row.get("view_count",         5000))
    accel  = n(row.get("growth_acceleration",  40))
    rv     = ravs_components(row)
    r_score = rv["total"]

    ttv_label, ttv_sub = time_to_virality(vp)
    rw_label,  rw_sub  = response_window(ttv_label)
    tier_name, tier_color = risk_tier(r_score)

    # ── KPI Cards ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Viral Probability</div>
            <div class="kpi-value-orange">{int(vp*100)}%</div>
            <div class="kpi-sub-orange">Critical Threshold</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Time to Virality</div>
            <div class="kpi-value-blue">{ttv_label}</div>
            <div class="kpi-sub-blue">{ttv_sub}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">RAVS Score</div>
            <div style="font-size:44px;font-weight:800;color:{tier_color};line-height:1;">{r_score}</div>
            <div style="font-size:11px;color:{tier_color};margin-top:5px;">Out of 100</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Response Window</div>
            <div class="kpi-value-blue">{rw_label}</div>
            <div class="kpi-sub-blue">{rw_sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="sec-hdr">PREDICTED ENGAGEMENT TRAJECTORY</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_trajectory(views, vph, accel),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            f"<div style='font-size:10px;color:{ORANGE};text-align:right;margin-top:-10px;'>"
            f"⚠ {rw_label} proactive window before peak reach</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown('<div class="sec-hdr">SIGNAL CONTRIBUTION</div>', unsafe_allow_html=True)
        sigs = signal_contributions(row)
        st.plotly_chart(
            chart_signal_bars(sigs),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown('<div class="sec-hdr">EVENT TIMING</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_event_timing(),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ======================================
# PAGE: RAVS ENGINE
# ======================================

def page_ravs_engine(df: pd.DataFrame, alert_df: pd.DataFrame):
    if len(df) == 0:
        st.info("No data available — run the pipeline first.")
        return

    sort_col = next((c for c in ["ravs_score", "viral_probability", "virality_score"] if c in df.columns), None)
    if sort_col:
        _tmp = df.copy()
        _tmp[sort_col] = pd.to_numeric(_tmp[sort_col], errors="coerce").fillna(0)
        top = _tmp.sort_values(sort_col, ascending=False).iloc[0].to_dict()
    else:
        top = df.iloc[0].to_dict()

    comps = ravs_components(top)
    total = comps["total"]
    tier_name, tier_color = risk_tier(total)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    left_col, mid_col, right_col = st.columns([2, 2, 3])

    # ── Component bars ───────────────────────────────────────
    with left_col:
        st.markdown('<div class="sec-hdr">RAVS COMPONENTS</div>', unsafe_allow_html=True)

        for label in ["Speed of Engagement", "Regulatory Relevance", "Safety Implications", "Influencer Reach", "Topic Sensitivity"]:
            pct = comps[label]
            bar_color = RED if pct >= 70 else "#ff9900" if pct >= 45 else BLUE
            st.markdown(f"""
            <div class="ravs-row">
                <div class="ravs-row-hdr">
                    <span class="ravs-lbl">{label}</span>
                    <span style="font-size:12px;font-weight:700;color:{bar_color};">{pct}%</span>
                </div>
                <div class="ravs-track">
                    <div class="ravs-fill" style="width:{pct}%;background:linear-gradient(90deg,{BLUE},{bar_color});"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Gauge ────────────────────────────────────────────────
    with mid_col:
        st.markdown('<div class="sec-hdr">RAVS SCORE</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_gauge(total, tier_name, tier_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(f"""
        <div style="text-align:center;padding:9px 10px;background:#0e1c30;
                    border-radius:8px;border:1px solid {tier_color}44;margin-top:-8px;">
            <div style="font-size:10px;color:{tier_color};font-weight:600;">
                Alert sent to Comms &amp; Regulatory Teams
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Risk tiers ───────────────────────────────────────────
    with right_col:
        st.markdown('<div class="sec-hdr">RISK CLASSIFICATION</div>', unsafe_allow_html=True)

        tiers_def = [
            ("CRITICAL", "70–100", RED,      ["Immediate escalation", "PV + Regulatory + Comms", "CEO brief required"]),
            ("REGION",   "35–69",  "#ff9900", ["Watch list", "Weekly review", "No immediate action"]),
            ("LOW",      "0–34",   GREEN,    ["Log only", "Monthly report", "No escalation needed"]),
        ]

        tc1, tc2, tc3 = st.columns(3)
        for col_obj, (name, rng, col, items) in zip([tc1, tc2, tc3], tiers_def):
            active = (
                (name == "CRITICAL" and total >= 70) or
                (name == "REGION"   and 35 <= total < 70) or
                (name == "LOW"      and total < 35)
            )
            border = f"2px solid {col}" if active else f"1px solid {col}44"
            items_html = "".join(f'<div class="tier-item">• {i}</div>' for i in items)
            with col_obj:
                st.markdown(f"""
                <div class="tier-card" style="border:{border};border-radius:8px;">
                    <div class="tier-name" style="color:{col};">{name}</div>
                    <div class="tier-range">({rng})</div>
                    {items_html}
                </div>""", unsafe_allow_html=True)

    # ── Top Scored Narratives ────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">TOP SCORED NARRATIVES</div>', unsafe_allow_html=True)

    score_col = next((c for c in ["ravs_score", "viral_probability"] if c in df.columns), None)

    if score_col and "cluster_topic_name" in df.columns:
        _tmp = df.copy()
        _tmp[score_col] = pd.to_numeric(_tmp[score_col], errors="coerce").fillna(0)

        agg_dict = {score_col: "mean", "video_id": "count"}
        if "trend_score" in _tmp.columns:
            agg_dict["trend_score"] = "mean"

        top10 = (
            _tmp.groupby("cluster_topic_name")
            .agg(agg_dict)
            .reset_index()
            .rename(columns={"video_id": "video_count"})
            .sort_values(score_col, ascending=False)
            .head(10)
        )

        st.markdown("""
        <div class="nar-table">
            <div class="nar-thead">
                <div>NARRATIVE TOPIC</div>
                <div style="text-align:center;">RAVS</div>
                <div style="text-align:center;">TREND</div>
            </div>
        """, unsafe_allow_html=True)

        for _, row_data in top10.iterrows():
            txt = str(row_data["cluster_topic_name"])
            txt = txt[:88] + "…" if len(txt) > 88 else txt
            sv  = n(row_data[score_col])
            score_disp = int(sv * 100) if sv <= 1.0 else int(sv)
            ts  = n(row_data.get("trend_score", 0)) if "trend_score" in row_data.index else 0
            tr_label = "↑ Rising" if ts > 0.3 else "→ Stable"
            tr_color = ORANGE if ts > 0.3 else BLUE
            _, pill_col = risk_tier(score_disp)

            st.markdown(f"""
            <div class="nar-row">
                <div style="color:#b0c4d8;">{txt}</div>
                <div style="text-align:center;">
                    <span class="nar-pill" style="background:{pill_col}22;color:{pill_col};">{score_disp}</span>
                </div>
                <div style="text-align:center;color:{tr_color};font-size:10px;font-weight:600;">{tr_label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    elif score_col:
        title_col = next((c for c in ["topic_keywords", "title_en", "title", "analysis_text_clean"] if c in df.columns), None)
        if title_col:
            top10 = (
                df[[title_col, score_col] + ([c for c in ["trend_score"] if c in df.columns])]
                .assign(**{score_col: lambda x: pd.to_numeric(x[score_col], errors="coerce").fillna(0)})
                .sort_values(score_col, ascending=False)
                .head(10)
            )

            st.markdown("""
            <div class="nar-table">
                <div class="nar-thead">
                    <div>NARRATIVE</div>
                    <div style="text-align:center;">RAVS</div>
                    <div style="text-align:center;">TREND</div>
                </div>
            """, unsafe_allow_html=True)

            for _, row_data in top10.iterrows():
                txt = str(row_data[title_col])
                txt = txt[:88] + "…" if len(txt) > 88 else txt
                sv  = n(row_data[score_col])
                score_disp = int(sv * 100) if sv <= 1.0 else int(sv)
                ts  = n(row_data.get("trend_score", 0)) if "trend_score" in row_data.index else 0
                tr_label = "↑ Rising" if ts > 0.3 else "→ Stable"
                tr_color = ORANGE if ts > 0.3 else BLUE
                _, pill_col = risk_tier(score_disp)

                st.markdown(f"""
                <div class="nar-row">
                    <div style="color:#b0c4d8;">{txt}</div>
                    <div style="text-align:center;">
                        <span class="nar-pill" style="background:{pill_col}22;color:{pill_col};">{score_disp}</span>
                    </div>
                    <div style="text-align:center;color:{tr_color};font-size:10px;font-weight:600;">{tr_label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ======================================
# PAGE: DASHBOARD
# ======================================

def page_dashboard(df: pd.DataFrame, alert_df: pd.DataFrame):
    st.markdown('<div class="sec-hdr">PLATFORM OVERVIEW</div>', unsafe_allow_html=True)

    kpis = [
        ("Total Videos",        len(df)          if len(df) > 0 else "—",  BLUE),
        ("Active Alerts",       len(alert_df)     if len(alert_df) > 0 else "—", RED),
        ("Emerging Narratives",
         int(df["emerging_narrative"].eq("emerging").sum()) if "emerging_narrative" in df.columns else "—",
         ORANGE),
        ("High Risk (RAVS≥70)",
         int((df["ravs_score"] >= 70).sum()) if "ravs_score" in df.columns else "—",
         RED),
        ("Avg Viral Score",
         f"{df['viral_probability'].mean():.2f}" if "viral_probability" in df.columns and len(df) > 0 else "—",
         BLUE),
    ]

    cols = st.columns(5)
    for col, (lbl, val, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div style="font-size:32px;font-weight:800;color:{color};line-height:1.1;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="sec-hdr">VIRAL PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
        if "viral_probability" in df.columns and len(df) > 0:
            fig = px.histogram(
                df.dropna(subset=["viral_probability"]),
                x="viral_probability", nbins=20,
                color_discrete_sequence=[BLUE],
            )
            _dark(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with cr:
        st.markdown('<div class="sec-hdr">SENTIMENT DISTRIBUTION</div>', unsafe_allow_html=True)
        if "sentiment_label" in df.columns and len(df) > 0:
            counts = df["sentiment_label"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]
            fig = px.pie(
                counts, names="sentiment", values="count",
                color_discrete_sequence=[BLUE, ORANGE, RED, GREEN, "#aa00ff"],
            )
            fig.update_layout(
                paper_bgcolor=PLOT_PAPER, font_color=PLOT_TEXT,
                height=280, margin=dict(l=8, r=8, t=8, b=8),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if len(alert_df) > 0:
        st.markdown('<div class="sec-hdr">ACTIVE ALERTS</div>', unsafe_allow_html=True)
        disp = [c for c in ["title", "alert_type", "alert_priority", "ravs_score", "viral_probability", "pharma_risk_score"] if c in alert_df.columns]
        st.dataframe(alert_df[disp].head(10), use_container_width=True, hide_index=True)


# ======================================
# PAGE: NARRATIVE MONITOR
# ======================================

def page_narrative_monitor(df: pd.DataFrame):
    if len(df) == 0:
        st.info("No narrative data. Run the pipeline first.")
        return

    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="sec-hdr">CLUSTER VIRAL PROBABILITY</div>', unsafe_allow_html=True)
        if "narrative_cluster_id" in df.columns and "viral_probability" in df.columns:
            cdf = (
                df.groupby("narrative_cluster_id")
                  .agg(count=("video_id", "count"), avg_vp=("viral_probability", "mean"))
                  .reset_index()
                  .sort_values("avg_vp", ascending=False)
                  .head(15)
            )
            fig = px.bar(cdf, x="avg_vp", y="narrative_cluster_id", orientation="h",
                         color="avg_vp", color_continuous_scale=[PLOT_BG, ORANGE])
            _dark(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with cr:
        st.markdown('<div class="sec-hdr">TOPIC VOLUME vs VIRALITY</div>', unsafe_allow_html=True)
        if "topic_keywords" in df.columns and "viral_probability" in df.columns:
            tdf = (
                df.groupby("topic_keywords")
                  .agg(count=("video_id", "count"), avg_vp=("viral_probability", "mean"))
                  .reset_index()
                  .sort_values("avg_vp", ascending=False)
                  .head(12)
            )
            fig = px.scatter(tdf, x="count", y="avg_vp", size="count",
                             color="avg_vp", color_continuous_scale=[PLOT_BG, ORANGE],
                             hover_data=["topic_keywords"])
            _dark(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec-hdr">NARRATIVE FEED</div>', unsafe_allow_html=True)
    feed_cols = [c for c in ["title", "topic_keywords", "viral_probability", "ravs_score", "sentiment_label", "emerging_narrative"] if c in df.columns]
    sort_df = df.sort_values("viral_probability", ascending=False) if "viral_probability" in df.columns else df
    st.dataframe(sort_df[feed_cols].head(50), use_container_width=True, hide_index=True)


# ======================================
# PAGE: ALERTS
# ======================================

def page_alerts(alert_df: pd.DataFrame):
    if len(alert_df) == 0:
        st.info("No alerts yet — run the full pipeline first.")
        return

    c1, c2, c3 = st.columns(3)
    for col_obj, priority, color in zip([c1, c2, c3], ["critical", "high", "medium"], [RED, "#ff9900", BLUE]):
        if "alert_priority" in alert_df.columns:
            cnt = int((alert_df["alert_priority"].str.lower() == priority).sum())
            with col_obj:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{color}55;">
                    <div class="kpi-label">{priority.upper()} ALERTS</div>
                    <div style="font-size:38px;font-weight:800;color:{color};line-height:1;">{cnt}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">ALERT DETAILS</div>', unsafe_allow_html=True)
    disp = [c for c in ["title", "alert_type", "alert_priority", "alert_reason", "ravs_score", "viral_probability", "pharma_risk_score"] if c in alert_df.columns]
    st.dataframe(alert_df[disp], use_container_width=True, hide_index=True)


# ======================================
# PAGE: PHARMACOVIGILANCE
# ======================================

def page_pharmacovigilance(df: pd.DataFrame):
    st.markdown('<div class="sec-hdr">SAFETY SIGNAL MONITORING</div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.info("No data available.")
        return

    c1, c2, c3 = st.columns(3)
    for col_obj, metric, lbl, color in zip(
        [c1, c2, c3],
        ["toxicity_score", "pharma_risk_score", "viral_probability"],
        ["Avg Toxicity",   "Avg Pharma Risk",    "Avg Viral Score"],
        [RED,              "#ff9900",             BLUE],
    ):
        if metric in df.columns:
            val = round(float(df[metric].mean()), 3)
            with col_obj:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{lbl}</div>
                    <div style="font-size:34px;font-weight:800;color:{color};line-height:1.1;">{val}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="sec-hdr">TOXICITY vs VIRALITY</div>', unsafe_allow_html=True)
        if "toxicity_score" in df.columns and "viral_probability" in df.columns:
            fig = px.scatter(
                df.dropna(subset=["toxicity_score", "viral_probability"]),
                x="toxicity_score", y="viral_probability",
                color="pharma_risk_label" if "pharma_risk_label" in df.columns else None,
                color_discrete_sequence=[BLUE, ORANGE, RED, "#aa00ff"],
                opacity=0.65,
            )
            _dark(fig, 280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with cr:
        st.markdown('<div class="sec-hdr">PHARMA RISK DISTRIBUTION</div>', unsafe_allow_html=True)
        if "pharma_risk_label" in df.columns:
            rdf = df["pharma_risk_label"].value_counts().reset_index()
            rdf.columns = ["label", "count"]
            fig = px.bar(rdf, x="label", y="count", color="label",
                         color_discrete_sequence=[ORANGE, BLUE, RED])
            _dark(fig, 280)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ======================================
# MAIN
# ======================================

narrative_df, alert_df, features_df = load_all_data()

main_df = narrative_df if len(narrative_df) > 0 else features_df

_needs_ravs = (
    len(main_df) > 0 and (
        "ravs_score" not in main_df.columns
        or main_df["ravs_score"].isna().all()
    )
)
if _needs_ravs:
    try:
        from youtube_alert_engine import compute_ravs_score
        main_df = compute_ravs_score(main_df.copy())
    except Exception:
        main_df["ravs_score"] = 0.0

page, indication = render_sidebar(main_df)

df = main_df.copy()
if indication != "All Products" and len(df) > 0:
    kws = INDICATION_GROUPS.get(indication, [])
    if kws and "keyword" in df.columns:
        pattern = "|".join(re.escape(k) for k in kws)
        mask = df["keyword"].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
        if mask.sum() > 0:
            df = df[mask]

PAGE_TITLES = {
    "Dashboard":          "Platform Overview",
    "Narrative Monitor":  "Narrative Monitor",
    "Alerts":             "Active Alerts",
    "Virality Predictor": "Virality Predictor",
    "RAVS Engine":        "RAVS Engine",
    "Pharmacovigilance":  "Pharmacovigilance",
}
st.markdown(
    f"<h2 style='color:#d0dae8;font-size:18px;font-weight:700;margin:0 0 2px 0;"
    f"padding:0;letter-spacing:.5px;'>{PAGE_TITLES.get(page, page)}</h2>",
    unsafe_allow_html=True,
)

if   page == "Dashboard":          page_dashboard(df, alert_df)
elif page == "Narrative Monitor":  page_narrative_monitor(df)
elif page == "Alerts":             page_alerts(alert_df)
elif page == "Virality Predictor": page_virality_predictor(df)
elif page == "RAVS Engine":        page_ravs_engine(df, alert_df)
elif page == "Pharmacovigilance":  page_pharmacovigilance(df)
