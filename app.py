"""
KosWatt - Smart Energy Agentic AI
Streamlit Exhibition Dashboard

Entry point: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Import core agent ────────────────────────────────────────────────────────
# The agent module is extracted from the notebook as a standalone importable.
# If running from the project root, koswatt_agent.py must be present.
from koswatt_agent import core_koswatt_agent


# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "KosWatt | Smart Energy AI",
    page_icon   = "W",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)


# ─── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d0d0d;
        color: #e8e8e8;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2a2a2a;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1300px;
    }

    /* ── Header ── */
    .kw-header {
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 1.2rem;
        margin-bottom: 2rem;
    }
    .kw-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .kw-subtitle {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Section labels ── */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 1rem;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e1e1e;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #555;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1;
    }
    .metric-unit {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #444;
        margin-left: 4px;
    }

    /* ── Device status pill ── */
    .device-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .pill-on {
        display: inline-block;
        background: #1a2e1a;
        color: #4ade80;
        border: 1px solid #2d5a2d;
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        letter-spacing: 1px;
    }
    .pill-off {
        display: inline-block;
        background: #1a1a1a;
        color: #555;
        border: 1px solid #2a2a2a;
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        letter-spacing: 1px;
    }
    .pill-eco {
        display: inline-block;
        background: #1a2a2e;
        color: #38bdf8;
        border: 1px solid #1e4a5a;
        border-radius: 2px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        letter-spacing: 1px;
    }

    /* ── Result panels ── */
    .panel-stable {
        background: #0d1f0d;
        border: 1px solid #1f4a1f;
        border-left: 4px solid #4ade80;
        border-radius: 4px;
        padding: 1.5rem 2rem;
    }
    .panel-alert {
        background: #1f0d0d;
        border: 1px solid #4a1f1f;
        border-left: 4px solid #f87171;
        border-radius: 4px;
        padding: 1.5rem 2rem;
    }
    .panel-medium {
        background: #1f1a0d;
        border: 1px solid #4a3a1f;
        border-left: 4px solid #fbbf24;
        border-radius: 4px;
        padding: 1.5rem 2rem;
    }
    .panel-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* ── Action list ── */
    .action-item {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #f87171;
        padding: 6px 0;
        border-bottom: 1px solid #2a1a1a;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .action-index {
        color: #444;
        font-size: 0.75rem;
        min-width: 20px;
    }

    /* ── Watt delta display ── */
    .watt-delta {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-top: 1.2rem;
    }
    .watt-before {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        color: #555;
        text-decoration: line-through;
    }
    .watt-after {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #4ade80;
    }
    .watt-saved {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #4ade80;
        background: #0d1f0d;
        border: 1px solid #1f4a1f;
        padding: 3px 10px;
        border-radius: 2px;
    }

    /* ── Sidebar overrides ── */
    .sidebar-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #444;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid #1e1e1e;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kw-header">
    <p class="kw-title">KosWatt</p>
    <p class="kw-subtitle">Smart Energy Agentic AI &nbsp;|&nbsp; Boarding House Energy Monitor</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar: Simulation Controls ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family: IBM Plex Mono, monospace; font-size: 1rem; font-weight:600; color:#fff; margin-bottom:4px;">Simulation Control</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem; color:#555; margin-bottom:1.5rem;">Configure room parameters to test the AI agent</p>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Room State</p>', unsafe_allow_html=True)

    occupancy_opt = st.radio(
        "Occupancy",
        options=["Room Empty", "Room Occupied"],
        index=0
    )
    occupancy = 0 if occupancy_opt == "Room Empty" else 1

    temperature = st.slider(
        "Room Temperature (C)",
        min_value=20, max_value=40, value=28, step=1
    )

    time_opt = st.radio(
        "Time of Day",
        options=["Daytime", "Nighttime"],
        index=0
    )
    time_of_day = 0 if time_opt == "Daytime" else 1

    st.markdown('<p class="sidebar-section">Device Switch State</p>', unsafe_allow_html=True)
    st.caption("Initial state before AI intervention")

    status_ac   = st.toggle("AC Unit",  value=True)
    status_lamp = st.toggle("Lamp",     value=False)
    status_tv   = st.toggle("TV",       value=True)


# ─── Core computation ─────────────────────────────────────────────────────────
result = core_koswatt_agent(
    occupancy   = occupancy,
    temperature = temperature,
    time_of_day = time_of_day,
    status_ac   = status_ac,
    status_lamp = status_lamp,
    status_tv   = status_tv
)

fuzzy_score    = result['fuzzy_score']
waste_category = result['waste_category']
action_seq     = result['action_sequence']
initial_watt   = result['initial_watt']
final_state    = result['final_state']
final_watt     = result['final_watt']


# ─── Gauge chart builder ──────────────────────────────────────────────────────
def build_gauge(score):
    """
    Plotly Gauge chart with three color zones:
      0-35  : Green  (Low waste)
      35-65 : Amber  (Medium waste)
      65-100: Red    (High waste)
    """
    if score <= 35:
        needle_color = "#4ade80"
    elif score <= 65:
        needle_color = "#fbbf24"
    else:
        needle_color = "#f87171"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number = {
            'font' : {'size': 40, 'family': 'IBM Plex Mono', 'color': '#ffffff'},
            'suffix': ''
        },
        gauge = {
            'axis': {
                'range'    : [0, 100],
                'tickwidth': 1,
                'tickcolor': "#333",
                'tickfont' : {'color': '#444', 'size': 10},
            },
            'bar'       : {'color': needle_color, 'thickness': 0.22},
            'bgcolor'   : '#141414',
            'borderwidth': 0,
            'steps'     : [
                {'range': [0, 35],  'color': '#0d1f0d'},
                {'range': [35, 65], 'color': '#1f1a0d'},
                {'range': [65, 100],'color': '#1f0d0d'},
            ],
            'threshold' : {
                'line' : {'color': '#555', 'width': 2},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor = '#141414',
        plot_bgcolor  = '#141414',
        font          = {'color': '#aaa', 'family': 'IBM Plex Sans'},
        margin        = dict(l=20, r=20, t=30, b=10),
        height        = 220,
    )
    return fig


# ─── Device status HTML builder ───────────────────────────────────────────────
def device_pill(label, is_on, is_eco=False):
    if is_eco:
        badge = '<span class="pill-eco">ECO</span>'
    elif is_on:
        badge = '<span class="pill-on">ON</span>'
    else:
        badge = '<span class="pill-off">OFF</span>'
    return f'<div class="device-row">{badge} <span style="color:#aaa">{label}</span></div>'


# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1: Current Conditions (Before AI)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">01 &nbsp; Current Conditions &nbsp; / &nbsp; Before AI Intervention</p>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1, 1.6])

with col_a:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Power Draw</div>
        <div>
            <span class="metric-value">{initial_watt}</span>
            <span class="metric-unit">W</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Context summary
    occ_text  = "Occupied" if occupancy else "Empty"
    tod_text  = "Night" if time_of_day else "Day"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Context</div>
        <div style="font-family: IBM Plex Mono, monospace; font-size: 0.85rem; line-height: 2; color: #aaa;">
            Room: <span style="color:#fff">{occ_text}</span><br>
            Temp: <span style="color:#fff">{temperature} C</span><br>
            Time: <span style="color:#fff">{tod_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    ac_state   = device_pill("AC Unit", status_ac)
    lamp_state = device_pill("Lamp",    status_lamp)
    tv_state   = device_pill("TV",      status_tv)
    st.markdown(f"""
    <div class="metric-card" style="height: 100%">
        <div class="metric-label">Device States</div>
        {ac_state}{lamp_state}{tv_state}
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="metric-card"><div class="metric-label">Waste Score</div>', unsafe_allow_html=True)
    st.plotly_chart(build_gauge(fuzzy_score), use_container_width=True, config={'displayModeBar': False})

    # Category badge under gauge
    if waste_category == 'HIGH WASTE':
        badge_style = "background:#2a0000; color:#f87171; border:1px solid #5a2020;"
    elif waste_category == 'MEDIUM WASTE':
        badge_style = "background:#2a1f00; color:#fbbf24; border:1px solid #5a4020;"
    else:
        badge_style = "background:#002a00; color:#4ade80; border:1px solid #205a20;"

    st.markdown(f"""
    <div style="text-align:center; margin-top:-8px;">
        <span style="font-family: IBM Plex Mono, monospace; font-size: 0.75rem;
                     padding: 4px 16px; border-radius: 2px; letter-spacing: 2px;
                     {badge_style}">{waste_category}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2: After AI Automation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">02 &nbsp; After AI Automation &nbsp; / &nbsp; Agent Decision</p>', unsafe_allow_html=True)

if waste_category == 'LOW WASTE':
    # ── Stable: no action needed ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="panel-stable">
        <p class="panel-title" style="color:#4ade80">System Stable</p>
        <p style="font-size:0.95rem; color:#aaa; margin:0;">
            Waste score <strong style="color:#4ade80">{fuzzy_score}</strong> is within acceptable range.
            No automation action required. All devices operating within ethical parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif waste_category == 'MEDIUM WASTE':
    # ── Medium: monitoring, no intervention ───────────────────────────────────
    st.markdown(f"""
    <div class="panel-medium">
        <p class="panel-title" style="color:#fbbf24">Ethical Tolerance Active</p>
        <p style="font-size:0.95rem; color:#aaa; margin:0 0 0.8rem;">
            Waste score <strong style="color:#fbbf24">{fuzzy_score}</strong>.
            Fuzzy guardrails detected contextual justification (thermal comfort or safety lighting).
            Autonomous intervention withheld.
        </p>
        <p style="font-size: 0.8rem; color: #666; margin:0; font-family: IBM Plex Mono, monospace;">
            MONITORING &nbsp; | &nbsp; No STRIPS plan triggered
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── High waste: STRIPS triggered ─────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Action sequence list
        actions_html = ""
        for i, action in enumerate(action_seq, 1):
            actions_html += f"""
            <div class="action-item">
                <span class="action-index">{i:02d}</span>
                <span>{action}</span>
            </div>"""

        if not action_seq:
            actions_html = '<p style="color:#555; font-size:0.85rem;">No further actions required.</p>'

        st.markdown(f"""
        <div class="panel-alert">
            <p class="panel-title" style="color:#f87171">AI Action Triggered</p>
            <p style="font-family: IBM Plex Mono, monospace; font-size: 0.75rem; color:#666; margin-bottom:1rem;">
                STRIPS PLANNER &nbsp; | &nbsp; {len(action_seq)} action(s) queued
            </p>
            {actions_html}
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Final device states after execution
        ac_eco   = final_state.get('ac_eco', False)
        ac_on    = final_state.get('ac_on', False)
        lamp_on  = final_state.get('lamp_on', False)
        tv_on    = final_state.get('tv_on', False)

        ac_final   = device_pill("AC Unit", ac_on,  is_eco=ac_eco)
        lamp_final = device_pill("Lamp",    lamp_on)
        tv_final   = device_pill("TV",      tv_on)

        watt_saved = initial_watt - final_watt

        st.markdown(f"""
        <div class="panel-alert">
            <p class="panel-title" style="color:#f87171">Post-Execution State</p>
            {ac_final}{lamp_final}{tv_final}
            <div class="watt-delta">
                <span class="watt-before">{initial_watt} W</span>
                <span class="watt-after">{final_watt} W</span>
                <span class="watt-saved">-{watt_saved} W saved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top: 1px solid #1e1e1e; padding-top: 1rem;">
    <p style="font-family: IBM Plex Mono, monospace; font-size: 0.65rem;
              color: #333; letter-spacing: 1.5px; text-align: center; margin:0;">
        KOSWATT &nbsp;|&nbsp; SMART ENERGY AGENTIC AI &nbsp;|&nbsp;
        FUZZY LOGIC + STRIPS PLANNING
    </p>
</div>
""", unsafe_allow_html=True)
