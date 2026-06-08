"""
KosWatt - Smart Energy Agentic AI
Streamlit Exhibition Dashboard

Entry point: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from koswatt_agent import (
    core_koswatt_agent,
    apply_actions_to_state,
    calculate_post_action_watt,
    AC_ECO_THRESHOLD,
    AC_ECO_WATTS,
    AC_HOT_WATTS,
    AC_COOL_WATTS,
    LAMP_WATTS,
    TV_WATTS,
    STANDBY_WATTS,
    AUTONOMY_LEVELS,
)

#  Human-readable labels 
ACTION_LABELS = {
    'TURN_OFF_TV'    : 'Turn off TV',
    'TURN_OFF_LAMP'  : 'Turn off lamp',
    'TURN_OFF_AC'    : 'Turn off AC unit',
    'TURN_OFF_AC_ECO': 'Turn off AC unit (was in ECO mode)',
    'SET_AC_TO_ECO'  : 'Set AC to ECO mode',
}
ADVISORY_LABELS = {
    'SUGGEST_AC_ECO'           : 'Switch AC to ECO mode',
    'SUGGEST_LAMP_OFF'         : 'Consider turning off the lamp',
    # Advisory-mode labels (agent suggests but does not execute STRIPS actions)
    'SUGGEST_TURN_OFF_TV'      : 'Turn off TV',
    'SUGGEST_TURN_OFF_LAMP'    : 'Turn off lamp',
    'SUGGEST_TURN_OFF_AC'      : 'Turn off AC unit',
    'SUGGEST_TURN_OFF_AC_ECO'  : 'Turn off AC unit (was in ECO mode)',
    'SUGGEST_SET_AC_TO_ECO'    : 'Set AC to ECO mode',
}

AUTONOMY_DISPLAY = {
    'autonomous': 'Autonomous',
    'confirm':    'Confirm First',
    'advisory':   'Advisory Only',
}


#  Page configuration 
st.set_page_config(
    page_title            = "KosWatt | Smart Energy AI",
    page_icon             = "W",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)


#  Session state initialisation 
if 'log' not in st.session_state:
    st.session_state.log = []
if 'last_log_hash' not in st.session_state:
    st.session_state.last_log_hash = None
if '_dataset' not in st.session_state:
    st.session_state['_dataset'] = None
if '_batch_results' not in st.session_state:
    st.session_state['_batch_results'] = None


#  Global styles 
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

    /*  Header  */
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

    /*  Section labels  */
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

    /*  Metric cards  */
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

    /*  Device status pill  */
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

    /*  Result panels  */
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
    .panel-advisory {
        background: #0d1520;
        border: 1px solid #1a2d40;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 1.2rem 2rem;
        margin-top: 1rem;
    }
    .panel-reasoning {
        background: #111111;
        border: 1px solid #1e1e1e;
        border-left: 4px solid #333333;
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

    /*  Action list (HIGH WASTE)  */
    .action-item {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #f87171;
        padding: 7px 0;
        border-bottom: 1px solid #2a1a1a;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .action-item:last-child {
        border-bottom: none;
    }
    .action-index {
        color: #444;
        font-size: 0.75rem;
        min-width: 20px;
    }

    /*  Advisory action list (MEDIUM WASTE)  */
    .advisory-item {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        padding: 7px 0;
        border-bottom: 1px solid #1a2d40;
        line-height: 1.5;
    }
    .advisory-item:last-child {
        border-bottom: none;
    }
    .advisory-note {
        font-size: 0.75rem;
        color: #4a7a99;
        margin-top: 2px;
    }

    /*  Reasoning lines  */
    .reasoning-line {
        font-size: 0.875rem;
        color: #888888;
        padding: 10px 0;
        border-bottom: 1px solid #1a1a1a;
        line-height: 1.65;
    }
    .reasoning-line:last-child {
        border-bottom: none;
    }

    /*  Watt delta display  */
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

    /*  Sidebar overrides  */
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

    /*  Pending confirmation panel  */
    .panel-confirm {
        background: #0d0d1f;
        border: 1px solid #2a2a5a;
        border-left: 4px solid #818cf8;
        border-radius: 4px;
        padding: 1.5rem 2rem;
    }
    .action-item-pending {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #818cf8;
        padding: 7px 0;
        border-bottom: 1px solid #1a1a3a;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .action-item-pending:last-child { border-bottom: none; }

    /*  Safety override panel  */
    .panel-safety {
        background: #1a0d1f;
        border: 1px solid #4a2a5a;
        border-left: 4px solid #c084fc;
        border-radius: 4px;
        padding: 1.5rem 2rem;
    }

    /*  Autonomy level badge  */
    .autonomy-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 2px;
        margin-bottom: 1rem;
    }
    .autonomy-advisory  { background: #0d1520; color: #38bdf8; border: 1px solid #1a2d40; }
    .autonomy-confirm   { background: #0d0d1f; color: #818cf8; border: 1px solid #2a2a5a; }
    .autonomy-autonomous{ background: #1f0d0d; color: #f87171; border: 1px solid #4a1f1f; }

    /*  Confidence indicator  */
    .confidence-bar-wrap {
        margin: 6px 0 2px;
        height: 4px;
        background: #1e1e1e;
        border-radius: 2px;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.3s;
    }

    /*  Session stats (sidebar)  */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .stat-row:last-child {
        border-bottom: none;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #444;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


#  Header 
st.markdown("""
<div class="kw-header">
    <p class="kw-title">KosWatt</p>
    <p class="kw-subtitle">Smart Energy Agentic AI &nbsp;|&nbsp; Boarding House Energy Monitor</p>
</div>
""", unsafe_allow_html=True)


#  Gauge chart builder 
def build_gauge(score):
    if score <= 35:
        needle_color = "#4ade80"
    elif score <= 65:
        needle_color = "#fbbf24"
    else:
        needle_color = "#f87171"

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = score,
        number = {
            'font'  : {'size': 40, 'family': 'IBM Plex Mono', 'color': '#ffffff'},
            'suffix': ''
        },
        gauge = {
            'axis': {
                'range'    : [0, 100],
                'tickwidth': 1,
                'tickcolor': "#333",
                'tickvals' : [0, 35, 65, 100],
                'tickfont' : {'color': '#444', 'size': 10},
            },
            'bar'       : {'color': needle_color, 'thickness': 0.22},
            'bgcolor'   : '#141414',
            'borderwidth': 0,
            'steps'     : [
                {'range': [0,   35],  'color': '#0d1f0d'},
                {'range': [35,  65],  'color': '#1f1a0d'},
                {'range': [65, 100],  'color': '#1f0d0d'},
            ],
            'threshold' : {
                'line'     : {'color': '#555', 'width': 2},
                'thickness': 0.75,
                'value'    : 65
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


#  Device status HTML builder 
def device_pill(label, is_on, is_eco=False):
    if is_eco:
        badge = '<span class="pill-eco">ECO</span>'
    elif is_on:
        badge = '<span class="pill-on">ON</span>'
    else:
        badge = '<span class="pill-off">OFF</span>'
    return f'<div class="device-row">{badge} <span style="color:#aaa">{label}</span></div>'


#  Sidebar: Simulation Controls 
with st.sidebar:

    #  Dataset section 
    st.markdown(
        '<p style="font-family: IBM Plex Mono, monospace; font-size: 1rem;'
        ' font-weight:600; color:#fff; margin-bottom:4px;">Dataset</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.75rem; color:#555; margin-bottom:0.75rem;">'
        'Load boarding_house_dataset.csv to replay real rows</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Load CSV", type='csv', label_visibility='collapsed')
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state['_dataset']      = df_uploaded
        st.session_state['_batch_results'] = None   # invalidate cached batch

    if st.session_state['_dataset'] is not None:
        df_ds  = st.session_state['_dataset']
        n_rows = len(df_ds)
        row_idx = st.slider(
            f"Row (of {n_rows})",
            min_value=0, max_value=n_rows - 1, value=0, step=1,
            key='_row_slider'
        )
        preview = df_ds.iloc[row_idx]
        st.markdown(
            f'<p style="font-family: IBM Plex Mono, monospace; font-size: 0.7rem;'
            f' color:#555; margin:2px 0;">'
            f'{preview["date"]} &nbsp; {preview["time"]} &nbsp;|&nbsp; '
            f'{preview["global_active_power"]} kW</p>',
            unsafe_allow_html=True
        )
        if st.button("Load Row into Controls", use_container_width=True):
            row = df_ds.iloc[row_idx]
            st.session_state['_occ_radio']   = 'Room Occupied' if int(row['occupancy']) == 1 else 'Room Empty'
            st.session_state['_temp_slider'] = int(row['temperature'])
            st.session_state['_tod_radio']   = 'Nighttime' if float(row['time_of_day']) >= 0.5 else 'Daytime'
            st.session_state['_ac_on']       = bool(int(row['ac_on']))
            st.session_state['_ac_eco']      = bool(int(row['ac_eco'])) if int(row['ac_on']) else False
            st.session_state['_lamp']        = bool(int(row['lamp_on']))
            st.session_state['_tv']          = bool(int(row['tv_on']))
            st.session_state['_conf_slider'] = float(row['occupancy_confidence'])
            st.rerun()

        if st.button("Run Batch Analysis", use_container_width=True):
            progress = st.progress(0, text="Running agent on dataset rows...")
            results  = []
            sample   = df_ds.head(500)   # cap at 500 for UI responsiveness
            for i, (_, row) in enumerate(sample.iterrows()):
                res = core_koswatt_agent(
                    occupancy            = int(row['occupancy']),
                    temperature          = int(row['temperature']),
                    time_of_day          = float(row['time_of_day']),
                    status_ac            = bool(int(row['ac_on'])),
                    status_ac_eco        = bool(int(row['ac_eco'])),
                    status_lamp          = bool(int(row['lamp_on'])),
                    status_tv            = bool(int(row['tv_on'])),
                    autonomy_level       = 'autonomous',
                    occupancy_confidence = float(row['occupancy_confidence']),
                )
                results.append({
                    'waste_category' : res['waste_category'],
                    'fuzzy_score'    : res['fuzzy_score'],
                    'actions'        : res['action_sequence'],
                    'watt_before'    : res['initial_watt'],
                    'watt_after'     : res['final_watt'],
                    'watt_saved'     : max(0, res['initial_watt'] - res['final_watt']),
                })
                if i % 50 == 0:
                    progress.progress((i + 1) / len(sample),
                                      text=f"Row {i+1}/{len(sample)}…")
            progress.empty()
            st.session_state['_batch_results'] = results

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e1e1e; margin: 0.5rem 0 1rem;">', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family: IBM Plex Mono, monospace; font-size: 1rem;'
        ' font-weight:600; color:#fff; margin-bottom:4px;">Simulation Control</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="font-size:0.75rem; color:#555; margin-bottom:1.5rem;">'
        'Configure room parameters to test the AI agent</p>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="sidebar-section">Room State</p>', unsafe_allow_html=True)

    occupancy_opt = st.radio(
        "Occupancy",
        options=["Room Empty", "Room Occupied"],
        index=0,
        key='_occ_radio'
    )
    occupancy = 0 if occupancy_opt == "Room Empty" else 1

    temperature = st.slider(
        "Room Temperature (C)",
        min_value=20, max_value=40, value=28, step=1,
        key='_temp_slider'
    )

    time_opt = st.radio(
        "Time of Day",
        options=["Daytime", "Nighttime"],
        index=0,
        key='_tod_radio'
    )
    time_of_day = 0 if time_opt == "Daytime" else 1

    st.markdown('<p class="sidebar-section">Device Switch State</p>', unsafe_allow_html=True)
    st.caption("Initial state before AI intervention")

    status_ac   = st.toggle("AC Unit", value=True,  key='_ac_on')
    if not status_ac:
        st.session_state['_ac_eco'] = False
    status_ac_eco = st.toggle("AC in ECO mode", key='_ac_eco', value=False, disabled=not status_ac)
    status_lamp = st.toggle("Lamp", value=False, key='_lamp')
    status_tv   = st.toggle("TV",   value=True,  key='_tv')

    #  Agent Settings 
    st.markdown('<p class="sidebar-section">Agent Settings</p>', unsafe_allow_html=True)

    autonomy_choice = st.radio(
        "Autonomy Level",
        options=list(AUTONOMY_DISPLAY.keys()),
        index=0,
        format_func=lambda x: AUTONOMY_DISPLAY[x],
    )
    autonomy_level = autonomy_choice
    st.caption(AUTONOMY_LEVELS[autonomy_level])

    # Confidence slider — only meaningful when room reads empty
    if occupancy == 0:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        occupancy_confidence = st.slider(
            "Sensor Confidence",
            min_value=0.0, max_value=1.0, value=1.0, step=0.05,
            key='_conf_slider',
            help="How confident is the occupancy sensor in its 'empty' reading? "
                 "Lower values make the agent more conservative (e.g. sleeping person)."
        )
        conf_pct   = int(occupancy_confidence * 100)
        if occupancy_confidence >= 0.8:
            bar_color = "#4ade80"
        elif occupancy_confidence >= 0.5:
            bar_color = "#fbbf24"
        else:
            bar_color = "#f87171"
        st.markdown(f"""
        <div style="font-family: IBM Plex Mono, monospace; font-size: 0.7rem; color: #555; margin-top:-4px;">
            Confidence: <span style="color:{bar_color}">{conf_pct}%</span>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill"
                 style="width:{conf_pct}%; background:{bar_color};"></div>
        </div>
        """, unsafe_allow_html=True)
        if occupancy_confidence < 0.5:
            st.caption("⚠ Below 50% — autonomous action suspended (safety override).")
    else:
        occupancy_confidence = 1.0

    #  Session stats 
    st.markdown('<p class="sidebar-section">Session Stats</p>', unsafe_allow_html=True)

    log = st.session_state.log
    total_readings  = f"{len(log)}+" if len(log) >= 100 else len(log)
    high_events     = sum(1 for e in log if e['waste_category'] == 'HIGH WASTE')
    medium_events   = sum(1 for e in log if e['waste_category'] == 'MEDIUM WASTE')
    avg_saved_watts = (
        round(sum(e['watt_saved'] for e in log if e['waste_category'] == 'HIGH WASTE') / high_events)
        if high_events else 0
    )

    st.markdown(f"""
    <div style="padding: 4px 0;">
        <div class="stat-row">
            <span class="stat-label">Readings</span>
            <span class="stat-value">{total_readings}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">High waste events</span>
            <span class="stat-value">{high_events}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Medium waste events</span>
            <span class="stat-value">{medium_events}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Avg W saved per action</span>
            <span class="stat-value">{avg_saved_watts} W</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


#  Core computation 
result = core_koswatt_agent(
    occupancy            = occupancy,
    temperature          = temperature,
    time_of_day          = time_of_day,
    status_ac            = status_ac,
    status_lamp          = status_lamp,
    status_tv            = status_tv,
    status_ac_eco        = status_ac_eco,
    autonomy_level       = autonomy_level,
    occupancy_confidence = occupancy_confidence,
)

fuzzy_score           = result['fuzzy_score']
waste_category        = result['waste_category']
action_seq            = result['action_sequence']
initial_watt          = result['initial_watt']
final_state           = result['final_state']
final_watt            = result['final_watt']
medium_recs           = result['medium_recommendations']
reasoning             = result['reasoning']
awaiting_confirmation = result['awaiting_confirmation']
effective_occupancy   = result['effective_occupancy']


#  Session logging (deduplicated by input hash) 
current_hash = (occupancy, temperature, time_of_day, status_ac, status_ac_eco,
                status_lamp, status_tv, autonomy_level, occupancy_confidence)
if current_hash != st.session_state.last_log_hash:
    st.session_state.last_log_hash = current_hash
    # Confirm mode: final_state == start_state (nothing executed yet).
    # Simulate projected savings so the session stat remains meaningful.
    # Autonomous mode: final_state already has actions applied — use final_watt
    # directly.  Advisory mode: action_seq is always [] so the else branch runs.
    if action_seq and awaiting_confirmation:
        _sim_state = apply_actions_to_state(final_state, action_seq)
        _sim_watt  = calculate_post_action_watt(_sim_state, temperature)
        _log_saved = max(0, initial_watt - _sim_watt)
    else:
        _log_saved = max(0, initial_watt - final_watt)
    st.session_state.log.append({
        'waste_category': waste_category,
        'initial_watt'  : initial_watt,
        'final_watt'    : final_watt,
        'watt_saved'    : _log_saved,
        'fuzzy_score'   : fuzzy_score,
    })
    if len(st.session_state.log) > 100:
        st.session_state.log = st.session_state.log[-100:]


# =============================================================================
# ROW 01 -- Current Conditions / Before AI Intervention
# =============================================================================
st.markdown(
    '<p class="section-label">01 &nbsp; Current Conditions &nbsp; / &nbsp; Before AI Intervention</p>',
    unsafe_allow_html=True
)

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

    occ_text = "Occupied" if occupancy else "Empty"
    tod_text = "Night"    if time_of_day else "Day"

    # Show effective occupancy with adaptive confidence fallback
    conf_pct = int(occupancy_confidence * 100)
    if occupancy_confidence < 1.0:
        eff_occ_html = (
            f'Eff. Occupancy: <span style="color:#fff">{effective_occupancy:.2f}</span>'
            f' <span style="color:#555; font-size:0.75rem;">({conf_pct}% conf.)</span><br>'
        )
    else:
        eff_occ_html = f'Eff. Occupancy: <span style="color:#fff">{effective_occupancy:.0f}</span> <span style="color:#555; font-size:0.75rem;">(Max Confidence)</span><br>'

    # Build per-device watt breakdown for the context card
    ac_draw = AC_ECO_WATTS if (status_ac and status_ac_eco) else (
              AC_HOT_WATTS if (status_ac and temperature > 30) else
              AC_COOL_WATTS if status_ac else 0)
    lamp_draw    = LAMP_WATTS if status_lamp else 0
    tv_draw      = TV_WATTS   if status_tv   else 0
    standby_draw = STANDBY_WATTS

    def watt_row(label, watts):
        color = "#aaa" if watts > 0 else "#333"
        return (f'<div style="display:flex; justify-content:space-between; '
                f'padding:3px 0; border-bottom:1px solid #1a1a1a;">'
                f'<span style="color:#555; font-size:0.75rem;">{label}</span>'
                f'<span style="font-family:IBM Plex Mono,monospace; font-size:0.75rem; color:{color};">'
                f'{watts} W</span></div>')

    breakdown_html = (
        watt_row("AC", ac_draw) +
        watt_row("Lamp", lamp_draw) +
        watt_row("TV", tv_draw) +
        watt_row("Standby", standby_draw)
    )

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Context</div>
        <div style="font-family: IBM Plex Mono, monospace; font-size: 0.85rem; line-height: 2; color: #aaa; margin-bottom: 0.8rem;">
            Room: <span style="color:#fff">{occ_text}</span><br>
            Temp: <span style="color:#fff">{temperature} C</span><br>
            Time: <span style="color:#fff">{tod_text}</span><br>
            {eff_occ_html}
        </div>
        <div style="border-top: 1px solid #1e1e1e; padding-top: 0.6rem;">
            <div style="font-size:0.65rem; letter-spacing:1.5px; color:#444; text-transform:uppercase; margin-bottom:6px;">Draw Breakdown</div>
            {breakdown_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    ac_state   = device_pill("AC Unit", status_ac, is_eco=(status_ac and status_ac_eco))
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
        <br>
        <span class="autonomy-badge autonomy-{autonomy_level}" style="margin-top:6px;">
            {AUTONOMY_DISPLAY[autonomy_level]}
        </span>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)


#  Helper: renders an action/suggestion list as HTML 
def _actions_html(items, label_map, item_css_class):
    html = ""
    for i, item in enumerate(items, 1):
        label = label_map.get(item, item)
        html += f"""
        <div class="{item_css_class}">
            <span class="action-index">{i:02d}</span>
            <span>{label}</span>
        </div>"""
    if not html:
        html = '<p style="color:#555; font-size:0.85rem;">No further actions required.</p>'
    return html


# =============================================================================
# ROW 02 -- After AI Automation / Agent Decision
# =============================================================================
st.markdown(
    '<p class="section-label">02 &nbsp; After AI Automation &nbsp; / &nbsp; Agent Decision</p>',
    unsafe_allow_html=True
)

if waste_category == 'LOW WASTE':
    st.markdown(f"""
    <div class="panel-stable">
        <p class="panel-title" style="color:#4ade80">System Stable</p>
        <p style="font-size:0.95rem; color:#aaa; margin:0;">
            Waste score <strong style="color:#4ade80">{fuzzy_score}</strong> is within
            acceptable range. No automation action required. All devices operating
            within ethical parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif waste_category == 'MEDIUM WASTE':
    st.markdown(f"""
    <div class="panel-medium">
        <p class="panel-title" style="color:#fbbf24">Ethical Tolerance Active</p>
        <p style="font-size:0.95rem; color:#aaa; margin:0 0 0.8rem;">
            Waste score <strong style="color:#fbbf24">{fuzzy_score}</strong>.
            Fuzzy guardrails detected contextual justification for continued device
            operation (thermal comfort or safety lighting). Autonomous intervention
            withheld.
        </p>
        <p style="font-size: 0.8rem; color: #666; margin:0; font-family: IBM Plex Mono, monospace;">
            MONITORING &nbsp; | &nbsp; No STRIPS plan triggered
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Advisory recommendations panel (non-binding)
    if medium_recs:
        advisory_items_html = ""
        for rec in medium_recs:
            label = ADVISORY_LABELS.get(rec, rec)
            if rec == 'SUGGEST_AC_ECO':
                current_ac_watts = AC_HOT_WATTS if temperature > 30 else AC_COOL_WATTS
                ac_saving        = current_ac_watts - AC_ECO_WATTS
                advisory_items_html += f"""
                <div class="advisory-item">
                    {label}
                    <div class="advisory-note">
                        Would reduce AC draw by approximately {ac_saving} W.
                        Advisory only -- no autonomous action executed.
                    </div>
                </div>"""
            elif rec == 'SUGGEST_LAMP_OFF':
                advisory_items_html += f"""
                <div class="advisory-item">
                    {label}
                    <div class="advisory-note">
                        Lamp draw is {LAMP_WATTS} W. Safety and deterrence
                        justification applies at night; shutoff is not
                        mandatory. Advisory only -- no autonomous action executed.
                    </div>
                </div>"""

        st.markdown(f"""
        <div class="panel-advisory">
            <p class="panel-title" style="color:#38bdf8; margin-bottom:0.8rem;">
                Non-Binding Recommendations
            </p>
            {advisory_items_html}
        </div>
        """, unsafe_allow_html=True)

else:
    # HIGH WASTE — behaviour depends on autonomy level and confidence

    #  Safety override: confidence too low to act 
    # Checked here (not only in the agent) so the correct panel is shown even
    # when the fuzzy dead zone (effective_occupancy=0.5 at confidence=0) caused
    # the agent to classify LOW/MEDIUM rather than HIGH.  The agent's reasoning
    # trace will already contain the PRIORITY 1: SAFETY OVERRIDE entry.
    # Uses < 0.5 (not == 0) to match the agent's own condition.
    safety_override_active = (occupancy_confidence < 0.5 and occupancy < 0.5)

    if safety_override_active:
        st.markdown(f"""
        <div class="panel-safety">
            <p class="panel-title" style="color:#c084fc">Safety Override Active</p>
            <p style="font-size:0.95rem; color:#aaa; margin:0 0 0.8rem;">
                Waste score <strong style="color:#f87171">{fuzzy_score}</strong> qualifies
                as HIGH WASTE, but the occupancy sensor confidence is only
                <strong style="color:#c084fc">{int(occupancy_confidence*100)}%</strong>
                — below the 50% threshold for autonomous action.
            </p>
            <p style="font-size:0.85rem; color:#7a5a8a; margin:0; font-family: IBM Plex Mono, monospace;">
                PRIORITY 1: SAFETY &nbsp;|&nbsp; Risk of false-empty reading (e.g. sleeping
                occupant). Autonomous action suspended. Raise sensor confidence to re-enable.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if medium_recs:
            adv_html = _actions_html(medium_recs, ADVISORY_LABELS, 'advisory-item')
            st.markdown(f"""
            <div class="panel-advisory">
                <p class="panel-title" style="color:#38bdf8; margin-bottom:0.8rem;">
                    Non-Binding Recommendations (Safety Override)
                </p>
                {adv_html}
            </div>
            """, unsafe_allow_html=True)

    #  Advisory: no execution, show recommendations 
    elif autonomy_level == 'advisory':
        adv_html = _actions_html(medium_recs, ADVISORY_LABELS, 'advisory-item') if medium_recs else \
                   '<p style="color:#555; font-size:0.85rem;">No recommendations generated.</p>'
        st.markdown(f"""
        <div class="panel-advisory">
            <p class="panel-title" style="color:#38bdf8">Advisory Mode — No Action Taken</p>
            <p style="font-size:0.95rem; color:#aaa; margin:0 0 0.8rem;">
                Waste score <strong style="color:#f87171">{fuzzy_score}</strong> qualifies
                as HIGH WASTE. Autonomy is <strong style="color:#38bdf8">ADVISORY</strong>
                — the STRIPS plan has been computed but will not execute.
                Devices remain unchanged.
            </p>
            {adv_html}
        </div>
        """, unsafe_allow_html=True)

    #  Confirm: plan ready, awaiting approval 
    elif awaiting_confirmation:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            pending_html = _actions_html(action_seq, ACTION_LABELS, 'action-item-pending')
            st.markdown(f"""
            <div class="panel-confirm">
                <p class="panel-title" style="color:#818cf8">Awaiting Confirmation</p>
                <p style="font-family: IBM Plex Mono, monospace; font-size: 0.75rem;
                          color:#444; margin-bottom:1rem;">
                    STRIPS PLANNER &nbsp;|&nbsp; {len(action_seq)} action(s) pending approval
                </p>
                {pending_html}
                <p style="font-size:0.75rem; color:#444; margin-top:1rem; margin-bottom:0;
                          font-family: IBM Plex Mono, monospace;">
                    Devices unchanged until approved. Switch to AUTONOMOUS to auto-execute.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            ac_eco  = final_state.get('ac_eco', False)
            ac_on   = final_state.get('ac_on',  False)
            lamp_on = final_state.get('lamp_on', False)
            tv_on   = final_state.get('tv_on',  False)
            # In confirm mode, final_state == start_state (nothing executed yet).
            # Simulate what post-approval draw would be.
            projected_state = apply_actions_to_state(final_state, action_seq)
            projected_watt  = calculate_post_action_watt(projected_state, temperature)
            watt_saved_projected = max(0, initial_watt - projected_watt)
            ac_f    = device_pill("AC Unit", ac_on,  is_eco=ac_eco)
            lamp_f  = device_pill("Lamp",    lamp_on)
            tv_f    = device_pill("TV",      tv_on)
            st.markdown(f"""
            <div class="panel-confirm">
                <p class="panel-title" style="color:#818cf8">Current State (Unchanged)</p>
                {ac_f}{lamp_f}{tv_f}
                <div class="watt-delta">
                    <span class="watt-before" style="text-decoration:none; color:#818cf8;">{initial_watt} W</span>
                    <span style="font-family: IBM Plex Mono, monospace; font-size:0.8rem; color:#444;">
                        → {projected_watt} W if approved
                        <span style="color:#818cf8">({watt_saved_projected} W saved)</span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    #  Autonomous: STRIPS executed 
    else:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            actions_html = _actions_html(action_seq, ACTION_LABELS, 'action-item')
            st.markdown(f"""
            <div class="panel-alert">
                <p class="panel-title" style="color:#f87171">AI Action Triggered</p>
                <p style="font-family: IBM Plex Mono, monospace; font-size: 0.75rem;
                          color:#666; margin-bottom:1rem;">
                    STRIPS PLANNER &nbsp; | &nbsp; {len(action_seq)} action(s) executed
                </p>
                {actions_html}
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            ac_eco   = final_state.get('ac_eco', False)
            ac_on    = final_state.get('ac_on',  False)
            lamp_on  = final_state.get('lamp_on', False)
            tv_on    = final_state.get('tv_on',  False)

            ac_final   = device_pill("AC Unit", ac_on,  is_eco=ac_eco)
            lamp_final = device_pill("Lamp",    lamp_on)
            tv_final   = device_pill("TV",      tv_on)

            watt_saved = max(0, initial_watt - final_watt)

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

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)


# =============================================================================
# ROW 03 -- Agent Reasoning / Decision Trace
# =============================================================================
st.markdown(
    '<p class="section-label">03 &nbsp; Agent Reasoning &nbsp; / &nbsp; Decision Trace</p>',
    unsafe_allow_html=True
)

reasoning_lines_html = ""
for line in reasoning:
    reasoning_lines_html += f'<div class="reasoning-line">{line}</div>'

if not reasoning_lines_html:
    reasoning_lines_html = '<div class="reasoning-line" style="color:#444;">No reasoning trace available.</div>'

st.markdown(f"""
<div class="panel-reasoning">
    <p class="panel-title" style="color:#555; margin-bottom:1rem;">Internal Decision Log</p>
    {reasoning_lines_html}
</div>
""", unsafe_allow_html=True)


# =============================================================================
# ROW 04 -- Batch Dataset Analysis
# =============================================================================
batch_results = st.session_state.get('_batch_results')
if batch_results:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label">04 &nbsp; Batch Dataset Analysis</p>',
        unsafe_allow_html=True
    )

    n_batch      = len(batch_results)
    cats         = [r['waste_category'] for r in batch_results]
    low_n        = cats.count('LOW WASTE')
    med_n        = cats.count('MEDIUM WASTE')
    high_n       = cats.count('HIGH WASTE')
    total_saved  = sum(r['watt_saved'] for r in batch_results)
    all_actions  = [a for r in batch_results for a in r['actions']]
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, label, val, color in [
        (col_a, "Rows Processed",   str(n_batch),           "#aaa"),
        (col_b, "HIGH WASTE",       str(high_n),            "#f87171"),
        (col_c, "MEDIUM WASTE",     str(med_n),             "#fbbf24"),
        (col_d, "Total W Saved",    f"{total_saved:,} W",   "#4ade80"),
    ]:
        col.markdown(f"""
        <div style="background:#111; border:1px solid #1e1e1e; border-radius:8px;
                    padding:1rem; text-align:center;">
            <p style="font-family: IBM Plex Mono, monospace; font-size:0.7rem;
                      color:#555; margin:0 0 0.3rem; letter-spacing:1px;">{label}</p>
            <p style="font-size:1.4rem; font-weight:700; color:{color}; margin:0;">{val}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_cat = go.Figure(go.Bar(
            x     = ['LOW WASTE', 'MEDIUM WASTE', 'HIGH WASTE'],
            y     = [low_n, med_n, high_n],
            marker_color = ['#4ade80', '#fbbf24', '#f87171'],
            text  = [low_n, med_n, high_n],
            textposition = 'outside',
            textfont     = {'color': '#aaa', 'size': 11,
                            'family': 'IBM Plex Mono'},
        ))
        fig_cat.update_layout(
            title      = dict(text='Waste Classification Distribution',
                              font=dict(color='#555', size=12,
                                        family='IBM Plex Mono'),
                              x=0),
            paper_bgcolor = 'rgba(0,0,0,0)',
            plot_bgcolor  = 'rgba(0,0,0,0)',
            xaxis = dict(tickfont=dict(color='#555', size=10,
                                       family='IBM Plex Mono'),
                         showgrid=False),
            yaxis = dict(tickfont=dict(color='#555', size=10),
                         gridcolor='#1e1e1e', showgrid=True),
            margin = dict(l=10, r=10, t=40, b=10),
            height = 260,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        if action_counts:
            labels = [ACTION_LABELS.get(k, k) for k in action_counts]
            values = list(action_counts.values())
            fig_act = go.Figure(go.Bar(
                x    = values,
                y    = labels,
                orientation  = 'h',
                marker_color = '#818cf8',
                text         = values,
                textposition = 'outside',
                textfont     = {'color': '#aaa', 'size': 11,
                                'family': 'IBM Plex Mono'},
            ))
            fig_act.update_layout(
                title      = dict(text='Actions Executed by STRIPS Planner',
                                  font=dict(color='#555', size=12,
                                            family='IBM Plex Mono'),
                                  x=0),
                paper_bgcolor = 'rgba(0,0,0,0)',
                plot_bgcolor  = 'rgba(0,0,0,0)',
                xaxis = dict(tickfont=dict(color='#555', size=10),
                             gridcolor='#1e1e1e'),
                yaxis = dict(tickfont=dict(color='#555', size=10,
                                           family='IBM Plex Mono'),
                             showgrid=False),
                margin = dict(l=10, r=10, t=40, b=10),
                height = 260,
            )
            st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.markdown(
                '<p style="color:#555; font-family: IBM Plex Mono, monospace;'
                ' font-size:0.8rem; padding-top:3rem; text-align:center;">'
                'No STRIPS actions executed across batch.</p>',
                unsafe_allow_html=True
            )


#  Footer 
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
