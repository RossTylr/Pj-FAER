"""
System Schematic Page - Live visualization of hospital configuration.

Displays a graphical representation of the hospital patient flow,
updating in real-time as configuration changes on Arrivals and Resources pages.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from app.components.schematic import (
    build_capacity_graph_from_params,
    build_feedback_diagram
)

st.set_page_config(page_title="Schematic", page_icon="", layout="wide")
st.title("System Schematic")

st.info("This diagram updates live as you change configuration on Arrivals and Resources pages.")

# ============ MAIN SCHEMATIC ============
st.header("Patient Flow Diagram")

st.markdown("""
| Arrow | Meaning |
|-------|---------|
| **Blue** | Normal patient flow |
| **Purple (dashed)** | Diagnostics loop (patient keeps ED bay) |
| **Red** | BLOCKING feedback |
""")

# Build from session state
schematic = build_capacity_graph_from_params(
    n_ambulances=st.session_state.get('n_ambulances', 10),
    n_helicopters=st.session_state.get('n_helicopters', 2),
    n_handover=st.session_state.get('n_handover_bays', 4),
    n_triage=st.session_state.get('n_triage', 3),
    n_ed_bays=st.session_state.get('n_ed_bays', 20),
    n_ct=st.session_state.get('ct_capacity', 2),
    n_xray=st.session_state.get('xray_capacity', 3),
    n_bloods=st.session_state.get('bloods_capacity', 5),
    n_theatre=st.session_state.get('n_theatre_tables', 2),
    n_itu=st.session_state.get('n_itu_beds', 6),
    n_ward=st.session_state.get('n_ward_beds', 30),
    ct_enabled=st.session_state.get('ct_enabled', True),
    xray_enabled=st.session_state.get('xray_enabled', True),
    bloods_enabled=st.session_state.get('bloods_enabled', True),
)

st.graphviz_chart(schematic, use_container_width=True)

# ============ CURRENT CONFIG SUMMARY ============
st.markdown("---")
st.header("Current Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Arrivals & Front Door")
    config1 = {
        'Ambulances': st.session_state.get('n_ambulances', 10),
        'HEMS': st.session_state.get('n_helicopters', 2),
        'Handover Bays': st.session_state.get('n_handover_bays', 4),
        'Triage': st.session_state.get('n_triage', 3),
        'ED Bays': st.session_state.get('n_ed_bays', 20),
    }
    st.dataframe(
        pd.DataFrame([config1]).T.rename(columns={0: 'Count'}),
        use_container_width=True
    )

with col2:
    st.subheader("Downstream")
    config2 = {
        'CT Scanners': st.session_state.get('ct_capacity', 2) if st.session_state.get('ct_enabled', True) else '-',
        'X-ray': st.session_state.get('xray_capacity', 3) if st.session_state.get('xray_enabled', True) else '-',
        'Theatre': st.session_state.get('n_theatre_tables', 2),
        'ITU': st.session_state.get('n_itu_beds', 6),
        'Ward': st.session_state.get('n_ward_beds', 30),
    }
    st.dataframe(
        pd.DataFrame([config2]).T.rename(columns={0: 'Count'}),
        use_container_width=True
    )

# ============ FEEDBACK LOOPS EXPLANATION ============
st.markdown("---")
st.header("Understanding Feedback Loops")

with st.expander("Why do feedback loops matter?", expanded=True):
    st.markdown("""
    ### The Blocking Cascade

    When downstream beds fill up, it causes a chain reaction:

    1. **Ward Full** -> Patients can't leave ITU -> ITU can't accept from ED
    2. **ED Full** -> Patients board in bays -> Handover bays blocked
    3. **Handover Blocked** -> Ambulances queue -> 999 response delays

    ### Key Insight

    > **Adding ED capacity alone often doesn't help.**
    > The bottleneck is usually downstream (Ward beds, discharge processes).
    """)

    st.graphviz_chart(build_feedback_diagram(), use_container_width=True)

# ============ CAPACITY CHECK ============
st.markdown("---")
st.header("Quick Capacity Check")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Bed Ratios**")
    ed = st.session_state.get('n_ed_bays', 20)
    itu = st.session_state.get('n_itu_beds', 6)
    ward = st.session_state.get('n_ward_beds', 30)

    st.write(f"ED : Ward = {ed} : {ward} = 1 : {ward/ed:.1f}")
    st.write(f"ITU : Ward = {itu} : {ward} = 1 : {ward/itu:.1f}")
    st.metric("Total Inpatient", itu + ward)

with col2:
    st.markdown("**Potential Issues**")
    issues = []

    if ward < ed:
        issues.append("Ward beds < ED bays")
    if itu < 3:
        issues.append("Very low ITU capacity")
    if st.session_state.get('ct_enabled') and st.session_state.get('ct_capacity', 2) < 2:
        issues.append("Single CT scanner")

    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("No obvious issues")

# ============ ARRIVAL CONFIGURATION ============
st.markdown("---")
st.header("Arrival Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Arrival Model**")
    arrival_model = st.session_state.get('arrival_model', 'profile_24h')
    day_type = st.session_state.get('day_type', 'weekday')
    demand_mult = st.session_state.get('demand_multiplier', 1.0)

    st.write(f"Model: **{arrival_model}**")
    if arrival_model == 'profile_24h':
        st.write(f"Day type: **{day_type}**")
    st.write(f"Demand scale: **{demand_mult:.1f}x**")

with col2:
    st.markdown("**Stream Multipliers**")
    st.write(f"Ambulance: **{st.session_state.get('ambulance_rate_multiplier', 1.0):.1f}x**")
    st.write(f"HEMS: **{st.session_state.get('helicopter_rate_multiplier', 1.0):.1f}x**")
    st.write(f"Walk-in: **{st.session_state.get('walkin_rate_multiplier', 1.0):.1f}x**")

# Navigation
st.markdown("---")
st.info("**Next**: Go to **Run** to simulate with this configuration")
