"""System Schematic Page - Interactive visualization of hospital configuration and results.

Displays a graphical representation of the hospital patient flow using an interactive
React/SVG component, with support for both configuration-only and post-simulation modes.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from app.components.react_schematic.component import is_built, react_schematic
from app.components.react_schematic.data import (
    SchematicData,
    build_schematic_from_config,
    build_schematic_from_results,
    to_dict,
)
from app.components.schematic import build_feedback_diagram


# ============ HELPER FUNCTIONS ============

def render_node_details(node_id: str, data: SchematicData):
    """Render detailed metrics for a clicked node."""
    st.markdown("---")
    node = data.nodes[node_id]

    # Horizontal layout for node details
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"### {node.label}")
        st.markdown(f"Type: `{node.node_type}`")
        status_emoji = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}.get(
            node.status, "⚪"
        )
        st.markdown(f"Status: {status_emoji} **{node.status.title()}**")

    with col2:
        if node.capacity:
            st.metric("Utilisation", f"{node.utilisation:.0%}")
            st.progress(min(node.utilisation, 1.0))
        else:
            st.metric("Throughput", f"{node.throughput_per_hour:.1f}/hr")

    with col3:
        st.metric("Mean Wait", f"{node.mean_wait_mins:.1f} min")
        if node.capacity:
            st.metric("Occupied", f"{node.occupied}/{node.capacity}")

    with col4:
        # Show connected edges
        st.markdown("**Flows**")
        incoming = [e for e in data.edges if e.target == node_id]
        outgoing = [e for e in data.edges if e.source == node_id]

        if incoming:
            for e in incoming:
                status = "🔴" if e.is_blocked else "🟢"
                source_label = data.nodes[e.source].label
                st.markdown(f"{status} ← {source_label}: {e.volume_per_hour:.1f}/hr")

        if outgoing:
            for e in outgoing:
                status = "🔴" if e.is_blocked else "🟢"
                target_label = data.nodes[e.target].label
                st.markdown(f"{status} → {target_label}: {e.volume_per_hour:.1f}/hr")


def render_fallback_schematic(data: SchematicData):
    """Render a static SVG fallback when React component isn't built.

    Layout: Left-to-right crucifix pattern

                              [ITU]
                                ↑
    [Arrivals] → [Triage] → [ED Bays] → [Theatre] → [Discharge]
                                ↓
                             [Ward]
    """

    def get_color(status: str) -> tuple:
        """Get fill and stroke colors for a status."""
        colors = {
            "normal": ("#f0fff4", "#28a745"),
            "warning": ("#fffbf0", "#ffc107"),
            "critical": ("#fff5f5", "#dc3545"),
        }
        return colors.get(status, ("#f8f9fa", "#6c757d"))

    ed = data.nodes.get("ed_bays")
    itu = data.nodes.get("itu")
    ward = data.nodes.get("ward")
    triage = data.nodes.get("triage")
    theatre = data.nodes.get("theatre")

    # Get status colors
    ed_fill, ed_stroke = get_color(ed.status if ed else "normal")
    itu_fill, itu_stroke = get_color(itu.status if itu else "normal")
    ward_fill, ward_stroke = get_color(ward.status if ward else "normal")
    triage_fill, triage_stroke = get_color(triage.status if triage else "normal")
    theatre_fill, theatre_stroke = get_color(theatre.status if theatre else "normal")

    # Check for blocked edges
    itu_blocked = any(e.is_blocked for e in data.edges if e.target == "itu")
    ward_blocked = any(e.is_blocked for e in data.edges if e.target == "ward")

    svg_content = f"""
    <svg width="1000" height="480" style="font-family: system-ui; background: #fafafa;">
        <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#adb5bd"/>
            </marker>
            <marker id="blocked-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#dc3545"/>
            </marker>
        </defs>

        <!-- Header -->
        <text x="20" y="25" font-size="16" font-weight="bold" fill="#333">System Schematic — {data.timestamp}</text>
        <text x="20" y="45" font-size="11" fill="#666">
            In System: <tspan font-weight="bold" fill="#333">{data.total_in_system}</tspan>  |
            24hr Throughput: <tspan font-weight="bold" fill="#333">{data.total_throughput_24h}</tspan>  |
            Status: <tspan font-weight="bold" fill="{ed_stroke}">{data.overall_status.upper()}</tspan>
        </text>

        <!-- === MAIN HORIZONTAL FLOW === -->
        <path d="M 120 235 L 175 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 295 235 L 355 235" stroke="#adb5bd" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
        <path d="M 475 235 L 535 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 655 235 L 715 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

        <!-- === VERTICAL FLOWS (crucifix arms) === -->
        <!-- ED → ITU (up) -->
        <path d="M 415 185 L 415 130" stroke="{'#dc3545' if itu_blocked else '#adb5bd'}"
              stroke-width="2" {'stroke-dasharray="6,3"' if itu_blocked else ''} fill="none"
              marker-end="url({'#blocked-arrow' if itu_blocked else '#arrow'})"/>

        <!-- ED → Ward (down) -->
        <path d="M 415 285 L 415 340" stroke="{'#dc3545' if ward_blocked else '#adb5bd'}"
              stroke-width="2" {'stroke-dasharray="6,3"' if ward_blocked else ''} fill="none"
              marker-end="url({'#blocked-arrow' if ward_blocked else '#arrow'})"/>

        <!-- === ARRIVALS (left side) === -->
        <g transform="translate(20, 135)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">Ambulance</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">{data.nodes.get('ambulance', type('', (), {{'throughput_per_hour': 0}})()).throughput_per_hour:.1f}/hr</text>
        </g>

        <g transform="translate(20, 210)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">Walk-in</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">{data.nodes.get('walkin', type('', (), {{'throughput_per_hour': 0}})()).throughput_per_hour:.1f}/hr</text>
        </g>

        <g transform="translate(20, 285)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">HEMS</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">{data.nodes.get('hems', type('', (), {{'throughput_per_hour': 0}})()).throughput_per_hour:.1f}/hr</text>
        </g>

        <!-- === TRIAGE === -->
        <g transform="translate(180, 195)">
            <rect width="110" height="80" rx="6" fill="{triage_fill}" stroke="{triage_stroke}" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="{triage_stroke}"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Triage</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="{int(90 * (triage.utilisation if triage else 0))}" height="8" rx="4" fill="{triage_stroke}"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">{triage.occupied if triage else 0}/{triage.capacity if triage else 0} ({triage.utilisation*100 if triage else 0:.0f}%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: {triage.mean_wait_mins if triage else 0:.0f}m</text>
        </g>

        <!-- === ED BAYS (center hub) === -->
        <g transform="translate(360, 190)">
            <rect width="110" height="90" rx="6" fill="{ed_fill}" stroke="{ed_stroke}" stroke-width="3"/>
            <circle cx="98" cy="12" r="5" fill="{ed_stroke}"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">ED Bays</text>
            <rect x="10" y="32" width="90" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="32" width="{int(90 * (ed.utilisation if ed else 0))}" height="10" rx="5" fill="{ed_stroke}"/>
            <text x="55" y="58" text-anchor="middle" font-size="10" fill="#333">{ed.occupied if ed else 0}/{ed.capacity if ed else 0} ({ed.utilisation*100 if ed else 0:.0f}%)</text>
            <text x="55" y="76" text-anchor="middle" font-size="9" fill="#666">Wait: {ed.mean_wait_mins if ed else 0:.0f}m</text>
        </g>

        <!-- === ITU (top) === -->
        <g transform="translate(360, 45)">
            <rect width="110" height="80" rx="6" fill="{itu_fill}" stroke="{itu_stroke}" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="{itu_stroke}"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">ITU</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="{int(90 * (itu.utilisation if itu else 0))}" height="8" rx="4" fill="{itu_stroke}"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">{itu.occupied if itu else 0}/{itu.capacity if itu else 0} ({itu.utilisation*100 if itu else 0:.0f}%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: {itu.mean_wait_mins if itu else 0:.0f}m</text>
        </g>

        <!-- === WARD (bottom) === -->
        <g transform="translate(360, 345)">
            <rect width="110" height="80" rx="6" fill="{ward_fill}" stroke="{ward_stroke}" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="{ward_stroke}"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Ward</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="{int(90 * (ward.utilisation if ward else 0))}" height="8" rx="4" fill="{ward_stroke}"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">{ward.occupied if ward else 0}/{ward.capacity if ward else 0} ({ward.utilisation*100 if ward else 0:.0f}%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: {ward.mean_wait_mins if ward else 0:.0f}m</text>
        </g>

        <!-- === THEATRE === -->
        <g transform="translate(540, 195)">
            <rect width="110" height="80" rx="6" fill="{theatre_fill}" stroke="{theatre_stroke}" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="{theatre_stroke}"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Theatre</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="{int(90 * (theatre.utilisation if theatre else 0))}" height="8" rx="4" fill="{theatre_stroke}"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">{theatre.occupied if theatre else 0}/{theatre.capacity if theatre else 0} ({theatre.utilisation*100 if theatre else 0:.0f}%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: {theatre.mean_wait_mins if theatre else 0:.0f}m</text>
        </g>

        <!-- === DISCHARGE === -->
        <g transform="translate(720, 195)">
            <rect width="100" height="80" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#6a1b9a">Discharge</text>
            <text x="50" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{data.nodes.get('discharge', type('', (), {{'throughput_per_hour': 0}})()).throughput_per_hour:.1f}/hr</text>
            <text x="50" y="65" text-anchor="middle" font-size="9" fill="#666">Exit</text>
        </g>

        <!-- Legend -->
        <g transform="translate(830, 320)">
            <rect width="155" height="115" fill="white" stroke="#dee2e6" rx="6" fill-opacity="0.95"/>
            <text x="10" y="18" font-size="11" font-weight="bold" fill="#333">Legend</text>
            <circle cx="18" cy="35" r="5" fill="#28a745"/>
            <text x="30" y="39" font-size="9" fill="#333">Normal (&lt;70%)</text>
            <circle cx="18" cy="52" r="5" fill="#ffc107"/>
            <text x="30" y="56" font-size="9" fill="#333">Warning (70-90%)</text>
            <circle cx="18" cy="69" r="5" fill="#dc3545"/>
            <text x="30" y="73" font-size="9" fill="#333">Critical (&gt;90%)</text>
            <line x1="12" y1="88" x2="35" y2="88" stroke="#dc3545" stroke-width="2" stroke-dasharray="4,2"/>
            <text x="42" y="92" font-size="9" fill="#333">Blocked Flow</text>
            <circle cx="18" cy="105" r="4" fill="#1976d2"/>
            <text x="30" y="109" font-size="9" fill="#333">Entry Point</text>
        </g>
    </svg>
    """

    st.components.v1.html(svg_content, height=490)


# ============ PAGE SETUP ============
st.set_page_config(page_title="Schematic", page_icon="", layout="wide")
st.title("System Schematic")

# ============ CHECK FOR RESULTS ============
has_results = st.session_state.get("run_complete", False)
results = st.session_state.get("run_results")
scenario = st.session_state.get("run_scenario")

# Mode indicator
if has_results and results:
    st.success("Showing post-simulation snapshot. Click nodes for details.")
else:
    st.info(
        "Showing configured capacities. Run simulation on the **Run** page to see live data."
    )

# ============ BUILD SCHEMATIC DATA ============
if has_results and results and scenario:
    run_length_hours = scenario.run_length / 60  # Convert minutes to hours
    schematic_data = build_schematic_from_results(results, scenario, run_length_hours)
else:
    schematic_data = build_schematic_from_config(dict(st.session_state))

# ============ RENDER SCHEMATIC ============
st.header("Patient Flow Diagram")

# Legend
st.markdown("""
| Indicator | Meaning |
|-----------|---------|
| **Green** | Normal utilisation (<70%) |
| **Yellow** | Warning utilisation (70-90%) |
| **Red** | Critical utilisation (>90%) |
| **Dashed red line** | Blocked flow |
""")

if is_built():
    # Render React component
    clicked = react_schematic(
        data=to_dict(schematic_data),
        width=1400,
        height=750,
        key="main_schematic",
    )

    # Node detail panel
    if clicked and clicked in schematic_data.nodes:
        render_node_details(clicked, schematic_data)
    else:
        st.caption("*Click a node in the schematic to see details*")
else:
    # Fallback to inline SVG when React not built
    st.warning("""
    **React component not built.** Using fallback visualization.

    To enable the interactive schematic, run:
    ```bash
    cd app/components/react_schematic
    npm install
    npm run build
    ```
    """)
    render_fallback_schematic(schematic_data)

# ============ CURRENT CONFIG SUMMARY ============
st.markdown("---")
st.header("Current Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Arrivals & Front Door")
    config1 = {
        "Ambulances": st.session_state.get("n_ambulances", 10),
        "HEMS": st.session_state.get("n_helicopters", 2),
        "Handover Bays": st.session_state.get("n_handover_bays", 4),
        "Triage": st.session_state.get("n_triage", 3),
        "ED Bays": st.session_state.get("n_ed_bays", 20),
    }
    st.dataframe(
        pd.DataFrame([config1]).T.rename(columns={0: "Count"}), use_container_width=True
    )

with col2:
    st.subheader("Downstream")
    config2 = {
        "CT Scanners": (
            st.session_state.get("ct_capacity", 2)
            if st.session_state.get("ct_enabled", True)
            else "-"
        ),
        "X-ray": (
            st.session_state.get("xray_capacity", 3)
            if st.session_state.get("xray_enabled", True)
            else "-"
        ),
        "Theatre": st.session_state.get("n_theatre_tables", 2),
        "ITU": st.session_state.get("n_itu_beds", 6),
        "Ward": st.session_state.get("n_ward_beds", 30),
    }
    st.dataframe(
        pd.DataFrame([config2]).T.rename(columns={0: "Count"}), use_container_width=True
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
    ed = st.session_state.get("n_ed_bays", 20)
    itu = st.session_state.get("n_itu_beds", 6)
    ward = st.session_state.get("n_ward_beds", 30)

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
    if st.session_state.get("ct_enabled") and st.session_state.get("ct_capacity", 2) < 2:
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
    arrival_model = st.session_state.get("arrival_model", "profile_24h")
    day_type = st.session_state.get("day_type", "weekday")
    demand_mult = st.session_state.get("demand_multiplier", 1.0)

    st.write(f"Model: **{arrival_model}**")
    if arrival_model == "profile_24h":
        st.write(f"Day type: **{day_type}**")
    st.write(f"Demand scale: **{demand_mult:.1f}x**")

with col2:
    st.markdown("**Stream Multipliers**")
    st.write(
        f"Ambulance: **{st.session_state.get('ambulance_rate_multiplier', 1.0):.1f}x**"
    )
    st.write(
        f"HEMS: **{st.session_state.get('helicopter_rate_multiplier', 1.0):.1f}x**"
    )
    st.write(
        f"Walk-in: **{st.session_state.get('walkin_rate_multiplier', 1.0):.1f}x**"
    )

# Navigation
st.markdown("---")
st.info("**Next**: Go to **Run** to simulate with this configuration")
