"""
Schematic Demo - Pure Streamlit Implementation

Run with: streamlit run faer/demos/schematic_streamlit.py

This demo uses only native Streamlit components + custom CSS/HTML.
COMPLETELY ISOLATED - no imports from main application.
"""

import streamlit as st
from sample_data import create_sample_data, NodeState, SchematicData


# === CONFIGURATION ===

STATUS_COLORS = {
    "normal": {"bg": "#f0fff4", "border": "#28a745", "icon": "🟢"},
    "warning": {"bg": "#fffbf0", "border": "#ffc107", "icon": "🟡"},
    "critical": {"bg": "#fff5f5", "border": "#dc3545", "icon": "🔴"},
}


# === STYLING ===

def inject_css():
    """Inject custom CSS for node styling."""
    st.markdown("""
    <style>
    .node-container {
        border: 2px solid #ccc;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 4px;
        text-align: center;
        min-height: 120px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .node-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .node-normal { border-color: #28a745; background: #f0fff4; }
    .node-warning { border-color: #ffc107; background: #fffbf0; }
    .node-critical { border-color: #dc3545; background: #fff5f5; }

    .node-label {
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .node-stats {
        font-size: 12px;
        color: #666;
    }
    .progress-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .progress-normal { background: #28a745; }
    .progress-warning { background: #ffc107; }
    .progress-critical { background: #dc3545; }

    .flow-arrow {
        text-align: center;
        font-size: 24px;
        color: #666;
        padding: 4px 0;
    }
    .flow-blocked {
        color: #dc3545;
    }

    .entry-node {
        background: #e3f2fd !important;
        border-color: #1976d2 !important;
    }
    .exit-node {
        background: #fce4ec !important;
        border-color: #c2185b !important;
    }

    .section-header {
        color: #495057;
        font-size: 14px;
        font-weight: 600;
        margin: 16px 0 8px 0;
        padding-left: 4px;
    }

    .blocking-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
        margin-top: 4px;
    }

    .legend-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-top: 16px;
    }

    /* SVG Flow lines */
    .flow-svg {
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# === COMPONENTS ===

def render_node(node: NodeState, show_blocking: bool = False):
    """Render a single node with status styling."""
    status_class = f"node-{node.status}"
    colors = STATUS_COLORS[node.status]

    # Special styling for entry/exit nodes
    extra_class = ""
    if node.node_type == "entry":
        extra_class = "entry-node"
    elif node.node_type == "exit":
        extra_class = "exit-node"

    # Utilisation bar
    util_pct = node.utilisation * 100
    progress_class = f"progress-{node.status}"

    if node.capacity:
        capacity_text = f"{node.occupied} / {node.capacity}"
        progress_html = f'''
        <div class="progress-bar">
            <div class="progress-fill {progress_class}" style="width: {util_pct}%"></div>
        </div>
        '''
    else:
        capacity_text = f"{node.throughput_per_hour:.1f}/hr"
        progress_html = ""

    blocking_html = ""
    if show_blocking:
        blocking_html = '<div class="blocking-warning">⚠️ BLOCKING</div>'

    st.markdown(f'''
    <div class="node-container {status_class} {extra_class}">
        <div class="node-label">{node.label} {colors["icon"]}</div>
        <div class="node-stats">{capacity_text}</div>
        {progress_html}
        <div class="node-stats">Wait: {node.mean_wait_mins:.1f} min</div>
        {blocking_html}
    </div>
    ''', unsafe_allow_html=True)


def render_flow_arrow(blocked: bool = False, label: str = ""):
    """Render a vertical flow arrow between nodes."""
    arrow_class = "flow-blocked" if blocked else ""
    arrow = "⬇️" if not blocked else "⛔"
    label_html = f'<div style="font-size: 10px; color: #666;">{label}</div>' if label else ""
    st.markdown(f'''
    <div class="flow-arrow {arrow_class}">
        {label_html}
        {arrow}
    </div>
    ''', unsafe_allow_html=True)


def render_horizontal_flow(blocked: bool = False, label: str = ""):
    """Render a horizontal flow indicator."""
    arrow = "→" if not blocked else "⊗"
    color = "#dc3545" if blocked else "#666"
    label_html = f'<span style="font-size: 10px; color: {color};">{label} </span>' if label else ""
    st.markdown(f'''
    <div style="text-align: center; padding: 8px;">
        {label_html}
        <span style="font-size: 24px; color: {color};">{arrow}</span>
    </div>
    ''', unsafe_allow_html=True)


# === MAIN LAYOUT ===

def render_schematic(data: SchematicData):
    """Render the complete schematic."""

    # Header metrics
    st.markdown(f"### 🏥 System Schematic — {data.timestamp}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("In System", data.total_in_system)
    with col2:
        st.metric("24h Throughput", data.total_throughput_24h)
    with col3:
        status_icon = STATUS_COLORS[data.overall_status]["icon"]
        st.metric("Status", f"{status_icon} {data.overall_status.title()}")
    with col4:
        blocked_edges = sum(1 for e in data.edges if e.is_blocked)
        st.metric("Blocked Flows", blocked_edges, delta=None if blocked_edges == 0 else "!", delta_color="inverse")

    st.markdown("---")

    # === ARRIVALS ROW ===
    st.markdown('<div class="section-header">📥 Arrivals</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        render_node(data.nodes["ambulance"])
    with cols[1]:
        render_node(data.nodes["walkin"])
    with cols[2]:
        render_node(data.nodes["hems"])

    render_flow_arrow(label="to Triage")

    # === TRIAGE ROW ===
    st.markdown('<div class="section-header">🏷️ Assessment</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_node(data.nodes["triage"])

    render_flow_arrow(label="to ED")

    # === ED ROW ===
    st.markdown('<div class="section-header">🚨 Emergency Department</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_node(data.nodes["ed_bays"])

    render_flow_arrow(label="to Downstream")

    # === DOWNSTREAM ROW ===
    st.markdown('<div class="section-header">🏨 Downstream</div>', unsafe_allow_html=True)
    cols = st.columns(4)

    # Check for blocking on each node
    itu_blocked = any(e.is_blocked for e in data.edges if e.target == "itu")
    ward_blocked = any(e.is_blocked for e in data.edges if e.source == "itu" and e.target == "ward")

    with cols[0]:
        render_node(data.nodes["theatre"])
    with cols[1]:
        render_node(data.nodes["itu"], show_blocking=itu_blocked)
    with cols[2]:
        render_node(data.nodes["ward"], show_blocking=ward_blocked)
    with cols[3]:
        render_node(data.nodes["discharge"])

    # === FLOW SUMMARY ===
    st.markdown("---")

    # Show blocked flows detail
    blocked_flows = [e for e in data.edges if e.is_blocked]
    if blocked_flows:
        st.markdown("#### ⚠️ Blocked Flows")
        for edge in blocked_flows:
            source_label = data.nodes[edge.source].label
            target_label = data.nodes[edge.target].label
            st.markdown(f"- **{source_label}** → **{target_label}** ({edge.volume_per_hour:.1f}/hr blocked)")

    # === LEGEND ===
    st.markdown("---")
    st.markdown('<div class="legend-box">', unsafe_allow_html=True)
    st.markdown("**Legend**")
    leg_cols = st.columns(5)
    with leg_cols[0]:
        st.markdown("🟢 Normal (<70%)")
    with leg_cols[1]:
        st.markdown("🟡 Warning (70-90%)")
    with leg_cols[2]:
        st.markdown("🔴 Critical (>90%)")
    with leg_cols[3]:
        st.markdown("⛔ Blocked flow")
    with leg_cols[4]:
        st.markdown("🔵 Entry / 🩷 Exit")
    st.markdown('</div>', unsafe_allow_html=True)


def render_interactive_controls(data: SchematicData) -> SchematicData:
    """Render interactive controls for adjusting sample data."""
    with st.expander("🔧 Adjust Sample Data", expanded=False):
        st.markdown("Use these sliders to see how the schematic responds to changes.")

        col1, col2 = st.columns(2)

        with col1:
            ed_occupied = st.slider(
                "ED Occupied",
                0, 30,
                data.nodes["ed_bays"].occupied,
                key="ed_slider"
            )
            triage_occupied = st.slider(
                "Triage Occupied",
                0, 2,
                data.nodes["triage"].occupied,
                key="triage_slider"
            )

        with col2:
            itu_occupied = st.slider(
                "ITU Occupied",
                0, 10,
                data.nodes["itu"].occupied,
                key="itu_slider"
            )
            ward_occupied = st.slider(
                "Ward Occupied",
                0, 50,
                data.nodes["ward"].occupied,
                key="ward_slider"
            )

        # Update data with new values
        data.nodes["ed_bays"].occupied = ed_occupied
        data.nodes["triage"].occupied = triage_occupied
        data.nodes["itu"].occupied = itu_occupied
        data.nodes["ward"].occupied = ward_occupied

        # Update blocking based on ITU status
        if itu_occupied >= 10:
            for edge in data.edges:
                if edge.target == "itu":
                    edge.is_blocked = True
        else:
            for edge in data.edges:
                if edge.target == "itu":
                    edge.is_blocked = False

        # Update overall status
        max_util = max(
            data.nodes["ed_bays"].utilisation,
            data.nodes["itu"].utilisation,
            data.nodes["ward"].utilisation
        )
        if max_util >= 0.90:
            data.overall_status = "critical"
        elif max_util >= 0.70:
            data.overall_status = "warning"
        else:
            data.overall_status = "normal"

        data.total_in_system = ed_occupied + itu_occupied + ward_occupied + triage_occupied

    return data


# === MAIN ===

def main():
    st.set_page_config(
        page_title="Schematic Demo - Streamlit",
        page_icon="🏥",
        layout="wide",
    )

    inject_css()

    st.title("Schematic Demo: Pure Streamlit")
    st.caption("This demo uses only native Streamlit components + custom CSS/HTML")

    # Load sample data
    data = create_sample_data()

    # Interactive controls (returns modified data)
    data = render_interactive_controls(data)

    # Render schematic
    render_schematic(data)

    # Technical notes
    with st.expander("📋 Technical Notes"):
        st.markdown("""
        **Approach**: Pure Streamlit + CSS

        **Pros**:
        - No build step required
        - Easy to iterate and modify
        - Native Streamlit state management
        - Works immediately

        **Cons**:
        - Limited flow line rendering (no SVG arrows between nodes)
        - Hover effects limited to CSS only
        - Layout constrained by Streamlit column system
        - Cannot easily draw diagonal connections

        **Visual Quality**: 3/5
        **Flow Lines**: 2/5 (limited to vertical arrows)
        **Interactivity**: 3/5
        **Dev Time**: Low
        **Maintenance**: Easy
        """)


if __name__ == "__main__":
    main()
