"""
Schematic Demo - React Component Implementation

Run with: streamlit run faer/demos/schematic_react.py

Prerequisites:
1. cd faer/demos/components/react_schematic
2. npm install
3. npm run build

This demo uses a custom React/SVG component for richer interactivity.
COMPLETELY ISOLATED - no imports from main application.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from sample_data import create_sample_data, to_dict, SchematicData


# === COMPONENT DECLARATION ===

# Path to compiled React component
COMPONENT_PATH = Path(__file__).parent / "components" / "react_schematic" / "build"

# Declare the component
_react_schematic = components.declare_component(
    "react_schematic",
    path=str(COMPONENT_PATH),
)


def react_schematic(
    data: dict,
    width: int = 900,
    height: int = 550,
    key: str = None
) -> str | None:
    """
    Render React-based schematic component.

    Args:
        data: SchematicData as JSON-serializable dict
        width: Component width in pixels
        height: Component height in pixels
        key: Streamlit component key

    Returns:
        Clicked node ID if any, else None
    """
    return _react_schematic(
        data=data,
        width=width,
        height=height,
        key=key,
        default=None,
    )


# === MAIN ===

def render_sidebar_controls(data: SchematicData) -> SchematicData:
    """Render controls in sidebar for adjusting sample data."""
    st.sidebar.header("🔧 Adjust Data")
    st.sidebar.caption("Modify values to see real-time updates")

    ed_occupied = st.sidebar.slider(
        "ED Occupied",
        0, 30,
        data.nodes["ed_bays"].occupied,
        key="ed_slider_react"
    )
    itu_occupied = st.sidebar.slider(
        "ITU Occupied",
        0, 10,
        data.nodes["itu"].occupied,
        key="itu_slider_react"
    )
    ward_occupied = st.sidebar.slider(
        "Ward Occupied",
        0, 50,
        data.nodes["ward"].occupied,
        key="ward_slider_react"
    )
    triage_occupied = st.sidebar.slider(
        "Triage Occupied",
        0, 2,
        data.nodes["triage"].occupied,
        key="triage_slider_react"
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


def main():
    st.set_page_config(
        page_title="Schematic Demo - React",
        page_icon="⚛️",
        layout="wide",
    )

    st.title("Schematic Demo: React Component")
    st.caption("This demo uses a custom React/SVG component for richer interactivity")

    # Check if component is built
    bundle_path = COMPONENT_PATH / "bundle.js"
    if not bundle_path.exists():
        st.warning(f"""
        **React component not yet built.**

        The React component needs to be compiled before use. Run these commands:

        ```bash
        cd faer/demos/components/react_schematic
        npm install
        npm run build
        ```

        For now, showing a **fallback HTML/SVG version** below.
        """)

        # Fallback to inline HTML/SVG rendering
        render_fallback_schematic()
        return

    # Load sample data
    data = create_sample_data()

    # Sidebar controls
    data = render_sidebar_controls(data)

    # Convert to dict for React component
    data_dict = to_dict(data)

    # Main layout
    col1, col2 = st.columns([4, 1])

    with col1:
        # Render React schematic
        clicked = react_schematic(
            data=data_dict,
            width=900,
            height=550,
            key="main_schematic",
        )

        if clicked:
            st.info(f"Selected node: **{clicked}**")

    with col2:
        st.markdown("### Node Details")

        if clicked and clicked in data.nodes:
            node = data.nodes[clicked]
            st.markdown(f"**{node.label}**")
            st.markdown(f"Type: `{node.node_type}`")

            if node.capacity:
                st.metric("Utilisation", f"{node.utilisation:.0%}")
                st.progress(node.utilisation)

            st.metric("Mean Wait", f"{node.mean_wait_mins:.1f} min")
            st.metric("Throughput", f"{node.throughput_per_hour:.1f}/hr")

            # Show connected edges
            st.markdown("---")
            st.markdown("**Flows**")
            incoming = [e for e in data.edges if e.target == clicked]
            outgoing = [e for e in data.edges if e.source == clicked]

            if incoming:
                st.markdown("*Incoming:*")
                for e in incoming:
                    status = "🔴" if e.is_blocked else "🟢"
                    st.markdown(f"- {status} {data.nodes[e.source].label}: {e.volume_per_hour:.1f}/hr")

            if outgoing:
                st.markdown("*Outgoing:*")
                for e in outgoing:
                    status = "🔴" if e.is_blocked else "🟢"
                    st.markdown(f"- {status} → {data.nodes[e.target].label}: {e.volume_per_hour:.1f}/hr")
        else:
            st.markdown("*Click a node in the schematic to see details*")

    # Technical notes
    with st.expander("📋 Technical Notes"):
        st.markdown("""
        **Approach**: Custom React Component + SVG

        **Pros**:
        - Full SVG control for flow lines and arrows
        - Rich interactivity (hover, click, selection)
        - Smooth animations and transitions
        - Professional visual appearance
        - Scalable vector graphics

        **Cons**:
        - Requires npm build step
        - More complex development setup
        - Additional dependencies (React, webpack)
        - Longer initial development time

        **Visual Quality**: 5/5
        **Flow Lines**: 5/5 (curved SVG paths with arrows)
        **Interactivity**: 5/5 (hover, click, selection, tooltips)
        **Dev Time**: Higher
        **Maintenance**: Moderate (TypeScript helps)
        """)


def render_fallback_schematic():
    """Render a fallback SVG schematic when React component isn't built."""
    data = create_sample_data()

    # Build inline SVG
    svg_content = """
    <svg width="900" height="550" style="font-family: system-ui; background: #fafafa;">
        <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#adb5bd"/>
            </marker>
            <marker id="blocked-arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#dc3545"/>
            </marker>
        </defs>

        <!-- Header -->
        <text x="20" y="30" font-size="18" font-weight="bold" fill="#333">🏥 System Schematic — Hour 14.5</text>
        <text x="20" y="50" font-size="12" fill="#666">In System: <tspan font-weight="bold" fill="#333">73</tspan></text>

        <!-- Section labels -->
        <text x="20" y="75" font-size="11" fill="#666" font-weight="500">📥 Arrivals</text>
        <text x="20" y="195" font-size="11" fill="#666" font-weight="500">🏷️ Assessment</text>
        <text x="20" y="315" font-size="11" fill="#666" font-weight="500">🚨 Emergency Dept</text>
        <text x="20" y="455" font-size="11" fill="#666" font-weight="500">🏨 Downstream</text>

        <!-- Edges -->
        <path d="M 150 125 C 150 162.5, 450 162.5, 450 155" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 450 125 L 450 155" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 750 125 C 750 162.5, 450 162.5, 450 155" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 450 245 L 450 275" stroke="#adb5bd" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
        <path d="M 450 365 C 450 412.5, 150 412.5, 150 415" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 450 365 C 450 412.5, 350 412.5, 350 415" stroke="#dc3545" stroke-width="3" stroke-dasharray="8,4" fill="none" marker-end="url(#blocked-arrow)"/>
        <path d="M 450 365 C 450 412.5, 550 412.5, 550 415" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
        <path d="M 450 365 C 450 412.5, 750 412.5, 750 415" stroke="#adb5bd" stroke-width="3" fill="none" marker-end="url(#arrow)"/>

        <!-- Entry nodes -->
        <g transform="translate(85, 35)">
            <rect width="130" height="90" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#1565c0">Ambulance</text>
            <text x="65" y="50" text-anchor="middle" font-size="12" fill="#333">12.0/hr</text>
        </g>

        <g transform="translate(385, 35)">
            <rect width="130" height="90" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#1565c0">Walk-in</text>
            <text x="65" y="50" text-anchor="middle" font-size="12" fill="#333">8.0/hr</text>
        </g>

        <g transform="translate(685, 35)">
            <rect width="130" height="90" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#1565c0">HEMS</text>
            <text x="65" y="50" text-anchor="middle" font-size="12" fill="#333">0.5/hr</text>
        </g>

        <!-- Triage -->
        <g transform="translate(385, 155)">
            <rect width="130" height="90" rx="8" fill="#f0fff4" stroke="#28a745" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#166534">Triage</text>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="35" width="55" height="10" rx="5" fill="#28a745"/>
            <text x="65" y="60" text-anchor="middle" font-size="12" fill="#333">1/2 (50%)</text>
            <text x="65" y="78" text-anchor="middle" font-size="10" fill="#666">Wait: 3m</text>
        </g>

        <!-- ED Bays -->
        <g transform="translate(385, 275)">
            <rect width="130" height="90" rx="8" fill="#fffbf0" stroke="#ffc107" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#ffc107"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">ED Bays</text>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="35" width="88" height="10" rx="5" fill="#ffc107"/>
            <text x="65" y="60" text-anchor="middle" font-size="12" fill="#333">24/30 (80%)</text>
            <text x="65" y="78" text-anchor="middle" font-size="10" fill="#666">Wait: 23m</text>
        </g>

        <!-- Downstream nodes -->
        <g transform="translate(85, 415)">
            <rect width="130" height="90" rx="8" fill="#f0fff4" stroke="#28a745" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#166534">Theatre</text>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="35" width="55" height="10" rx="5" fill="#28a745"/>
            <text x="65" y="60" text-anchor="middle" font-size="12" fill="#333">1/2 (50%)</text>
            <text x="65" y="78" text-anchor="middle" font-size="10" fill="#666">Wait: 45m</text>
        </g>

        <g transform="translate(285, 415)">
            <rect width="130" height="90" rx="8" fill="#fff5f5" stroke="#dc3545" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#dc3545"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#991b1b">ITU</text>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#dc3545"/>
            <text x="65" y="60" text-anchor="middle" font-size="12" fill="#333">10/10 (100%)</text>
            <text x="65" y="78" text-anchor="middle" font-size="10" fill="#666">Wait: 120m</text>
        </g>

        <g transform="translate(485, 415)">
            <rect width="130" height="90" rx="8" fill="#fffbf0" stroke="#ffc107" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#ffc107"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">Ward</text>
            <rect x="10" y="35" width="110" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="35" width="84" height="10" rx="5" fill="#ffc107"/>
            <text x="65" y="60" text-anchor="middle" font-size="12" fill="#333">38/50 (76%)</text>
            <text x="65" y="78" text-anchor="middle" font-size="10" fill="#666">Wait: 35m</text>
        </g>

        <g transform="translate(685, 415)">
            <rect width="130" height="90" rx="8" fill="#fce4ec" stroke="#c2185b" stroke-width="2"/>
            <circle cx="115" cy="15" r="6" fill="#28a745"/>
            <text x="65" y="22" text-anchor="middle" font-size="13" font-weight="bold" fill="#880e4f">Discharge</text>
            <text x="65" y="50" text-anchor="middle" font-size="12" fill="#333">15.0/hr</text>
        </g>

        <!-- Legend -->
        <g transform="translate(720, 440)">
            <rect width="165" height="100" fill="white" stroke="#dee2e6" rx="6" fill-opacity="0.95"/>
            <text x="10" y="20" font-size="12" font-weight="bold" fill="#333">Legend</text>
            <circle cx="20" cy="38" r="5" fill="#28a745"/>
            <text x="32" y="42" font-size="10" fill="#333">Normal (&lt;70%)</text>
            <circle cx="20" cy="55" r="5" fill="#ffc107"/>
            <text x="32" y="59" font-size="10" fill="#333">Warning (70-90%)</text>
            <circle cx="20" cy="72" r="5" fill="#dc3545"/>
            <text x="32" y="76" font-size="10" fill="#333">Critical (&gt;90%)</text>
            <line x1="15" y1="88" x2="40" y2="88" stroke="#dc3545" stroke-width="2" stroke-dasharray="4,2"/>
            <text x="48" y="92" font-size="10" fill="#333">Blocked</text>
        </g>
    </svg>
    """

    st.components.v1.html(svg_content, height=560)

    st.info("""
    **This is a static fallback.** Build the React component for full interactivity:

    ```bash
    cd faer/demos/components/react_schematic
    npm install
    npm run build
    ```
    """)


if __name__ == "__main__":
    main()
