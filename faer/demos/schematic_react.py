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

    # Full-width schematic (1400x1000 viewBox, 20% larger elements)
    clicked = react_schematic(
        data=data_dict,
        width=1400,
        height=750,
        key="main_schematic",
    )

    # Node details below schematic
    if clicked and clicked in data.nodes:
        st.markdown("---")
        node = data.nodes[clicked]

        # Horizontal layout for node details
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"### {node.label}")
            st.markdown(f"Type: `{node.node_type}`")

        with col2:
            if node.capacity:
                st.metric("Utilisation", f"{node.utilisation:.0%}")
                st.progress(node.utilisation)
            else:
                st.metric("Throughput", f"{node.throughput_per_hour:.1f}/hr")

        with col3:
            st.metric("Mean Wait", f"{node.mean_wait_mins:.1f} min")
            if node.capacity:
                st.metric("Occupied", f"{node.occupied}/{node.capacity}")

        with col4:
            # Show connected edges
            st.markdown("**Flows**")
            incoming = [e for e in data.edges if e.target == clicked]
            outgoing = [e for e in data.edges if e.source == clicked]

            if incoming:
                for e in incoming:
                    status = "🔴" if e.is_blocked else "🟢"
                    st.markdown(f"{status} ← {data.nodes[e.source].label}: {e.volume_per_hour:.1f}/hr")

            if outgoing:
                for e in outgoing:
                    status = "🔴" if e.is_blocked else "🟢"
                    st.markdown(f"{status} → {data.nodes[e.target].label}: {e.volume_per_hour:.1f}/hr")
    else:
        st.caption("*Click a node in the schematic to see details*")

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
    """Render a fallback SVG schematic when React component isn't built.

    Layout: Left-to-right crucifix pattern

                              [ITU]
                                ↑
    [Arrivals] → [Triage] → [ED Bays] → [Theatre] → [Discharge]
                                ↓
                             [Ward]
    """
    data = create_sample_data()

    # Build inline SVG - left-to-right crucifix layout
    svg_content = """
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
        <text x="20" y="25" font-size="16" font-weight="bold" fill="#333">System Schematic — Hour 14.5</text>
        <text x="20" y="45" font-size="11" fill="#666">In System: <tspan font-weight="bold" fill="#333">73</tspan>  |  24hr Throughput: <tspan font-weight="bold" fill="#333">432</tspan></text>

        <!-- === MAIN HORIZONTAL FLOW (Y=235 center line) === -->

        <!-- Arrivals → Triage -->
        <path d="M 120 235 L 175 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

        <!-- Triage → ED -->
        <path d="M 295 235 L 355 235" stroke="#adb5bd" stroke-width="3" fill="none" marker-end="url(#arrow)"/>

        <!-- ED → Theatre -->
        <path d="M 475 235 L 535 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

        <!-- Theatre → Discharge -->
        <path d="M 655 235 L 715 235" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

        <!-- === VERTICAL FLOWS FROM ED (crucifix arms) === -->

        <!-- ED → ITU (up) - BLOCKED -->
        <path d="M 415 185 L 415 130" stroke="#dc3545" stroke-width="2" stroke-dasharray="6,3" fill="none" marker-end="url(#blocked-arrow)"/>

        <!-- ED → Ward (down) -->
        <path d="M 415 285 L 415 340" stroke="#adb5bd" stroke-width="2" fill="none" marker-end="url(#arrow)"/>

        <!-- === SECONDARY FLOWS === -->

        <!-- Theatre → ITU (curved up-left) - BLOCKED -->
        <path d="M 560 185 C 560 130, 480 90, 480 90" stroke="#dc3545" stroke-width="1.5" stroke-dasharray="4,2" fill="none" marker-end="url(#blocked-arrow)"/>

        <!-- ITU → Ward (curved down) - BLOCKED -->
        <path d="M 350 90 C 310 90, 310 350, 350 390" stroke="#dc3545" stroke-width="1.5" stroke-dasharray="4,2" fill="none" marker-end="url(#blocked-arrow)"/>

        <!-- Ward → Discharge (curved right) -->
        <path d="M 480 390 C 600 390, 750 320, 780 275" stroke="#adb5bd" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>

        <!-- ITU → Discharge (curved right) -->
        <path d="M 480 70 C 600 50, 750 150, 780 195" stroke="#adb5bd" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>

        <!-- === ARRIVALS (left side, stacked vertically) === -->
        <g transform="translate(20, 135)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">Ambulance</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">12.0/hr</text>
        </g>

        <g transform="translate(20, 210)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">Walk-in</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">8.0/hr</text>
        </g>

        <g transform="translate(20, 285)">
            <rect width="100" height="50" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565c0">HEMS</text>
            <text x="50" y="36" text-anchor="middle" font-size="10" fill="#333">0.5/hr</text>
        </g>

        <!-- === TRIAGE (assessment) === -->
        <g transform="translate(180, 195)">
            <rect width="110" height="80" rx="6" fill="#f0fff4" stroke="#28a745" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="#28a745"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#166534">Triage</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="45" height="8" rx="4" fill="#28a745"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">1/2 (50%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: 3m</text>
        </g>

        <!-- === ED BAYS (center - main hub) === -->
        <g transform="translate(360, 190)">
            <rect width="110" height="90" rx="6" fill="#fffbf0" stroke="#ffc107" stroke-width="3"/>
            <circle cx="98" cy="12" r="5" fill="#ffc107"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#92400e">ED Bays</text>
            <rect x="10" y="32" width="90" height="10" rx="5" fill="#e9ecef"/>
            <rect x="10" y="32" width="72" height="10" rx="5" fill="#ffc107"/>
            <text x="55" y="58" text-anchor="middle" font-size="10" fill="#333">24/30 (80%)</text>
            <text x="55" y="76" text-anchor="middle" font-size="9" fill="#666">Wait: 23m</text>
        </g>

        <!-- === ITU (top arm of crucifix) - CRITICAL === -->
        <g transform="translate(360, 45)">
            <rect width="110" height="80" rx="6" fill="#fff5f5" stroke="#dc3545" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="#dc3545"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#991b1b">ITU</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#dc3545"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">10/10 (100%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: 120m</text>
        </g>

        <!-- === WARD (bottom arm of crucifix) === -->
        <g transform="translate(360, 345)">
            <rect width="110" height="80" rx="6" fill="#fffbf0" stroke="#ffc107" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="#ffc107"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#92400e">Ward</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="68" height="8" rx="4" fill="#ffc107"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">38/50 (76%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: 35m</text>
        </g>

        <!-- === THEATRE (surgery, right of ED) === -->
        <g transform="translate(540, 195)">
            <rect width="110" height="80" rx="6" fill="#f0fff4" stroke="#28a745" stroke-width="2"/>
            <circle cx="98" cy="12" r="5" fill="#28a745"/>
            <text x="55" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#166534">Theatre</text>
            <rect x="10" y="30" width="90" height="8" rx="4" fill="#e9ecef"/>
            <rect x="10" y="30" width="45" height="8" rx="4" fill="#28a745"/>
            <text x="55" y="52" text-anchor="middle" font-size="10" fill="#333">1/2 (50%)</text>
            <text x="55" y="68" text-anchor="middle" font-size="9" fill="#666">Wait: 45m</text>
        </g>

        <!-- === DISCHARGE (exit, far right) === -->
        <g transform="translate(720, 195)">
            <rect width="100" height="80" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
            <circle cx="88" cy="12" r="5" fill="#28a745"/>
            <text x="50" y="18" text-anchor="middle" font-size="12" font-weight="bold" fill="#6a1b9a">Discharge</text>
            <text x="50" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">15.0/hr</text>
            <text x="50" y="65" text-anchor="middle" font-size="9" fill="#666">Exit</text>
        </g>

        <!-- Legend (bottom right) -->
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

        <!-- Section labels -->
        <text x="45" y="125" font-size="9" fill="#888" font-weight="500">ARRIVALS</text>
        <text x="202" y="185" font-size="9" fill="#888" font-weight="500">ASSESS</text>
        <text x="382" y="180" font-size="9" fill="#888" font-weight="500">EMERGENCY</text>
        <text x="567" y="185" font-size="9" fill="#888" font-weight="500">SURGERY</text>
        <text x="745" y="185" font-size="9" fill="#888" font-weight="500">EXIT</text>
    </svg>
    """

    st.components.v1.html(svg_content, height=490)

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
