# CLAUDE.md - Schematic Demo Tabs

## Context

You are working on two demo schematic tabs for Pj-FAER, comparing Pure Streamlit vs React component approaches. These are ISOLATED sandbox pages for evaluation - they receive data but have NO dependencies on or from the main application.

**Key Principle**: These demos exist to compare rendering approaches. They should be self-contained, use identical sample data, and produce visually comparable output.

---

## Feature Scope

### What These Demos Do

- Render a hospital flow schematic with identical data
- Display node status (capacity, utilisation, wait times)
- Show flow between nodes with volume indicators
- Indicate blocking/critical states visually
- Allow visual comparison of Streamlit vs React approaches

### What These Demos Do NOT Do

- Connect to live simulation
- Modify any state
- Import from main application code
- Export to main application code
- Persist any data

### Isolation Contract

```
┌─────────────────────────────────────────────────────────────────┐
│  MAIN APPLICATION                                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐
│  │  faer/                                                      │
│  │  ├── ui/app.py         ← Main app (DO NOT MODIFY)          │
│  │  ├── model/            ← Simulation (DO NOT IMPORT)         │
│  │  └── ...                                                    │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
│                         ❌ NO CONNECTION ❌                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐
│  │  faer/demos/           ← ISOLATED SANDBOX                   │
│  │  ├── schematic_streamlit.py                                 │
│  │  ├── schematic_react.py                                     │
│  │  ├── sample_data.py    ← Shared mock data                   │
│  │  └── components/       ← React component source             │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tech options | Pure Streamlit vs React | Compare native vs custom component |
| Isolation | Complete sandbox | No risk to main app during iteration |
| Data | Shared sample_data.py | Identical input for fair comparison |
| Evaluation | Side-by-side criteria | Visual quality, interactivity, dev effort |

---

## Directory Structure

```
faer/
├── demos/
│   ├── __init__.py
│   ├── README.md                    # How to run demos
│   ├── sample_data.py               # Shared data structures + mock data
│   ├── schematic_streamlit.py       # Demo 1: Pure Streamlit
│   ├── schematic_react.py           # Demo 2: React wrapper
│   └── components/
│       └── react_schematic/
│           ├── package.json
│           ├── tsconfig.json
│           ├── src/
│           │   ├── index.tsx        # Streamlit component entry
│           │   ├── Schematic.tsx    # Main schematic component
│           │   ├── nodes/
│           │   │   ├── ResourceNode.tsx
│           │   │   ├── EntryNode.tsx
│           │   │   └── ExitNode.tsx
│           │   ├── edges/
│           │   │   └── FlowEdge.tsx
│           │   └── styles/
│           │       └── schematic.css
│           └── build/
│               └── bundle.js        # Compiled output
```

---

## Shared Data Model

### sample_data.py

This file defines the data contract that BOTH demos must render.

```python
"""
Shared data structures for schematic demos.

Both Streamlit and React demos consume this same data format.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class NodeState:
    """State of a single resource node."""
    id: str
    label: str
    node_type: str  # "entry" | "resource" | "process" | "exit"
    
    # Capacity (None for entry/exit nodes)
    capacity: Optional[int]
    occupied: int
    
    # Performance metrics
    throughput_per_hour: float
    mean_wait_mins: float
    
    @property
    def utilisation(self) -> float:
        """Calculate utilisation ratio."""
        if self.capacity is None or self.capacity == 0:
            return 0.0
        return min(1.0, self.occupied / self.capacity)
    
    @property
    def status(self) -> str:
        """Determine status based on utilisation."""
        if self.utilisation >= 0.90:
            return "critical"
        elif self.utilisation >= 0.70:
            return "warning"
        return "normal"


@dataclass
class FlowEdge:
    """Flow connection between two nodes."""
    source: str      # Source node ID
    target: str      # Target node ID
    volume_per_hour: float
    is_blocked: bool = False


@dataclass
class SchematicData:
    """Complete schematic state for rendering."""
    timestamp: str   # Display label (e.g., "Hour 14.5")
    nodes: Dict[str, NodeState]
    edges: List[FlowEdge]
    
    # Summary metrics
    total_in_system: int
    total_throughput_24h: int
    overall_status: str  # "normal" | "warning" | "critical"


def create_sample_data() -> SchematicData:
    """
    Create sample schematic data for demos.
    
    This represents a hospital mid-simulation with:
    - ED at 80% capacity (warning)
    - ITU at 100% capacity (critical, blocking)
    - Some downstream blocking occurring
    """
    nodes = {
        # Entry nodes
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=12.0,
            mean_wait_mins=0.0,
        ),
        "walkin": NodeState(
            id="walkin",
            label="Walk-in",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=8.0,
            mean_wait_mins=0.0,
        ),
        "hems": NodeState(
            id="hems",
            label="HEMS",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=0.5,
            mean_wait_mins=0.0,
        ),
        
        # Process nodes
        "triage": NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=2,
            occupied=1,
            throughput_per_hour=20.5,
            mean_wait_mins=3.2,
        ),
        
        # Resource nodes
        "ed_bays": NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=30,
            occupied=24,
            throughput_per_hour=18.0,
            mean_wait_mins=23.0,
        ),
        "theatre": NodeState(
            id="theatre",
            label="Theatre",
            node_type="resource",
            capacity=2,
            occupied=1,
            throughput_per_hour=1.5,
            mean_wait_mins=45.0,
        ),
        "itu": NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=10,
            occupied=10,  # Full!
            throughput_per_hour=0.8,
            mean_wait_mins=120.0,
        ),
        "ward": NodeState(
            id="ward",
            label="Ward",
            node_type="resource",
            capacity=50,
            occupied=38,
            throughput_per_hour=3.2,
            mean_wait_mins=35.0,
        ),
        
        # Exit node
        "discharge": NodeState(
            id="discharge",
            label="Discharge",
            node_type="exit",
            capacity=None,
            occupied=0,
            throughput_per_hour=15.0,
            mean_wait_mins=0.0,
        ),
    }
    
    edges = [
        # Arrivals → Triage
        FlowEdge(source="ambulance", target="triage", volume_per_hour=12.0),
        FlowEdge(source="walkin", target="triage", volume_per_hour=8.0),
        FlowEdge(source="hems", target="triage", volume_per_hour=0.5),
        
        # Triage → ED
        FlowEdge(source="triage", target="ed_bays", volume_per_hour=20.5),
        
        # ED → Downstream
        FlowEdge(source="ed_bays", target="theatre", volume_per_hour=1.5),
        FlowEdge(source="ed_bays", target="itu", volume_per_hour=0.8, is_blocked=True),
        FlowEdge(source="ed_bays", target="ward", volume_per_hour=5.0),
        FlowEdge(source="ed_bays", target="discharge", volume_per_hour=10.0),
        
        # Theatre → Post-op
        FlowEdge(source="theatre", target="itu", volume_per_hour=0.5, is_blocked=True),
        FlowEdge(source="theatre", target="ward", volume_per_hour=0.8),
        
        # ITU → Step-down/Discharge
        FlowEdge(source="itu", target="ward", volume_per_hour=0.6, is_blocked=True),
        FlowEdge(source="itu", target="discharge", volume_per_hour=0.2),
        
        # Ward → Discharge
        FlowEdge(source="ward", target="discharge", volume_per_hour=4.0),
    ]
    
    return SchematicData(
        timestamp="Hour 14.5",
        nodes=nodes,
        edges=edges,
        total_in_system=73,
        total_throughput_24h=432,
        overall_status="warning",
    )


def to_dict(data: SchematicData) -> dict:
    """Convert SchematicData to JSON-serializable dict for React component."""
    return {
        "timestamp": data.timestamp,
        "nodes": {
            k: {
                "id": v.id,
                "label": v.label,
                "node_type": v.node_type,
                "capacity": v.capacity,
                "occupied": v.occupied,
                "throughput_per_hour": v.throughput_per_hour,
                "mean_wait_mins": v.mean_wait_mins,
                "utilisation": v.utilisation,
                "status": v.status,
            }
            for k, v in data.nodes.items()
        },
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "volume_per_hour": e.volume_per_hour,
                "is_blocked": e.is_blocked,
            }
            for e in data.edges
        ],
        "total_in_system": data.total_in_system,
        "total_throughput_24h": data.total_throughput_24h,
        "overall_status": data.overall_status,
    }
```

---

## Demo 1: Pure Streamlit

### File: schematic_streamlit.py

```python
"""
Schematic Demo - Pure Streamlit Implementation

Run with: streamlit run faer/demos/schematic_streamlit.py
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
        background: #e3f2fd;
        border-color: #1976d2;
    }
    .exit-node {
        background: #fce4ec;
        border-color: #c2185b;
    }
    </style>
    """, unsafe_allow_html=True)


# === COMPONENTS ===

def render_node(node: NodeState):
    """Render a single node."""
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
    
    st.markdown(f'''
    <div class="node-container {status_class} {extra_class}">
        <div class="node-label">{node.label} {colors["icon"]}</div>
        <div class="node-stats">{capacity_text}</div>
        {progress_html}
        <div class="node-stats">Wait: {node.mean_wait_mins:.1f} min</div>
    </div>
    ''', unsafe_allow_html=True)


def render_flow_arrow(blocked: bool = False):
    """Render a flow arrow between nodes."""
    arrow_class = "flow-blocked" if blocked else ""
    arrow = "🔻" if not blocked else "⛔"
    st.markdown(f'<div class="flow-arrow {arrow_class}">{arrow}</div>', unsafe_allow_html=True)


def render_horizontal_arrow(blocked: bool = False):
    """Render horizontal flow arrow."""
    arrow = "→" if not blocked else "⊗"
    color = "#dc3545" if blocked else "#666"
    st.markdown(f'<span style="font-size: 24px; color: {color};">{arrow}</span>', unsafe_allow_html=True)


# === MAIN LAYOUT ===

def render_schematic(data: SchematicData):
    """Render the complete schematic."""
    
    # Header
    st.markdown(f"### 🏥 System Schematic — {data.timestamp}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("In System", data.total_in_system)
    with col2:
        st.metric("24h Throughput", data.total_throughput_24h)
    with col3:
        status_icon = STATUS_COLORS[data.overall_status]["icon"]
        st.metric("Status", f"{status_icon} {data.overall_status.title()}")
    
    st.markdown("---")
    
    # === ARRIVALS ROW ===
    st.markdown("**Arrivals**")
    cols = st.columns(3)
    with cols[0]:
        render_node(data.nodes["ambulance"])
    with cols[1]:
        render_node(data.nodes["walkin"])
    with cols[2]:
        render_node(data.nodes["hems"])
    
    render_flow_arrow()
    
    # === TRIAGE ROW ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_node(data.nodes["triage"])
    
    render_flow_arrow()
    
    # === ED ROW ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_node(data.nodes["ed_bays"])
    
    render_flow_arrow()
    
    # === DOWNSTREAM ROW ===
    st.markdown("**Downstream**")
    cols = st.columns(4)
    
    with cols[0]:
        render_node(data.nodes["theatre"])
    with cols[1]:
        # Find if ED→ITU is blocked
        itu_blocked = any(e.is_blocked for e in data.edges if e.target == "itu")
        render_node(data.nodes["itu"])
        if itu_blocked:
            st.caption("⚠️ BLOCKING")
    with cols[2]:
        render_node(data.nodes["ward"])
    with cols[3]:
        render_node(data.nodes["discharge"])
    
    # === LEGEND ===
    st.markdown("---")
    st.markdown("**Legend**")
    leg_cols = st.columns(4)
    with leg_cols[0]:
        st.markdown("🟢 Normal (<70%)")
    with leg_cols[1]:
        st.markdown("🟡 Warning (70-90%)")
    with leg_cols[2]:
        st.markdown("🔴 Critical (>90%)")
    with leg_cols[3]:
        st.markdown("⛔ Blocked flow")


# === MAIN ===

def main():
    st.set_page_config(
        page_title="Schematic Demo - Streamlit",
        page_icon="🏥",
        layout="wide",
    )
    
    inject_css()
    
    st.title("Schematic Demo: Pure Streamlit")
    st.caption("This demo uses only native Streamlit components + custom CSS")
    
    # Load sample data
    data = create_sample_data()
    
    # Render schematic
    render_schematic(data)
    
    # Controls for testing
    with st.expander("🔧 Adjust Sample Data"):
        st.markdown("Use these sliders to see how the schematic responds to changes.")
        
        ed_occupied = st.slider("ED Occupied", 0, 30, 24)
        itu_occupied = st.slider("ITU Occupied", 0, 10, 10)
        
        if ed_occupied != 24 or itu_occupied != 10:
            data.nodes["ed_bays"].occupied = ed_occupied
            data.nodes["itu"].occupied = itu_occupied
            st.experimental_rerun()


if __name__ == "__main__":
    main()
```

---

## Demo 2: React Component

### File: schematic_react.py

```python
"""
Schematic Demo - React Component Implementation

Run with: streamlit run faer/demos/schematic_react.py

Prerequisites:
1. cd faer/demos/components/react_schematic
2. npm install
3. npm run build
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from sample_data import create_sample_data, to_dict


# === COMPONENT DECLARATION ===

# Path to compiled React component
COMPONENT_PATH = Path(__file__).parent / "components" / "react_schematic" / "build"

# Declare the component
_react_schematic = components.declare_component(
    "react_schematic",
    path=str(COMPONENT_PATH),
)


def react_schematic(data: dict, width: int = 900, height: int = 600, key: str = None):
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

def main():
    st.set_page_config(
        page_title="Schematic Demo - React",
        page_icon="⚛️",
        layout="wide",
    )
    
    st.title("Schematic Demo: React Component")
    st.caption("This demo uses a custom React/SVG component")
    
    # Check if component is built
    if not COMPONENT_PATH.exists():
        st.error(f"""
        React component not built. Please run:
        ```
        cd faer/demos/components/react_schematic
        npm install
        npm run build
        ```
        """)
        return
    
    # Load sample data
    data = create_sample_data()
    data_dict = to_dict(data)
    
    # Render React schematic
    col1, col2 = st.columns([3, 1])
    
    with col1:
        clicked = react_schematic(
            data=data_dict,
            width=900,
            height=600,
            key="main_schematic",
        )
        
        if clicked:
            st.info(f"Clicked node: {clicked}")
    
    with col2:
        st.markdown("### Node Details")
        st.markdown("Click a node in the schematic to see details here.")
        
        if clicked and clicked in data.nodes:
            node = data.nodes[clicked]
            st.markdown(f"**{node.label}**")
            st.metric("Utilisation", f"{node.utilisation:.0%}")
            st.metric("Mean Wait", f"{node.mean_wait_mins:.1f} min")
            st.metric("Throughput", f"{node.throughput_per_hour:.1f}/hr")
    
    # Controls for testing
    with st.expander("🔧 Adjust Sample Data"):
        st.markdown("Use these sliders to see how the schematic responds to changes.")
        
        ed_occupied = st.slider("ED Occupied", 0, 30, data.nodes["ed_bays"].occupied)
        itu_occupied = st.slider("ITU Occupied", 0, 10, data.nodes["itu"].occupied)
        
        if st.button("Update Schematic"):
            # This would require state management - simplified for demo
            st.info("In production, this would update the schematic in real-time")


if __name__ == "__main__":
    main()
```

### React Component: package.json

```json
{
  "name": "react-schematic",
  "version": "1.0.0",
  "private": true,
  "main": "build/bundle.js",
  "scripts": {
    "build": "webpack --mode production",
    "dev": "webpack --mode development --watch"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "streamlit-component-lib": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "css-loader": "^6.8.1",
    "style-loader": "^3.3.3",
    "ts-loader": "^9.4.4",
    "typescript": "^5.1.6",
    "webpack": "^5.88.2",
    "webpack-cli": "^5.1.4"
  }
}
```

### React Component: Schematic.tsx (Outline)

```tsx
/**
 * Main Schematic Component
 * 
 * Renders hospital flow schematic as SVG with:
 * - Node boxes showing capacity/utilisation
 * - Flow edges with volume indicators
 * - Status colours and blocking indicators
 * - Click interaction for node details
 */

import React, { useState, useEffect } from "react";
import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";

interface NodeData {
  id: string;
  label: string;
  node_type: string;
  capacity: number | null;
  occupied: number;
  throughput_per_hour: number;
  mean_wait_mins: number;
  utilisation: number;
  status: string;
}

interface EdgeData {
  source: string;
  target: string;
  volume_per_hour: number;
  is_blocked: boolean;
}

interface SchematicData {
  timestamp: string;
  nodes: Record<string, NodeData>;
  edges: EdgeData[];
  total_in_system: number;
  total_throughput_24h: number;
  overall_status: string;
}

interface SchematicProps extends ComponentProps {
  args: {
    data: SchematicData;
    width: number;
    height: number;
  };
}

// Node positions (hardcoded layout for demo)
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  ambulance: { x: 150, y: 60 },
  walkin: { x: 450, y: 60 },
  hems: { x: 750, y: 60 },
  triage: { x: 450, y: 160 },
  ed_bays: { x: 450, y: 280 },
  theatre: { x: 150, y: 420 },
  itu: { x: 350, y: 420 },
  ward: { x: 550, y: 420 },
  discharge: { x: 750, y: 420 },
};

const STATUS_COLORS = {
  normal: { fill: "#f0fff4", stroke: "#28a745" },
  warning: { fill: "#fffbf0", stroke: "#ffc107" },
  critical: { fill: "#fff5f5", stroke: "#dc3545" },
};

const Schematic: React.FC<SchematicProps> = ({ args }) => {
  const { data, width, height } = args;
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const handleNodeClick = (nodeId: string) => {
    Streamlit.setComponentValue(nodeId);
  };

  const renderNode = (node: NodeData) => {
    const pos = NODE_POSITIONS[node.id];
    if (!pos) return null;

    const colors = STATUS_COLORS[node.status as keyof typeof STATUS_COLORS] || STATUS_COLORS.normal;
    const isHovered = hoveredNode === node.id;
    const nodeWidth = 120;
    const nodeHeight = 80;

    return (
      <g
        key={node.id}
        transform={`translate(${pos.x - nodeWidth / 2}, ${pos.y - nodeHeight / 2})`}
        onClick={() => handleNodeClick(node.id)}
        onMouseEnter={() => setHoveredNode(node.id)}
        onMouseLeave={() => setHoveredNode(null)}
        style={{ cursor: "pointer" }}
      >
        {/* Node box */}
        <rect
          width={nodeWidth}
          height={nodeHeight}
          rx={8}
          fill={colors.fill}
          stroke={colors.stroke}
          strokeWidth={isHovered ? 3 : 2}
        />

        {/* Label */}
        <text x={nodeWidth / 2} y={20} textAnchor="middle" fontSize={12} fontWeight="bold">
          {node.label}
        </text>

        {/* Utilisation bar */}
        {node.capacity && (
          <>
            <rect x={10} y={30} width={nodeWidth - 20} height={8} rx={4} fill="#e9ecef" />
            <rect
              x={10}
              y={30}
              width={(nodeWidth - 20) * node.utilisation}
              height={8}
              rx={4}
              fill={colors.stroke}
            />
          </>
        )}

        {/* Stats */}
        <text x={nodeWidth / 2} y={55} textAnchor="middle" fontSize={11}>
          {node.capacity ? `${node.occupied}/${node.capacity}` : `${node.throughput_per_hour.toFixed(1)}/hr`}
        </text>
        <text x={nodeWidth / 2} y={70} textAnchor="middle" fontSize={10} fill="#666">
          Wait: {node.mean_wait_mins.toFixed(1)}m
        </text>
      </g>
    );
  };

  const renderEdge = (edge: EdgeData, index: number) => {
    const source = NODE_POSITIONS[edge.source];
    const target = NODE_POSITIONS[edge.target];
    if (!source || !target) return null;

    const strokeColor = edge.is_blocked ? "#dc3545" : "#adb5bd";
    const strokeDash = edge.is_blocked ? "8,4" : undefined;

    return (
      <g key={index}>
        <line
          x1={source.x}
          y1={source.y + 40}
          x2={target.x}
          y2={target.y - 40}
          stroke={strokeColor}
          strokeWidth={2}
          strokeDasharray={strokeDash}
          markerEnd="url(#arrowhead)"
        />
      </g>
    );
  };

  return (
    <svg width={width} height={height} style={{ fontFamily: "sans-serif" }}>
      {/* Definitions */}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
        </marker>
      </defs>

      {/* Title */}
      <text x={20} y={30} fontSize={16} fontWeight="bold">
        🏥 System Schematic — {data.timestamp}
      </text>

      {/* Edges (behind nodes) */}
      <g className="edges">
        {data.edges.map((edge, i) => renderEdge(edge, i))}
      </g>

      {/* Nodes */}
      <g className="nodes">
        {Object.values(data.nodes).map((node) => renderNode(node))}
      </g>

      {/* Legend */}
      <g transform={`translate(${width - 180}, ${height - 80})`}>
        <rect width={160} height={70} fill="white" stroke="#ccc" rx={4} />
        <text x={10} y={20} fontSize={11}>🟢 Normal (&lt;70%)</text>
        <text x={10} y={38} fontSize={11}>🟡 Warning (70-90%)</text>
        <text x={10} y={56} fontSize={11}>🔴 Critical (&gt;90%)</text>
      </g>
    </svg>
  );
};

export default withStreamlitConnection(Schematic);
```

---

## Evaluation Criteria

After building both demos, compare:

| Criterion | Weight | Streamlit | React | Notes |
|-----------|--------|-----------|-------|-------|
| **Visual Quality** | 25% | | | Professional appearance |
| **Flow Lines** | 15% | | | Can show arrows/connections |
| **Interactivity** | 20% | | | Hover, click, drill-down |
| **Responsiveness** | 10% | | | Resize behaviour |
| **Dev Time** | 15% | | | Hours to build |
| **Maintenance** | 10% | | | Ease of updates |
| **Dependencies** | 5% | | | External tooling needed |

### Scoring Guide

- 5 = Excellent
- 4 = Good
- 3 = Adequate
- 2 = Poor
- 1 = Unacceptable

---

## Running the Demos

```bash
# Demo 1: Pure Streamlit
streamlit run faer/demos/schematic_streamlit.py

# Demo 2: React (requires build first)
cd faer/demos/components/react_schematic
npm install
npm run build
cd ../../../..
streamlit run faer/demos/schematic_react.py
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `faer/demos/__init__.py` | Package marker |
| `faer/demos/README.md` | How to run demos |
| `faer/demos/sample_data.py` | Shared data structures |
| `faer/demos/schematic_streamlit.py` | Pure Streamlit demo |
| `faer/demos/schematic_react.py` | React wrapper demo |
| `faer/demos/components/react_schematic/package.json` | React dependencies |
| `faer/demos/components/react_schematic/tsconfig.json` | TypeScript config |
| `faer/demos/components/react_schematic/webpack.config.js` | Build config |
| `faer/demos/components/react_schematic/src/index.tsx` | Component entry |
| `faer/demos/components/react_schematic/src/Schematic.tsx` | Main component |

---

## Implementation Order

1. **Create directory structure**
2. **sample_data.py** - Data contract both demos use
3. **schematic_streamlit.py** - Faster to iterate, do first
4. **React scaffolding** - package.json, tsconfig, webpack
5. **Schematic.tsx** - React component
6. **Build and test React**
7. **Side-by-side evaluation**

---

## Notes for Implementation

- Keep demos COMPLETELY ISOLATED from main app
- Both demos must render IDENTICAL data
- Focus on visual quality and interactivity comparison
- Document any limitations discovered
- Take screenshots for comparison document
- Don't over-engineer - these are evaluation demos
