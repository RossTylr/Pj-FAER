# Hospital Schematic Integration Instructions for FAER-M

> **For Claude Code in Cursor** - Implementation Guide for Modular Hospital Visualization

## Overview

This document provides instructions for integrating the React-based hospital schematic visualization with FAER-M (Framework for Acute and Emergency Resources - Model). The architecture is designed to be **modular and expandable** - supporting additional beds, departments, and custom hospital configurations.

---

## Current Architecture Summary

### Data Flow Pattern
```
Python Simulation (FAER-M)
        ↓
Builder Functions (data.py)
        ↓
SchematicData (dataclass)
        ↓
to_dict() JSON serialization
        ↓
React Component (Schematic.tsx)
        ↓
SVG Render (Interactive)
```

### Key Files
| File | Purpose |
|------|---------|
| `app/components/react_schematic/data.py` | Python dataclasses & builder functions |
| `app/components/react_schematic/src/Schematic.tsx` | React SVG renderer |
| `src/faer/core/entities.py` | Core enums (NodeType, BedState, etc.) |
| `src/faer/core/scenario.py` | Configuration dataclasses |

---

## Part 1: Adding New Beds (Capacity Expansion)

### Step 1.1: Update Configuration Dataclass

When adding new bed types or expanding capacity, modify the appropriate config in `src/faer/core/scenario.py`:

```
Location: src/faer/core/scenario.py

For existing node types:
- Modify the `capacity` default value in the relevant config dataclass
- Example: ITUConfig.capacity, WardConfig.capacity

For NEW bed types within existing departments:
- Add a new field to the config dataclass
- Example: Add `high_dependency_beds: int = 4` to ITUConfig
```

### Step 1.2: Session State Integration

Ensure Streamlit persists the new capacity:

```
Location: app/pages/2_Resources.py

Pattern to follow:
1. Add number_input widget for new bed type
2. Store in st.session_state with descriptive key
3. Pass to scenario builder
```

### Step 1.3: Schematic Data Builder

Update the builder to include new capacity:

```
Location: app/components/react_schematic/data.py

In build_schematic_from_config():
1. Read new capacity from session_state
2. Create NodeState with new capacity
3. Add to nodes dict

In build_schematic_from_results():
1. Calculate utilisation from simulation results
2. Map result keys to node IDs
```

---

## Part 2: Adding New Departments (Horizontal Expansion)

### Step 2.1: Define Department Type

Add new NodeType enum value:

```
Location: src/faer/core/entities.py

Add to NodeType enum:
- Choose next available integer value
- Use descriptive name (e.g., RADIOLOGY, PAEDIATRIC_ED, BURNS_UNIT)
```

### Step 2.2: Create Department Config

Add new configuration dataclass:

```
Location: src/faer/core/scenario.py

Pattern to follow:
@dataclass
class NewDeptConfig:
    capacity: int = X
    los_mean_hours: float = Y
    los_cv: float = Z
    enabled: bool = True
    # Add routing probabilities to other departments
    to_ward_prob: float = 0.0
    to_itu_prob: float = 0.0
```

### Step 2.3: Update FullScenario

Add config reference to FullScenario:

```
Location: src/faer/core/scenario.py

In FullScenario dataclass:
1. Add field: new_dept: NewDeptConfig = field(default_factory=NewDeptConfig)
2. Add to routing matrix generation
```

### Step 2.4: React Schematic - Add Node Position

Define visual placement:

```
Location: app/components/react_schematic/src/Schematic.tsx

In NODE_POSITIONS object, add coordinates:
- Consider the crucifix layout pattern
- Horizontal flow: Entry → Triage → ED → Theatre → Exit
- Vertical branches: ITU (above), Ward (below)

Placement guidelines:
- Main lane Y = 500 (horizontal center)
- Above lane Y < 500 (e.g., ITU at 230)
- Below lane Y > 500 (e.g., Ward at 770)
- Spacing X ~ 270 between nodes
```

### Step 2.5: React Schematic - Edge Routing

Add connection paths:

```
Location: app/components/react_schematic/src/Schematic.tsx

In renderEdge() function:
1. Add case for new source/target combinations
2. Choose path type:
   - Straight horizontal: same Y coordinate
   - Curved entry: from entry nodes (X=120)
   - Vertical: crucifix arm connections
```

### Step 2.6: Python Data Builder - Department Node

Add node creation:

```
Location: app/components/react_schematic/data.py

In build_schematic_from_config():
    if new_dept_enabled:
        nodes["new_dept"] = NodeState(
            id="new_dept",
            label="New Department",
            node_type="resource",
            capacity=new_dept_capacity,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0
        )

In build_schematic_from_results():
    # Map simulation results to node state
    # Calculate utilisation from results['util_new_dept']
```

### Step 2.7: Add Flow Edges

Define connections in data builder:

```
Location: app/components/react_schematic/data.py

Add FlowEdge objects:
edges.append(FlowEdge(
    source="source_node_id",
    target="new_dept",
    volume_per_hour=calculated_flow,
    is_blocked=util_new_dept > 0.95
))
```

---

## Part 3: Modular Department Registry Pattern

For maximum flexibility, implement a registry pattern:

### Step 3.1: Department Registry

Create a central department definition:

```
Suggested location: src/faer/core/departments.py

Pattern:
@dataclass
class DepartmentDefinition:
    id: str                    # Unique identifier
    label: str                 # Display name
    node_type: str             # "resource" | "process" | "entry" | "exit"
    category: str              # "emergency" | "downstream" | "diagnostics"
    default_capacity: int
    position: Tuple[int, int]  # (x, y) for schematic
    color: str                 # Status color override (optional)

DEPARTMENT_REGISTRY: Dict[str, DepartmentDefinition] = {
    "ed_bays": DepartmentDefinition(
        id="ed_bays",
        label="ED Bays",
        node_type="resource",
        category="emergency",
        default_capacity=12,
        position=(660, 500),
        color=None
    ),
    # ... more departments
}
```

### Step 3.2: Dynamic Node Generation

React component reads from registry:

```
Location: app/components/react_schematic/src/Schematic.tsx

Instead of hardcoded NODE_POSITIONS:
- Receive positions from Python via props
- Iterate over nodes object to render
- Use node.category for styling decisions
```

### Step 3.3: Routing Matrix

Define allowed transitions:

```
Location: src/faer/core/scenario.py

Pattern:
@dataclass
class RoutingRule:
    source: str
    target: str
    probability: float
    priority_filter: Optional[List[int]] = None  # P1-P4 filter

ROUTING_MATRIX: List[RoutingRule] = [
    RoutingRule("triage", "ed_bays", 1.0),
    RoutingRule("ed_bays", "ward", 0.40),
    RoutingRule("ed_bays", "itu", 0.10),
    # ...
]
```

---

## Part 4: Visualization Customization

### Step 4.1: Status Thresholds

Make thresholds configurable:

```
Location: app/components/react_schematic/data.py

Instead of hardcoded 0.70/0.90:

@dataclass
class StatusThresholds:
    warning: float = 0.70
    critical: float = 0.90
    blocked: float = 0.95
```

### Step 4.2: Color Schemes

Support multiple themes:

```
Location: app/components/react_schematic/src/Schematic.tsx

const THEMES = {
    default: {
        normal: "#28a745",
        warning: "#ffc107",
        critical: "#dc3545",
        entry: "#1976d2",
        exit: "#7b1fa2"
    },
    accessible: {
        // High contrast colors
    }
};
```

### Step 4.3: Layout Algorithms

For auto-positioning new departments:

```
Layout Strategy Options:

1. Fixed Grid: Predefined positions in registry
2. Force-Directed: Auto-arrange based on connections
3. Layered: Organize by category (entry → process → downstream → exit)
4. Custom: User-defined coordinates in config

Recommended: Start with Fixed Grid, migrate to Layered as departments grow
```

---

## Part 5: Integration Checklist

When adding a new department, complete these steps:

### Python Side
- [ ] Add NodeType enum value in `entities.py`
- [ ] Create Config dataclass in `scenario.py`
- [ ] Add to FullScenario dataclass
- [ ] Update routing rules
- [ ] Add to ResultsCollector metrics
- [ ] Update `build_schematic_from_config()`
- [ ] Update `build_schematic_from_results()`
- [ ] Add FlowEdge connections
- [ ] Update Streamlit Resources page

### React Side
- [ ] Add position to NODE_POSITIONS
- [ ] Add edge routing cases
- [ ] Update legend if new node type
- [ ] Test click interaction
- [ ] Verify responsive scaling

### Testing
- [ ] Unit test for new config validation
- [ ] Integration test for simulation flow
- [ ] Visual regression test for schematic
- [ ] Performance test with increased nodes

---

## Part 6: Code Patterns to Follow

### Pattern 1: Node State Creation
```python
NodeState(
    id="unique_id",           # matches React NODE_POSITIONS key
    label="Display Name",     # shown on schematic
    node_type="resource",     # determines rendering style
    capacity=10,              # None for entry/exit nodes
    occupied=5,               # current patients
    throughput_per_hour=2.5,  # for entry/exit display
    mean_wait_mins=15.0       # shown below capacity
)
```

### Pattern 2: Flow Edge Creation
```python
FlowEdge(
    source="source_node_id",
    target="target_node_id",
    volume_per_hour=3.5,      # shown as label if >= 2.0
    is_blocked=False          # red dashed if True
)
```

### Pattern 3: Utilisation Calculation
```python
@property
def utilisation(self) -> float:
    if self.capacity is None or self.capacity == 0:
        return 0.0
    return min(1.0, self.occupied / self.capacity)
```

### Pattern 4: Status Determination
```python
@property
def status(self) -> str:
    if self.utilisation >= 0.90:
        return "critical"
    elif self.utilisation >= 0.70:
        return "warning"
    return "normal"
```

---

## Part 7: Common Pitfalls to Avoid

1. **Mismatched IDs**: Node `id` in Python must match key in React `NODE_POSITIONS`

2. **Missing Edges**: Every node needs at least one incoming or outgoing edge

3. **Overlapping Positions**: Check X/Y coordinates don't collide

4. **Capacity vs Throughput**:
   - Resource nodes: use `capacity` (integer)
   - Entry/Exit nodes: use `throughput_per_hour` (float), set `capacity=None`

5. **Session State Keys**: Use consistent naming: `st.session_state.n_<dept>_beds`

6. **TypeScript Types**: Update `NodeData` interface if adding new fields

7. **Routing Loops**: Ensure routing probabilities sum to ≤ 1.0

8. **Build Step**: After React changes, run `npm run build` in `app/components/react_schematic/`

---

## Part 8: Future Expansion Ideas

### 8.1 Multi-Building Support
- Add `building: str` field to NodeState
- Group nodes by building in layout
- Add building selector in UI

### 8.2 Time-Based Visualization
- Animate schematic over simulation time
- Slider to scrub through timeline
- Heatmap overlay for historical utilisation

### 8.3 Scenario Comparison
- Side-by-side schematics (use MiniSchematic)
- Diff view highlighting changed capacities
- A/B testing visualization

### 8.4 Real-Time Integration
- WebSocket for live updates
- Connect to hospital information systems
- Dashboard mode with auto-refresh

### 8.5 Configuration Import/Export
- JSON schema for department definitions
- Import hospital configs from file
- Share configurations between instances

---

## Quick Reference: File Modifications by Task

| Task | Files to Modify |
|------|----------------|
| Add beds to existing dept | `scenario.py`, `2_Resources.py`, `data.py` |
| Add new department | `entities.py`, `scenario.py`, `data.py`, `Schematic.tsx`, `2_Resources.py` |
| Change routing | `scenario.py`, `data.py` (edges) |
| Change layout | `Schematic.tsx` (NODE_POSITIONS) |
| Change colors | `Schematic.tsx` (getStatusColor, CSS) |
| Add metrics | `entities.py`, `collector.py`, `data.py` |

---

## Commands Reference

```bash
# Install dependencies
pip install -e ".[dev]"
cd app/components/react_schematic && npm install

# Build React component
cd app/components/react_schematic && npm run build

# Run Streamlit app
streamlit run app/Home.py

# Run tests
pytest tests/ -v

# Type checking (if configured)
mypy src/faer/
```

---

*Document Version: 1.0*
*Last Updated: January 2026*
*For: Claude Code in Cursor - FAER-M Integration*
