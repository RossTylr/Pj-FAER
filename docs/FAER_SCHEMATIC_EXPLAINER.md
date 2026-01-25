# FAER-M Schematic Integration Explainer

> A concise guide for Claude Code integration with the hospital visualization system

---

## TL;DR

The schematic system visualizes hospital patient flow using a **Python → JSON → React** pipeline. Departments are **nodes**, patient pathways are **edges**. The system supports two extension patterns:

1. **Scaling** — Departments grow/shrink at runtime (surge beds)
2. **Bolt-on** — New departments attach without breaking existing flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FAER-M Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Simulation          Transform           Render                │
│   ──────────          ─────────           ──────                │
│   SimPy Model    →    SchematicData   →   React/SVG             │
│   (Python)            (dataclass)         (TypeScript)          │
│                                                                 │
│   Produces:           Serializes:         Displays:             │
│   • Occupancy         • Nodes             • Interactive nodes   │
│   • Utilisation       • Edges             • Flow edges          │
│   • Queue times       • Metrics           • Status colors       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Abstractions

### 1. Node

A hospital resource or location (ED, Ward, ITU, etc.)

```python
@dataclass
class NodeState:
    id: str                    # Unique key: "ed_bays", "itu", "ward"
    label: str                 # Display: "ED Bays", "ITU", "Ward"
    node_type: str             # "entry" | "resource" | "process" | "exit"
    capacity: Optional[int]    # Beds/slots (None for entry/exit)
    occupied: int              # Current patients
    utilisation: float         # 0.0–1.0 (computed property)
    status: str                # "normal" | "warning" | "critical"
```

**Design rationale**: Nodes are value objects—immutable snapshots of state at a point in time. Compute derived properties (utilisation, status) rather than storing them.

### 2. Edge

A patient pathway between nodes.

```python
@dataclass
class FlowEdge:
    source: str           # Origin node ID
    target: str           # Destination node ID
    volume_per_hour: float
    is_blocked: bool      # True if destination util > 95%
```

**Design rationale**: Edges are directional. Bidirectional flows (e.g., ED ↔ CT) use two edges. Blocking is a derived state from destination utilisation.

### 3. SchematicData

Complete visualization state for one moment in time.

```python
@dataclass
class SchematicData:
    timestamp: str
    nodes: Dict[str, NodeState]
    edges: List[FlowEdge]
    total_in_system: int
    overall_status: str
```

**Design rationale**: Single source of truth for the React component. Serialize once, render anywhere.

---

## The Layout Model

Hospital flow follows a **crucifix pattern**:

```
                         ┌─────┐
                         │ ITU │
                         └──┬──┘
                            │
┌─────────┐  ┌────────┐  ┌──┴───┐  ┌─────────┐  ┌───────────┐
│ Arrivals├──┤ Triage ├──┤  ED  ├──┤ Theatre ├──┤ Discharge │
└─────────┘  └────────┘  └──┬───┘  └─────────┘  └───────────┘
                            │
                         ┌──┴──┐
                         │Ward │
                         └─────┘
```

**Zones** (Y-axis positioning):
| Zone | Y Range | Contains |
|------|---------|----------|
| Upstream | 200–400 | ITU, Diagnostics |
| Main Lane | 500 | Entry → Triage → ED → Theatre → Exit |
| Downstream | 600–800 | Ward, Step-down, Holding |

**Why crucifix?** Matches real hospital patient flow. Critical care branches up, general care branches down. Main lane represents the emergency pathway.

---

## Extension Pattern 1: Scaling

Departments can **grow** during high pressure and **shrink** when pressure eases.

### The Mechanism

```
┌─────────────────────────────────────────────────────┐
│              DynamicCapacityResource                │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░│              │   │
│   └─────────────────────────────────────────────┘   │
│    ├── baseline: 20 ───┤├ surge ┤├─ available ─┤   │
│                         (active)   (inactive)       │
│                                                     │
│   current_capacity = baseline + surge_active        │
│   max_capacity = baseline + surge_available         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### OPEL Triggers

NHS Operational Pressures Escalation Levels drive automatic scaling:

```python
OPEL_THRESHOLDS = {
    1: 0.70,  # Normal
    2: 0.85,  # Moderate pressure → flow focus
    3: 0.90,  # Severe pressure → activate surge
    4: 0.95,  # Crisis → diversion
}
```

### Integration Point

```python
# In scaling monitor (background SimPy process)
if resource.utilisation > OPEL_THRESHOLDS[3]:
    resource.add_capacity(surge_beds)

# Schematic reflects this
node.surge_active = resource.current_capacity - resource.baseline
```

### Schematic Visualization

```
┌──────────────────────────────────┐
│  ED Bays              [OPEL 3]  │
├──────────────────────────────────┤
│  ████████████████░░░░  24/28    │
│  └─ baseline ─┘└surge┘          │
└──────────────────────────────────┘
```

**Key insight**: The SimPy resource is created at `max_capacity`. Scaling controls which slots are "active". This avoids recreating resources mid-simulation.

---

## Extension Pattern 2: Bolt-On

New departments attach to existing flow without modifying core simulation logic.

### The Pattern

```python
@dataclass
class BoltOnConfig:
    enabled: bool = False      # Master switch
    capacity: int = 10
    # ... department-specific params

# In resource creation
if config.enabled:
    resources.new_dept = simpy.Resource(env, config.capacity)
else:
    resources.new_dept = None

# In patient process
if resources.new_dept is not None:
    # Use the bolt-on
    with resources.new_dept.request() as req:
        yield req
        # ...
```

### Bolt-On Categories

| Category | Example | Attachment Point | Flow Pattern |
|----------|---------|------------------|--------------|
| **Holding** | Discharge Lounge | Before exit | Unidirectional |
| **Step-Down** | HDU | Between ITU & Ward | Unidirectional |
| **Diagnostic** | CT Suite | Parallel to ED | Bidirectional |
| **Satellite** | Urgent Care | Parallel to Entry | Diversion |
| **Overflow** | Winter Ward | Parallel to Ward | Spillover |

### Example: Adding HDU (Step-Down Unit)

**Before:**
```
ITU → Ward
```

**After:**
```
ITU → HDU → Ward
```

**Implementation:**

```python
# 1. Config
@dataclass
class HDUConfig:
    enabled: bool = False
    capacity: int = 4
    from_itu_prob: float = 0.85

# 2. Conditional routing
if scenario.hdu.enabled:
    routing["itu"]["hdu"] = scenario.hdu.from_itu_prob
    routing["itu"]["ward"] = 0.0  # Redirect through HDU
    routing["hdu"]["ward"] = 0.90
else:
    routing["itu"]["ward"] = 0.85  # Original path
```

**Schematic updates automatically** because edges are derived from active routing rules.

---

## Data Flow Deep Dive

### Pre-Simulation (Config Only)

```python
def build_schematic_from_config(session_state) -> SchematicData:
    """Build visualization from UI configuration before running sim"""
    nodes = {}

    # Always include core departments
    nodes["ed_bays"] = NodeState(
        id="ed_bays",
        capacity=session_state.get("n_ed_bays", 20),
        occupied=0,  # No simulation yet
        # ...
    )

    # Conditionally include bolt-ons
    if session_state.get("hdu_enabled", False):
        nodes["hdu"] = NodeState(...)

    return SchematicData(nodes=nodes, edges=derive_edges(nodes))
```

### Post-Simulation (With Results)

```python
def build_schematic_from_results(results, scenario) -> SchematicData:
    """Build visualization from simulation output"""
    nodes = {}

    nodes["ed_bays"] = NodeState(
        id="ed_bays",
        capacity=scenario.n_ed_bays,
        occupied=int(results["mean_occupancy_ed"]),
        throughput_per_hour=results["throughput_ed"] / run_hours,
        mean_wait_mins=results["mean_wait_ed"],
    )

    # Derive blocking from utilisation
    edges = []
    ward_blocked = results["util_ward"] > 0.95
    edges.append(FlowEdge("ed_bays", "ward", volume, is_blocked=ward_blocked))

    return SchematicData(nodes=nodes, edges=edges)
```

### React Rendering

```typescript
// Schematic.tsx
const Schematic: React.FC<{data: SchematicData}> = ({data}) => {
  return (
    <svg viewBox="0 0 1400 1000">
      {/* Render edges first (behind nodes) */}
      {data.edges.map(edge => <Edge key={`${edge.source}-${edge.target}`} {...edge} />)}

      {/* Render nodes */}
      {Object.values(data.nodes).map(node => <Node key={node.id} {...node} />)}
    </svg>
  );
};
```

---

## The Registry Pattern

For large-scale extensibility, define departments in a central registry:

```python
DEPARTMENT_REGISTRY = {
    "ed_bays": DepartmentDef(
        id="ed_bays",
        label="ED Bays",
        category="emergency",
        default_capacity=20,
        supports_scaling=True,
        zone="main",
        targets=["theatre", "ward", "itu", "discharge"],
    ),
    "hdu": DepartmentDef(
        id="hdu",
        label="HDU",
        category="step_down",
        default_capacity=4,
        is_bolt_on=True,
        zone="downstream",
        sources=["itu"],
        targets=["ward"],
    ),
}
```

**Benefits:**
- UI auto-generated from registry
- Schematic positions derived from zone/category
- Routing validation against allowed connections
- Single source of truth for department metadata

---

## Quick Reference

### Status Colors

| Status | Utilisation | Color | Hex |
|--------|-------------|-------|-----|
| Normal | < 70% | Green | `#28a745` |
| Warning | 70–90% | Yellow | `#ffc107` |
| Critical | ≥ 90% | Red | `#dc3545` |

### Node Types

| Type | Has Capacity | Shows | Examples |
|------|--------------|-------|----------|
| `entry` | No | Throughput/hr | Ambulance, Walk-in |
| `resource` | Yes | Occupied/Capacity | ED, Ward, ITU |
| `process` | Yes | Occupied/Capacity | Triage, Handover |
| `exit` | No | Throughput/hr | Discharge |

### File Locations

| Purpose | Path |
|---------|------|
| Python data models | `app/components/react_schematic/data.py` |
| React component | `app/components/react_schematic/src/Schematic.tsx` |
| Core enums | `src/faer/core/entities.py` |
| Scenario config | `src/faer/core/scenario.py` |
| Scaling logic | `src/faer/core/scaling.py` |
| Dynamic resources | `src/faer/model/dynamic_resource.py` |

---

## Integration Checklist

### Adding Surge Capacity

```
□ Define surge beds in CapacityScalingConfig
□ Add OPEL trigger rule
□ Extend NodeState with baseline/surge fields
□ Update React capacity bar renderer
□ Test scale-up and graceful scale-down
```

### Adding Bolt-On Department

```
□ Create XxxConfig dataclass with enabled: bool = False
□ Add to FullScenario
□ Create resource conditionally in model
□ Update routing rules when enabled
□ Add node to schematic builder (conditional)
□ Add position to layout zones
□ Add UI toggle in Streamlit
□ Write tests for enabled/disabled states
```

---

## Design Principles

1. **Immutable snapshots** — SchematicData is a moment in time, not a live view
2. **Derived state** — Compute utilisation/status from raw values, don't store
3. **Conditional composition** — Bolt-ons use `if enabled` guards, not inheritance
4. **Registry-driven** — Department metadata lives in one place
5. **Graceful degradation** — Disabled bolt-ons don't break simulation
6. **Separation of concerns** — Simulation produces data, schematic visualizes it

---

## Common Patterns

### Pattern: Conditional Edge

```python
# Edge only exists if both nodes are active
if source_id in nodes and target_id in nodes:
    edges.append(FlowEdge(source_id, target_id, volume))
```

### Pattern: Graceful Scale-Down

```python
# Don't remove occupied beds immediately
def remove_capacity(self, amount, graceful=True):
    if graceful:
        self._pending_removals += amount  # Remove when freed
    else:
        self._remove_empty_slots(amount)  # Remove now if empty
```

### Pattern: Bidirectional Diagnostic

```python
# Patient goes to CT and returns to ED
edges.append(FlowEdge("ed_bays", "ct_suite", rate))
edges.append(FlowEdge("ct_suite", "ed_bays", rate))  # Return path
```

### Pattern: Diversion Satellite

```python
# Some arrivals bypass main ED
if ucc_enabled and patient.priority in [3, 4]:
    if rng.random() < diversion_rate:
        yield from ucc_pathway(patient)
        return  # Don't enter main ED
```

---

## Testing Strategy

| Layer | Test Focus | Example |
|-------|------------|---------|
| Unit | Data transforms | `NodeState.utilisation` computes correctly |
| Integration | Bolt-on enable/disable | HDU routing changes when toggled |
| Snapshot | Schematic output | JSON matches expected structure |
| Visual | React rendering | Nodes appear in correct positions |

```python
def test_bolt_on_isolation():
    """Disabling bolt-on doesn't affect core simulation"""
    scenario_with = FullScenario(hdu=HDUConfig(enabled=True))
    scenario_without = FullScenario(hdu=HDUConfig(enabled=False))

    # Core metrics should be similar (within stochastic variance)
    results_with = run(scenario_with)
    results_without = run(scenario_without)

    assert abs(results_with["util_ed"] - results_without["util_ed"]) < 0.05
```

---

## Summary

The FAER-M schematic system is built on three ideas:

1. **Nodes and edges** — Simple graph model of hospital flow
2. **Scaling** — Dynamic capacity via OPEL triggers
3. **Bolt-ons** — Modular departments with enable/disable flags

The architecture separates simulation (SimPy) from visualization (React) via a clean data contract (SchematicData). Extensions follow consistent patterns: config dataclass → conditional resource → conditional routing → conditional schematic node.

---

*Version 1.0 | January 2026 | For Claude Code in Cursor*
