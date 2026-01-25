# Hospital Schematic Integration Instructions for FAER-M

> **For Claude Code in Cursor** - Implementation Guide for Modular Hospital Visualization

## Overview

This document provides instructions for integrating the React-based hospital schematic visualization with FAER-M (Framework for Acute and Emergency Resources - Model). The architecture is designed to be **modular and expandable** - supporting:

- **Dynamic capacity scaling** - Departments that grow/shrink at runtime
- **Bolt-on modules** - New departments added without disrupting existing flow
- **OPEL-triggered surge** - Automatic capacity activation based on pressure

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

| Layer | File | Purpose |
|-------|------|---------|
| **Data Models** | `app/components/react_schematic/data.py` | Python dataclasses & builder functions |
| **React UI** | `app/components/react_schematic/src/Schematic.tsx` | SVG renderer |
| **Core Enums** | `src/faer/core/entities.py` | NodeType, BedState, etc. |
| **Config** | `src/faer/core/scenario.py` | Department configuration dataclasses |
| **Scaling** | `src/faer/core/scaling.py` | OPEL levels, triggers, actions |
| **Dynamic Resources** | `src/faer/model/dynamic_resource.py` | Runtime capacity changes |
| **Scaling Monitor** | `src/faer/model/scaling_monitor.py` | Background trigger evaluation |

---

## Part 1: Department Size Scaling (Vertical Growth)

Departments can **increase in size** dynamically during simulation through the `DynamicCapacityResource` pattern.

### 1.1 Understanding the Dynamic Resource Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DynamicCapacityResource                   │
├─────────────────────────────────────────────────────────────┤
│  initial_capacity: 20    ←── Baseline beds (always active)  │
│  max_capacity: 30        ←── Ceiling (SimPy resource size)  │
│  current_capacity: 24    ←── Active slots (runtime value)   │
│  surge_beds: 10          ←── Available to activate          │
├─────────────────────────────────────────────────────────────┤
│  add_capacity(4)         →   current: 20 → 24               │
│  remove_capacity(2)      →   current: 24 → 22               │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: SimPy resource is created at `max_capacity`, but only `current_capacity` slots are active. Scaling adds/removes from the active pool.

### 1.2 Configuration for Scalable Departments

```
Location: src/faer/core/scaling.py

@dataclass
class CapacityScalingConfig:
    enabled: bool = False

    # Baseline capacities (always available)
    baseline_ed_bays: int = 20
    baseline_ward_beds: int = 30
    baseline_itu_beds: int = 6

    # Surge capacities (activated on demand)
    opel_3_ed_surge_beds: int = 4
    opel_3_ward_surge_beds: int = 8
    opel_3_itu_surge_beds: int = 2

    # Maximum ceiling
    max_ed_bays: int = 28
    max_ward_beds: int = 45
    max_itu_beds: int = 10
```

### 1.3 OPEL-Triggered Scaling

The NHS OPEL (Operational Pressures Escalation Levels) framework drives automatic scaling:

```
OPEL Level │ Threshold │ Actions
───────────┼───────────┼──────────────────────────────────────
OPEL 1     │ < 70%     │ Normal operations
OPEL 2     │ 70-85%    │ Flow focus, discharge push
OPEL 3     │ 85-95%    │ Surge capacity activated
OPEL 4     │ > 95%     │ Full escalation, diversions
```

**Schematic visualization must show**:
- Current OPEL level badge
- Surge beds as distinct visual element
- Capacity bar showing baseline vs surge

### 1.4 Implementing Scalable Node in Schematic

**Step 1: Extend NodeState for scaling**

```
Location: app/components/react_schematic/data.py

@dataclass
class NodeState:
    id: str
    label: str
    node_type: str
    capacity: Optional[int]           # Current active capacity
    baseline_capacity: Optional[int]  # NEW: Always-on beds
    max_capacity: Optional[int]       # NEW: Ceiling
    surge_active: int = 0             # NEW: Currently activated surge beds
    occupied: int
    throughput_per_hour: float
    mean_wait_mins: float

    @property
    def utilisation(self) -> float:
        if self.capacity is None or self.capacity == 0:
            return 0.0
        return min(1.0, self.occupied / self.capacity)

    @property
    def surge_available(self) -> int:
        """Beds that could still be activated"""
        if self.max_capacity and self.capacity:
            return self.max_capacity - self.capacity
        return 0
```

**Step 2: React component renders scaling state**

```
Location: app/components/react_schematic/src/Schematic.tsx

// Extended interface
interface NodeData {
  id: string;
  label: string;
  capacity: number | null;
  baseline_capacity: number | null;  // NEW
  max_capacity: number | null;       // NEW
  surge_active: number;              // NEW
  occupied: number;
  // ... existing fields
}

// Render capacity bar with surge indicator
const renderCapacityBar = (node: NodeData) => {
  const baselineWidth = (node.baseline_capacity / node.max_capacity) * barWidth;
  const surgeWidth = (node.surge_active / node.max_capacity) * barWidth;

  return (
    <g>
      {/* Baseline capacity (solid) */}
      <rect fill="#28a745" width={baselineWidth} />

      {/* Surge capacity (hatched pattern) */}
      <rect fill="url(#surgePattern)" x={baselineWidth} width={surgeWidth} />

      {/* Available surge (ghost) */}
      <rect fill="#e0e0e0" x={baselineWidth + surgeWidth}
            width={barWidth - baselineWidth - surgeWidth} />
    </g>
  );
};
```

**Step 3: Visual indicator for surge status**

```
Schematic Display Pattern:

┌──────────────────────────────────┐
│  ED Bays            [OPEL 3]    │  ← OPEL badge top-right
├──────────────────────────────────┤
│  ████████████░░░░  24/28        │  ← Baseline + Surge / Max
│  ├── 20 ──┤├ +4 ┤               │  ← Visual breakdown
│   baseline  surge               │
├──────────────────────────────────┤
│  Wait: 12 min                   │
└──────────────────────────────────┘
```

### 1.5 Scale-Up Trigger Integration

```
Location: src/faer/core/scaling.py

@dataclass
class ScalingTrigger:
    trigger_type: TriggerType
    resource_id: str
    threshold: float
    sustain_minutes: float = 5.0    # Must hold for 5 min before firing

@dataclass
class ScalingAction:
    action_type: ActionType         # ADD_CAPACITY, REMOVE_CAPACITY, etc.
    target_resource: str
    amount: int

@dataclass
class ScalingRule:
    name: str
    trigger: ScalingTrigger
    action: ScalingAction
    cooldown_minutes: float = 15.0  # Prevent oscillation
    max_activations: int = 3        # Limit per simulation
```

**Example: Auto-generate OPEL 3 surge rule**

```python
def create_opel_rules(config: OPELConfig) -> List[ScalingRule]:
    rules = []

    # ED surge at OPEL 3
    if config.opel_3_ed_surge_beds > 0:
        rules.append(ScalingRule(
            name="OPEL3_ED_Surge",
            trigger=ScalingTrigger(
                trigger_type=TriggerType.UTILIZATION_ABOVE,
                resource_id="ed_bays",
                threshold=0.90,
                sustain_minutes=5.0
            ),
            action=ScalingAction(
                action_type=ActionType.ADD_CAPACITY,
                target_resource="ed_bays",
                amount=config.opel_3_ed_surge_beds
            ),
            cooldown_minutes=30.0
        ))

    return rules
```

### 1.6 Graceful Scale-Down

When pressure eases, capacity reduces **gracefully** (waits for beds to empty):

```
Location: src/faer/model/dynamic_resource.py

def remove_capacity(self, amount: int, graceful: bool = True) -> int:
    """
    Remove capacity from the resource.

    Args:
        amount: Number of slots to remove
        graceful: If True, wait for slots to be released before removing
                  If False, only remove currently empty slots
    """
    if graceful:
        # Mark slots for deactivation when patient leaves
        self._pending_removals += amount
    else:
        # Only remove empty slots immediately
        removable = self.current_capacity - self.occupied
        actual_remove = min(amount, removable)
        self.current_capacity -= actual_remove
        return actual_remove
```

**Schematic should show pending removals**:
```
┌──────────────────────────────────┐
│  Ward Beds                       │
├──────────────────────────────────┤
│  ████████████████░░  38/45      │
│       ↓ scaling down to 30      │  ← Indicator for pending
└──────────────────────────────────┘
```

---

## Part 2: Bolt-On Departments (Horizontal Extension)

New departments can be **attached modularly** without rewriting core simulation logic.

### 2.1 Bolt-On Architecture Pattern

```
                    ┌─────────────┐
                    │     ITU     │
                    └──────┬──────┘
                           │
┌─────────┐  ┌─────────┐  ┌┴────────┐  ┌─────────┐  ┌──────────┐
│ Arrivals├──┤ Triage  ├──┤ ED Bays ├──┤ Theatre ├──┤ Discharge│
└─────────┘  └─────────┘  └┬────────┘  └─────────┘  └────┬─────┘
                           │                              │
                    ┌──────┴──────┐              ┌────────┴───────┐
                    │    Ward     │              │ Discharge      │
                    └─────────────┘              │ Lounge [BOLT]  │
                                                 └────────────────┘
```

**Bolt-on characteristics**:
- Connects to existing nodes via edges
- Has own configuration dataclass
- Can be enabled/disabled independently
- Doesn't break simulation if disabled

### 2.2 Bolt-On Department Types

| Type | Example | Connection Point | Purpose |
|------|---------|------------------|---------|
| **Holding Area** | Discharge Lounge | Exit pathway | Decouple bed from discharge |
| **Diagnostic Pod** | CT Suite, MRI | ED Bays (parallel) | Specialist investigation |
| **Satellite Unit** | Urgent Care Centre | Arrivals (parallel) | Offload minor cases |
| **Step-Down Unit** | HDU | Between ITU & Ward | Intermediate care level |
| **Overflow Wing** | Winter Ward | Ward (parallel) | Surge capacity pod |

### 2.3 Implementing a Bolt-On: Discharge Lounge Example

**Step 1: Configuration dataclass**

```
Location: src/faer/core/scenario.py

@dataclass
class DischargeLoungeConfig:
    enabled: bool = False
    capacity: int = 10
    max_wait_mins: float = 120.0
    position: Tuple[int, int] = (1200, 700)  # Schematic coordinates
```

**Step 2: Add to FullScenario**

```python
@dataclass
class FullScenario:
    # ... existing configs
    discharge_lounge: DischargeLoungeConfig = field(
        default_factory=DischargeLoungeConfig
    )
```

**Step 3: SimPy resource creation (conditional)**

```
Location: src/faer/model/full_model.py

def create_resources(env, scenario):
    resources = AEResources(...)

    # Bolt-on: Discharge Lounge
    if scenario.discharge_lounge.enabled:
        resources.discharge_lounge = simpy.Resource(
            env,
            capacity=scenario.discharge_lounge.capacity
        )
    else:
        resources.discharge_lounge = None

    return resources
```

**Step 4: Patient routing (conditional)**

```python
def patient_discharge_process(env, patient, resources, scenario):
    # Standard pathway: leave ward/ITU bed
    yield release_bed(patient.current_bed)

    # Bolt-on pathway: use lounge if enabled
    if resources.discharge_lounge is not None:
        with resources.discharge_lounge.request() as req:
            yield req
            # Wait for transport/paperwork
            yield env.timeout(sample_lounge_time())

    # Depart system
    record_departure(patient)
```

**Step 5: Schematic data builder (conditional node)**

```
Location: app/components/react_schematic/data.py

def build_schematic_from_config(session_state) -> SchematicData:
    nodes = {}
    edges = []

    # ... existing nodes

    # Bolt-on: Discharge Lounge
    if session_state.get("discharge_lounge_enabled", False):
        nodes["discharge_lounge"] = NodeState(
            id="discharge_lounge",
            label="Discharge Lounge",
            node_type="process",
            capacity=session_state.get("discharge_lounge_capacity", 10),
            baseline_capacity=session_state.get("discharge_lounge_capacity", 10),
            max_capacity=session_state.get("discharge_lounge_capacity", 10),
            surge_active=0,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0
        )

        # Edge from Ward to Lounge
        edges.append(FlowEdge(
            source="ward",
            target="discharge_lounge",
            volume_per_hour=0.0,
            is_blocked=False
        ))

        # Edge from Lounge to Discharge
        edges.append(FlowEdge(
            source="discharge_lounge",
            target="discharge",
            volume_per_hour=0.0,
            is_blocked=False
        ))
```

**Step 6: React schematic (dynamic positioning)**

```
Location: app/components/react_schematic/src/Schematic.tsx

// Read positions from data, not hardcoded
const getNodePosition = (nodeId: string, nodeData: NodeData) => {
  // First check if position provided in data
  if (nodeData.position) {
    return nodeData.position;
  }

  // Fall back to defaults for known nodes
  const defaults: Record<string, {x: number, y: number}> = {
    ed_bays: { x: 660, y: 500 },
    ward: { x: 930, y: 770 },
    // ... etc
  };

  return defaults[nodeId] || { x: 100, y: 100 };
};
```

### 2.4 Bolt-On: Step-Down Unit (HDU)

Pattern for intermediate care between ITU and Ward:

```
                    ┌─────────────┐
                    │     ITU     │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │    HDU      │  ← NEW BOLT-ON
                    │ (Step-Down) │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │    Ward     │
                    └─────────────┘
```

**Configuration**:

```python
@dataclass
class HDUConfig:
    enabled: bool = False
    capacity: int = 4
    los_mean_hours: float = 24.0
    los_cv: float = 0.6
    from_itu_prob: float = 0.85      # ITU patients stepping down
    direct_to_ward_prob: float = 0.90 # HDU patients going to ward
    direct_discharge_prob: float = 0.10
    position: Tuple[int, int] = (930, 500)  # Between ITU and Ward
```

**Routing modification**:

```python
# Before bolt-on:
RoutingRule("itu", "ward", probability=0.85)
RoutingRule("itu", "discharge", probability=0.05)

# After bolt-on (when HDU enabled):
RoutingRule("itu", "hdu", probability=0.85)      # Step-down to HDU
RoutingRule("itu", "discharge", probability=0.05)
RoutingRule("hdu", "ward", probability=0.90)     # HDU to Ward
RoutingRule("hdu", "discharge", probability=0.10)
```

### 2.5 Bolt-On: Diagnostic Pod

Parallel pathway for specialist investigations:

```
                              ┌─────────────┐
                         ┌────┤  CT Suite   │
                         │    └──────┬──────┘
┌─────────┐  ┌─────────┐ │           │
│ Triage  ├──┤ ED Bays ├─┼───────────┼────► Continue flow
└─────────┘  └─────────┘ │           │
                         │    ┌──────┴──────┐
                         └────┤   X-Ray     │
                              └─────────────┘
```

**Key difference**: Patients visit diagnostic pod and **return** to ED bay (not a one-way flow).

```python
@dataclass
class DiagnosticPodConfig:
    enabled: bool = False
    capacity: int = 2
    scan_time_mean: float = 15.0  # minutes
    scan_time_cv: float = 0.3
    position: Tuple[int, int] = (660, 300)  # Above ED

    # Usage probability by priority
    usage_by_priority: Dict[int, float] = field(default_factory=lambda: {
        1: 0.80,  # P1 patients 80% need CT
        2: 0.50,
        3: 0.20,
        4: 0.05,
    })
```

**Schematic edge pattern** (bidirectional):

```python
edges.append(FlowEdge("ed_bays", "ct_suite", volume_per_hour=2.0))
edges.append(FlowEdge("ct_suite", "ed_bays", volume_per_hour=2.0))  # Return path
```

### 2.6 Bolt-On: Satellite Urgent Care Centre

Arrival-parallel pathway to offload minor cases:

```
┌─────────────┐
│  Ambulance  ├──┐
└─────────────┘  │
                 │    ┌─────────┐    ┌─────────────┐
┌─────────────┐  ├────┤ Triage  ├────┤   ED Bays   │
│   Walk-in   ├──┤    └─────────┘    └─────────────┘
└─────────────┘  │
                 │    ┌─────────────────────────┐
                 └────┤  Urgent Care Centre    │──► Minor discharge
                      │  [BOLT-ON SATELLITE]   │
                      └─────────────────────────┘
```

**Diversion logic**:

```python
@dataclass
class UCCConfig:
    enabled: bool = False
    capacity: int = 8
    assessment_time_mean: float = 15.0

    # Diversion criteria
    accept_priorities: List[int] = field(default_factory=lambda: [3, 4])
    accept_arrival_modes: List[str] = field(default_factory=lambda: ["walkin"])
    diversion_rate: float = 0.30  # 30% of eligible patients diverted
```

---

## Part 3: Schematic Layout Expansion Strategies

As departments grow and bolt-on, the schematic must adapt.

### 3.1 Layout Zones

```
┌─────────────────────────────────────────────────────────────────────┐
│                           UPSTREAM ZONE                             │
│                    (Diagnostics, Specialty Consults)                │
│                           Y: 200-400                                │
├─────────────────────────────────────────────────────────────────────┤
│                           MAIN LANE                                 │
│              Entry → Triage → ED → Theatre → Exit                   │
│                           Y: 500                                    │
├─────────────────────────────────────────────────────────────────────┤
│                         DOWNSTREAM ZONE                             │
│                    (Ward, Step-Down, Discharge)                     │
│                           Y: 600-800                                │
├─────────────────────────────────────────────────────────────────────┤
│                          SATELLITE ZONE                             │
│                    (UCC, Overflow, External)                        │
│                           Y: 900+                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Dynamic Position Calculation

```
Location: app/components/react_schematic/data.py

def calculate_node_positions(nodes: Dict[str, NodeState]) -> Dict[str, Tuple[int, int]]:
    """Auto-calculate positions based on node categories"""

    positions = {}

    # Group nodes by category
    entry_nodes = [n for n in nodes.values() if n.node_type == "entry"]
    process_nodes = [n for n in nodes.values() if n.node_type == "process"]
    resource_nodes = [n for n in nodes.values() if n.node_type == "resource"]
    exit_nodes = [n for n in nodes.values() if n.node_type == "exit"]

    # Entry column (X = 120)
    for i, node in enumerate(entry_nodes):
        positions[node.id] = (120, 380 + i * 120)

    # Main lane resources (X increments of 270)
    main_lane = ["triage", "ed_bays", "theatre"]
    for i, node_id in enumerate(main_lane):
        if node_id in nodes:
            positions[node_id] = (390 + i * 270, 500)

    # Upstream (Y = 230)
    upstream = ["itu", "ct_suite", "xray"]
    for i, node_id in enumerate(upstream):
        if node_id in nodes:
            positions[node_id] = (660 + i * 200, 230)

    # Downstream (Y = 770)
    downstream = ["ward", "hdu", "discharge_lounge"]
    for i, node_id in enumerate(downstream):
        if node_id in nodes:
            positions[node_id] = (660 + i * 200, 770)

    # Exit (X = 1200)
    positions["discharge"] = (1200, 500)

    return positions
```

### 3.3 Viewport Scaling

As nodes increase, expand the viewBox:

```typescript
// Schematic.tsx
const calculateViewBox = (nodeCount: number): string => {
  const baseWidth = 1400;
  const baseHeight = 1000;

  // Expand for additional nodes
  const extraWidth = Math.max(0, (nodeCount - 10) * 150);
  const extraHeight = Math.max(0, (nodeCount - 10) * 100);

  return `0 0 ${baseWidth + extraWidth} ${baseHeight + extraHeight}`;
};
```

---

## Part 4: Modular Department Registry

For maximum flexibility, maintain a central registry of all possible departments.

### 4.1 Registry Definition

```
Location: src/faer/core/departments.py (NEW FILE)

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class DepartmentCategory(Enum):
    ENTRY = "entry"
    TRIAGE = "triage"
    EMERGENCY = "emergency"
    DIAGNOSTIC = "diagnostic"
    THEATRE = "theatre"
    CRITICAL_CARE = "critical_care"
    STEP_DOWN = "step_down"
    WARD = "ward"
    HOLDING = "holding"
    EXIT = "exit"
    SATELLITE = "satellite"

@dataclass
class DepartmentDefinition:
    id: str
    label: str
    category: DepartmentCategory
    node_type: str  # "entry" | "resource" | "process" | "exit"

    # Capacity settings
    default_capacity: int
    min_capacity: int = 1
    max_capacity: Optional[int] = None
    supports_scaling: bool = False

    # Position hints
    zone: str = "main"  # "upstream" | "main" | "downstream" | "satellite"
    lane_order: int = 0  # Ordering within zone

    # Routing
    default_sources: List[str] = field(default_factory=list)
    default_targets: List[str] = field(default_factory=list)
    is_bidirectional: bool = False  # For diagnostics

    # Feature flags
    is_bolt_on: bool = False
    requires_config: bool = True

# Central Registry
DEPARTMENT_REGISTRY: Dict[str, DepartmentDefinition] = {
    # Entry points
    "ambulance": DepartmentDefinition(
        id="ambulance", label="Ambulance", category=DepartmentCategory.ENTRY,
        node_type="entry", default_capacity=0, zone="entry", lane_order=0,
        default_targets=["triage"], is_bolt_on=False, requires_config=False
    ),

    # Core departments
    "ed_bays": DepartmentDefinition(
        id="ed_bays", label="ED Bays", category=DepartmentCategory.EMERGENCY,
        node_type="resource", default_capacity=20, min_capacity=8, max_capacity=40,
        supports_scaling=True, zone="main", lane_order=2,
        default_sources=["triage"], default_targets=["theatre", "ward", "itu", "discharge"]
    ),

    # Bolt-ons
    "discharge_lounge": DepartmentDefinition(
        id="discharge_lounge", label="Discharge Lounge",
        category=DepartmentCategory.HOLDING, node_type="process",
        default_capacity=10, zone="downstream", lane_order=3,
        default_sources=["ward", "itu"], default_targets=["discharge"],
        is_bolt_on=True
    ),

    "hdu": DepartmentDefinition(
        id="hdu", label="HDU (Step-Down)", category=DepartmentCategory.STEP_DOWN,
        node_type="resource", default_capacity=4, max_capacity=8,
        supports_scaling=True, zone="downstream", lane_order=1,
        default_sources=["itu"], default_targets=["ward", "discharge"],
        is_bolt_on=True
    ),

    "ct_suite": DepartmentDefinition(
        id="ct_suite", label="CT Suite", category=DepartmentCategory.DIAGNOSTIC,
        node_type="resource", default_capacity=2, zone="upstream", lane_order=0,
        default_sources=["ed_bays"], default_targets=["ed_bays"],
        is_bidirectional=True, is_bolt_on=True
    ),

    # ... more departments
}
```

### 4.2 Registry-Driven UI Generation

```
Location: app/pages/2_Resources.py

from faer.core.departments import DEPARTMENT_REGISTRY, DepartmentCategory

def render_department_config():
    st.header("Department Configuration")

    # Group by category
    categories = {}
    for dept_id, dept in DEPARTMENT_REGISTRY.items():
        cat = dept.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(dept)

    # Render each category
    for cat_name, depts in categories.items():
        with st.expander(f"{cat_name.replace('_', ' ').title()}", expanded=True):
            for dept in depts:
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{dept.label}**")

                with col2:
                    if dept.is_bolt_on:
                        enabled = st.checkbox(
                            "Enable",
                            key=f"{dept.id}_enabled",
                            value=st.session_state.get(f"{dept.id}_enabled", False)
                        )
                    else:
                        enabled = True

                with col3:
                    if enabled and dept.default_capacity > 0:
                        capacity = st.number_input(
                            "Capacity",
                            min_value=dept.min_capacity,
                            max_value=dept.max_capacity or 100,
                            value=st.session_state.get(f"n_{dept.id}", dept.default_capacity),
                            key=f"n_{dept.id}"
                        )
```

### 4.3 Registry-Driven Schematic Building

```python
def build_schematic_from_registry(
    session_state,
    results: Optional[Dict] = None
) -> SchematicData:
    nodes = {}
    edges = []

    # Build nodes from registry
    for dept_id, dept in DEPARTMENT_REGISTRY.items():
        # Skip disabled bolt-ons
        if dept.is_bolt_on:
            if not session_state.get(f"{dept_id}_enabled", False):
                continue

        # Get capacity (from session_state or default)
        capacity = session_state.get(f"n_{dept_id}", dept.default_capacity)
        if dept.node_type == "entry" or dept.node_type == "exit":
            capacity = None

        # Get occupancy from results if available
        occupied = 0
        if results and f"occupied_{dept_id}" in results:
            occupied = results[f"occupied_{dept_id}"]

        nodes[dept_id] = NodeState(
            id=dept_id,
            label=dept.label,
            node_type=dept.node_type,
            capacity=capacity,
            occupied=occupied,
            # ... etc
        )

    # Build edges from registry
    for dept_id, dept in DEPARTMENT_REGISTRY.items():
        if dept_id not in nodes:
            continue

        for target_id in dept.default_targets:
            if target_id in nodes:
                edges.append(FlowEdge(
                    source=dept_id,
                    target=target_id,
                    volume_per_hour=0.0,
                    is_blocked=False
                ))

    return SchematicData(
        timestamp=datetime.now().isoformat(),
        nodes=nodes,
        edges=edges,
        total_in_system=sum(n.occupied for n in nodes.values()),
        total_throughput_24h=0,
        overall_status="normal"
    )
```

---

## Part 5: Integration Checklist

### Adding Surge Capacity to Existing Department

- [ ] Update `CapacityScalingConfig` with surge bed count
- [ ] Add to OPEL rule generation in `create_opel_rules()`
- [ ] Extend `NodeState` with `baseline_capacity`, `max_capacity`, `surge_active`
- [ ] Update React `renderCapacityBar()` for surge visualization
- [ ] Update `build_schematic_from_results()` to include scaling state
- [ ] Add scaling metrics to results collector
- [ ] Test graceful scale-down behavior

### Adding Bolt-On Department

- [ ] Create config dataclass (e.g., `HDUConfig`)
- [ ] Add to `FullScenario` with `enabled: bool = False`
- [ ] Add to `DEPARTMENT_REGISTRY`
- [ ] Create SimPy resource conditionally in `create_resources()`
- [ ] Add routing rules (conditional on enabled)
- [ ] Update `build_schematic_from_config()` with conditional node
- [ ] Add position to zone layout calculation
- [ ] Update React edge routing for new connections
- [ ] Add enable/disable toggle in Streamlit UI
- [ ] Write tests for enabled and disabled states

---

## Part 6: Quick Reference Tables

### Scaling Triggers

| Trigger Type | Use Case | Example Threshold |
|-------------|----------|-------------------|
| `UTILIZATION_ABOVE` | Activate surge | 0.90 (90%) |
| `UTILIZATION_BELOW` | Deactivate surge | 0.70 (70%) |
| `QUEUE_LENGTH_ABOVE` | Pressure indicator | 5 patients |
| `TIME_OF_DAY` | Shift-based capacity | 08:00-20:00 |

### Scaling Actions

| Action Type | Effect | Schematic Update |
|------------|--------|------------------|
| `ADD_CAPACITY` | Activate surge beds | Surge bar extends |
| `REMOVE_CAPACITY` | Deactivate when empty | Surge bar shrinks |
| `ACCELERATE_DISCHARGE` | Reduce LoS | Badge on exit node |
| `ACTIVATE_DISCHARGE_LOUNGE` | Enable bolt-on | Node appears |
| `DIVERT_ARRIVALS` | Redirect to satellite | Entry edge redirects |

### Bolt-On Categories

| Category | Zone | Visual Pattern |
|----------|------|----------------|
| Diagnostic | Upstream | Bidirectional edges |
| Step-Down | Between critical & ward | Vertical in crucifix |
| Holding | Near exit | Precedes discharge |
| Satellite | Below main | Parallel entry path |
| Overflow | Parallel to ward | Same Y as parent |

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

# Run tests (including scaling tests)
pytest tests/test_scaling_integration.py -v
pytest tests/test_dynamic_resource.py -v

# Run all tests
pytest tests/ -v
```

---

*Document Version: 2.0*
*Last Updated: January 2026*
*For: Claude Code in Cursor - FAER-M Integration*
