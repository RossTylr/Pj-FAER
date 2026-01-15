# Capacity Scaling Model - Implementation Plan

## Executive Summary

This plan outlines the implementation of **dynamic capacity scaling** for Pj-FAER, enabling hospitals to model step-up/step-down of beds and resources in response to patient flow peaks. The feature allows evaluation of surge protocols, discharge acceleration, and adaptive resource management.

---

## 1. Problem Statement

### Current Limitations
The existing model uses **static capacity** - all resources (beds, staff, equipment) are fixed integers created at simulation start. This doesn't reflect real hospital operations where:

1. **Surge protocols** activate additional capacity during peaks
2. **Discharge pushes** accelerate patient flow during congestion
3. **Escalation policies** open surge beds when utilization crosses thresholds
4. **De-escalation** returns capacity to baseline when pressure eases
5. **Shift changes** vary staffing levels throughout the day

### Business Value
- **What-if analysis**: "If we open 5 surge beds at 80% ED utilization, what's the impact?"
- **Protocol validation**: Test surge policies before real-world implementation
- **Cost-benefit analysis**: Balance additional capacity costs vs. wait time reduction
- **Discharge planning**: Model the impact of earlier discharge windows

---

## 2. Core Concepts

### 2.1 Capacity Scaling Types

| Type | Description | Trigger | Example |
|------|-------------|---------|---------|
| **Threshold-Based** | Automatic scaling when utilization crosses a threshold | Utilization ≥ X% | Open surge ward at 85% ED utilization |
| **Time-Based** | Scheduled capacity changes at specific times | Clock time | Extra staff 08:00-16:00 |
| **Event-Based** | Capacity change triggered by external event | Major incident | Activate disaster protocol |
| **Manual Override** | User-defined capacity changes at specific simulation times | User-specified | "At t=480, add 10 ward beds" |

### 2.2 Scaling Actions

| Action | Description | Resources Affected |
|--------|-------------|-------------------|
| **Add Capacity** | Increase resource count | ED bays, Ward beds, ITU beds, Theatre tables |
| **Remove Capacity** | Decrease resource count (graceful - wait for current occupants) | All resources |
| **Accelerate Discharge** | Reduce LoS for eligible patients | Ward, ITU (via discharge probability boost) |
| **Divert Arrivals** | Redirect lower-acuity patients | Ambulance arrivals (to other facilities) |
| **Pool Resources** | Temporarily combine separate resource pools | ED bays + surge area |

### 2.3 Discharge Expansion Options

| Option | Mechanism | Impact |
|--------|-----------|--------|
| **Early Discharge Window** | Reduce LoS mean by X% for medically fit patients | Faster ward turnover |
| **Discharge Lounge** | Move discharged patients to holding area, freeing bed immediately | Instant bed availability |
| **Home Care Pathway** | Increase probability of early discharge | Reduced admission rate |
| **Social Care Acceleration** | Reduce delayed discharge (bed blocking) duration | Ward bed throughput |

---

## 3. Data Model

### 3.1 New Configuration Classes

```python
# src/faer/core/scaling.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable

class ScalingTriggerType(Enum):
    UTILIZATION_ABOVE = "utilization_above"
    UTILIZATION_BELOW = "utilization_below"
    QUEUE_LENGTH_ABOVE = "queue_length_above"
    TIME_OF_DAY = "time_of_day"
    SIMULATION_TIME = "simulation_time"
    MANUAL = "manual"

class ScalingActionType(Enum):
    ADD_CAPACITY = "add_capacity"
    REMOVE_CAPACITY = "remove_capacity"
    ACCELERATE_DISCHARGE = "accelerate_discharge"
    ACTIVATE_DISCHARGE_LOUNGE = "activate_discharge_lounge"
    DIVERT_ARRIVALS = "divert_arrivals"

@dataclass
class ScalingTrigger:
    """Defines when a scaling action should be triggered."""
    trigger_type: ScalingTriggerType
    resource: str  # e.g., "ed_bays", "ward_beds"
    threshold: float  # Utilization %, queue length, or time

    # Hysteresis to prevent oscillation
    cooldown_mins: float = 60.0  # Min time between activations
    sustain_mins: float = 15.0   # How long condition must hold before triggering

    # For time-based triggers
    start_time: Optional[float] = None  # Minutes from midnight
    end_time: Optional[float] = None

@dataclass
class ScalingAction:
    """Defines what happens when a trigger fires."""
    action_type: ScalingActionType
    resource: str
    magnitude: int = 0  # Number of beds/staff to add/remove

    # For discharge acceleration
    los_reduction_pct: float = 0.0  # Reduce LoS by this percentage
    discharge_probability_boost: float = 0.0  # Increase early discharge prob

    # For arrival diversion
    diversion_rate: float = 0.0  # Fraction of arrivals to divert
    diversion_priorities: List[str] = field(default_factory=lambda: ["P4", "P3"])

@dataclass
class ScalingRule:
    """A complete scaling rule: trigger + action + de-escalation."""
    name: str
    trigger: ScalingTrigger
    action: ScalingAction

    # De-escalation (optional)
    auto_deescalate: bool = True
    deescalation_threshold: Optional[float] = None  # If util drops below this
    deescalation_delay_mins: float = 30.0  # Wait before de-escalating

    # Constraints
    max_activations: int = -1  # -1 = unlimited
    enabled: bool = True

@dataclass
class CapacityScalingConfig:
    """Master configuration for all scaling behaviors."""
    enabled: bool = False
    rules: List[ScalingRule] = field(default_factory=list)

    # Global settings
    evaluation_interval_mins: float = 5.0  # How often to check triggers
    max_simultaneous_actions: int = 3  # Limit concurrent scaling

    # Discharge lounge (if activated)
    discharge_lounge_capacity: int = 10
    discharge_lounge_max_wait_mins: float = 120.0  # Max time in lounge

    # Baseline capacity (for calculating deltas)
    baseline_ed_bays: int = 20
    baseline_ward_beds: int = 30
    baseline_itu_beds: int = 6
```

### 3.2 Scenario Integration

```python
# Addition to src/faer/core/scenario.py

@dataclass
class FullScenario:
    # ... existing fields ...

    # Capacity Scaling (new)
    capacity_scaling: CapacityScalingConfig = field(
        default_factory=CapacityScalingConfig
    )
```

### 3.3 Results Tracking

```python
# Addition to src/faer/results/collector.py

@dataclass
class ScalingEvent:
    """Records a capacity scaling event."""
    time: float
    rule_name: str
    action_type: str
    resource: str
    old_capacity: int
    new_capacity: int
    trigger_value: float  # The utilization/queue length that triggered it

@dataclass
class ScalingMetrics:
    """Aggregated metrics for scaling behavior."""
    total_scale_up_events: int = 0
    total_scale_down_events: int = 0
    total_additional_bed_hours: float = 0.0  # Cumulative surge capacity used
    avg_time_at_surge: float = 0.0  # % of run time at elevated capacity
    discharge_lounge_utilization: float = 0.0
    patients_diverted: int = 0

    # Per-rule metrics
    rule_activations: Dict[str, int] = field(default_factory=dict)
    rule_effectiveness: Dict[str, float] = field(default_factory=dict)  # Wait time reduction
```

---

## 4. Simulation Logic

### 4.1 Dynamic Resource Management

SimPy resources cannot change capacity after creation. **Solution**: Use a custom wrapper that manages multiple underlying resources.

```python
# src/faer/model/dynamic_resource.py

import simpy
from typing import List, Optional

class DynamicCapacityResource:
    """
    Wrapper around SimPy resources that supports capacity changes.

    Strategy: Maintain a pool of individual "slot" resources.
    Adding capacity = activate more slots.
    Removing capacity = deactivate slots (gracefully, after current user releases).
    """

    def __init__(self, env: simpy.Environment, name: str,
                 initial_capacity: int, max_capacity: int,
                 is_priority: bool = False):
        self.env = env
        self.name = name
        self.current_capacity = initial_capacity
        self.max_capacity = max_capacity
        self.is_priority = is_priority

        # Create the actual SimPy resource with max capacity
        # We control effective capacity through our own queue management
        if is_priority:
            self._resource = simpy.PriorityResource(env, capacity=max_capacity)
        else:
            self._resource = simpy.Resource(env, capacity=max_capacity)

        # Track which "slots" are active
        self._active_slots = initial_capacity
        self._pending_deactivations = 0

        # Event log
        self.capacity_log: List[tuple] = [(0.0, initial_capacity)]

    @property
    def count(self) -> int:
        """Number of resources currently in use."""
        return self._resource.count

    @property
    def capacity(self) -> int:
        """Current effective capacity."""
        return self._active_slots - self._pending_deactivations

    def request(self, priority: int = 0):
        """
        Request a resource slot.
        Modified to respect current capacity limits.
        """
        # If at or over effective capacity, queue the request
        if self.is_priority:
            return self._resource.request(priority=priority)
        else:
            return self._resource.request()

    def add_capacity(self, amount: int) -> int:
        """
        Increase capacity by `amount` slots.
        Returns actual amount added (may be less if at max).
        """
        actual_add = min(amount, self.max_capacity - self._active_slots)
        self._active_slots += actual_add
        self.capacity_log.append((self.env.now, self._active_slots))
        return actual_add

    def remove_capacity(self, amount: int, graceful: bool = True) -> int:
        """
        Decrease capacity by `amount` slots.
        If graceful=True, waits for current occupants to leave before deactivating.
        Returns actual amount that will be removed.
        """
        actual_remove = min(amount, self._active_slots - self._pending_deactivations)

        if graceful:
            # Mark slots for deactivation when released
            self._pending_deactivations += actual_remove
        else:
            # Immediate (only if slots are empty)
            empty_slots = self._active_slots - self.count
            immediate_remove = min(actual_remove, empty_slots)
            self._active_slots -= immediate_remove
            self.capacity_log.append((self.env.now, self._active_slots))

        return actual_remove
```

### 4.2 Capacity Monitor Process

```python
# src/faer/model/scaling_monitor.py

def capacity_scaling_monitor(env: simpy.Environment,
                             resources: Dict[str, DynamicCapacityResource],
                             config: CapacityScalingConfig,
                             results: FullResultsCollector):
    """
    Background process that monitors utilization and triggers scaling actions.
    Runs every `evaluation_interval_mins` minutes.
    """
    rule_states = {rule.name: RuleState() for rule in config.rules}

    while True:
        yield env.timeout(config.evaluation_interval_mins)

        for rule in config.rules:
            if not rule.enabled:
                continue

            state = rule_states[rule.name]

            # Check cooldown
            if env.now - state.last_activation < rule.trigger.cooldown_mins:
                continue

            # Evaluate trigger
            triggered, trigger_value = evaluate_trigger(
                rule.trigger, resources, env.now
            )

            if triggered and not state.is_active:
                # Check sustain period
                if state.trigger_start is None:
                    state.trigger_start = env.now
                elif env.now - state.trigger_start >= rule.trigger.sustain_mins:
                    # Execute action
                    execute_scaling_action(rule.action, resources, env, results)
                    state.is_active = True
                    state.last_activation = env.now
                    state.trigger_start = None

                    results.record_scaling_event(ScalingEvent(
                        time=env.now,
                        rule_name=rule.name,
                        action_type=rule.action.action_type.value,
                        resource=rule.action.resource,
                        old_capacity=...,
                        new_capacity=...,
                        trigger_value=trigger_value
                    ))

            elif not triggered and state.is_active and rule.auto_deescalate:
                # Check de-escalation
                if rule.deescalation_threshold is not None:
                    _, current_value = evaluate_trigger(rule.trigger, resources, env.now)
                    if current_value < rule.deescalation_threshold:
                        if state.deescalation_start is None:
                            state.deescalation_start = env.now
                        elif env.now - state.deescalation_start >= rule.deescalation_delay_mins:
                            # Reverse the action
                            reverse_scaling_action(rule.action, resources, env, results)
                            state.is_active = False
                            state.deescalation_start = None
            else:
                state.trigger_start = None
                state.deescalation_start = None


def evaluate_trigger(trigger: ScalingTrigger,
                     resources: Dict[str, DynamicCapacityResource],
                     current_time: float) -> tuple[bool, float]:
    """Evaluate whether a trigger condition is met."""

    if trigger.trigger_type == ScalingTriggerType.UTILIZATION_ABOVE:
        resource = resources.get(trigger.resource)
        if resource:
            utilization = resource.count / resource.capacity if resource.capacity > 0 else 1.0
            return utilization >= trigger.threshold, utilization

    elif trigger.trigger_type == ScalingTriggerType.UTILIZATION_BELOW:
        resource = resources.get(trigger.resource)
        if resource:
            utilization = resource.count / resource.capacity if resource.capacity > 0 else 1.0
            return utilization <= trigger.threshold, utilization

    elif trigger.trigger_type == ScalingTriggerType.QUEUE_LENGTH_ABOVE:
        resource = resources.get(trigger.resource)
        if resource:
            queue_len = len(resource._resource.queue)
            return queue_len >= trigger.threshold, queue_len

    elif trigger.trigger_type == ScalingTriggerType.TIME_OF_DAY:
        time_of_day = current_time % 1440  # Minutes since midnight
        in_window = trigger.start_time <= time_of_day < trigger.end_time
        return in_window, time_of_day

    elif trigger.trigger_type == ScalingTriggerType.SIMULATION_TIME:
        return current_time >= trigger.threshold, current_time

    return False, 0.0
```

### 4.3 Discharge Acceleration

```python
# src/faer/model/discharge.py

class DischargeManager:
    """
    Manages discharge processes including acceleration and discharge lounge.
    """

    def __init__(self, env: simpy.Environment, config: CapacityScalingConfig):
        self.env = env
        self.config = config
        self.acceleration_active = False
        self.los_reduction_factor = 1.0  # Multiplier on LoS

        # Discharge lounge
        if config.discharge_lounge_capacity > 0:
            self.discharge_lounge = simpy.Resource(
                env, capacity=config.discharge_lounge_capacity
            )
        else:
            self.discharge_lounge = None

        self.lounge_queue: List[Patient] = []

    def activate_acceleration(self, los_reduction_pct: float):
        """Activate discharge acceleration."""
        self.acceleration_active = True
        self.los_reduction_factor = 1.0 - (los_reduction_pct / 100.0)

    def deactivate_acceleration(self):
        """Return to normal discharge patterns."""
        self.acceleration_active = False
        self.los_reduction_factor = 1.0

    def get_adjusted_los(self, base_los: float) -> float:
        """Get LoS adjusted for any active acceleration."""
        return base_los * self.los_reduction_factor

    def try_discharge_lounge(self, patient: Patient, ward_bed) -> bool:
        """
        Attempt to move patient to discharge lounge, freeing ward bed immediately.
        Returns True if successful.
        """
        if self.discharge_lounge is None:
            return False

        if self.discharge_lounge.count < self.discharge_lounge.capacity:
            # Move to lounge
            self.lounge_queue.append(patient)
            ward_bed.release()  # Free the bed immediately
            return True

        return False
```

---

## 5. UI Design

### 5.1 New Tab: "Capacity Scaling" (Page 10)

**Location**: `app/pages/10_Capacity_Scaling.py`

**Follows**: Results page (6_Results.py) - allows users to see baseline results, then explore "what if we had scaling?"

**Layout**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  CAPACITY SCALING                                          [?] Help │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Enable Scaling  │  │ Evaluation      │  │ Max Concurrent  │     │
│  │ [x] On          │  │ Interval: 5 min │  │ Actions: 3      │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  ══════════════════════════════════════════════════════════════════ │
│  SCALING RULES                                           [+ Add Rule]│
│  ══════════════════════════════════════════════════════════════════ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Rule 1: ED Surge Protocol                        [Edit] [Delete]││
│  │ ─────────────────────────────────────────────────────────────── ││
│  │ Trigger: ED Bay utilization ≥ 85% for 15 min                   ││
│  │ Action:  Add 5 surge ED bays                                   ││
│  │ De-escalate: When utilization < 70% for 30 min                 ││
│  │ Cooldown: 60 min between activations                           ││
│  │ Status: [●] Enabled                                            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Rule 2: Discharge Acceleration                   [Edit] [Delete]││
│  │ ─────────────────────────────────────────────────────────────── ││
│  │ Trigger: Ward utilization ≥ 90% for 15 min                     ││
│  │ Action:  Accelerate discharge (reduce LoS by 15%)              ││
│  │ De-escalate: When utilization < 75% for 30 min                 ││
│  │ Status: [●] Enabled                                            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Rule 3: Shift Change Staffing                    [Edit] [Delete]││
│  │ ─────────────────────────────────────────────────────────────── ││
│  │ Trigger: Time of day 08:00 - 20:00                             ││
│  │ Action:  Add 1 triage nurse                                    ││
│  │ Status: [●] Enabled                                            ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  ══════════════════════════════════════════════════════════════════ │
│  DISCHARGE LOUNGE                                                   │
│  ══════════════════════════════════════════════════════════════════ │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ Enable Lounge   │  │ Capacity        │  │ Max Wait        │     │
│  │ [x] On          │  │ 10 spaces       │  │ 120 min         │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
│  ══════════════════════════════════════════════════════════════════ │
│  PRESET PROTOCOLS                                                   │
│  ══════════════════════════════════════════════════════════════════ │
│                                                                     │
│  [ Standard Surge ]  [ Full Escalation ]  [ Discharge Push ]       │
│  [ Winter Pressure ] [ Major Incident ]   [ Clear All ]            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Rule Editor Modal

```
┌─────────────────────────────────────────────────────────────────────┐
│  ADD/EDIT SCALING RULE                                      [X]     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Rule Name: [ED Surge Protocol                              ]       │
│                                                                     │
│  ── TRIGGER ─────────────────────────────────────────────────────── │
│                                                                     │
│  Type: [Utilization Threshold ▼]                                    │
│                                                                     │
│  Resource:     [ED Bays ▼]                                          │
│  Condition:    [Above ▼]  Threshold: [85] %                         │
│  Sustain for:  [15] minutes before triggering                       │
│  Cooldown:     [60] minutes between activations                     │
│                                                                     │
│  ── ACTION ──────────────────────────────────────────────────────── │
│                                                                     │
│  Action Type:  [Add Capacity ▼]                                     │
│                                                                     │
│  Resource:     [ED Bays ▼]                                          │
│  Amount:       [5] additional beds/staff                            │
│                                                                     │
│  ── DE-ESCALATION ───────────────────────────────────────────────── │
│                                                                     │
│  [x] Auto de-escalate                                               │
│  When utilization drops below: [70] %                               │
│  Wait [30] minutes before de-escalating                             │
│                                                                     │
│                                        [ Cancel ]  [ Save Rule ]    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Results Integration

On the Results page (6_Results.py), add a new section when scaling is enabled:

```
══════════════════════════════════════════════════════════════════════
CAPACITY SCALING ANALYSIS
══════════════════════════════════════════════════════════════════════

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ Scale-Up Events      │  │ Additional Bed-Hours │  │ Time at Surge        │
│       12             │  │     156.4 hrs        │  │      18.3%           │
│ ±2.1 (95% CI)        │  │ ±23.1 (95% CI)       │  │ ±3.2% (95% CI)       │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

SCALING TIMELINE
────────────────────────────────────────────────────────────────────────
[Interactive Plotly chart showing capacity over time]
- X-axis: Simulation time (hours)
- Y-axis: Capacity (beds)
- Annotations: Scale-up/down events
- Overlay: Utilization line

RULE EFFECTIVENESS
────────────────────────────────────────────────────────────────────────
┌────────────────────────┬─────────────┬────────────────┬──────────────┐
│ Rule                   │ Activations │ Avg Duration   │ Wait Δ       │
├────────────────────────┼─────────────┼────────────────┼──────────────┤
│ ED Surge Protocol      │     8.2     │    45.3 min    │  -12.4 min   │
│ Discharge Acceleration │     4.1     │    62.8 min    │   -8.7 min   │
│ Shift Change Staffing  │     1.0     │   720.0 min    │   -3.2 min   │
└────────────────────────┴─────────────┴────────────────┴──────────────┘
```

### 5.4 Compare Page Enhancement

Add capacity scaling comparison in 7_Compare.py:

```
SCENARIO A (Baseline)          vs          SCENARIO B (With Scaling)
────────────────────────────────────────────────────────────────────
                                            Scaling Rules: 3 active
                                            Scale-up events: 12.3
                                            Additional bed-hours: 156.4

Mean ED Wait:    45.2 min                   32.8 min  (-27.4%)
P95 ED Wait:     98.4 min                   67.2 min  (-31.7%)
Ward Utilization: 94.2%                     86.1%     (-8.1 pp)
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Foundation)

**Files to create/modify**:
- `src/faer/core/scaling.py` - Data classes (ScalingTrigger, ScalingAction, ScalingRule, CapacityScalingConfig)
- `src/faer/core/scenario.py` - Add `capacity_scaling` field to FullScenario
- `src/faer/model/dynamic_resource.py` - DynamicCapacityResource wrapper

**Tasks**:
1. Implement all data classes for scaling configuration
2. Implement DynamicCapacityResource with add/remove capacity
3. Add capacity_scaling field to FullScenario
4. Write unit tests for dynamic resource behavior

**Acceptance**: Dynamic resource can change capacity mid-test, logs changes correctly.

### Phase 2: Simulation Logic

**Files to create/modify**:
- `src/faer/model/scaling_monitor.py` - Capacity scaling monitor process
- `src/faer/model/full_model.py` - Integrate scaling into main simulation
- `src/faer/results/collector.py` - Add ScalingEvent and ScalingMetrics

**Tasks**:
1. Implement `capacity_scaling_monitor` background process
2. Implement trigger evaluation functions
3. Implement action execution functions
4. Integrate monitor into `run_full_simulation`
5. Update ResultsCollector to track scaling events
6. Write tests for each trigger type and action type

**Acceptance**: Simulation responds to utilization thresholds, adds capacity, logs events.

### Phase 3: Discharge Acceleration

**Files to create/modify**:
- `src/faer/model/discharge.py` - DischargeManager class
- `src/faer/model/full_model.py` - Integrate discharge logic

**Tasks**:
1. Implement DischargeManager with LoS acceleration
2. Implement discharge lounge resource and process
3. Modify ward/ITU patient processes to use DischargeManager
4. Add discharge metrics to results
5. Write tests for discharge acceleration

**Acceptance**: LoS reduction activates when triggered, discharge lounge works.

### Phase 4: Streamlit UI

**Files to create/modify**:
- `app/pages/10_Capacity_Scaling.py` - New tab
- `app/pages/5_Run.py` - Pass scaling config to simulation
- `app/pages/6_Results.py` - Add scaling analysis section

**Tasks**:
1. Create 10_Capacity_Scaling.py with rule configuration UI
2. Implement rule editor modal
3. Add preset protocol buttons
4. Store scaling config in session state
5. Update Run page to include scaling config in scenario
6. Add scaling results section to Results page
7. Add scaling timeline visualization (Plotly)

**Acceptance**: User can configure rules, run simulation, see scaling results.

### Phase 5: Comparison & Refinement

**Files to modify**:
- `app/pages/7_Compare.py` - Add scaling comparison
- `app/pages/8_Sensitivity.py` - Add scaling parameters to sensitivity

**Tasks**:
1. Add scaling comparison to Compare page
2. Enable scaling threshold sensitivity analysis
3. Add scaling rule effectiveness metrics
4. Polish UI based on user feedback

**Acceptance**: Full integration, comparison works, sensitivity works.

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/test_dynamic_resource.py

def test_add_capacity():
    env = simpy.Environment()
    resource = DynamicCapacityResource(env, "test", 5, 20)

    assert resource.capacity == 5
    resource.add_capacity(3)
    assert resource.capacity == 8

def test_remove_capacity_graceful():
    """Capacity reduction waits for current occupants."""
    env = simpy.Environment()
    resource = DynamicCapacityResource(env, "test", 5, 20)

    # Fill all slots
    requests = [resource.request() for _ in range(5)]
    env.run()

    # Request removal - should not immediately reduce
    resource.remove_capacity(2, graceful=True)
    assert resource.capacity == 5  # Still 5 until occupants leave

# tests/test_scaling_triggers.py

def test_utilization_trigger():
    """Trigger fires when utilization exceeds threshold."""
    # Setup simulation with 80% utilization
    # Assert trigger fires at 85% threshold

def test_cooldown_prevents_rapid_scaling():
    """Cooldown prevents oscillation."""
    # Setup scenario that would trigger multiple times
    # Assert only one activation in cooldown period

def test_sustain_requirement():
    """Brief spikes don't trigger scaling."""
    # Setup brief spike above threshold
    # Assert no scaling (sustain not met)
```

### Integration Tests

```python
# tests/test_scaling_integration.py

def test_surge_protocol_end_to_end():
    """Full test of ED surge protocol."""
    scenario = FullScenario(
        capacity_scaling=CapacityScalingConfig(
            enabled=True,
            rules=[
                ScalingRule(
                    name="ED Surge",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.85,
                        sustain_mins=15
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=5
                    )
                )
            ]
        )
    )

    results = run_full_simulation(scenario)

    assert results.scaling_metrics.total_scale_up_events > 0
    assert results.scaling_metrics.total_additional_bed_hours > 0
```

---

## 8. Configuration Examples

### Example 1: Standard Surge Protocol

```python
ScalingRule(
    name="ED Surge Protocol",
    trigger=ScalingTrigger(
        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
        resource="ed_bays",
        threshold=0.85,
        sustain_mins=15,
        cooldown_mins=60
    ),
    action=ScalingAction(
        action_type=ScalingActionType.ADD_CAPACITY,
        resource="ed_bays",
        magnitude=5
    ),
    auto_deescalate=True,
    deescalation_threshold=0.70,
    deescalation_delay_mins=30
)
```

### Example 2: Discharge Push

```python
ScalingRule(
    name="Discharge Acceleration",
    trigger=ScalingTrigger(
        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
        resource="ward_beds",
        threshold=0.90,
        sustain_mins=15
    ),
    action=ScalingAction(
        action_type=ScalingActionType.ACCELERATE_DISCHARGE,
        resource="ward_beds",
        los_reduction_pct=15.0,
        discharge_probability_boost=0.1
    ),
    auto_deescalate=True,
    deescalation_threshold=0.75
)
```

### Example 3: Time-Based Staffing

```python
ScalingRule(
    name="Day Shift Extra Triage",
    trigger=ScalingTrigger(
        trigger_type=ScalingTriggerType.TIME_OF_DAY,
        resource="triage",
        threshold=0,  # Not used for time triggers
        start_time=480,   # 08:00
        end_time=1200     # 20:00
    ),
    action=ScalingAction(
        action_type=ScalingActionType.ADD_CAPACITY,
        resource="triage",
        magnitude=1
    ),
    auto_deescalate=True  # Will deactivate at end_time
)
```

### Example 4: Major Incident Surge

```python
ScalingRule(
    name="Major Incident Full Surge",
    trigger=ScalingTrigger(
        trigger_type=ScalingTriggerType.SIMULATION_TIME,
        resource="ed_bays",
        threshold=480  # Activate at t=480 (8 hours)
    ),
    action=ScalingAction(
        action_type=ScalingActionType.ADD_CAPACITY,
        resource="ed_bays",
        magnitude=10
    ),
    auto_deescalate=False,  # Stay active until end
    max_activations=1
)
```

---

## 9. Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| SimPy resource capacity is immutable | Use DynamicCapacityResource wrapper with max_capacity allocation |
| Oscillation (rapid scale up/down) | Cooldown periods, hysteresis, sustain requirements |
| Performance impact of frequent monitoring | Configurable evaluation interval, efficient trigger evaluation |
| Complex configuration UX | Preset protocols, sensible defaults, validation |
| Reproducibility with dynamic behavior | Include scaling events in deterministic seed handling |

---

## 10. Success Metrics

1. **Functional**: All trigger types work correctly
2. **Performance**: < 5% simulation time increase with scaling enabled
3. **Usability**: Users can configure rules in < 2 minutes
4. **Effectiveness**: Demonstrates measurable wait time reduction with surge protocols
5. **Reproducibility**: Same seed + config = identical scaling events

---

## 11. Future Enhancements

- **ML-based triggers**: Predict surge based on arrival patterns
- **Cost modeling**: Associate costs with additional capacity
- **Staff fatigue**: Model performance degradation at sustained surge
- **Cross-resource coordination**: Orchestrate multiple resource changes together
- **Historical replay**: Apply scaling rules to past scenarios

---

## Appendix: Key File Paths

| Component | Path |
|-----------|------|
| Scaling config classes | `src/faer/core/scaling.py` (new) |
| Dynamic resource | `src/faer/model/dynamic_resource.py` (new) |
| Scaling monitor | `src/faer/model/scaling_monitor.py` (new) |
| Discharge manager | `src/faer/model/discharge.py` (new) |
| UI tab | `app/pages/10_Capacity_Scaling.py` (new) |
| Results integration | `app/pages/6_Results.py` (modify) |
| Scenario integration | `src/faer/core/scenario.py` (modify) |
| Simulation integration | `src/faer/model/full_model.py` (modify) |
| Tests | `tests/test_scaling_*.py` (new) |
