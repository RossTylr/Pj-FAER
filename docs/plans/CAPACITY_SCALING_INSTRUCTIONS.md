# Capacity Scaling Model - Claude Code Instructions

## Overview

Implement dynamic capacity scaling for Pj-FAER to model step-up/step-down of hospital beds and resources in response to patient flow peaks. This enables evaluation of surge protocols, discharge acceleration, and adaptive resource management.

---

## Phase 1: Core Data Classes

### Task 1.1: Create `src/faer/core/scaling.py`

Create new file with these dataclasses:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class ScalingTriggerType(Enum):
    UTILIZATION_ABOVE = "utilization_above"
    UTILIZATION_BELOW = "utilization_below"
    QUEUE_LENGTH_ABOVE = "queue_length_above"
    TIME_OF_DAY = "time_of_day"
    SIMULATION_TIME = "simulation_time"

class ScalingActionType(Enum):
    ADD_CAPACITY = "add_capacity"
    REMOVE_CAPACITY = "remove_capacity"
    ACCELERATE_DISCHARGE = "accelerate_discharge"
    ACTIVATE_DISCHARGE_LOUNGE = "activate_discharge_lounge"

@dataclass
class ScalingTrigger:
    trigger_type: ScalingTriggerType
    resource: str  # "ed_bays", "ward_beds", "itu_beds", "triage"
    threshold: float  # Utilization %, queue length, or time
    cooldown_mins: float = 60.0
    sustain_mins: float = 15.0
    start_time: Optional[float] = None  # For TIME_OF_DAY
    end_time: Optional[float] = None

@dataclass
class ScalingAction:
    action_type: ScalingActionType
    resource: str
    magnitude: int = 0
    los_reduction_pct: float = 0.0
    discharge_probability_boost: float = 0.0

@dataclass
class ScalingRule:
    name: str
    trigger: ScalingTrigger
    action: ScalingAction
    auto_deescalate: bool = True
    deescalation_threshold: Optional[float] = None
    deescalation_delay_mins: float = 30.0
    max_activations: int = -1
    enabled: bool = True

@dataclass
class CapacityScalingConfig:
    enabled: bool = False
    rules: List[ScalingRule] = field(default_factory=list)
    evaluation_interval_mins: float = 5.0
    discharge_lounge_capacity: int = 10
    discharge_lounge_max_wait_mins: float = 120.0
```

### Task 1.2: Update `src/faer/core/scenario.py`

Add to `FullScenario` dataclass:

```python
from faer.core.scaling import CapacityScalingConfig

@dataclass
class FullScenario:
    # ... existing fields ...
    capacity_scaling: CapacityScalingConfig = field(default_factory=CapacityScalingConfig)
```

### Task 1.3: Create `src/faer/model/dynamic_resource.py`

SimPy resources can't change capacity after creation. Create a wrapper:

```python
import simpy
from typing import List

class DynamicCapacityResource:
    """Wrapper that manages capacity changes via slot activation/deactivation."""

    def __init__(self, env: simpy.Environment, name: str,
                 initial_capacity: int, max_capacity: int,
                 is_priority: bool = False):
        self.env = env
        self.name = name
        self._active_slots = initial_capacity
        self.max_capacity = max_capacity
        self.is_priority = is_priority
        self._pending_deactivations = 0

        # Create resource with max capacity, control effective capacity ourselves
        if is_priority:
            self._resource = simpy.PriorityResource(env, capacity=max_capacity)
        else:
            self._resource = simpy.Resource(env, capacity=max_capacity)

        self.capacity_log: List[tuple] = [(0.0, initial_capacity)]

    @property
    def count(self) -> int:
        return self._resource.count

    @property
    def capacity(self) -> int:
        return self._active_slots - self._pending_deactivations

    @property
    def queue(self):
        return self._resource.queue

    def request(self, priority: int = 0):
        if self.is_priority:
            return self._resource.request(priority=priority)
        return self._resource.request()

    def add_capacity(self, amount: int) -> int:
        actual_add = min(amount, self.max_capacity - self._active_slots)
        self._active_slots += actual_add
        self.capacity_log.append((self.env.now, self._active_slots))
        return actual_add

    def remove_capacity(self, amount: int, graceful: bool = True) -> int:
        actual_remove = min(amount, self._active_slots - self._pending_deactivations)
        if graceful:
            self._pending_deactivations += actual_remove
        else:
            empty_slots = self._active_slots - self.count
            immediate_remove = min(actual_remove, empty_slots)
            self._active_slots -= immediate_remove
            self.capacity_log.append((self.env.now, self._active_slots))
        return actual_remove
```

### Task 1.4: Write tests `tests/test_dynamic_resource.py`

```python
import simpy
import pytest
from faer.model.dynamic_resource import DynamicCapacityResource

def test_initial_capacity():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 5, 20)
    assert res.capacity == 5

def test_add_capacity():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 5, 20)
    added = res.add_capacity(3)
    assert added == 3
    assert res.capacity == 8

def test_add_capacity_respects_max():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 18, 20)
    added = res.add_capacity(5)
    assert added == 2
    assert res.capacity == 20

def test_capacity_log():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 5, 20)
    res.add_capacity(3)
    assert len(res.capacity_log) == 2
    assert res.capacity_log[-1] == (0.0, 8)
```

---

## Phase 2: Scaling Monitor

### Task 2.1: Create `src/faer/model/scaling_monitor.py`

```python
import simpy
from dataclasses import dataclass, field
from typing import Dict
from faer.core.scaling import (
    ScalingTriggerType, ScalingActionType, ScalingRule, CapacityScalingConfig
)
from faer.model.dynamic_resource import DynamicCapacityResource

@dataclass
class RuleState:
    is_active: bool = False
    last_activation: float = -9999.0
    trigger_start: float = None
    deescalation_start: float = None
    activation_count: int = 0

def evaluate_trigger(trigger, resources: Dict[str, DynamicCapacityResource],
                     current_time: float) -> tuple:
    """Returns (triggered: bool, value: float)"""

    if trigger.trigger_type == ScalingTriggerType.UTILIZATION_ABOVE:
        res = resources.get(trigger.resource)
        if res and res.capacity > 0:
            util = res.count / res.capacity
            return util >= trigger.threshold, util

    elif trigger.trigger_type == ScalingTriggerType.UTILIZATION_BELOW:
        res = resources.get(trigger.resource)
        if res and res.capacity > 0:
            util = res.count / res.capacity
            return util <= trigger.threshold, util

    elif trigger.trigger_type == ScalingTriggerType.QUEUE_LENGTH_ABOVE:
        res = resources.get(trigger.resource)
        if res:
            qlen = len(res.queue)
            return qlen >= trigger.threshold, qlen

    elif trigger.trigger_type == ScalingTriggerType.TIME_OF_DAY:
        time_of_day = current_time % 1440
        in_window = trigger.start_time <= time_of_day < trigger.end_time
        return in_window, time_of_day

    elif trigger.trigger_type == ScalingTriggerType.SIMULATION_TIME:
        return current_time >= trigger.threshold, current_time

    return False, 0.0

def execute_action(action, resources: Dict[str, DynamicCapacityResource],
                   discharge_manager=None) -> dict:
    """Execute scaling action, return details for logging."""
    res = resources.get(action.resource)
    result = {"resource": action.resource, "action": action.action_type.value}

    if action.action_type == ScalingActionType.ADD_CAPACITY and res:
        old_cap = res.capacity
        added = res.add_capacity(action.magnitude)
        result.update({"old": old_cap, "new": res.capacity, "delta": added})

    elif action.action_type == ScalingActionType.REMOVE_CAPACITY and res:
        old_cap = res.capacity
        removed = res.remove_capacity(action.magnitude)
        result.update({"old": old_cap, "new": res.capacity, "delta": -removed})

    elif action.action_type == ScalingActionType.ACCELERATE_DISCHARGE:
        if discharge_manager:
            discharge_manager.activate_acceleration(action.los_reduction_pct)
        result.update({"los_reduction_pct": action.los_reduction_pct})

    return result

def capacity_scaling_monitor(env: simpy.Environment,
                             resources: Dict[str, DynamicCapacityResource],
                             config: CapacityScalingConfig,
                             results_collector,
                             discharge_manager=None):
    """Background process monitoring utilization and triggering scaling."""

    rule_states = {rule.name: RuleState() for rule in config.rules}

    while True:
        yield env.timeout(config.evaluation_interval_mins)

        for rule in config.rules:
            if not rule.enabled:
                continue

            state = rule_states[rule.name]

            # Check max activations
            if rule.max_activations > 0 and state.activation_count >= rule.max_activations:
                continue

            # Check cooldown
            if env.now - state.last_activation < rule.trigger.cooldown_mins:
                continue

            triggered, value = evaluate_trigger(rule.trigger, resources, env.now)

            if triggered and not state.is_active:
                # Check sustain period
                if state.trigger_start is None:
                    state.trigger_start = env.now
                elif env.now - state.trigger_start >= rule.trigger.sustain_mins:
                    # Execute action
                    result = execute_action(rule.action, resources, discharge_manager)
                    state.is_active = True
                    state.last_activation = env.now
                    state.activation_count += 1
                    state.trigger_start = None

                    # Log event
                    if results_collector and hasattr(results_collector, 'record_scaling_event'):
                        results_collector.record_scaling_event(
                            env.now, rule.name, result, value, "scale_up"
                        )

            elif not triggered:
                state.trigger_start = None

                # Check de-escalation
                if state.is_active and rule.auto_deescalate:
                    if rule.deescalation_threshold is not None:
                        _, current_val = evaluate_trigger(rule.trigger, resources, env.now)
                        should_deescalate = current_val < rule.deescalation_threshold
                    else:
                        should_deescalate = True

                    if should_deescalate:
                        if state.deescalation_start is None:
                            state.deescalation_start = env.now
                        elif env.now - state.deescalation_start >= rule.deescalation_delay_mins:
                            # Reverse action
                            reverse_action = _get_reverse_action(rule.action)
                            result = execute_action(reverse_action, resources, discharge_manager)
                            state.is_active = False
                            state.deescalation_start = None

                            if results_collector and hasattr(results_collector, 'record_scaling_event'):
                                results_collector.record_scaling_event(
                                    env.now, rule.name, result, current_val, "scale_down"
                                )
                    else:
                        state.deescalation_start = None

def _get_reverse_action(action):
    """Create reverse action for de-escalation."""
    from copy import deepcopy
    reverse = deepcopy(action)
    if action.action_type == ScalingActionType.ADD_CAPACITY:
        reverse.action_type = ScalingActionType.REMOVE_CAPACITY
    elif action.action_type == ScalingActionType.REMOVE_CAPACITY:
        reverse.action_type = ScalingActionType.ADD_CAPACITY
    return reverse
```

### Task 2.2: Update `src/faer/results/collector.py`

Add scaling event tracking to `FullResultsCollector`:

```python
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ScalingEvent:
    time: float
    rule_name: str
    action_type: str
    resource: str
    old_capacity: int
    new_capacity: int
    trigger_value: float
    direction: str  # "scale_up" or "scale_down"

# Add to FullResultsCollector class:
scaling_events: List[ScalingEvent] = field(default_factory=list)

def record_scaling_event(self, time, rule_name, result, trigger_value, direction):
    self.scaling_events.append(ScalingEvent(
        time=time,
        rule_name=rule_name,
        action_type=result.get("action", ""),
        resource=result.get("resource", ""),
        old_capacity=result.get("old", 0),
        new_capacity=result.get("new", 0),
        trigger_value=trigger_value,
        direction=direction
    ))
```

### Task 2.3: Integrate into `src/faer/model/full_model.py`

In `run_full_simulation()`, after creating resources:

```python
from faer.model.scaling_monitor import capacity_scaling_monitor
from faer.model.dynamic_resource import DynamicCapacityResource

def run_full_simulation(scenario: FullScenario):
    env = simpy.Environment()

    # Create dynamic resources if scaling enabled
    if scenario.capacity_scaling.enabled:
        dynamic_resources = {
            "ed_bays": DynamicCapacityResource(
                env, "ed_bays", scenario.n_ed_bays,
                scenario.n_ed_bays + 20, is_priority=True
            ),
            "ward_beds": DynamicCapacityResource(
                env, "ward_beds", scenario.ward_config.capacity,
                scenario.ward_config.capacity + 30
            ),
            "itu_beds": DynamicCapacityResource(
                env, "itu_beds", scenario.itu_config.capacity,
                scenario.itu_config.capacity + 10
            ),
            "triage": DynamicCapacityResource(
                env, "triage", scenario.n_triage,
                scenario.n_triage + 5, is_priority=True
            ),
        }

        # Start scaling monitor
        env.process(capacity_scaling_monitor(
            env, dynamic_resources, scenario.capacity_scaling, results
        ))

        # Use dynamic_resources._resource for patient processes
    else:
        # Use standard SimPy resources (existing code)
        pass
```

---

## Phase 3: Discharge Manager

### Task 3.1: Create `src/faer/model/discharge.py`

```python
import simpy
from typing import List, Optional
from faer.core.scaling import CapacityScalingConfig

class DischargeManager:
    """Manages discharge acceleration and discharge lounge."""

    def __init__(self, env: simpy.Environment, config: CapacityScalingConfig):
        self.env = env
        self.config = config
        self.acceleration_active = False
        self.los_reduction_factor = 1.0

        # Discharge lounge
        self.discharge_lounge = None
        if config.discharge_lounge_capacity > 0:
            self.discharge_lounge = simpy.Resource(
                env, capacity=config.discharge_lounge_capacity
            )

        self.lounge_patients: List = []
        self.lounge_log: List[tuple] = []

    def activate_acceleration(self, los_reduction_pct: float):
        self.acceleration_active = True
        self.los_reduction_factor = 1.0 - (los_reduction_pct / 100.0)

    def deactivate_acceleration(self):
        self.acceleration_active = False
        self.los_reduction_factor = 1.0

    def get_adjusted_los(self, base_los: float) -> float:
        """Apply LoS reduction if acceleration active."""
        return base_los * self.los_reduction_factor

    def try_move_to_lounge(self, patient, bed_request) -> bool:
        """Try to move patient to discharge lounge, freeing bed immediately."""
        if self.discharge_lounge is None:
            return False

        if self.discharge_lounge.count < self.discharge_lounge.capacity:
            self.lounge_patients.append(patient)
            self.lounge_log.append((self.env.now, patient.id, "entered"))
            return True

        return False

    def discharge_lounge_process(self, patient, bed_resource):
        """Process for patient in discharge lounge."""
        with self.discharge_lounge.request() as req:
            yield req

            # Release the ward bed immediately
            # Patient waits in lounge for transport/pickup
            max_wait = self.config.discharge_lounge_max_wait_mins
            yield self.env.timeout(min(patient.remaining_discharge_time, max_wait))

            self.lounge_patients.remove(patient)
            self.lounge_log.append((self.env.now, patient.id, "departed"))
```

### Task 3.2: Integrate with ward process

In the ward patient process, use discharge manager:

```python
# In ward stay process
if discharge_manager:
    base_los = rng.lognormal(...)
    adjusted_los = discharge_manager.get_adjusted_los(base_los)
    yield env.timeout(adjusted_los)
else:
    yield env.timeout(base_los)
```

---

## Phase 4: Streamlit UI

### Task 4.1: Create `app/pages/10_Capacity_Scaling.py`

```python
import streamlit as st
from faer.core.scaling import (
    ScalingTriggerType, ScalingActionType, ScalingTrigger,
    ScalingAction, ScalingRule, CapacityScalingConfig
)

st.set_page_config(page_title="Capacity Scaling", page_icon="📈", layout="wide")
st.title("Capacity Scaling Configuration")

# Initialize session state
if "scaling_config" not in st.session_state:
    st.session_state.scaling_config = CapacityScalingConfig()

if "scaling_rules" not in st.session_state:
    st.session_state.scaling_rules = []

# Global settings
col1, col2, col3 = st.columns(3)
with col1:
    enabled = st.toggle("Enable Capacity Scaling",
                        value=st.session_state.scaling_config.enabled)
with col2:
    eval_interval = st.number_input("Evaluation Interval (min)",
                                     min_value=1.0, max_value=30.0, value=5.0)
with col3:
    max_actions = st.number_input("Max Concurrent Actions",
                                   min_value=1, max_value=10, value=3)

st.divider()

# Scaling Rules
st.subheader("Scaling Rules")

if st.button("➕ Add New Rule"):
    st.session_state.show_rule_editor = True
    st.session_state.editing_rule_idx = None

# Display existing rules
for idx, rule in enumerate(st.session_state.scaling_rules):
    with st.expander(f"Rule {idx+1}: {rule.name}", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Trigger:** {rule.trigger.resource} "
                     f"{rule.trigger.trigger_type.value} {rule.trigger.threshold}")
            st.write(f"**Action:** {rule.action.action_type.value} "
                     f"{rule.action.magnitude} {rule.action.resource}")
            if rule.auto_deescalate:
                st.write(f"**De-escalate:** below {rule.deescalation_threshold}")
        with col2:
            if st.button("Edit", key=f"edit_{idx}"):
                st.session_state.show_rule_editor = True
                st.session_state.editing_rule_idx = idx
            if st.button("Delete", key=f"del_{idx}"):
                st.session_state.scaling_rules.pop(idx)
                st.rerun()

st.divider()

# Discharge Lounge
st.subheader("Discharge Lounge")
col1, col2, col3 = st.columns(3)
with col1:
    lounge_enabled = st.toggle("Enable Discharge Lounge", value=False)
with col2:
    lounge_capacity = st.number_input("Lounge Capacity",
                                       min_value=0, max_value=30, value=10)
with col3:
    lounge_max_wait = st.number_input("Max Wait (min)",
                                       min_value=30, max_value=240, value=120)

st.divider()

# Preset Protocols
st.subheader("Preset Protocols")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚨 Standard Surge"):
        st.session_state.scaling_rules.append(
            ScalingRule(
                name="ED Surge Protocol",
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
                ),
                deescalation_threshold=0.70
            )
        )
        st.rerun()
with col2:
    if st.button("🏥 Discharge Push"):
        st.session_state.scaling_rules.append(
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
                    los_reduction_pct=15.0
                ),
                deescalation_threshold=0.75
            )
        )
        st.rerun()
with col3:
    if st.button("🗑️ Clear All Rules"):
        st.session_state.scaling_rules = []
        st.rerun()

# Build config from session state
st.session_state.scaling_config = CapacityScalingConfig(
    enabled=enabled,
    rules=st.session_state.scaling_rules,
    evaluation_interval_mins=eval_interval,
    discharge_lounge_capacity=lounge_capacity if lounge_enabled else 0,
    discharge_lounge_max_wait_mins=lounge_max_wait
)

# Show config summary
if enabled and st.session_state.scaling_rules:
    st.success(f"✅ Scaling enabled with {len(st.session_state.scaling_rules)} rules")
```

### Task 4.2: Rule Editor Modal

Add to `10_Capacity_Scaling.py`:

```python
# Rule Editor (in dialog or expander)
if st.session_state.get("show_rule_editor", False):
    st.subheader("Rule Editor")

    rule_name = st.text_input("Rule Name", value="New Rule")

    st.write("**Trigger**")
    trigger_type = st.selectbox("Trigger Type",
        options=[t.value for t in ScalingTriggerType])
    trigger_resource = st.selectbox("Resource",
        options=["ed_bays", "ward_beds", "itu_beds", "triage"])
    trigger_threshold = st.slider("Threshold", 0.0, 1.0, 0.85)
    sustain_mins = st.number_input("Sustain (min)", value=15.0)
    cooldown_mins = st.number_input("Cooldown (min)", value=60.0)

    st.write("**Action**")
    action_type = st.selectbox("Action Type",
        options=[a.value for a in ScalingActionType])
    action_resource = st.selectbox("Target Resource",
        options=["ed_bays", "ward_beds", "itu_beds", "triage"])

    if action_type == "add_capacity" or action_type == "remove_capacity":
        magnitude = st.number_input("Amount", min_value=1, max_value=20, value=5)
    elif action_type == "accelerate_discharge":
        los_reduction = st.slider("LoS Reduction %", 0, 30, 15)

    st.write("**De-escalation**")
    auto_deescalate = st.checkbox("Auto de-escalate", value=True)
    if auto_deescalate:
        deesc_threshold = st.slider("De-escalate below", 0.0, 1.0, 0.70)
        deesc_delay = st.number_input("Delay (min)", value=30.0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Rule"):
            # Build and save rule
            new_rule = ScalingRule(
                name=rule_name,
                trigger=ScalingTrigger(
                    trigger_type=ScalingTriggerType(trigger_type),
                    resource=trigger_resource,
                    threshold=trigger_threshold,
                    sustain_mins=sustain_mins,
                    cooldown_mins=cooldown_mins
                ),
                action=ScalingAction(
                    action_type=ScalingActionType(action_type),
                    resource=action_resource,
                    magnitude=magnitude if action_type in ["add_capacity", "remove_capacity"] else 0,
                    los_reduction_pct=los_reduction if action_type == "accelerate_discharge" else 0
                ),
                auto_deescalate=auto_deescalate,
                deescalation_threshold=deesc_threshold if auto_deescalate else None,
                deescalation_delay_mins=deesc_delay if auto_deescalate else 30
            )

            if st.session_state.editing_rule_idx is not None:
                st.session_state.scaling_rules[st.session_state.editing_rule_idx] = new_rule
            else:
                st.session_state.scaling_rules.append(new_rule)

            st.session_state.show_rule_editor = False
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.session_state.show_rule_editor = False
            st.rerun()
```

### Task 4.3: Update `app/pages/5_Run.py`

Pass scaling config to scenario:

```python
# When building FullScenario, include scaling config
scenario = FullScenario(
    # ... existing parameters ...
    capacity_scaling=st.session_state.get("scaling_config", CapacityScalingConfig())
)
```

### Task 4.4: Update `app/pages/6_Results.py`

Add scaling results section:

```python
# After existing results display
if st.session_state.get("scaling_config", {}).get("enabled", False):
    st.divider()
    st.subheader("📈 Capacity Scaling Analysis")

    results = st.session_state.run_results
    scaling_events = results.get("scaling_events", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        scale_ups = len([e for e in scaling_events if e["direction"] == "scale_up"])
        st.metric("Scale-Up Events", scale_ups)
    with col2:
        # Calculate additional bed-hours
        st.metric("Additional Bed-Hours", f"{results.get('additional_bed_hours', 0):.1f}")
    with col3:
        st.metric("Time at Surge", f"{results.get('pct_time_at_surge', 0):.1f}%")

    # Scaling timeline chart
    if scaling_events:
        import plotly.graph_objects as go

        fig = go.Figure()
        times = [e["time"] for e in scaling_events]
        capacities = [e["new_capacity"] for e in scaling_events]

        fig.add_trace(go.Scatter(
            x=times, y=capacities,
            mode='lines+markers',
            name='Capacity',
            line=dict(shape='hv')
        ))

        fig.update_layout(
            title="Capacity Over Time",
            xaxis_title="Simulation Time (min)",
            yaxis_title="Capacity"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rule effectiveness table
    st.write("**Rule Effectiveness**")
    rule_stats = {}  # Aggregate from scaling_events
    for event in scaling_events:
        rule = event["rule_name"]
        if rule not in rule_stats:
            rule_stats[rule] = {"activations": 0, "total_duration": 0}
        rule_stats[rule]["activations"] += 1

    import pandas as pd
    df = pd.DataFrame([
        {"Rule": k, "Activations": v["activations"]}
        for k, v in rule_stats.items()
    ])
    st.dataframe(df, use_container_width=True)
```

---

## Phase 5: Testing

### Task 5.1: Create `tests/test_scaling_triggers.py`

```python
import pytest
import simpy
from faer.core.scaling import ScalingTrigger, ScalingTriggerType
from faer.model.dynamic_resource import DynamicCapacityResource
from faer.model.scaling_monitor import evaluate_trigger

def test_utilization_above_trigger():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 10, 20)

    # Fill 9 of 10 slots (90% utilization)
    for _ in range(9):
        env.process(occupy_resource(env, res))
    env.run(until=1)

    trigger = ScalingTrigger(
        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
        resource="test",
        threshold=0.85
    )

    triggered, value = evaluate_trigger(trigger, {"test": res}, 0)
    assert triggered is True
    assert value == 0.9

def test_utilization_below_threshold_no_trigger():
    env = simpy.Environment()
    res = DynamicCapacityResource(env, "test", 10, 20)

    # Fill 5 of 10 slots (50% utilization)
    for _ in range(5):
        env.process(occupy_resource(env, res))
    env.run(until=1)

    trigger = ScalingTrigger(
        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
        resource="test",
        threshold=0.85
    )

    triggered, value = evaluate_trigger(trigger, {"test": res}, 0)
    assert triggered is False
    assert value == 0.5

def occupy_resource(env, res):
    with res.request() as req:
        yield req
        yield env.timeout(1000)  # Hold forever
```

### Task 5.2: Create `tests/test_scaling_integration.py`

```python
import pytest
from faer.core.scenario import FullScenario
from faer.core.scaling import (
    CapacityScalingConfig, ScalingRule, ScalingTrigger,
    ScalingAction, ScalingTriggerType, ScalingActionType
)
from faer.model.full_model import run_full_simulation

def test_surge_protocol_activates():
    """Integration test: scaling activates under load."""
    config = CapacityScalingConfig(
        enabled=True,
        rules=[
            ScalingRule(
                name="ED Surge",
                trigger=ScalingTrigger(
                    trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                    resource="ed_bays",
                    threshold=0.80,
                    sustain_mins=5,
                    cooldown_mins=30
                ),
                action=ScalingAction(
                    action_type=ScalingActionType.ADD_CAPACITY,
                    resource="ed_bays",
                    magnitude=5
                )
            )
        ],
        evaluation_interval_mins=2.0
    )

    scenario = FullScenario(
        run_length=480,
        n_ed_bays=10,
        arrival_rate=12.0,  # High load
        capacity_scaling=config,
        random_seed=42
    )

    results = run_full_simulation(scenario)

    # Should have triggered at least once
    assert len(results.scaling_events) > 0
    assert any(e.direction == "scale_up" for e in results.scaling_events)

def test_scaling_disabled_no_events():
    """No scaling events when disabled."""
    config = CapacityScalingConfig(enabled=False)

    scenario = FullScenario(
        run_length=480,
        capacity_scaling=config,
        random_seed=42
    )

    results = run_full_simulation(scenario)
    assert len(results.scaling_events) == 0
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/faer/core/scaling.py` | CREATE | Data classes for scaling config |
| `src/faer/core/scenario.py` | MODIFY | Add capacity_scaling field |
| `src/faer/model/dynamic_resource.py` | CREATE | Dynamic capacity wrapper |
| `src/faer/model/scaling_monitor.py` | CREATE | Monitor process & trigger logic |
| `src/faer/model/discharge.py` | CREATE | Discharge acceleration manager |
| `src/faer/model/full_model.py` | MODIFY | Integrate scaling into simulation |
| `src/faer/results/collector.py` | MODIFY | Add scaling event tracking |
| `app/pages/10_Capacity_Scaling.py` | CREATE | Streamlit configuration UI |
| `app/pages/5_Run.py` | MODIFY | Pass scaling config to scenario |
| `app/pages/6_Results.py` | MODIFY | Add scaling results display |
| `tests/test_dynamic_resource.py` | CREATE | Unit tests for dynamic resource |
| `tests/test_scaling_triggers.py` | CREATE | Unit tests for triggers |
| `tests/test_scaling_integration.py` | CREATE | Integration tests |

---

## Execution Order

1. Phase 1: Core data classes + dynamic resource + tests
2. Phase 2: Scaling monitor + results tracking + integration
3. Phase 3: Discharge manager
4. Phase 4: Streamlit UI (tab, run integration, results display)
5. Phase 5: Full integration testing

Run `pytest tests/test_scaling*.py -v` after each phase to verify.
