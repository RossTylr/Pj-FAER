"""
Capacity Scaling Monitor for Pj-FAER.

Background SimPy process that monitors resource utilisation and triggers
scaling actions based on configured rules. Implements OPEL escalation
framework and custom threshold-based scaling.
"""

import simpy
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from copy import deepcopy

from faer.core.scaling import (
    ScalingTriggerType,
    ScalingActionType,
    ScalingTrigger,
    ScalingAction,
    ScalingRule,
    CapacityScalingConfig,
    OPELLevel,
    create_opel_rules,
)
from faer.model.dynamic_resource import DynamicCapacityResource


@dataclass
class RuleState:
    """
    Tracks the runtime state of a scaling rule.

    Attributes:
        is_active: Whether the rule's action is currently in effect.
        last_activation: Simulation time of last activation.
        trigger_start: When the trigger condition first became true.
        deescalation_start: When de-escalation conditions first became true.
        activation_count: Number of times this rule has been activated.
    """
    is_active: bool = False
    last_activation: float = -9999.0
    trigger_start: Optional[float] = None
    deescalation_start: Optional[float] = None
    activation_count: int = 0


@dataclass
class ScalingEvent:
    """
    Records a capacity scaling event for results tracking.

    Attributes:
        time: Simulation time of the event.
        rule_name: Name of the rule that triggered.
        action_type: Type of action taken.
        resource: Resource affected.
        old_capacity: Capacity before the change.
        new_capacity: Capacity after the change.
        trigger_value: The utilisation/queue value that triggered the event.
        direction: "scale_up" or "scale_down".
    """
    time: float
    rule_name: str
    action_type: str
    resource: str
    old_capacity: int
    new_capacity: int
    trigger_value: float
    direction: str  # "scale_up" or "scale_down"


@dataclass
class OPELStatus:
    """
    Tracks current OPEL level and time spent at each level.

    Attributes:
        current_level: Current OPEL level (1-4).
        level_start_time: When the current level started.
        time_at_level: Accumulated time at each OPEL level.
        transitions: Number of OPEL level changes.
        peak_level: Highest OPEL level reached.
    """
    current_level: OPELLevel = OPELLevel.OPEL_1
    level_start_time: float = 0.0
    time_at_level: Dict[OPELLevel, float] = field(default_factory=lambda: {
        OPELLevel.OPEL_1: 0.0,
        OPELLevel.OPEL_2: 0.0,
        OPELLevel.OPEL_3: 0.0,
        OPELLevel.OPEL_4: 0.0,
    })
    transitions: int = 0
    peak_level: OPELLevel = OPELLevel.OPEL_1


def evaluate_trigger(
    trigger: ScalingTrigger,
    resources: Dict[str, DynamicCapacityResource],
    current_time: float
) -> tuple[bool, float]:
    """
    Evaluate whether a trigger condition is met.

    Args:
        trigger: The trigger to evaluate.
        resources: Dictionary of available resources.
        current_time: Current simulation time.

    Returns:
        Tuple of (triggered: bool, current_value: float).
    """
    if trigger.trigger_type == ScalingTriggerType.UTILIZATION_ABOVE:
        resource = resources.get(trigger.resource)
        if resource and resource.capacity > 0:
            utilisation = resource.count / resource.capacity
            return utilisation >= trigger.threshold, utilisation
        elif resource:
            # Zero capacity - consider fully utilised
            return True, 1.0

    elif trigger.trigger_type == ScalingTriggerType.UTILIZATION_BELOW:
        resource = resources.get(trigger.resource)
        if resource and resource.capacity > 0:
            utilisation = resource.count / resource.capacity
            return utilisation <= trigger.threshold, utilisation
        elif resource:
            return False, 1.0

    elif trigger.trigger_type == ScalingTriggerType.QUEUE_LENGTH_ABOVE:
        resource = resources.get(trigger.resource)
        if resource:
            queue_len = len(resource.queue)
            return queue_len >= trigger.threshold, float(queue_len)

    elif trigger.trigger_type == ScalingTriggerType.TIME_OF_DAY:
        # Time of day in minutes from midnight (simulation time % 1440)
        time_of_day = current_time % 1440
        if trigger.start_time is not None and trigger.end_time is not None:
            in_window = trigger.start_time <= time_of_day < trigger.end_time
            return in_window, time_of_day

    elif trigger.trigger_type == ScalingTriggerType.SIMULATION_TIME:
        return current_time >= trigger.threshold, current_time

    return False, 0.0


def execute_action(
    action: ScalingAction,
    resources: Dict[str, DynamicCapacityResource],
    discharge_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute a scaling action.

    Args:
        action: The action to execute.
        resources: Dictionary of available resources.
        discharge_manager: Optional discharge manager for LoS adjustments.

    Returns:
        Dictionary with action details for logging.
    """
    result = {
        "resource": action.resource,
        "action": action.action_type.value,
        "old": 0,
        "new": 0,
        "delta": 0,
    }

    resource = resources.get(action.resource)

    if action.action_type == ScalingActionType.ADD_CAPACITY and resource:
        old_capacity = resource.capacity
        added = resource.add_capacity(action.magnitude, reason=f"scaling_{action.action_type.value}")
        result.update({
            "old": old_capacity,
            "new": resource.capacity,
            "delta": added
        })

    elif action.action_type == ScalingActionType.REMOVE_CAPACITY and resource:
        old_capacity = resource.capacity
        removed = resource.remove_capacity(action.magnitude, graceful=True, reason=f"scaling_{action.action_type.value}")
        result.update({
            "old": old_capacity,
            "new": resource.capacity,
            "delta": -removed
        })

    elif action.action_type == ScalingActionType.ACCELERATE_DISCHARGE:
        if discharge_manager:
            discharge_manager.activate_acceleration(action.los_reduction_pct)
        result.update({
            "los_reduction_pct": action.los_reduction_pct
        })

    elif action.action_type == ScalingActionType.ACTIVATE_DISCHARGE_LOUNGE:
        if discharge_manager:
            discharge_manager.activate_lounge()
        result.update({
            "lounge_activated": True
        })

    elif action.action_type == ScalingActionType.DIVERT_ARRIVALS:
        # Diversion is handled by the arrival generator checking this state
        result.update({
            "diversion_rate": action.diversion_rate,
            "diversion_priorities": action.diversion_priorities
        })

    return result


def _get_reverse_action(action: ScalingAction) -> ScalingAction:
    """
    Create the reverse action for de-escalation.

    Args:
        action: The original action to reverse.

    Returns:
        A new ScalingAction that reverses the original.
    """
    reverse = deepcopy(action)

    if action.action_type == ScalingActionType.ADD_CAPACITY:
        reverse.action_type = ScalingActionType.REMOVE_CAPACITY
    elif action.action_type == ScalingActionType.REMOVE_CAPACITY:
        reverse.action_type = ScalingActionType.ADD_CAPACITY

    return reverse


def determine_opel_level(
    resources: Dict[str, DynamicCapacityResource],
    config: CapacityScalingConfig
) -> OPELLevel:
    """
    Determine current OPEL level based on resource utilisation.

    Args:
        resources: Dictionary of resources to check.
        config: Capacity scaling configuration with OPEL thresholds.

    Returns:
        Current OPEL level.
    """
    opel_config = config.opel_config

    # Get ED and Ward utilisation
    ed_util = 0.0
    ward_util = 0.0

    ed_resource = resources.get("ed_bays")
    if ed_resource and ed_resource.capacity > 0:
        ed_util = ed_resource.count / ed_resource.capacity

    ward_resource = resources.get("ward_beds")
    if ward_resource and ward_resource.capacity > 0:
        ward_util = ward_resource.count / ward_resource.capacity

    # Check from highest to lowest
    if ed_util >= opel_config.opel_4_ed_threshold or ward_util >= opel_config.opel_4_ward_threshold:
        return OPELLevel.OPEL_4
    elif ed_util >= opel_config.opel_3_ed_threshold or ward_util >= opel_config.opel_3_ward_threshold:
        return OPELLevel.OPEL_3
    elif ed_util >= opel_config.opel_2_ed_threshold or ward_util >= opel_config.opel_2_ward_threshold:
        return OPELLevel.OPEL_2
    else:
        return OPELLevel.OPEL_1


class ScalingMonitor:
    """
    Manages capacity scaling monitoring and actions.

    This class encapsulates all scaling state and provides the monitor process
    that runs in the background checking triggers and executing actions.
    """

    def __init__(
        self,
        env: simpy.Environment,
        resources: Dict[str, DynamicCapacityResource],
        config: CapacityScalingConfig,
        discharge_manager: Optional[Any] = None
    ):
        """
        Initialize the scaling monitor.

        Args:
            env: SimPy environment.
            resources: Dictionary of dynamic resources to monitor.
            config: Capacity scaling configuration.
            discharge_manager: Optional discharge manager for LoS adjustments.
        """
        self.env = env
        self.resources = resources
        self.config = config
        self.discharge_manager = discharge_manager

        # Build combined rules list (custom + OPEL-generated)
        self.rules: List[ScalingRule] = list(config.rules)
        if config.opel_config.enabled:
            self.rules.extend(create_opel_rules(config.opel_config))

        # Initialize rule states
        self.rule_states: Dict[str, RuleState] = {
            rule.name: RuleState() for rule in self.rules
        }

        # OPEL tracking
        self.opel_status = OPELStatus()

        # Event log
        self.scaling_events: List[ScalingEvent] = []

        # Diversion state
        self.diversion_active = False
        self.diversion_rate = 0.0
        self.diversion_priorities: List[str] = []

    def run(self):
        """
        Generator function for the monitoring process.

        Yields SimPy events and should be started with env.process().
        """
        while True:
            yield self.env.timeout(self.config.evaluation_interval_mins)

            # Update OPEL status
            self._update_opel_status()

            # Process each rule
            for rule in self.rules:
                if not rule.enabled:
                    continue

                self._process_rule(rule)

    def _update_opel_status(self):
        """Update OPEL level tracking."""
        if not self.config.opel_config.enabled:
            return

        new_level = determine_opel_level(self.resources, self.config)

        if new_level != self.opel_status.current_level:
            # Record time at previous level
            time_at_prev = self.env.now - self.opel_status.level_start_time
            self.opel_status.time_at_level[self.opel_status.current_level] += time_at_prev

            # Update to new level
            self.opel_status.current_level = new_level
            self.opel_status.level_start_time = self.env.now
            self.opel_status.transitions += 1

            # Track peak
            if new_level.value > self.opel_status.peak_level.value:
                self.opel_status.peak_level = new_level

    def _process_rule(self, rule: ScalingRule):
        """Process a single scaling rule."""
        state = self.rule_states[rule.name]

        # Check max activations
        if rule.max_activations > 0 and state.activation_count >= rule.max_activations:
            return

        # Check cooldown
        if self.env.now - state.last_activation < rule.trigger.cooldown_mins:
            return

        # Evaluate trigger
        triggered, trigger_value = evaluate_trigger(
            rule.trigger, self.resources, self.env.now
        )

        if triggered and not state.is_active:
            # Check sustain period
            if state.trigger_start is None:
                state.trigger_start = self.env.now
            elif self.env.now - state.trigger_start >= rule.trigger.sustain_mins:
                # Execute action
                result = execute_action(
                    rule.action, self.resources, self.discharge_manager
                )

                # Update state
                state.is_active = True
                state.last_activation = self.env.now
                state.activation_count += 1
                state.trigger_start = None

                # Track diversion state
                if rule.action.action_type == ScalingActionType.DIVERT_ARRIVALS:
                    self.diversion_active = True
                    self.diversion_rate = rule.action.diversion_rate
                    self.diversion_priorities = rule.action.diversion_priorities

                # Record event
                self.scaling_events.append(ScalingEvent(
                    time=self.env.now,
                    rule_name=rule.name,
                    action_type=rule.action.action_type.value,
                    resource=rule.action.resource,
                    old_capacity=result.get("old", 0),
                    new_capacity=result.get("new", 0),
                    trigger_value=trigger_value,
                    direction="scale_up"
                ))

        elif not triggered:
            state.trigger_start = None

            # Check de-escalation
            if state.is_active and rule.auto_deescalate:
                should_deescalate = False

                if rule.deescalation_threshold is not None:
                    _, current_value = evaluate_trigger(
                        rule.trigger, self.resources, self.env.now
                    )
                    should_deescalate = current_value < rule.deescalation_threshold
                else:
                    should_deescalate = True

                if should_deescalate:
                    if state.deescalation_start is None:
                        state.deescalation_start = self.env.now
                    elif self.env.now - state.deescalation_start >= rule.deescalation_delay_mins:
                        # Execute reverse action
                        reverse_action = _get_reverse_action(rule.action)
                        result = execute_action(
                            reverse_action, self.resources, self.discharge_manager
                        )

                        # Update state
                        state.is_active = False
                        state.deescalation_start = None

                        # Clear diversion if this was a diversion rule
                        if rule.action.action_type == ScalingActionType.DIVERT_ARRIVALS:
                            self.diversion_active = False
                            self.diversion_rate = 0.0
                            self.diversion_priorities = []

                        # Deactivate discharge acceleration if applicable
                        if rule.action.action_type == ScalingActionType.ACCELERATE_DISCHARGE:
                            if self.discharge_manager:
                                self.discharge_manager.deactivate_acceleration()

                        # Record event
                        _, current_value = evaluate_trigger(
                            rule.trigger, self.resources, self.env.now
                        )
                        self.scaling_events.append(ScalingEvent(
                            time=self.env.now,
                            rule_name=rule.name,
                            action_type=reverse_action.action_type.value,
                            resource=reverse_action.resource,
                            old_capacity=result.get("old", 0),
                            new_capacity=result.get("new", 0),
                            trigger_value=current_value,
                            direction="scale_down"
                        ))
                else:
                    state.deescalation_start = None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated scaling metrics.

        Returns:
            Dictionary with scaling statistics.
        """
        # Finalize OPEL time at current level
        if self.config.opel_config.enabled:
            time_at_current = self.env.now - self.opel_status.level_start_time
            final_times = dict(self.opel_status.time_at_level)
            final_times[self.opel_status.current_level] += time_at_current
        else:
            final_times = {}

        scale_ups = sum(1 for e in self.scaling_events if e.direction == "scale_up")
        scale_downs = sum(1 for e in self.scaling_events if e.direction == "scale_down")

        # Calculate rule-level metrics
        rule_metrics = {}
        for rule_name, state in self.rule_states.items():
            rule_metrics[rule_name] = {
                "activations": state.activation_count,
                "is_active": state.is_active,
            }

        return {
            "total_scale_up_events": scale_ups,
            "total_scale_down_events": scale_downs,
            "total_scaling_events": len(self.scaling_events),
            "opel_transitions": self.opel_status.transitions if self.config.opel_config.enabled else 0,
            "opel_peak_level": self.opel_status.peak_level.value if self.config.opel_config.enabled else 1,
            "opel_time_at_level": final_times,
            "rule_metrics": rule_metrics,
            "events": self.scaling_events,
        }

    def should_divert(self, priority: str) -> bool:
        """
        Check if an arrival should be diverted.

        Args:
            priority: Priority level of the arriving patient (e.g., "P3").

        Returns:
            True if the patient should be diverted.
        """
        if not self.diversion_active:
            return False

        if priority not in self.diversion_priorities:
            return False

        # Use environment's random state if available, otherwise simple check
        # In practice, this should use the scenario's RNG
        import random
        return random.random() < self.diversion_rate


def capacity_scaling_monitor(
    env: simpy.Environment,
    resources: Dict[str, DynamicCapacityResource],
    config: CapacityScalingConfig,
    results_collector: Optional[Any] = None,
    discharge_manager: Optional[Any] = None
) -> ScalingMonitor:
    """
    Factory function to create and start a scaling monitor.

    Args:
        env: SimPy environment.
        resources: Dictionary of dynamic resources.
        config: Capacity scaling configuration.
        results_collector: Optional results collector for event recording.
        discharge_manager: Optional discharge manager.

    Returns:
        The ScalingMonitor instance (also starts the process).
    """
    monitor = ScalingMonitor(
        env=env,
        resources=resources,
        config=config,
        discharge_manager=discharge_manager
    )

    # Start the monitor process
    env.process(monitor.run())

    return monitor
