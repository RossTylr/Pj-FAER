"""
Capacity Scaling Configuration for Pj-FAER.

Defines data classes for dynamic capacity scaling including:
- Scaling triggers (utilisation, queue length, time-based)
- Scaling actions (add/remove capacity, discharge acceleration)
- OPEL framework integration (NHS England escalation levels)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ScalingTriggerType(Enum):
    """Types of triggers that can activate scaling rules."""
    UTILIZATION_ABOVE = "utilization_above"
    UTILIZATION_BELOW = "utilization_below"
    QUEUE_LENGTH_ABOVE = "queue_length_above"
    TIME_OF_DAY = "time_of_day"
    SIMULATION_TIME = "simulation_time"


class ScalingActionType(Enum):
    """Types of actions that can be taken when a trigger fires."""
    ADD_CAPACITY = "add_capacity"
    REMOVE_CAPACITY = "remove_capacity"
    ACCELERATE_DISCHARGE = "accelerate_discharge"
    ACTIVATE_DISCHARGE_LOUNGE = "activate_discharge_lounge"
    DIVERT_ARRIVALS = "divert_arrivals"


class OPELLevel(Enum):
    """
    NHS Operational Pressures Escalation Levels.

    OPEL is the NHS England standard framework for communicating and
    responding to capacity pressure across acute trusts.
    """
    OPEL_1 = 1  # Normal ops. Demand within capacity. No escalation needed.
    OPEL_2 = 2  # Moderate pressure. Local actions to address. Focus on flow & discharge.
    OPEL_3 = 3  # Severe pressure. Surge capacity opened. Discharge push active.
    OPEL_4 = 4  # Crisis. Unable to deliver safe care. Mutual aid, diverts.


@dataclass
class ScalingTrigger:
    """
    Defines when a scaling action should be triggered.

    Attributes:
        trigger_type: The type of condition to monitor.
        resource: The resource to monitor (e.g., "ed_bays", "ward_beds").
        threshold: The threshold value (utilisation %, queue length, or time).
        cooldown_mins: Minimum time between activations to prevent oscillation.
        sustain_mins: How long condition must hold before triggering.
        start_time: For TIME_OF_DAY triggers, start of active window (mins from midnight).
        end_time: For TIME_OF_DAY triggers, end of active window (mins from midnight).
    """
    trigger_type: ScalingTriggerType
    resource: str
    threshold: float
    cooldown_mins: float = 60.0
    sustain_mins: float = 15.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ScalingAction:
    """
    Defines what happens when a trigger fires.

    Attributes:
        action_type: The type of action to take.
        resource: The resource to act upon.
        magnitude: Number of beds/staff to add/remove.
        los_reduction_pct: For discharge acceleration, reduce LoS by this percentage.
        discharge_probability_boost: Increase early discharge probability.
        diversion_rate: Fraction of arrivals to divert (0.0 to 1.0).
        diversion_priorities: Which priority levels to divert (e.g., ["P4", "P3"]).
    """
    action_type: ScalingActionType
    resource: str
    magnitude: int = 0
    los_reduction_pct: float = 0.0
    discharge_probability_boost: float = 0.0
    diversion_rate: float = 0.0
    diversion_priorities: List[str] = field(default_factory=lambda: ["P4", "P3"])


@dataclass
class ScalingRule:
    """
    A complete scaling rule combining trigger, action, and de-escalation.

    Attributes:
        name: Human-readable name for the rule.
        trigger: The condition that activates this rule.
        action: What to do when triggered.
        auto_deescalate: Whether to automatically reverse when condition clears.
        deescalation_threshold: Threshold below which to de-escalate.
        deescalation_delay_mins: Wait time before de-escalating.
        max_activations: Maximum times this rule can fire (-1 = unlimited).
        enabled: Whether this rule is active.
    """
    name: str
    trigger: ScalingTrigger
    action: ScalingAction
    auto_deescalate: bool = True
    deescalation_threshold: Optional[float] = None
    deescalation_delay_mins: float = 30.0
    max_activations: int = -1
    enabled: bool = True


@dataclass
class OPELConfig:
    """
    OPEL-based escalation configuration.

    NHS trusts use OPEL 1-4 to communicate pressure status:
    - OPEL 1: Normal operations (<85% utilisation)
    - OPEL 2: Moderate pressure (85-90%) - focus on flow & discharge
    - OPEL 3: Severe pressure (90-95%) - surge capacity, discharge push
    - OPEL 4: Crisis (>95%) - mutual aid, diverts, unable to deliver safe care

    Thresholds can be customised per trust based on local capacity.
    """
    enabled: bool = False

    # OPEL 2 thresholds (85% = "functionally full")
    opel_2_ed_threshold: float = 0.85
    opel_2_ward_threshold: float = 0.85

    # OPEL 3 thresholds (90% = corridor care likely, 4hr target at risk)
    opel_3_ed_threshold: float = 0.90
    opel_3_ward_threshold: float = 0.90

    # OPEL 4 thresholds (95% = unsafe staffing ratios, care compromised)
    opel_4_ed_threshold: float = 0.95
    opel_4_ward_threshold: float = 0.95

    # OPEL 3 actions
    opel_3_surge_beds: int = 5
    opel_3_los_reduction_pct: float = 10.0
    opel_3_enable_lounge: bool = True

    # OPEL 4 actions
    opel_4_surge_beds: int = 10
    opel_4_los_reduction_pct: float = 20.0
    opel_4_enable_divert: bool = True
    opel_4_divert_priorities: List[str] = field(default_factory=lambda: ["P4", "P3"])


@dataclass
class CapacityScalingConfig:
    """
    Master configuration for all capacity scaling behaviors.

    Attributes:
        enabled: Master switch for capacity scaling.
        rules: List of custom scaling rules.
        opel_config: OPEL framework configuration.
        evaluation_interval_mins: How often to check triggers.
        max_simultaneous_actions: Limit on concurrent scaling actions.
        discharge_lounge_capacity: Number of spaces in discharge lounge.
        discharge_lounge_max_wait_mins: Maximum time in lounge before departure.
        baseline_ed_bays: Baseline ED capacity for calculating surge delta.
        baseline_ward_beds: Baseline ward capacity.
        baseline_itu_beds: Baseline ITU capacity.
    """
    enabled: bool = False
    rules: List[ScalingRule] = field(default_factory=list)
    opel_config: OPELConfig = field(default_factory=OPELConfig)

    # Global settings
    evaluation_interval_mins: float = 5.0
    max_simultaneous_actions: int = 3

    # Discharge lounge
    discharge_lounge_capacity: int = 10
    discharge_lounge_max_wait_mins: float = 120.0

    # Baseline capacity (for calculating deltas)
    baseline_ed_bays: int = 20
    baseline_ward_beds: int = 30
    baseline_itu_beds: int = 6


def create_opel_rules(config: OPELConfig) -> List[ScalingRule]:
    """
    Generate ScalingRule objects from OPEL configuration.

    Converts the NHS OPEL framework thresholds and actions into
    concrete scaling rules that the simulation engine can execute.

    Args:
        config: OPEL configuration with thresholds and actions.

    Returns:
        List of ScalingRule objects implementing OPEL responses.
    """
    rules = []

    if not config.enabled:
        return rules

    # OPEL 3 - ED Surge
    rules.append(ScalingRule(
        name="OPEL 3: ED Surge",
        trigger=ScalingTrigger(
            trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
            resource="ed_bays",
            threshold=config.opel_3_ed_threshold,
            sustain_mins=15.0,
            cooldown_mins=60.0
        ),
        action=ScalingAction(
            action_type=ScalingActionType.ADD_CAPACITY,
            resource="ed_bays",
            magnitude=config.opel_3_surge_beds
        ),
        auto_deescalate=True,
        deescalation_threshold=config.opel_2_ed_threshold - 0.05,  # 80%
        deescalation_delay_mins=30.0
    ))

    # OPEL 3 - Discharge Acceleration
    rules.append(ScalingRule(
        name="OPEL 3: Discharge Push",
        trigger=ScalingTrigger(
            trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
            resource="ward_beds",
            threshold=config.opel_3_ward_threshold,
            sustain_mins=15.0,
            cooldown_mins=60.0
        ),
        action=ScalingAction(
            action_type=ScalingActionType.ACCELERATE_DISCHARGE,
            resource="ward_beds",
            los_reduction_pct=config.opel_3_los_reduction_pct
        ),
        auto_deescalate=True,
        deescalation_threshold=config.opel_2_ward_threshold - 0.05,  # 80%
        deescalation_delay_mins=30.0
    ))

    # OPEL 3 - Discharge Lounge
    if config.opel_3_enable_lounge:
        rules.append(ScalingRule(
            name="OPEL 3: Discharge Lounge",
            trigger=ScalingTrigger(
                trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                resource="ward_beds",
                threshold=config.opel_3_ward_threshold,
                sustain_mins=15.0,
                cooldown_mins=60.0
            ),
            action=ScalingAction(
                action_type=ScalingActionType.ACTIVATE_DISCHARGE_LOUNGE,
                resource="discharge_lounge"
            ),
            auto_deescalate=True,
            deescalation_threshold=config.opel_2_ward_threshold - 0.05,
            deescalation_delay_mins=30.0
        ))

    # OPEL 4 - Full Surge
    rules.append(ScalingRule(
        name="OPEL 4: Full Surge",
        trigger=ScalingTrigger(
            trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
            resource="ed_bays",
            threshold=config.opel_4_ed_threshold,
            sustain_mins=10.0,  # Faster response at OPEL 4
            cooldown_mins=60.0
        ),
        action=ScalingAction(
            action_type=ScalingActionType.ADD_CAPACITY,
            resource="ed_bays",
            magnitude=config.opel_4_surge_beds
        ),
        auto_deescalate=True,
        deescalation_threshold=config.opel_3_ed_threshold - 0.05,  # 85%
        deescalation_delay_mins=60.0  # Longer wait before de-escalating from OPEL 4
    ))

    # OPEL 4 - Aggressive Discharge
    rules.append(ScalingRule(
        name="OPEL 4: Aggressive Discharge",
        trigger=ScalingTrigger(
            trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
            resource="ward_beds",
            threshold=config.opel_4_ward_threshold,
            sustain_mins=10.0,
            cooldown_mins=60.0
        ),
        action=ScalingAction(
            action_type=ScalingActionType.ACCELERATE_DISCHARGE,
            resource="ward_beds",
            los_reduction_pct=config.opel_4_los_reduction_pct
        ),
        auto_deescalate=True,
        deescalation_threshold=config.opel_3_ward_threshold - 0.05,  # 85%
        deescalation_delay_mins=60.0
    ))

    # OPEL 4 - Ambulance Diversion
    if config.opel_4_enable_divert:
        rules.append(ScalingRule(
            name="OPEL 4: Ambulance Divert",
            trigger=ScalingTrigger(
                trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                resource="ed_bays",
                threshold=config.opel_4_ed_threshold,
                sustain_mins=10.0,
                cooldown_mins=120.0  # Diversion is serious - longer cooldown
            ),
            action=ScalingAction(
                action_type=ScalingActionType.DIVERT_ARRIVALS,
                resource="arrivals",
                diversion_rate=0.3,  # Divert 30% of eligible arrivals
                diversion_priorities=config.opel_4_divert_priorities
            ),
            auto_deescalate=True,
            deescalation_threshold=config.opel_3_ed_threshold - 0.05,  # 85%
            deescalation_delay_mins=60.0
        ))

    return rules
