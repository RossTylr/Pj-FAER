"""Parameter-Metric Linkage Knowledge Base.

This module defines the known relationships between simulation parameters
and their expected impact on metrics. This enables:

1. **Guided Parameter Selection**: Suggest relevant parameters for a given metric
2. **Impact Prediction**: Estimate expected effect direction/magnitude
3. **Sensitivity Guidance**: Recommend which parameters to sweep first
4. **Insight Contextualization**: Explain why a metric changed

The knowledge base is built from:
- Queuing theory principles (utilization → wait times)
- NHS operational standards (4-hour standard, handover targets)
- Clinical evidence (ICU occupancy → mortality)
- DES best practices (warm-up, run length effects)

Example usage:
    from faer.agents.parameter_metrics import (
        get_affected_metrics,
        get_influencing_parameters,
        get_parameter_guidance,
        PARAMETER_METRIC_MAP,
    )

    # What metrics does n_ed_bays affect?
    metrics = get_affected_metrics("n_ed_bays")
    # ['util_ed_bays', 'mean_treatment_wait', 'p_delay', ...]

    # What parameters influence mean_treatment_wait?
    params = get_influencing_parameters("mean_treatment_wait")
    # ['n_ed_bays', 'n_triage', 'demand_multiplier', ...]

    # Get guidance for a parameter
    guidance = get_parameter_guidance("n_ed_bays")
    # ParameterGuidance with expected effects, clinical context, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EffectDirection(Enum):
    """Direction of parameter's effect on a metric.

    INCREASE: Increasing parameter increases metric value
    DECREASE: Increasing parameter decreases metric value
    NONLINEAR: Relationship is non-monotonic (e.g., utilization ceiling effects)
    INDIRECT: Effect is mediated through other resources
    """
    INCREASE = "increase"
    DECREASE = "decrease"
    NONLINEAR = "nonlinear"
    INDIRECT = "indirect"


class EffectMagnitude(Enum):
    """Expected magnitude of effect.

    Based on queuing theory and empirical simulation results.
    """
    STRONG = "strong"      # Elasticity > 0.5 (parameter is primary driver)
    MODERATE = "moderate"  # Elasticity 0.2-0.5 (noticeable effect)
    WEAK = "weak"          # Elasticity < 0.2 (secondary influence)
    THRESHOLD = "threshold"  # Effect only appears beyond certain values
    NONLINEAR = "nonlinear"  # Non-monotonic relationship (e.g., queuing theory exponential)
    INDIRECT = "indirect"  # Effect mediated through other resources


class MetricCategory(Enum):
    """Categories of metrics for grouping and display."""
    FLOW = "Patient Flow"
    UTILIZATION = "Resource Utilization"
    WAIT_TIME = "Wait Times"
    HANDOVER = "Handover & Transport"
    DOWNSTREAM = "Downstream Care"
    AEROMED = "Aeromedical"
    CAPACITY_SCALING = "Capacity Scaling"


@dataclass(frozen=True)
class ParameterEffect:
    """Describes how a parameter affects a specific metric.

    Attributes:
        metric: Name of affected metric
        direction: Expected direction of effect
        magnitude: Expected magnitude of effect
        explanation: Brief explanation of the relationship
        clinical_note: Optional clinical context
        queuing_theory_basis: Optional theoretical grounding
    """
    metric: str
    direction: EffectDirection
    magnitude: EffectMagnitude
    explanation: str
    clinical_note: str = ""
    queuing_theory_basis: str = ""


@dataclass
class ParameterGuidance:
    """Comprehensive guidance for a parameter.

    Provides all information needed to understand a parameter's
    role in the system and guide experimentation.

    Attributes:
        parameter: Parameter name (dot notation for nested)
        display_name: Human-readable name
        description: What this parameter represents
        affected_metrics: List of ParameterEffect for each affected metric
        primary_metrics: Metrics most directly affected (for quick reference)
        typical_range: Tuple of (min, max) typical values
        clinical_context: NHS/clinical relevance
        diminishing_returns_threshold: Value beyond which adding more has little effect
        interaction_notes: Known interactions with other parameters
    """
    parameter: str
    display_name: str
    description: str
    affected_metrics: list[ParameterEffect]
    primary_metrics: list[str]
    typical_range: tuple[float, float]
    clinical_context: str
    diminishing_returns_threshold: Optional[float] = None
    interaction_notes: list[str] = field(default_factory=list)


# ============ PARAMETER KNOWLEDGE BASE ============

PARAMETER_GUIDANCE: dict[str, ParameterGuidance] = {
    # === FRONT DOOR RESOURCES ===
    "n_ed_bays": ParameterGuidance(
        parameter="n_ed_bays",
        display_name="ED Bays",
        description="Number of treatment bays in the Emergency Department",
        affected_metrics=[
            ParameterEffect(
                metric="util_ed_bays",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More bays = lower utilization at same demand",
                queuing_theory_basis="Utilization = λ/(μ·c) decreases with capacity c",
            ),
            ParameterEffect(
                metric="mean_treatment_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More bays reduce queue for treatment",
                queuing_theory_basis="M/M/c queue: wait time decreases exponentially as c increases near ρ=1",
            ),
            ParameterEffect(
                metric="p_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More bays = more likely immediate treatment",
            ),
            ParameterEffect(
                metric="mean_boarding_time",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="More ED capacity allows faster processing, reducing boarding",
                clinical_note="Boarding primarily driven by downstream capacity",
            ),
            ParameterEffect(
                metric="mean_handover_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="ED saturation causes handover delays; more bays helps",
            ),
        ],
        primary_metrics=["util_ed_bays", "mean_treatment_wait", "p_delay"],
        typical_range=(10, 40),
        clinical_context="NHS Type 1 EDs typically have 15-40 bays depending on catchment",
        diminishing_returns_threshold=35,
        interaction_notes=[
            "Interacts with n_triage: fast triage with few bays creates bottleneck",
            "Downstream blocking (ward/ITU full) can negate ED bay additions",
        ],
    ),

    "n_triage": ParameterGuidance(
        parameter="n_triage",
        display_name="Triage Clinicians",
        description="Number of triage nurses/clinicians assessing incoming patients",
        affected_metrics=[
            ParameterEffect(
                metric="util_triage",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More clinicians = lower utilization",
            ),
            ParameterEffect(
                metric="mean_triage_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More clinicians process patients faster",
            ),
            ParameterEffect(
                metric="mean_treatment_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.WEAK,
                explanation="Faster triage gets patients to treatment sooner",
                clinical_note="Only helps if ED bays are available",
            ),
        ],
        primary_metrics=["util_triage", "mean_triage_wait"],
        typical_range=(1, 6),
        clinical_context="Triage is typically 2-5 minutes; bottleneck rare unless understaffed",
        diminishing_returns_threshold=4,
        interaction_notes=[
            "Adding triage without ED bays just moves the queue",
        ],
    ),

    "n_handover_bays": ParameterGuidance(
        parameter="n_handover_bays",
        display_name="Handover Bays",
        description="Physical spaces for ambulance crew handover to ED staff",
        affected_metrics=[
            ParameterEffect(
                metric="util_handover",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More bays = lower handover utilization",
            ),
            ParameterEffect(
                metric="mean_handover_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More physical space reduces crew wait times",
                clinical_note="NHS England target: handover within 15 minutes",
            ),
            ParameterEffect(
                metric="max_handover_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="Peak delays reduced with more capacity",
            ),
        ],
        primary_metrics=["mean_handover_delay", "util_handover"],
        typical_range=(2, 8),
        clinical_context="Ambulance handover delays impact 999 response times in community",
        diminishing_returns_threshold=6,
        interaction_notes=[
            "Handover delays often caused by ED saturation, not bay count",
            "Adding bays doesn't help if ED can't accept patients",
        ],
    ),

    # === DEMAND PARAMETERS ===
    "demand_multiplier": ParameterGuidance(
        parameter="demand_multiplier",
        display_name="Demand Level",
        description="Multiplier on baseline arrival rate (1.0 = baseline)",
        affected_metrics=[
            ParameterEffect(
                metric="arrivals",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="Direct linear scaling of patient volume",
            ),
            ParameterEffect(
                metric="p_delay",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More patients = higher probability of waiting",
                queuing_theory_basis="Delay probability increases sharply as ρ→1",
            ),
            ParameterEffect(
                metric="util_ed_bays",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More arrivals = higher resource utilization",
            ),
            ParameterEffect(
                metric="mean_treatment_wait",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.NONLINEAR,
                explanation="Wait times increase exponentially as utilization approaches 100%",
                queuing_theory_basis="Kingman's formula: E[W] ∝ ρ/(1-ρ)",
            ),
            ParameterEffect(
                metric="mean_boarding_time",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Higher demand stresses downstream capacity",
            ),
        ],
        primary_metrics=["arrivals", "p_delay", "util_ed_bays", "mean_treatment_wait"],
        typical_range=(0.5, 2.0),
        clinical_context="Winter surge typically 1.2-1.5x; major incident 2-3x",
        interaction_notes=[
            "High demand exposes downstream bottlenecks (ward, ITU)",
            "Test at 1.5x to assess winter resilience",
        ],
    ),

    "p_resus": ParameterGuidance(
        parameter="p_resus",
        display_name="Resus Proportion",
        description="Proportion of arrivals requiring resuscitation (P1 acuity)",
        affected_metrics=[
            ParameterEffect(
                metric="util_ed_bays",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="P1 patients have longer treatment times",
                clinical_note="Resus cases typically 60-120 min vs 30-45 min for majors",
            ),
            ParameterEffect(
                metric="mean_treatment_wait",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Complex cases slow throughput",
            ),
            ParameterEffect(
                metric="util_itu",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="P1 patients more likely to need ITU admission",
            ),
            ParameterEffect(
                metric="util_theatre",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Trauma resus often requires emergency surgery",
            ),
        ],
        primary_metrics=["util_itu", "util_ed_bays"],
        typical_range=(0.02, 0.15),
        clinical_context="Typical UK ED: 3-8% P1. Trauma centre: 8-15%",
        interaction_notes=[
            "High P1 with limited ITU causes ED boarding for critical patients",
            "CBRN/MCI events can spike P1 proportion to 20-40%",
        ],
    ),

    # === DOWNSTREAM RESOURCES ===
    "itu_config.capacity": ParameterGuidance(
        parameter="itu_config.capacity",
        display_name="ITU Beds",
        description="Number of Intensive Treatment Unit beds",
        affected_metrics=[
            ParameterEffect(
                metric="util_itu",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More beds = lower utilization",
            ),
            ParameterEffect(
                metric="mean_itu_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More beds reduce queue for critical care",
            ),
            ParameterEffect(
                metric="mean_boarding_time",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Critical patients board in ED when ITU full",
                clinical_note="P1 boarding is particularly harmful",
            ),
            ParameterEffect(
                metric="util_theatre",
                direction=EffectDirection.INDIRECT,
                magnitude=EffectMagnitude.MODERATE,
                explanation="ITU shortage can block post-op cases, reducing theatre throughput",
            ),
        ],
        primary_metrics=["util_itu", "mean_boarding_time"],
        typical_range=(4, 20),
        clinical_context="ITU at >90% utilization associated with increased mortality",
        diminishing_returns_threshold=15,
        interaction_notes=[
            "ITU-theatre interaction: major surgery requires ITU post-op",
            "Step-down beds can increase effective ITU capacity",
        ],
    ),

    "ward_config.capacity": ParameterGuidance(
        parameter="ward_config.capacity",
        display_name="Ward Beds",
        description="Number of general inpatient ward beds",
        affected_metrics=[
            ParameterEffect(
                metric="util_ward",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More beds = lower ward utilization",
            ),
            ParameterEffect(
                metric="mean_boarding_time",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="Ward availability is primary driver of ED boarding",
                clinical_note="Classic NHS flow problem: ward full → ED boards",
            ),
            ParameterEffect(
                metric="mean_handover_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.INDIRECT,
                explanation="Ward → ED → Handover cascade when beds blocked",
            ),
            ParameterEffect(
                metric="p_delay",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Downstream flow improves front-door performance",
            ),
        ],
        primary_metrics=["util_ward", "mean_boarding_time"],
        typical_range=(20, 80),
        clinical_context="Ward occupancy >92% associated with increased mortality risk",
        diminishing_returns_threshold=60,
        interaction_notes=[
            "Discharge processes often more impactful than bed count",
            "Length of stay reduction can increase effective capacity",
        ],
    ),

    "theatre_config.n_tables": ParameterGuidance(
        parameter="theatre_config.n_tables",
        display_name="Theatre Tables",
        description="Number of operating theatre tables available",
        affected_metrics=[
            ParameterEffect(
                metric="util_theatre",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More tables = lower utilization",
            ),
            ParameterEffect(
                metric="mean_theatre_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="Emergency surgical cases get faster access",
                clinical_note="Time-critical for trauma, AAA, appendix",
            ),
        ],
        primary_metrics=["util_theatre", "mean_theatre_wait"],
        typical_range=(1, 6),
        clinical_context="Emergency theatres should have dedicated capacity, not share with electives",
        diminishing_returns_threshold=4,
        interaction_notes=[
            "Requires matching anaesthetic and surgical staff",
            "ITU capacity limits can block post-op, reducing effective theatre capacity",
        ],
    ),

    # === CAPACITY SCALING PARAMETERS ===
    "capacity_scaling.opel_config.opel_3_ed_threshold": ParameterGuidance(
        parameter="capacity_scaling.opel_config.opel_3_ed_threshold",
        display_name="OPEL 3 ED Trigger",
        description="ED utilization threshold that triggers OPEL 3 (severe pressure)",
        affected_metrics=[
            ParameterEffect(
                metric="opel_peak_level",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Lower threshold = earlier escalation = more time at OPEL 3",
            ),
            ParameterEffect(
                metric="pct_time_at_surge",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="Lower threshold triggers surge more often",
            ),
            ParameterEffect(
                metric="mean_treatment_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Earlier surge activation can prevent delays",
            ),
            ParameterEffect(
                metric="total_additional_bed_hours",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More time at surge = more surge bed-hours used",
            ),
        ],
        primary_metrics=["pct_time_at_surge", "opel_peak_level"],
        typical_range=(0.80, 0.95),
        clinical_context="OPEL 3 typically triggers at 85-90% occupancy",
        interaction_notes=[
            "Too low = constant escalation (alert fatigue)",
            "Too high = escalation comes too late to help",
            "Balance between responsiveness and resource efficiency",
        ],
    ),

    "capacity_scaling.opel_config.opel_3_surge_beds": ParameterGuidance(
        parameter="capacity_scaling.opel_config.opel_3_surge_beds",
        display_name="OPEL 3 Surge Beds",
        description="Additional beds activated at OPEL 3",
        affected_metrics=[
            ParameterEffect(
                metric="mean_boarding_time",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Surge capacity provides buffer for boarders",
            ),
            ParameterEffect(
                metric="total_additional_bed_hours",
                direction=EffectDirection.INCREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More surge beds = more additional bed-hours when activated",
            ),
            ParameterEffect(
                metric="opel_peak_level",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.MODERATE,
                explanation="Adequate surge may prevent escalation to OPEL 4",
            ),
        ],
        primary_metrics=["mean_boarding_time", "total_additional_bed_hours"],
        typical_range=(2, 10),
        clinical_context="Surge beds typically from corridor spaces, discharge lounge",
        interaction_notes=[
            "Requires staff to be effective - unfunded surge is unsafe",
            "Interaction with discharge acceleration compounds effect",
        ],
    ),

    # === AEROMED PARAMETERS ===
    "aeromed_config.hems.slots_per_day": ParameterGuidance(
        parameter="aeromed_config.hems.slots_per_day",
        display_name="HEMS Slots/Day",
        description="Number of HEMS evacuation slots available per day",
        affected_metrics=[
            ParameterEffect(
                metric="aeromed_slots_missed",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More slots = fewer missed evacuation opportunities",
            ),
            ParameterEffect(
                metric="mean_aeromed_slot_wait",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.STRONG,
                explanation="More slots reduces wait for next available slot",
            ),
            ParameterEffect(
                metric="util_ward",
                direction=EffectDirection.DECREASE,
                magnitude=EffectMagnitude.WEAK,
                explanation="Evacuations free up ward beds",
                clinical_note="Only relevant if eligible patients are waiting",
            ),
        ],
        primary_metrics=["aeromed_slots_missed", "mean_aeromed_slot_wait"],
        typical_range=(2, 12),
        clinical_context="HEMS slots limited by aircraft availability, crew duty hours, weather",
        interaction_notes=[
            "Effect depends on having evacuable patients ready",
            "Weather/night restrictions may reduce effective slots",
        ],
    ),
}


# ============ METRIC KNOWLEDGE BASE ============

@dataclass(frozen=True)
class MetricInfo:
    """Information about a metric.

    Attributes:
        name: Metric identifier
        display_name: Human-readable name
        category: MetricCategory for grouping
        description: What this metric measures
        unit: Unit of measurement (min, %, count, etc.)
        clinical_interpretation: How to interpret the value clinically
        nhs_target: Optional NHS standard or target
        good_direction: "lower" or "higher" - which direction is better
        warning_threshold: Value at which this metric becomes concerning
        critical_threshold: Value at which this metric is critical
    """
    name: str
    display_name: str
    category: MetricCategory
    description: str
    unit: str
    clinical_interpretation: str
    good_direction: str  # "lower" or "higher"
    nhs_target: Optional[str] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


METRIC_INFO: dict[str, MetricInfo] = {
    # Flow metrics
    "p_delay": MetricInfo(
        name="p_delay",
        display_name="P(Delay)",
        category=MetricCategory.FLOW,
        description="Proportion of patients who waited for treatment",
        unit="%",
        clinical_interpretation="Higher values indicate capacity insufficient for demand",
        good_direction="lower",
        warning_threshold=0.30,
        critical_threshold=0.50,
    ),
    "mean_treatment_wait": MetricInfo(
        name="mean_treatment_wait",
        display_name="Mean Treatment Wait",
        category=MetricCategory.WAIT_TIME,
        description="Average time from triage to start of treatment",
        unit="min",
        clinical_interpretation="Longer waits increase deterioration risk",
        good_direction="lower",
        warning_threshold=60.0,
        critical_threshold=120.0,
    ),
    "p95_treatment_wait": MetricInfo(
        name="p95_treatment_wait",
        display_name="P95 Treatment Wait",
        category=MetricCategory.WAIT_TIME,
        description="95th percentile of treatment wait (worst 5%)",
        unit="min",
        clinical_interpretation="Captures worst-case patient experience",
        good_direction="lower",
        nhs_target="<240 min (4-hour standard)",
        warning_threshold=180.0,
        critical_threshold=240.0,
    ),
    "mean_system_time": MetricInfo(
        name="mean_system_time",
        display_name="Mean System Time",
        category=MetricCategory.FLOW,
        description="Average total time from arrival to departure",
        unit="min",
        clinical_interpretation="Overall efficiency of patient journey",
        good_direction="lower",
        warning_threshold=180.0,
        critical_threshold=300.0,
    ),
    # Utilization metrics
    "util_ed_bays": MetricInfo(
        name="util_ed_bays",
        display_name="ED Bay Utilisation",
        category=MetricCategory.UTILIZATION,
        description="Proportion of ED bays occupied (time-weighted)",
        unit="%",
        clinical_interpretation=">85% indicates strain, >95% is crisis",
        good_direction="lower",
        warning_threshold=0.85,
        critical_threshold=0.95,
    ),
    "util_itu": MetricInfo(
        name="util_itu",
        display_name="ITU Utilisation",
        category=MetricCategory.UTILIZATION,
        description="Proportion of ITU beds occupied",
        unit="%",
        clinical_interpretation=">90% associated with increased mortality",
        good_direction="lower",
        nhs_target="<85% for safe operation",
        warning_threshold=0.85,
        critical_threshold=0.90,
    ),
    "util_ward": MetricInfo(
        name="util_ward",
        display_name="Ward Utilisation",
        category=MetricCategory.UTILIZATION,
        description="Proportion of ward beds occupied",
        unit="%",
        clinical_interpretation=">92% causes flow problems and mortality risk",
        good_direction="lower",
        warning_threshold=0.90,
        critical_threshold=0.95,
    ),
    # Handover metrics
    "mean_handover_delay": MetricInfo(
        name="mean_handover_delay",
        display_name="Mean Handover Delay",
        category=MetricCategory.HANDOVER,
        description="Average ambulance crew wait time at handover",
        unit="min",
        clinical_interpretation="Delays reduce ambulance availability for 999 calls",
        good_direction="lower",
        nhs_target="<15 min",
        warning_threshold=15.0,
        critical_threshold=30.0,
    ),
    # Boarding metrics
    "mean_boarding_time": MetricInfo(
        name="mean_boarding_time",
        display_name="Mean Boarding Time",
        category=MetricCategory.DOWNSTREAM,
        description="Average time admitted patients wait in ED for bed",
        unit="min",
        clinical_interpretation="Boarding ties up ED capacity and harms boarding patients",
        good_direction="lower",
        warning_threshold=30.0,
        critical_threshold=60.0,
    ),
    # Capacity scaling metrics
    "opel_peak_level": MetricInfo(
        name="opel_peak_level",
        display_name="Peak OPEL Level",
        category=MetricCategory.CAPACITY_SCALING,
        description="Highest OPEL escalation level reached (1-4)",
        unit="level",
        clinical_interpretation="OPEL 4 = critical, requires system-level response",
        good_direction="lower",
        warning_threshold=3.0,
        critical_threshold=4.0,
    ),
    "pct_time_at_surge": MetricInfo(
        name="pct_time_at_surge",
        display_name="% Time at Surge",
        category=MetricCategory.CAPACITY_SCALING,
        description="Percentage of simulation time with surge capacity active",
        unit="%",
        clinical_interpretation="High values indicate baseline capacity insufficient",
        good_direction="lower",
        warning_threshold=20.0,
        critical_threshold=50.0,
    ),
    "patients_diverted": MetricInfo(
        name="patients_diverted",
        display_name="Patients Diverted",
        category=MetricCategory.CAPACITY_SCALING,
        description="Number of patients sent to alternative facility",
        unit="count",
        clinical_interpretation="Diversion indicates system at breaking point",
        good_direction="lower",
        warning_threshold=1.0,
        critical_threshold=5.0,
    ),
    "aeromed_slots_missed": MetricInfo(
        name="aeromed_slots_missed",
        display_name="Aeromed Slots Missed",
        category=MetricCategory.AEROMED,
        description="Evacuation slots that couldn't be used",
        unit="count",
        clinical_interpretation="Missed slots extend patient stay",
        good_direction="lower",
        warning_threshold=1.0,
        critical_threshold=3.0,
    ),
}


# ============ LOOKUP FUNCTIONS ============

def get_affected_metrics(parameter: str) -> list[str]:
    """Get list of metrics affected by a parameter.

    Args:
        parameter: Parameter name (dot notation for nested)

    Returns:
        List of metric names, sorted by effect magnitude
    """
    guidance = PARAMETER_GUIDANCE.get(parameter)
    if not guidance:
        return []

    # Sort by magnitude (strong first)
    magnitude_order = {
        EffectMagnitude.STRONG: 0,
        EffectMagnitude.MODERATE: 1,
        EffectMagnitude.NONLINEAR: 2,
        EffectMagnitude.WEAK: 3,
        EffectMagnitude.THRESHOLD: 4,
    }

    sorted_effects = sorted(
        guidance.affected_metrics,
        key=lambda e: magnitude_order.get(e.magnitude, 5)
    )

    return [e.metric for e in sorted_effects]


def get_influencing_parameters(metric: str) -> list[str]:
    """Get parameters that influence a metric.

    Args:
        metric: Metric name

    Returns:
        List of parameter names, sorted by effect magnitude
    """
    influencers = []

    for param_name, guidance in PARAMETER_GUIDANCE.items():
        for effect in guidance.affected_metrics:
            if effect.metric == metric:
                influencers.append((param_name, effect.magnitude))
                break

    # Sort by magnitude (strong first)
    magnitude_order = {
        EffectMagnitude.STRONG: 0,
        EffectMagnitude.MODERATE: 1,
        EffectMagnitude.NONLINEAR: 2,
        EffectMagnitude.WEAK: 3,
        EffectMagnitude.THRESHOLD: 4,
    }

    influencers.sort(key=lambda x: magnitude_order.get(x[1], 5))
    return [p for p, _ in influencers]


def get_parameter_guidance(parameter: str) -> Optional[ParameterGuidance]:
    """Get comprehensive guidance for a parameter.

    Args:
        parameter: Parameter name

    Returns:
        ParameterGuidance or None if not found
    """
    return PARAMETER_GUIDANCE.get(parameter)


def get_metric_info(metric: str) -> Optional[MetricInfo]:
    """Get information about a metric.

    Args:
        metric: Metric name

    Returns:
        MetricInfo or None if not found
    """
    return METRIC_INFO.get(metric)


def get_recommended_parameters_for_metric(metric: str, top_n: int = 3) -> list[dict]:
    """Get recommended parameters to tune for a specific metric.

    Args:
        metric: Metric to improve
        top_n: Number of top recommendations

    Returns:
        List of dicts with parameter, display_name, direction, explanation
    """
    params = get_influencing_parameters(metric)[:top_n]
    results = []

    for param in params:
        guidance = get_parameter_guidance(param)
        if not guidance:
            continue

        # Find the effect for this metric
        for effect in guidance.affected_metrics:
            if effect.metric == metric:
                results.append({
                    "parameter": param,
                    "display_name": guidance.display_name,
                    "direction": effect.direction.value,
                    "magnitude": effect.magnitude.value,
                    "explanation": effect.explanation,
                    "typical_range": guidance.typical_range,
                })
                break

    return results


def get_sweep_suggestions(metric: str) -> str:
    """Get natural language suggestions for sensitivity analysis.

    Args:
        metric: Metric user wants to optimize

    Returns:
        Markdown-formatted suggestion text
    """
    params = get_recommended_parameters_for_metric(metric, top_n=3)
    metric_info = get_metric_info(metric)

    if not params:
        return f"No parameter guidance available for {metric}."

    lines = [f"**To improve {metric_info.display_name if metric_info else metric}:**\n"]

    for i, p in enumerate(params, 1):
        direction_verb = "increase" if p["direction"] == "decrease" else "decrease"
        if metric_info and metric_info.good_direction == "higher":
            direction_verb = "decrease" if p["direction"] == "decrease" else "increase"

        lines.append(
            f"{i}. **{p['display_name']}** ({p['magnitude']} effect)\n"
            f"   - {direction_verb.capitalize()} this parameter\n"
            f"   - {p['explanation']}\n"
            f"   - Try range: {p['typical_range'][0]} - {p['typical_range'][1]}\n"
        )

    return "\n".join(lines)
