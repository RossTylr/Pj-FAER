"""Heuristic Clinical Shadow Agent.

This module implements a rule-based clinical shadow agent that applies
NHS and clinical guideline thresholds to simulation metrics. It serves as:

1. A working baseline before LLM integration
2. A testable, deterministic reference implementation
3. A fallback when LLM services are unavailable

The agent uses configurable threshold rules that can be customized
for different clinical contexts or jurisdictions.

Example usage:
    from faer.agents.shadow import HeuristicShadowAgent

    agent = HeuristicShadowAgent()
    insights = agent.analyze(metrics_summary)

    for insight in insights:
        if insight.severity == Severity.CRITICAL:
            print(f"ALERT: {insight.title}")
            print(f"  {insight.message}")
            print(f"  Recommendation: {insight.recommendation}")
"""

from dataclasses import dataclass
from typing import Iterator

from .interface import (
    ClinicalInsight,
    InsightCategory,
    MetricsSummary,
    Severity,
)


@dataclass
class ClinicalThreshold:
    """Configurable threshold for a clinical rule.

    Defines a single-metric threshold that triggers an insight
    when breached. Multiple thresholds can be combined for
    comprehensive coverage.

    Attributes:
        metric: Name of metric to check (must match MetricsSummary field)
        threshold: Numeric threshold value
        operator: Comparison operator ("gt", "lt", "gte", "lte")
        severity: Insight severity if threshold breached
        category: Insight category for filtering
        title: Short headline (≤80 chars)
        message_template: Format string with {value} placeholder
        recommendation: Actionable guidance
    """

    metric: str
    threshold: float
    operator: str  # "gt", "lt", "gte", "lte"
    severity: Severity
    category: InsightCategory
    title: str
    message_template: str
    recommendation: str


# NHS/Clinical guideline thresholds
# These are based on NHS Constitution standards and clinical best practices
NHS_THRESHOLDS = [
    # 4-hour treatment standard (NHS Constitution)
    ClinicalThreshold(
        metric="p95_treatment_wait",
        threshold=240.0,
        operator="gt",
        severity=Severity.CRITICAL,
        category=InsightCategory.WAIT_TIME,
        title="4-Hour Standard Breach",
        message_template=(
            "P95 treatment wait of {value:.0f} minutes exceeds NHS 4-hour standard. "
            "Patients waiting this long face increased risk of deterioration, "
            "sepsis progression, and adverse outcomes. The 4-hour target is a "
            "clinical safety standard, not just an operational metric."
        ),
        recommendation=(
            "Immediate capacity review required. Consider activating surge "
            "protocol, calling in additional staff, or initiating divert."
        ),
    ),
    # ITU capacity threshold (evidence-based 90% danger zone)
    ClinicalThreshold(
        metric="util_itu",
        threshold=0.90,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.CAPACITY,
        title="ITU Capacity Critical",
        message_template=(
            "ITU utilization at {value:.0%} indicates near-saturation. "
            "At this level, new critical admissions will experience delays, "
            "post-operative patients may not have beds, and ED boarding is "
            "likely to increase. Research shows mortality increases when "
            "ICU occupancy exceeds 90%."
        ),
        recommendation=(
            "Prepare ITU step-down transfers. Alert bed management team. "
            "Review elective surgical schedule for cases requiring ITU post-op."
        ),
    ),
    # ED bay saturation threshold
    ClinicalThreshold(
        metric="util_ed_bays",
        threshold=0.85,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.CAPACITY,
        title="ED Bay Saturation",
        message_template=(
            "ED bay utilization at {value:.0%} approaching saturation. "
            "Expect triage bottlenecks and ambulance handover delays. "
            "Patient flow through department will degrade, increasing "
            "wait times for all acuity levels."
        ),
        recommendation=(
            "Activate surge protocol. Consider see-and-treat pathway for "
            "minors. Expedite discharges and transfers. Review bay turnover."
        ),
    ),
    # Majority patients delayed
    ClinicalThreshold(
        metric="p_delay",
        threshold=0.50,
        operator="gt",
        severity=Severity.MEDIUM,
        category=InsightCategory.FLOW_BOTTLENECK,
        title="Majority Patients Delayed",
        message_template=(
            "{value:.0%} of patients experienced treatment delays. "
            "This indicates systemic flow issues rather than isolated incidents. "
            "When more than half of patients wait for treatment, the department "
            "is operating beyond sustainable capacity."
        ),
        recommendation=(
            "Review triage efficiency and treatment bay turnover. "
            "Consider streaming pathways to separate patient flows."
        ),
    ),
    # Ambulance handover delays (NHS England 15-min target)
    ClinicalThreshold(
        metric="mean_handover_delay",
        threshold=30.0,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.HANDOVER,
        title="Ambulance Handover Delays",
        message_template=(
            "Mean ambulance handover delay of {value:.0f} minutes. "
            "Crews are unable to respond to new 999 calls while waiting. "
            "Community response times will be impacted, potentially affecting "
            "out-of-hospital cardiac arrest and stroke outcomes."
        ),
        recommendation=(
            "Prioritize handover bay clearance. Consider corridor care protocol "
            "if clinically appropriate. Alert ambulance service of delays."
        ),
    ),
    # Extended ED boarding
    ClinicalThreshold(
        metric="mean_boarding_time",
        threshold=60.0,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.BOARDING,
        title="Extended ED Boarding",
        message_template=(
            "Patients boarding in ED for average {value:.0f} minutes awaiting "
            "downstream beds. This ties up ED capacity, delays new arrivals, "
            "and is associated with worse outcomes for boarding patients who "
            "require inpatient-level care."
        ),
        recommendation=(
            "Escalate bed management. Review discharge timing on wards. "
            "Consider opening overflow capacity."
        ),
    ),
    # Aeromedical slots missed
    ClinicalThreshold(
        metric="aeromed_slots_missed",
        threshold=1.0,
        operator="gte",
        severity=Severity.HIGH,
        category=InsightCategory.AEROMEDICAL,
        title="Aeromedical Slots Missed",
        message_template=(
            "{value:.0f} aeromedical evacuation slots missed. "
            "Patients requiring strategic evacuation are experiencing delays, "
            "consuming ward bed-days that could otherwise be freed. Each missed "
            "slot extends patient stay and reduces throughput capacity."
        ),
        recommendation=(
            "Review aeromed scheduling and patient prioritization. "
            "Consider additional slots or alternative transport."
        ),
    ),
    # Theatre utilization approaching capacity
    ClinicalThreshold(
        metric="util_theatre",
        threshold=0.85,
        operator="gt",
        severity=Severity.MEDIUM,
        category=InsightCategory.CAPACITY,
        title="Theatre Utilization High",
        message_template=(
            "Theatre utilization at {value:.0%}. Emergency surgical cases may "
            "experience delays as capacity is limited. Consider impact on "
            "time-sensitive procedures."
        ),
        recommendation=(
            "Review theatre schedule for flexibility. Ensure emergency "
            "slot availability. Consider postponing non-urgent electives."
        ),
    ),
    # Ward capacity approaching saturation
    ClinicalThreshold(
        metric="util_ward",
        threshold=0.90,
        operator="gt",
        severity=Severity.MEDIUM,
        category=InsightCategory.CAPACITY,
        title="Ward Capacity Near Saturation",
        message_template=(
            "Ward utilization at {value:.0%}. Limited discharge capacity "
            "will cause upstream blocking in ED and post-operative areas."
        ),
        recommendation=(
            "Accelerate discharge planning. Review patients medically fit "
            "for discharge. Engage discharge coordination team."
        ),
    ),
]


class HeuristicShadowAgent:
    """Rule-based clinical shadow agent using configurable thresholds.

    This agent applies clinical guidelines as threshold rules against
    simulation metrics. It's deterministic and testable, serving as
    the foundation before LLM integration.

    The agent evaluates:
    1. Single-metric threshold rules (NHS standards)
    2. Compound rules (multi-metric risk patterns)

    Attributes:
        thresholds: List of ClinicalThreshold rules to apply
        name: Agent identifier ("heuristic_shadow")
        description: Human-readable description

    Example:
        agent = HeuristicShadowAgent()
        insights = agent.analyze(metrics_summary)

        # With custom thresholds
        custom_thresholds = [
            ClinicalThreshold(
                metric="p95_treatment_wait",
                threshold=180.0,  # Stricter than NHS
                ...
            )
        ]
        strict_agent = HeuristicShadowAgent(thresholds=custom_thresholds)
    """

    def __init__(self, thresholds: list[ClinicalThreshold] | None = None):
        """Initialize with threshold rules.

        Args:
            thresholds: Custom thresholds. Defaults to NHS_THRESHOLDS.
        """
        self.thresholds = thresholds if thresholds is not None else NHS_THRESHOLDS

    @property
    def name(self) -> str:
        return "heuristic_shadow"

    @property
    def description(self) -> str:
        return "Rule-based clinical risk detector using NHS/clinical thresholds"

    def analyze(self, metrics: MetricsSummary) -> list[ClinicalInsight]:
        """Apply all threshold rules and return triggered insights.

        Args:
            metrics: Standardized simulation output summary

        Returns:
            List of ClinicalInsight objects, sorted by severity
            (CRITICAL first, then HIGH, MEDIUM, LOW, INFO)
        """
        insights = list(self._evaluate_thresholds(metrics))
        insights.extend(self._evaluate_compound_rules(metrics))

        # Sort by severity (CRITICAL first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        insights.sort(key=lambda x: severity_order[x.severity])

        return insights

    def _evaluate_thresholds(
        self, metrics: MetricsSummary
    ) -> Iterator[ClinicalInsight]:
        """Evaluate single-metric threshold rules.

        Iterates through configured thresholds and yields insights
        for any breached rules.

        Args:
            metrics: Metrics to evaluate

        Yields:
            ClinicalInsight for each triggered threshold
        """
        for rule in self.thresholds:
            value = getattr(metrics, rule.metric, None)
            if value is None:
                continue

            triggered = self._check_threshold(value, rule.threshold, rule.operator)
            if triggered:
                yield ClinicalInsight(
                    severity=rule.severity,
                    category=rule.category,
                    title=rule.title,
                    message=rule.message_template.format(value=value),
                    impact_metric=rule.metric,
                    evidence={rule.metric: value, "threshold": rule.threshold},
                    recommendation=rule.recommendation,
                    source_agent=self.name,
                )

    def _evaluate_compound_rules(
        self, metrics: MetricsSummary
    ) -> Iterator[ClinicalInsight]:
        """Evaluate multi-metric compound rules.

        These rules detect risk patterns that require multiple metrics
        to be in certain states simultaneously. They capture emergent
        risks that single-metric rules would miss.

        Args:
            metrics: Metrics to evaluate

        Yields:
            ClinicalInsight for each triggered compound rule
        """
        # Guard against division by zero
        total_arrivals = max(metrics.arrivals, 1)

        # Compound rule 1: High acuity + long waits = severe risk
        # When critical patients (P1) represent >8% of arrivals AND
        # mean treatment wait exceeds 60 minutes, mortality risk increases
        resus_count = metrics.arrivals_by_priority.get("P1", 0)
        resus_ratio = resus_count / total_arrivals

        if resus_ratio > 0.08 and metrics.mean_treatment_wait > 60:
            yield ClinicalInsight(
                severity=Severity.CRITICAL,
                category=InsightCategory.PATIENT_SAFETY,
                title="High Acuity with Treatment Delays",
                message=(
                    f"Resuscitation cases represent {resus_ratio:.1%} of arrivals "
                    f"while mean treatment wait is {metrics.mean_treatment_wait:.0f} "
                    "minutes. This combination significantly elevates mortality risk "
                    "for critical patients. P1 patients deteriorate rapidly when "
                    "treatment is delayed."
                ),
                impact_metric="mortality_risk",
                evidence={
                    "resus_ratio": resus_ratio,
                    "mean_treatment_wait": metrics.mean_treatment_wait,
                    "P1_arrivals": resus_count,
                },
                recommendation=(
                    "Immediate senior clinical review. Consider divert protocol. "
                    "Ensure P1 patients are not waiting for non-critical resources."
                ),
                source_agent=self.name,
            )

        # Compound rule 2: ITU near capacity + theatre active = downstream gridlock
        # When ITU is >85% full and theatre is >70% utilized, post-op patients
        # may not have ITU beds, causing theatre cancellations and ED cascade
        if metrics.util_itu > 0.85 and metrics.util_theatre > 0.70:
            yield ClinicalInsight(
                severity=Severity.HIGH,
                category=InsightCategory.DOWNSTREAM,
                title="Downstream Gridlock Risk",
                message=(
                    f"ITU at {metrics.util_itu:.0%} utilization with theatre at "
                    f"{metrics.util_theatre:.0%}. Post-operative patients requiring "
                    "ITU may not have beds available, causing theatre cancellations "
                    "and ED boarding cascade. This is a classic flow failure mode."
                ),
                impact_metric="flow_cascade",
                evidence={
                    "util_itu": metrics.util_itu,
                    "util_theatre": metrics.util_theatre,
                },
                recommendation=(
                    "Review elective theatre schedule. Prepare ITU step-downs. "
                    "Alert surgical teams to potential case delays."
                ),
                source_agent=self.name,
            )

        # Compound rule 3: Low triage util but high treatment wait = processing bottleneck
        # When triage is underutilized but treatment waits are high, the bottleneck
        # is in treatment bays, not assessment - focus capacity there
        if metrics.util_triage < 0.50 and metrics.mean_treatment_wait > 90:
            yield ClinicalInsight(
                severity=Severity.MEDIUM,
                category=InsightCategory.FLOW_BOTTLENECK,
                title="Treatment Bottleneck Identified",
                message=(
                    f"Triage utilization is only {metrics.util_triage:.0%} but "
                    f"treatment wait averages {metrics.mean_treatment_wait:.0f} minutes. "
                    "The bottleneck is clearly in treatment bays, not triage assessment. "
                    "Adding triage capacity would not improve patient flow."
                ),
                impact_metric="bottleneck_location",
                evidence={
                    "util_triage": metrics.util_triage,
                    "mean_treatment_wait": metrics.mean_treatment_wait,
                },
                recommendation=(
                    "Focus capacity increase on ED treatment bays, not triage. "
                    "Review bay turnover time and discharge processes."
                ),
                source_agent=self.name,
            )

        # Compound rule 4: High boarding + high handover delay = ambulance crisis
        # When ED boarding backs up AND handover delays are high, the system
        # is failing both admitted patients and incoming ambulances
        if metrics.mean_boarding_time > 30 and metrics.mean_handover_delay > 20:
            yield ClinicalInsight(
                severity=Severity.HIGH,
                category=InsightCategory.COMPOUND_RISK,
                title="Dual Blocking Crisis",
                message=(
                    f"Both ED boarding ({metrics.mean_boarding_time:.0f} min average) "
                    f"and ambulance handover ({metrics.mean_handover_delay:.0f} min) "
                    "are elevated. This indicates full-system blocking from downstream "
                    "to pre-hospital care. Both admitted patients and new arrivals "
                    "are affected."
                ),
                impact_metric="system_blocking",
                evidence={
                    "mean_boarding_time": metrics.mean_boarding_time,
                    "mean_handover_delay": metrics.mean_handover_delay,
                },
                recommendation=(
                    "Activate hospital-wide full capacity protocol. "
                    "Escalate to executive on-call. Consider declaring OPEL 4."
                ),
                source_agent=self.name,
            )

        # Compound rule 5: Ward full + aeromed slots available = evacuation opportunity
        # When ward is near capacity but aeromed is available, there may be
        # opportunity to free beds through strategic evacuation
        if (
            metrics.util_ward > 0.85
            and metrics.aeromed_total > 0
            and metrics.aeromed_slots_missed == 0
        ):
            yield ClinicalInsight(
                severity=Severity.INFO,
                category=InsightCategory.AEROMEDICAL,
                title="Aeromedical Capacity Available",
                message=(
                    f"Ward utilization at {metrics.util_ward:.0%} with aeromedical "
                    "evacuation slots available (no missed slots). Consider "
                    "prioritizing eligible patients for evacuation to free beds."
                ),
                impact_metric="bed_optimization",
                evidence={
                    "util_ward": metrics.util_ward,
                    "aeromed_total": metrics.aeromed_total,
                    "aeromed_slots_missed": metrics.aeromed_slots_missed,
                },
                recommendation=(
                    "Review patient list for aeromedical eligibility. "
                    "Coordinate with evacuation team to maximize throughput."
                ),
                source_agent=self.name,
            )

    @staticmethod
    def _check_threshold(value: float, threshold: float, operator: str) -> bool:
        """Check if value triggers threshold based on operator.

        Args:
            value: Actual metric value
            threshold: Threshold to compare against
            operator: Comparison operator ("gt", "lt", "gte", "lte")

        Returns:
            True if threshold is breached, False otherwise
        """
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
        }
        return ops.get(operator, lambda v, t: False)(value, threshold)

    def health_check(self) -> bool:
        """Heuristic agent is always healthy (no external dependencies).

        Returns:
            Always True for heuristic agent
        """
        return True
