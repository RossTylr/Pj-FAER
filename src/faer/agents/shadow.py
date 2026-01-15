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
    StructuralAssessment,
    PeakLoadAnalysis,
    ExpertPerspective,
    SystemEvaluation,
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
        soft_margin: Fractional buffer zone around threshold (e.g., 0.05 = 5%)
            where uncertainty messaging applies
        severity_when_uncertain: Optional downgraded severity when CI overlaps
            threshold. If None, uses original severity.
        uncertainty_message_template: Alternative message template when
            threshold breach is uncertain. If empty, uses message_template.
    """

    metric: str
    threshold: float
    operator: str  # "gt", "lt", "gte", "lte"
    severity: Severity
    category: InsightCategory
    title: str
    message_template: str
    recommendation: str
    # Uncertainty-aware fields (defaults for backward compatibility)
    soft_margin: float = 0.0
    severity_when_uncertain: Severity | None = None
    uncertainty_message_template: str = ""


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
        """Evaluate single-metric threshold rules with uncertainty awareness.

        Iterates through configured thresholds and yields insights
        for any breached rules. When confidence intervals are available,
        the insight includes confidence level and uncertainty notes.

        Args:
            metrics: Metrics to evaluate (may include ci_bounds)

        Yields:
            ClinicalInsight for each triggered threshold, with confidence info
        """
        for rule in self.thresholds:
            value = getattr(metrics, rule.metric, None)
            if value is None:
                continue

            # Get CI bounds if available
            ci_bounds = metrics.ci_bounds.get(rule.metric)
            if ci_bounds:
                ci_lower, ci_upper = ci_bounds
            else:
                # No CI available - use point estimate
                ci_lower, ci_upper = value, value

            # Check if point estimate triggers threshold
            triggered = self._check_threshold(value, rule.threshold, rule.operator)

            # Compute confidence and overlap
            confidence_level = self._compute_confidence_level(
                value, ci_lower, ci_upper, rule.threshold
            )
            threshold_overlap = self._ci_overlaps_threshold(
                ci_lower, ci_upper, rule.threshold, rule.operator, rule.soft_margin
            )

            if triggered:
                # Threshold breached - yield insight with confidence info
                yield self._create_insight_with_confidence(
                    rule=rule,
                    value=value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    confidence_level=confidence_level,
                    threshold_overlap=threshold_overlap,
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

    @staticmethod
    def _compute_confidence_level(
        mean: float,
        ci_lower: float,
        ci_upper: float,
        threshold: float,
    ) -> str:
        """Compute confidence level based on CI relationship to threshold.

        Args:
            mean: Point estimate of the metric
            ci_lower: Lower bound of confidence interval
            ci_upper: Upper bound of confidence interval
            threshold: Threshold value being evaluated

        Returns:
            "high", "medium", or "low" confidence level
        """
        # Single replication or no CI - low confidence
        if ci_lower == ci_upper:
            return "low"

        ci_width = ci_upper - ci_lower
        distance_to_threshold = abs(mean - threshold)

        # High confidence: threshold is well outside CI (distance > 2x CI width)
        if distance_to_threshold > 2 * ci_width:
            return "high"
        # Medium confidence: threshold is outside but close to CI
        elif distance_to_threshold > ci_width:
            return "medium"
        # Low confidence: threshold is within or near CI
        else:
            return "low"

    @staticmethod
    def _ci_overlaps_threshold(
        ci_lower: float,
        ci_upper: float,
        threshold: float,
        operator: str,
        soft_margin: float = 0.0,
    ) -> bool:
        """Check if confidence interval overlaps threshold boundary.

        Args:
            ci_lower: Lower bound of CI
            ci_upper: Upper bound of CI
            threshold: Threshold value
            operator: Comparison operator ("gt", "lt", "gte", "lte")
            soft_margin: Fractional buffer zone (e.g., 0.05 = 5%)

        Returns:
            True if CI spans the threshold zone, False otherwise
        """
        # Apply soft margin to create threshold zone
        margin = abs(threshold) * soft_margin if threshold != 0 else soft_margin
        threshold_low = threshold - margin
        threshold_high = threshold + margin

        # CI overlaps if it spans the threshold zone
        return ci_lower <= threshold_high and ci_upper >= threshold_low

    def _create_insight_with_confidence(
        self,
        rule: ClinicalThreshold,
        value: float,
        ci_lower: float,
        ci_upper: float,
        confidence_level: str,
        threshold_overlap: bool,
    ) -> ClinicalInsight:
        """Create a ClinicalInsight with uncertainty-aware fields populated.

        Args:
            rule: The threshold rule that triggered
            value: Current metric value
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            confidence_level: "high", "medium", or "low"
            threshold_overlap: Whether CI overlaps threshold

        Returns:
            ClinicalInsight with confidence fields populated
        """
        # Determine severity - may be downgraded if uncertain
        if threshold_overlap and rule.severity_when_uncertain is not None:
            severity = rule.severity_when_uncertain
        else:
            severity = rule.severity

        # Select message template
        if threshold_overlap and rule.uncertainty_message_template:
            message = rule.uncertainty_message_template.format(value=value)
        else:
            message = rule.message_template.format(value=value)

        # Build uncertainty note
        uncertainty_note = ""
        if threshold_overlap:
            uncertainty_note = (
                f"Note: The 95% confidence interval [{ci_lower:.2f}, {ci_upper:.2f}] "
                f"overlaps the threshold of {rule.threshold:.2f}. "
                f"More replications may improve confidence."
            )
        elif confidence_level == "low":
            uncertainty_note = (
                "Note: Limited replications may affect reliability of this alert."
            )

        # Build title - add uncertainty marker if needed
        title = rule.title
        if threshold_overlap and len(title) <= 68:  # Leave room for marker
            title = f"{title} (Uncertain)"

        return ClinicalInsight(
            severity=severity,
            category=rule.category,
            title=title,
            message=message,
            impact_metric=rule.metric,
            evidence={
                rule.metric: value,
                "threshold": rule.threshold,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            },
            recommendation=rule.recommendation,
            source_agent=self.name,
            confidence_level=confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            threshold_overlap=threshold_overlap,
            uncertainty_note=uncertainty_note,
        )

    def health_check(self) -> bool:
        """Heuristic agent is always healthy (no external dependencies).

        Returns:
            Always True for heuristic agent
        """
        return True

    # ============ SYSTEM EVALUATION METHODS ============

    def evaluate_system(self, metrics: MetricsSummary) -> SystemEvaluation:
        """Comprehensive system evaluation with multi-expert perspectives.

        Combines structural analysis, peak load assessment, and expert
        perspectives from multiple clinical and operational viewpoints.

        Args:
            metrics: Standardized simulation output summary

        Returns:
            SystemEvaluation with comprehensive analysis
        """
        structural = self._assess_structure(metrics)
        peak_load = self._analyze_peak_load(metrics)
        expert_perspectives = self._generate_expert_perspectives(metrics)

        # Determine overall status
        max_severity = Severity.INFO
        for perspective in expert_perspectives:
            if perspective.severity.value == "CRITICAL":
                max_severity = Severity.CRITICAL
                break
            elif perspective.severity.value == "HIGH" and max_severity.value != "CRITICAL":
                max_severity = Severity.HIGH
            elif perspective.severity.value == "MEDIUM" and max_severity.value not in ("CRITICAL", "HIGH"):
                max_severity = Severity.MEDIUM

        status_map = {
            Severity.CRITICAL: "critical",
            Severity.HIGH: "warning",
            Severity.MEDIUM: "caution",
            Severity.LOW: "normal",
            Severity.INFO: "normal",
        }
        overall_status = status_map[max_severity]

        # Generate executive summary
        summary = self._generate_executive_summary(
            metrics, structural, peak_load, expert_perspectives, overall_status
        )

        return SystemEvaluation(
            structural=structural,
            peak_load=peak_load,
            expert_perspectives=expert_perspectives,
            overall_system_status=overall_status,
            summary_text=summary,
        )

    def _assess_structure(self, metrics: MetricsSummary) -> StructuralAssessment:
        """Assess structural strengths and weaknesses of the system.

        Analyzes resource configuration and utilisation patterns to identify
        architectural strengths, vulnerabilities, and bottleneck cascades.

        Args:
            metrics: Metrics summary

        Returns:
            StructuralAssessment with strengths, weaknesses, and bottleneck chain
        """
        strengths = []
        weaknesses = []
        bottleneck_chain = []

        # Calculate headroom for each resource
        headroom = {
            "triage": 1.0 - metrics.util_triage,
            "ed_bays": 1.0 - metrics.util_ed_bays,
            "itu": 1.0 - metrics.util_itu,
            "ward": 1.0 - metrics.util_ward,
            "theatre": 1.0 - metrics.util_theatre,
        }

        # Identify bottleneck chain (ordered by utilisation, highest first)
        sorted_resources = sorted(headroom.items(), key=lambda x: x[1])
        bottleneck_chain = [r[0] for r in sorted_resources if r[1] < 0.30]

        # Structural strengths
        if metrics.util_triage < 0.70:
            strengths.append("Triage capacity adequate - patients assessed promptly")
        if metrics.p_delay < 0.30:
            strengths.append("Low delay probability - most patients treated immediately")
        if metrics.util_itu < 0.80 and metrics.util_ward < 0.80:
            strengths.append("Downstream capacity buffer - reduces ED boarding risk")
        if metrics.mean_system_time < 180:
            strengths.append("Good patient throughput - system time under 3 hours")
        if not bottleneck_chain:
            strengths.append("Balanced resource utilisation - no critical bottlenecks")

        # Structural weaknesses
        if metrics.util_ed_bays > 0.85:
            weaknesses.append("ED bay saturation risk - primary constraint on flow")
        if metrics.util_itu > 0.90:
            weaknesses.append("ITU near capacity - limits post-operative and critical admissions")
        if metrics.util_ward > 0.90:
            weaknesses.append("Ward saturation - causes upstream blocking")
        if metrics.mean_treatment_wait > 60:
            weaknesses.append("High treatment wait times - capacity/demand mismatch")
        if metrics.mean_boarding_time > 30:
            weaknesses.append("ED boarding occurring - downstream flow impaired")
        if metrics.mean_handover_delay > 15:
            weaknesses.append("Ambulance handover delays - impacts pre-hospital response")
        if len(bottleneck_chain) > 2:
            weaknesses.append("Multi-resource bottleneck cascade - systemic capacity issue")

        # Calculate resilience score (0-100)
        # Higher score = more able to absorb surge
        resilience_factors = [
            min(headroom["ed_bays"] * 50, 25),  # ED headroom (max 25)
            min(headroom["itu"] * 30, 15),       # ITU headroom (max 15)
            min(headroom["ward"] * 30, 15),      # Ward headroom (max 15)
            max(0, 15 - (metrics.mean_treatment_wait / 10)),  # Wait time penalty
            max(0, 15 - (metrics.p_delay * 30)),              # Delay probability penalty
            max(0, 15 - (metrics.mean_boarding_time / 6)),    # Boarding penalty
        ]
        resilience_score = max(0, min(100, sum(resilience_factors)))

        return StructuralAssessment(
            strengths=strengths,
            weaknesses=weaknesses,
            bottleneck_chain=bottleneck_chain,
            resilience_score=resilience_score,
            headroom_by_resource=headroom,
        )

    def _analyze_peak_load(self, metrics: MetricsSummary) -> PeakLoadAnalysis:
        """Analyze peak load and overloading patterns.

        Examines arrival patterns and capacity stress to identify surge
        conditions, bolus arrivals, and system recovery characteristics.

        Args:
            metrics: Metrics summary with raw data access

        Returns:
            PeakLoadAnalysis with peak/overload assessment
        """
        raw = metrics.raw_metrics

        # Get arrival data from raw metrics
        arrivals_list = raw.get("arrivals", [metrics.arrivals])
        mean_arrivals = sum(arrivals_list) / len(arrivals_list) if arrivals_list else metrics.arrivals

        # Estimate run length from scenario (default 8 hours)
        # Try to infer from metrics or use reasonable default
        run_length_hours = 8.0

        mean_arrival_rate = mean_arrivals / run_length_hours if run_length_hours > 0 else 0

        # Peak analysis - estimate from P95 system times and delay patterns
        # High P95 relative to mean indicates peaks
        p95_wait = metrics.p95_treatment_wait
        mean_wait = metrics.mean_treatment_wait
        peak_indicator = p95_wait / max(mean_wait, 1.0) if mean_wait > 0 else 1.0

        # Estimate peak arrival rate from variability
        peak_to_mean_ratio = max(1.0, min(3.0, peak_indicator))
        peak_arrival_rate = mean_arrival_rate * peak_to_mean_ratio

        # Time above capacity estimation
        # Based on delay probability and utilisation
        time_above_capacity = min(100, metrics.p_delay * 100 + max(0, (metrics.util_ed_bays - 0.85) * 200))

        # Bolus pattern detection
        # High peak-to-mean ratio with high P95 suggests bolus arrivals
        bolus_detected = peak_to_mean_ratio > 1.5 and metrics.p95_treatment_wait > 60

        # Surge headroom estimation (minutes of surge system can absorb)
        # Based on queue capacity and service rate
        ed_headroom = 1.0 - metrics.util_ed_bays
        estimated_surge_minutes = max(0, ed_headroom * 60 * 2)  # Rough estimate

        # Queue buildup rate (patients/hour during peak)
        queue_buildup = max(0, peak_arrival_rate - (mean_arrival_rate * (1 - metrics.p_delay)))

        return PeakLoadAnalysis(
            peak_arrival_rate=peak_arrival_rate,
            mean_arrival_rate=mean_arrival_rate,
            peak_to_mean_ratio=peak_to_mean_ratio,
            time_above_capacity_pct=time_above_capacity,
            bolus_pattern_detected=bolus_detected,
            estimated_surge_headroom=estimated_surge_minutes,
            queue_buildup_rate=queue_buildup,
        )

    def _generate_expert_perspectives(
        self, metrics: MetricsSummary
    ) -> list[ExpertPerspective]:
        """Generate analysis from multiple expert perspectives.

        Creates domain-specific assessments from clinical and operational
        viewpoints, each with their own focus areas and concerns.

        Args:
            metrics: Metrics summary

        Returns:
            List of ExpertPerspective objects
        """
        perspectives = []

        # EM Consultant Perspective
        perspectives.append(self._em_consultant_perspective(metrics))

        # Anaesthetist Perspective
        perspectives.append(self._anaesthetist_perspective(metrics))

        # Surgeon Perspective
        perspectives.append(self._surgeon_perspective(metrics))

        # CBRN Specialist (if relevant metrics present)
        if metrics.mean_treatment_wait > 0:  # Always include for completeness
            perspectives.append(self._cbrn_specialist_perspective(metrics))

        # NHS Health Data Scientist Perspective
        perspectives.append(self._data_scientist_perspective(metrics))

        # Paramedic/Pre-hospital Perspective
        perspectives.append(self._paramedic_perspective(metrics))

        return perspectives

    def _em_consultant_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """Emergency Medicine Consultant perspective.

        Focus: Triage priorities (P1-P4), overall ED flow, 4-hour standard.
        """
        concerns = []
        recommendations = []

        # P1 patient concerns
        p1_count = metrics.arrivals_by_priority.get("P1", 0)
        if p1_count > 0 and metrics.mean_treatment_wait > 30:
            concerns.append(
                f"P1 (immediate) patients may be waiting - {metrics.mean_treatment_wait:.0f}min mean wait"
            )
            recommendations.append("Ensure P1 patients bypass queue with dedicated resus pathway")

        # 4-hour standard
        if metrics.p95_treatment_wait > 180:
            concerns.append(
                f"P95 treatment wait {metrics.p95_treatment_wait:.0f}min threatens 4-hour standard"
            )
            recommendations.append("Review patient flow, consider streaming minors separately")

        # Boarding concerns
        if metrics.mean_boarding_time > 30:
            concerns.append(
                f"ED boarding average {metrics.mean_boarding_time:.0f}min reduces capacity for new patients"
            )
            recommendations.append("Escalate bed management, review discharge processes")

        # Delay probability
        if metrics.p_delay > 0.5:
            concerns.append(f"{metrics.p_delay:.0%} of patients delayed - unsustainable demand")

        # Severity assessment
        if metrics.p95_treatment_wait > 240 or metrics.mean_boarding_time > 60:
            severity = Severity.CRITICAL
            assessment = "ED under severe pressure. Patient safety risks present."
        elif metrics.p_delay > 0.5 or metrics.mean_treatment_wait > 60:
            severity = Severity.HIGH
            assessment = "ED flow compromised. Action needed to prevent deterioration."
        elif metrics.util_ed_bays > 0.80:
            severity = Severity.MEDIUM
            assessment = "ED operating near capacity. Monitor for deterioration."
        else:
            severity = Severity.INFO
            assessment = "ED operating within acceptable parameters."

        if not concerns:
            concerns.append("No immediate clinical concerns from ED perspective")

        return ExpertPerspective(
            expert_role="em_consultant",
            expert_title="Emergency Medicine Consultant",
            focus_area="Triage priorities, ED flow, clinical safety",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Continue current protocols"],
            key_metrics={
                "mean_treatment_wait": metrics.mean_treatment_wait,
                "p95_treatment_wait": metrics.p95_treatment_wait,
                "p_delay": metrics.p_delay,
                "util_ed_bays": metrics.util_ed_bays,
                "mean_boarding_time": metrics.mean_boarding_time,
            },
            severity=severity,
        )

    def _anaesthetist_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """Anaesthetist perspective.

        Focus: ITU pathways, P1 stabilisation, post-operative routing.
        """
        concerns = []
        recommendations = []

        # ITU capacity
        if metrics.util_itu > 0.90:
            concerns.append(
                f"ITU at {metrics.util_itu:.0%} - critical care capacity exhausted"
            )
            recommendations.append("Prepare step-down transfers, consider ITU admission criteria review")
        elif metrics.util_itu > 0.80:
            concerns.append(
                f"ITU at {metrics.util_itu:.0%} - limited surge capacity for new critical patients"
            )

        # Theatre-ITU flow
        if metrics.util_theatre > 0.70 and metrics.util_itu > 0.85:
            concerns.append(
                "Theatre-ITU bottleneck: post-op patients may not have ITU beds available"
            )
            recommendations.append("Review elective schedule, prioritise cases not requiring ITU post-op")

        # ITU wait times
        if metrics.mean_itu_wait > 30:
            concerns.append(
                f"ITU bed wait averaging {metrics.mean_itu_wait:.0f}min - delays critical transfers"
            )

        # P1 stabilisation concerns
        p1_arrivals = metrics.arrivals_by_priority.get("P1", 0)
        if p1_arrivals > 0:
            if metrics.mean_treatment_wait > 15:
                concerns.append(
                    "P1 stabilisation may be delayed - assess airway/critical care response times"
                )

        # Severity
        if metrics.util_itu > 0.95:
            severity = Severity.CRITICAL
            assessment = "Critical care capacity crisis. Post-op and ED critical patients at risk."
        elif metrics.util_itu > 0.85:
            severity = Severity.HIGH
            assessment = "Critical care under pressure. Limited capacity for new deteriorations."
        elif metrics.util_itu > 0.70:
            severity = Severity.MEDIUM
            assessment = "Critical care operating at moderate load. Monitor for surges."
        else:
            severity = Severity.INFO
            assessment = "Critical care capacity adequate for current demand."

        if not concerns:
            concerns.append("ITU capacity and critical care pathways operating normally")

        return ExpertPerspective(
            expert_role="anaesthetist",
            expert_title="Consultant Anaesthetist",
            focus_area="Critical care, ITU pathways, airway management",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Maintain current ITU protocols"],
            key_metrics={
                "util_itu": metrics.util_itu,
                "mean_itu_wait": metrics.mean_itu_wait,
                "util_theatre": metrics.util_theatre,
                "itu_admissions": metrics.itu_admissions,
            },
            severity=severity,
        )

    def _surgeon_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """Surgeon perspective.

        Focus: Theatre capacity, trauma cases, surgical pathway timing.
        """
        concerns = []
        recommendations = []

        # Theatre utilisation
        if metrics.util_theatre > 0.85:
            concerns.append(
                f"Theatre utilisation at {metrics.util_theatre:.0%} - emergency slot availability limited"
            )
            recommendations.append("Review emergency theatre scheduling, consider postponing non-urgent electives")
        elif metrics.util_theatre > 0.70:
            concerns.append(
                f"Theatre utilisation at {metrics.util_theatre:.0%} - monitor for emergency capacity"
            )

        # Theatre wait times
        if metrics.mean_theatre_wait > 60:
            concerns.append(
                f"Mean theatre wait {metrics.mean_theatre_wait:.0f}min - time-sensitive procedures at risk"
            )
            recommendations.append("Prioritise life-threatening surgical cases, consider 2nd emergency theatre")

        # Downstream blocking affecting theatre
        if metrics.util_itu > 0.90 and metrics.util_theatre > 0.50:
            concerns.append(
                "ITU saturation may force theatre cancellations for cases requiring ITU post-op"
            )

        # High acuity arrivals
        p1_count = metrics.arrivals_by_priority.get("P1", 0)
        total_arrivals = max(metrics.arrivals, 1)
        if p1_count / total_arrivals > 0.10:
            concerns.append(
                f"High proportion of P1 arrivals ({p1_count/total_arrivals:.0%}) - increased surgical demand likely"
            )

        # Severity
        if metrics.util_theatre > 0.90 or metrics.mean_theatre_wait > 120:
            severity = Severity.CRITICAL
            assessment = "Theatre capacity critical. Emergency surgical capacity compromised."
        elif metrics.util_theatre > 0.80:
            severity = Severity.HIGH
            assessment = "Theatre utilisation high. Limited flexibility for emergency cases."
        elif metrics.util_theatre > 0.60:
            severity = Severity.MEDIUM
            assessment = "Theatre operating at moderate load. Emergency capacity available."
        else:
            severity = Severity.INFO
            assessment = "Theatre capacity adequate for emergency surgical demand."

        if not concerns:
            concerns.append("Theatre capacity and surgical pathways operating normally")

        return ExpertPerspective(
            expert_role="surgeon",
            expert_title="Trauma Surgeon",
            focus_area="Theatre capacity, trauma surgery, time-critical procedures",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Maintain current surgical protocols"],
            key_metrics={
                "util_theatre": metrics.util_theatre,
                "mean_theatre_wait": metrics.mean_theatre_wait,
                "theatre_admissions": metrics.theatre_admissions,
            },
            severity=severity,
        )

    def _cbrn_specialist_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """CBRN Specialist perspective.

        Focus: Decontamination workflows, contamination protocols, surge capacity.
        Note: CBRN-specific metrics may not be present in standard simulations.
        """
        concerns = []
        recommendations = []

        # Assess surge capacity from CBRN/MCI perspective
        # CBRN events typically cause bolus arrivals with decon delays

        # ED capacity for MCI
        if metrics.util_ed_bays > 0.80:
            concerns.append(
                f"ED at {metrics.util_ed_bays:.0%} - limited surge capacity for contaminated casualties"
            )
            recommendations.append("Identify decon pathway that doesn't consume ED bays")

        # Treatment delays indicating capacity issues
        if metrics.mean_treatment_wait > 45:
            concerns.append(
                f"Current wait times ({metrics.mean_treatment_wait:.0f}min) would compound decon delays (15-45min)"
            )
            recommendations.append("Review MCI protocols for parallel decon/triage processing")

        # Downstream capacity for contaminated patients
        if metrics.util_ward > 0.85:
            concerns.append(
                "Limited isolation capacity - contaminated patients may overwhelm ward infection control"
            )

        # P1 handling capacity
        p1_arrivals = metrics.arrivals_by_priority.get("P1", 0)
        if p1_arrivals > 5 and metrics.p_delay > 0.3:
            concerns.append(
                "Current system struggles with high-acuity load - CBRN P1 casualties would face critical delays"
            )

        # Severity - CBRN perspective is about preparedness
        if metrics.util_ed_bays > 0.90 and metrics.p_delay > 0.5:
            severity = Severity.HIGH
            assessment = "System has minimal surge capacity. CBRN event would quickly overwhelm current resources."
        elif metrics.util_ed_bays > 0.80:
            severity = Severity.MEDIUM
            assessment = "Limited CBRN/MCI surge capacity. Recommend contingency planning."
        else:
            severity = Severity.INFO
            assessment = "System has some surge capacity for CBRN/MCI events."

        if not concerns:
            concerns.append("Adequate baseline capacity for CBRN surge response")

        return ExpertPerspective(
            expert_role="cbrn_specialist",
            expert_title="CBRN Specialist",
            focus_area="Decontamination workflows, MCI surge capacity, contamination protocols",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Review CBRN response protocols annually"],
            key_metrics={
                "util_ed_bays": metrics.util_ed_bays,
                "mean_treatment_wait": metrics.mean_treatment_wait,
                "p_delay": metrics.p_delay,
                "util_ward": metrics.util_ward,
            },
            severity=severity,
        )

    def _data_scientist_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """NHS Health Data Scientist perspective.

        Focus: Statistical confidence, metrics design, operational research insights.
        """
        concerns = []
        recommendations = []

        n_reps = metrics.n_replications

        # Replication confidence
        if n_reps < 10:
            concerns.append(
                f"Only {n_reps} replications - confidence intervals may be wide"
            )
            recommendations.append("Consider 20+ replications for robust statistical inference")

        # CI overlap concerns
        for metric_name, (ci_low, ci_high) in metrics.ci_bounds.items():
            ci_width = ci_high - ci_low
            mean_val = getattr(metrics, metric_name, ci_low)
            if mean_val > 0 and ci_width / mean_val > 0.3:
                concerns.append(
                    f"{metric_name}: CI width {ci_width:.2f} is >30% of mean - high uncertainty"
                )

        # Utilisation ceiling effects
        if metrics.util_ed_bays > 0.95:
            concerns.append(
                "ED utilisation >95% - queuing theory predicts exponential wait time growth"
            )
            recommendations.append("Target <85% utilisation to maintain linear wait behaviour")

        # Variability indicators
        if metrics.p95_treatment_wait > 2 * metrics.mean_treatment_wait:
            concerns.append(
                f"High wait time variability (P95 = {metrics.p95_treatment_wait/metrics.mean_treatment_wait:.1f}x mean)"
            )
            recommendations.append("Investigate causes of wait time variability (arrivals? service times?)")

        # Delay probability interpretation
        if metrics.p_delay > 0.5:
            concerns.append(
                f"P(delay) = {metrics.p_delay:.0%} indicates system operating above effective capacity"
            )

        # Severity
        if n_reps < 5 or len(concerns) > 3:
            severity = Severity.MEDIUM
            assessment = "Statistical reliability concerns. Results should be interpreted with caution."
        elif n_reps >= 20 and len(concerns) <= 1:
            severity = Severity.INFO
            assessment = "Good statistical basis for results. Confidence intervals are reliable."
        else:
            severity = Severity.LOW
            assessment = "Adequate statistical basis. Consider additional replications for key decisions."

        if not concerns:
            concerns.append("Statistical metrics indicate robust simulation results")

        return ExpertPerspective(
            expert_role="data_scientist",
            expert_title="NHS Health Data Scientist",
            focus_area="Statistical confidence, CI computation, operational research metrics",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Results statistically sound"],
            key_metrics={
                "n_replications": float(n_reps),
                "p_delay": metrics.p_delay,
                "p95_treatment_wait": metrics.p95_treatment_wait,
                "mean_treatment_wait": metrics.mean_treatment_wait,
            },
            severity=severity,
        )

    def _paramedic_perspective(self, metrics: MetricsSummary) -> ExpertPerspective:
        """Paramedic/Pre-hospital perspective.

        Focus: Bolus arrival patterns, ambulance turnaround, handover delays.
        """
        concerns = []
        recommendations = []

        # Handover delays - critical for community response
        if metrics.mean_handover_delay > 30:
            concerns.append(
                f"Mean handover delay {metrics.mean_handover_delay:.0f}min - crews unavailable for 999 calls"
            )
            recommendations.append("Implement rapid handover protocol, consider corridor care")
        elif metrics.mean_handover_delay > 15:
            concerns.append(
                f"Handover delays ({metrics.mean_handover_delay:.0f}min) affecting ambulance availability"
            )

        # Max handover (indicates worst case)
        if metrics.max_handover_delay > 60:
            concerns.append(
                f"Max handover delay {metrics.max_handover_delay:.0f}min - some crews severely impacted"
            )
            recommendations.append("Review handover bay capacity and flow to prevent extreme delays")

        # Ambulance arrival volume
        ambulance_arrivals = metrics.arrivals_by_mode.get("ambulance", 0)
        helicopter_arrivals = metrics.arrivals_by_mode.get("helicopter", 0)
        total_arrivals = max(metrics.arrivals, 1)
        conveyance_rate = (ambulance_arrivals + helicopter_arrivals) / total_arrivals

        if conveyance_rate > 0.6:
            concerns.append(
                f"High conveyance rate ({conveyance_rate:.0%}) - ambulance service heavily committed"
            )

        # ED blocking affecting pre-hospital
        if metrics.util_ed_bays > 0.90 and metrics.mean_handover_delay > 0:
            concerns.append(
                "ED saturation causing ambulance service degradation - community at risk"
            )
            recommendations.append("Consider divert, increase handover bays, expedite discharges")

        # Bolus arrival risk (multi-wave delivery)
        p95_to_mean = metrics.p95_treatment_wait / max(metrics.mean_treatment_wait, 1)
        if p95_to_mean > 2.0 and ambulance_arrivals > 0:
            concerns.append(
                "Wait time variability suggests bolus arrivals - multi-ambulance waves likely"
            )
            recommendations.append("Coordinate with ambulance service on arrival spacing if possible")

        # Severity
        if metrics.mean_handover_delay > 45:
            severity = Severity.CRITICAL
            assessment = "Critical handover delays. Community emergency response severely impacted."
        elif metrics.mean_handover_delay > 20:
            severity = Severity.HIGH
            assessment = "Significant handover delays. Ambulance availability reduced."
        elif metrics.mean_handover_delay > 10:
            severity = Severity.MEDIUM
            assessment = "Moderate handover impact. Monitor for deterioration."
        else:
            severity = Severity.INFO
            assessment = "Handover times acceptable. Ambulance service operating normally."

        if not concerns:
            concerns.append("Ambulance handover and pre-hospital flow operating normally")

        return ExpertPerspective(
            expert_role="paramedic",
            expert_title="Paramedic/Pre-hospital Lead",
            focus_area="Ambulance turnaround, handover delays, bolus arrival patterns",
            assessment=assessment,
            concerns=concerns,
            recommendations=recommendations if recommendations else ["Maintain current handover protocols"],
            key_metrics={
                "mean_handover_delay": metrics.mean_handover_delay,
                "max_handover_delay": metrics.max_handover_delay,
                "ambulance_arrivals": ambulance_arrivals,
                "helicopter_arrivals": helicopter_arrivals,
            },
            severity=severity,
        )

    def _generate_executive_summary(
        self,
        metrics: MetricsSummary,
        structural: StructuralAssessment,
        peak_load: PeakLoadAnalysis,
        perspectives: list[ExpertPerspective],
        status: str,
    ) -> str:
        """Generate executive summary from all analyses.

        Args:
            metrics: Metrics summary
            structural: Structural assessment
            peak_load: Peak load analysis
            perspectives: List of expert perspectives
            status: Overall system status

        Returns:
            Executive summary text
        """
        # Count concerns by severity
        critical_concerns = sum(1 for p in perspectives if p.severity == Severity.CRITICAL)
        high_concerns = sum(1 for p in perspectives if p.severity == Severity.HIGH)

        # Build summary
        lines = []

        # Status headline
        status_emoji = {"critical": "🔴", "warning": "🟠", "caution": "🟡", "normal": "🟢"}.get(status, "⚪")
        lines.append(f"**System Status: {status_emoji} {status.upper()}**")
        lines.append("")

        # Key metrics summary
        lines.append(f"- **Resilience Score**: {structural.resilience_score:.0f}/100")
        lines.append(f"- **P(Delay)**: {metrics.p_delay:.0%}")
        lines.append(f"- **ED Utilisation**: {metrics.util_ed_bays:.0%}")
        lines.append(f"- **Peak/Mean Ratio**: {peak_load.peak_to_mean_ratio:.1f}x")
        lines.append("")

        # Bottleneck chain
        if structural.bottleneck_chain:
            lines.append(f"**Bottleneck Chain**: {' → '.join(structural.bottleneck_chain)}")
            lines.append("")

        # Critical concerns
        if critical_concerns > 0:
            lines.append(f"**{critical_concerns} CRITICAL concern(s)** require immediate attention.")
        elif high_concerns > 0:
            lines.append(f"**{high_concerns} HIGH priority concern(s)** require review.")
        else:
            lines.append("No critical or high priority concerns identified.")

        # Top recommendations (from highest severity perspectives)
        top_recs = []
        for p in sorted(perspectives, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x.severity.value, 3)):
            for rec in p.recommendations[:1]:  # Take first recommendation from each
                if rec not in top_recs and len(top_recs) < 3:
                    top_recs.append(rec)

        if top_recs:
            lines.append("")
            lines.append("**Top Recommendations**:")
            for rec in top_recs:
                lines.append(f"- {rec}")

        return "\n".join(lines)
