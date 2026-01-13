"""Tests for uncertainty-aware clinical insights.

These tests verify the new confidence level computation, CI overlap detection,
and insight generation with uncertainty quantification.
"""

import pytest

from faer.agents import (
    HeuristicShadowAgent,
    MetricsSummary,
    ClinicalThreshold,
    ClinicalInsight,
    Severity,
    InsightCategory,
    SiteThresholdConfig,
)


class TestConfidenceLevelComputation:
    """Test confidence level calculation based on CI relationship to threshold."""

    def test_high_confidence_when_far_from_threshold(self):
        """Mean well beyond CI from threshold -> high confidence."""
        # Distance (0.10) > 2 * CI width (0.04) -> high confidence
        # CI width = 0.02, 2 * CI width = 0.04, distance = 0.10
        conf = HeuristicShadowAgent._compute_confidence_level(
            mean=1.00,
            ci_lower=0.99,
            ci_upper=1.01,  # CI width = 0.02
            threshold=0.90,  # distance = 0.10 > 2 * 0.02 = 0.04
        )
        assert conf == "high"

    def test_medium_confidence_when_close_to_threshold(self):
        """Distance between 1x and 2x CI width -> medium confidence."""
        # Distance (0.03) > CI width (0.02) but < 2 * CI width (0.04)
        conf = HeuristicShadowAgent._compute_confidence_level(
            mean=0.93,
            ci_lower=0.92,
            ci_upper=0.94,
            threshold=0.90,
        )
        assert conf == "medium"

    def test_low_confidence_when_ci_spans_threshold(self):
        """Distance less than CI width -> low confidence."""
        # Mean 0.91, CI [0.88, 0.94], threshold 0.90
        # Distance (0.01) < CI width (0.06) -> low
        conf = HeuristicShadowAgent._compute_confidence_level(
            mean=0.91,
            ci_lower=0.88,
            ci_upper=0.94,
            threshold=0.90,
        )
        assert conf == "low"

    def test_low_confidence_when_no_ci(self):
        """Single replication (CI width = 0) -> low confidence."""
        conf = HeuristicShadowAgent._compute_confidence_level(
            mean=0.95,
            ci_lower=0.95,  # Same as mean = no CI
            ci_upper=0.95,
            threshold=0.90,
        )
        assert conf == "low"


class TestCIOverlapDetection:
    """Test CI threshold overlap detection."""

    def test_ci_clearly_above_threshold_no_overlap(self):
        """CI entirely above threshold -> no overlap."""
        overlaps = HeuristicShadowAgent._ci_overlaps_threshold(
            ci_lower=0.92,
            ci_upper=0.98,
            threshold=0.90,
            operator="gt",
            soft_margin=0.0,
        )
        assert not overlaps

    def test_ci_clearly_below_threshold_no_overlap(self):
        """CI entirely below threshold -> no overlap."""
        overlaps = HeuristicShadowAgent._ci_overlaps_threshold(
            ci_lower=0.80,
            ci_upper=0.85,
            threshold=0.90,
            operator="gt",
            soft_margin=0.0,
        )
        assert not overlaps

    def test_ci_spans_threshold_has_overlap(self):
        """CI spans threshold -> overlap."""
        overlaps = HeuristicShadowAgent._ci_overlaps_threshold(
            ci_lower=0.88,
            ci_upper=0.93,
            threshold=0.90,
            operator="gt",
            soft_margin=0.0,
        )
        assert overlaps

    def test_soft_margin_extends_overlap_zone(self):
        """Soft margin creates larger overlap zone."""
        # Without margin: CI [0.92, 0.95] doesn't overlap threshold 0.90
        overlaps_no_margin = HeuristicShadowAgent._ci_overlaps_threshold(
            ci_lower=0.92,
            ci_upper=0.95,
            threshold=0.90,
            operator="gt",
            soft_margin=0.0,
        )
        assert not overlaps_no_margin

        # With 5% margin: threshold zone becomes [0.855, 0.945]
        # CI [0.92, 0.95] overlaps this zone
        overlaps_with_margin = HeuristicShadowAgent._ci_overlaps_threshold(
            ci_lower=0.92,
            ci_upper=0.95,
            threshold=0.90,
            operator="gt",
            soft_margin=0.05,
        )
        assert overlaps_with_margin


class TestInsightWithConfidenceCreation:
    """Test insight creation with confidence fields."""

    def test_confident_insight_has_full_severity(self):
        """When threshold_overlap is False, use full severity."""
        agent = HeuristicShadowAgent()
        rule = ClinicalThreshold(
            metric="util_itu",
            threshold=0.90,
            operator="gt",
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="ITU Capacity Critical",
            message_template="ITU at {value:.0%}",
            recommendation="Review step-downs",
            severity_when_uncertain=Severity.MEDIUM,
        )

        insight = agent._create_insight_with_confidence(
            rule=rule,
            value=0.95,
            ci_lower=0.93,
            ci_upper=0.97,
            confidence_level="high",
            threshold_overlap=False,
        )

        assert insight.severity == Severity.HIGH  # Full severity
        assert insight.confidence_level == "high"
        assert not insight.threshold_overlap
        assert insight.uncertainty_note == ""  # No uncertainty note

    def test_uncertain_insight_has_downgraded_severity(self):
        """When threshold_overlap is True, use severity_when_uncertain."""
        agent = HeuristicShadowAgent()
        rule = ClinicalThreshold(
            metric="util_itu",
            threshold=0.90,
            operator="gt",
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="ITU Capacity Critical",
            message_template="ITU at {value:.0%}",
            recommendation="Review step-downs",
            severity_when_uncertain=Severity.MEDIUM,
        )

        insight = agent._create_insight_with_confidence(
            rule=rule,
            value=0.91,
            ci_lower=0.88,
            ci_upper=0.94,
            confidence_level="low",
            threshold_overlap=True,
        )

        assert insight.severity == Severity.MEDIUM  # Downgraded
        assert insight.confidence_level == "low"
        assert insight.threshold_overlap
        assert "overlaps the threshold" in insight.uncertainty_note
        assert "(Uncertain)" in insight.title

    def test_insight_ci_bounds_in_evidence(self):
        """CI bounds should be included in evidence dict."""
        agent = HeuristicShadowAgent()
        rule = ClinicalThreshold(
            metric="p_delay",
            threshold=0.50,
            operator="gt",
            severity=Severity.MEDIUM,
            category=InsightCategory.FLOW_BOTTLENECK,
            title="Majority Patients Delayed",
            message_template="Delay at {value:.0%}",
            recommendation="Review flow",
        )

        insight = agent._create_insight_with_confidence(
            rule=rule,
            value=0.55,
            ci_lower=0.52,
            ci_upper=0.58,
            confidence_level="high",
            threshold_overlap=False,
        )

        assert insight.ci_lower == 0.52
        assert insight.ci_upper == 0.58
        assert insight.evidence["ci_lower"] == 0.52
        assert insight.evidence["ci_upper"] == 0.58


class TestMetricsSummaryCIComputation:
    """Test CI computation in MetricsSummary."""

    def test_ci_bounds_populated_from_replications(self):
        """from_run_results should populate ci_bounds from multiple reps."""
        results = {
            "arrivals": [100, 102, 98, 101, 99],
            "p_delay": [0.45, 0.48, 0.42, 0.46, 0.44],
            "util_itu": [0.85, 0.87, 0.83, 0.86, 0.84],
            "mean_treatment_wait": [30, 32, 28, 31, 29],
        }

        metrics = MetricsSummary.from_run_results(results, "Test")

        assert "p_delay" in metrics.ci_bounds
        assert "util_itu" in metrics.ci_bounds
        assert "mean_treatment_wait" in metrics.ci_bounds

        # Check bounds are sensible (lower < mean < upper)
        p_delay_lower, p_delay_upper = metrics.ci_bounds["p_delay"]
        assert p_delay_lower < metrics.p_delay < p_delay_upper

    def test_single_replication_has_no_ci(self):
        """Single replication should not compute CI."""
        results = {
            "arrivals": [100],
            "p_delay": [0.45],
        }

        metrics = MetricsSummary.from_run_results(results, "Test")

        # CI bounds should be empty for single replication
        assert "p_delay" not in metrics.ci_bounds

    def test_ci_computation_can_be_disabled(self):
        """Setting compute_confidence_intervals=False should skip CI calc."""
        results = {
            "arrivals": [100, 102, 98],
            "p_delay": [0.45, 0.48, 0.42],
        }

        metrics = MetricsSummary.from_run_results(
            results, "Test", compute_confidence_intervals=False
        )

        assert len(metrics.ci_bounds) == 0


class TestBackwardCompatibility:
    """Ensure existing code continues to work."""

    def test_old_style_insight_creation(self):
        """ClinicalInsight without new fields should work."""
        insight = ClinicalInsight(
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="Test Alert",
            message="Test message",
            impact_metric="test_metric",
            evidence={"value": 1.0},
        )

        assert insight.severity == Severity.HIGH
        assert insight.confidence_level == "high"  # Default
        assert insight.ci_lower is None  # Default
        assert insight.ci_upper is None  # Default
        assert not insight.threshold_overlap  # Default False
        assert insight.uncertainty_note == ""  # Default

    def test_old_style_threshold_creation(self):
        """ClinicalThreshold without new fields should work."""
        threshold = ClinicalThreshold(
            metric="test",
            threshold=0.5,
            operator="gt",
            severity=Severity.MEDIUM,
            category=InsightCategory.CAPACITY,
            title="Test",
            message_template="Test {value}",
            recommendation="Test",
        )

        assert threshold.soft_margin == 0.0  # Default
        assert threshold.severity_when_uncertain is None  # Default
        assert threshold.uncertainty_message_template == ""  # Default


class TestSiteConfiguration:
    """Test site-specific threshold configuration."""

    def test_build_thresholds_with_nhs_base(self):
        """Building thresholds from NHS base should include all defaults."""
        config = SiteThresholdConfig(
            site_id="test",
            site_name="Test Hospital",
            base_thresholds="NHS",
        )

        thresholds = config.build_thresholds()

        # Should have at least as many as NHS_THRESHOLDS
        from faer.agents import NHS_THRESHOLDS

        assert len(thresholds) >= len(NHS_THRESHOLDS)

    def test_build_thresholds_with_overrides(self):
        """Overrides should modify base thresholds."""
        config = SiteThresholdConfig(
            site_id="test",
            site_name="Test Hospital",
            base_thresholds="NHS",
            overrides={
                "p_delay": {"threshold": 0.60, "soft_margin": 0.05},
            },
        )

        thresholds = config.build_thresholds()

        # Find p_delay threshold
        p_delay_threshold = next(
            (t for t in thresholds if t.metric == "p_delay"), None
        )

        assert p_delay_threshold is not None
        assert p_delay_threshold.threshold == 0.60
        assert p_delay_threshold.soft_margin == 0.05

    def test_build_thresholds_with_custom(self):
        """Custom thresholds should be added."""
        config = SiteThresholdConfig(
            site_id="test",
            site_name="Test Hospital",
            base_thresholds="NONE",
            custom_thresholds=[
                {
                    "metric": "custom_metric",
                    "threshold": 100.0,
                    "operator": "gt",
                    "severity": "HIGH",
                    "category": "CAPACITY",
                    "title": "Custom Alert",
                    "message_template": "Custom {value}",
                    "recommendation": "Do something",
                }
            ],
        )

        thresholds = config.build_thresholds()

        assert len(thresholds) == 1
        assert thresholds[0].metric == "custom_metric"
        assert thresholds[0].threshold == 100.0


class TestIntegration:
    """Integration tests for end-to-end uncertainty flow."""

    def test_agent_analyze_with_ci_bounds(self):
        """Agent should produce insights with confidence info when CIs available."""
        # Create metrics with CI bounds
        results = {
            "arrivals": [100, 105, 95, 102, 98],
            "util_itu": [0.92, 0.94, 0.91, 0.93, 0.92],  # Above 0.90 threshold
            "p_delay": [0.55, 0.58, 0.52, 0.56, 0.54],  # Above 0.50 threshold
            # Required fields
            "arrivals_P1": [5, 6, 4, 5, 5],
            "arrivals_P2": [20, 22, 18, 21, 19],
            "arrivals_P3": [50, 52, 48, 51, 49],
            "arrivals_P4": [25, 25, 25, 25, 25],
            "mean_triage_wait": [5, 6, 4, 5, 5],
            "mean_treatment_wait": [30, 32, 28, 31, 29],
            "p95_treatment_wait": [60, 65, 55, 62, 58],
            "mean_system_time": [90, 95, 85, 92, 88],
            "p95_system_time": [120, 125, 115, 122, 118],
            "util_triage": [0.5, 0.52, 0.48, 0.51, 0.49],
            "util_ed_bays": [0.7, 0.72, 0.68, 0.71, 0.69],
            "util_ward": [0.8, 0.82, 0.78, 0.81, 0.79],
            "util_theatre": [0.6, 0.62, 0.58, 0.61, 0.59],
            "mean_boarding_time": [20, 22, 18, 21, 19],
            "p_boarding": [0.3, 0.32, 0.28, 0.31, 0.29],
            "mean_handover_delay": [10, 12, 8, 11, 9],
            "max_handover_delay": [30, 35, 25, 32, 28],
            "itu_admissions": [2, 3, 1, 2, 2],
            "mean_itu_wait": [15, 17, 13, 16, 14],
            "ward_admissions": [10, 12, 8, 11, 9],
            "mean_ward_wait": [25, 27, 23, 26, 24],
            "theatre_admissions": [5, 6, 4, 5, 5],
            "mean_theatre_wait": [20, 22, 18, 21, 19],
        }

        metrics = MetricsSummary.from_run_results(results, "Integration Test")
        agent = HeuristicShadowAgent()
        insights = agent.analyze(metrics)

        # Should have at least one insight (ITU above threshold)
        assert len(insights) > 0

        # Check that insights have confidence info
        for insight in insights:
            assert hasattr(insight, "confidence_level")
            assert insight.confidence_level in ["high", "medium", "low"]
            # CI bounds should be present when available
            if insight.impact_metric in metrics.ci_bounds:
                assert insight.ci_lower is not None
                assert insight.ci_upper is not None
