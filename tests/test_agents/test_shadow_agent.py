"""Tests for HeuristicShadowAgent.

These tests verify that the heuristic shadow agent correctly:
1. Detects issues in "bad" scenarios
2. Produces no alerts for "healthy" scenarios
3. Applies compound rules correctly
4. Produces deterministic, reproducible output
5. Generates valid insight objects
"""

import pytest
from faer.agents.shadow import HeuristicShadowAgent, ClinicalThreshold, NHS_THRESHOLDS
from faer.agents.interface import (
    MetricsSummary,
    Severity,
    InsightCategory,
)


@pytest.fixture
def healthy_metrics() -> MetricsSummary:
    """Metrics that should not trigger any alerts."""
    return MetricsSummary(
        scenario_name="healthy",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=200,
        arrivals_by_priority={"P1": 10, "P2": 60, "P3": 80, "P4": 50},
        arrivals_by_mode={"ambulance": 100, "helicopter": 10, "walk_in": 90},
        mean_triage_wait=5.0,
        mean_treatment_wait=30.0,
        p95_treatment_wait=90.0,  # Well under 240
        mean_system_time=120.0,
        p95_system_time=200.0,
        p_delay=0.25,  # Under 0.5
        util_triage=0.40,
        util_ed_bays=0.65,  # Under 0.85
        util_itu=0.70,  # Under 0.90
        util_ward=0.60,
        util_theatre=0.50,
        itu_admissions=15,
        mean_itu_wait=20.0,
        ward_admissions=50,
        mean_ward_wait=15.0,
        theatre_admissions=10,
        mean_theatre_wait=30.0,
        mean_boarding_time=10.0,
        p_boarding=0.10,
        mean_handover_delay=10.0,  # Under 30
        max_handover_delay=25.0,
    )


@pytest.fixture
def critical_metrics() -> MetricsSummary:
    """Metrics that should trigger CRITICAL alerts."""
    return MetricsSummary(
        scenario_name="critical",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=300,
        arrivals_by_priority={"P1": 35, "P2": 100, "P3": 110, "P4": 55},
        arrivals_by_mode={"ambulance": 200, "helicopter": 25, "walk_in": 75},
        mean_triage_wait=20.0,
        mean_treatment_wait=150.0,  # High - triggers compound rule
        p95_treatment_wait=300.0,  # Over 240 - CRITICAL
        mean_system_time=280.0,
        p95_system_time=450.0,
        p_delay=0.70,  # Over 0.5 - triggers alert
        util_triage=0.55,
        util_ed_bays=0.95,  # Over 0.85 - HIGH
        util_itu=0.94,  # Over 0.90 - HIGH
        util_ward=0.80,
        util_theatre=0.75,  # triggers compound with ITU
        itu_admissions=30,
        mean_itu_wait=90.0,
        ward_admissions=70,
        mean_ward_wait=60.0,
        theatre_admissions=20,
        mean_theatre_wait=45.0,
        mean_boarding_time=75.0,  # Over 60 - HIGH
        p_boarding=0.40,
        mean_handover_delay=45.0,  # Over 30 - HIGH
        max_handover_delay=120.0,
    )


@pytest.fixture
def edge_case_metrics() -> MetricsSummary:
    """Metrics at threshold boundaries."""
    return MetricsSummary(
        scenario_name="edge_case",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=200,
        arrivals_by_priority={"P1": 10, "P2": 60, "P3": 80, "P4": 50},
        arrivals_by_mode={"ambulance": 100, "helicopter": 10, "walk_in": 90},
        mean_triage_wait=5.0,
        mean_treatment_wait=30.0,
        p95_treatment_wait=240.0,  # Exactly at threshold (not over)
        mean_system_time=120.0,
        p95_system_time=200.0,
        p_delay=0.50,  # Exactly at threshold (not over)
        util_triage=0.40,
        util_ed_bays=0.85,  # Exactly at threshold (not over)
        util_itu=0.90,  # Exactly at threshold (not over)
        util_ward=0.60,
        util_theatre=0.50,
        itu_admissions=15,
        mean_itu_wait=20.0,
        ward_admissions=50,
        mean_ward_wait=15.0,
        theatre_admissions=10,
        mean_theatre_wait=30.0,
        mean_boarding_time=60.0,  # Exactly at threshold
        p_boarding=0.10,
        mean_handover_delay=30.0,  # Exactly at threshold
        max_handover_delay=45.0,
    )


class TestHeuristicShadowAgent:
    """Test suite for HeuristicShadowAgent."""

    def test_healthy_scenario_no_critical_alerts(self, healthy_metrics):
        """Healthy scenario should produce no CRITICAL alerts."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(healthy_metrics)

        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) == 0, "Healthy scenario should have no CRITICAL alerts"

    def test_healthy_scenario_no_high_alerts(self, healthy_metrics):
        """Healthy scenario should produce no HIGH alerts."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(healthy_metrics)

        high = [i for i in insights if i.severity == Severity.HIGH]
        assert len(high) == 0, "Healthy scenario should have no HIGH alerts"

    def test_critical_scenario_detects_issues(self, critical_metrics):
        """Critical scenario should trigger multiple alerts."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        assert len(insights) > 0, "Should detect issues in critical scenario"

        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) >= 1, "Should have at least one CRITICAL alert"

    def test_detects_4_hour_breach(self, critical_metrics):
        """Should detect 4-hour treatment standard breach."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "4-Hour Standard Breach" in titles, "Should detect 4-hour breach"

    def test_detects_itu_capacity(self, critical_metrics):
        """Should detect ITU capacity critical."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "ITU Capacity Critical" in titles, "Should detect ITU capacity issue"

    def test_detects_ed_saturation(self, critical_metrics):
        """Should detect ED bay saturation."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "ED Bay Saturation" in titles, "Should detect ED saturation"

    def test_detects_handover_delays(self, critical_metrics):
        """Should detect ambulance handover delays."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "Ambulance Handover Delays" in titles, "Should detect handover delays"

    def test_insights_sorted_by_severity(self, critical_metrics):
        """Insights should be sorted CRITICAL first."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        if len(insights) > 1:
            severity_order = {
                Severity.CRITICAL: 0,
                Severity.HIGH: 1,
                Severity.MEDIUM: 2,
                Severity.LOW: 3,
                Severity.INFO: 4,
            }
            severity_values = [severity_order[i.severity] for i in insights]
            assert severity_values == sorted(
                severity_values
            ), "Insights should be sorted by severity"

    def test_insight_has_required_fields(self, critical_metrics):
        """Each insight should have all required fields."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        for insight in insights:
            assert insight.severity is not None, "Severity required"
            assert insight.category is not None, "Category required"
            assert len(insight.title) > 0, "Title required"
            assert len(insight.title) <= 80, "Title must be ≤80 chars"
            assert len(insight.message) > 0, "Message required"
            assert len(insight.impact_metric) > 0, "Impact metric required"
            assert isinstance(insight.evidence, dict), "Evidence must be dict"
            assert insight.source_agent == "heuristic_shadow", "Source agent set"

    def test_compound_rule_high_acuity_long_wait(self, critical_metrics):
        """Should detect high acuity + long wait compound risk."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert (
            "High Acuity with Treatment Delays" in titles
        ), "Should detect compound risk"

    def test_compound_rule_downstream_gridlock(self, critical_metrics):
        """Should detect downstream gridlock risk."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "Downstream Gridlock Risk" in titles, "Should detect gridlock"

    def test_agent_health_check(self):
        """Heuristic agent should always pass health check."""
        agent = HeuristicShadowAgent()
        assert agent.health_check() is True

    def test_agent_name_and_description(self):
        """Agent should have name and description."""
        agent = HeuristicShadowAgent()
        assert agent.name == "heuristic_shadow"
        assert len(agent.description) > 0

    def test_deterministic_output(self, critical_metrics):
        """Same input should always produce same output."""
        agent = HeuristicShadowAgent()

        insights1 = agent.analyze(critical_metrics)
        insights2 = agent.analyze(critical_metrics)

        assert len(insights1) == len(insights2), "Should produce same count"
        for i1, i2 in zip(insights1, insights2):
            assert i1.title == i2.title, "Titles should match"
            assert i1.severity == i2.severity, "Severities should match"
            assert i1.message == i2.message, "Messages should match"

    def test_edge_case_at_threshold_not_triggered(self, edge_case_metrics):
        """Values exactly at threshold should NOT trigger (using 'gt' operator)."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(edge_case_metrics)

        # p95_treatment_wait=240 should NOT trigger "4-Hour Standard Breach"
        # because rule uses "gt" (greater than), not "gte"
        titles = [i.title for i in insights]
        assert "4-Hour Standard Breach" not in titles, "Exactly at threshold shouldn't trigger"

    def test_custom_thresholds(self, healthy_metrics):
        """Agent should accept custom thresholds."""
        custom_thresholds = [
            ClinicalThreshold(
                metric="p95_treatment_wait",
                threshold=60.0,  # Much stricter than NHS
                operator="gt",
                severity=Severity.HIGH,
                category=InsightCategory.WAIT_TIME,
                title="Custom Wait Breach",
                message_template="Wait of {value:.0f} exceeds custom threshold",
                recommendation="Take action",
            )
        ]

        agent = HeuristicShadowAgent(thresholds=custom_thresholds)
        insights = agent.analyze(healthy_metrics)

        # healthy_metrics has p95_treatment_wait=90, which is >60
        titles = [i.title for i in insights]
        assert "Custom Wait Breach" in titles, "Custom threshold should trigger"

    def test_empty_thresholds(self, critical_metrics):
        """Agent with no thresholds should only evaluate compound rules."""
        agent = HeuristicShadowAgent(thresholds=[])
        insights = agent.analyze(critical_metrics)

        # Should still get compound rules
        titles = [i.title for i in insights]
        assert "High Acuity with Treatment Delays" in titles


class TestMetricsSummary:
    """Tests for MetricsSummary data model."""

    def test_from_run_results_basic(self):
        """Should create MetricsSummary from results dict."""
        results = {
            "arrivals": [100, 110, 105],
            "mean_treatment_wait": [30, 35, 32],
            "p95_treatment_wait": [90, 95, 92],
            "p_delay": [0.2, 0.25, 0.22],
            "util_ed_bays": [0.6, 0.65, 0.62],
        }

        metrics = MetricsSummary.from_run_results(results, "test_scenario")

        assert metrics.scenario_name == "test_scenario"
        assert metrics.n_replications == 3
        assert metrics.arrivals == pytest.approx(105, rel=0.01)  # mean of [100,110,105]
        assert metrics.mean_treatment_wait == pytest.approx(32.33, rel=0.01)

    def test_from_run_results_missing_fields(self):
        """Should handle missing fields with defaults."""
        results = {"arrivals": [100]}

        metrics = MetricsSummary.from_run_results(results)

        assert metrics.arrivals == 100
        assert metrics.util_itu == 0.0  # default
        assert metrics.aeromed_total == 0.0  # default

    def test_from_run_results_empty(self):
        """Should handle empty results."""
        results = {}

        metrics = MetricsSummary.from_run_results(results)

        assert metrics.arrivals == 0.0
        assert metrics.n_replications == 1


class TestClinicalInsight:
    """Tests for ClinicalInsight data model."""

    def test_valid_insight(self):
        """Should create valid insight."""
        from faer.agents.interface import ClinicalInsight

        insight = ClinicalInsight(
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="Test Insight",
            message="This is a test message",
            impact_metric="util_ed_bays",
            evidence={"util_ed_bays": 0.95},
            recommendation="Do something",
            source_agent="test",
        )

        assert insight.severity == Severity.HIGH
        assert insight.title == "Test Insight"

    def test_title_too_long_raises(self):
        """Should raise ValueError if title > 80 chars."""
        from faer.agents.interface import ClinicalInsight

        with pytest.raises(ValueError, match="Title must be"):
            ClinicalInsight(
                severity=Severity.HIGH,
                category=InsightCategory.CAPACITY,
                title="X" * 81,  # 81 chars
                message="Test",
                impact_metric="test",
                evidence={},
            )

    def test_insight_is_immutable(self):
        """Insight should be frozen (immutable)."""
        from faer.agents.interface import ClinicalInsight

        insight = ClinicalInsight(
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="Test",
            message="Test",
            impact_metric="test",
            evidence={},
        )

        with pytest.raises(AttributeError):
            insight.title = "New Title"  # type: ignore
