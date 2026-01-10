"""Tests for AgentOrchestrator.

These tests verify that the orchestrator correctly:
1. Registers and manages agents
2. Runs all registered agents
3. Handles agent failures gracefully
4. Aggregates and deduplicates insights
5. Tracks execution timing
"""

import pytest
from faer.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorConfig,
    OrchestratorResult,
)
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.interface import (
    MetricsSummary,
    Severity,
    ClinicalInsight,
    InsightCategory,
    AgentExecutionError,
)


@pytest.fixture
def critical_metrics() -> MetricsSummary:
    """Metrics that should trigger alerts."""
    return MetricsSummary(
        scenario_name="critical",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=300,
        arrivals_by_priority={"P1": 35, "P2": 100, "P3": 110, "P4": 55},
        arrivals_by_mode={"ambulance": 200, "helicopter": 25, "walk_in": 75},
        mean_triage_wait=20.0,
        mean_treatment_wait=150.0,
        p95_treatment_wait=300.0,
        mean_system_time=280.0,
        p95_system_time=450.0,
        p_delay=0.70,
        util_triage=0.55,
        util_ed_bays=0.95,
        util_itu=0.94,
        util_ward=0.80,
        util_theatre=0.75,
        itu_admissions=30,
        mean_itu_wait=90.0,
        ward_admissions=70,
        mean_ward_wait=60.0,
        theatre_admissions=20,
        mean_theatre_wait=45.0,
        mean_boarding_time=75.0,
        p_boarding=0.40,
        mean_handover_delay=45.0,
        max_handover_delay=120.0,
    )


@pytest.fixture
def healthy_metrics() -> MetricsSummary:
    """Metrics that should not trigger alerts."""
    return MetricsSummary(
        scenario_name="healthy",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=200,
        arrivals_by_priority={"P1": 10, "P2": 60, "P3": 80, "P4": 50},
        arrivals_by_mode={"ambulance": 100, "helicopter": 10, "walk_in": 90},
        mean_triage_wait=5.0,
        mean_treatment_wait=30.0,
        p95_treatment_wait=90.0,
        mean_system_time=120.0,
        p95_system_time=200.0,
        p_delay=0.25,
        util_triage=0.40,
        util_ed_bays=0.65,
        util_itu=0.70,
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
        mean_handover_delay=10.0,
        max_handover_delay=25.0,
    )


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        name: str = "mock_agent",
        insights: list[ClinicalInsight] | None = None,
        should_fail: bool = False,
        health_ok: bool = True,
    ):
        self._name = name
        self._insights = insights or []
        self._should_fail = should_fail
        self._health_ok = health_ok

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Mock agent for testing"

    def analyze(self, metrics: MetricsSummary) -> list[ClinicalInsight]:
        if self._should_fail:
            raise ValueError("Intentional failure")
        return self._insights

    def health_check(self) -> bool:
        return self._health_ok


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""

    def test_register_single_agent(self):
        """Should register an agent successfully."""
        orchestrator = AgentOrchestrator()
        agent = HeuristicShadowAgent()

        orchestrator.register(agent)

        assert "heuristic_shadow" in orchestrator.registered_agents

    def test_register_duplicate_raises(self):
        """Should raise error when registering duplicate agent name."""
        orchestrator = AgentOrchestrator()
        agent1 = MockAgent(name="test")
        agent2 = MockAgent(name="test")

        orchestrator.register(agent1)

        with pytest.raises(ValueError, match="already registered"):
            orchestrator.register(agent2)

    def test_unregister_agent(self):
        """Should unregister an agent."""
        orchestrator = AgentOrchestrator()
        agent = MockAgent(name="test")

        orchestrator.register(agent)
        orchestrator.unregister("test")

        assert "test" not in orchestrator.registered_agents

    def test_unregister_nonexistent_silent(self):
        """Should silently succeed when unregistering nonexistent agent."""
        orchestrator = AgentOrchestrator()

        # Should not raise
        orchestrator.unregister("nonexistent")

    def test_run_single_agent(self, critical_metrics):
        """Should run single registered agent."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        assert len(result.agent_results) == 1
        assert result.agent_results[0].success is True
        assert len(result.get_all_insights()) > 0

    def test_run_multiple_agents(self, critical_metrics):
        """Should run multiple registered agents."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(MockAgent(name="agent1"))
        orchestrator.register(MockAgent(name="agent2"))

        result = orchestrator.run_all(critical_metrics)

        assert len(result.agent_results) == 2
        assert all(r.success for r in result.agent_results)

    def test_fail_open_continues_on_error(self, critical_metrics):
        """Should continue if agent fails (fail_open=True)."""
        config = OrchestratorConfig(fail_open=True)
        orchestrator = AgentOrchestrator(config)

        failing_agent = MockAgent(name="failing", should_fail=True)
        working_agent = HeuristicShadowAgent()

        orchestrator.register(failing_agent)
        orchestrator.register(working_agent)

        result = orchestrator.run_all(critical_metrics)

        assert len(result.agent_results) == 2
        assert result.agent_results[0].success is False
        assert result.agent_results[1].success is True
        assert len(result.get_all_insights()) > 0

    def test_fail_closed_raises_on_error(self, critical_metrics):
        """Should raise if agent fails and fail_open=False."""
        config = OrchestratorConfig(fail_open=False)
        orchestrator = AgentOrchestrator(config)

        failing_agent = MockAgent(name="failing", should_fail=True)
        orchestrator.register(failing_agent)

        with pytest.raises(AgentExecutionError):
            orchestrator.run_all(critical_metrics)

    def test_health_check_failure(self, critical_metrics):
        """Should handle agent health check failure."""
        config = OrchestratorConfig(fail_open=True)
        orchestrator = AgentOrchestrator(config)

        unhealthy_agent = MockAgent(name="unhealthy", health_ok=False)
        orchestrator.register(unhealthy_agent)

        result = orchestrator.run_all(critical_metrics)

        assert result.agent_results[0].success is False
        assert "health check" in result.agent_results[0].error_message.lower()

    def test_execution_time_tracked(self, critical_metrics):
        """Should track execution time."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        assert result.total_execution_time_ms > 0
        assert result.agent_results[0].execution_time_ms > 0

    def test_max_insights_per_agent(self, critical_metrics):
        """Should limit insights per agent."""
        # Create agent that produces many insights
        many_insights = [
            ClinicalInsight(
                severity=Severity.LOW,
                category=InsightCategory.CAPACITY,
                title=f"Insight {i}",
                message="Test",
                impact_metric="test",
                evidence={},
                source_agent="mock",
            )
            for i in range(20)
        ]

        config = OrchestratorConfig(max_insights_per_agent=5)
        orchestrator = AgentOrchestrator(config)
        orchestrator.register(MockAgent(name="prolific", insights=many_insights))

        result = orchestrator.run_all(critical_metrics)

        assert len(result.agent_results[0].insights) == 5

    def test_result_hook_called(self, critical_metrics):
        """Should call result hooks after each agent."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(MockAgent(name="test"))

        hook_calls = []

        def hook(result):
            hook_calls.append(result.agent_name)

        orchestrator.add_result_hook(hook)
        orchestrator.run_all(critical_metrics)

        assert hook_calls == ["test"]

    def test_hook_error_doesnt_crash(self, critical_metrics):
        """Hook errors should not crash orchestrator."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(MockAgent(name="test"))

        def bad_hook(result):
            raise ValueError("Hook error")

        orchestrator.add_result_hook(bad_hook)

        # Should not raise
        result = orchestrator.run_all(critical_metrics)
        assert result.agent_results[0].success is True


class TestOrchestratorResult:
    """Tests for OrchestratorResult."""

    def test_get_all_insights(self, critical_metrics):
        """Should aggregate insights from all agents."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        insights = result.get_all_insights()

        assert len(insights) > 0

    def test_get_insights_by_severity(self, critical_metrics):
        """Should filter insights by severity."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        critical = result.get_insights_by_severity(Severity.CRITICAL)

        assert all(i.severity == Severity.CRITICAL for i in critical)

    def test_get_critical_insights(self, critical_metrics):
        """Should get only CRITICAL insights."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        critical = result.get_critical_insights()

        assert all(i.severity == Severity.CRITICAL for i in critical)

    def test_get_high_and_critical(self, critical_metrics):
        """Should get HIGH and CRITICAL insights."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        important = result.get_high_and_critical()

        assert all(
            i.severity in (Severity.CRITICAL, Severity.HIGH) for i in important
        )

    def test_all_succeeded(self, critical_metrics):
        """Should report if all agents succeeded."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        assert result.all_succeeded is True

    def test_failed_agents_list(self, critical_metrics):
        """Should list failed agents."""
        config = OrchestratorConfig(fail_open=True)
        orchestrator = AgentOrchestrator(config)
        orchestrator.register(MockAgent(name="failing", should_fail=True))
        orchestrator.register(MockAgent(name="working"))

        result = orchestrator.run_all(critical_metrics)

        assert "failing" in result.failed_agents
        assert "working" not in result.failed_agents

    def test_summary_statistics(self, critical_metrics):
        """Should produce correct summary stats."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        summary = result.summary

        assert summary["agents_run"] == 1
        assert summary["agents_succeeded"] == 1
        assert summary["agents_failed"] == 0
        assert summary["total_insights"] > 0
        assert "critical_count" in summary
        assert "high_count" in summary
        assert "execution_time_ms" in summary

    def test_to_dict_serializable(self, critical_metrics):
        """Should produce serializable dict."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        data = result.to_dict()

        # Should be JSON-serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0

        # Should have expected structure
        assert "summary" in data
        assert "insights" in data
        assert "agent_results" in data

    def test_deduplication(self, critical_metrics):
        """Should deduplicate insights with same title."""
        # Create two agents that produce same insight
        same_insight = ClinicalInsight(
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="Duplicate Title",
            message="Test",
            impact_metric="test",
            evidence={},
            source_agent="agent1",
        )

        config = OrchestratorConfig(deduplicate_insights=True)
        orchestrator = AgentOrchestrator(config)
        orchestrator.register(MockAgent(name="agent1", insights=[same_insight]))
        orchestrator.register(
            MockAgent(
                name="agent2",
                insights=[
                    ClinicalInsight(
                        severity=Severity.HIGH,
                        category=InsightCategory.CAPACITY,
                        title="Duplicate Title",  # Same title
                        message="Different message",
                        impact_metric="test",
                        evidence={},
                        source_agent="agent2",
                    )
                ],
            )
        )

        result = orchestrator.run_all(critical_metrics)
        insights = result.get_all_insights()

        # Should only have one insight with "Duplicate Title"
        duplicate_titles = [i for i in insights if i.title == "Duplicate Title"]
        assert len(duplicate_titles) == 1

    def test_no_deduplication_when_disabled(self, critical_metrics):
        """Should not deduplicate when disabled."""
        same_insight = ClinicalInsight(
            severity=Severity.HIGH,
            category=InsightCategory.CAPACITY,
            title="Duplicate Title",
            message="Test",
            impact_metric="test",
            evidence={},
            source_agent="agent1",
        )

        config = OrchestratorConfig(deduplicate_insights=False)
        orchestrator = AgentOrchestrator(config)
        orchestrator.register(MockAgent(name="agent1", insights=[same_insight]))
        orchestrator.register(
            MockAgent(
                name="agent2",
                insights=[
                    ClinicalInsight(
                        severity=Severity.HIGH,
                        category=InsightCategory.CAPACITY,
                        title="Duplicate Title",
                        message="Test",
                        impact_metric="test",
                        evidence={},
                        source_agent="agent2",
                    )
                ],
            )
        )

        result = orchestrator.run_all(critical_metrics)
        insights = result.get_all_insights()

        duplicate_titles = [i for i in insights if i.title == "Duplicate Title"]
        assert len(duplicate_titles) == 2
