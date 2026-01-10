"""FAER Agent Layer - Clinical Intelligence for Simulation Output.

This module provides autonomous agents that analyze simulation results
and generate clinical insights, capacity recommendations, and narrative reports.

Example usage:
    from faer.agents import HeuristicShadowAgent, AgentOrchestrator, MetricsSummary

    # Convert simulation results to metrics
    metrics = MetricsSummary.from_run_results(results)

    # Run agents
    orchestrator = AgentOrchestrator()
    orchestrator.register(HeuristicShadowAgent())
    result = orchestrator.run_all(metrics)

    # Get insights
    for insight in result.get_all_insights():
        print(f"[{insight.severity}] {insight.title}")
"""

from .interface import (
    Severity,
    InsightCategory,
    ClinicalInsight,
    CapacityRecommendation,
    MetricsSummary,
    ClinicalAgent,
    AgentResult,
    AgentExecutionError,
)
from .shadow import HeuristicShadowAgent, ClinicalThreshold, NHS_THRESHOLDS
from .orchestrator import AgentOrchestrator, OrchestratorConfig, OrchestratorResult

__all__ = [
    # Data models
    "Severity",
    "InsightCategory",
    "ClinicalInsight",
    "CapacityRecommendation",
    "MetricsSummary",
    # Protocols
    "ClinicalAgent",
    "AgentResult",
    "AgentExecutionError",
    # Agents
    "HeuristicShadowAgent",
    "ClinicalThreshold",
    "NHS_THRESHOLDS",
    # Orchestration
    "AgentOrchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
]
