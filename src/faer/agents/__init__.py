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

    # Use site-specific thresholds
    from faer.agents import load_site_config
    config = load_site_config(Path("config/thresholds/my_hospital.yaml"))
    agent = HeuristicShadowAgent(thresholds=config.build_thresholds())
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
    # System evaluation models
    StructuralAssessment,
    PeakLoadAnalysis,
    ExpertPerspective,
    SystemEvaluation,
)
from .shadow import HeuristicShadowAgent, ClinicalThreshold, NHS_THRESHOLDS
from .orchestrator import AgentOrchestrator, OrchestratorConfig, OrchestratorResult
from .threshold_config import (
    SiteThresholdConfig,
    load_site_config,
    save_site_config,
    get_default_config_dir,
    list_available_configs,
)

__all__ = [
    # Data models
    "Severity",
    "InsightCategory",
    "ClinicalInsight",
    "CapacityRecommendation",
    "MetricsSummary",
    # System evaluation models
    "StructuralAssessment",
    "PeakLoadAnalysis",
    "ExpertPerspective",
    "SystemEvaluation",
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
    # Site Configuration
    "SiteThresholdConfig",
    "load_site_config",
    "save_site_config",
    "get_default_config_dir",
    "list_available_configs",
]
