"""Agent Orchestrator - Coordinates multiple agents and aggregates outputs.

The orchestrator provides:
- Agent registration and management
- Sequential or parallel execution (future)
- Timeout handling
- Failure isolation (fail-open by default)
- Insight deduplication and aggregation
- Execution metrics and observability

Example usage:
    from faer.agents.orchestrator import AgentOrchestrator, OrchestratorConfig
    from faer.agents.shadow import HeuristicShadowAgent

    # Create orchestrator with custom config
    config = OrchestratorConfig(timeout_ms=5000, fail_open=True)
    orchestrator = AgentOrchestrator(config)

    # Register agents
    orchestrator.register(HeuristicShadowAgent())
    # orchestrator.register(CapacityAdvisorAgent())  # Future

    # Run all agents
    result = orchestrator.run_all(metrics_summary)

    # Access aggregated insights
    for insight in result.get_all_insights():
        print(f"[{insight.severity}] {insight.title}")

    # Check execution stats
    print(f"Executed in {result.total_execution_time_ms:.1f}ms")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .interface import (
    AgentExecutionError,
    AgentResult,
    ClinicalAgent,
    ClinicalInsight,
    MetricsSummary,
    Severity,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for agent orchestration.

    Attributes:
        timeout_ms: Maximum time per agent (milliseconds). Default 5000.
        fail_open: Continue on agent failure if True. Default True.
        deduplicate_insights: Remove duplicate insights. Default True.
        max_insights_per_agent: Limit insights per agent. Default 10.
    """

    timeout_ms: float = 5000.0
    fail_open: bool = True
    deduplicate_insights: bool = True
    max_insights_per_agent: int = 10


class AgentOrchestrator:
    """Coordinates multiple agents and aggregates their outputs.

    The orchestrator manages agent lifecycle and execution:
    1. Registration: Agents are registered with unique names
    2. Execution: All registered agents run against provided metrics
    3. Aggregation: Results are collected and deduplicated
    4. Failure handling: Agent failures don't crash the system

    Thread Safety:
        Currently single-threaded. Future versions may support
        parallel agent execution.

    Attributes:
        config: OrchestratorConfig with execution settings
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize orchestrator with optional config.

        Args:
            config: Orchestration settings. Uses defaults if None.
        """
        self.config = config or OrchestratorConfig()
        self._agents: dict[str, ClinicalAgent] = {}
        self._hooks: list[Callable[[AgentResult], None]] = []

    def register(self, agent: ClinicalAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: Agent instance implementing ClinicalAgent protocol

        Raises:
            ValueError: If agent with same name already registered
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} - {agent.description}")

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator.

        Args:
            agent_name: Name of agent to remove

        Note:
            Silently succeeds if agent not found.
        """
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    def add_result_hook(self, hook: Callable[[AgentResult], None]) -> None:
        """Add a hook called after each agent execution.

        Hooks receive the AgentResult after each agent completes.
        Use for logging, metrics, or side effects.

        Args:
            hook: Callable that receives AgentResult
        """
        self._hooks.append(hook)

    @property
    def registered_agents(self) -> list[str]:
        """List of registered agent names."""
        return list(self._agents.keys())

    def run_all(self, metrics: MetricsSummary) -> "OrchestratorResult":
        """Execute all registered agents and collect results.

        Runs each agent sequentially, collects insights, and
        handles failures according to fail_open configuration.

        Args:
            metrics: Simulation output to analyze

        Returns:
            OrchestratorResult with all agent outputs and metadata

        Note:
            If fail_open is False and an agent fails, raises
            AgentExecutionError. Otherwise, failed agents are
            recorded but execution continues.
        """
        results = []

        for name, agent in self._agents.items():
            result = self._run_agent(agent, metrics)
            results.append(result)

            # Execute hooks
            for hook in self._hooks:
                try:
                    hook(result)
                except Exception as e:
                    logger.warning(f"Hook error for {name}: {e}")

        return OrchestratorResult(
            agent_results=results,
            config=self.config,
        )

    def _run_agent(
        self, agent: ClinicalAgent, metrics: MetricsSummary
    ) -> AgentResult:
        """Execute a single agent with error handling and timing.

        Args:
            agent: Agent to execute
            metrics: Metrics to analyze

        Returns:
            AgentResult with insights or error details
        """
        start_time = time.time()

        try:
            # Health check first
            if not agent.health_check():
                raise AgentExecutionError(
                    f"Agent {agent.name} failed health check"
                )

            # Run analysis
            insights = agent.analyze(metrics)

            # Apply limits
            if len(insights) > self.config.max_insights_per_agent:
                logger.warning(
                    f"Agent {agent.name} produced {len(insights)} insights, "
                    f"truncated to {self.config.max_insights_per_agent}"
                )
                insights = insights[: self.config.max_insights_per_agent]

            execution_time = (time.time() - start_time) * 1000

            logger.debug(
                f"Agent {agent.name} completed in {execution_time:.1f}ms "
                f"with {len(insights)} insights"
            )

            return AgentResult(
                agent_name=agent.name,
                execution_time_ms=execution_time,
                insights=insights,
                success=True,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Agent {agent.name} failed: {e}")

            if not self.config.fail_open:
                raise AgentExecutionError(
                    f"Agent {agent.name} failed: {e}"
                ) from e

            return AgentResult(
                agent_name=agent.name,
                execution_time_ms=execution_time,
                insights=[],
                success=False,
                error_message=str(e),
            )


@dataclass
class OrchestratorResult:
    """Aggregated results from all agents.

    Provides methods to access and filter insights from all agents,
    along with execution metadata.

    Attributes:
        agent_results: List of results from each agent
        config: Config used for this execution
    """

    agent_results: list[AgentResult]
    config: OrchestratorConfig

    def get_all_insights(self) -> list[ClinicalInsight]:
        """Get all insights from all agents, optionally deduplicated.

        Returns:
            List of ClinicalInsight from successful agents.
            If config.deduplicate_insights is True, removes duplicates
            by title (keeps first occurrence).
        """
        all_insights = []
        for result in self.agent_results:
            if result.success:
                all_insights.extend(result.insights)

        if self.config.deduplicate_insights:
            # Deduplicate by title (keep first occurrence)
            seen_titles: set[str] = set()
            unique_insights = []
            for insight in all_insights:
                if insight.title not in seen_titles:
                    seen_titles.add(insight.title)
                    unique_insights.append(insight)
            return unique_insights

        return all_insights

    def get_insights_by_severity(self, severity: Severity) -> list[ClinicalInsight]:
        """Filter insights by severity level.

        Args:
            severity: Severity level to filter for

        Returns:
            List of insights matching the severity
        """
        return [i for i in self.get_all_insights() if i.severity == severity]

    def get_critical_insights(self) -> list[ClinicalInsight]:
        """Get only CRITICAL severity insights.

        Convenience method for accessing highest-priority items.

        Returns:
            List of CRITICAL severity insights
        """
        return self.get_insights_by_severity(Severity.CRITICAL)

    def get_high_and_critical(self) -> list[ClinicalInsight]:
        """Get HIGH and CRITICAL severity insights.

        Returns:
            List of HIGH and CRITICAL insights
        """
        return [
            i
            for i in self.get_all_insights()
            if i.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

    @property
    def total_execution_time_ms(self) -> float:
        """Total time across all agents (milliseconds)."""
        return sum(r.execution_time_ms for r in self.agent_results)

    @property
    def all_succeeded(self) -> bool:
        """Check if all agents completed successfully."""
        return all(r.success for r in self.agent_results)

    @property
    def failed_agents(self) -> list[str]:
        """List of agent names that failed."""
        return [r.agent_name for r in self.agent_results if not r.success]

    @property
    def summary(self) -> dict:
        """Summary statistics for logging/display.

        Returns:
            Dict with counts of agents, insights by severity, and timing
        """
        insights = self.get_all_insights()
        return {
            "agents_run": len(self.agent_results),
            "agents_succeeded": sum(1 for r in self.agent_results if r.success),
            "agents_failed": len(self.failed_agents),
            "total_insights": len(insights),
            "critical_count": len(
                [i for i in insights if i.severity == Severity.CRITICAL]
            ),
            "high_count": len([i for i in insights if i.severity == Severity.HIGH]),
            "medium_count": len(
                [i for i in insights if i.severity == Severity.MEDIUM]
            ),
            "low_count": len([i for i in insights if i.severity == Severity.LOW]),
            "execution_time_ms": self.total_execution_time_ms,
        }

    def to_dict(self) -> dict:
        """Convert result to serializable dictionary.

        Useful for storing results or sending to frontend.

        Returns:
            Dict representation of results
        """
        return {
            "summary": self.summary,
            "insights": [
                {
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "title": i.title,
                    "message": i.message,
                    "impact_metric": i.impact_metric,
                    "evidence": i.evidence,
                    "recommendation": i.recommendation,
                    "source_agent": i.source_agent,
                }
                for i in self.get_all_insights()
            ],
            "agent_results": [
                {
                    "agent_name": r.agent_name,
                    "execution_time_ms": r.execution_time_ms,
                    "success": r.success,
                    "insight_count": len(r.insights),
                    "error_message": r.error_message,
                }
                for r in self.agent_results
            ],
        }
