# Product Requirements Document: Clinical Shadow Agent Integration
## Pj-FAER Agent Layer Extension

**Version**: 1.0
**Status**: Draft
**Author**: Engineering Team
**Date**: 2026-01-10

---

## Executive Summary

This PRD defines the integration of autonomous clinical intelligence agents into the FAER (Framework for Acute and Emergency Resources) simulation platform. The agent layer transforms raw simulation outputs into actionable clinical insights, enabling hospital operations teams to identify risks, optimize capacity, and make evidence-based decisions.

The proposed architecture follows **Interface-First Monolith** principles—building agent modules as if they were microservices (strict contracts, decoupled state) while deploying as a unified Streamlit application. This approach maximizes iteration velocity while preserving architectural optionality for future cloud-native deployment.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Agent Taxonomy](#4-agent-taxonomy)
5. [Technical Specification](#5-technical-specification)
6. [Implementation Phases](#6-implementation-phases)
7. [Integration Points](#7-integration-points)
8. [API Contracts](#8-api-contracts)
9. [Testing Strategy](#9-testing-strategy)
10. [Observability](#10-observability)
11. [Success Metrics](#11-success-metrics)
12. [Appendices](#12-appendices)

---

## 1. Problem Statement

### Current State

FAER produces **100+ KPIs** per simulation run:
- Utilization metrics (ED bays, ITU, theatre, ward)
- Wait time distributions (P50, P95, max)
- Admission rates, boarding times, handover delays
- Aeromedical slot misses, bed-days blocked

However, these metrics require **expert interpretation**:
- What does "ITU utilization at 94%" mean for patient safety?
- When P95 treatment wait hits 180 minutes, what clinical risks emerge?
- How do cascading blockages (ED → ITU → Ward) translate to mortality risk?

### The Gap

Simulation experts can run scenarios. Clinical experts can interpret risks. But:
1. **Translation friction**: Simulation output → Clinical action requires manual synthesis
2. **Pattern blindness**: Humans miss subtle multi-variate risk signatures
3. **Scenario fatigue**: Running 30+ configurations produces overwhelming data
4. **No memory**: Each run starts fresh; historical patterns aren't leveraged

### The Opportunity

An **agent layer** can bridge simulation output to clinical decision support by:
- Detecting risk patterns in metric combinations
- Generating natural-language clinical assessments
- Recommending capacity interventions
- Learning from accumulated scenario history

---

## 2. Goals & Non-Goals

### Goals

| Priority | Goal | Success Criteria |
|----------|------|------------------|
| P0 | **Risk Detection** | Agent identifies ≥80% of "known bad" scenarios (validated against clinical rules) |
| P0 | **Zero Disruption** | Existing FAER workflow unchanged; agent is additive |
| P1 | **Natural Language Output** | Insights readable by non-technical clinical staff |
| P1 | **Explainability** | Every insight cites source metrics and thresholds |
| P2 | **LLM Upgrade Path** | Swap heuristics for LLM without architectural change |
| P2 | **Scenario Memory** | Compare current run against historical baselines |

### Non-Goals (v1.0)

- **Real-time intervention**: Agents observe completed runs, not live simulation
- **Automated control**: No automatic parameter modification
- **External data integration**: No EHR, staffing, or weather feeds
- **Multi-tenancy**: Single-user, local deployment only
- **Production SLA**: Research tool, not clinical-grade system

---

## 3. Architecture Overview

### Design Principles

1. **Pure Functions**: Agents are stateless transformers: `f(SimulationOutput) → Insights`
2. **Contract-First**: Strict interfaces between layers; implementations swappable
3. **Fail-Safe**: Agent failures never crash simulation; graceful degradation
4. **Testable**: Deterministic heuristics enable unit testing before LLM integration

### System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FAER Platform                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Streamlit   │───▶│  Simulation  │───▶│  Results Collector   │  │
│  │  UI (Input)  │    │  Engine      │    │  (Metrics + Logs)    │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│                                          ┌───────────▼───────────┐  │
│                                          │    AGENT LAYER        │  │
│                                          │  ┌─────────────────┐  │  │
│                                          │  │ AgentOrchestrator│  │  │
│                                          │  └────────┬────────┘  │  │
│                                          │           │           │  │
│                                          │  ┌────────▼────────┐  │  │
│                                          │  │  Agent Registry │  │  │
│                                          │  │  ├─ Shadow      │  │  │
│                                          │  │  ├─ Capacity    │  │  │
│                                          │  │  ├─ Narrative   │  │  │
│                                          │  │  └─ Comparator  │  │  │
│                                          │  └────────┬────────┘  │  │
│                                          │           │           │  │
│                                          │  ┌────────▼────────┐  │  │
│                                          │  │ InsightAggregator│ │  │
│                                          │  └─────────────────┘  │  │
│                                          └───────────┬───────────┘  │
│                                                      │              │
│  ┌──────────────────────────────────────────────────▼────────────┐  │
│  │                    Streamlit UI (Output)                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────────┐  │  │
│  │  │ KPI Cards  │  │ Charts     │  │ 🔴 Agent Insights Panel │  │  │
│  │  └────────────┘  └────────────┘  └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | State |
|-------|---------------|-------|
| Simulation Engine | Run SimPy model, collect events | Stateful (env, resources) |
| Results Collector | Aggregate metrics, compute KPIs | Stateful (logs, patients) |
| **Agent Layer** | Transform metrics → insights | **Stateless** |
| UI | Display results + insights | Session state |

---

## 4. Agent Taxonomy

### 4.1 Clinical Shadow Agent

**Purpose**: Detect risk patterns invisible to raw metric inspection.

**Input**: `MetricsSummary` (aggregated KPIs from simulation run)

**Output**: `List[ClinicalInsight]` with severity, message, evidence

**Modes**:
- **Heuristic Mode** (v1.0): Rule-based thresholds from clinical guidelines
- **LLM Mode** (v2.0): GPT-4/Claude analysis with structured output

**Example Rules**:
```python
# 4-hour treatment wait breaches NHS safety standard
if p95_treatment_wait > 240:
    yield ClinicalInsight(severity="CRITICAL", ...)

# ITU at 90%+ signals imminent ED blocking cascade
if util_itu > 0.90 and mean_itu_wait > 60:
    yield ClinicalInsight(severity="HIGH", ...)

# Compound risk: long waits + high acuity arrivals
if p_delay > 0.5 and arrivals_resus / arrivals > 0.10:
    yield ClinicalInsight(severity="HIGH", ...)
```

### 4.2 Capacity Advisor Agent

**Purpose**: Recommend specific interventions based on bottleneck analysis.

**Input**: `MetricsSummary` + `FullScenario` (current configuration)

**Output**: `List[CapacityRecommendation]` with action, rationale, expected impact

**Logic**:
1. Identify binding constraint (highest utilization resource)
2. Simulate marginal capacity increase (e.g., +1 bed)
3. Estimate impact using Little's Law heuristics
4. Generate prioritized recommendations

**Example Output**:
```
RECOMMENDATION: Add 1 ITU bed
RATIONALE: ITU at 94% utilization is causing 45-min average boarding delays
EXPECTED IMPACT: Reduce mean treatment wait by ~12 minutes
CONFIDENCE: Medium (based on queuing theory approximation)
```

### 4.3 Narrative Agent (LLM-powered)

**Purpose**: Generate executive summary for non-technical stakeholders.

**Input**: `MetricsSummary` + `List[ClinicalInsight]` + `scenario_name`

**Output**: `NarrativeReport` (3-5 paragraph natural language summary)

**Implementation**: LLM prompt engineering with structured context injection

**Example Prompt**:
```
You are a clinical operations analyst. Given these simulation results:
- ED Utilization: {util_ed_bays}%
- Mean Treatment Wait: {mean_treatment_wait} minutes
- P95 System Time: {p95_system_time} minutes
- ITU Admissions: {itu_admissions}, Wait: {mean_itu_wait} minutes

And these detected risks:
{insights_formatted}

Write a 3-paragraph briefing for the Hospital Chief of Medicine covering:
1. Overall system performance assessment
2. Key risk areas requiring attention
3. Recommended next steps
```

### 4.4 Scenario Comparator Agent

**Purpose**: Compare current run against baseline/historical scenarios.

**Input**: `MetricsSummary` (current) + `List[MetricsSummary]` (comparisons)

**Output**: `ComparisonReport` with delta analysis, trend detection

**Use Cases**:
- "How does this scenario compare to last week's run?"
- "What's the impact of adding 2 ED bays vs. 1 ITU bed?"
- "Show me the Pareto frontier of cost vs. wait time"

---

## 5. Technical Specification

### 5.1 Directory Structure

```
pj_faer/
├── src/faer/
│   ├── agents/                    # NEW: Agent layer
│   │   ├── __init__.py
│   │   ├── interface.py           # Abstract base classes, data models
│   │   ├── shadow.py              # ClinicalShadowAgent implementations
│   │   ├── capacity.py            # CapacityAdvisorAgent
│   │   ├── narrative.py           # NarrativeAgent (LLM-powered)
│   │   ├── comparator.py          # ScenarioComparatorAgent
│   │   ├── orchestrator.py        # AgentOrchestrator (coordination)
│   │   ├── rules/                 # Clinical rule definitions
│   │   │   ├── __init__.py
│   │   │   ├── nhs_standards.py   # NHS clinical thresholds
│   │   │   └── custom_rules.py    # User-defined rules
│   │   └── prompts/               # LLM prompt templates
│   │       ├── narrative.txt
│   │       └── analysis.txt
│   └── ...existing modules...
├── app/
│   └── pages/
│       └── 9_Insights.py          # NEW: Agent insights page
├── tests/
│   └── test_agents/               # NEW: Agent test suite
│       ├── test_shadow_agent.py
│       ├── test_capacity_agent.py
│       └── test_orchestrator.py
└── notebooks/
    └── agent_prototyping.ipynb    # NEW: Interactive development
```

### 5.2 Core Data Models

```python
# src/faer/agents/interface.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Optional
import pandas as pd

class Severity(Enum):
    """Insight severity levels aligned with clinical escalation protocols."""
    CRITICAL = "CRITICAL"  # Immediate action required
    HIGH = "HIGH"          # Urgent attention needed
    MEDIUM = "MEDIUM"      # Monitor closely
    LOW = "LOW"            # Awareness item
    INFO = "INFO"          # Informational only


class InsightCategory(Enum):
    """Clinical insight categories for filtering and routing."""
    WAIT_TIME = "wait_time"
    CAPACITY = "capacity"
    PATIENT_SAFETY = "patient_safety"
    FLOW_BOTTLENECK = "flow_bottleneck"
    RESOURCE_UTILIZATION = "resource_utilization"
    BOARDING = "boarding"
    DOWNSTREAM = "downstream"
    AEROMEDICAL = "aeromedical"


@dataclass(frozen=True)
class ClinicalInsight:
    """Immutable clinical insight generated by an agent."""

    severity: Severity
    category: InsightCategory
    title: str                          # Short headline (≤10 words)
    message: str                        # Detailed explanation
    impact_metric: str                  # Primary metric affected
    evidence: dict[str, float]          # Supporting metric values
    recommendation: Optional[str] = None
    source_agent: str = ""              # Agent that generated this insight

    def __post_init__(self):
        if len(self.title) > 80:
            raise ValueError("Title must be ≤80 characters")


@dataclass(frozen=True)
class CapacityRecommendation:
    """Capacity intervention recommendation."""

    resource: str                       # Resource to modify
    action: str                         # "increase", "decrease", "rebalance"
    magnitude: int                      # Suggested change amount
    rationale: str                      # Why this recommendation
    expected_impact: dict[str, float]   # Estimated metric improvements
    confidence: str                     # "high", "medium", "low"
    cost_indicator: str                 # "low", "medium", "high"


@dataclass
class MetricsSummary:
    """Standardized metrics container for agent consumption.

    This is the primary interface between simulation output and agents.
    Agents should ONLY depend on this contract, never on internal
    ResultsCollector implementation details.
    """

    # Core identifiers
    scenario_name: str
    run_timestamp: str
    n_replications: int

    # Demand metrics
    arrivals: float
    arrivals_by_priority: dict[str, float]
    arrivals_by_mode: dict[str, float]

    # Wait time metrics (minutes)
    mean_triage_wait: float
    mean_treatment_wait: float
    p95_treatment_wait: float
    mean_system_time: float
    p95_system_time: float
    p_delay: float  # Proportion who waited for treatment

    # Utilization metrics (0-1 scale)
    util_triage: float
    util_ed_bays: float
    util_itu: float
    util_ward: float
    util_theatre: float

    # Downstream metrics
    itu_admissions: float
    mean_itu_wait: float
    ward_admissions: float
    mean_ward_wait: float
    theatre_admissions: float
    mean_theatre_wait: float

    # Boarding/blocking metrics
    mean_boarding_time: float
    p_boarding: float

    # Handover metrics
    mean_handover_delay: float
    max_handover_delay: float

    # Aeromedical metrics (optional)
    aeromed_total: float = 0.0
    aeromed_slots_missed: float = 0.0
    mean_aeromed_slot_wait: float = 0.0

    # Confidence intervals (optional, for display)
    ci_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Raw data access (for advanced agents)
    raw_metrics: dict = field(default_factory=dict)

    @classmethod
    def from_run_results(cls, results: dict, scenario_name: str = "Unnamed") -> "MetricsSummary":
        """Factory method to create MetricsSummary from simulation output."""
        # Extract mean values from replication results
        def mean_of(key: str, default: float = 0.0) -> float:
            values = results.get(key, [default])
            return sum(values) / len(values) if values else default

        return cls(
            scenario_name=scenario_name,
            run_timestamp=pd.Timestamp.now().isoformat(),
            n_replications=len(results.get('arrivals', [1])),
            arrivals=mean_of('arrivals'),
            arrivals_by_priority={
                'P1': mean_of('arrivals_P1'),
                'P2': mean_of('arrivals_P2'),
                'P3': mean_of('arrivals_P3'),
                'P4': mean_of('arrivals_P4'),
            },
            arrivals_by_mode={
                'ambulance': mean_of('arrivals_ambulance', 0),
                'helicopter': mean_of('arrivals_helicopter', 0),
                'walk_in': mean_of('arrivals_walkin', 0),
            },
            mean_triage_wait=mean_of('mean_triage_wait'),
            mean_treatment_wait=mean_of('mean_treatment_wait'),
            p95_treatment_wait=mean_of('p95_treatment_wait'),
            mean_system_time=mean_of('mean_system_time'),
            p95_system_time=mean_of('p95_system_time'),
            p_delay=mean_of('p_delay'),
            util_triage=mean_of('util_triage'),
            util_ed_bays=mean_of('util_ed_bays'),
            util_itu=mean_of('util_itu', 0),
            util_ward=mean_of('util_ward', 0),
            util_theatre=mean_of('util_theatre', 0),
            itu_admissions=mean_of('itu_admissions', 0),
            mean_itu_wait=mean_of('mean_itu_wait', 0),
            ward_admissions=mean_of('ward_admissions', 0),
            mean_ward_wait=mean_of('mean_ward_wait', 0),
            theatre_admissions=mean_of('theatre_admissions', 0),
            mean_theatre_wait=mean_of('mean_theatre_wait', 0),
            mean_boarding_time=mean_of('mean_boarding_time', 0),
            p_boarding=mean_of('p_boarding', 0),
            mean_handover_delay=mean_of('mean_handover_delay', 0),
            max_handover_delay=mean_of('max_handover_delay', 0),
            aeromed_total=mean_of('aeromed_total', 0),
            aeromed_slots_missed=mean_of('aeromed_slots_missed', 0),
            mean_aeromed_slot_wait=mean_of('mean_aeromed_slot_wait', 0),
            raw_metrics=results,
        )


class ClinicalAgent(Protocol):
    """Protocol defining the agent interface contract.

    All agents must implement this interface. This enables:
    - Dependency injection for testing
    - Runtime agent swapping (heuristic → LLM)
    - Future microservice decomposition
    """

    @property
    def name(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def description(self) -> str:
        """Human-readable agent purpose."""
        ...

    def analyze(self, metrics: MetricsSummary) -> list[ClinicalInsight]:
        """Core analysis method: metrics → insights.

        Args:
            metrics: Standardized simulation output summary

        Returns:
            List of clinical insights (may be empty)

        Raises:
            AgentExecutionError: If analysis fails unrecoverably
        """
        ...

    def health_check(self) -> bool:
        """Verify agent is operational (e.g., LLM API accessible)."""
        ...


@dataclass
class AgentResult:
    """Container for agent execution result with metadata."""

    agent_name: str
    execution_time_ms: float
    insights: list[ClinicalInsight]
    success: bool
    error_message: Optional[str] = None


class AgentExecutionError(Exception):
    """Raised when an agent fails to execute."""
    pass
```

### 5.3 Clinical Shadow Agent Implementation

```python
# src/faer/agents/shadow.py

from dataclasses import dataclass
from typing import Iterator
from .interface import (
    ClinicalAgent, ClinicalInsight, MetricsSummary,
    Severity, InsightCategory
)


@dataclass
class ClinicalThreshold:
    """Configurable threshold for clinical rule."""
    metric: str
    threshold: float
    operator: str  # "gt", "lt", "gte", "lte"
    severity: Severity
    category: InsightCategory
    title: str
    message_template: str
    recommendation: str


# NHS/Clinical guideline thresholds
NHS_THRESHOLDS = [
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
            "sepsis progression, and adverse outcomes."
        ),
        recommendation="Immediate capacity review required. Consider escalation protocol."
    ),
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
            "and ED boarding is likely to increase."
        ),
        recommendation="Prepare ITU step-down transfers. Alert bed management."
    ),
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
            "Patient flow through department will degrade."
        ),
        recommendation="Activate surge protocol. Consider see-and-treat for minors."
    ),
    ClinicalThreshold(
        metric="p_delay",
        threshold=0.50,
        operator="gt",
        severity=Severity.MEDIUM,
        category=InsightCategory.FLOW_BOTTLENECK,
        title="Majority Patients Delayed",
        message_template=(
            "{value:.0%} of patients experienced treatment delays. "
            "This indicates systemic flow issues rather than isolated incidents."
        ),
        recommendation="Review triage efficiency and treatment bay turnover."
    ),
    ClinicalThreshold(
        metric="mean_handover_delay",
        threshold=30.0,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.BOARDING,
        title="Ambulance Handover Delays",
        message_template=(
            "Mean ambulance handover delay of {value:.0f} minutes. "
            "Crews are unable to respond to new 999 calls while waiting. "
            "Community response times will be impacted."
        ),
        recommendation="Prioritize handover bay clearance. Consider corridor care protocol."
    ),
    ClinicalThreshold(
        metric="mean_boarding_time",
        threshold=60.0,
        operator="gt",
        severity=Severity.HIGH,
        category=InsightCategory.BOARDING,
        title="Extended ED Boarding",
        message_template=(
            "Patients boarding in ED for average {value:.0f} minutes awaiting "
            "downstream beds. This ties up ED capacity and delays new arrivals."
        ),
        recommendation="Escalate bed management. Review discharge timing."
    ),
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
            "consuming ward bed-days that could be freed."
        ),
        recommendation="Review aeromed scheduling. Consider additional slots."
    ),
]


class HeuristicShadowAgent:
    """Rule-based clinical shadow agent using configurable thresholds.

    This agent applies clinical guidelines as threshold rules against
    simulation metrics. It's deterministic and testable, serving as
    the foundation before LLM integration.

    Example usage:
        agent = HeuristicShadowAgent()
        insights = agent.analyze(metrics_summary)
        for insight in insights:
            print(f"[{insight.severity}] {insight.title}")
    """

    def __init__(self, thresholds: list[ClinicalThreshold] | None = None):
        """Initialize with threshold rules.

        Args:
            thresholds: Custom thresholds. Defaults to NHS_THRESHOLDS.
        """
        self.thresholds = thresholds or NHS_THRESHOLDS

    @property
    def name(self) -> str:
        return "heuristic_shadow"

    @property
    def description(self) -> str:
        return "Rule-based clinical risk detector using NHS/clinical thresholds"

    def analyze(self, metrics: MetricsSummary) -> list[ClinicalInsight]:
        """Apply all threshold rules and return triggered insights."""
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

    def _evaluate_thresholds(self, metrics: MetricsSummary) -> Iterator[ClinicalInsight]:
        """Evaluate single-metric threshold rules."""
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

    def _evaluate_compound_rules(self, metrics: MetricsSummary) -> Iterator[ClinicalInsight]:
        """Evaluate multi-metric compound rules.

        These rules detect risk patterns that require multiple metrics
        to be in certain states simultaneously.
        """
        # Compound rule: High acuity + long waits = severe risk
        resus_ratio = metrics.arrivals_by_priority.get('P1', 0) / max(metrics.arrivals, 1)
        if resus_ratio > 0.08 and metrics.mean_treatment_wait > 60:
            yield ClinicalInsight(
                severity=Severity.CRITICAL,
                category=InsightCategory.PATIENT_SAFETY,
                title="High Acuity with Treatment Delays",
                message=(
                    f"Resuscitation cases represent {resus_ratio:.1%} of arrivals "
                    f"while mean treatment wait is {metrics.mean_treatment_wait:.0f} minutes. "
                    "This combination significantly elevates mortality risk for critical patients."
                ),
                impact_metric="mortality_risk",
                evidence={
                    "resus_ratio": resus_ratio,
                    "mean_treatment_wait": metrics.mean_treatment_wait,
                },
                recommendation="Immediate senior clinical review. Consider divert protocol.",
                source_agent=self.name,
            )

        # Compound rule: ITU near capacity + theatre active = downstream gridlock
        if metrics.util_itu > 0.85 and metrics.util_theatre > 0.70:
            yield ClinicalInsight(
                severity=Severity.HIGH,
                category=InsightCategory.DOWNSTREAM,
                title="Downstream Gridlock Risk",
                message=(
                    f"ITU at {metrics.util_itu:.0%} utilization with theatre at "
                    f"{metrics.util_theatre:.0%}. Post-operative patients may not have "
                    "ITU beds, causing theatre cancellations and ED boarding cascade."
                ),
                impact_metric="flow_cascade",
                evidence={
                    "util_itu": metrics.util_itu,
                    "util_theatre": metrics.util_theatre,
                },
                recommendation="Review elective theatre schedule. Prepare ITU step-downs.",
                source_agent=self.name,
            )

        # Compound rule: Low triage util but high treatment wait = processing bottleneck
        if metrics.util_triage < 0.50 and metrics.mean_treatment_wait > 90:
            yield ClinicalInsight(
                severity=Severity.MEDIUM,
                category=InsightCategory.FLOW_BOTTLENECK,
                title="Treatment Bottleneck Identified",
                message=(
                    f"Triage utilization is only {metrics.util_triage:.0%} but treatment "
                    f"wait averages {metrics.mean_treatment_wait:.0f} minutes. "
                    "Bottleneck is in treatment bays, not triage assessment."
                ),
                impact_metric="bottleneck_location",
                evidence={
                    "util_triage": metrics.util_triage,
                    "mean_treatment_wait": metrics.mean_treatment_wait,
                },
                recommendation="Focus capacity increase on ED treatment bays, not triage.",
                source_agent=self.name,
            )

    @staticmethod
    def _check_threshold(value: float, threshold: float, operator: str) -> bool:
        """Check if value triggers threshold based on operator."""
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
        }
        return ops.get(operator, lambda v, t: False)(value, threshold)

    def health_check(self) -> bool:
        """Heuristic agent is always healthy (no external dependencies)."""
        return True
```

### 5.4 Agent Orchestrator

```python
# src/faer/agents/orchestrator.py

import time
import logging
from dataclasses import dataclass, field
from typing import Callable
from .interface import (
    ClinicalAgent, ClinicalInsight, MetricsSummary,
    AgentResult, AgentExecutionError, Severity
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for agent orchestration."""
    timeout_ms: float = 5000.0          # Max time per agent
    fail_open: bool = True              # Continue on agent failure
    deduplicate_insights: bool = True   # Remove duplicate insights
    max_insights_per_agent: int = 10    # Limit insights per agent


class AgentOrchestrator:
    """Coordinates multiple agents and aggregates their outputs.

    The orchestrator provides:
    - Parallel/sequential agent execution
    - Timeout handling
    - Failure isolation (fail-open by default)
    - Insight deduplication
    - Execution metrics

    Example usage:
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())
        orchestrator.register(CapacityAdvisorAgent())

        results = orchestrator.run_all(metrics_summary)
        all_insights = results.get_all_insights()
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        self._agents: dict[str, ClinicalAgent] = {}
        self._hooks: list[Callable[[AgentResult], None]] = []

    def register(self, agent: ClinicalAgent) -> None:
        """Register an agent with the orchestrator."""
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    def unregister(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator."""
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    def add_result_hook(self, hook: Callable[[AgentResult], None]) -> None:
        """Add a hook called after each agent execution."""
        self._hooks.append(hook)

    def run_all(self, metrics: MetricsSummary) -> "OrchestratorResult":
        """Execute all registered agents and collect results.

        Args:
            metrics: Simulation output to analyze

        Returns:
            OrchestratorResult with all agent outputs and metadata
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

    def _run_agent(self, agent: ClinicalAgent, metrics: MetricsSummary) -> AgentResult:
        """Execute a single agent with error handling and timing."""
        start_time = time.time()

        try:
            # Health check first
            if not agent.health_check():
                raise AgentExecutionError(f"Agent {agent.name} failed health check")

            # Run analysis
            insights = agent.analyze(metrics)

            # Apply limits
            if len(insights) > self.config.max_insights_per_agent:
                insights = insights[:self.config.max_insights_per_agent]
                logger.warning(
                    f"Agent {agent.name} produced {len(insights)} insights, "
                    f"truncated to {self.config.max_insights_per_agent}"
                )

            execution_time = (time.time() - start_time) * 1000

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
                raise

            return AgentResult(
                agent_name=agent.name,
                execution_time_ms=execution_time,
                insights=[],
                success=False,
                error_message=str(e),
            )


@dataclass
class OrchestratorResult:
    """Aggregated results from all agents."""

    agent_results: list[AgentResult]
    config: OrchestratorConfig

    def get_all_insights(self) -> list[ClinicalInsight]:
        """Get all insights from all agents, optionally deduplicated."""
        all_insights = []
        for result in self.agent_results:
            if result.success:
                all_insights.extend(result.insights)

        if self.config.deduplicate_insights:
            # Deduplicate by title (keep first occurrence)
            seen_titles = set()
            unique_insights = []
            for insight in all_insights:
                if insight.title not in seen_titles:
                    seen_titles.add(insight.title)
                    unique_insights.append(insight)
            return unique_insights

        return all_insights

    def get_insights_by_severity(self, severity: Severity) -> list[ClinicalInsight]:
        """Filter insights by severity level."""
        return [i for i in self.get_all_insights() if i.severity == severity]

    def get_critical_insights(self) -> list[ClinicalInsight]:
        """Get only CRITICAL severity insights."""
        return self.get_insights_by_severity(Severity.CRITICAL)

    @property
    def total_execution_time_ms(self) -> float:
        """Total time across all agents."""
        return sum(r.execution_time_ms for r in self.agent_results)

    @property
    def all_succeeded(self) -> bool:
        """Check if all agents completed successfully."""
        return all(r.success for r in self.agent_results)

    @property
    def summary(self) -> dict:
        """Summary statistics for logging/display."""
        insights = self.get_all_insights()
        return {
            "agents_run": len(self.agent_results),
            "agents_succeeded": sum(1 for r in self.agent_results if r.success),
            "total_insights": len(insights),
            "critical_count": len([i for i in insights if i.severity == Severity.CRITICAL]),
            "high_count": len([i for i in insights if i.severity == Severity.HIGH]),
            "execution_time_ms": self.total_execution_time_ms,
        }
```

---

## 6. Implementation Phases

### Phase A: Foundation (Days 1-2)

**Goal**: Working agent infrastructure with deterministic heuristics

**Deliverables**:
1. `agents/interface.py` - Data models and protocols
2. `agents/shadow.py` - HeuristicShadowAgent with NHS thresholds
3. `agents/orchestrator.py` - Agent coordination
4. `tests/test_agents/` - Unit tests for all components
5. `notebooks/agent_prototyping.ipynb` - Interactive testing

**Acceptance Criteria**:
- [ ] `HeuristicShadowAgent.analyze(metrics)` returns insights for known-bad scenarios
- [ ] Agent produces zero insights for "healthy" scenario
- [ ] All tests pass with deterministic inputs

**Integration**:
```python
# Minimal integration test
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.interface import MetricsSummary

# Create "bad" scenario metrics
bad_metrics = MetricsSummary(
    scenario_name="stress_test",
    run_timestamp="2026-01-10T12:00:00",
    n_replications=10,
    arrivals=300,
    arrivals_by_priority={'P1': 30, 'P2': 90, 'P3': 120, 'P4': 60},
    arrivals_by_mode={'ambulance': 200, 'helicopter': 20, 'walk_in': 80},
    mean_triage_wait=15.0,
    mean_treatment_wait=120.0,  # Bad!
    p95_treatment_wait=280.0,   # Very bad! (>240 threshold)
    mean_system_time=240.0,
    p95_system_time=400.0,
    p_delay=0.65,               # Bad! (>0.5 threshold)
    util_triage=0.45,
    util_ed_bays=0.92,          # Bad! (>0.85 threshold)
    util_itu=0.88,
    util_ward=0.75,
    util_theatre=0.60,
    itu_admissions=25,
    mean_itu_wait=45.0,
    ward_admissions=80,
    mean_ward_wait=30.0,
    theatre_admissions=15,
    mean_theatre_wait=60.0,
    mean_boarding_time=45.0,
    p_boarding=0.30,
    mean_handover_delay=25.0,
    max_handover_delay=90.0,
)

agent = HeuristicShadowAgent()
insights = agent.analyze(bad_metrics)

assert len(insights) > 0, "Should detect issues in bad scenario"
assert any(i.severity.value == "CRITICAL" for i in insights), "Should have CRITICAL alert"
print(f"Detected {len(insights)} issues:")
for i in insights:
    print(f"  [{i.severity.value}] {i.title}")
```

### Phase B: Streamlit Integration (Days 3-4)

**Goal**: Agent insights displayed in UI after simulation run

**Deliverables**:
1. `app/pages/9_Insights.py` - New insights page
2. Modifications to `app/pages/5_Run.py` - Trigger agents after simulation
3. Insight display components (cards, filters, severity badges)

**UI Wireframe**:
```
┌─────────────────────────────────────────────────────────────┐
│  🏥 FAER Clinical Insights                    [Run: 12:34] │
├─────────────────────────────────────────────────────────────┤
│  Filter: [All ▼] [CRITICAL ●] [HIGH ●] [MEDIUM ○] [LOW ○]  │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🔴 CRITICAL: 4-Hour Standard Breach                   │  │
│  │                                                       │  │
│  │ P95 treatment wait of 280 minutes exceeds NHS 4-hour │  │
│  │ standard. Patients waiting this long face increased  │  │
│  │ risk of deterioration, sepsis progression, and       │  │
│  │ adverse outcomes.                                    │  │
│  │                                                       │  │
│  │ Evidence: p95_treatment_wait = 280 (threshold: 240)  │  │
│  │                                                       │  │
│  │ 💡 Immediate capacity review required. Consider      │  │
│  │    escalation protocol.                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 🟠 HIGH: ED Bay Saturation                            │  │
│  │ ...                                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Streamlit Integration Code**:
```python
# app/pages/9_Insights.py

import streamlit as st
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.orchestrator import AgentOrchestrator
from faer.agents.interface import MetricsSummary, Severity

st.set_page_config(page_title="Clinical Insights", page_icon="🔍")
st.title("🔍 Clinical Shadow Analysis")

# Check for run results
if 'run_results' not in st.session_state:
    st.warning("No simulation results available. Run a scenario first.")
    st.page_link("pages/5_Run.py", label="Go to Run page")
    st.stop()

# Convert results to MetricsSummary
metrics = MetricsSummary.from_run_results(
    st.session_state['run_results'],
    scenario_name=st.session_state.get('scenario_name', 'Unnamed')
)

# Run agents
@st.cache_data
def run_agents(_metrics_dict: dict) -> list:
    """Run agents on metrics (cached to avoid re-running)."""
    metrics = MetricsSummary.from_run_results(_metrics_dict)
    orchestrator = AgentOrchestrator()
    orchestrator.register(HeuristicShadowAgent())
    result = orchestrator.run_all(metrics)
    return result.get_all_insights()

with st.spinner("Analyzing simulation results..."):
    insights = run_agents(st.session_state['run_results'])

# Summary stats
col1, col2, col3, col4 = st.columns(4)
critical = len([i for i in insights if i.severity == Severity.CRITICAL])
high = len([i for i in insights if i.severity == Severity.HIGH])
medium = len([i for i in insights if i.severity == Severity.MEDIUM])
low = len([i for i in insights if i.severity == Severity.LOW])

col1.metric("🔴 Critical", critical)
col2.metric("🟠 High", high)
col3.metric("🟡 Medium", medium)
col4.metric("🟢 Low", low)

# Severity filter
st.divider()
severity_filter = st.multiselect(
    "Filter by severity",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
    default=["CRITICAL", "HIGH", "MEDIUM"]
)

# Display insights
filtered = [i for i in insights if i.severity.value in severity_filter]

if not filtered:
    st.success("✅ No issues detected at selected severity levels")
else:
    for insight in filtered:
        severity_colors = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🟢",
            Severity.INFO: "🔵",
        }
        icon = severity_colors.get(insight.severity, "⚪")

        with st.expander(f"{icon} **{insight.severity.value}**: {insight.title}", expanded=(insight.severity == Severity.CRITICAL)):
            st.write(insight.message)

            st.caption("**Evidence**")
            evidence_str = ", ".join(f"`{k}={v:.2f}`" for k, v in insight.evidence.items())
            st.code(evidence_str)

            if insight.recommendation:
                st.info(f"💡 **Recommendation**: {insight.recommendation}")
```

### Phase C: Capacity Advisor (Days 5-6)

**Goal**: Recommendations for capacity interventions

**Deliverables**:
1. `agents/capacity.py` - CapacityAdvisorAgent
2. Little's Law heuristics for impact estimation
3. Integration with insights page

### Phase D: LLM Integration (Days 7-10)

**Goal**: Natural language narratives and enhanced analysis

**Deliverables**:
1. `agents/narrative.py` - LLM-powered narrative agent
2. Prompt templates in `agents/prompts/`
3. API key configuration (environment variables)
4. Fallback to heuristics if LLM unavailable

**LLM Agent Skeleton**:
```python
# src/faer/agents/narrative.py

import os
from dataclasses import dataclass
from typing import Optional
from .interface import ClinicalInsight, MetricsSummary

# Import will depend on chosen LLM provider
# from openai import OpenAI
# from anthropic import Anthropic


@dataclass
class NarrativeReport:
    """Structured narrative report."""
    executive_summary: str
    risk_assessment: str
    recommendations: str
    confidence_note: str
    source_metrics: dict


class NarrativeAgent:
    """LLM-powered narrative generation agent.

    Transforms simulation metrics and clinical insights into
    natural language reports suitable for executive briefings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def name(self) -> str:
        return "narrative_agent"

    @property
    def description(self) -> str:
        return "LLM-powered narrative report generator"

    def health_check(self) -> bool:
        """Verify API key is configured and service is accessible."""
        if not self.api_key:
            return False
        # Could add API ping here
        return True

    def generate_narrative(
        self,
        metrics: MetricsSummary,
        insights: list[ClinicalInsight],
    ) -> NarrativeReport:
        """Generate narrative report from metrics and insights."""

        if not self.health_check():
            return self._fallback_narrative(metrics, insights)

        prompt = self._build_prompt(metrics, insights)

        # LLM call would go here
        # response = self._client.chat.completions.create(...)

        # For now, return fallback
        return self._fallback_narrative(metrics, insights)

    def _build_prompt(self, metrics: MetricsSummary, insights: list[ClinicalInsight]) -> str:
        """Construct LLM prompt from metrics and insights."""

        insights_text = "\n".join([
            f"- [{i.severity.value}] {i.title}: {i.message}"
            for i in insights
        ])

        return f"""You are a clinical operations analyst writing a briefing for hospital leadership.

## Simulation Results: {metrics.scenario_name}
- Run duration: {metrics.n_replications} replications
- Total arrivals: {metrics.arrivals:.0f}
- ED utilization: {metrics.util_ed_bays:.1%}
- Mean treatment wait: {metrics.mean_treatment_wait:.0f} minutes
- P95 treatment wait: {metrics.p95_treatment_wait:.0f} minutes
- ITU utilization: {metrics.util_itu:.1%}
- ITU admissions: {metrics.itu_admissions:.0f}
- Proportion delayed: {metrics.p_delay:.1%}

## Detected Issues
{insights_text if insights else "No significant issues detected."}

## Task
Write a concise 3-paragraph briefing covering:
1. Overall system performance (1-2 sentences)
2. Key risk areas and their clinical implications (2-3 sentences)
3. Priority recommendations for operations team (2-3 bullet points)

Use clinical terminology appropriate for hospital leadership. Be direct about risks without being alarmist.
"""

    def _fallback_narrative(
        self,
        metrics: MetricsSummary,
        insights: list[ClinicalInsight],
    ) -> NarrativeReport:
        """Generate template-based narrative when LLM unavailable."""

        critical_count = len([i for i in insights if i.severity.value == "CRITICAL"])
        high_count = len([i for i in insights if i.severity.value == "HIGH"])

        if critical_count > 0:
            risk_level = "critical"
            summary = f"Simulation indicates CRITICAL operational risks requiring immediate attention."
        elif high_count > 0:
            risk_level = "elevated"
            summary = f"Simulation indicates elevated operational risks requiring monitoring."
        else:
            risk_level = "acceptable"
            summary = "Simulation indicates acceptable operational performance."

        return NarrativeReport(
            executive_summary=summary,
            risk_assessment=f"Detected {len(insights)} issues ({critical_count} critical, {high_count} high priority).",
            recommendations="Review detected insights for specific recommendations.",
            confidence_note="This is a template-based summary. Enable LLM for enhanced narrative.",
            source_metrics={
                "arrivals": metrics.arrivals,
                "util_ed_bays": metrics.util_ed_bays,
                "mean_treatment_wait": metrics.mean_treatment_wait,
            }
        )
```

---

## 7. Integration Points

### 7.1 Simulation → Agents

The agent layer consumes simulation output via `MetricsSummary`:

```python
# In app/pages/5_Run.py (after simulation completes)

from faer.agents.interface import MetricsSummary
from faer.agents.orchestrator import AgentOrchestrator
from faer.agents.shadow import HeuristicShadowAgent

# Existing code runs simulation
results = multiple_replications(scenario, n_reps=n_reps)
st.session_state['run_results'] = results

# NEW: Run agent analysis
metrics = MetricsSummary.from_run_results(results, scenario_name="Current Run")

orchestrator = AgentOrchestrator()
orchestrator.register(HeuristicShadowAgent())
# orchestrator.register(CapacityAdvisorAgent())  # Phase C
# orchestrator.register(NarrativeAgent())        # Phase D

agent_results = orchestrator.run_all(metrics)
st.session_state['agent_insights'] = agent_results.get_all_insights()
st.session_state['agent_summary'] = agent_results.summary
```

### 7.2 UI Integration Points

| Page | Integration | Priority |
|------|-------------|----------|
| `5_Run.py` | Trigger agents after run | P0 |
| `6_Results.py` | Show insight count badge | P1 |
| `9_Insights.py` | Full insights display | P0 |
| `7_Compare.py` | Comparative insights | P2 |

### 7.3 Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FullScenario ──▶ run_full_simulation() ──▶ results: dict       │
│                                               │                  │
│                                               ▼                  │
│                              MetricsSummary.from_run_results()   │
│                                               │                  │
│                                               ▼                  │
│                                        MetricsSummary            │
│                                               │                  │
│                    ┌──────────────────────────┼────────────┐    │
│                    │                          │            │    │
│                    ▼                          ▼            ▼    │
│            HeuristicShadow           CapacityAdvisor  Narrative │
│                    │                          │            │    │
│                    ▼                          ▼            ▼    │
│            ClinicalInsight[]        Recommendation[]    Report  │
│                    │                          │            │    │
│                    └──────────────────────────┼────────────┘    │
│                                               │                  │
│                                               ▼                  │
│                                      OrchestratorResult         │
│                                               │                  │
│                                               ▼                  │
│                                   st.session_state['insights']  │
│                                               │                  │
│                                               ▼                  │
│                                        9_Insights.py            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. API Contracts

### 8.1 Agent Protocol

All agents must implement:

```python
class ClinicalAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def analyze(self, metrics: MetricsSummary) -> list[ClinicalInsight]: ...

    def health_check(self) -> bool: ...
```

### 8.2 Insight Schema

```python
@dataclass(frozen=True)
class ClinicalInsight:
    severity: Severity              # Enum: CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: InsightCategory       # Enum: wait_time, capacity, safety, etc.
    title: str                      # ≤80 characters
    message: str                    # Detailed explanation
    impact_metric: str              # Primary affected metric
    evidence: dict[str, float]      # Supporting data
    recommendation: Optional[str]   # Actionable guidance
    source_agent: str               # Agent that generated this
```

### 8.3 Orchestrator Result

```python
@dataclass
class OrchestratorResult:
    agent_results: list[AgentResult]

    def get_all_insights(self) -> list[ClinicalInsight]: ...
    def get_insights_by_severity(self, severity: Severity) -> list[ClinicalInsight]: ...
    def get_critical_insights(self) -> list[ClinicalInsight]: ...

    @property
    def total_execution_time_ms(self) -> float: ...
    @property
    def all_succeeded(self) -> bool: ...
    @property
    def summary(self) -> dict: ...
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_agents/test_shadow_agent.py

import pytest
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.interface import MetricsSummary, Severity


@pytest.fixture
def healthy_metrics() -> MetricsSummary:
    """Metrics that should not trigger any alerts."""
    return MetricsSummary(
        scenario_name="healthy",
        run_timestamp="2026-01-10T00:00:00",
        n_replications=10,
        arrivals=200,
        arrivals_by_priority={'P1': 10, 'P2': 60, 'P3': 80, 'P4': 50},
        arrivals_by_mode={'ambulance': 100, 'helicopter': 10, 'walk_in': 90},
        mean_triage_wait=5.0,
        mean_treatment_wait=30.0,
        p95_treatment_wait=90.0,      # Well under 240
        mean_system_time=120.0,
        p95_system_time=200.0,
        p_delay=0.25,                 # Under 0.5
        util_triage=0.40,
        util_ed_bays=0.65,            # Under 0.85
        util_itu=0.70,                # Under 0.90
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
        mean_handover_delay=10.0,     # Under 30
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
        arrivals_by_priority={'P1': 35, 'P2': 100, 'P3': 110, 'P4': 55},
        arrivals_by_mode={'ambulance': 200, 'helicopter': 25, 'walk_in': 75},
        mean_triage_wait=20.0,
        mean_treatment_wait=150.0,
        p95_treatment_wait=300.0,     # Over 240 - CRITICAL
        mean_system_time=280.0,
        p95_system_time=450.0,
        p_delay=0.70,                 # Over 0.5 - triggers alert
        util_triage=0.55,
        util_ed_bays=0.95,            # Over 0.85 - HIGH
        util_itu=0.94,                # Over 0.90 - HIGH
        util_ward=0.80,
        util_theatre=0.75,
        itu_admissions=30,
        mean_itu_wait=90.0,
        ward_admissions=70,
        mean_ward_wait=60.0,
        theatre_admissions=20,
        mean_theatre_wait=45.0,
        mean_boarding_time=75.0,      # Over 60 - HIGH
        p_boarding=0.40,
        mean_handover_delay=45.0,     # Over 30 - HIGH
        max_handover_delay=120.0,
    )


class TestHeuristicShadowAgent:
    """Test suite for HeuristicShadowAgent."""

    def test_healthy_scenario_no_alerts(self, healthy_metrics):
        """Healthy scenario should produce no/minimal alerts."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(healthy_metrics)

        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) == 0, "Healthy scenario should have no CRITICAL alerts"

    def test_critical_scenario_detects_issues(self, critical_metrics):
        """Critical scenario should trigger multiple alerts."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        assert len(insights) > 0, "Should detect issues"

        critical = [i for i in insights if i.severity == Severity.CRITICAL]
        assert len(critical) >= 1, "Should have at least one CRITICAL alert"

        # Should detect 4-hour breach
        titles = [i.title for i in insights]
        assert "4-Hour Standard Breach" in titles

    def test_insights_sorted_by_severity(self, critical_metrics):
        """Insights should be sorted CRITICAL first."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        if len(insights) > 1:
            # First insight should be most severe
            severities = [i.severity for i in insights]
            severity_values = [s.value for s in severities]
            assert severity_values == sorted(severity_values, key=lambda x: {
                "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4
            }.get(x, 5))

    def test_insight_has_required_fields(self, critical_metrics):
        """Each insight should have all required fields."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        for insight in insights:
            assert insight.severity is not None
            assert insight.category is not None
            assert len(insight.title) > 0
            assert len(insight.title) <= 80
            assert len(insight.message) > 0
            assert len(insight.impact_metric) > 0
            assert isinstance(insight.evidence, dict)
            assert insight.source_agent == "heuristic_shadow"

    def test_compound_rule_high_acuity_long_wait(self, critical_metrics):
        """Should detect high acuity + long wait compound risk."""
        agent = HeuristicShadowAgent()
        insights = agent.analyze(critical_metrics)

        titles = [i.title for i in insights]
        assert "High Acuity with Treatment Delays" in titles

    def test_agent_health_check(self):
        """Heuristic agent should always pass health check."""
        agent = HeuristicShadowAgent()
        assert agent.health_check() is True

    def test_deterministic_output(self, critical_metrics):
        """Same input should always produce same output."""
        agent = HeuristicShadowAgent()

        insights1 = agent.analyze(critical_metrics)
        insights2 = agent.analyze(critical_metrics)

        assert len(insights1) == len(insights2)
        for i1, i2 in zip(insights1, insights2):
            assert i1.title == i2.title
            assert i1.severity == i2.severity
```

### 9.2 Integration Tests

```python
# tests/test_agents/test_orchestrator.py

import pytest
from faer.agents.orchestrator import AgentOrchestrator, OrchestratorConfig
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.interface import MetricsSummary, Severity


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""

    def test_register_and_run_single_agent(self, critical_metrics):
        """Should run single registered agent."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        assert len(result.agent_results) == 1
        assert result.agent_results[0].success is True
        assert len(result.get_all_insights()) > 0

    def test_fail_open_continues_on_error(self, critical_metrics):
        """Should continue if agent fails (fail_open=True)."""

        class FailingAgent:
            @property
            def name(self): return "failing"
            @property
            def description(self): return "Always fails"
            def analyze(self, m): raise Exception("Intentional failure")
            def health_check(self): return True

        config = OrchestratorConfig(fail_open=True)
        orchestrator = AgentOrchestrator(config)
        orchestrator.register(FailingAgent())
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        # Should have 2 results
        assert len(result.agent_results) == 2

        # One failed, one succeeded
        assert result.agent_results[0].success is False
        assert result.agent_results[1].success is True

        # Should still get insights from working agent
        assert len(result.get_all_insights()) > 0

    def test_execution_time_tracked(self, critical_metrics):
        """Should track execution time."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)

        assert result.total_execution_time_ms > 0
        assert result.agent_results[0].execution_time_ms > 0

    def test_summary_statistics(self, critical_metrics):
        """Should produce correct summary stats."""
        orchestrator = AgentOrchestrator()
        orchestrator.register(HeuristicShadowAgent())

        result = orchestrator.run_all(critical_metrics)
        summary = result.summary

        assert summary['agents_run'] == 1
        assert summary['agents_succeeded'] == 1
        assert summary['total_insights'] > 0
        assert 'critical_count' in summary
```

### 9.3 End-to-End Test

```python
# tests/test_agents/test_e2e.py

import pytest
from faer.model.full_model import run_full_simulation
from faer.core.scenario import FullScenario
from faer.agents.interface import MetricsSummary
from faer.agents.shadow import HeuristicShadowAgent


class TestEndToEnd:
    """End-to-end tests: simulation → agents → insights."""

    def test_simulation_to_insights_pipeline(self):
        """Full pipeline from scenario to insights."""
        # Configure a stressed scenario
        scenario = FullScenario(
            run_length=480.0,  # 8 hours
            n_triage=1,        # Understaffed
            n_ed_bays=4,       # Limited bays
            arrival_rate=12.0, # High demand
            random_seed=42,
        )

        # Run simulation
        results = run_full_simulation(scenario)

        # Convert to metrics
        metrics = MetricsSummary.from_run_results(
            {'arrivals': [results['arrivals']],
             'p95_treatment_wait': [results['p95_treatment_wait']],
             # ... other metrics
            },
            scenario_name="stress_test"
        )

        # Run agent
        agent = HeuristicShadowAgent()
        insights = agent.analyze(metrics)

        # Should detect issues in stressed scenario
        # (Exact assertions depend on scenario configuration)
        assert isinstance(insights, list)
```

---

## 10. Observability

### 10.1 Logging

```python
import logging

# Configure agent logging
logging.getLogger("faer.agents").setLevel(logging.INFO)

# Log format includes agent name and execution context
formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
```

### 10.2 Metrics (Future)

For production deployment, consider:
- Agent execution time (histogram)
- Insights generated per run (counter)
- Agent failures (counter by agent name)
- LLM API latency (histogram)

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Detection accuracy | ≥80% of "known bad" scenarios flagged | Test suite with labeled scenarios |
| False positive rate | ≤10% | Manual review of flagged scenarios |
| Agent execution time | ≤500ms (heuristic), ≤3s (LLM) | Performance tests |
| User adoption | 70% of runs have insights viewed | Analytics (future) |
| Code coverage | ≥90% for agent module | pytest-cov |

---

## 12. Appendices

### A. Glossary

| Term | Definition |
|------|------------|
| Clinical Shadow | Agent that observes simulation output and identifies risks |
| Insight | Structured clinical observation with severity and evidence |
| Orchestrator | Coordinator that runs multiple agents and aggregates results |
| Fail-open | Design pattern where component failures don't crash the system |
| Interface-first | Building modules as if they were services, deployed as monolith |

### B. NHS Clinical Thresholds Reference

| Metric | Threshold | Source |
|--------|-----------|--------|
| Treatment wait | ≤4 hours (240 min) | NHS Constitution |
| Ambulance handover | ≤15 minutes | NHS England |
| Trolley wait | ≤12 hours | NHS operational |
| DTaC (Decision to Admit → Admit) | ≤4 hours | NHS operational |

### C. Dependencies

```toml
# Additional dependencies for agent layer
[project.optional-dependencies]
agents = [
    "openai>=1.0.0",      # For LLM integration
    "anthropic>=0.18.0",  # Alternative LLM
    "tiktoken>=0.5.0",    # Token counting
]
```

### D. Future Enhancements (Out of Scope)

1. **Real-time intervention**: Agents modify running simulation
2. **Multi-agent collaboration**: Agents debate insights
3. **Scenario optimization**: Agents suggest parameter changes
4. **Historical memory**: Agents learn from past runs
5. **External data fusion**: EHR, staffing, weather integration

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-10 | Engineering | Initial PRD |

---

*End of Document*
