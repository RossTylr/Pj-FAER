"""Results and metrics layer: event logging, KPI computation, cost modelling."""

from faer.results.collector import ResultsCollector
from faer.results.costs import (
    CostConfig,
    CostBreakdown,
    CostByPriority,
    calculate_costs,
    calculate_patient_cost,
    calculate_costs_by_priority,
    compare_scenario_costs,
    format_currency,
)

__all__ = [
    "ResultsCollector",
    "CostConfig",
    "CostBreakdown",
    "CostByPriority",
    "calculate_costs",
    "calculate_patient_cost",
    "calculate_costs_by_priority",
    "compare_scenario_costs",
    "format_currency",
]
