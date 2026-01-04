"""Experimentation layer: replication runner, CI analysis."""

from faer.experiment.runner import multiple_replications, run_scenario_comparison
from faer.experiment.analysis import (
    compute_ci,
    run_until_precision,
    estimate_required_reps,
    compare_scenarios,
)

__all__ = [
    "multiple_replications",
    "run_scenario_comparison",
    "compute_ci",
    "run_until_precision",
    "estimate_required_reps",
    "compare_scenarios",
]
