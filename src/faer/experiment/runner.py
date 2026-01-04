"""Single and batch simulation runners."""

from typing import Dict, List, Optional, Callable, Any

from faer.core.scenario import Scenario
from faer.model.processes import run_simulation


def multiple_replications(
    scenario: Scenario,
    n_reps: int = 30,
    metric_names: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, List[float]]:
    """Run multiple replications and collect specified metrics.

    Each replication uses a different random seed (base_seed + rep_number)
    to ensure independent samples.

    Args:
        scenario: Base scenario configuration.
        n_reps: Number of replications to run.
        metric_names: List of metric names to collect. If None, collects
            p_delay, mean_queue_time, and utilisation.
        progress_callback: Optional callback(current_rep, total_reps) for
            progress reporting.

    Returns:
        Dictionary mapping metric names to lists of values across replications.
    """
    if metric_names is None:
        metric_names = ["p_delay", "mean_queue_time", "utilisation"]

    results: Dict[str, List[float]] = {name: [] for name in metric_names}

    for rep in range(n_reps):
        # Create scenario with different seed for this rep
        rep_scenario = scenario.clone_with_seed(scenario.random_seed + rep)

        # Run simulation
        run_results = run_simulation(rep_scenario)

        # Collect specified metrics
        for name in metric_names:
            if name in run_results:
                results[name].append(run_results[name])

        # Report progress if callback provided
        if progress_callback is not None:
            progress_callback(rep + 1, n_reps)

    return results


def run_scenario_comparison(
    scenarios: Dict[str, Scenario],
    n_reps: int = 30,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Run multiple scenarios for comparison.

    Args:
        scenarios: Dictionary mapping scenario names to Scenario objects.
        n_reps: Number of replications per scenario.
        metric_names: Metrics to collect.

    Returns:
        Nested dictionary: {scenario_name: {metric_name: [values]}}.
    """
    all_results = {}

    for name, scenario in scenarios.items():
        all_results[name] = multiple_replications(
            scenario, n_reps=n_reps, metric_names=metric_names
        )

    return all_results
