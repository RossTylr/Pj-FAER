"""Confidence interval and precision guidance analysis."""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from faer.core.scenario import Scenario
from faer.model.processes import run_simulation


def compute_ci(values: List[float], confidence: float = 0.95) -> Dict:
    """Compute confidence interval for a metric.

    Args:
        values: List of metric values from replications.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dictionary containing:
        - mean: Sample mean
        - std: Sample standard deviation
        - se: Standard error
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - ci_half_width: Half-width of CI
        - n: Sample size
    """
    n = len(values)
    if n < 2:
        mean = values[0] if n == 1 else 0.0
        return {
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_lower": mean,
            "ci_upper": mean,
            "ci_half_width": 0.0,
            "n": n,
        }

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    se = float(stats.sem(arr))

    # t-critical value for confidence level
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    half_width = t_crit * se

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "ci_half_width": half_width,
        "n": n,
    }


def run_until_precision(
    scenario: Scenario,
    target_metric: str = "p_delay",
    target_half_width: float = 0.02,
    max_reps: int = 200,
    min_reps: int = 10,
    batch_size: int = 5,
    confidence: float = 0.95,
) -> Dict:
    """Run replications until target precision is achieved.

    Runs batches of replications, checking precision after each batch.
    Stops when the CI half-width is less than or equal to the target,
    or when max_reps is reached.

    Args:
        scenario: Base scenario configuration.
        target_metric: Metric to track for precision (default "p_delay").
        target_half_width: Target CI half-width to achieve.
        max_reps: Maximum number of replications to run.
        min_reps: Minimum replications before checking precision.
        batch_size: Number of reps to run between precision checks.
        confidence: Confidence level for CI calculation.

    Returns:
        Dictionary containing:
        - converged: Whether target precision was achieved
        - n_reps: Total replications run
        - values: List of all metric values
        - ci: Final CI statistics
    """
    values: List[float] = []

    for rep in range(max_reps):
        # Create scenario with different seed for this rep
        rep_scenario = scenario.clone_with_seed(scenario.random_seed + rep)

        # Run simulation
        run_results = run_simulation(rep_scenario)
        values.append(run_results[target_metric])

        # Check precision after min_reps, then every batch_size
        if rep >= min_reps - 1 and (rep + 1) % batch_size == 0:
            ci = compute_ci(values, confidence)

            if ci["ci_half_width"] <= target_half_width:
                return {
                    "converged": True,
                    "n_reps": rep + 1,
                    "values": values,
                    "ci": ci,
                }

    # Did not converge within max_reps
    return {
        "converged": False,
        "n_reps": max_reps,
        "values": values,
        "ci": compute_ci(values, confidence),
    }


def estimate_required_reps(
    pilot_values: List[float],
    target_half_width: float,
    confidence: float = 0.95,
) -> int:
    """Estimate replications needed to achieve target precision.

    Uses pilot run data to estimate sample variance and project
    the number of replications needed.

    Args:
        pilot_values: Metric values from pilot replications.
        target_half_width: Desired CI half-width.
        confidence: Confidence level.

    Returns:
        Estimated number of replications needed.
    """
    if len(pilot_values) < 2:
        return 100  # Default estimate

    n = len(pilot_values)
    std = float(np.std(pilot_values, ddof=1))

    # t-critical value (approximate with large sample)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)

    # Required n: (t * std / target)^2
    if target_half_width <= 0:
        return 1000  # Very tight precision

    required_n = (t_crit * std / target_half_width) ** 2
    return max(int(np.ceil(required_n)), n)


def compare_scenarios(
    results_a: Dict[str, List[float]],
    results_b: Dict[str, List[float]],
    metric: str,
    confidence: float = 0.95,
) -> Dict:
    """Compare a metric between two scenarios.

    Computes CIs for both and determines if difference is significant
    (i.e., CIs don't overlap).

    Args:
        results_a: Results from scenario A.
        results_b: Results from scenario B.
        metric: Metric name to compare.
        confidence: Confidence level.

    Returns:
        Dictionary with comparison results.
    """
    ci_a = compute_ci(results_a[metric], confidence)
    ci_b = compute_ci(results_b[metric], confidence)

    # Check if CIs overlap
    overlap = not (ci_a["ci_upper"] < ci_b["ci_lower"] or ci_b["ci_upper"] < ci_a["ci_lower"])

    difference = ci_b["mean"] - ci_a["mean"]
    percent_change = (difference / ci_a["mean"] * 100) if ci_a["mean"] != 0 else 0.0

    return {
        "metric": metric,
        "scenario_a": ci_a,
        "scenario_b": ci_b,
        "difference": difference,
        "percent_change": percent_change,
        "significant": not overlap,
        "confidence": confidence,
    }
