"""Confidence interval, sensitivity analysis and experimentation tools.

Phase 6 additions:
- sensitivity_sweep(): Vary one parameter, measure metric impact
- find_breaking_point(): Binary search for threshold crossing
- SweepResult, BreakingPointResult: Structured result classes
"""

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import stats

from faer.core.scenario import Scenario, FullScenario
from faer.model.processes import run_simulation
from faer.model.full_model import run_full_simulation


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


# =============================================================================
# Phase 6: Sensitivity Analysis and Breaking Point Functions
# =============================================================================


@dataclass
class SweepResult:
    """Result of a sensitivity sweep.

    Attributes:
        parameter: Name of the parameter that was varied.
        values: List of parameter values tested.
        metric: Name of the metric measured.
        results: DataFrame with columns: value, mean, std, ci_lower, ci_upper, n_reps.
    """
    parameter: str
    values: List[float]
    metric: str
    results: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return self.results


@dataclass
class BreakingPointResult:
    """Result of breaking point search.

    Attributes:
        parameter: Name of the parameter varied.
        metric: Name of the metric monitored.
        threshold: The threshold value being searched for.
        direction: 'above' or 'below' indicating threshold crossing direction.
        breaking_point: The parameter value where threshold is crossed.
        metric_at_break: The metric value at the breaking point.
        confidence_interval: (lower, upper) CI for the metric at breaking point.
        search_history: List of dicts recording the binary search progress.
    """
    parameter: str
    metric: str
    threshold: float
    direction: str
    breaking_point: float
    metric_at_break: float
    confidence_interval: tuple
    search_history: List[Dict]

    def summary(self) -> str:
        """Human-readable summary of the breaking point."""
        return (
            f"The {self.metric} crosses {self.threshold} "
            f"when {self.parameter} reaches {self.breaking_point:.2f}\n"
            f"At this point, {self.metric} = {self.metric_at_break:.3f} "
            f"(95% CI: {self.confidence_interval[0]:.3f} - {self.confidence_interval[1]:.3f})"
        )


def set_nested_param(
    scenario: Union[Scenario, FullScenario],
    param_path: str,
    value: Any
) -> Union[Scenario, FullScenario]:
    """Set a parameter on scenario, supporting nested paths.

    Creates a deep copy of the scenario and sets the specified parameter.

    Args:
        scenario: The scenario to modify (Scenario or FullScenario).
        param_path: Dot-separated path to the parameter.
            Examples: 'n_ed_bays', 'demand_multiplier', 'node_configs.SURGERY.capacity'
        value: The value to set.

    Returns:
        A new scenario instance with the parameter set.

    Raises:
        ValueError: If the parameter path cannot be navigated or set.
    """
    scenario = copy.deepcopy(scenario)

    parts = param_path.split('.')
    obj = scenario

    for part in parts[:-1]:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise ValueError(f"Cannot navigate to {part} in {param_path}")

    final_attr = parts[-1]
    if hasattr(obj, final_attr):
        setattr(obj, final_attr, value)
    elif isinstance(obj, dict):
        obj[final_attr] = value
    else:
        raise ValueError(f"Cannot set {final_attr} in {param_path}")

    return scenario


def _run_single_rep(scenario: Union[Scenario, FullScenario], seed_offset: int) -> Dict[str, float]:
    """Run single replication with seed offset (helper for parallel execution)."""
    scenario = copy.deepcopy(scenario)
    scenario.random_seed = scenario.random_seed + seed_offset

    if isinstance(scenario, FullScenario):
        # Re-initialize RNG streams after seed change
        scenario.__post_init__()
        results = run_full_simulation(scenario)
    else:
        results = run_simulation(scenario)

    return results


def sensitivity_sweep(
    base_scenario: Union[Scenario, FullScenario],
    param_path: str,
    values: List[float],
    metric: str,
    n_reps: int = 30,
    parallel: bool = False,
    confidence: float = 0.95
) -> SweepResult:
    """Vary one parameter across values, measuring impact on metric.

    Runs multiple replications at each parameter value and computes
    confidence intervals for the metric.

    Args:
        base_scenario: Starting scenario configuration.
        param_path: Parameter to vary (e.g., 'n_ed_bays', 'demand_multiplier').
        values: List of values to test.
        metric: Metric to measure (e.g., 'mean_wait_time', 'p_delay').
        n_reps: Replications per value (default 30 for statistical validity).
        parallel: Use parallel execution (default False - not fully supported).
        confidence: Confidence level for intervals (default 0.95).

    Returns:
        SweepResult with DataFrame containing mean, std, CI for each value.

    Example:
        >>> result = sensitivity_sweep(
        ...     scenario,
        ...     'n_ed_bays',
        ...     values=[10, 15, 20, 25, 30],
        ...     metric='mean_treatment_wait',
        ...     n_reps=30
        ... )
        >>> print(result.to_dataframe())

    Plain Language:
        This function tests "what if" questions by changing one thing at a time.
        For example: "If I add ED beds from 15 to 25, how much do wait times improve?"
    """
    results_data = []

    for value in values:
        # Create scenario with this parameter value
        test_scenario = set_nested_param(base_scenario, param_path, value)

        # Run replications
        metric_values = []
        for i in range(n_reps):
            rep_results = _run_single_rep(test_scenario, i)
            if metric in rep_results:
                metric_values.append(rep_results[metric])

        # Compute statistics
        if metric_values:
            mean = np.mean(metric_values)
            std = np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0.0
            n = len(metric_values)

            # Confidence interval
            if n > 1:
                t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
                ci_half = t_crit * std / np.sqrt(n)
            else:
                ci_half = 0.0

            results_data.append({
                'value': value,
                'mean': mean,
                'std': std,
                'ci_lower': mean - ci_half,
                'ci_upper': mean + ci_half,
                'n_reps': n
            })
        else:
            results_data.append({
                'value': value,
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_reps': 0
            })

    return SweepResult(
        parameter=param_path,
        values=values,
        metric=metric,
        results=pd.DataFrame(results_data)
    )


def find_breaking_point(
    base_scenario: Union[Scenario, FullScenario],
    param_path: str,
    metric: str,
    threshold: float,
    search_range: tuple,
    direction: str = "above",
    n_reps: int = 30,
    tolerance: float = 0.05,
    max_iterations: int = 20
) -> BreakingPointResult:
    """Find parameter value where metric crosses threshold.

    Uses binary search to efficiently find the breaking point where
    a metric crosses a specified threshold.

    Args:
        base_scenario: Starting scenario configuration.
        param_path: Parameter to vary (e.g., 'demand_multiplier').
        metric: Metric to monitor (e.g., 'p_delay').
        threshold: Value to find crossing of.
        search_range: (min, max) range to search.
        direction: 'above' (find where metric exceeds threshold) or
                   'below' (find where metric drops below).
        n_reps: Replications for statistical confidence.
        tolerance: Stop when range narrows to this fraction.
        max_iterations: Maximum search iterations.

    Returns:
        BreakingPointResult with breaking point and confidence.

    Example:
        >>> # At what demand does P(delay) exceed 50%?
        >>> result = find_breaking_point(
        ...     scenario,
        ...     'demand_multiplier',
        ...     'p_delay',
        ...     threshold=0.5,
        ...     search_range=(1.0, 3.0),
        ...     direction='above'
        ... )
        >>> print(result.summary())

    Plain Language:
        Finds exactly when something becomes a problem.
        Example: "At what patient volume do wait times exceed 30 minutes?"
    """
    low, high = search_range
    history = []

    def evaluate(value):
        """Run scenario and get metric mean and CI."""
        test_scenario = set_nested_param(base_scenario, param_path, value)

        metric_values = []
        for i in range(n_reps):
            rep_results = _run_single_rep(test_scenario, i)
            if metric in rep_results:
                metric_values.append(rep_results[metric])

        if not metric_values:
            return None, None, None

        mean = np.mean(metric_values)
        std = np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0.0
        n = len(metric_values)

        if n > 1:
            t_crit = stats.t.ppf(0.975, n - 1)
            ci_half = t_crit * std / np.sqrt(n)
        else:
            ci_half = 0.0

        return mean, (mean - ci_half, mean + ci_half), metric_values

    def crosses_threshold(mean, ci):
        """Check if we're above/below threshold."""
        if direction == "above":
            return mean > threshold
        else:
            return mean < threshold

    # Binary search
    for iteration in range(max_iterations):
        mid = (low + high) / 2
        mean, ci, values = evaluate(mid)

        history.append({
            'iteration': iteration,
            'value': mid,
            'mean': mean,
            'ci': ci,
            'low': low,
            'high': high
        })

        if mean is None:
            break

        # Check if we've converged
        if high > 0 and (high - low) / high < tolerance:
            break

        # Binary search step
        if crosses_threshold(mean, ci):
            high = mid
        else:
            low = mid

    # Final evaluation at breaking point
    breaking_point = (low + high) / 2
    final_mean, final_ci, _ = evaluate(breaking_point)

    return BreakingPointResult(
        parameter=param_path,
        metric=metric,
        threshold=threshold,
        direction=direction,
        breaking_point=breaking_point,
        metric_at_break=final_mean if final_mean is not None else 0.0,
        confidence_interval=final_ci if final_ci is not None else (0.0, 0.0),
        search_history=history
    )


@dataclass
class BottleneckResult:
    """Result of bottleneck analysis.

    Attributes:
        primary_bottleneck: Name of the primary constraining resource.
        bottleneck_score: 0-1 score indicating constraint severity.
        utilisation_ranking: List of (node, utilisation) sorted by utilisation.
        blocking_attribution: Dict mapping node -> % of blocking caused.
        recommendation: Plain language recommendation.
        details: Additional analysis details.
    """
    primary_bottleneck: str
    bottleneck_score: float
    utilisation_ranking: List[tuple]
    blocking_attribution: Dict[str, float]
    recommendation: str
    details: Dict[str, Any]

    def summary(self) -> str:
        """Plain language summary of bottleneck analysis."""
        ranking_str = "\n".join([
            f"  {i+1}. {node}: {util:.0%}"
            for i, (node, util) in enumerate(self.utilisation_ranking[:5])
        ])
        return (
            f"**Primary Bottleneck**: {self.primary_bottleneck}\n\n"
            f"**Recommendation**: {self.recommendation}\n\n"
            f"**Utilisation Ranking**:\n{ranking_str}"
        )


def identify_bottlenecks(
    results: Dict[str, Any],
    scenario: Union[Scenario, FullScenario]
) -> BottleneckResult:
    """Analyse simulation results to identify system bottlenecks.

    A bottleneck is identified by:
    1. High utilisation (>85%)
    2. Causing blocking/boarding upstream
    3. Limiting overall throughput

    Args:
        results: Simulation results dictionary with metrics.
        scenario: Scenario configuration.

    Returns:
        BottleneckResult with primary bottleneck and recommendations.

    Plain Language:
        Finds which part of the hospital is limiting overall performance.
        If Ward is the bottleneck, adding ED beds won't help - patients
        will just board longer in ED.
    """
    metrics = results

    # 1. Gather utilisation data
    utilisation = {}
    util_keys = ['util_triage', 'util_ed_bays', 'util_handover', 'util_ambulance_fleet']
    util_names = ['triage', 'ed_bays', 'handover', 'ambulance_fleet']

    for key, name in zip(util_keys, util_names):
        if key in metrics:
            utilisation[name] = metrics[key]

    # Sort by utilisation (descending)
    util_ranking = sorted(utilisation.items(), key=lambda x: x[1], reverse=True)

    # 2. Simplified blocking attribution
    blocking_attr = {}
    for node, util in util_ranking:
        if util > 0.85:
            blocking_attr[node] = util

    # 3. Identify primary bottleneck
    primary = None
    bottleneck_score = 0

    for node, util in util_ranking:
        if util > 0.85:
            score = util
            # Add context-specific weighting
            if node == 'ed_bays' and metrics.get('mean_treatment_wait', 0) > 30:
                score += 0.1
            if node == 'handover' and metrics.get('util_ambulance_fleet', 0) > 0.7:
                score += 0.1

            if score > bottleneck_score:
                primary = node
                bottleneck_score = score

    # 4. Generate recommendation
    if primary is None:
        recommendation = "No critical bottleneck identified. System has adequate capacity."
    elif primary == 'ed_bays':
        recommendation = (
            "ED bays are the primary constraint. Consider:\n"
            "- Increasing ED bay capacity\n"
            "- Rapid assessment and treatment (RAT) model\n"
            "- Streaming low-acuity patients to primary care\n"
            "- Reducing ED length of stay through pull from wards"
        )
    elif primary == 'triage':
        recommendation = (
            "Triage is the primary constraint. Consider:\n"
            "- Adding triage clinicians\n"
            "- Implementing streaming at front door\n"
            "- Using physician at triage model\n"
            "- Rapid assessment zones"
        )
    elif primary == 'handover':
        recommendation = (
            "Ambulance handover is the primary constraint. Consider:\n"
            "- Dedicated handover team\n"
            "- Fit-to-sit policy\n"
            "- Corridor care protocol (with safety mitigations)\n"
            "- Address downstream ED flow to release handover bays"
        )
    elif primary == 'ambulance_fleet':
        recommendation = (
            "Ambulance fleet is the primary constraint. Consider:\n"
            "- Increasing fleet size\n"
            "- Reducing turnaround time\n"
            "- Addressing handover delays at ED\n"
            "- Alternative transport for low-acuity patients"
        )
    else:
        recommendation = f"Consider increasing capacity at {primary}."

    return BottleneckResult(
        primary_bottleneck=primary or "None",
        bottleneck_score=bottleneck_score,
        utilisation_ranking=util_ranking,
        blocking_attribution=blocking_attr,
        recommendation=recommendation,
        details={'metrics': metrics}
    )
