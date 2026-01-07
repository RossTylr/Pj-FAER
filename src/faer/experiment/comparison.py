"""Scenario comparison tools with statistical testing (Phase 6).

This module provides functions for comparing two simulation scenarios
with proper statistical testing to determine if differences are significant.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats

from faer.core.scenario import Scenario, FullScenario
from faer.model.processes import run_simulation
from faer.model.full_model import run_full_simulation


@dataclass
class ComparisonResult:
    """Result of comparing two scenarios.

    Attributes:
        scenario_a_name: Display name for scenario A.
        scenario_b_name: Display name for scenario B.
        metrics: DataFrame with detailed comparison for each metric.
        summary: Plain language summary of the comparison.
    """
    scenario_a_name: str
    scenario_b_name: str
    metrics: pd.DataFrame
    summary: str

    def significant_differences(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return only metrics with significant differences.

        Args:
            alpha: Significance level (default 0.05).

        Returns:
            DataFrame containing only rows where p_value < alpha.
        """
        return self.metrics[self.metrics['p_value'] < alpha]


def _run_replications(
    scenario: Union[Scenario, FullScenario],
    n_reps: int
) -> List[Dict]:
    """Run multiple replications of a scenario.

    Args:
        scenario: The scenario to run.
        n_reps: Number of replications.

    Returns:
        List of result dictionaries, one per replication.
    """
    results = []
    is_full = isinstance(scenario, FullScenario)

    for i in range(n_reps):
        rep_scenario = copy.deepcopy(scenario)
        rep_scenario.random_seed = scenario.random_seed + i

        if is_full:
            rep_scenario.__post_init__()
            result = run_full_simulation(rep_scenario)
        else:
            result = run_simulation(rep_scenario)

        results.append(result)

    return results


def _effect_magnitude(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        String description of effect magnitude.
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _generate_summary(
    df: pd.DataFrame,
    name_a: str,
    name_b: str,
    alpha: float
) -> str:
    """Generate plain language summary of comparison.

    Args:
        df: Comparison results DataFrame.
        name_a: Name of scenario A.
        name_b: Name of scenario B.
        alpha: Significance level.

    Returns:
        Markdown-formatted summary string.
    """
    sig_improvements = df[(df['significant']) & (df['difference'] < 0)]
    sig_degradations = df[(df['significant']) & (df['difference'] > 0)]

    lines = [f"## Comparison: {name_a} vs {name_b}\n"]

    if len(sig_improvements) > 0:
        lines.append(f"### Significant Improvements ({name_b} is better):\n")
        for _, row in sig_improvements.iterrows():
            lines.append(
                f"- **{row['metric']}**: {abs(row['pct_difference']):.1f}% reduction "
                f"({row['effect_magnitude']} effect)\n"
            )

    if len(sig_degradations) > 0:
        lines.append(f"\n### Significant Degradations ({name_b} is worse):\n")
        for _, row in sig_degradations.iterrows():
            lines.append(
                f"- **{row['metric']}**: {row['pct_difference']:.1f}% increase "
                f"({row['effect_magnitude']} effect)\n"
            )

    no_change = df[~df['significant']]
    if len(no_change) > 0:
        lines.append(f"\n### No Significant Change:\n")
        for _, row in no_change.iterrows():
            lines.append(f"- {row['metric']}\n")

    return "".join(lines)


def compare_scenarios(
    scenario_a: Union[Scenario, FullScenario],
    scenario_b: Union[Scenario, FullScenario],
    metrics: List[str],
    n_reps: int = 50,
    scenario_a_name: str = "Scenario A",
    scenario_b_name: str = "Scenario B",
    alpha: float = 0.05
) -> ComparisonResult:
    """Compare two scenarios with statistical testing.

    Runs both scenarios multiple times and uses Mann-Whitney U test
    to determine if differences are statistically significant.

    Args:
        scenario_a: First scenario (often "current state" or "baseline").
        scenario_b: Second scenario (often "proposed change").
        metrics: List of metrics to compare.
        n_reps: Replications per scenario (recommend 50 for reliable stats).
        scenario_a_name: Display name for scenario A.
        scenario_b_name: Display name for scenario B.
        alpha: Significance level (default 0.05).

    Returns:
        ComparisonResult with detailed comparison and plain language summary.

    Example:
        >>> current = FullScenario(n_ed_bays=20)
        >>> proposed = FullScenario(n_ed_bays=25)
        >>> result = compare_scenarios(
        ...     current, proposed,
        ...     metrics=['mean_treatment_wait', 'p_delay', 'util_ed_bays'],
        ...     scenario_a_name="Current (20 beds)",
        ...     scenario_b_name="Proposed (25 beds)"
        ... )
        >>> print(result.summary)

    Plain Language:
        Compares two scenarios to see which performs better.
        Uses statistical tests to determine if differences are real
        or just due to random variation.
    """
    # Run replications for both scenarios
    results_a = _run_replications(scenario_a, n_reps)
    results_b = _run_replications(scenario_b, n_reps)

    # Compare each metric
    comparison_data = []

    for metric in metrics:
        values_a = [r.get(metric, np.nan) for r in results_a]
        values_b = [r.get(metric, np.nan) for r in results_b]

        # Remove NaNs
        values_a = [v for v in values_a if not np.isnan(v)]
        values_b = [v for v in values_b if not np.isnan(v)]

        if not values_a or not values_b:
            continue

        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1) if len(values_a) > 1 else 0.0
        std_b = np.std(values_b, ddof=1) if len(values_b) > 1 else 0.0

        # Statistical test (Mann-Whitney U - non-parametric)
        try:
            statistic, p_value = stats.mannwhitneyu(
                values_a, values_b, alternative='two-sided'
            )
        except ValueError:
            # Handle edge cases (all values identical)
            p_value = 1.0

        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2) if (std_a > 0 or std_b > 0) else 1.0
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

        # Difference
        diff = mean_b - mean_a
        pct_diff = (diff / mean_a * 100) if mean_a != 0 else 0.0

        comparison_data.append({
            'metric': metric,
            f'{scenario_a_name}_mean': mean_a,
            f'{scenario_a_name}_std': std_a,
            f'{scenario_b_name}_mean': mean_b,
            f'{scenario_b_name}_std': std_b,
            'difference': diff,
            'pct_difference': pct_diff,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'effect_magnitude': _effect_magnitude(effect_size)
        })

    df = pd.DataFrame(comparison_data)

    # Generate plain language summary
    summary = _generate_summary(df, scenario_a_name, scenario_b_name, alpha)

    return ComparisonResult(
        scenario_a_name=scenario_a_name,
        scenario_b_name=scenario_b_name,
        metrics=df,
        summary=summary
    )


def quick_compare(
    scenario_a: Union[Scenario, FullScenario],
    scenario_b: Union[Scenario, FullScenario],
    n_reps: int = 30
) -> ComparisonResult:
    """Quick comparison with default metrics.

    Convenience function for rapid comparison using common metrics.

    Args:
        scenario_a: First scenario.
        scenario_b: Second scenario.
        n_reps: Number of replications.

    Returns:
        ComparisonResult with default metrics compared.
    """
    default_metrics = [
        'arrivals',
        'p_delay',
        'mean_treatment_wait',
        'mean_system_time',
        'util_triage',
        'util_ed_bays',
    ]

    return compare_scenarios(
        scenario_a,
        scenario_b,
        metrics=default_metrics,
        n_reps=n_reps,
        scenario_a_name="Baseline",
        scenario_b_name="Modified"
    )
