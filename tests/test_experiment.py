"""Tests for experimentation module."""

import numpy as np
import pytest

from faer.core.scenario import Scenario
from faer.experiment.runner import multiple_replications
from faer.experiment.analysis import (
    compute_ci,
    run_until_precision,
    estimate_required_reps,
)


class TestComputeCI:
    """Test confidence interval computation."""

    def test_ci_known_values(self):
        """CI computed correctly for known data."""
        values = [10.0, 12.0, 14.0, 16.0, 18.0]

        ci = compute_ci(values, confidence=0.95)

        assert ci["mean"] == 14.0
        assert ci["n"] == 5
        assert ci["ci_lower"] < ci["mean"]
        assert ci["ci_upper"] > ci["mean"]

    def test_ci_single_value(self):
        """CI handles single value."""
        ci = compute_ci([5.0])

        assert ci["mean"] == 5.0
        assert ci["ci_half_width"] == 0.0
        assert ci["n"] == 1

    def test_ci_empty(self):
        """CI handles empty list."""
        ci = compute_ci([])

        assert ci["mean"] == 0.0
        assert ci["n"] == 0

    def test_ci_half_width_decreases_with_n(self):
        """CI half-width decreases with more samples."""
        rng = np.random.default_rng(42)
        values_10 = list(rng.normal(50, 10, 10))
        values_100 = list(rng.normal(50, 10, 100))

        ci_10 = compute_ci(values_10)
        ci_100 = compute_ci(values_100)

        # With same variance, more samples should give tighter CI
        assert ci_100["ci_half_width"] < ci_10["ci_half_width"]


class TestMultipleReplications:
    """Test multiple replications runner."""

    def test_runs_correct_number(self):
        """Runs specified number of replications."""
        scenario = Scenario(run_length=60.0, random_seed=42)

        results = multiple_replications(scenario, n_reps=5)

        assert len(results["p_delay"]) == 5
        assert len(results["mean_queue_time"]) == 5
        assert len(results["utilisation"]) == 5

    def test_different_seeds_per_rep(self):
        """Each replication uses different seed."""
        scenario = Scenario(run_length=60.0, random_seed=42)

        results = multiple_replications(scenario, n_reps=10)

        # Values should vary (not all identical)
        p_delays = results["p_delay"]
        assert len(set(p_delays)) > 1  # Not all same

    def test_reproducibility(self):
        """Same base seed produces same replication results."""
        scenario1 = Scenario(run_length=60.0, random_seed=42)
        scenario2 = Scenario(run_length=60.0, random_seed=42)

        results1 = multiple_replications(scenario1, n_reps=5)
        results2 = multiple_replications(scenario2, n_reps=5)

        assert results1["p_delay"] == results2["p_delay"]

    def test_custom_metrics(self):
        """Can collect custom metric list."""
        scenario = Scenario(run_length=60.0)

        results = multiple_replications(
            scenario,
            n_reps=3,
            metric_names=["arrivals", "departures"],
        )

        assert "arrivals" in results
        assert "departures" in results
        assert "p_delay" not in results

    def test_progress_callback(self):
        """Progress callback is called correctly."""
        scenario = Scenario(run_length=60.0)
        progress = []

        def callback(current, total):
            progress.append((current, total))

        multiple_replications(scenario, n_reps=5, progress_callback=callback)

        assert len(progress) == 5
        assert progress[0] == (1, 5)
        assert progress[-1] == (5, 5)


class TestRunUntilPrecision:
    """Test precision-based stopping."""

    def test_converges_with_loose_target(self):
        """Converges with achievable target precision."""
        scenario = Scenario(run_length=120.0, random_seed=42)

        result = run_until_precision(
            scenario,
            target_metric="p_delay",
            target_half_width=0.2,  # Very loose target
            min_reps=10,
            max_reps=50,
        )

        assert result["converged"] is True
        assert result["n_reps"] <= 50
        assert result["ci"]["ci_half_width"] <= 0.2

    def test_hits_max_with_tight_target(self):
        """Hits max_reps with very tight target."""
        scenario = Scenario(run_length=60.0, random_seed=42)

        result = run_until_precision(
            scenario,
            target_metric="p_delay",
            target_half_width=0.001,  # Very tight target
            min_reps=5,
            max_reps=20,
        )

        assert result["converged"] is False
        assert result["n_reps"] == 20

    def test_returns_all_values(self):
        """Returns all collected metric values."""
        scenario = Scenario(run_length=60.0, random_seed=42)

        result = run_until_precision(
            scenario,
            min_reps=10,
            max_reps=15,
        )

        assert len(result["values"]) == result["n_reps"]


class TestEstimateRequiredReps:
    """Test replication estimation."""

    def test_estimate_from_pilot(self):
        """Estimates required reps from pilot data."""
        pilot = [0.5, 0.6, 0.55, 0.45, 0.5]

        n_required = estimate_required_reps(pilot, target_half_width=0.05)

        assert n_required > len(pilot)

    def test_returns_minimum(self):
        """Returns at least pilot size."""
        pilot = [0.5, 0.5, 0.5, 0.5, 0.5]  # No variance

        n_required = estimate_required_reps(pilot, target_half_width=0.05)

        assert n_required >= len(pilot)

    def test_single_value(self):
        """Handles single pilot value."""
        n_required = estimate_required_reps([0.5], target_half_width=0.05)

        assert n_required == 100  # Default estimate
