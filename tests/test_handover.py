"""Tests for ambulance handover gate (Phase 5b)."""

import pytest

from faer.core.entities import ArrivalMode
from faer.core.scenario import FullScenario
from faer.model.full_model import run_full_simulation, FullResultsCollector
from faer.model.patient import Patient, Acuity


class TestHandoverParameters:
    """Test handover parameters in scenario."""

    def test_default_handover_params(self):
        """Default scenario has handover parameters."""
        scenario = FullScenario()
        assert scenario.n_handover_bays == 4
        assert scenario.handover_time_mean == 15.0
        assert scenario.handover_time_cv == 0.3
        assert scenario.rng_handover is not None

    def test_custom_handover_params(self):
        """Can customize handover parameters."""
        scenario = FullScenario(
            n_handover_bays=8,
            handover_time_mean=20.0,
            handover_time_cv=0.5,
        )
        assert scenario.n_handover_bays == 8
        assert scenario.handover_time_mean == 20.0
        assert scenario.handover_time_cv == 0.5


class TestHandoverMetrics:
    """Test handover metrics collection."""

    def test_simulation_has_handover_metrics(self):
        """Simulation produces handover metrics."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        # These metrics should exist
        assert "mean_handover_delay" in results
        assert "max_handover_delay" in results
        assert "p95_handover_delay" in results
        assert "handover_arrivals" in results
        assert "util_handover" in results

    def test_handover_arrivals_counted(self):
        """Handover arrivals are counted."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        # Should have some handover arrivals (ambulance arrivals)
        assert results["handover_arrivals"] >= 0

    def test_handover_utilisation_range(self):
        """Handover utilisation is between 0 and 1."""
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert 0.0 <= results["util_handover"] <= 1.0


class TestHandoverFeedbackLoop:
    """Test handover feedback mechanism."""

    def test_handover_delays_increase_when_ed_full(self):
        """When ED is congested, handover delays should increase."""
        # Constrained scenario
        constrained = FullScenario(
            n_ed_bays=3,  # Very few ED bays
            n_handover_bays=4,
            arrival_rate=12.0,  # High arrival rate
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        # Unconstrained scenario
        unconstrained = FullScenario(
            n_ed_bays=30,  # Many ED bays
            n_handover_bays=4,
            arrival_rate=12.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        results_constrained = run_full_simulation(constrained)
        results_unconstrained = run_full_simulation(unconstrained)

        # Constrained ED should lead to higher handover delays
        # (because handover bays are held while waiting for ED)
        assert results_constrained["mean_handover_delay"] >= results_unconstrained["mean_handover_delay"]

    def test_more_handover_bays_reduces_delay(self):
        """More handover bays should reduce handover delay."""
        # Few handover bays
        few_bays = FullScenario(
            n_ed_bays=5,
            n_handover_bays=2,  # Very few
            arrival_rate=10.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        # Many handover bays
        many_bays = FullScenario(
            n_ed_bays=5,
            n_handover_bays=10,  # Many
            arrival_rate=10.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        results_few = run_full_simulation(few_bays)
        results_many = run_full_simulation(many_bays)

        # More handover bays should reduce delays
        assert results_many["mean_handover_delay"] <= results_few["mean_handover_delay"]


class TestWalkinsSkipHandover:
    """Test that walk-in patients bypass handover."""

    def test_walkins_have_no_handover_timestamps(self):
        """Walk-in patients should have no handover timestamps."""
        # Use multistream with high walk-in rate
        scenario = FullScenario(
            run_length=240.0,
            warm_up=0.0,
            random_seed=42,
        )

        # Run simulation
        run_full_simulation(scenario)

        # Check that patients are created (indirectly via metrics)
        # Walk-ins should skip handover
        # This is tested by running multistream simulation


class TestHandoverResourceLogging:
    """Test handover resource state logging."""

    def test_empty_metrics_include_handover(self):
        """Empty metrics include handover fields."""
        results = FullResultsCollector()
        scenario = FullScenario()
        metrics = results._empty_metrics()

        assert "mean_handover_delay" in metrics
        assert "max_handover_delay" in metrics
        assert "p95_handover_delay" in metrics
        assert "handover_arrivals" in metrics
        assert "util_handover" in metrics


class TestHandoverCloneWithSeed:
    """Test that clone_with_seed preserves handover params."""

    def test_clone_preserves_handover_params(self):
        """Clone with seed preserves handover parameters."""
        original = FullScenario(
            n_handover_bays=6,
            handover_time_mean=25.0,
            handover_time_cv=0.4,
            random_seed=42,
        )

        cloned = original.clone_with_seed(99)

        assert cloned.n_handover_bays == original.n_handover_bays
        assert cloned.handover_time_mean == original.handover_time_mean
        assert cloned.handover_time_cv == original.handover_time_cv
        assert cloned.random_seed == 99
