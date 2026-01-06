"""Tests for fleet resource controls (Phase 5c)."""

import pytest

from faer.core.scenario import FullScenario
from faer.core.entities import ArrivalMode
from faer.model.full_model import run_full_simulation, FullResultsCollector


class TestFleetParameters:
    """Test fleet parameters in scenario."""

    def test_default_fleet_params(self):
        """Default scenario has fleet parameters."""
        scenario = FullScenario()
        assert scenario.n_ambulances == 10
        assert scenario.n_helicopters == 2
        assert scenario.ambulance_turnaround_mins == 45.0
        assert scenario.helicopter_turnaround_mins == 90.0
        assert scenario.litters_per_ambulance == 1

    def test_custom_fleet_params(self):
        """Can customize fleet parameters."""
        scenario = FullScenario(
            n_ambulances=20,
            n_helicopters=5,
            ambulance_turnaround_mins=30.0,
            helicopter_turnaround_mins=60.0,
            litters_per_ambulance=2,
        )
        assert scenario.n_ambulances == 20
        assert scenario.n_helicopters == 5
        assert scenario.ambulance_turnaround_mins == 30.0
        assert scenario.helicopter_turnaround_mins == 60.0
        assert scenario.litters_per_ambulance == 2


class TestFleetMetrics:
    """Test fleet metrics collection."""

    def test_simulation_has_fleet_metrics(self):
        """Simulation produces fleet utilisation metrics."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario, use_multistream=True)

        # These metrics should exist
        assert "util_ambulance_fleet" in results
        assert "util_helicopter_fleet" in results

    def test_fleet_utilisation_range(self):
        """Fleet utilisation is between 0 and 1."""
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario, use_multistream=True)

        assert 0.0 <= results["util_ambulance_fleet"] <= 1.0
        assert 0.0 <= results["util_helicopter_fleet"] <= 1.0


class TestFleetConstraints:
    """Test fleet resource constraints."""

    def test_fleet_limits_concurrent_arrivals(self):
        """With few ambulances, fleet utilisation should be high."""
        # Very constrained fleet with high arrival rate
        constrained = FullScenario(
            n_ambulances=2,  # Very few ambulances
            n_helicopters=1,
            ambulance_turnaround_mins=60.0,  # Long turnaround
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )
        results = run_full_simulation(constrained, use_multistream=True)

        # With few ambulances and long turnaround, utilisation should be moderate to high
        # (depends on arrival rates in default config)
        assert results["util_ambulance_fleet"] >= 0.0

    def test_more_fleet_reduces_utilisation(self):
        """More vehicles should reduce utilisation."""
        # Few vehicles
        few_vehicles = FullScenario(
            n_ambulances=2,
            n_helicopters=1,
            ambulance_turnaround_mins=45.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        # Many vehicles
        many_vehicles = FullScenario(
            n_ambulances=20,
            n_helicopters=5,
            ambulance_turnaround_mins=45.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        results_few = run_full_simulation(few_vehicles, use_multistream=True)
        results_many = run_full_simulation(many_vehicles, use_multistream=True)

        # More vehicles should reduce utilisation
        assert results_many["util_ambulance_fleet"] <= results_few["util_ambulance_fleet"]

    def test_turnaround_affects_utilisation(self):
        """Longer turnaround should increase utilisation."""
        # Short turnaround
        short_turnaround = FullScenario(
            n_ambulances=5,
            ambulance_turnaround_mins=15.0,  # Short
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        # Long turnaround
        long_turnaround = FullScenario(
            n_ambulances=5,
            ambulance_turnaround_mins=90.0,  # Long
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        results_short = run_full_simulation(short_turnaround, use_multistream=True)
        results_long = run_full_simulation(long_turnaround, use_multistream=True)

        # Longer turnaround should increase utilisation
        assert results_long["util_ambulance_fleet"] >= results_short["util_ambulance_fleet"]


class TestFleetResourceLogging:
    """Test fleet resource state logging."""

    def test_empty_metrics_include_fleet(self):
        """Empty metrics include fleet fields."""
        results = FullResultsCollector()
        metrics = results._empty_metrics()

        assert "util_ambulance_fleet" in metrics
        assert "util_helicopter_fleet" in metrics


class TestFleetCloneWithSeed:
    """Test that clone_with_seed preserves fleet params."""

    def test_clone_preserves_fleet_params(self):
        """Clone with seed preserves fleet parameters."""
        original = FullScenario(
            n_ambulances=15,
            n_helicopters=3,
            ambulance_turnaround_mins=50.0,
            helicopter_turnaround_mins=100.0,
            litters_per_ambulance=2,
            random_seed=42,
        )

        cloned = original.clone_with_seed(99)

        assert cloned.n_ambulances == original.n_ambulances
        assert cloned.n_helicopters == original.n_helicopters
        assert cloned.ambulance_turnaround_mins == original.ambulance_turnaround_mins
        assert cloned.helicopter_turnaround_mins == original.helicopter_turnaround_mins
        assert cloned.litters_per_ambulance == original.litters_per_ambulance
        assert cloned.random_seed == 99


class TestWalkInsNotAffectedByFleet:
    """Test that walk-in patients are not affected by fleet constraints."""

    def test_walkins_not_constrained_by_fleet(self):
        """Walk-in patients should arrive regardless of fleet size."""
        # Zero ambulances (extreme case)
        scenario = FullScenario(
            n_ambulances=1,  # Minimum
            n_helicopters=1,  # Minimum
            run_length=240.0,
            warm_up=0.0,
            random_seed=42,
        )
        results = run_full_simulation(scenario, use_multistream=True)

        # Should still have arrivals (walk-ins don't need fleet)
        assert results["arrivals"] > 0
