"""Tests for simulation model processes."""

import pytest

from faer.core.scenario import Scenario
from faer.model.processes import run_simulation, sample_lognormal
import numpy as np


class TestLognormalSampling:
    """Test lognormal distribution sampling."""

    def test_sample_lognormal_mean(self):
        """Lognormal samples have approximately correct mean."""
        rng = np.random.default_rng(42)
        samples = [sample_lognormal(rng, mean=45.0, cv=0.5) for _ in range(10000)]

        sample_mean = np.mean(samples)
        assert 43.0 < sample_mean < 47.0  # Within ~5% of target

    def test_sample_lognormal_zero_cv(self):
        """Zero CV returns constant mean."""
        rng = np.random.default_rng(42)
        sample = sample_lognormal(rng, mean=45.0, cv=0.0)
        assert sample == 45.0


class TestBasicFlow:
    """Test basic simulation flow."""

    def test_arrivals_occur(self):
        """Patients arrive during simulation."""
        scenario = Scenario(run_length=60.0, arrival_rate=4.0)
        results = run_simulation(scenario)

        assert results["arrivals"] > 0

    def test_departures_occur(self):
        """Patients depart during simulation."""
        scenario = Scenario(run_length=120.0, arrival_rate=2.0, n_resus_bays=5)
        results = run_simulation(scenario)

        assert results["departures"] > 0

    def test_queue_times_recorded(self):
        """Queue times are recorded for each patient."""
        scenario = Scenario(run_length=60.0, arrival_rate=4.0)
        results = run_simulation(scenario)

        assert len(results["queue_times"]) > 0
        assert all(qt >= 0 for qt in results["queue_times"])

    def test_system_times_recorded(self):
        """System times are recorded for departing patients."""
        scenario = Scenario(run_length=120.0, arrival_rate=2.0, n_resus_bays=5)
        results = run_simulation(scenario)

        assert len(results["system_times"]) > 0
        assert all(st > 0 for st in results["system_times"])


class TestReproducibility:
    """Test simulation reproducibility."""

    def test_same_seed_same_results(self):
        """Same seed produces identical results."""
        scenario1 = Scenario(random_seed=42, run_length=120.0)
        scenario2 = Scenario(random_seed=42, run_length=120.0)

        results1 = run_simulation(scenario1)
        results2 = run_simulation(scenario2)

        assert results1["arrivals"] == results2["arrivals"]
        assert results1["departures"] == results2["departures"]
        assert results1["queue_times"] == results2["queue_times"]
        assert results1["system_times"] == results2["system_times"]

    def test_different_seeds_different_results(self):
        """Different seeds produce different results."""
        scenario1 = Scenario(random_seed=42, run_length=120.0)
        scenario2 = Scenario(random_seed=99, run_length=120.0)

        results1 = run_simulation(scenario1)
        results2 = run_simulation(scenario2)

        # Results should differ (extremely unlikely to be identical)
        assert results1["queue_times"] != results2["queue_times"]


class TestCapacityEffects:
    """Test effect of capacity on queuing."""

    def test_high_capacity_low_queuing(self):
        """High capacity relative to demand means little queuing."""
        scenario = Scenario(
            arrival_rate=2.0,  # Low arrival rate
            n_resus_bays=10,  # High capacity
            resus_mean=30.0,  # Short service
            run_length=240.0,
            random_seed=42,
        )
        results = run_simulation(scenario)

        # With excess capacity, most queue times should be ~0
        mean_queue = results["mean_queue_time"]
        assert mean_queue < 5.0  # Should be very low

    def test_low_capacity_high_queuing(self):
        """Low capacity relative to demand causes queuing."""
        scenario = Scenario(
            arrival_rate=6.0,  # High arrival rate
            n_resus_bays=1,  # Very low capacity
            resus_mean=45.0,  # Moderate service
            run_length=240.0,
            random_seed=42,
        )
        results = run_simulation(scenario)

        # With constrained capacity, expect delays
        p_delay = results["p_delay"]
        assert p_delay > 0.3  # Significant proportion delayed


class TestMetrics:
    """Test computed metrics."""

    def test_p_delay_range(self):
        """P(delay) is between 0 and 1."""
        scenario = Scenario(run_length=120.0)
        results = run_simulation(scenario)

        assert 0.0 <= results["p_delay"] <= 1.0

    def test_utilisation_range(self):
        """Utilisation is between 0 and 1."""
        scenario = Scenario(run_length=120.0)
        results = run_simulation(scenario)

        assert 0.0 <= results["utilisation"] <= 1.0

    def test_mean_queue_time_non_negative(self):
        """Mean queue time is non-negative."""
        scenario = Scenario(run_length=120.0)
        results = run_simulation(scenario)

        assert results["mean_queue_time"] >= 0.0

    def test_mean_system_time_exceeds_queue_time(self):
        """Mean system time should exceed mean queue time."""
        scenario = Scenario(run_length=240.0, arrival_rate=3.0)
        results = run_simulation(scenario)

        # System time = queue time + service time
        assert results["mean_system_time"] >= results["mean_queue_time"]


class TestArrivalRate:
    """Test arrival rate accuracy."""

    def test_arrival_count_approximates_rate(self):
        """Arrival count should approximate expected from rate."""
        scenario = Scenario(
            arrival_rate=6.0,  # 6 per hour
            run_length=480.0,  # 8 hours
            random_seed=42,
        )
        results = run_simulation(scenario)

        # Expected arrivals: 6 * 8 = 48
        # Allow 20% variance for randomness
        expected = 6.0 * (480.0 / 60.0)
        assert expected * 0.8 < results["arrivals"] < expected * 1.2
