"""Tests for Scenario configuration."""

import numpy as np
import pytest

from faer.core.scenario import Scenario


class TestScenarioDefaults:
    """Test default scenario creation."""

    def test_default_values(self):
        """Default scenario has expected parameter values."""
        scenario = Scenario()

        assert scenario.run_length == 480.0
        assert scenario.warm_up == 0.0
        assert scenario.arrival_rate == 4.0
        assert scenario.n_resus_bays == 2
        assert scenario.resus_mean == 45.0
        assert scenario.resus_cv == 0.5
        assert scenario.random_seed == 42

    def test_rng_streams_created(self):
        """RNG streams are created in __post_init__."""
        scenario = Scenario()

        assert scenario.rng_arrivals is not None
        assert scenario.rng_service is not None
        assert scenario.rng_routing is not None
        assert isinstance(scenario.rng_arrivals, np.random.Generator)

    def test_mean_iat_property(self):
        """Mean IAT is correctly computed from arrival rate."""
        scenario = Scenario(arrival_rate=4.0)
        assert scenario.mean_iat == 15.0  # 60 / 4 = 15 minutes

        scenario2 = Scenario(arrival_rate=6.0)
        assert scenario2.mean_iat == 10.0  # 60 / 6 = 10 minutes


class TestScenarioCustom:
    """Test custom parameter overrides."""

    def test_custom_parameters(self):
        """Custom parameters are set correctly."""
        scenario = Scenario(
            run_length=120.0,
            arrival_rate=6.0,
            n_resus_bays=4,
            resus_mean=30.0,
            random_seed=123,
        )

        assert scenario.run_length == 120.0
        assert scenario.arrival_rate == 6.0
        assert scenario.n_resus_bays == 4
        assert scenario.resus_mean == 30.0
        assert scenario.random_seed == 123


class TestScenarioReproducibility:
    """Test RNG reproducibility."""

    def test_same_seed_same_values(self):
        """Same seed produces identical random values."""
        scenario1 = Scenario(random_seed=42)
        scenario2 = Scenario(random_seed=42)

        # First arrival RNG value
        val1 = scenario1.rng_arrivals.exponential(15.0)
        val2 = scenario2.rng_arrivals.exponential(15.0)
        assert val1 == val2

        # First service RNG value
        val1 = scenario1.rng_service.exponential(45.0)
        val2 = scenario2.rng_service.exponential(45.0)
        assert val1 == val2

    def test_different_seeds_different_values(self):
        """Different seeds produce different random values."""
        scenario1 = Scenario(random_seed=42)
        scenario2 = Scenario(random_seed=99)

        val1 = scenario1.rng_arrivals.exponential(15.0)
        val2 = scenario2.rng_arrivals.exponential(15.0)
        assert val1 != val2

    def test_separate_rng_streams(self):
        """Different RNG streams are independent."""
        scenario = Scenario(random_seed=42)

        # Draw from arrivals
        arrival_val = scenario.rng_arrivals.exponential(15.0)

        # Create fresh scenario
        scenario2 = Scenario(random_seed=42)

        # Service stream should be unaffected by arrivals draw
        service_val1 = scenario.rng_service.exponential(45.0)
        service_val2 = scenario2.rng_service.exponential(45.0)
        assert service_val1 == service_val2


class TestScenarioClone:
    """Test scenario cloning."""

    def test_clone_with_seed(self):
        """Clone creates new scenario with different seed."""
        original = Scenario(
            run_length=120.0,
            arrival_rate=6.0,
            n_resus_bays=4,
        )

        cloned = original.clone_with_seed(99)

        # Parameters should match
        assert cloned.run_length == original.run_length
        assert cloned.arrival_rate == original.arrival_rate
        assert cloned.n_resus_bays == original.n_resus_bays

        # Seed should differ
        assert cloned.random_seed == 99
        assert cloned.random_seed != original.random_seed

        # RNGs should produce different values
        val1 = original.rng_arrivals.exponential(15.0)
        val2 = cloned.rng_arrivals.exponential(15.0)
        assert val1 != val2
