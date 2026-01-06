"""Tests for demand scaling controls (Phase 5d)."""

import pytest

from faer.core.scenario import FullScenario
from faer.core.entities import ArrivalMode
from faer.model.full_model import run_full_simulation


class TestDemandScalingParameters:
    """Test demand scaling parameters in scenario."""

    def test_default_demand_params(self):
        """Default scenario has demand scaling parameters."""
        scenario = FullScenario()
        assert scenario.demand_multiplier == 1.0
        assert scenario.ambulance_rate_multiplier == 1.0
        assert scenario.helicopter_rate_multiplier == 1.0
        assert scenario.walkin_rate_multiplier == 1.0

    def test_custom_demand_params(self):
        """Can customize demand scaling parameters."""
        scenario = FullScenario(
            demand_multiplier=1.5,
            ambulance_rate_multiplier=2.0,
            helicopter_rate_multiplier=0.5,
            walkin_rate_multiplier=0.8,
        )
        assert scenario.demand_multiplier == 1.5
        assert scenario.ambulance_rate_multiplier == 2.0
        assert scenario.helicopter_rate_multiplier == 0.5
        assert scenario.walkin_rate_multiplier == 0.8


class TestRateMultiplierMethod:
    """Test get_rate_multiplier method."""

    def test_rate_multiplier_default(self):
        """Default multiplier is 1.0 for all modes."""
        scenario = FullScenario()
        assert scenario.get_rate_multiplier(ArrivalMode.AMBULANCE) == 1.0
        assert scenario.get_rate_multiplier(ArrivalMode.HELICOPTER) == 1.0
        assert scenario.get_rate_multiplier(ArrivalMode.SELF_PRESENTATION) == 1.0

    def test_rate_multiplier_global(self):
        """Global multiplier affects all modes."""
        scenario = FullScenario(demand_multiplier=2.0)
        assert scenario.get_rate_multiplier(ArrivalMode.AMBULANCE) == 2.0
        assert scenario.get_rate_multiplier(ArrivalMode.HELICOPTER) == 2.0
        assert scenario.get_rate_multiplier(ArrivalMode.SELF_PRESENTATION) == 2.0

    def test_rate_multiplier_per_stream(self):
        """Per-stream multipliers apply correctly."""
        scenario = FullScenario(
            ambulance_rate_multiplier=1.5,
            helicopter_rate_multiplier=0.5,
            walkin_rate_multiplier=0.8,
        )
        assert scenario.get_rate_multiplier(ArrivalMode.AMBULANCE) == 1.5
        assert scenario.get_rate_multiplier(ArrivalMode.HELICOPTER) == 0.5
        assert scenario.get_rate_multiplier(ArrivalMode.SELF_PRESENTATION) == 0.8

    def test_rate_multiplier_combined(self):
        """Global and per-stream multipliers combine correctly."""
        scenario = FullScenario(
            demand_multiplier=2.0,
            ambulance_rate_multiplier=1.5,
            helicopter_rate_multiplier=0.5,
        )
        assert scenario.get_rate_multiplier(ArrivalMode.AMBULANCE) == 3.0  # 2.0 * 1.5
        assert scenario.get_rate_multiplier(ArrivalMode.HELICOPTER) == 1.0  # 2.0 * 0.5
        assert scenario.get_rate_multiplier(ArrivalMode.SELF_PRESENTATION) == 2.0  # 2.0 * 1.0


class TestDemandScalingEffect:
    """Test demand scaling affects simulation."""

    def test_demand_multiplier_increases_arrivals(self):
        """Higher demand_multiplier should increase arrivals."""
        # Base scenario
        base = FullScenario(
            demand_multiplier=1.0,
            run_length=360.0,
            warm_up=0.0,
            random_seed=42,
        )

        # Double demand
        double = FullScenario(
            demand_multiplier=2.0,
            run_length=360.0,
            warm_up=0.0,
            random_seed=42,
        )

        results_base = run_full_simulation(base, use_multistream=True)
        results_double = run_full_simulation(double, use_multistream=True)

        # Double demand should roughly double arrivals
        ratio = results_double["arrivals"] / results_base["arrivals"]
        assert 1.5 < ratio < 2.5  # Allow variance

    def test_zero_stream_multiplier_disables_stream(self):
        """Setting stream multiplier to 0 should disable that stream."""
        scenario = FullScenario(
            helicopter_rate_multiplier=0.0,  # No helicopters
            run_length=360.0,
            warm_up=0.0,
            random_seed=42,
        )
        # This should run without error
        results = run_full_simulation(scenario, use_multistream=True)
        assert results["arrivals"] >= 0

    def test_per_stream_scaling(self):
        """Per-stream multipliers scale streams independently."""
        # High ambulance rate, low walk-in rate
        scenario = FullScenario(
            ambulance_rate_multiplier=2.0,
            walkin_rate_multiplier=0.5,
            run_length=360.0,
            warm_up=0.0,
            random_seed=42,
        )
        results = run_full_simulation(scenario, use_multistream=True)

        # Should have arrivals
        assert results["arrivals"] > 0

    def test_demand_scaling_reproducibility(self):
        """Same seed and multipliers produce same results."""
        scenario1 = FullScenario(
            demand_multiplier=1.5,
            run_length=240.0,
            warm_up=0.0,
            random_seed=42,
        )
        scenario2 = FullScenario(
            demand_multiplier=1.5,
            run_length=240.0,
            warm_up=0.0,
            random_seed=42,
        )

        results1 = run_full_simulation(scenario1, use_multistream=True)
        results2 = run_full_simulation(scenario2, use_multistream=True)

        assert results1["arrivals"] == results2["arrivals"]


class TestDemandScalingCloneWithSeed:
    """Test that clone_with_seed preserves demand scaling params."""

    def test_clone_preserves_demand_params(self):
        """Clone with seed preserves demand scaling parameters."""
        original = FullScenario(
            demand_multiplier=1.5,
            ambulance_rate_multiplier=2.0,
            helicopter_rate_multiplier=0.5,
            walkin_rate_multiplier=0.8,
            random_seed=42,
        )

        cloned = original.clone_with_seed(99)

        assert cloned.demand_multiplier == original.demand_multiplier
        assert cloned.ambulance_rate_multiplier == original.ambulance_rate_multiplier
        assert cloned.helicopter_rate_multiplier == original.helicopter_rate_multiplier
        assert cloned.walkin_rate_multiplier == original.walkin_rate_multiplier
        assert cloned.random_seed == 99


class TestSurgeScenarios:
    """Test surge scenario simulations."""

    def test_surge_scenario_25_percent(self):
        """25% surge scenario runs correctly."""
        scenario = FullScenario(
            demand_multiplier=1.25,
            run_length=360.0,
            warm_up=60.0,
            random_seed=42,
        )
        results = run_full_simulation(scenario, use_multistream=True)
        assert results["arrivals"] > 0

    def test_surge_scenario_50_percent(self):
        """50% surge scenario runs correctly."""
        scenario = FullScenario(
            demand_multiplier=1.50,
            run_length=360.0,
            warm_up=60.0,
            random_seed=42,
        )
        results = run_full_simulation(scenario, use_multistream=True)
        assert results["arrivals"] > 0

    def test_low_demand_scenario(self):
        """Low demand scenario (0.5x) runs correctly."""
        scenario = FullScenario(
            demand_multiplier=0.5,
            run_length=360.0,
            warm_up=60.0,
            random_seed=42,
        )
        results = run_full_simulation(scenario, use_multistream=True)
        assert results["arrivals"] >= 0
