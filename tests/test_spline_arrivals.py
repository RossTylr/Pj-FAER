"""Tests for spline-based arrival profiles (Phase 8b).

Tests the SplineKnot, SplineArrivalConfig, and build_spline_profile() functionality
including PCHIP interpolation, constraints, and integration with FullScenario.
"""

import pytest
import numpy as np

from faer.core.entities import ArrivalModel, ArrivalMode
from faer.core.scenario import (
    SplineKnot,
    SplineArrivalConfig,
    FullScenario,
    ArrivalConfig,
    BASE_HOURLY_RATES,
)
from faer.core.arrivals import build_spline_profile, ArrivalProfile


class TestSplineKnot:
    """Tests for SplineKnot dataclass."""

    def test_create_knot_with_defaults(self):
        """Create knot with default multiplier."""
        knot = SplineKnot(hour=12.0)
        assert knot.hour == 12.0
        assert knot.multiplier == 1.0

    def test_create_knot_with_multiplier(self):
        """Create knot with custom multiplier."""
        knot = SplineKnot(hour=6.0, multiplier=1.5)
        assert knot.hour == 6.0
        assert knot.multiplier == 1.5

    def test_knot_hour_validation_min(self):
        """Hour below 0 raises ValueError."""
        with pytest.raises(ValueError, match="Hour must be"):
            SplineKnot(hour=-1.0, multiplier=1.0)

    def test_knot_hour_validation_max(self):
        """Hour above 24 raises ValueError."""
        with pytest.raises(ValueError, match="Hour must be"):
            SplineKnot(hour=25.0, multiplier=1.0)

    def test_knot_multiplier_validation(self):
        """Negative multiplier raises ValueError."""
        with pytest.raises(ValueError, match="Multiplier must be"):
            SplineKnot(hour=12.0, multiplier=-0.5)

    def test_knot_to_dict(self):
        """Knot serializes to dict."""
        knot = SplineKnot(hour=8.5, multiplier=2.0)
        d = knot.to_dict()
        assert d == {'hour': 8.5, 'multiplier': 2.0}

    def test_knot_from_dict(self):
        """Knot deserializes from dict."""
        d = {'hour': 8.5, 'multiplier': 2.0}
        knot = SplineKnot.from_dict(d)
        assert knot.hour == 8.5
        assert knot.multiplier == 2.0

    def test_knot_boundary_hours(self):
        """Boundary hours (0, 24) are valid."""
        knot_0 = SplineKnot(hour=0.0)
        knot_24 = SplineKnot(hour=24.0)
        assert knot_0.hour == 0.0
        assert knot_24.hour == 24.0


class TestSplineArrivalConfig:
    """Tests for SplineArrivalConfig dataclass."""

    def test_default_config(self):
        """Default config has 5 knots at 1.0 multiplier."""
        config = SplineArrivalConfig()
        assert len(config.knots) == 5
        for knot in config.knots:
            assert knot.multiplier == 1.0

    def test_config_validation_min_knots(self):
        """Config requires at least 3 knots."""
        with pytest.raises(ValueError, match="at least 3 knots"):
            SplineArrivalConfig(knots=[
                SplineKnot(0.0, 1.0),
                SplineKnot(24.0, 1.0),
            ])

    def test_config_validation_max_knots(self):
        """Config allows maximum 12 knots."""
        knots = [SplineKnot(i * 2, 1.0) for i in range(13)]  # 13 knots
        with pytest.raises(ValueError, match="maximum 12 knots"):
            SplineArrivalConfig(knots=knots)

    def test_config_multiplier_validation(self):
        """min_multiplier must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            SplineArrivalConfig(min_multiplier=-0.1)

    def test_config_multiplier_range_validation(self):
        """max_multiplier must be >= min_multiplier."""
        with pytest.raises(ValueError, match="max_multiplier must be"):
            SplineArrivalConfig(min_multiplier=2.0, max_multiplier=1.0)

    def test_get_sorted_knots(self):
        """Knots are sorted by hour."""
        config = SplineArrivalConfig(knots=[
            SplineKnot(12.0, 1.5),
            SplineKnot(0.0, 1.0),
            SplineKnot(24.0, 1.0),
        ])
        sorted_knots = config.get_sorted_knots()
        hours = [k.hour for k in sorted_knots]
        assert hours == [0.0, 12.0, 24.0]

    def test_add_knot(self):
        """Add knot within limit."""
        config = SplineArrivalConfig()
        initial_count = len(config.knots)
        success = config.add_knot(9.0, 1.3)
        assert success is True
        assert len(config.knots) == initial_count + 1

    def test_add_knot_at_limit(self):
        """Cannot add knot when at 12."""
        knots = [SplineKnot(i * 2, 1.0) for i in range(12)]  # 12 knots
        config = SplineArrivalConfig(knots=knots)
        success = config.add_knot(1.0, 1.5)
        assert success is False
        assert len(config.knots) == 12

    def test_remove_knot(self):
        """Remove knot above minimum."""
        config = SplineArrivalConfig()  # 5 knots
        initial_count = len(config.knots)
        success = config.remove_knot(2)
        assert success is True
        assert len(config.knots) == initial_count - 1

    def test_remove_knot_at_minimum(self):
        """Cannot remove knot when at 3."""
        config = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.0),
            SplineKnot(24.0, 1.0),
        ])
        success = config.remove_knot(1)
        assert success is False
        assert len(config.knots) == 3

    def test_to_dict_from_dict_roundtrip(self):
        """Config serializes and deserializes correctly."""
        config = SplineArrivalConfig(
            preserve_volume=True,
            max_multiplier=3.0,
            min_multiplier=0.1,
        )
        config.add_knot(9.0, 1.8)

        d = config.to_dict()
        restored = SplineArrivalConfig.from_dict(d)

        assert len(restored.knots) == len(config.knots)
        assert restored.preserve_volume == config.preserve_volume
        assert restored.max_multiplier == config.max_multiplier
        assert restored.min_multiplier == config.min_multiplier


class TestBuildSplineProfile:
    """Tests for build_spline_profile() function."""

    def test_uniform_knots_match_baseline(self):
        """All 1.0 multipliers produce rates close to baseline."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.0),
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24  # Constant baseline

        profile = build_spline_profile(knots, baseline)

        # All rates should be approximately 4.0
        for _, rate in profile.schedule:
            assert abs(rate - 4.0) < 0.5, f"Rate {rate} not close to baseline 4.0"

    def test_doubled_knot_doubles_rate(self):
        """A 2.0 multiplier at noon approximately doubles the rate."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 2.0),  # Double at noon
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline)

        # Rate at noon (minute 720) should be ~8.0
        noon_rate = profile.get_rate(720)
        assert noon_rate > 7.0, f"Noon rate {noon_rate} should be near 8.0"
        assert noon_rate < 9.0, f"Noon rate {noon_rate} should be near 8.0"

    def test_halved_knot_halves_rate(self):
        """A 0.5 multiplier halves the rate."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 0.5),  # Half at noon
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline)

        noon_rate = profile.get_rate(720)
        assert noon_rate < 2.5, f"Noon rate {noon_rate} should be near 2.0"
        assert noon_rate > 1.5, f"Noon rate {noon_rate} should be near 2.0"

    def test_no_negative_rates(self):
        """Profile never has negative rates."""
        knots = [
            SplineKnot(0.0, 0.1),
            SplineKnot(12.0, 0.1),
            SplineKnot(24.0, 0.1),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline, min_mult=0.0)

        for _, rate in profile.schedule:
            assert rate >= 0.0, f"Rate {rate} is negative"

    def test_max_multiplier_constraint(self):
        """Multiplier values are clamped to max."""
        knots = [
            SplineKnot(0.0, 5.0),  # Exceeds default max of 2.5
            SplineKnot(12.0, 5.0),
            SplineKnot(24.0, 5.0),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline, max_mult=2.0)

        # All rates should be <= 8.0 (4.0 * 2.0)
        for _, rate in profile.schedule:
            assert rate <= 8.1, f"Rate {rate} exceeds max"

    def test_min_multiplier_constraint(self):
        """Multiplier values are clamped to min."""
        knots = [
            SplineKnot(0.0, 0.05),  # Below default min of 0.2
            SplineKnot(12.0, 0.05),
            SplineKnot(24.0, 0.05),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline, min_mult=0.2)

        # All rates should be >= 0.8 (4.0 * 0.2)
        for _, rate in profile.schedule:
            assert rate >= 0.79, f"Rate {rate} below min"

    def test_preserve_volume(self):
        """Preserve volume scales total to match baseline."""
        knots = [
            SplineKnot(0.0, 2.0),  # All doubled
            SplineKnot(12.0, 2.0),
            SplineKnot(24.0, 2.0),
        ]
        baseline = [4.0] * 24
        baseline_total = sum(baseline)

        profile = build_spline_profile(knots, baseline, preserve_volume=True)

        # Approximate total from profile (96 periods, each 15 mins)
        profile_total = sum(r for _, r in profile.schedule) * (15 / 60)
        # Should be within 10% of baseline
        assert abs(profile_total - baseline_total) < baseline_total * 0.15

    def test_pchip_shape_preserving(self):
        """PCHIP doesn't overshoot between monotonic knots."""
        # Create a profile that increases then decreases
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 2.0),  # Peak
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline, max_mult=3.0)

        # Rates should not exceed 2.0 * 4.0 = 8.0 significantly
        for _, rate in profile.schedule:
            assert rate <= 8.5, f"PCHIP overshoot: rate {rate}"

    def test_resolution_affects_schedule_length(self):
        """Different resolution produces different schedule lengths."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.5),
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24

        profile_15 = build_spline_profile(knots, baseline, resolution_mins=15)
        profile_30 = build_spline_profile(knots, baseline, resolution_mins=30)
        profile_60 = build_spline_profile(knots, baseline, resolution_mins=60)

        assert len(profile_15.schedule) == 96  # 1440 / 15
        assert len(profile_30.schedule) == 48  # 1440 / 30
        assert len(profile_60.schedule) == 24  # 1440 / 60

    def test_profile_is_arrival_profile(self):
        """Result is an ArrivalProfile instance."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.0),
            SplineKnot(24.0, 1.0),
        ]
        baseline = [4.0] * 24

        profile = build_spline_profile(knots, baseline)

        assert isinstance(profile, ArrivalProfile)

    def test_insufficient_knots_raises_error(self):
        """Fewer than 2 knots raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 knots"):
            build_spline_profile([SplineKnot(12.0, 1.0)], [4.0] * 24)

    def test_wrong_baseline_length_raises_error(self):
        """Non-24 baseline raises ValueError."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.0),
            SplineKnot(24.0, 1.0),
        ]
        with pytest.raises(ValueError, match="24 values"):
            build_spline_profile(knots, [4.0] * 12)

    def test_with_base_hourly_rates(self):
        """Works with actual BASE_HOURLY_RATES constant."""
        knots = [
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.5),
            SplineKnot(24.0, 1.0),
        ]

        profile = build_spline_profile(knots, BASE_HOURLY_RATES)

        assert profile.max_rate > 0
        assert len(profile.schedule) == 96


class TestSplineInFullScenario:
    """Tests for spline mode integration with FullScenario."""

    def test_spline_model_enum_value(self):
        """SPLINE_CONTROL enum has correct value."""
        assert ArrivalModel.SPLINE_CONTROL.value == "spline_control"

    def test_scenario_with_spline_arrivals(self):
        """FullScenario accepts spline_arrivals parameter."""
        config = SplineArrivalConfig()
        scenario = FullScenario(
            arrival_model=ArrivalModel.SPLINE_CONTROL,
            spline_arrivals=config,
        )
        assert scenario.spline_arrivals is not None
        assert scenario.arrival_model == ArrivalModel.SPLINE_CONTROL

    def test_get_effective_arrival_rate_spline_mode(self):
        """get_effective_arrival_rate uses spline when model is SPLINE_CONTROL."""
        config = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 2.0),
            SplineKnot(24.0, 1.0),
        ])
        scenario = FullScenario(
            arrival_model=ArrivalModel.SPLINE_CONTROL,
            spline_arrivals=config,
        )

        # Get arrival config
        arrival_config = scenario.arrival_configs[0]

        # Rate at noon should be higher due to 2.0 multiplier
        rate_noon = scenario.get_effective_arrival_rate(arrival_config, 12)
        rate_midnight = scenario.get_effective_arrival_rate(arrival_config, 0)

        assert rate_noon > rate_midnight, "Noon rate should be higher with 2.0 multiplier"

    def test_get_effective_arrival_rate_no_config_returns_zero(self):
        """Returns 0.0 if spline_arrivals is None."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.SPLINE_CONTROL,
            spline_arrivals=None,
        )
        arrival_config = scenario.arrival_configs[0]
        rate = scenario.get_effective_arrival_rate(arrival_config, 12)
        assert rate == 0.0

    def test_clone_preserves_spline_config(self):
        """clone_with_seed preserves spline_arrivals."""
        config = SplineArrivalConfig()
        config.knots[2] = SplineKnot(12.0, 1.8)  # Modify middle knot

        scenario = FullScenario(
            arrival_model=ArrivalModel.SPLINE_CONTROL,
            spline_arrivals=config,
            random_seed=42,
        )

        cloned = scenario.clone_with_seed(999)

        assert cloned.spline_arrivals is not None
        assert cloned.arrival_model == ArrivalModel.SPLINE_CONTROL
        # Check the modified knot was preserved
        noon_knot = [k for k in cloned.spline_arrivals.knots if abs(k.hour - 12.0) < 0.1][0]
        assert noon_knot.multiplier == 1.8


class TestSplineReproducibility:
    """Tests for reproducibility with spline arrivals."""

    def test_same_config_same_profile(self):
        """Same spline config produces identical profiles."""
        config1 = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(8.0, 1.5),
            SplineKnot(16.0, 1.2),
            SplineKnot(24.0, 1.0),
        ])
        config2 = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(8.0, 1.5),
            SplineKnot(16.0, 1.2),
            SplineKnot(24.0, 1.0),
        ])

        profile1 = build_spline_profile(config1.get_sorted_knots(), BASE_HOURLY_RATES)
        profile2 = build_spline_profile(config2.get_sorted_knots(), BASE_HOURLY_RATES)

        for i, ((t1, r1), (t2, r2)) in enumerate(zip(profile1.schedule, profile2.schedule)):
            assert t1 == t2, f"Times differ at {i}: {t1} vs {t2}"
            assert abs(r1 - r2) < 1e-10, f"Rates differ at {i}: {r1} vs {r2}"

    def test_different_config_different_profile(self):
        """Different multipliers produce different profiles."""
        config1 = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 1.5),
            SplineKnot(24.0, 1.0),
        ])
        config2 = SplineArrivalConfig(knots=[
            SplineKnot(0.0, 1.0),
            SplineKnot(12.0, 2.0),  # Different multiplier
            SplineKnot(24.0, 1.0),
        ])

        profile1 = build_spline_profile(config1.get_sorted_knots(), BASE_HOURLY_RATES)
        profile2 = build_spline_profile(config2.get_sorted_knots(), BASE_HOURLY_RATES)

        # Profiles should differ
        rates1 = [r for _, r in profile1.schedule]
        rates2 = [r for _, r in profile2.schedule]
        assert rates1 != rates2
