"""Tests for Phase 6 arrival model configurations."""

import pytest
import numpy as np

from faer.core.entities import ArrivalMode, ArrivalModel, DayType, Priority
from faer.core.scenario import (
    FullScenario,
    ArrivalConfig,
    DetailedArrivalConfig,
    DAY_TYPE_MULTIPLIERS,
    DEMAND_LEVEL_MULTIPLIERS,
)


class TestArrivalModelEnum:
    """Tests for ArrivalModel enum."""

    def test_arrival_model_values(self):
        """ArrivalModel has expected values."""
        assert ArrivalModel.SIMPLE.value == "simple"
        assert ArrivalModel.PROFILE_24H.value == "profile_24h"
        assert ArrivalModel.DETAILED.value == "detailed"


class TestDayTypeEnum:
    """Tests for DayType enum."""

    def test_day_type_values(self):
        """DayType has expected values."""
        assert DayType.WEEKDAY.value == "weekday"
        assert DayType.MONDAY.value == "monday"
        assert DayType.FRIDAY_EVE.value == "friday_eve"
        assert DayType.SATURDAY_NIGHT.value == "sat_night"
        assert DayType.SUNDAY.value == "sunday"
        assert DayType.BANK_HOLIDAY.value == "bank_holiday"

    def test_all_day_types_have_multipliers(self):
        """All DayType values have multiplier configurations."""
        for day_type in DayType:
            assert day_type in DAY_TYPE_MULTIPLIERS
            assert 'overall' in DAY_TYPE_MULTIPLIERS[day_type]
            assert 'hourly_adjustments' in DAY_TYPE_MULTIPLIERS[day_type]


class TestDemandLevelMultipliers:
    """Tests for demand level presets."""

    def test_demand_levels_exist(self):
        """Expected demand levels are defined."""
        assert 'Low' in DEMAND_LEVEL_MULTIPLIERS
        assert 'Normal' in DEMAND_LEVEL_MULTIPLIERS
        assert 'Busy' in DEMAND_LEVEL_MULTIPLIERS
        assert 'Surge' in DEMAND_LEVEL_MULTIPLIERS
        assert 'Major Incident' in DEMAND_LEVEL_MULTIPLIERS

    def test_demand_level_ordering(self):
        """Demand levels increase monotonically."""
        assert DEMAND_LEVEL_MULTIPLIERS['Low'] < DEMAND_LEVEL_MULTIPLIERS['Normal']
        assert DEMAND_LEVEL_MULTIPLIERS['Normal'] < DEMAND_LEVEL_MULTIPLIERS['Busy']
        assert DEMAND_LEVEL_MULTIPLIERS['Busy'] < DEMAND_LEVEL_MULTIPLIERS['Surge']
        assert DEMAND_LEVEL_MULTIPLIERS['Surge'] < DEMAND_LEVEL_MULTIPLIERS['Major Incident']


class TestDetailedArrivalConfig:
    """Tests for DetailedArrivalConfig class."""

    def test_default_initialization(self):
        """Default DetailedArrivalConfig has zeros for all hours/modes."""
        config = DetailedArrivalConfig()
        for hour in range(24):
            for mode in ArrivalMode:
                assert config.get_rate(hour, mode) == 0.0

    def test_set_and_get_rate(self):
        """Can set and retrieve rates."""
        config = DetailedArrivalConfig()
        config.set_rate(10, ArrivalMode.AMBULANCE, 5)
        assert config.get_rate(10, ArrivalMode.AMBULANCE) == 5.0
        assert config.get_rate(10, ArrivalMode.HELICOPTER) == 0.0

    def test_hourly_counts_structure(self):
        """Hourly counts are properly structured."""
        config = DetailedArrivalConfig()
        assert len(config.hourly_counts) == 24
        for hour in range(24):
            assert ArrivalMode.AMBULANCE in config.hourly_counts[hour]
            assert ArrivalMode.HELICOPTER in config.hourly_counts[hour]
            assert ArrivalMode.SELF_PRESENTATION in config.hourly_counts[hour]


class TestSimpleArrivalModel:
    """Tests for ArrivalModel.SIMPLE behavior."""

    def test_simple_model_uses_average(self):
        """Simple model uses average rate across all hours."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.SIMPLE,
            demand_multiplier=1.0,
        )

        config = scenario.arrival_configs[0]  # Ambulance

        # Rate should be constant across hours (average)
        expected_avg = sum(config.hourly_rates) / 24

        rate_at_8 = scenario.get_effective_arrival_rate(config, 8)
        rate_at_16 = scenario.get_effective_arrival_rate(config, 16)
        rate_at_2 = scenario.get_effective_arrival_rate(config, 2)

        assert abs(rate_at_8 - expected_avg) < 0.01
        assert abs(rate_at_16 - expected_avg) < 0.01
        assert abs(rate_at_2 - expected_avg) < 0.01

    def test_simple_model_applies_demand_multiplier(self):
        """Simple model applies demand multiplier to average rate."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.SIMPLE,
            demand_multiplier=2.0,
        )

        config = scenario.arrival_configs[0]
        expected_avg = sum(config.hourly_rates) / 24

        rate = scenario.get_effective_arrival_rate(config, 8)
        assert abs(rate - expected_avg * 2.0) < 0.01


class TestProfile24hModel:
    """Tests for ArrivalModel.PROFILE_24H behavior."""

    def test_profile_uses_hourly_rates(self):
        """Profile model uses different rates for different hours."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.WEEKDAY,
        )

        config = scenario.arrival_configs[0]  # Ambulance

        rate_peak = scenario.get_effective_arrival_rate(config, 9)  # Morning peak
        rate_night = scenario.get_effective_arrival_rate(config, 3)  # Night

        # Peak should be higher than night (from default ambulance profile)
        assert rate_peak > rate_night

    def test_monday_morning_surge(self):
        """Monday has higher rates 07:00-11:00."""
        scenario_weekday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.WEEKDAY,
        )

        scenario_monday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.MONDAY,
        )

        config = scenario_weekday.arrival_configs[0]
        config_monday = scenario_monday.arrival_configs[0]

        rate_8am_weekday = scenario_weekday.get_effective_arrival_rate(config, 8)
        rate_8am_monday = scenario_monday.get_effective_arrival_rate(config_monday, 8)

        # Monday 8am should be higher (1.3x from multipliers)
        assert rate_8am_monday > rate_8am_weekday
        assert abs(rate_8am_monday / rate_8am_weekday - 1.3) < 0.01

    def test_saturday_night_surge(self):
        """Saturday night has higher rates 20:00-02:00."""
        scenario_weekday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.WEEKDAY,
        )

        scenario_saturday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.SATURDAY_NIGHT,
        )

        config = scenario_weekday.arrival_configs[0]
        config_saturday = scenario_saturday.arrival_configs[0]

        rate_22_weekday = scenario_weekday.get_effective_arrival_rate(config, 22)
        rate_22_saturday = scenario_saturday.get_effective_arrival_rate(config_saturday, 22)

        # Saturday 10pm should be higher (1.5x from multipliers)
        assert rate_22_saturday > rate_22_weekday
        assert abs(rate_22_saturday / rate_22_weekday - 1.5) < 0.01

    def test_sunday_overall_reduction(self):
        """Sunday has 15% overall reduction."""
        scenario_weekday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.WEEKDAY,
        )

        scenario_sunday = FullScenario(
            arrival_model=ArrivalModel.PROFILE_24H,
            day_type=DayType.SUNDAY,
        )

        config = scenario_weekday.arrival_configs[0]
        config_sunday = scenario_sunday.arrival_configs[0]

        # Use an hour without hourly adjustment
        rate_6am_weekday = scenario_weekday.get_effective_arrival_rate(config, 6)
        rate_6am_sunday = scenario_sunday.get_effective_arrival_rate(config_sunday, 6)

        # Sunday should be 85% of weekday
        assert abs(rate_6am_sunday / rate_6am_weekday - 0.85) < 0.01


class TestDetailedArrivalModel:
    """Tests for ArrivalModel.DETAILED behavior."""

    def test_detailed_model_uses_exact_counts(self):
        """Detailed model returns exact specified counts."""
        detailed = DetailedArrivalConfig()
        detailed.set_rate(10, ArrivalMode.AMBULANCE, 5)
        detailed.set_rate(10, ArrivalMode.HELICOPTER, 1)
        detailed.set_rate(10, ArrivalMode.SELF_PRESENTATION, 8)

        scenario = FullScenario(
            arrival_model=ArrivalModel.DETAILED,
            detailed_arrivals=detailed,
        )

        # Create config matching ambulance mode
        amb_config = ArrivalConfig(
            mode=ArrivalMode.AMBULANCE,
            hourly_rates=[0] * 24,  # Not used in detailed mode
            triage_mix={Priority.P1_IMMEDIATE: 1.0, Priority.P2_VERY_URGENT: 0.0,
                       Priority.P3_URGENT: 0.0, Priority.P4_STANDARD: 0.0}
        )

        rate = scenario.get_effective_arrival_rate(amb_config, 10)
        assert rate == 5.0

    def test_detailed_model_returns_zero_without_config(self):
        """Detailed model returns 0 if no detailed config provided."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.DETAILED,
            detailed_arrivals=None,
        )

        config = scenario.arrival_configs[0]
        rate = scenario.get_effective_arrival_rate(config, 10)
        assert rate == 0.0


class TestScenarioCloning:
    """Tests for scenario cloning with new Phase 6 fields."""

    def test_clone_preserves_arrival_model(self):
        """Clone preserves arrival_model setting."""
        scenario = FullScenario(
            arrival_model=ArrivalModel.DETAILED,
            day_type=DayType.MONDAY,
        )

        cloned = scenario.clone_with_seed(999)

        assert cloned.arrival_model == ArrivalModel.DETAILED
        assert cloned.day_type == DayType.MONDAY
        assert cloned.random_seed == 999

    def test_clone_preserves_detailed_arrivals(self):
        """Clone preserves detailed arrivals config."""
        detailed = DetailedArrivalConfig()
        detailed.set_rate(8, ArrivalMode.AMBULANCE, 10)

        scenario = FullScenario(
            arrival_model=ArrivalModel.DETAILED,
            detailed_arrivals=detailed,
        )

        cloned = scenario.clone_with_seed(999)

        assert cloned.detailed_arrivals is not None
        assert cloned.detailed_arrivals.get_rate(8, ArrivalMode.AMBULANCE) == 10.0
