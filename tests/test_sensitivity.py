"""Tests for Phase 6 sensitivity analysis and breaking point functions."""

import pytest
import numpy as np

from faer.core.scenario import Scenario, FullScenario
from faer.experiment.analysis import (
    sensitivity_sweep,
    find_breaking_point,
    identify_bottlenecks,
    set_nested_param,
    SweepResult,
    BreakingPointResult,
    BottleneckResult,
)


class TestSetNestedParam:
    """Tests for set_nested_param function."""

    def test_set_simple_param(self):
        """Can set a simple top-level parameter."""
        scenario = FullScenario()
        original_bays = scenario.n_ed_bays

        modified = set_nested_param(scenario, 'n_ed_bays', 25)

        assert modified.n_ed_bays == 25
        assert scenario.n_ed_bays == original_bays  # Original unchanged

    def test_set_demand_multiplier(self):
        """Can set demand_multiplier parameter."""
        scenario = FullScenario()
        modified = set_nested_param(scenario, 'demand_multiplier', 2.5)

        assert modified.demand_multiplier == 2.5

    def test_invalid_param_raises_error(self):
        """Invalid parameter path raises ValueError."""
        scenario = FullScenario()

        with pytest.raises(ValueError):
            set_nested_param(scenario, 'nonexistent_param', 10)


class TestSensitivitySweep:
    """Tests for sensitivity_sweep function."""

    def test_sweep_returns_result(self):
        """Sweep completes and returns SweepResult."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = sensitivity_sweep(
            scenario,
            'n_ed_bays',
            values=[10, 20],
            metric='arrivals',
            n_reps=2,
            parallel=False
        )

        assert isinstance(result, SweepResult)
        assert len(result.results) == 2
        assert 'mean' in result.results.columns
        assert 'ci_lower' in result.results.columns
        assert 'ci_upper' in result.results.columns

    def test_sweep_dataframe_conversion(self):
        """to_dataframe() returns DataFrame."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = sensitivity_sweep(
            scenario,
            'n_ed_bays',
            values=[15, 20],
            metric='arrivals',
            n_reps=2
        )

        df = result.to_dataframe()
        assert len(df) == 2
        assert df['value'].tolist() == [15, 20]

    def test_sweep_with_demand_multiplier(self):
        """Can sweep demand_multiplier parameter."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = sensitivity_sweep(
            scenario,
            'demand_multiplier',
            values=[0.5, 1.0, 1.5],
            metric='arrivals',
            n_reps=3
        )

        assert len(result.results) == 3
        # Higher demand should generally mean more arrivals
        df = result.to_dataframe()
        # Allow for some randomness but trend should be positive
        assert df.iloc[0]['mean'] <= df.iloc[2]['mean'] * 1.5  # Roughly proportional


class TestFindBreakingPoint:
    """Tests for find_breaking_point function."""

    def test_breaking_point_returns_result(self):
        """Breaking point finder returns BreakingPointResult."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = find_breaking_point(
            scenario,
            'demand_multiplier',
            'p_delay',
            threshold=0.5,
            search_range=(0.5, 3.0),
            direction='above',
            n_reps=2,
            max_iterations=3
        )

        assert isinstance(result, BreakingPointResult)
        assert result.parameter == 'demand_multiplier'
        assert result.metric == 'p_delay'
        assert result.threshold == 0.5
        assert result.direction == 'above'
        assert result.breaking_point >= 0.5
        assert result.breaking_point <= 3.0

    def test_breaking_point_has_search_history(self):
        """Breaking point result includes search history."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = find_breaking_point(
            scenario,
            'demand_multiplier',
            'p_delay',
            threshold=0.5,
            search_range=(1.0, 2.0),
            n_reps=2,
            max_iterations=3
        )

        assert len(result.search_history) > 0
        assert 'iteration' in result.search_history[0]
        assert 'value' in result.search_history[0]

    def test_breaking_point_summary(self):
        """Summary method returns formatted string."""
        scenario = FullScenario(run_length=60, warm_up=0)

        result = find_breaking_point(
            scenario,
            'demand_multiplier',
            'p_delay',
            threshold=0.5,
            search_range=(1.0, 2.0),
            n_reps=2,
            max_iterations=3
        )

        summary = result.summary()
        assert 'p_delay' in summary
        assert 'demand_multiplier' in summary
        assert '0.5' in summary  # threshold


class TestIdentifyBottlenecks:
    """Tests for identify_bottlenecks function."""

    def test_bottleneck_no_critical(self):
        """No bottleneck when all utilisation below 85%."""
        results = {
            'util_triage': 0.5,
            'util_ed_bays': 0.6,
            'util_handover': 0.4,
            'util_ambulance_fleet': 0.3,
        }
        scenario = FullScenario()

        bottleneck = identify_bottlenecks(results, scenario)

        assert isinstance(bottleneck, BottleneckResult)
        assert bottleneck.primary_bottleneck == "None"
        assert "adequate capacity" in bottleneck.recommendation.lower()

    def test_bottleneck_ed_bays_high(self):
        """ED bays identified as bottleneck when utilisation high."""
        results = {
            'util_triage': 0.5,
            'util_ed_bays': 0.92,  # High
            'util_handover': 0.4,
            'util_ambulance_fleet': 0.3,
            'mean_treatment_wait': 45,  # Adds to score
        }
        scenario = FullScenario()

        bottleneck = identify_bottlenecks(results, scenario)

        assert bottleneck.primary_bottleneck == "ed_bays"
        assert "ED bay" in bottleneck.recommendation

    def test_bottleneck_triage_high(self):
        """Triage identified as bottleneck when utilisation high."""
        results = {
            'util_triage': 0.95,  # Highest
            'util_ed_bays': 0.5,
            'util_handover': 0.4,
            'util_ambulance_fleet': 0.3,
        }
        scenario = FullScenario()

        bottleneck = identify_bottlenecks(results, scenario)

        assert bottleneck.primary_bottleneck == "triage"
        assert "triage" in bottleneck.recommendation.lower()

    def test_bottleneck_summary(self):
        """Summary method returns formatted output."""
        results = {
            'util_triage': 0.5,
            'util_ed_bays': 0.9,
            'util_handover': 0.4,
            'util_ambulance_fleet': 0.3,
        }
        scenario = FullScenario()

        bottleneck = identify_bottlenecks(results, scenario)
        summary = bottleneck.summary()

        assert "Primary Bottleneck" in summary
        assert "Recommendation" in summary
        assert "Utilisation Ranking" in summary

    def test_bottleneck_utilisation_ranking(self):
        """Utilisation ranking is sorted descending."""
        results = {
            'util_triage': 0.5,
            'util_ed_bays': 0.9,
            'util_handover': 0.7,
            'util_ambulance_fleet': 0.3,
        }
        scenario = FullScenario()

        bottleneck = identify_bottlenecks(results, scenario)

        # Should be sorted high to low
        utils = [u for _, u in bottleneck.utilisation_ranking]
        assert utils == sorted(utils, reverse=True)


class TestSweepResultDataClass:
    """Tests for SweepResult dataclass."""

    def test_sweep_result_attributes(self):
        """SweepResult has expected attributes."""
        import pandas as pd

        result = SweepResult(
            parameter='test_param',
            values=[1, 2, 3],
            metric='test_metric',
            results=pd.DataFrame({'value': [1, 2, 3], 'mean': [10, 20, 30]})
        )

        assert result.parameter == 'test_param'
        assert result.values == [1, 2, 3]
        assert result.metric == 'test_metric'
        assert len(result.results) == 3


class TestBreakingPointResultDataClass:
    """Tests for BreakingPointResult dataclass."""

    def test_breaking_point_result_attributes(self):
        """BreakingPointResult has expected attributes."""
        result = BreakingPointResult(
            parameter='demand',
            metric='wait_time',
            threshold=30.0,
            direction='above',
            breaking_point=1.5,
            metric_at_break=32.5,
            confidence_interval=(28.0, 37.0),
            search_history=[]
        )

        assert result.parameter == 'demand'
        assert result.threshold == 30.0
        assert result.breaking_point == 1.5
        assert result.metric_at_break == 32.5
