"""Tests for Phase 6 scenario comparison functionality."""

import pytest
import numpy as np

from faer.core.scenario import FullScenario
from faer.experiment.comparison import (
    compare_scenarios,
    quick_compare,
    ComparisonResult,
    _effect_magnitude,
)


class TestEffectMagnitude:
    """Tests for effect size interpretation."""

    def test_negligible_effect(self):
        """Small d values are negligible."""
        assert _effect_magnitude(0.1) == "negligible"
        assert _effect_magnitude(-0.15) == "negligible"

    def test_small_effect(self):
        """d between 0.2 and 0.5 is small."""
        assert _effect_magnitude(0.3) == "small"
        assert _effect_magnitude(-0.4) == "small"

    def test_medium_effect(self):
        """d between 0.5 and 0.8 is medium."""
        assert _effect_magnitude(0.6) == "medium"
        assert _effect_magnitude(-0.7) == "medium"

    def test_large_effect(self):
        """d >= 0.8 is large."""
        assert _effect_magnitude(0.9) == "large"
        assert _effect_magnitude(-1.5) == "large"


class TestCompareScenarios:
    """Tests for compare_scenarios function."""

    def test_comparison_returns_result(self):
        """Comparison returns ComparisonResult."""
        scenario_a = FullScenario(run_length=60, warm_up=0, n_ed_bays=20)
        scenario_b = FullScenario(run_length=60, warm_up=0, n_ed_bays=25)

        result = compare_scenarios(
            scenario_a,
            scenario_b,
            metrics=['arrivals', 'p_delay'],
            n_reps=3,
            scenario_a_name="Baseline",
            scenario_b_name="Increased"
        )

        assert isinstance(result, ComparisonResult)
        assert result.scenario_a_name == "Baseline"
        assert result.scenario_b_name == "Increased"

    def test_comparison_has_metrics_dataframe(self):
        """Result contains metrics DataFrame with expected columns."""
        scenario_a = FullScenario(run_length=60, warm_up=0)
        scenario_b = FullScenario(run_length=60, warm_up=0, demand_multiplier=1.5)

        result = compare_scenarios(
            scenario_a,
            scenario_b,
            metrics=['arrivals'],
            n_reps=3
        )

        df = result.metrics
        assert 'metric' in df.columns
        assert 'p_value' in df.columns
        assert 'significant' in df.columns
        assert 'effect_size' in df.columns
        assert 'effect_magnitude' in df.columns

    def test_comparison_has_summary(self):
        """Result has summary string."""
        scenario_a = FullScenario(run_length=60, warm_up=0)
        scenario_b = FullScenario(run_length=60, warm_up=0)

        result = compare_scenarios(
            scenario_a,
            scenario_b,
            metrics=['arrivals'],
            n_reps=3
        )

        assert isinstance(result.summary, str)
        assert "Comparison" in result.summary

    def test_significant_differences_filter(self):
        """significant_differences() filters by alpha."""
        scenario_a = FullScenario(run_length=60, warm_up=0, n_ed_bays=5)
        scenario_b = FullScenario(run_length=60, warm_up=0, n_ed_bays=50)

        result = compare_scenarios(
            scenario_a,
            scenario_b,
            metrics=['util_ed_bays', 'arrivals'],
            n_reps=5
        )

        # Get only significant results
        sig = result.significant_differences(alpha=0.05)
        assert isinstance(sig, type(result.metrics))
        # All rows in sig should have p_value < 0.05
        if len(sig) > 0:
            assert all(sig['p_value'] < 0.05)


class TestQuickCompare:
    """Tests for quick_compare convenience function."""

    def test_quick_compare_returns_result(self):
        """Quick compare returns ComparisonResult."""
        scenario_a = FullScenario(run_length=60, warm_up=0)
        scenario_b = FullScenario(run_length=60, warm_up=0)

        result = quick_compare(scenario_a, scenario_b, n_reps=2)

        assert isinstance(result, ComparisonResult)
        assert result.scenario_a_name == "Baseline"
        assert result.scenario_b_name == "Modified"

    def test_quick_compare_uses_default_metrics(self):
        """Quick compare uses default set of metrics."""
        scenario_a = FullScenario(run_length=60, warm_up=0)
        scenario_b = FullScenario(run_length=60, warm_up=0)

        result = quick_compare(scenario_a, scenario_b, n_reps=2)

        # Should have multiple metrics
        assert len(result.metrics) >= 1


class TestComparisonResultDataClass:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_attributes(self):
        """ComparisonResult has expected attributes."""
        import pandas as pd

        result = ComparisonResult(
            scenario_a_name="Test A",
            scenario_b_name="Test B",
            metrics=pd.DataFrame({'metric': ['test'], 'p_value': [0.1], 'significant': [False]}),
            summary="Test summary"
        )

        assert result.scenario_a_name == "Test A"
        assert result.scenario_b_name == "Test B"
        assert len(result.metrics) == 1
        assert result.summary == "Test summary"
