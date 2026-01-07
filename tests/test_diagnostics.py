"""Tests for diagnostic resources configuration (Phase 7a).

Tests the diagnostic service nodes (CT, X-ray, Bloods) configuration
and resource creation without the diagnostics loop implementation.
"""

import pytest
import simpy

from faer.core.entities import DiagnosticType, Priority
from faer.core.scenario import FullScenario, DiagnosticConfig
from faer.model.full_model import AEResources, FullResultsCollector, run_full_simulation


class TestDiagnosticConfig:
    """Tests for DiagnosticConfig dataclass."""

    def test_default_diagnostic_configs_created(self):
        """Default scenario has diagnostic configs for all types."""
        scenario = FullScenario()

        assert DiagnosticType.CT_SCAN in scenario.diagnostic_configs
        assert DiagnosticType.XRAY in scenario.diagnostic_configs
        assert DiagnosticType.BLOODS in scenario.diagnostic_configs

    def test_ct_scan_default_values(self):
        """CT scan has correct default configuration."""
        scenario = FullScenario()
        ct_config = scenario.diagnostic_configs[DiagnosticType.CT_SCAN]

        assert ct_config.capacity == 2
        assert ct_config.process_time_mean == 20.0
        assert ct_config.turnaround_time_mean == 30.0
        assert ct_config.enabled is True

    def test_xray_default_values(self):
        """X-ray has correct default configuration."""
        scenario = FullScenario()
        xray_config = scenario.diagnostic_configs[DiagnosticType.XRAY]

        assert xray_config.capacity == 3
        assert xray_config.process_time_mean == 10.0
        assert xray_config.turnaround_time_mean == 15.0

    def test_bloods_default_values(self):
        """Bloods has correct default configuration."""
        scenario = FullScenario()
        bloods_config = scenario.diagnostic_configs[DiagnosticType.BLOODS]

        assert bloods_config.capacity == 5
        assert bloods_config.process_time_mean == 5.0
        assert bloods_config.turnaround_time_mean == 45.0

    def test_p1_more_likely_to_need_ct(self):
        """P1 patients more likely to need CT than P4."""
        scenario = FullScenario()
        ct_config = scenario.diagnostic_configs[DiagnosticType.CT_SCAN]

        assert ct_config.probability_by_priority[Priority.P1_IMMEDIATE] > \
               ct_config.probability_by_priority[Priority.P4_STANDARD]

    def test_p1_more_likely_to_need_bloods(self):
        """P1 patients more likely to need bloods than P4."""
        scenario = FullScenario()
        bloods_config = scenario.diagnostic_configs[DiagnosticType.BLOODS]

        assert bloods_config.probability_by_priority[Priority.P1_IMMEDIATE] > \
               bloods_config.probability_by_priority[Priority.P4_STANDARD]

    def test_custom_diagnostic_config(self):
        """Can create custom diagnostic configuration."""
        custom_config = DiagnosticConfig(
            diagnostic_type=DiagnosticType.CT_SCAN,
            capacity=4,
            process_time_mean=15.0,
            turnaround_time_mean=20.0,
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.9,
                Priority.P2_VERY_URGENT: 0.5,
                Priority.P3_URGENT: 0.2,
                Priority.P4_STANDARD: 0.1,
            }
        )

        assert custom_config.capacity == 4
        assert custom_config.process_time_mean == 15.0


class TestDiagnosticResources:
    """Tests for diagnostic resource creation."""

    def test_diagnostic_resources_created(self):
        """Model creates diagnostic resources."""
        scenario = FullScenario()
        env = simpy.Environment()

        # Create diagnostic resources manually (as done in run_full_simulation)
        diagnostics = {}
        for diag_type, config in scenario.diagnostic_configs.items():
            if config.enabled:
                diagnostics[diag_type] = simpy.PriorityResource(
                    env, capacity=config.capacity
                )

        assert DiagnosticType.CT_SCAN in diagnostics
        assert DiagnosticType.XRAY in diagnostics
        assert DiagnosticType.BLOODS in diagnostics

    def test_diagnostic_resource_capacity(self):
        """Diagnostic resources have correct capacity."""
        scenario = FullScenario()
        env = simpy.Environment()

        diagnostics = {}
        for diag_type, config in scenario.diagnostic_configs.items():
            if config.enabled:
                diagnostics[diag_type] = simpy.PriorityResource(
                    env, capacity=config.capacity
                )

        assert diagnostics[DiagnosticType.CT_SCAN].capacity == 2
        assert diagnostics[DiagnosticType.XRAY].capacity == 3
        assert diagnostics[DiagnosticType.BLOODS].capacity == 5

    def test_disabled_diagnostic_not_created(self):
        """Disabled diagnostics are not created as resources."""
        scenario = FullScenario()
        # Disable CT scanner
        scenario.diagnostic_configs[DiagnosticType.CT_SCAN].enabled = False
        env = simpy.Environment()

        diagnostics = {}
        for diag_type, config in scenario.diagnostic_configs.items():
            if config.enabled:
                diagnostics[diag_type] = simpy.PriorityResource(
                    env, capacity=config.capacity
                )

        assert DiagnosticType.CT_SCAN not in diagnostics
        assert DiagnosticType.XRAY in diagnostics


class TestDiagnosticResultsCollector:
    """Tests for diagnostic metrics in FullResultsCollector."""

    def test_diagnostic_tracking_initialized(self):
        """Results collector initializes diagnostic tracking."""
        results = FullResultsCollector()

        for diag_type in DiagnosticType:
            assert diag_type in results.diagnostic_waits
            assert diag_type in results.diagnostic_turnarounds
            assert results.diagnostic_waits[diag_type] == []
            assert results.diagnostic_turnarounds[diag_type] == []

    def test_record_diagnostic_wait(self):
        """Can record diagnostic wait times."""
        results = FullResultsCollector()

        results.record_diagnostic_wait(DiagnosticType.CT_SCAN, 10.5)
        results.record_diagnostic_wait(DiagnosticType.CT_SCAN, 15.0)

        assert len(results.diagnostic_waits[DiagnosticType.CT_SCAN]) == 2
        assert results.diagnostic_waits[DiagnosticType.CT_SCAN][0] == 10.5
        assert results.diagnostic_waits[DiagnosticType.CT_SCAN][1] == 15.0

    def test_record_diagnostic_turnaround(self):
        """Can record diagnostic turnaround times."""
        results = FullResultsCollector()

        results.record_diagnostic_turnaround(DiagnosticType.BLOODS, 45.0)
        results.record_diagnostic_turnaround(DiagnosticType.BLOODS, 50.0)

        assert len(results.diagnostic_turnarounds[DiagnosticType.BLOODS]) == 2

    def test_diagnostic_resource_logs_initialized(self):
        """Resource logs include diagnostic resources."""
        results = FullResultsCollector()

        for diag_type in DiagnosticType:
            assert f"diag_{diag_type.name}" in results.resource_logs


class TestDiagnosticMetrics:
    """Tests for diagnostic metrics computation."""

    def test_empty_metrics_includes_diagnostics(self):
        """Empty metrics include diagnostic metrics."""
        results = FullResultsCollector()
        metrics = results._empty_metrics()

        for diag_type in DiagnosticType:
            assert f"mean_wait_{diag_type.name}" in metrics
            assert f"p95_wait_{diag_type.name}" in metrics
            assert f"mean_turnaround_{diag_type.name}" in metrics
            assert f"util_{diag_type.name}" in metrics
            assert f"count_{diag_type.name}" in metrics

    def test_simulation_returns_diagnostic_metrics(self):
        """Simulation returns diagnostic metrics (zeros when loop not implemented)."""
        scenario = FullScenario(
            run_length=60,
            warm_up=0,
            random_seed=42,
        )

        results = run_full_simulation(scenario, use_multistream=False)

        # Should have diagnostic metrics (all zeros until loop implemented)
        for diag_type in DiagnosticType:
            assert f"mean_wait_{diag_type.name}" in results
            assert f"util_{diag_type.name}" in results
            assert f"count_{diag_type.name}" in results


class TestScenarioClone:
    """Tests for scenario cloning with diagnostic configs."""

    def test_clone_preserves_diagnostic_configs(self):
        """Cloned scenario preserves diagnostic configurations."""
        scenario = FullScenario(random_seed=42)
        # Modify a config
        scenario.diagnostic_configs[DiagnosticType.CT_SCAN].capacity = 5

        cloned = scenario.clone_with_seed(123)

        # Should preserve the modified config
        assert cloned.diagnostic_configs[DiagnosticType.CT_SCAN].capacity == 5

    def test_clone_has_different_rngs(self):
        """Cloned scenario has different RNG streams."""
        scenario = FullScenario(random_seed=42)
        cloned = scenario.clone_with_seed(123)

        # Sample from both - should be different
        val1 = scenario.rng_diagnostics.random()
        val2 = cloned.rng_diagnostics.random()

        # Reset and sample again to ensure reproducibility
        scenario2 = FullScenario(random_seed=42)
        cloned2 = scenario2.clone_with_seed(123)

        val1_again = scenario2.rng_diagnostics.random()
        val2_again = cloned2.rng_diagnostics.random()

        assert val1 == val1_again
        assert val2 == val2_again
        assert val1 != val2
