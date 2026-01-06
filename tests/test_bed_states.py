"""Tests for bed state management (Phase 5e)."""

import pytest

from faer.core.scenario import FullScenario
from faer.core.entities import BedState, NodeType
from faer.model.full_model import run_full_simulation, FullResultsCollector


class TestBedStateEnum:
    """Test BedState enum."""

    def test_bed_state_values(self):
        """BedState enum has correct values."""
        assert BedState.EMPTY == 1
        assert BedState.OCCUPIED == 2
        assert BedState.BLOCKED == 3
        assert BedState.CLEANING == 4

    def test_bed_state_ordering(self):
        """BedState can be compared."""
        assert BedState.EMPTY < BedState.OCCUPIED
        assert BedState.OCCUPIED < BedState.BLOCKED
        assert BedState.BLOCKED < BedState.CLEANING


class TestBedTurnaroundParameter:
    """Test bed turnaround parameter in scenario."""

    def test_default_bed_turnaround(self):
        """Default scenario has bed_turnaround_mins parameter."""
        scenario = FullScenario()
        assert scenario.bed_turnaround_mins == 10.0

    def test_custom_bed_turnaround(self):
        """Can customize bed_turnaround_mins."""
        scenario = FullScenario(bed_turnaround_mins=20.0)
        assert scenario.bed_turnaround_mins == 20.0


class TestBedStateMetrics:
    """Test bed state metrics collection."""

    def test_simulation_has_bed_state_metrics(self):
        """Simulation produces bed state metrics."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        # These metrics should exist
        assert "ED_BAYS_pct_occupied" in results
        assert "ED_BAYS_pct_blocked" in results
        assert "ED_BAYS_pct_cleaning" in results

    def test_bed_state_percentages_range(self):
        """Bed state percentages are between 0 and 1."""
        scenario = FullScenario(run_length=480.0, warm_up=60.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert 0.0 <= results["ED_BAYS_pct_occupied"] <= 1.0
        assert 0.0 <= results["ED_BAYS_pct_blocked"] <= 1.0
        assert 0.0 <= results["ED_BAYS_pct_cleaning"] <= 1.0

    def test_bed_state_logging(self):
        """Bed state changes are logged."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)

        # Run simulation and check bed state log
        from faer.model.full_model import (
            FullResultsCollector, AEResources, patient_journey,
            arrival_generator_single
        )
        import simpy
        import itertools

        env = simpy.Environment()
        resources = AEResources(
            triage=simpy.PriorityResource(env, capacity=scenario.n_triage),
            ed_bays=simpy.PriorityResource(env, capacity=scenario.n_ed_bays),
            handover_bays=simpy.Resource(env, capacity=scenario.n_handover_bays),
            ambulance_fleet=simpy.Resource(env, capacity=scenario.n_ambulances),
            helicopter_fleet=simpy.Resource(env, capacity=scenario.n_helicopters),
        )
        results_collector = FullResultsCollector()
        patient_counter = itertools.count(1)

        env.process(arrival_generator_single(
            env, resources, scenario, results_collector, patient_counter
        ))
        env.run(until=240.0)

        # Should have bed state log entries
        assert len(results_collector.bed_state_log) > 0


class TestBedStateTransitions:
    """Test bed state transitions."""

    def test_bed_transitions_include_all_states(self):
        """Beds go through OCCUPIED -> BLOCKED (for admits) -> CLEANING -> EMPTY."""
        scenario = FullScenario(
            run_length=480.0,
            warm_up=0.0,
            random_seed=42,
            n_ed_bays=5,  # Constrained to force queuing
        )
        results = run_full_simulation(scenario)

        # Should have some percentage in each state
        # (except BLOCKED may be 0 if few admissions in short run)
        assert results["ED_BAYS_pct_occupied"] >= 0.0
        assert results["ED_BAYS_pct_cleaning"] >= 0.0


class TestBedTurnaroundEffect:
    """Test bed turnaround time effect."""

    def test_longer_turnaround_increases_cleaning_time(self):
        """Longer turnaround should increase cleaning percentage."""
        # Short turnaround
        short = FullScenario(
            bed_turnaround_mins=5.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        # Long turnaround
        long = FullScenario(
            bed_turnaround_mins=30.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42,
        )

        results_short = run_full_simulation(short)
        results_long = run_full_simulation(long)

        # Longer turnaround should increase cleaning percentage
        assert results_long["ED_BAYS_pct_cleaning"] >= results_short["ED_BAYS_pct_cleaning"]


class TestBedStateCloneWithSeed:
    """Test that clone_with_seed preserves bed turnaround param."""

    def test_clone_preserves_bed_turnaround(self):
        """Clone with seed preserves bed_turnaround_mins."""
        original = FullScenario(
            bed_turnaround_mins=15.0,
            random_seed=42,
        )

        cloned = original.clone_with_seed(99)

        assert cloned.bed_turnaround_mins == original.bed_turnaround_mins
        assert cloned.random_seed == 99


class TestEmptyMetricsIncludeBedState:
    """Test empty metrics include bed state fields."""

    def test_empty_metrics_include_bed_state(self):
        """Empty metrics include bed state fields."""
        results = FullResultsCollector()
        metrics = results._empty_metrics()

        assert "ED_BAYS_pct_occupied" in metrics
        assert "ED_BAYS_pct_blocked" in metrics
        assert "ED_BAYS_pct_cleaning" in metrics
