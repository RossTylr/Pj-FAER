"""Tests for two-phase acquire blocking pattern (Phase 5)."""

import pytest

from faer.core.entities import NodeType
from faer.core.scenario import FullScenario
from faer.model.full_model import run_full_simulation, FullResultsCollector
from faer.model.patient import Patient, Acuity


class TestBoardingMetrics:
    """Test boarding metrics collection."""

    def test_results_collector_has_boarding_events(self):
        """Results collector can record boarding events."""
        results = FullResultsCollector()

        results.record_boarding(patient_id=1, from_node="ed_bays", duration=30.0)
        results.record_boarding(patient_id=2, from_node="ed_bays", duration=45.0)

        assert len(results.boarding_events) == 2
        assert results.boarding_events[0] == (1, "ed_bays", 30.0)

    def test_zero_duration_not_recorded(self):
        """Zero duration boarding is not recorded."""
        results = FullResultsCollector()

        results.record_boarding(patient_id=1, from_node="ed_bays", duration=0.0)

        assert len(results.boarding_events) == 0

    def test_simulation_has_boarding_metrics(self):
        """Simulation produces boarding metrics."""
        scenario = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        # These metrics should exist (even if zero)
        assert "mean_boarding_time" in results
        assert "total_boarding_events" in results
        assert "p_boarding" in results


class TestPatientCurrentNode:
    """Test patient current_node tracking."""

    def test_patient_has_current_node(self):
        """Patient can track current node."""
        patient = Patient(
            id=1,
            arrival_time=0.0,
            acuity=Acuity.MAJORS,
        )
        assert patient.current_node is None

        patient.current_node = NodeType.ED_BAYS
        assert patient.current_node == NodeType.ED_BAYS

    def test_node_transitions(self):
        """Patient can transition between nodes."""
        patient = Patient(id=1, arrival_time=0.0, acuity=Acuity.RESUS)

        patient.current_node = NodeType.ED_BAYS
        assert patient.current_node == NodeType.ED_BAYS

        patient.current_node = NodeType.ITU
        assert patient.current_node == NodeType.ITU

        patient.current_node = NodeType.WARD
        assert patient.current_node == NodeType.WARD


class TestSimulationWithBlockingMetrics:
    """Test simulation produces blocking-related metrics."""

    def test_default_simulation_runs(self):
        """Default simulation still runs correctly."""
        scenario = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert results["arrivals"] > 0
        assert results["departures"] > 0

    def test_boarding_metrics_present(self):
        """Boarding metrics are present in results."""
        scenario = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert results["mean_boarding_time"] >= 0.0
        assert results["max_boarding_time"] >= 0.0
        assert results["total_boarding_events"] >= 0
        assert 0.0 <= results["p_boarding"] <= 1.0

    def test_empty_metrics_include_boarding(self):
        """Empty metrics include boarding fields."""
        results = FullResultsCollector()
        scenario = FullScenario()
        metrics = results._empty_metrics()

        assert "mean_boarding_time" in metrics
        assert "total_boarding_events" in metrics
        assert "p_boarding" in metrics
