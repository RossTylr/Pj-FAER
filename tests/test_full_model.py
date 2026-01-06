"""Tests for full A&E pathway model (Phase 5 - simplified ED)."""

import pytest
import numpy as np

from faer.core.scenario import FullScenario
from faer.model.patient import Patient, Acuity, Disposition
from faer.model.full_model import (
    run_full_simulation,
    assign_acuity,
)


class TestPatient:
    """Test Patient entity."""

    def test_create_patient(self):
        """Can create a patient with acuity."""
        patient = Patient(id=1, arrival_time=10.0, acuity=Acuity.MAJORS)

        assert patient.id == 1
        assert patient.arrival_time == 10.0
        assert patient.acuity == Acuity.MAJORS

    def test_patient_timestamps(self):
        """Patient timestamps compute correctly."""
        patient = Patient(id=1, arrival_time=0.0, acuity=Acuity.MAJORS)

        patient.record_triage(5.0, 10.0)
        patient.record_treatment(15.0, 60.0)
        patient.record_departure(65.0, Disposition.DISCHARGE)

        assert patient.triage_wait == 5.0
        assert patient.triage_duration == 5.0
        assert patient.treatment_wait == 5.0  # 15 - 10
        assert patient.treatment_duration == 45.0
        assert patient.system_time == 65.0

    def test_patient_is_admitted(self):
        """is_admitted property works correctly."""
        patient1 = Patient(id=1, arrival_time=0.0, acuity=Acuity.MAJORS)
        patient1.disposition = Disposition.ADMIT_WARD

        patient2 = Patient(id=2, arrival_time=0.0, acuity=Acuity.MINORS)
        patient2.disposition = Disposition.DISCHARGE

        assert patient1.is_admitted is True
        assert patient2.is_admitted is False


class TestFullScenario:
    """Test FullScenario configuration (Phase 5)."""

    def test_default_scenario(self):
        """Default FullScenario has valid values."""
        scenario = FullScenario()

        assert scenario.arrival_rate == 6.0
        assert scenario.n_triage == 2
        assert scenario.n_ed_bays == 20  # Phase 5: single ED pool

    def test_acuity_mix_validation(self):
        """Acuity mix must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            FullScenario(p_resus=0.5, p_majors=0.5, p_minors=0.5)

    def test_rng_streams_created(self):
        """RNG streams are created."""
        scenario = FullScenario()

        assert scenario.rng_arrivals is not None
        assert scenario.rng_acuity is not None
        assert scenario.rng_triage is not None
        assert scenario.rng_treatment is not None

    def test_clone_with_seed(self):
        """Clone creates new scenario with different seed."""
        original = FullScenario(random_seed=42)
        cloned = original.clone_with_seed(99)

        assert cloned.random_seed == 99
        assert cloned.arrival_rate == original.arrival_rate


class TestAcuityAssignment:
    """Test acuity assignment."""

    def test_acuity_distribution(self):
        """Acuity assignment follows probabilities."""
        scenario = FullScenario(
            p_resus=0.1, p_majors=0.6, p_minors=0.3, random_seed=42
        )

        counts = {Acuity.RESUS: 0, Acuity.MAJORS: 0, Acuity.MINORS: 0}
        for _ in range(1000):
            acuity = assign_acuity(scenario)
            counts[acuity] += 1

        # Check proportions are approximately correct (within 5%)
        assert 50 < counts[Acuity.RESUS] < 150  # ~10%
        assert 550 < counts[Acuity.MAJORS] < 650  # ~60%
        assert 250 < counts[Acuity.MINORS] < 350  # ~30%


class TestFullSimulation:
    """Test full A&E simulation (Phase 5)."""

    def test_simulation_runs(self):
        """Simulation runs without error."""
        # Need longer run to allow patients to complete treatment
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert results["arrivals"] > 0
        assert results["departures"] > 0

    def test_reproducibility(self):
        """Same seed produces same results."""
        scenario1 = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        scenario2 = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)

        results1 = run_full_simulation(scenario1)
        results2 = run_full_simulation(scenario2)

        assert results1["arrivals"] == results2["arrivals"]
        assert results1["departures"] == results2["departures"]

    def test_acuity_counts(self):
        """Arrivals are split by acuity."""
        scenario = FullScenario(
            run_length=480.0, warm_up=0.0, random_seed=42
        )
        results = run_full_simulation(scenario)

        total_by_acuity = (
            results["arrivals_resus"]
            + results["arrivals_majors"]
            + results["arrivals_minors"]
        )
        assert total_by_acuity == results["arrivals"]

    def test_p1_bypass_triage(self):
        """P1 patients have zero triage wait (bypass triage)."""
        scenario = FullScenario(
            p_resus=1.0, p_majors=0.0, p_minors=0.0,  # All Resus (P1)
            run_length=480.0, warm_up=0.0, random_seed=42
        )
        results = run_full_simulation(scenario)

        # P1 patients bypass triage, so triage wait should be 0
        assert results["mean_triage_wait"] == 0.0

    def test_admission_rate(self):
        """Admission rate is reasonable."""
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        # With default mix, expect some admissions
        assert 0.0 < results["admission_rate"] < 1.0
        assert results["admitted"] + results["discharged"] == results["departures"]

    def test_utilisation_range(self):
        """Utilisation is between 0 and 1 (Phase 5: single ED pool)."""
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert 0.0 <= results["util_triage"] <= 1.0
        assert 0.0 <= results["util_ed_bays"] <= 1.0

    def test_warm_up_excludes_early_patients(self):
        """Warm-up period patients are excluded from metrics."""
        # Without warm-up
        scenario1 = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results1 = run_full_simulation(scenario1)

        # With warm-up (same total time, but first 60 min is warm-up)
        scenario2 = FullScenario(run_length=420.0, warm_up=60.0, random_seed=42)
        results2 = run_full_simulation(scenario2)

        # With warm-up, fewer arrivals should be counted
        assert results2["arrivals"] < results1["arrivals"]


class TestResourceConstraints:
    """Test resource constraint effects (Phase 5)."""

    def test_low_capacity_increases_wait(self):
        """Low capacity leads to longer waits."""
        # High capacity
        high_cap = FullScenario(
            n_ed_bays=30, run_length=480.0, warm_up=0.0, random_seed=42
        )
        # Low capacity
        low_cap = FullScenario(
            n_ed_bays=5, run_length=480.0, warm_up=0.0, random_seed=42
        )

        results_high = run_full_simulation(high_cap)
        results_low = run_full_simulation(low_cap)

        # Low capacity should have higher mean treatment wait
        assert results_low["mean_treatment_wait"] >= results_high["mean_treatment_wait"]

    def test_high_utilisation_with_low_capacity(self):
        """Low capacity leads to higher utilisation."""
        low_cap = FullScenario(
            n_ed_bays=5,
            arrival_rate=10.0,  # High arrival rate
            run_length=480.0, warm_up=0.0, random_seed=42
        )
        results = run_full_simulation(low_cap)

        # Expect moderate to high utilisation
        assert results["util_ed_bays"] > 0.3


class TestPriorityQueuing:
    """Test priority queuing in single ED pool (Phase 5)."""

    def test_p1_shorter_wait_than_p4(self):
        """P1 patients should have shorter waits than P4."""
        # Constrained resources to force queuing
        scenario = FullScenario(
            n_ed_bays=5,
            arrival_rate=10.0,
            run_length=480.0,
            warm_up=60.0,
            random_seed=42
        )
        results = run_full_simulation(scenario)

        # P1 should have shorter wait than P4
        assert results["P1_mean_wait"] <= results["P4_mean_wait"]

    def test_priority_counts(self):
        """Arrivals are split by priority."""
        scenario = FullScenario(
            run_length=480.0, warm_up=0.0, random_seed=42
        )
        results = run_full_simulation(scenario)

        total_by_priority = (
            results["arrivals_P1"]
            + results["arrivals_P2"]
            + results["arrivals_P3"]
            + results["arrivals_P4"]
        )
        assert total_by_priority == results["arrivals"]
