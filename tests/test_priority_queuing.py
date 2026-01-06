"""Tests for priority-based queuing (Phase 5 - single ED pool)."""

import pytest
import simpy

from faer.model.patient import Priority, Patient, Acuity
from faer.core.scenario import FullScenario
from faer.model.full_model import (
    run_full_simulation,
    assign_priority,
    AEResources,
    FullResultsCollector,
    patient_journey,
)


class TestPriorityEnum:
    """Test Priority enum ordering and values."""

    def test_priority_ordering(self):
        """P1 < P2 < P3 < P4 (lower is more urgent)."""
        assert Priority.P1_IMMEDIATE < Priority.P2_VERY_URGENT
        assert Priority.P2_VERY_URGENT < Priority.P3_URGENT
        assert Priority.P3_URGENT < Priority.P4_STANDARD

    def test_priority_values(self):
        """Priority values are 1-4."""
        assert Priority.P1_IMMEDIATE.value == 1
        assert Priority.P2_VERY_URGENT.value == 2
        assert Priority.P3_URGENT.value == 3
        assert Priority.P4_STANDARD.value == 4

    def test_priority_is_int(self):
        """Priority can be used as integer for SimPy."""
        assert int(Priority.P1_IMMEDIATE) == 1
        assert Priority.P1_IMMEDIATE.value == 1


class TestPriorityAssignment:
    """Test priority assignment based on acuity."""

    def test_resus_always_p1(self):
        """Resus patients are always P1."""
        import numpy as np
        rng = np.random.default_rng(42)

        for _ in range(100):
            priority = assign_priority(Acuity.RESUS, rng)
            assert priority == Priority.P1_IMMEDIATE

    def test_majors_distribution(self):
        """Majors patients are P2 (70%) or P3 (30%)."""
        import numpy as np
        rng = np.random.default_rng(42)

        priorities = [assign_priority(Acuity.MAJORS, rng) for _ in range(1000)]
        p2_count = sum(1 for p in priorities if p == Priority.P2_VERY_URGENT)
        p3_count = sum(1 for p in priorities if p == Priority.P3_URGENT)

        # Check roughly 70% P2, 30% P3 (within 5%)
        assert 0.65 < p2_count / 1000 < 0.75
        assert 0.25 < p3_count / 1000 < 0.35

    def test_minors_distribution(self):
        """Minors patients are P3 (60%) or P4 (40%)."""
        import numpy as np
        rng = np.random.default_rng(42)

        priorities = [assign_priority(Acuity.MINORS, rng) for _ in range(1000)]
        p3_count = sum(1 for p in priorities if p == Priority.P3_URGENT)
        p4_count = sum(1 for p in priorities if p == Priority.P4_STANDARD)

        # Check roughly 60% P3, 40% P4 (within 5%)
        assert 0.55 < p3_count / 1000 < 0.65
        assert 0.35 < p4_count / 1000 < 0.45


class TestPatientPriority:
    """Test Patient class with priority."""

    def test_patient_has_priority(self):
        """Patient can be created with priority."""
        patient = Patient(
            id=1,
            arrival_time=0.0,
            acuity=Acuity.RESUS,
            priority=Priority.P1_IMMEDIATE,
        )
        assert patient.priority == Priority.P1_IMMEDIATE

    def test_patient_default_priority(self):
        """Patient has default priority P3."""
        patient = Patient(id=1, arrival_time=0.0, acuity=Acuity.MAJORS)
        assert patient.priority == Priority.P3_URGENT


class TestPriorityQueuing:
    """Test priority-based queuing behavior."""

    def test_p1_served_before_p4(self):
        """When P1 and P4 are both queuing, P1 gets served first."""
        env = simpy.Environment()
        resource = simpy.PriorityResource(env, capacity=1)

        service_order = []

        def patient_process(env, patient_id, priority):
            with resource.request(priority=priority.value) as req:
                yield req
                service_order.append((patient_id, priority))
                yield env.timeout(10)

        # Occupy the resource first
        def blocker(env):
            with resource.request(priority=Priority.P3_URGENT.value) as req:
                yield req
                yield env.timeout(5)  # Block for 5 time units

        # Start blocker
        env.process(blocker(env))

        # At time 1, P4 arrives (will queue)
        def delayed_p4(env):
            yield env.timeout(1)
            yield from patient_process(env, "P4_patient", Priority.P4_STANDARD)

        # At time 2, P1 arrives (should jump queue)
        def delayed_p1(env):
            yield env.timeout(2)
            yield from patient_process(env, "P1_patient", Priority.P1_IMMEDIATE)

        env.process(delayed_p4(env))
        env.process(delayed_p1(env))

        env.run()

        # P1 should be served before P4
        assert len(service_order) == 2
        assert service_order[0][0] == "P1_patient"
        assert service_order[1][0] == "P4_patient"

    def test_simulation_with_priorities(self):
        """Full simulation runs with priority queuing."""
        scenario = FullScenario(run_length=480.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario)

        assert results["arrivals"] > 0
        assert results["departures"] > 0

    def test_priority_reproducibility(self):
        """Same seed produces same priority assignments."""
        scenario1 = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)
        scenario2 = FullScenario(run_length=240.0, warm_up=0.0, random_seed=42)

        results1 = run_full_simulation(scenario1)
        results2 = run_full_simulation(scenario2)

        assert results1["arrivals"] == results2["arrivals"]
        assert results1["departures"] == results2["departures"]


class TestHighAcuityPrioritized:
    """Test that high acuity patients are appropriately prioritized."""

    def test_resus_has_shorter_wait_than_minors(self):
        """In constrained scenario, Resus (P1) waits less than Minors (P3/P4)."""
        # Create constrained scenario where queuing is likely (Phase 5: single pool)
        scenario = FullScenario(
            run_length=480.0,
            warm_up=60.0,
            arrival_rate=10.0,  # High arrival rate
            n_ed_bays=5,        # Constrained capacity
            random_seed=42,
        )
        results = run_full_simulation(scenario)

        # Resus patients should generally have lower wait times
        # (They're P1 and bypass triage)
        assert results["resus_mean_wait"] <= results["minors_mean_wait"]
