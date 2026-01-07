"""Tests for inter-facility transfers (Phase 7c).

Tests the transfer process where patients are transferred to other
hospitals/specialist centres after assessment in the ED.
"""

import pytest
import simpy

from faer.core.entities import (
    DiagnosticType, Priority, TransferType, TransferDestination,
)
from faer.core.scenario import FullScenario, TransferConfig
from faer.model.patient import Patient, Acuity, Disposition
from faer.model.full_model import (
    run_full_simulation,
    determine_transfer_required,
    select_transfer_destination,
    select_transfer_type,
    transfer_process,
    AEResources,
    FullResultsCollector,
)


class TestTransferConfig:
    """Tests for TransferConfig dataclass."""

    def test_default_transfer_config_created(self):
        """Default scenario has transfer config."""
        scenario = FullScenario()
        assert scenario.transfer_config is not None
        assert isinstance(scenario.transfer_config, TransferConfig)

    def test_transfer_enabled_by_default(self):
        """Transfers are enabled by default."""
        scenario = FullScenario()
        assert scenario.transfer_config.enabled is True

    def test_transfer_probabilities_by_priority(self):
        """Transfer probabilities vary by priority."""
        scenario = FullScenario()
        config = scenario.transfer_config

        # P1 should have higher transfer probability than P4
        assert config.probability_by_priority[Priority.P1_IMMEDIATE] > \
               config.probability_by_priority[Priority.P4_STANDARD]

    def test_default_transfer_resources(self):
        """Default scenario has transfer vehicle counts."""
        scenario = FullScenario()
        assert scenario.transfer_config.n_transfer_ambulances >= 1
        assert scenario.transfer_config.n_transfer_helicopters >= 1

    def test_custom_transfer_config(self):
        """Can create custom transfer configuration."""
        custom_config = TransferConfig(
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.5,
                Priority.P2_VERY_URGENT: 0.3,
                Priority.P3_URGENT: 0.1,
                Priority.P4_STANDARD: 0.05,
            },
            n_transfer_ambulances=5,
            n_transfer_helicopters=3,
            land_ambulance_wait_mean=30.0,
            helicopter_wait_mean=20.0,
        )

        assert custom_config.n_transfer_ambulances == 5
        assert custom_config.n_transfer_helicopters == 3
        assert custom_config.land_ambulance_wait_mean == 30.0


class TestDetermineTransferRequired:
    """Tests for determine_transfer_required function."""

    def test_p1_more_likely_to_need_transfer(self):
        """P1 patients should be more likely to need transfer."""
        scenario = FullScenario(random_seed=42)
        # Increase probabilities for testing
        scenario.transfer_config.probability_by_priority = {
            Priority.P1_IMMEDIATE: 0.5,
            Priority.P2_VERY_URGENT: 0.3,
            Priority.P3_URGENT: 0.1,
            Priority.P4_STANDARD: 0.05,
        }

        p1_count = 0
        p4_count = 0
        n_samples = 200

        for i in range(n_samples):
            p1_patient = Patient(id=i, arrival_time=0, acuity=Acuity.RESUS,
                                priority=Priority.P1_IMMEDIATE)
            p4_patient = Patient(id=i+1000, arrival_time=0, acuity=Acuity.MINORS,
                                priority=Priority.P4_STANDARD)

            if determine_transfer_required(p1_patient, scenario):
                p1_count += 1
            if determine_transfer_required(p4_patient, scenario):
                p4_count += 1

        # P1 should have significantly more transfers
        assert p1_count > p4_count

    def test_disabled_transfers_never_selected(self):
        """Disabled transfers should never be selected."""
        scenario = FullScenario(random_seed=42)
        scenario.transfer_config.enabled = False

        for i in range(50):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.RESUS,
                            priority=Priority.P1_IMMEDIATE)
            assert not determine_transfer_required(patient, scenario)

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce same transfer decisions."""
        scenario1 = FullScenario(random_seed=42)
        scenario1.transfer_config.probability_by_priority[Priority.P2_VERY_URGENT] = 0.5

        scenario2 = FullScenario(random_seed=42)
        scenario2.transfer_config.probability_by_priority[Priority.P2_VERY_URGENT] = 0.5

        patient1 = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                          priority=Priority.P2_VERY_URGENT)
        patient2 = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                          priority=Priority.P2_VERY_URGENT)

        result1 = determine_transfer_required(patient1, scenario1)
        result2 = determine_transfer_required(patient2, scenario2)

        assert result1 == result2


class TestSelectTransferDestination:
    """Tests for select_transfer_destination function."""

    def test_resus_patients_go_to_specialist_centres(self):
        """Resus patients should primarily go to specialist centres."""
        scenario = FullScenario(random_seed=42)
        destinations = []

        for i in range(100):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.RESUS,
                            priority=Priority.P1_IMMEDIATE)
            dest = select_transfer_destination(patient, scenario)
            destinations.append(dest)

        # Should have specialist destinations
        specialist = [
            TransferDestination.MAJOR_TRAUMA_CENTRE,
            TransferDestination.NEUROSURGERY,
            TransferDestination.CARDIAC_CENTRE,
            TransferDestination.PAEDIATRIC_ICU,
        ]
        specialist_count = sum(1 for d in destinations if d in specialist)
        assert specialist_count == len(destinations)  # All should be specialist

    def test_minors_primarily_go_to_regional(self):
        """Minors patients should primarily go to regional ICU."""
        scenario = FullScenario(random_seed=42)
        destinations = []

        for i in range(100):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.MINORS,
                            priority=Priority.P4_STANDARD)
            dest = select_transfer_destination(patient, scenario)
            destinations.append(dest)

        regional_count = sum(
            1 for d in destinations
            if d in (TransferDestination.REGIONAL_ICU, TransferDestination.BURNS_UNIT)
        )
        assert regional_count == len(destinations)


class TestSelectTransferType:
    """Tests for select_transfer_type function."""

    def test_p1_can_get_helicopter(self):
        """P1 patients should have a chance of helicopter transfer."""
        scenario = FullScenario(random_seed=42)
        scenario.transfer_config.helicopter_proportion_p1 = 0.3

        helicopter_count = 0
        n_samples = 200

        for i in range(n_samples):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.RESUS,
                            priority=Priority.P1_IMMEDIATE)
            transfer_type = select_transfer_type(patient, scenario)
            if transfer_type == TransferType.HELICOPTER:
                helicopter_count += 1

        # Should have some helicopter transfers (approximately 30%)
        assert helicopter_count > 0
        assert helicopter_count < n_samples  # Not all helicopters

    def test_p4_rarely_gets_helicopter(self):
        """P4 patients should rarely get helicopter transfer."""
        scenario = FullScenario(random_seed=42)

        helicopter_count = 0
        n_samples = 100

        for i in range(n_samples):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.MINORS,
                            priority=Priority.P4_STANDARD)
            transfer_type = select_transfer_type(patient, scenario)
            if transfer_type == TransferType.HELICOPTER:
                helicopter_count += 1

        # Should have no helicopter transfers for P4
        assert helicopter_count == 0

    def test_p1_can_get_critical_care(self):
        """P1 patients should have a chance of critical care ambulance."""
        scenario = FullScenario(random_seed=42)

        critical_care_count = 0
        n_samples = 200

        for i in range(n_samples):
            patient = Patient(id=i, arrival_time=0, acuity=Acuity.RESUS,
                            priority=Priority.P1_IMMEDIATE)
            transfer_type = select_transfer_type(patient, scenario)
            if transfer_type == TransferType.CRITICAL_CARE:
                critical_care_count += 1

        # Should have some critical care transfers
        assert critical_care_count > 0


class TestTransferProcess:
    """Tests for transfer_process generator."""

    def test_transfer_records_timestamps(self):
        """Transfer process should record all timestamps."""
        scenario = FullScenario(random_seed=42)
        env = simpy.Environment()

        resources = AEResources(
            triage=simpy.PriorityResource(env, capacity=2),
            ed_bays=simpy.PriorityResource(env, capacity=20),
            handover_bays=simpy.Resource(env, capacity=4),
            ambulance_fleet=simpy.Resource(env, capacity=10),
            helicopter_fleet=simpy.Resource(env, capacity=2),
            diagnostics={},
            transfer_ambulances=simpy.Resource(env, capacity=2),
            transfer_helicopters=simpy.Resource(env, capacity=1),
        )
        results = FullResultsCollector()

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                         priority=Priority.P2_VERY_URGENT)

        def run_transfer():
            yield from transfer_process(
                env, patient, resources, scenario, results
            )

        env.process(run_transfer())
        env.run()

        # Check timestamps were recorded
        assert patient.requires_transfer is True
        assert patient.transfer_decision_time is not None
        assert patient.transfer_requested_time is not None
        assert patient.transfer_vehicle_arrived_time is not None
        assert patient.transfer_departed_time is not None

    def test_transfer_records_type_and_destination(self):
        """Transfer should record type and destination."""
        scenario = FullScenario(random_seed=42)
        env = simpy.Environment()

        resources = AEResources(
            triage=simpy.PriorityResource(env, capacity=2),
            ed_bays=simpy.PriorityResource(env, capacity=20),
            handover_bays=simpy.Resource(env, capacity=4),
            ambulance_fleet=simpy.Resource(env, capacity=10),
            helicopter_fleet=simpy.Resource(env, capacity=2),
            diagnostics={},
            transfer_ambulances=simpy.Resource(env, capacity=2),
            transfer_helicopters=simpy.Resource(env, capacity=1),
        )
        results = FullResultsCollector()

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS,
                         priority=Priority.P1_IMMEDIATE)

        def run_transfer():
            yield from transfer_process(
                env, patient, resources, scenario, results
            )

        env.process(run_transfer())
        env.run()

        assert patient.transfer_type is not None
        assert patient.transfer_destination is not None

    def test_transfer_waits_recorded_in_results(self):
        """Transfer wait times should be recorded in results."""
        scenario = FullScenario(random_seed=42)
        env = simpy.Environment()

        # Only 1 transfer ambulance to force potential queuing
        resources = AEResources(
            triage=simpy.PriorityResource(env, capacity=2),
            ed_bays=simpy.PriorityResource(env, capacity=20),
            handover_bays=simpy.Resource(env, capacity=4),
            ambulance_fleet=simpy.Resource(env, capacity=10),
            helicopter_fleet=simpy.Resource(env, capacity=2),
            diagnostics={},
            transfer_ambulances=simpy.Resource(env, capacity=1),
            transfer_helicopters=simpy.Resource(env, capacity=1),
        )
        results = FullResultsCollector()

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MINORS,
                         priority=Priority.P3_URGENT)

        def run_transfer():
            yield from transfer_process(
                env, patient, resources, scenario, results
            )

        env.process(run_transfer())
        env.run()

        # Wait should be recorded for at least one transfer type
        total_waits = sum(len(waits) for waits in results.transfer_waits.values())
        assert total_waits >= 1


class TestTransfersInSimulation:
    """Tests for transfers integration in full simulation."""

    def test_simulation_with_transfers(self):
        """Simulation should run with transfers enabled."""
        scenario = FullScenario(
            run_length=120,
            warm_up=0,
            random_seed=42,
            arrival_rate=6.0,
        )

        results = run_full_simulation(scenario, use_multistream=False)

        # Should complete without error
        assert results['arrivals'] > 0
        assert results['departures'] > 0

        # Should have transfer metrics
        assert 'total_transfers' in results
        assert 'util_transfer_ambulances' in results
        assert 'util_transfer_helicopters' in results

    def test_simulation_counts_transfers_with_high_probability(self):
        """Simulation should count transfers when probability is high."""
        scenario = FullScenario(
            run_length=240,
            warm_up=0,
            random_seed=42,
            arrival_rate=10.0,
        )
        # Increase transfer probability for testing
        scenario.transfer_config.probability_by_priority = {
            Priority.P1_IMMEDIATE: 0.5,
            Priority.P2_VERY_URGENT: 0.3,
            Priority.P3_URGENT: 0.15,
            Priority.P4_STANDARD: 0.05,
        }

        results = run_full_simulation(scenario, use_multistream=False)

        # Should have some transfers
        assert results['total_transfers'] > 0

    def test_disabled_transfers_count_zero(self):
        """Disabled transfers should result in zero transfer count."""
        scenario = FullScenario(
            run_length=120,
            warm_up=0,
            random_seed=42,
            arrival_rate=6.0,
        )
        scenario.transfer_config.enabled = False

        results = run_full_simulation(scenario, use_multistream=False)

        assert results['total_transfers'] == 0


class TestPatientTransferTracking:
    """Tests for Patient class transfer tracking."""

    def test_patient_records_transfer_info(self):
        """Patient should track transfer information."""
        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                         priority=Priority.P2_VERY_URGENT)

        patient.record_transfer(
            transfer_type=TransferType.LAND_AMBULANCE,
            destination=TransferDestination.REGIONAL_ICU,
            decision_time=10.0,
            requested_time=15.0,
            vehicle_arrived_time=45.0,
            departed_time=60.0,
        )

        assert patient.requires_transfer is True
        assert patient.transfer_type == TransferType.LAND_AMBULANCE
        assert patient.transfer_destination == TransferDestination.REGIONAL_ICU
        assert patient.transfer_decision_time == 10.0
        assert patient.transfer_requested_time == 15.0
        assert patient.transfer_vehicle_arrived_time == 45.0
        assert patient.transfer_departed_time == 60.0

    def test_transfer_wait_time_property(self):
        """transfer_wait_time should return correct value."""
        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                         priority=Priority.P2_VERY_URGENT)

        patient.transfer_requested_time = 15.0
        patient.transfer_vehicle_arrived_time = 45.0

        assert patient.transfer_wait_time == 30.0

    def test_transfer_total_time_property(self):
        """transfer_total_time should return correct value."""
        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS,
                         priority=Priority.P2_VERY_URGENT)

        patient.transfer_decision_time = 10.0
        patient.transfer_departed_time = 60.0

        assert patient.transfer_total_time == 50.0


class TestTransferResultsCollector:
    """Tests for transfer metrics in FullResultsCollector."""

    def test_transfer_tracking_initialized(self):
        """Results collector initializes transfer tracking."""
        results = FullResultsCollector()

        for transfer_type in TransferType:
            assert transfer_type in results.transfer_waits
            assert transfer_type in results.transfer_total_times
            assert results.transfer_waits[transfer_type] == []
            assert results.transfer_total_times[transfer_type] == []

        for dest in TransferDestination:
            assert dest in results.transfers_by_destination
            assert results.transfers_by_destination[dest] == 0

    def test_record_transfer_wait(self):
        """Can record transfer wait times."""
        results = FullResultsCollector()

        results.record_transfer_wait(TransferType.LAND_AMBULANCE, 15.0)
        results.record_transfer_wait(TransferType.LAND_AMBULANCE, 20.0)

        assert len(results.transfer_waits[TransferType.LAND_AMBULANCE]) == 2
        assert results.transfer_waits[TransferType.LAND_AMBULANCE][0] == 15.0

    def test_record_transfer_destination(self):
        """Can record transfer destinations."""
        results = FullResultsCollector()

        results.record_transfer_destination(TransferDestination.MAJOR_TRAUMA_CENTRE)
        results.record_transfer_destination(TransferDestination.MAJOR_TRAUMA_CENTRE)
        results.record_transfer_destination(TransferDestination.CARDIAC_CENTRE)

        assert results.transfers_by_destination[TransferDestination.MAJOR_TRAUMA_CENTRE] == 2
        assert results.transfers_by_destination[TransferDestination.CARDIAC_CENTRE] == 1


class TestTransferMetrics:
    """Tests for transfer metrics computation."""

    def test_empty_metrics_includes_transfers(self):
        """Empty metrics include transfer metrics."""
        results = FullResultsCollector()
        metrics = results._empty_metrics()

        for transfer_type in TransferType:
            assert f"mean_transfer_wait_{transfer_type.name}" in metrics
            assert f"p95_transfer_wait_{transfer_type.name}" in metrics
            assert f"count_transfer_{transfer_type.name}" in metrics

        assert "util_transfer_ambulances" in metrics
        assert "util_transfer_helicopters" in metrics
        assert "total_transfers" in metrics

        for dest in TransferDestination:
            assert f"transfers_to_{dest.name}" in metrics


class TestScenarioClone:
    """Tests for scenario cloning with transfer config."""

    def test_clone_preserves_transfer_config(self):
        """Cloned scenario preserves transfer configuration."""
        scenario = FullScenario(random_seed=42)
        scenario.transfer_config.n_transfer_ambulances = 5
        scenario.transfer_config.helicopter_wait_mean = 25.0

        cloned = scenario.clone_with_seed(123)

        assert cloned.transfer_config.n_transfer_ambulances == 5
        assert cloned.transfer_config.helicopter_wait_mean == 25.0

    def test_clone_has_different_transfer_rngs(self):
        """Cloned scenario has different transfer RNG streams."""
        scenario = FullScenario(random_seed=42)
        cloned = scenario.clone_with_seed(123)

        # Sample from both - should be different
        val1 = scenario.rng_transfer.random()
        val2 = cloned.rng_transfer.random()

        # Reset and sample again to ensure reproducibility
        scenario2 = FullScenario(random_seed=42)
        cloned2 = scenario2.clone_with_seed(123)

        val1_again = scenario2.rng_transfer.random()
        val2_again = cloned2.rng_transfer.random()

        assert val1 == val1_again
        assert val2 == val2_again
        assert val1 != val2
