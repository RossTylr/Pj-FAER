"""Tests for aeromedical evacuation processes (Phase 10).

Tests cover:
- AeromedConfig, HEMSConfig, FixedWingConfig, MissedSlotConfig dataclasses
- Patient aeromed fields and properties
- HEMS evacuation process
- Fixed-wing evacuation with slot scheduling
- Missed slot handling and blocking cascade
- Discharge pathway selection
"""

import pytest
import simpy
import numpy as np


# =============================================================================
# Config Dataclass Tests
# =============================================================================


class TestAeromedConfig:
    """Test AeromedConfig and sub-configs."""

    def test_default_aeromed_config_disabled(self):
        """AeromedConfig is disabled by default."""
        from faer.core.scenario import AeromedConfig

        config = AeromedConfig()
        assert config.enabled is False
        assert config.p1_aeromed_probability == 0.05
        assert config.fixedwing_proportion == 0.3

    def test_aeromed_config_has_hems(self):
        """AeromedConfig includes HEMSConfig."""
        from faer.core.scenario import AeromedConfig, HEMSConfig

        config = AeromedConfig()
        assert isinstance(config.hems, HEMSConfig)
        assert config.hems.enabled is True
        assert config.hems.slots_per_day == 6

    def test_aeromed_config_has_fixedwing(self):
        """AeromedConfig includes FixedWingConfig."""
        from faer.core.scenario import AeromedConfig, FixedWingConfig

        config = AeromedConfig()
        assert isinstance(config.fixedwing, FixedWingConfig)
        assert config.fixedwing.enabled is False  # Disabled by default
        assert config.fixedwing.slots_per_segment == [1, 0]

    def test_aeromed_config_has_missed_slot(self):
        """AeromedConfig includes MissedSlotConfig."""
        from faer.core.scenario import AeromedConfig, MissedSlotConfig

        config = AeromedConfig()
        assert isinstance(config.missed_slot, MissedSlotConfig)
        assert config.missed_slot.requires_restabilisation is False


class TestHEMSConfig:
    """Test HEMSConfig dataclass."""

    def test_hems_default_values(self):
        """HEMSConfig has sensible defaults."""
        from faer.core.scenario import HEMSConfig

        config = HEMSConfig()
        assert config.enabled is True
        assert config.slots_per_day == 6
        assert config.operating_start_hour == 7
        assert config.operating_end_hour == 21
        assert config.stabilisation_mins == (30.0, 120.0)
        assert config.transfer_to_helipad_mins == (15.0, 45.0)
        assert config.flight_duration_mins == (15.0, 60.0)

    def test_hems_custom_values(self):
        """HEMSConfig accepts custom values."""
        from faer.core.scenario import HEMSConfig

        config = HEMSConfig(
            slots_per_day=4,
            operating_start_hour=8,
            operating_end_hour=20,
        )
        assert config.slots_per_day == 4
        assert config.operating_start_hour == 8
        assert config.operating_end_hour == 20


class TestFixedWingConfig:
    """Test FixedWingConfig dataclass."""

    def test_fixedwing_default_values(self):
        """FixedWingConfig has sensible defaults."""
        from faer.core.scenario import FixedWingConfig

        config = FixedWingConfig()
        assert config.enabled is False
        assert config.slots_per_segment == [1, 0]  # 1 AM slot, 0 PM slots
        assert config.departure_hour_in_segment == 2
        assert config.cutoff_hours_before == 4
        assert config.stabilisation_mins == (120.0, 240.0)

    def test_fixedwing_custom_slot_pattern(self):
        """FixedWingConfig accepts custom 5-day slot pattern."""
        from faer.core.scenario import FixedWingConfig

        # 5-day pattern: 2 slots AM, 1 slot PM each day
        slots = [2, 1] * 5  # 10 segments = 5 days
        config = FixedWingConfig(
            enabled=True,
            slots_per_segment=slots,
        )
        assert len(config.slots_per_segment) == 10
        assert sum(config.slots_per_segment) == 15  # Total slots over 5 days


class TestMissedSlotConfig:
    """Test MissedSlotConfig dataclass."""

    def test_missed_slot_default_no_restab(self):
        """MissedSlotConfig defaults to no re-stabilisation."""
        from faer.core.scenario import MissedSlotConfig

        config = MissedSlotConfig()
        assert config.requires_restabilisation is False
        assert config.restabilisation_factor == 0.3
        assert config.max_wait_before_restab_hours == 24.0

    def test_missed_slot_with_restab(self):
        """MissedSlotConfig can require re-stabilisation."""
        from faer.core.scenario import MissedSlotConfig

        config = MissedSlotConfig(
            requires_restabilisation=True,
            restabilisation_factor=0.5,
            max_wait_before_restab_hours=12.0,
        )
        assert config.requires_restabilisation is True
        assert config.restabilisation_factor == 0.5


# =============================================================================
# FullScenario Aeromed Integration Tests
# =============================================================================


class TestScenarioAeromedConfig:
    """Test aeromed config integration with FullScenario."""

    def test_scenario_has_aeromed_config(self):
        """FullScenario initializes aeromed_config."""
        from faer.core.scenario import FullScenario

        scenario = FullScenario()
        assert scenario.aeromed_config is not None
        assert scenario.aeromed_config.enabled is False

    def test_scenario_has_aeromed_rng(self):
        """FullScenario initializes rng_aeromed."""
        from faer.core.scenario import FullScenario

        scenario = FullScenario()
        assert scenario.rng_aeromed is not None
        assert isinstance(scenario.rng_aeromed, np.random.Generator)

    def test_scenario_clone_preserves_aeromed(self):
        """clone_with_seed preserves aeromed_config."""
        from faer.core.scenario import FullScenario, AeromedConfig

        scenario = FullScenario(
            aeromed_config=AeromedConfig(enabled=True, p1_aeromed_probability=0.10),
        )
        cloned = scenario.clone_with_seed(123)

        assert cloned.aeromed_config.enabled is True
        assert cloned.aeromed_config.p1_aeromed_probability == 0.10

    def test_scenario_clone_has_different_aeromed_rng(self):
        """clone_with_seed creates fresh rng_aeromed."""
        from faer.core.scenario import FullScenario

        scenario = FullScenario(random_seed=42)
        cloned = scenario.clone_with_seed(99)

        # Generate values from each RNG
        val1 = scenario.rng_aeromed.random()
        val2 = cloned.rng_aeromed.random()

        # Should be different due to different seeds
        assert val1 != val2


# =============================================================================
# Patient Aeromed Fields Tests
# =============================================================================


class TestPatientAeromedFields:
    """Test Patient aeromed tracking fields."""

    def test_patient_aeromed_defaults(self):
        """Patient has default aeromed field values."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)

        assert patient.requires_aeromed is False
        assert patient.aeromed_type is None
        assert patient.aeromed_stabilisation_start is None
        assert patient.aeromed_stabilisation_end is None
        assert patient.aeromed_wait_for_slot is None
        assert patient.aeromed_slot_missed is False
        assert patient.aeromed_departure is None

    def test_patient_aeromed_stabilisation_duration(self):
        """Patient computes aeromed stabilisation duration correctly."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)
        patient.aeromed_stabilisation_start = 100
        patient.aeromed_stabilisation_end = 190

        assert patient.aeromed_stabilisation_duration == 90

    def test_patient_aeromed_total_time(self):
        """Patient computes total aeromed process time."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)
        patient.aeromed_stabilisation_start = 100
        patient.aeromed_departure = 250

        assert patient.aeromed_total_time == 150

    def test_patient_record_aeromed_evacuation(self):
        """Patient records aeromed evacuation timestamps."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)
        patient.record_aeromed_evacuation(
            aeromed_type="HEMS",
            stabilisation_start=100,
            stabilisation_end=160,
            wait_for_slot=30,
            departure=220,
            slot_missed=False,
        )

        assert patient.requires_aeromed is True
        assert patient.aeromed_type == "HEMS"
        assert patient.aeromed_stabilisation_start == 100
        assert patient.aeromed_stabilisation_end == 160
        assert patient.aeromed_wait_for_slot == 30
        assert patient.aeromed_departure == 220
        assert patient.aeromed_slot_missed is False
        assert "aeromed_hems" in patient.resources_used


# =============================================================================
# HEMS Slot Resource Tests
# =============================================================================


class TestHEMSSlotBehaviour:
    """Test HEMS slot resource behaviour."""

    def test_hems_slots_limit_concurrent_evacuations(self):
        """HEMS slots limit concurrent evacuations."""
        env = simpy.Environment()
        hems_slots = simpy.Resource(env, capacity=2)

        events = []

        def patient_hems(env, patient_id, hems_slots):
            arrival = env.now
            with hems_slots.request() as req:
                yield req
                wait = env.now - arrival
                events.append((patient_id, 'slot_acquired', wait))
                yield env.timeout(60)  # Flight + transfer time

        # 3 patients need HEMS, only 2 slots available
        for i in range(3):
            env.process(patient_hems(env, i, hems_slots))

        env.run(until=200)

        # First 2 should get slots immediately, third waits
        waits = [e[2] for e in events]
        assert waits[0] == 0
        assert waits[1] == 0
        assert waits[2] == 60  # Waits for first to finish


# =============================================================================
# Fixed-Wing Slot Scheduling Tests
# =============================================================================


class TestFixedWingSlotScheduling:
    """Test fixed-wing slot scheduling logic."""

    def test_find_next_slot_simple(self):
        """Find next slot in simple 1/day pattern."""
        from faer.core.scenario import FixedWingConfig

        config = FixedWingConfig(
            enabled=True,
            slots_per_segment=[1, 0],  # 1 AM, 0 PM
            departure_hour_in_segment=2,  # 02:00 departure
        )

        # At time 0 (midnight), next slot is at hour 2 = 120 mins
        # Segment 0 starts at 0, departure at 0 + 2*60 = 120
        segment_duration = 12 * 60  # 720 mins
        current_time = 0

        # Find next slot (manual calculation for now)
        # Current segment 0, slots_per_segment[0] = 1, departure at 120
        expected_next_slot = 120  # 02:00

        # This will be tested against the actual function once implemented
        assert config.slots_per_segment[0] == 1
        assert config.departure_hour_in_segment == 2

    def test_slot_pattern_repeats(self):
        """Slot pattern repeats after configured segments."""
        from faer.core.scenario import FixedWingConfig

        config = FixedWingConfig(
            enabled=True,
            slots_per_segment=[1, 0, 2, 1],  # 4 segments = 2 days
        )

        # Pattern should repeat after day 2
        pattern_length = len(config.slots_per_segment)
        assert pattern_length == 4

        # Segment 4 (day 3 AM) should mirror segment 0
        segment_4_index = 4 % pattern_length
        assert segment_4_index == 0
        assert config.slots_per_segment[segment_4_index] == 1


# =============================================================================
# Discharge Pathway Selection Tests
# =============================================================================


class TestDischargePathwaySelection:
    """Test discharge pathway determination logic."""

    def test_p1_can_get_aeromed(self):
        """P1 patients can be selected for aeromed."""
        from faer.core.entities import Priority
        from faer.model.patient import Patient, Acuity

        patient = Patient(
            id=1,
            arrival_time=0,
            acuity=Acuity.RESUS,
            priority=Priority.P1_IMMEDIATE,
        )

        # With 100% probability, P1 should always get aeromed
        # This tests the eligibility, actual selection uses RNG
        assert patient.priority == Priority.P1_IMMEDIATE
        assert patient.acuity == Acuity.RESUS

    def test_p4_ineligible_for_aeromed(self):
        """P4 patients are not eligible for aeromed."""
        from faer.core.entities import Priority
        from faer.model.patient import Patient, Acuity

        patient = Patient(
            id=1,
            arrival_time=0,
            acuity=Acuity.MINORS,
            priority=Priority.P4_STANDARD,
        )

        # P4 patients should never get aeromed (per planning doc: P1 only)
        assert patient.priority == Priority.P4_STANDARD
        # Actual eligibility check will be in determine_discharge_pathway()


# =============================================================================
# Missed Slot Cascade Tests
# =============================================================================


class TestMissedSlotCascade:
    """Test missed slot blocking behaviour."""

    def test_missed_slot_holds_ward_bed(self):
        """Patient missing slot continues to hold ward bed."""
        env = simpy.Environment()
        ward_beds = simpy.Resource(env, capacity=1)

        events = []

        def aeromed_patient(env, patient_id, ward_beds, slot_delay):
            """Patient who misses slot and waits."""
            # Acquire ward bed
            with ward_beds.request() as req:
                yield req
                events.append((patient_id, 'ward_acquired', env.now))

                # Wait for slot (simulating missed slot)
                yield env.timeout(slot_delay)

                events.append((patient_id, 'departed', env.now))
            # Ward bed released on departure

        def waiting_patient(env, patient_id, ward_beds):
            """Patient waiting for ward bed."""
            arrival = env.now
            with ward_beds.request() as req:
                yield req
                wait = env.now - arrival
                events.append((patient_id, 'ward_acquired', env.now, wait))

        # Aeromed patient holds bed for 200 mins (missed slot wait)
        env.process(aeromed_patient(env, 0, ward_beds, 200))
        # Second patient arrives and must wait
        env.process(waiting_patient(env, 1, ward_beds))

        env.run(until=500)

        # Patient 1 should wait until patient 0 departs at t=200
        p1_event = [e for e in events if e[0] == 1 and e[1] == 'ward_acquired'][0]
        assert p1_event[2] == 200  # Acquired at time 200
        assert p1_event[3] == 200  # Waited 200 mins


# =============================================================================
# Operating Hours Tests
# =============================================================================


class TestHEMSOperatingHours:
    """Test HEMS operating hours constraints."""

    def test_hems_within_operating_hours(self):
        """HEMS operates within configured hours."""
        from faer.core.scenario import HEMSConfig

        config = HEMSConfig(
            operating_start_hour=7,
            operating_end_hour=21,
        )

        # Check various hours
        def within_hours(hour):
            return config.operating_start_hour <= hour < config.operating_end_hour

        assert within_hours(7) is True   # Start hour OK
        assert within_hours(12) is True  # Midday OK
        assert within_hours(20) is True  # Before end OK
        assert within_hours(21) is False  # End hour NOT OK
        assert within_hours(6) is False   # Before start NOT OK
        assert within_hours(23) is False  # Night NOT OK

    def test_hems_overnight_wait(self):
        """Patient arriving after hours waits until morning."""
        from faer.core.scenario import HEMSConfig

        config = HEMSConfig(
            operating_start_hour=7,
            operating_end_hour=21,
        )

        # Patient ready at 22:00 (hour 22)
        current_hour = 22

        # Should wait until 07:00 next day
        # Hours to wait: (24 - 22) + 7 = 9 hours = 540 mins
        hours_to_wait = (24 - current_hour) + config.operating_start_hour
        assert hours_to_wait == 9
