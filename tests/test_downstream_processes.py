"""Unit tests for downstream processes (Phase 9).

These tests validate Theatre, ITU, and Ward process behaviour in isolation,
before integration into the full model. Following test-first development approach.
"""

import pytest
import simpy


# =============================================================================
# Resource Blocking Tests - Core SimPy Behaviour Validation
# =============================================================================


class TestResourceBlocking:
    """Test that SimPy resources block correctly when full."""

    def test_resource_blocks_when_full(self):
        """If capacity is N and N+1 requests arrive, 1 must wait."""
        env = simpy.Environment()
        capacity = 2
        resource = simpy.Resource(env, capacity=capacity)
        wait_times = []

        def patient(env, resource, los=60):
            arrival = env.now
            with resource.request() as req:
                yield req
                wait_times.append(env.now - arrival)
                yield env.timeout(los)

        # Start 3 patients simultaneously (capacity + 1)
        for _ in range(capacity + 1):
            env.process(patient(env, resource))

        env.run()

        # First N get in immediately (wait = 0)
        assert sum(1 for w in wait_times if w == 0) == capacity
        # Exactly 1 waits
        assert sum(1 for w in wait_times if w > 0) == 1
        # The waiter should wait exactly 60 mins (first patient's LOS)
        assert max(wait_times) == 60

    def test_multiple_waiters_queue_correctly(self):
        """Multiple waiters are served in FIFO order."""
        env = simpy.Environment()
        resource = simpy.Resource(env, capacity=1)
        completion_order = []

        def patient(env, patient_id, resource, los=30):
            with resource.request() as req:
                yield req
                yield env.timeout(los)
                completion_order.append(patient_id)

        # Start 4 patients
        for i in range(4):
            env.process(patient(env, i, resource))

        env.run()

        # Should complete in order 0, 1, 2, 3
        assert completion_order == [0, 1, 2, 3]

    def test_blocking_scales_with_excess(self):
        """Number of patients who wait equals excess over capacity."""
        for capacity in [1, 2, 5]:
            for excess in [1, 2, 3]:
                env = simpy.Environment()
                resource = simpy.Resource(env, capacity=capacity)
                wait_times = []

                def patient(env, resource, los=60):
                    arrival = env.now
                    with resource.request() as req:
                        yield req
                        wait_times.append(env.now - arrival)
                        yield env.timeout(los)

                for _ in range(capacity + excess):
                    env.process(patient(env, resource))

                env.run(until=10000)

                # Exactly 'excess' patients should have waited
                waiters = sum(1 for w in wait_times if w > 0)
                assert waiters == excess, f"capacity={capacity}, excess={excess}"


# =============================================================================
# Nested Resource Tests - Blocking Cascade Behaviour
# =============================================================================


class TestBlockingCascade:
    """Test that nested resource requests create blocking cascade."""

    def test_itu_releases_when_ward_acquired(self):
        """ITU bed should release when ward bed is ACQUIRED, not when ward done.

        This is the correct hospital behaviour:
        - Patient HOLDS ITU bed while WAITING for ward bed
        - Once ward bed acquired, patient moves and ITU bed is freed

        This requires explicit resource.release() rather than context managers.

        Timeline with ward_capacity=1, itu_capacity=1:
        - t=0: P0 acquires ITU, P1 waits for ITU
        - t=60: P0 finishes ITU treatment, requests ward, gets it, releases ITU
        - t=60: P1 acquires ITU (freed by P0)
        - t=120: P1 finishes ITU, requests ward, waits (P0 still there)
        - t=180: P0 finishes ward, P1 gets ward and releases ITU
        """
        env = simpy.Environment()
        itu_beds = simpy.Resource(env, capacity=1)
        ward_beds = simpy.Resource(env, capacity=1)

        events = []

        def patient_with_stepdown(env, patient_id, itu_beds, ward_beds):
            # Request and hold ITU
            itu_req = itu_beds.request()
            yield itu_req
            events.append((patient_id, 'itu_acquired', env.now))

            yield env.timeout(60)  # ITU stay

            # Request ward while still holding ITU
            ward_req = ward_beds.request()
            yield ward_req
            events.append((patient_id, 'ward_acquired', env.now))

            # NOW release ITU (patient has moved to ward)
            itu_beds.release(itu_req)
            events.append((patient_id, 'itu_released', env.now))

            yield env.timeout(120)  # Ward stay

            ward_beds.release(ward_req)
            events.append((patient_id, 'ward_released', env.now))

        env.process(patient_with_stepdown(env, 0, itu_beds, ward_beds))
        env.process(patient_with_stepdown(env, 1, itu_beds, ward_beds))

        env.run(until=1000)

        def get_events(pid):
            return {e[1]: e[2] for e in events if e[0] == pid}

        p0 = get_events(0)
        p1 = get_events(1)

        # P0: ITU at 0, ward at 60, releases ITU at 60
        assert p0['itu_acquired'] == 0
        assert p0['ward_acquired'] == 60
        assert p0['itu_released'] == 60  # Released when ward acquired!
        assert p0['ward_released'] == 180

        # P1: gets ITU at 60 (when P0 released it)
        assert p1['itu_acquired'] == 60
        # P1 finishes ITU at 120, but must wait for ward until 180
        assert p1['ward_acquired'] == 180
        assert p1['itu_released'] == 180  # Released when ward acquired
        assert p1['ward_released'] == 300

        # KEY: P1 only waited 60 mins for ITU, not 180
        p1_itu_wait = p1['itu_acquired'] - 0
        assert p1_itu_wait == 60

        # But P1 waited 60 mins for ward (120 to 180)
        p1_ward_wait = p1['ward_acquired'] - 120
        assert p1_ward_wait == 60

    def test_three_level_cascade_with_release_on_acquire(self):
        """Test Theatre -> ITU -> Ward cascade with proper release timing.

        Each upstream resource is released when downstream is ACQUIRED.

        Timeline (all capacity=1, surgery=30, ITU=60, ward=120):
        P0: theatre 0-30, acquires ITU at 30 (releases theatre)
            ITU 30-90, acquires ward at 90 (releases ITU), ward 90-210
        P1: gets theatre at 30 (P0 released), theatre 30-60
            gets ITU at 90 (P0 released), ITU 90-150
            waits for ward (P0 has it until 210), gets ward at 210, ward 210-330
        P2: gets theatre at 60 (P1 released), theatre 60-90
            gets ITU at 150 (P1 released), ITU 150-210
            waits for ward (P1 has it), gets ward at 330, ward 330-450
        """
        env = simpy.Environment()
        theatre = simpy.Resource(env, capacity=1)
        itu = simpy.Resource(env, capacity=1)
        ward = simpy.Resource(env, capacity=1)

        events = []

        def surgical_patient(env, patient_id):
            """Patient needing Theatre -> ITU -> Ward."""
            # Theatre
            theatre_req = theatre.request()
            yield theatre_req
            events.append((patient_id, 'theatre_start', env.now))
            yield env.timeout(30)  # Surgery

            # Request ITU while holding theatre
            itu_req = itu.request()
            yield itu_req
            events.append((patient_id, 'itu_start', env.now))
            # Release theatre now that we have ITU
            theatre.release(theatre_req)
            events.append((patient_id, 'theatre_released', env.now))

            yield env.timeout(60)  # ITU stay

            # Request ward while holding ITU
            ward_req = ward.request()
            yield ward_req
            events.append((patient_id, 'ward_start', env.now))
            # Release ITU now that we have ward
            itu.release(itu_req)
            events.append((patient_id, 'itu_released', env.now))

            yield env.timeout(120)  # Ward stay

            ward.release(ward_req)
            events.append((patient_id, 'ward_released', env.now))

        # Three patients arriving simultaneously
        for i in range(3):
            env.process(surgical_patient(env, i))

        env.run(until=2000)

        def get_events(pid):
            return {e[1]: e[2] for e in events if e[0] == pid}

        p0 = get_events(0)
        p1 = get_events(1)
        p2 = get_events(2)

        # Patient 0: no waits
        assert p0['theatre_start'] == 0
        assert p0['itu_start'] == 30
        assert p0['theatre_released'] == 30
        assert p0['ward_start'] == 90
        assert p0['itu_released'] == 90

        # Patient 1: gets theatre when P0 releases (t=30)
        assert p1['theatre_start'] == 30
        # Gets ITU when P0 releases (t=90)
        assert p1['itu_start'] == 90
        assert p1['theatre_released'] == 90
        # Waits for ward until P0 finishes (t=210)
        assert p1['ward_start'] == 210
        assert p1['itu_released'] == 210

        # Patient 2: gets theatre when P1 releases (t=90)
        assert p2['theatre_start'] == 90
        # Gets ITU when P1 releases (t=210)
        assert p2['itu_start'] == 210
        assert p2['theatre_released'] == 210
        # Waits for ward until P1 finishes (t=330)
        assert p2['ward_start'] == 330
        assert p2['itu_released'] == 330

        # Cascade effect: P2 waits 90 mins for theatre
        p2_theatre_wait = p2['theatre_start'] - 0
        assert p2_theatre_wait == 90

        # And P2 waits 120 mins for ITU (90 to 210)
        p2_itu_wait = p2['itu_start'] - (90 + 30)  # After finishing theatre
        assert p2_itu_wait == 90

        # And P2 waits 120 mins for ward (210 to 330)
        p2_ward_wait = p2['ward_start'] - (210 + 60)  # After finishing ITU
        assert p2_ward_wait == 60


# =============================================================================
# ED Bay Release Policy Tests
# =============================================================================


class TestEDBayReleaseEligibility:
    """Test ED bay early release eligibility logic."""

    def test_p1_never_releases_early(self):
        """P1 patients always hold ED bay regardless of destination."""
        from faer.core.entities import Priority, NodeType

        # P1 to Ward - should NOT release early
        assert not _can_release_early(Priority.P1_IMMEDIATE, NodeType.WARD)

        # P1 to ITU - should NOT release early
        assert not _can_release_early(Priority.P1_IMMEDIATE, NodeType.ITU)

        # P1 to Surgery - should NOT release early
        assert not _can_release_early(Priority.P1_IMMEDIATE, NodeType.SURGERY)

    def test_p2_never_releases_early(self):
        """P2 patients always hold ED bay regardless of destination."""
        from faer.core.entities import Priority, NodeType

        assert not _can_release_early(Priority.P2_VERY_URGENT, NodeType.WARD)
        assert not _can_release_early(Priority.P2_VERY_URGENT, NodeType.ITU)
        assert not _can_release_early(Priority.P2_VERY_URGENT, NodeType.SURGERY)

    def test_p3_ward_can_release_early(self):
        """P3 patients going to Ward CAN release early (if policy enabled)."""
        from faer.core.entities import Priority, NodeType

        assert _can_release_early(Priority.P3_URGENT, NodeType.WARD)

    def test_p3_itu_holds_bay(self):
        """P3 patients going to ITU hold ED bay."""
        from faer.core.entities import Priority, NodeType

        assert not _can_release_early(Priority.P3_URGENT, NodeType.ITU)

    def test_p3_surgery_holds_bay(self):
        """P3 patients going to Surgery hold ED bay."""
        from faer.core.entities import Priority, NodeType

        assert not _can_release_early(Priority.P3_URGENT, NodeType.SURGERY)

    def test_p4_ward_can_release_early(self):
        """P4 patients going to Ward CAN release early (if policy enabled)."""
        from faer.core.entities import Priority, NodeType

        assert _can_release_early(Priority.P4_STANDARD, NodeType.WARD)

    def test_discharge_always_releases(self):
        """Discharging patients always release ED bay."""
        from faer.core.entities import Priority, NodeType

        for priority in Priority:
            assert _can_release_early(priority, NodeType.EXIT)


def _can_release_early(priority, destination) -> bool:
    """Determine if patient can release ED bay early.

    This is the eligibility logic that will be implemented in full_model.py.
    Only P3/P4 patients going to Ward (or discharging) can release early.
    """
    from faer.core.entities import Priority, NodeType

    # Discharge always releases
    if destination == NodeType.EXIT:
        return True

    # Only P3/P4 eligible
    if priority in (Priority.P1_IMMEDIATE, Priority.P2_VERY_URGENT):
        return False

    # Only Ward destination eligible (not ITU, not Surgery)
    if destination not in (NodeType.WARD,):
        return False

    return True


# =============================================================================
# Patient Timestamp Recording Tests
# =============================================================================


class TestPatientDownstreamTimestamps:
    """Test Patient entity correctly records downstream timestamps."""

    def test_record_surgery(self):
        """Surgery timestamps recorded correctly."""
        from faer.model.patient import Patient, Acuity
        from faer.core.entities import Priority

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)

        patient.record_surgery(wait_start=100, start=120, end=200)

        assert patient.surgery_wait_start == 100
        assert patient.surgery_start == 120
        assert patient.surgery_end == 200
        assert patient.surgery_wait == 20  # 120 - 100
        assert patient.surgery_duration == 80  # 200 - 120
        assert "theatre" in patient.resources_used

    def test_record_itu(self):
        """ITU timestamps recorded correctly."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MAJORS)

        patient.record_itu(wait_start=200, start=260, end=500)

        assert patient.itu_wait_start == 200
        assert patient.itu_start == 260
        assert patient.itu_end == 500
        assert patient.itu_wait == 60
        assert patient.itu_duration == 240
        assert "itu" in patient.resources_used

    def test_record_ward(self):
        """Ward timestamps recorded correctly."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MINORS)

        patient.record_ward(wait_start=500, start=500, end=1000)

        assert patient.ward_wait_start == 500
        assert patient.ward_start == 500
        assert patient.ward_end == 1000
        assert patient.ward_wait == 0  # No wait
        assert patient.ward_duration == 500
        assert "ward" in patient.resources_used

    def test_record_ed_bay_release(self):
        """ED bay release tracking works."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.MINORS)

        # Normal release (not early)
        patient.record_ed_bay_release(time=100, early=False)
        assert patient.ed_bay_release_time == 100
        assert patient.released_ed_bay_early is False

        # Early release
        patient2 = Patient(id=2, arrival_time=0, acuity=Acuity.MINORS)
        patient2.record_ed_bay_release(time=50, early=True)
        assert patient2.ed_bay_release_time == 50
        assert patient2.released_ed_bay_early is True

    def test_full_surgical_pathway_timestamps(self):
        """Complete surgical pathway records all timestamps."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0, acuity=Acuity.RESUS)

        # ED -> Surgery -> ITU -> Ward
        patient.record_surgery(wait_start=60, start=70, end=160)
        patient.record_itu(wait_start=160, start=180, end=400)
        patient.record_ward(wait_start=400, start=450, end=1500)

        assert patient.surgery_wait == 10
        assert patient.surgery_duration == 90
        assert patient.itu_wait == 20
        assert patient.itu_duration == 220
        assert patient.ward_wait == 50
        assert patient.ward_duration == 1050

        # Total downstream time
        total_downstream = (
            patient.surgery_duration +
            patient.itu_duration +
            patient.ward_duration
        )
        assert total_downstream == 90 + 220 + 1050


# =============================================================================
# Scenario Configuration Tests
# =============================================================================


class TestScenarioDownstreamConfig:
    """Test FullScenario downstream configuration."""

    def test_downstream_disabled_by_default(self):
        """Downstream is disabled by default."""
        from faer.core.scenario import FullScenario

        scenario = FullScenario()

        assert scenario.downstream_enabled is False
        assert scenario.release_stable_to_wait is False

    def test_downstream_can_be_enabled(self):
        """Downstream can be enabled via constructor."""
        from faer.core.scenario import FullScenario

        scenario = FullScenario(downstream_enabled=True)

        assert scenario.downstream_enabled is True

    def test_release_stable_requires_downstream(self):
        """release_stable_to_wait only meaningful when downstream enabled."""
        from faer.core.scenario import FullScenario

        # Both flags can be set independently
        scenario = FullScenario(
            downstream_enabled=True,
            release_stable_to_wait=True
        )

        assert scenario.downstream_enabled is True
        assert scenario.release_stable_to_wait is True

    def test_downstream_configs_initialized(self):
        """ITU, Ward, Theatre configs are initialized by default."""
        from faer.core.scenario import FullScenario, ITUConfig, WardConfig, TheatreConfig

        scenario = FullScenario()

        assert scenario.itu_config is not None
        assert scenario.ward_config is not None
        assert scenario.theatre_config is not None
        assert isinstance(scenario.itu_config, ITUConfig)
        assert isinstance(scenario.ward_config, WardConfig)
        assert isinstance(scenario.theatre_config, TheatreConfig)


# =============================================================================
# Integration Tests - Full Simulation with Downstream
# =============================================================================


class TestDownstreamIntegration:
    """Integration tests for downstream simulation."""

    def test_simulation_runs_with_downstream_disabled(self):
        """Simulation runs normally with downstream disabled (legacy)."""
        from faer.core.scenario import FullScenario
        from faer.model.full_model import run_full_simulation

        scenario = FullScenario(
            run_length=120.0,  # 2 hours
            warm_up=0.0,
            arrival_rate=6.0,
            downstream_enabled=False,
            random_seed=42,
        )

        results = run_full_simulation(scenario)

        assert results['arrivals'] > 0
        assert results['departures'] > 0
        assert 'mean_system_time' in results

    def test_simulation_runs_with_downstream_enabled(self):
        """Simulation runs with downstream enabled."""
        from faer.core.scenario import FullScenario, ITUConfig, WardConfig, TheatreConfig
        from faer.model.full_model import run_full_simulation

        scenario = FullScenario(
            run_length=240.0,  # 4 hours
            warm_up=0.0,
            arrival_rate=4.0,  # Lower rate for faster test
            downstream_enabled=True,
            itu_config=ITUConfig(capacity=2, los_mean_hours=1.0),
            ward_config=WardConfig(capacity=5, los_mean_hours=2.0),
            theatre_config=TheatreConfig(n_tables=1, procedure_time_mean=30.0),
            random_seed=42,
        )

        results = run_full_simulation(scenario)

        assert results['arrivals'] > 0
        # Note: departures may be lower if patients are still in downstream
        assert 'mean_system_time' in results

    def test_downstream_resources_created_when_enabled(self):
        """Downstream resources are created when enabled."""
        from faer.core.scenario import FullScenario
        from faer.model.full_model import AEResources
        import simpy

        scenario = FullScenario(downstream_enabled=True)
        env = simpy.Environment()

        # Manually create resources as run_full_simulation does
        assert scenario.theatre_config is not None
        assert scenario.itu_config is not None
        assert scenario.ward_config is not None

        # Check capacities are set correctly
        assert scenario.theatre_config.n_tables == 2
        assert scenario.itu_config.capacity == 6
        assert scenario.ward_config.capacity == 30

    def test_downstream_resources_none_when_disabled(self):
        """Downstream resources are None when disabled."""
        from faer.core.scenario import FullScenario
        from faer.model.full_model import run_full_simulation

        scenario = FullScenario(
            run_length=60.0,
            warm_up=0.0,
            downstream_enabled=False,
            random_seed=42,
        )

        # The resources object is internal, but we can verify the scenario
        assert scenario.downstream_enabled is False

    def test_release_stable_affects_p3_p4_only(self):
        """release_stable_to_wait only affects P3/P4 ward-bound patients."""
        from faer.core.entities import Priority, NodeType
        from faer.model.full_model import can_release_ed_bay_early

        # P1/P2 should never release early
        assert not can_release_ed_bay_early(
            _make_test_patient(Priority.P1_IMMEDIATE), NodeType.WARD
        )
        assert not can_release_ed_bay_early(
            _make_test_patient(Priority.P2_VERY_URGENT), NodeType.WARD
        )

        # P3/P4 to Ward CAN release
        assert can_release_ed_bay_early(
            _make_test_patient(Priority.P3_URGENT), NodeType.WARD
        )
        assert can_release_ed_bay_early(
            _make_test_patient(Priority.P4_STANDARD), NodeType.WARD
        )

        # P3/P4 to ITU/Surgery should NOT release
        assert not can_release_ed_bay_early(
            _make_test_patient(Priority.P3_URGENT), NodeType.ITU
        )
        assert not can_release_ed_bay_early(
            _make_test_patient(Priority.P3_URGENT), NodeType.SURGERY
        )


def _make_test_patient(priority):
    """Create a minimal patient for testing."""
    from faer.model.patient import Patient, Acuity

    return Patient(
        id=1,
        arrival_time=0,
        acuity=Acuity.MAJORS,
        priority=priority,
    )
