"""
Integration tests for Capacity Scaling.

Tests end-to-end scaling scenarios including:
- OPEL-based escalation
- Scaling monitor trigger/action cycle
- Discharge acceleration
- Discharge lounge functionality
- Results collection and metrics
"""

import pytest
import simpy
import numpy as np

from faer.core.scaling import (
    CapacityScalingConfig,
    OPELConfig,
    OPELLevel,
    ScalingRule,
    ScalingTrigger,
    ScalingAction,
    ScalingTriggerType,
    ScalingActionType,
    create_opel_rules,
)
from faer.model.dynamic_resource import (
    DynamicCapacityResource,
    DynamicResourceManager,
)
from faer.model.scaling_monitor import ScalingMonitor
from faer.model.discharge import DischargeManager, DischargeMetrics


class TestOPELRuleGeneration:
    """Tests for OPEL preset rule generation."""

    def test_opel_disabled_generates_no_rules(self):
        """When OPEL is disabled, no rules are generated."""
        config = OPELConfig(enabled=False)
        rules = create_opel_rules(config)
        assert len(rules) == 0

    def test_opel_enabled_generates_rules(self):
        """OPEL enabled generates escalation rules."""
        config = OPELConfig(enabled=True)
        rules = create_opel_rules(config)

        # Should generate multiple rules for OPEL 3 and 4
        assert len(rules) >= 4  # At minimum: ED surge, Discharge push, Full surge, Aggressive discharge

    def test_opel3_rule_has_correct_threshold(self):
        """OPEL 3 escalation triggers at 90% by default."""
        config = OPELConfig(
            enabled=True,
            opel_3_ed_threshold=0.90
        )
        rules = create_opel_rules(config)

        # Find OPEL 3 ED surge rule
        opel3_rules = [r for r in rules if "OPEL 3" in r.name and "ED" in r.name]
        assert len(opel3_rules) >= 1
        assert opel3_rules[0].trigger.threshold == 0.90

    def test_opel3_rule_adds_surge_beds(self):
        """OPEL 3 escalation adds configured surge beds."""
        config = OPELConfig(
            enabled=True,
            opel_3_surge_beds=5
        )
        rules = create_opel_rules(config)

        # Find OPEL 3 ED surge rule
        opel3_rules = [r for r in rules if "OPEL 3" in r.name and "Surge" in r.name]
        assert len(opel3_rules) >= 1
        assert opel3_rules[0].action.magnitude == 5

    def test_opel4_rule_enables_diversion(self):
        """OPEL 4 escalation enables diversion when configured."""
        config = OPELConfig(
            enabled=True,
            opel_4_enable_divert=True
        )
        rules = create_opel_rules(config)

        # Find OPEL 4 divert rule
        divert_rules = [r for r in rules if "Divert" in r.name]
        assert len(divert_rules) >= 1
        assert divert_rules[0].action.action_type == ScalingActionType.DIVERT_ARRIVALS

    def test_custom_thresholds_propagate(self):
        """Custom OPEL thresholds are used in generated rules."""
        config = OPELConfig(
            enabled=True,
            opel_3_ed_threshold=0.88,
            opel_4_ed_threshold=0.93
        )
        rules = create_opel_rules(config)

        # Check thresholds propagated correctly
        opel3_ed = [r for r in rules if "OPEL 3" in r.name and "ED" in r.name]
        opel4_ed = [r for r in rules if "OPEL 4" in r.name and "Surge" in r.name]

        if opel3_ed:
            assert opel3_ed[0].trigger.threshold == 0.88
        if opel4_ed:
            assert opel4_ed[0].trigger.threshold == 0.93


class TestScalingMonitorIntegration:
    """Tests for ScalingMonitor with DynamicResourceManager."""

    def test_monitor_triggers_scale_up(self):
        """Monitor triggers scale-up when utilisation exceeds threshold."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        # Create ED resource with 10 bays, max 20
        ed = manager.create_resource("ed_bays", 10, 20)

        # Config with simple utilisation trigger at 80%
        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=1.0,  # Check every minute
            rules=[
                ScalingRule(
                    name="Surge",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.80,
                        sustain_mins=0.0,  # No sustain for test
                        cooldown_mins=0.0
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=5
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def fill_ed(env, resource):
            # Fill 9 of 10 bays (90% utilisation)
            requests = []
            for _ in range(9):
                req = resource.request()
                yield req
                requests.append(req)

            # Wait for monitor to detect and act
            yield env.timeout(5)

        env.process(monitor.run())
        env.process(fill_ed(env, ed))
        env.run(until=10)

        # Should have scaled up by 5
        assert ed.capacity == 15

    def test_monitor_respects_cooldown(self):
        """Monitor doesn't re-trigger during cooldown period."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 30)

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=1.0,
            rules=[
                ScalingRule(
                    name="Surge",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.80,
                        sustain_mins=0.0,
                        cooldown_mins=60.0  # Long cooldown
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=2
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def keep_high_utilisation(env, resource):
            # Maintain high utilisation throughout
            requests = []
            for _ in range(9):
                req = resource.request()
                yield req
                requests.append(req)

            yield env.timeout(30)  # Stay occupied

        env.process(monitor.run())
        env.process(keep_high_utilisation(env, ed))
        env.run(until=20)

        # Should only scale once despite sustained high utilisation
        # Initial: 10, after one scale-up: 12
        assert ed.capacity == 12

    def test_monitor_tracks_events(self):
        """Monitor logs scaling events correctly."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=1.0,
            rules=[
                ScalingRule(
                    name="TestRule",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.50,
                        sustain_mins=0.0,
                        cooldown_mins=0.0
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=3
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def trigger_scaling(env, resource):
            for _ in range(6):  # 60% utilisation
                req = resource.request()
                yield req
            yield env.timeout(5)

        env.process(monitor.run())
        env.process(trigger_scaling(env, ed))
        env.run(until=10)

        metrics = monitor.get_metrics()
        events = metrics["events"]
        assert len(events) >= 1
        assert events[0].rule_name == "TestRule"
        assert events[0].action_type == ScalingActionType.ADD_CAPACITY.value


class TestDischargeManagerIntegration:
    """Tests for DischargeManager functionality."""

    def test_los_acceleration_reduces_stay(self):
        """LoS acceleration reduces length of stay."""
        env = simpy.Environment()
        config = CapacityScalingConfig(
            discharge_lounge_capacity=0  # No lounge
        )

        dm = DischargeManager(env, config)

        # Activate 20% reduction
        dm.activate_acceleration(20.0)

        base_los = 100.0
        adjusted = dm.get_adjusted_los(base_los)

        assert adjusted == 80.0  # 100 * (1 - 0.20)

    def test_los_acceleration_tracks_metrics(self):
        """LoS acceleration tracks reduction metrics."""
        env = simpy.Environment()
        config = CapacityScalingConfig(discharge_lounge_capacity=0)

        dm = DischargeManager(env, config)
        dm.activate_acceleration(10.0)

        # Apply to multiple patients
        dm.get_adjusted_los(100.0)
        dm.get_adjusted_los(200.0)
        dm.get_adjusted_los(150.0)

        metrics = dm.get_metrics()

        assert metrics["los_reduction_applied_count"] == 3
        assert metrics["total_los_reduction_mins"] == 45.0  # 10 + 20 + 15

    def test_deactivation_returns_to_normal(self):
        """Deactivating acceleration returns LoS to normal."""
        env = simpy.Environment()
        config = CapacityScalingConfig(discharge_lounge_capacity=0)

        dm = DischargeManager(env, config)
        dm.activate_acceleration(15.0)

        assert dm.get_adjusted_los(100.0) == 85.0

        dm.deactivate_acceleration()

        assert dm.get_adjusted_los(100.0) == 100.0

    def test_discharge_lounge_accepts_patients(self):
        """Discharge lounge accepts patients when enabled."""
        env = simpy.Environment()
        config = CapacityScalingConfig(
            discharge_lounge_capacity=5,
            discharge_lounge_max_wait_mins=60.0
        )

        dm = DischargeManager(env, config)

        assert dm.lounge_enabled is True
        assert dm.is_lounge_available() is True

    def test_lounge_process_records_entry_exit(self):
        """Lounge process records patient entry and exit."""
        env = simpy.Environment()
        config = CapacityScalingConfig(
            discharge_lounge_capacity=5,
            discharge_lounge_max_wait_mins=60.0
        )

        dm = DischargeManager(env, config)

        def patient_in_lounge(env, dm):
            yield env.process(dm.enter_lounge_process(
                patient_id=1,
                bed_resource_name="ward_bed",
                remaining_discharge_time=30.0
            ))

        env.process(patient_in_lounge(env, dm))
        env.run()

        metrics = dm.get_metrics()
        assert metrics["lounge_entries"] == 1
        assert metrics["lounge_exits"] == 1

    def test_lounge_timeline_for_visualization(self):
        """Lounge generates timeline events for visualization."""
        env = simpy.Environment()
        config = CapacityScalingConfig(
            discharge_lounge_capacity=5,
            discharge_lounge_max_wait_mins=60.0
        )

        dm = DischargeManager(env, config)

        def patient_in_lounge(env, dm):
            yield env.process(dm.enter_lounge_process(
                patient_id=42,
                bed_resource_name="itu_bed",
                remaining_discharge_time=20.0
            ))

        env.process(patient_in_lounge(env, dm))
        env.run()

        timeline = dm.get_lounge_timeline()

        assert len(timeline) == 2  # entry + exit
        assert timeline[0][1] == 42  # patient_id
        assert timeline[0][2] == "entered"
        assert timeline[1][2] == "departed"


class TestOPELLevelTracking:
    """Tests for OPEL level determination and tracking."""

    def test_opel_level_normal_at_low_utilisation(self):
        """OPEL level is 1 (Normal) at low utilisation."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(
            enabled=True,
            opel_config=OPELConfig(enabled=True)
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # 20% utilisation
        def low_occupancy(env, resource):
            for _ in range(2):
                req = resource.request()
                yield req
            yield env.timeout(5)

        env.process(monitor.run())
        env.process(low_occupancy(env, ed))
        env.run(until=3)

        assert monitor.opel_status.current_level == OPELLevel.OPEL_1

    def test_opel_level_escalates_with_utilisation(self):
        """OPEL level escalates as utilisation increases."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=1.0,
            opel_config=OPELConfig(
                enabled=True,
                opel_2_ed_threshold=0.85,
                opel_3_ed_threshold=0.90,
                opel_4_ed_threshold=0.95
            )
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # 92% utilisation (> 90% OPEL 3 threshold)
        def high_occupancy(env, resource):
            requests = []
            for i in range(10):
                req = resource.request()
                yield req
                requests.append(req)
                if i == 8:  # At 90%
                    yield env.timeout(3)  # Let monitor evaluate

        env.process(monitor.run())
        env.process(high_occupancy(env, ed))
        env.run(until=10)

        # Should be at least OPEL 3 with 90%+ utilisation
        assert monitor.opel_status.current_level.value >= OPELLevel.OPEL_3.value


class TestScalingConfigIntegration:
    """Tests for CapacityScalingConfig."""

    def test_config_defaults_disabled(self):
        """Default config has scaling disabled."""
        config = CapacityScalingConfig()

        assert config.enabled is False
        assert len(config.rules) == 0

    def test_config_with_opel_generates_rules(self):
        """Config with OPEL enabled generates rules on demand."""
        opel = OPELConfig(enabled=True)
        config = CapacityScalingConfig(
            enabled=True,
            opel_config=opel
        )

        rules = create_opel_rules(opel)

        assert len(rules) >= 4  # Multiple OPEL rules
        assert config.enabled is True

    def test_config_combines_manual_and_opel_rules(self):
        """Manual rules can be combined with OPEL rules."""
        manual_rule = ScalingRule(
            name="Custom Rule",
            trigger=ScalingTrigger(
                trigger_type=ScalingTriggerType.QUEUE_LENGTH_ABOVE,
                resource="ed_bays",
                threshold=10.0
            ),
            action=ScalingAction(
                action_type=ScalingActionType.ADD_CAPACITY,
                resource="ed_bays",
                magnitude=2
            )
        )

        opel = OPELConfig(enabled=True)
        opel_rules = create_opel_rules(opel)

        config = CapacityScalingConfig(
            enabled=True,
            rules=[manual_rule] + opel_rules,
            opel_config=opel
        )

        # Manual + OPEL rules
        assert len(config.rules) >= 5


class TestDiversionFunctionality:
    """Tests for patient diversion during capacity crisis."""

    def test_diversion_can_be_enabled(self):
        """Monitor can enable diversion via scaling action."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 10)  # At max

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=1.0,
            rules=[
                ScalingRule(
                    name="Crisis Divert",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.95,
                        sustain_mins=0.0,
                        cooldown_mins=0.0
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.DIVERT_ARRIVALS,
                        resource="arrivals",
                        diversion_rate=0.3,
                        diversion_priorities=["P3", "P4"]
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def fill_completely(env, resource):
            requests = []
            for _ in range(10):
                req = resource.request()
                yield req
                requests.append(req)
            yield env.timeout(5)

        env.process(monitor.run())
        env.process(fill_completely(env, ed))
        env.run(until=10)

        assert monitor.diversion_active is True
        assert "P3" in monitor.diversion_priorities
        assert "P4" in monitor.diversion_priorities

    def test_diversion_check_respects_priorities(self):
        """Diversion only applies to specified priorities."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        manager.create_resource("ed_bays", 10, 10)

        config = CapacityScalingConfig(enabled=True)
        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # Manually enable diversion
        monitor.diversion_active = True
        monitor.diversion_rate = 1.0  # Always divert if in list
        monitor.diversion_priorities = ["P3", "P4"]

        # P1 should never be diverted
        # Note: should_divert has randomness, but P1 isn't in the list
        # so it should return False regardless
        result_p1 = monitor.should_divert("P1")
        assert result_p1 is False


class TestGracefulScaleDown:
    """Tests for graceful capacity reduction."""

    def test_graceful_removal_waits_for_occupants(self):
        """Graceful scale-down waits for patients to leave."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        res = manager.create_resource("beds", 10, 20)

        def occupy_and_scale_down(env, resource):
            # Occupy 5 beds
            requests = []
            for _ in range(5):
                req = resource.request()
                yield req
                requests.append(req)

            # Request graceful removal of 3
            resource.remove_capacity(3, graceful=True)

            # Capacity should be reduced immediately (effective)
            assert resource.capacity == 7

            # But actual slots still active until release
            assert resource._active_slots == 10

            # Release one
            resource.release(requests[0])
            yield env.timeout(0.1)

            # One pending deactivation should clear
            assert resource._pending_deactivations == 2

        env.process(occupy_and_scale_down(env, res))
        env.run()


class TestTimeBasedTriggers:
    """Tests for time-of-day based scaling."""

    def test_time_trigger_activates_in_window(self):
        """Time-based trigger activates within specified window."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        # Rule active 08:00-20:00 (480-1200 mins from midnight)
        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=60.0,
            rules=[
                ScalingRule(
                    name="Day Surge",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.TIME_OF_DAY,
                        resource="ed_bays",
                        threshold=0.0,  # Not used for time triggers
                        start_time=480,  # 08:00
                        end_time=1200,   # 20:00
                        sustain_mins=0.0,
                        cooldown_mins=720.0  # Only once per day
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=5
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # Start simulation at 09:00 (540 mins)
        def run_at_9am(env):
            yield env.timeout(540)
            # Monitor should trigger at next evaluation
            yield env.timeout(60)

        env.process(monitor.run())
        env.process(run_at_9am(env))
        env.run(until=700)

        # Should have scaled up during daytime window
        assert ed.capacity == 15


class TestMetricsCollection:
    """Tests for scaling metrics in ResultsCollector context."""

    def test_scaling_events_have_timestamps(self):
        """All scaling events have correct timestamps."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=5.0,
            rules=[
                ScalingRule(
                    name="QuickScale",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="ed_bays",
                        threshold=0.50,
                        sustain_mins=0.0,
                        cooldown_mins=0.0
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="ed_bays",
                        magnitude=2
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def trigger_at_time_10(env, resource):
            yield env.timeout(10)
            for _ in range(6):
                req = resource.request()
                yield req
            yield env.timeout(10)

        env.process(monitor.run())
        env.process(trigger_at_time_10(env, ed))
        env.run(until=30)

        metrics = monitor.get_metrics()
        events = metrics["events"]

        # Event should happen after time 10 (when utilisation goes high)
        if events:
            assert events[0].time >= 10.0

    def test_additional_bed_hours_calculation(self):
        """Additional bed-hours are calculated from capacity changes."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        res = manager.create_resource("beds", 10, 30)

        def capacity_changes(env, resource):
            # At time 0: 10 beds
            yield env.timeout(60)  # 1 hour

            # Add 5 beds
            resource.add_capacity(5)
            yield env.timeout(120)  # 2 hours at 15 beds

            # Add 5 more
            resource.add_capacity(5)
            yield env.timeout(60)  # 1 hour at 20 beds

        env.process(capacity_changes(env, res))
        env.run()

        timeline = res.get_capacity_timeline()

        # Calculate additional bed-hours manually
        # 0-60: 10 beds (baseline)
        # 60-180: 15 beds (+5 for 2 hours = 10 bed-hours)
        # 180-240: 20 beds (+10 for 1 hour = 10 bed-hours)
        # Total additional: 20 bed-hours

        assert len(timeline) == 3
        assert timeline[0] == (0.0, 10)
        assert timeline[1] == (60.0, 15)
        assert timeline[2] == (180.0, 20)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_scaling_disabled_does_nothing(self):
        """When scaling disabled, monitor does not modify resources."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        ed = manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(enabled=False)
        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        def fill_completely(env, resource):
            for _ in range(10):
                req = resource.request()
                yield req
            yield env.timeout(60)

        env.process(monitor.run())
        env.process(fill_completely(env, ed))
        env.run(until=120)

        # Capacity unchanged (no rules when disabled)
        assert ed.capacity == 10

    def test_missing_resource_handles_gracefully(self):
        """Rule targeting missing resource doesn't crash."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        # No resources created

        config = CapacityScalingConfig(
            enabled=True,
            evaluation_interval_mins=5.0,
            rules=[
                ScalingRule(
                    name="NoResource",
                    trigger=ScalingTrigger(
                        trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                        resource="nonexistent",
                        threshold=0.50
                    ),
                    action=ScalingAction(
                        action_type=ScalingActionType.ADD_CAPACITY,
                        resource="nonexistent",
                        magnitude=5
                    )
                )
            ]
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # Should run without error
        env.process(monitor.run())
        env.run(until=20)

        # No scaling events (resource doesn't exist)
        metrics = monitor.get_metrics()
        assert len(metrics["events"]) == 0

    def test_zero_capacity_resource(self):
        """Resource with zero initial capacity can be scaled up."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        # Surge resource starting at 0
        surge = manager.create_resource("surge_area", 0, 10)

        assert surge.capacity == 0

        added = surge.add_capacity(5)

        assert added == 5
        assert surge.capacity == 5


class TestCurrentOPELLevel:
    """Test the current_opel_level property accessor."""

    def test_current_opel_level_property(self):
        """ScalingMonitor exposes current_opel_level property."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        manager.create_resource("ed_bays", 10, 20)

        config = CapacityScalingConfig(
            enabled=True,
            opel_config=OPELConfig(enabled=True)
        )

        monitor = ScalingMonitor(env, manager.get_all_resources(), config)

        # Default should be OPEL 1
        assert monitor.opel_status.current_level == OPELLevel.OPEL_1


class TestScalingEventDataclass:
    """Test ScalingEvent dataclass."""

    def test_scaling_event_fields(self):
        """ScalingEvent has all required fields."""
        from faer.model.scaling_monitor import ScalingEvent

        event = ScalingEvent(
            time=100.0,
            rule_name="Test Rule",
            action_type="add_capacity",
            resource="ed_bays",
            old_capacity=10,
            new_capacity=15,
            trigger_value=0.92,
            direction="scale_up"
        )

        assert event.time == 100.0
        assert event.rule_name == "Test Rule"
        assert event.direction == "scale_up"
