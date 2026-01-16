"""
Tests for DynamicCapacityResource.

Verifies that capacity can be changed at runtime and that
the resource behaves correctly with additions and removals.
"""

import pytest
import simpy

from faer.model.dynamic_resource import (
    DynamicCapacityResource,
    DynamicResourceManager,
    CapacityChangeEvent,
)


class TestDynamicCapacityResource:
    """Tests for DynamicCapacityResource class."""

    def test_initial_capacity(self):
        """Resource starts with specified initial capacity."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        assert res.capacity == 5
        assert res.count == 0
        assert res.max_capacity == 20

    def test_add_capacity(self):
        """Capacity can be increased."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        added = res.add_capacity(3)

        assert added == 3
        assert res.capacity == 8

    def test_add_capacity_respects_max(self):
        """Cannot exceed maximum capacity."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 18, 20)

        added = res.add_capacity(5)

        assert added == 2
        assert res.capacity == 20

    def test_add_capacity_at_max_returns_zero(self):
        """Adding capacity when already at max returns zero."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 20, 20)

        added = res.add_capacity(5)

        assert added == 0
        assert res.capacity == 20

    def test_remove_capacity_graceful(self):
        """Graceful removal marks slots for deactivation."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 10, 20)

        removed = res.remove_capacity(3, graceful=True)

        assert removed == 3
        # Effective capacity reduced immediately
        assert res.capacity == 7

    def test_remove_capacity_immediate_empty_slots(self):
        """Immediate removal only removes empty slots."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 10, 20)

        # No occupants, so all slots can be removed immediately
        removed = res.remove_capacity(3, graceful=False)

        assert removed == 3
        assert res.capacity == 7

    def test_capacity_log_initial_entry(self):
        """Capacity log contains initial entry."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        assert len(res.capacity_log) == 1
        assert res.capacity_log[0].new_capacity == 5
        assert res.capacity_log[0].reason == "initial"

    def test_capacity_log_records_changes(self):
        """Capacity changes are logged."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        res.add_capacity(3, reason="surge")
        res.remove_capacity(2, graceful=True, reason="de-escalate")

        assert len(res.capacity_log) == 3
        assert res.capacity_log[1].new_capacity == 8
        assert res.capacity_log[1].reason == "surge"

    def test_request_basic(self):
        """Basic request/release cycle works."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        def process(env, res):
            with res.request() as req:
                yield req
                assert res.count == 1
                yield env.timeout(10)
            assert res.count == 0

        env.process(process(env, res))
        env.run()

    def test_priority_resource(self):
        """Priority resource can be created and used."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 1, 5, is_priority=True)
        served_order = []

        def process(env, res, name, priority):
            with res.request(priority=priority) as req:
                yield req
                served_order.append(name)
                yield env.timeout(10)

        # Note: With max_capacity=5, all can be served immediately
        # This test just verifies priority resource works without error
        env.process(process(env, res, "P1", 1))
        env.process(process(env, res, "P2", 2))
        env.process(process(env, res, "P3", 3))

        env.run()

        # All served (order depends on process start, not priority when capacity available)
        assert len(served_order) == 3
        assert set(served_order) == {"P1", "P2", "P3"}

    def test_utilisation_calculation(self):
        """Utilisation is calculated correctly."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 10, 20)

        def occupy(env, res, count):
            requests = []
            for _ in range(count):
                req = res.request()
                yield req
                requests.append(req)
            # Check utilisation with 5 of 10 slots used
            assert res.utilisation == 0.5
            yield env.timeout(10)

        env.process(occupy(env, res, 5))
        env.run()

    def test_utilisation_zero_capacity(self):
        """Utilisation handles zero capacity edge case."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 1, 10)

        # Remove all capacity
        res.remove_capacity(1, graceful=False)

        assert res.capacity == 0
        assert res.utilisation == 0.0  # No occupants

    def test_queue_length(self):
        """Queue length is tracked correctly."""
        env = simpy.Environment()
        # Note: max_capacity=1 ensures requests will queue
        res = DynamicCapacityResource(env, "test", 1, 1)
        queue_lengths = []

        def fill_and_queue(env, res):
            # Fill the single slot
            req1 = res.request()
            yield req1

            # Start more requests that will queue (since max_capacity=1)
            def queuer():
                req2 = res.request()
                yield req2
                yield env.timeout(5)

            env.process(queuer())
            env.process(queuer())

            yield env.timeout(0.1)  # Small delay for processes to start
            queue_lengths.append(res.queue_length)

        env.process(fill_and_queue(env, res))
        env.run(until=2)

        # Two processes should be queued
        assert queue_lengths[0] == 2

    def test_get_capacity_timeline(self):
        """Capacity timeline for plotting is generated correctly."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        def change_capacity(env, res):
            yield env.timeout(10)
            res.add_capacity(3)
            yield env.timeout(10)
            res.add_capacity(2)

        env.process(change_capacity(env, res))
        env.run()

        timeline = res.get_capacity_timeline()

        assert len(timeline) == 3
        assert timeline[0] == (0.0, 5)
        assert timeline[1] == (10.0, 8)
        assert timeline[2] == (20.0, 10)

    def test_get_metrics(self):
        """Metrics are calculated correctly."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        def changes(env, res):
            yield env.timeout(10)
            res.add_capacity(3)  # Scale up
            yield env.timeout(10)
            res.add_capacity(2)  # Scale up
            yield env.timeout(10)
            res.remove_capacity(4, graceful=False)  # Scale down

        env.process(changes(env, res))
        env.run()

        metrics = res.get_metrics()

        assert metrics["scale_up_events"] == 2
        assert metrics["scale_down_events"] == 1
        assert metrics["total_capacity_changes"] == 3
        assert metrics["max_capacity_reached"] == 10
        assert metrics["min_capacity_reached"] == 5

    def test_no_changes_metrics(self):
        """Metrics handle no changes case."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)

        metrics = res.get_metrics()

        assert metrics["scale_up_events"] == 0
        assert metrics["scale_down_events"] == 0


class TestDynamicResourceManager:
    """Tests for DynamicResourceManager class."""

    def test_create_resource(self):
        """Manager creates and registers resources."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        res = manager.create_resource("ed_bays", 10, 30)

        assert res is not None
        assert res.capacity == 10
        assert manager.get_resource("ed_bays") is res

    def test_get_nonexistent_resource(self):
        """Getting nonexistent resource returns None."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        assert manager.get_resource("nonexistent") is None

    def test_get_all_resources(self):
        """Can retrieve all managed resources."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        manager.create_resource("ed_bays", 10, 30)
        manager.create_resource("ward_beds", 20, 50)

        resources = manager.get_all_resources()

        assert len(resources) == 2
        assert "ed_bays" in resources
        assert "ward_beds" in resources

    def test_get_utilisation_summary(self):
        """Utilisation summary returns all resource utilisations."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        manager.create_resource("ed_bays", 10, 30)
        manager.create_resource("ward_beds", 20, 50)

        summary = manager.get_utilisation_summary()

        assert summary["ed_bays"] == 0.0
        assert summary["ward_beds"] == 0.0

    def test_get_aggregated_metrics(self):
        """Aggregated metrics sum across all resources."""
        env = simpy.Environment()
        manager = DynamicResourceManager(env)

        res1 = manager.create_resource("ed_bays", 10, 30)
        res2 = manager.create_resource("ward_beds", 20, 50)

        def changes(env):
            yield env.timeout(10)
            res1.add_capacity(5)
            res2.add_capacity(10)
            yield env.timeout(10)
            res1.remove_capacity(2, graceful=False)

        env.process(changes(env))
        env.run()

        metrics = manager.get_aggregated_metrics()

        assert metrics["total_scale_up_events"] == 2
        assert metrics["total_scale_down_events"] == 1
        assert "ed_bays" in metrics["by_resource"]
        assert "ward_beds" in metrics["by_resource"]


class TestGracefulDeactivation:
    """Tests for graceful capacity removal."""

    def test_graceful_removal_waits_for_release(self):
        """Graceful removal waits for occupants to leave."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)
        capacity_during_occupancy = None

        def occupy_and_remove(env, res):
            nonlocal capacity_during_occupancy

            # Occupy 3 slots
            requests = []
            for _ in range(3):
                req = res.request()
                yield req
                requests.append(req)

            # Request graceful removal of 2 slots
            res.remove_capacity(2, graceful=True)

            # Effective capacity should be reduced
            capacity_during_occupancy = res.capacity

            # Wait then release
            yield env.timeout(10)

            # Release the requests (manually handle for test)
            for req in requests:
                res.release(req)

        env.process(occupy_and_remove(env, res))
        env.run()

        # During occupancy, effective capacity was 3 (5 - 2 pending)
        assert capacity_during_occupancy == 3

    def test_slots_deactivate_on_release(self):
        """Pending deactivations happen when slots are released."""
        env = simpy.Environment()
        res = DynamicCapacityResource(env, "test", 5, 20)
        final_capacity = None

        def test_process(env, res):
            nonlocal final_capacity

            # Occupy 3 slots
            req1 = res.request()
            yield req1
            req2 = res.request()
            yield req2
            req3 = res.request()
            yield req3

            # Request graceful removal of 2
            res.remove_capacity(2, graceful=True)
            assert res._pending_deactivations == 2

            # Release one - should deactivate one slot
            res.release(req1)
            assert res._pending_deactivations == 1
            assert res._active_slots == 4

            # Release another - should deactivate another slot
            res.release(req2)
            assert res._pending_deactivations == 0
            assert res._active_slots == 3

            res.release(req3)
            final_capacity = res.capacity

        env.process(test_process(env, res))
        env.run()

        assert final_capacity == 3
