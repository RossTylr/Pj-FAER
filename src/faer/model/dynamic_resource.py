"""
Dynamic Capacity Resource for Pj-FAER.

SimPy resources have fixed capacity after creation. This module provides
a wrapper that enables runtime capacity changes for modelling surge protocols,
step-up/step-down beds, and adaptive resource management.

Strategy: Create resource with maximum possible capacity, then control
effective capacity through slot activation/deactivation tracking.
"""

import simpy
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CapacityChangeEvent:
    """Record of a capacity change."""
    time: float
    old_capacity: int
    new_capacity: int
    reason: str = ""


class DynamicCapacityResource:
    """
    Wrapper around SimPy resources that supports capacity changes.

    This enables modelling of:
    - Surge bed activation when utilisation exceeds threshold
    - Step-down when pressure eases
    - Shift-based staffing changes
    - OPEL-triggered capacity adjustments

    The wrapper creates a SimPy resource with maximum capacity and tracks
    which "slots" are active. Adding capacity activates more slots.
    Removing capacity deactivates slots (gracefully waiting for release).

    Attributes:
        env: SimPy environment.
        name: Resource identifier (e.g., "ed_bays", "ward_beds").
        current_capacity: Effective capacity at this moment.
        max_capacity: Maximum possible capacity (resource ceiling).
        is_priority: Whether to use PriorityResource for priority queuing.
    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        initial_capacity: int,
        max_capacity: int,
        is_priority: bool = False
    ):
        """
        Initialize a dynamic capacity resource.

        Args:
            env: SimPy environment.
            name: Resource name for logging and identification.
            initial_capacity: Starting number of active slots.
            max_capacity: Maximum slots that can ever be active.
            is_priority: Use PriorityResource for priority-based queuing.
        """
        self.env = env
        self.name = name
        self._active_slots = initial_capacity
        self.max_capacity = max_capacity
        self.is_priority = is_priority
        self._pending_deactivations = 0

        # Create underlying SimPy resource with max capacity
        # We control effective capacity through our tracking
        if is_priority:
            self._resource = simpy.PriorityResource(env, capacity=max_capacity)
        else:
            self._resource = simpy.Resource(env, capacity=max_capacity)

        # Capacity change log for metrics and visualisation
        self.capacity_log: List[CapacityChangeEvent] = [
            CapacityChangeEvent(
                time=0.0,
                old_capacity=initial_capacity,
                new_capacity=initial_capacity,
                reason="initial"
            )
        ]

    @property
    def count(self) -> int:
        """Number of resources currently in use."""
        return self._resource.count

    @property
    def capacity(self) -> int:
        """Current effective capacity (active slots minus pending deactivations)."""
        return self._active_slots - self._pending_deactivations

    @property
    def queue(self) -> list:
        """Current queue of waiting requests."""
        return self._resource.queue

    @property
    def queue_length(self) -> int:
        """Number of requests waiting in queue."""
        return len(self._resource.queue)

    @property
    def utilisation(self) -> float:
        """Current utilisation as fraction (0.0 to 1.0+)."""
        if self.capacity <= 0:
            return 1.0 if self.count > 0 else 0.0
        return self.count / self.capacity

    def request(self, priority: int = 0):
        """
        Request a resource slot.

        For priority resources, lower priority numbers are served first.

        Args:
            priority: Priority level (only used if is_priority=True).

        Returns:
            SimPy request event.
        """
        if self.is_priority:
            return self._resource.request(priority=priority)
        return self._resource.request()

    def release(self, request):
        """
        Release a resource slot.

        If there are pending deactivations and this release frees a slot
        that should be deactivated, handle the deactivation.

        Args:
            request: The request object to release.
        """
        self._resource.release(request)

        # Handle graceful deactivation
        if self._pending_deactivations > 0:
            # Check if we can now deactivate a slot
            if self._active_slots - self.count > 0:
                self._active_slots -= 1
                self._pending_deactivations -= 1
                self.capacity_log.append(CapacityChangeEvent(
                    time=self.env.now,
                    old_capacity=self._active_slots + 1,
                    new_capacity=self._active_slots,
                    reason="graceful_deactivation"
                ))

    def add_capacity(self, amount: int, reason: str = "") -> int:
        """
        Increase capacity by activating additional slots.

        Args:
            amount: Number of slots to activate.
            reason: Reason for the change (for logging).

        Returns:
            Actual number of slots activated (may be less if at max).
        """
        old_capacity = self._active_slots
        actual_add = min(amount, self.max_capacity - self._active_slots)

        if actual_add > 0:
            self._active_slots += actual_add
            self.capacity_log.append(CapacityChangeEvent(
                time=self.env.now,
                old_capacity=old_capacity,
                new_capacity=self._active_slots,
                reason=reason or f"add_{actual_add}"
            ))

        return actual_add

    def remove_capacity(self, amount: int, graceful: bool = True, reason: str = "") -> int:
        """
        Decrease capacity by deactivating slots.

        Args:
            amount: Number of slots to deactivate.
            graceful: If True, wait for current occupants to leave.
                     If False, only remove empty slots immediately.
            reason: Reason for the change (for logging).

        Returns:
            Actual number of slots that will be removed.
        """
        # Can't remove more than we have (minus already pending)
        available_to_remove = self._active_slots - self._pending_deactivations
        actual_remove = min(amount, available_to_remove)

        if actual_remove <= 0:
            return 0

        old_capacity = self._active_slots

        if graceful:
            # Mark slots for deactivation when released
            self._pending_deactivations += actual_remove
            self.capacity_log.append(CapacityChangeEvent(
                time=self.env.now,
                old_capacity=old_capacity,
                new_capacity=self.capacity,  # Effective capacity reduced
                reason=reason or f"pending_remove_{actual_remove}"
            ))
        else:
            # Immediate removal - only remove empty slots
            empty_slots = self._active_slots - self.count
            immediate_remove = min(actual_remove, empty_slots)
            if immediate_remove > 0:
                self._active_slots -= immediate_remove
                self.capacity_log.append(CapacityChangeEvent(
                    time=self.env.now,
                    old_capacity=old_capacity,
                    new_capacity=self._active_slots,
                    reason=reason or f"immediate_remove_{immediate_remove}"
                ))
            actual_remove = immediate_remove

        return actual_remove

    def get_capacity_timeline(self) -> List[Tuple[float, int]]:
        """
        Get capacity over time for plotting.

        Returns:
            List of (time, capacity) tuples.
        """
        return [(event.time, event.new_capacity) for event in self.capacity_log]

    def get_metrics(self) -> dict:
        """
        Calculate capacity scaling metrics.

        Returns:
            Dictionary with scaling statistics.
        """
        if len(self.capacity_log) <= 1:
            return {
                "scale_up_events": 0,
                "scale_down_events": 0,
                "total_capacity_changes": 0,
                "max_capacity_reached": self._active_slots,
                "min_capacity_reached": self._active_slots,
            }

        scale_ups = 0
        scale_downs = 0
        max_cap = self.capacity_log[0].new_capacity
        min_cap = self.capacity_log[0].new_capacity

        for i in range(1, len(self.capacity_log)):
            prev = self.capacity_log[i - 1].new_capacity
            curr = self.capacity_log[i].new_capacity

            if curr > prev:
                scale_ups += 1
            elif curr < prev:
                scale_downs += 1

            max_cap = max(max_cap, curr)
            min_cap = min(min_cap, curr)

        return {
            "scale_up_events": scale_ups,
            "scale_down_events": scale_downs,
            "total_capacity_changes": scale_ups + scale_downs,
            "max_capacity_reached": max_cap,
            "min_capacity_reached": min_cap,
        }


class DynamicResourceManager:
    """
    Manages multiple dynamic resources for a simulation.

    Provides a central point for accessing and modifying resources,
    and aggregating metrics across all managed resources.
    """

    def __init__(self, env: simpy.Environment):
        """
        Initialize the resource manager.

        Args:
            env: SimPy environment.
        """
        self.env = env
        self._resources: dict[str, DynamicCapacityResource] = {}

    def create_resource(
        self,
        name: str,
        initial_capacity: int,
        max_capacity: int,
        is_priority: bool = False
    ) -> DynamicCapacityResource:
        """
        Create and register a new dynamic resource.

        Args:
            name: Unique resource identifier.
            initial_capacity: Starting capacity.
            max_capacity: Maximum possible capacity.
            is_priority: Use priority queuing.

        Returns:
            The created DynamicCapacityResource.
        """
        resource = DynamicCapacityResource(
            env=self.env,
            name=name,
            initial_capacity=initial_capacity,
            max_capacity=max_capacity,
            is_priority=is_priority
        )
        self._resources[name] = resource
        return resource

    def get_resource(self, name: str) -> Optional[DynamicCapacityResource]:
        """Get a resource by name."""
        return self._resources.get(name)

    def get_all_resources(self) -> dict[str, DynamicCapacityResource]:
        """Get all managed resources."""
        return self._resources.copy()

    def get_utilisation_summary(self) -> dict[str, float]:
        """Get current utilisation of all resources."""
        return {
            name: resource.utilisation
            for name, resource in self._resources.items()
        }

    def get_aggregated_metrics(self) -> dict:
        """
        Aggregate metrics across all managed resources.

        Returns:
            Dictionary with combined scaling statistics.
        """
        total_scale_ups = 0
        total_scale_downs = 0
        resource_metrics = {}

        for name, resource in self._resources.items():
            metrics = resource.get_metrics()
            total_scale_ups += metrics["scale_up_events"]
            total_scale_downs += metrics["scale_down_events"]
            resource_metrics[name] = metrics

        return {
            "total_scale_up_events": total_scale_ups,
            "total_scale_down_events": total_scale_downs,
            "by_resource": resource_metrics
        }
