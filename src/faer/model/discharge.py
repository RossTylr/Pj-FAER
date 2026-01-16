"""
Discharge Manager for Pj-FAER.

Manages discharge acceleration and discharge lounge functionality
as part of capacity scaling. Enables modelling of discharge push
protocols during OPEL 3/4 escalation.
"""

import simpy
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple

from faer.core.scaling import CapacityScalingConfig


@dataclass
class DischargeLoungeEntry:
    """Record of a patient in the discharge lounge."""
    patient_id: int
    entry_time: float
    original_bed_resource: str


@dataclass
class DischargeMetrics:
    """Metrics for discharge acceleration and lounge usage."""
    total_lounge_entries: int = 0
    total_lounge_exits: int = 0
    total_time_in_lounge: float = 0.0
    acceleration_activations: int = 0
    los_reduction_applied_count: int = 0
    total_los_reduction_mins: float = 0.0


class DischargeManager:
    """
    Manages discharge acceleration and discharge lounge.

    This class provides:
    1. Length-of-stay (LoS) acceleration during capacity pressure
    2. Discharge lounge functionality to free beds faster
    3. Metrics tracking for discharge-related interventions

    Discharge acceleration reduces remaining LoS for patients who are
    medically fit for discharge, modelling "discharge push" protocols.

    The discharge lounge allows patients who are medically fit but
    awaiting transport/discharge paperwork to wait in a holding area,
    freeing their bed for new admissions.
    """

    def __init__(
        self,
        env: simpy.Environment,
        config: CapacityScalingConfig
    ):
        """
        Initialize the discharge manager.

        Args:
            env: SimPy environment.
            config: Capacity scaling configuration.
        """
        self.env = env
        self.config = config

        # Acceleration state
        self.acceleration_active = False
        self.los_reduction_factor = 1.0  # Multiplier on remaining LoS (1.0 = no reduction)
        self._los_reduction_pct = 0.0

        # Discharge lounge
        self.lounge_enabled = False
        self.discharge_lounge: Optional[simpy.Resource] = None

        if config.discharge_lounge_capacity > 0:
            self.discharge_lounge = simpy.Resource(
                env, capacity=config.discharge_lounge_capacity
            )
            self.lounge_enabled = True

        # Current lounge patients
        self.lounge_patients: List[DischargeLoungeEntry] = []

        # Metrics
        self.metrics = DischargeMetrics()

        # Event log for visualization
        self.lounge_log: List[Tuple[float, int, str, str]] = []  # (time, patient_id, action, bed_type)

    def activate_acceleration(self, los_reduction_pct: float) -> None:
        """
        Activate discharge acceleration.

        Args:
            los_reduction_pct: Percentage reduction in LoS (e.g., 15 means 15% reduction).
        """
        self.acceleration_active = True
        self._los_reduction_pct = los_reduction_pct
        self.los_reduction_factor = 1.0 - (los_reduction_pct / 100.0)
        self.metrics.acceleration_activations += 1

    def deactivate_acceleration(self) -> None:
        """Return to normal discharge patterns."""
        self.acceleration_active = False
        self.los_reduction_factor = 1.0
        self._los_reduction_pct = 0.0

    def activate_lounge(self) -> None:
        """Activate the discharge lounge (if configured)."""
        self.lounge_enabled = True

    def deactivate_lounge(self) -> None:
        """Deactivate the discharge lounge."""
        self.lounge_enabled = False

    def get_adjusted_los(self, base_los: float) -> float:
        """
        Get LoS adjusted for any active acceleration.

        Args:
            base_los: Base length of stay in minutes.

        Returns:
            Adjusted LoS (may be reduced if acceleration active).
        """
        if self.acceleration_active:
            adjusted = base_los * self.los_reduction_factor
            self.metrics.los_reduction_applied_count += 1
            self.metrics.total_los_reduction_mins += (base_los - adjusted)
            return adjusted
        return base_los

    def is_lounge_available(self) -> bool:
        """
        Check if discharge lounge has capacity.

        Returns:
            True if lounge exists and has space.
        """
        if not self.lounge_enabled or self.discharge_lounge is None:
            return False
        return self.discharge_lounge.count < self.discharge_lounge.capacity

    def try_move_to_lounge(
        self,
        patient_id: int,
        bed_resource_name: str
    ) -> bool:
        """
        Attempt to move patient to discharge lounge.

        This is a check-only method. The actual move should be done
        by calling enter_lounge_process().

        Args:
            patient_id: Patient identifier.
            bed_resource_name: Name of the bed resource being freed.

        Returns:
            True if lounge has space and patient can be moved.
        """
        return self.is_lounge_available()

    def enter_lounge_process(
        self,
        patient_id: int,
        bed_resource_name: str,
        remaining_discharge_time: float
    ):
        """
        SimPy process for patient waiting in discharge lounge.

        Patient occupies lounge space while waiting for final discharge
        (transport, paperwork, etc.). Their ward/ITU bed is freed immediately.

        Args:
            patient_id: Patient identifier.
            bed_resource_name: Name of the bed resource that was freed.
            remaining_discharge_time: Time patient still needs before final discharge.

        Yields:
            SimPy events.
        """
        if self.discharge_lounge is None:
            return

        with self.discharge_lounge.request() as req:
            yield req

            # Record entry
            entry = DischargeLoungeEntry(
                patient_id=patient_id,
                entry_time=self.env.now,
                original_bed_resource=bed_resource_name
            )
            self.lounge_patients.append(entry)
            self.lounge_log.append((self.env.now, patient_id, "entered", bed_resource_name))
            self.metrics.total_lounge_entries += 1

            # Wait in lounge (capped by max wait time)
            max_wait = self.config.discharge_lounge_max_wait_mins
            wait_time = min(remaining_discharge_time, max_wait)
            yield self.env.timeout(wait_time)

            # Record exit
            self.lounge_patients.remove(entry)
            time_in_lounge = self.env.now - entry.entry_time
            self.metrics.total_lounge_exits += 1
            self.metrics.total_time_in_lounge += time_in_lounge
            self.lounge_log.append((self.env.now, patient_id, "departed", bed_resource_name))

    def get_lounge_occupancy(self) -> int:
        """Get current number of patients in discharge lounge."""
        if self.discharge_lounge is None:
            return 0
        return self.discharge_lounge.count

    def get_lounge_queue(self) -> int:
        """Get number of patients waiting for lounge space."""
        if self.discharge_lounge is None:
            return 0
        return len(self.discharge_lounge.queue)

    def get_metrics(self) -> dict:
        """
        Get discharge manager metrics.

        Returns:
            Dictionary with discharge-related statistics.
        """
        avg_time_in_lounge = 0.0
        if self.metrics.total_lounge_exits > 0:
            avg_time_in_lounge = self.metrics.total_time_in_lounge / self.metrics.total_lounge_exits

        avg_los_reduction = 0.0
        if self.metrics.los_reduction_applied_count > 0:
            avg_los_reduction = self.metrics.total_los_reduction_mins / self.metrics.los_reduction_applied_count

        return {
            "acceleration_active": self.acceleration_active,
            "los_reduction_factor": self.los_reduction_factor,
            "los_reduction_pct": self._los_reduction_pct,
            "acceleration_activations": self.metrics.acceleration_activations,
            "los_reduction_applied_count": self.metrics.los_reduction_applied_count,
            "total_los_reduction_mins": self.metrics.total_los_reduction_mins,
            "avg_los_reduction_mins": avg_los_reduction,
            "lounge_enabled": self.lounge_enabled,
            "lounge_capacity": self.config.discharge_lounge_capacity,
            "lounge_entries": self.metrics.total_lounge_entries,
            "lounge_exits": self.metrics.total_lounge_exits,
            "total_time_in_lounge": self.metrics.total_time_in_lounge,
            "avg_time_in_lounge": avg_time_in_lounge,
            "current_lounge_occupancy": self.get_lounge_occupancy(),
        }

    def get_lounge_timeline(self) -> List[Tuple[float, int, str, str]]:
        """
        Get lounge events for visualization.

        Returns:
            List of (time, patient_id, action, bed_type) tuples.
        """
        return list(self.lounge_log)
