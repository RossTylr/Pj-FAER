"""Event logging during simulation runs."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# ============== Scaling Event Tracking (Phase 12) ==============

@dataclass
class ScalingEventRecord:
    """
    Record of a capacity scaling event.

    Attributes:
        time: Simulation time of the event.
        rule_name: Name of the rule that triggered.
        action_type: Type of action taken (e.g., "add_capacity").
        resource: Resource affected (e.g., "ed_bays").
        old_capacity: Capacity before the change.
        new_capacity: Capacity after the change.
        trigger_value: The utilisation/queue value that triggered.
        direction: "scale_up" or "scale_down".
    """
    time: float
    rule_name: str
    action_type: str
    resource: str
    old_capacity: int
    new_capacity: int
    trigger_value: float
    direction: str


@dataclass
class ScalingMetrics:
    """
    Aggregated metrics for capacity scaling behavior.

    Attributes:
        total_scale_up_events: Number of scale-up events.
        total_scale_down_events: Number of scale-down events.
        total_additional_bed_hours: Cumulative surge capacity used (bed-hours).
        pct_time_at_surge: Percentage of run time at elevated capacity.
        opel_peak_level: Highest OPEL level reached (1-4).
        opel_transitions: Number of OPEL level changes.
        opel_time_at_level: Time spent at each OPEL level.
        patients_diverted: Number of patients diverted.
        rule_activations: Activation count per rule.
    """
    total_scale_up_events: int = 0
    total_scale_down_events: int = 0
    total_additional_bed_hours: float = 0.0
    pct_time_at_surge: float = 0.0
    opel_peak_level: int = 1
    opel_transitions: int = 0
    opel_time_at_level: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})
    patients_diverted: int = 0
    rule_activations: Dict[str, int] = field(default_factory=dict)


@dataclass
class ResultsCollector:
    """Collect and compute simulation metrics.

    This class accumulates events during a simulation run and computes
    summary metrics after the run completes.

    Attributes:
        arrivals: Total count of patient arrivals.
        departures: Total count of patient departures.
        queue_times: List of queue times (minutes).
        system_times: List of total system times (minutes).
        resource_log: List of (time, n_busy) tuples for utilisation.
        scaling_events: List of capacity scaling events (Phase 12).
        scaling_metrics: Aggregated scaling metrics (Phase 12).
        patients_diverted: Count of diverted patients (Phase 12).
    """

    arrivals: int = 0
    departures: int = 0
    queue_times: List[float] = field(default_factory=list)
    system_times: List[float] = field(default_factory=list)
    resource_log: List[Tuple[float, int]] = field(default_factory=list)

    # Phase 12: Capacity Scaling
    scaling_events: List[ScalingEventRecord] = field(default_factory=list)
    scaling_metrics: Optional[ScalingMetrics] = None
    patients_diverted: int = 0

    def __post_init__(self) -> None:
        """Initialize with starting state."""
        if not self.resource_log:
            self.resource_log = [(0.0, 0)]
        if self.scaling_metrics is None:
            self.scaling_metrics = ScalingMetrics()

    def record_arrival(self) -> None:
        """Record a patient arrival."""
        self.arrivals += 1

    def record_queue_time(self, wait: float) -> None:
        """Record a patient's queue time.

        Args:
            wait: Time spent waiting in queue (minutes).
        """
        self.queue_times.append(wait)

    def record_system_time(self, time: float) -> None:
        """Record a patient's total system time and increment departures.

        Args:
            time: Total time in system (minutes).
        """
        self.system_times.append(time)
        self.departures += 1

    def record_resource_state(self, time: float, n_busy: int) -> None:
        """Record resource utilisation state change.

        Args:
            time: Simulation time of state change.
            n_busy: Number of resources currently in use.
        """
        self.resource_log.append((time, n_busy))

    def compute_metrics(self, run_length: float, capacity: int) -> Dict:
        """Compute all KPIs from collected data.

        Args:
            run_length: Total simulation time (minutes).
            capacity: Resource capacity.

        Returns:
            Dictionary containing:
            - arrivals, departures: Counts
            - p_delay: Proportion of patients who waited
            - mean_queue_time, median_queue_time, p95_queue_time
            - mean_system_time
            - utilisation: Time-weighted resource utilisation
            - throughput_per_hour: Departures per hour
        """
        queue_times = np.array(self.queue_times) if self.queue_times else np.array([0.0])
        system_times = np.array(self.system_times) if self.system_times else np.array([0.0])

        # P(delay) = P(queue_time > 0)
        p_delay = float(np.mean(queue_times > 0)) if len(self.queue_times) > 0 else 0.0

        # Queue time quantiles
        if len(self.queue_times) > 0:
            mean_queue = float(np.mean(queue_times))
            median_queue = float(np.percentile(queue_times, 50))
            p95_queue = float(np.percentile(queue_times, 95))
        else:
            mean_queue = median_queue = p95_queue = 0.0

        # System time
        mean_system = float(np.mean(system_times)) if len(self.system_times) > 0 else 0.0

        # Time-weighted utilisation
        utilisation = self._compute_utilisation(run_length, capacity)

        # Throughput
        throughput = self.departures / (run_length / 60) if run_length > 0 else 0.0

        return {
            "arrivals": self.arrivals,
            "departures": self.departures,
            "p_delay": p_delay,
            "mean_queue_time": mean_queue,
            "median_queue_time": median_queue,
            "p95_queue_time": p95_queue,
            "mean_system_time": mean_system,
            "utilisation": utilisation,
            "throughput_per_hour": throughput,
        }

    def _compute_utilisation(self, run_length: float, capacity: int) -> float:
        """Compute time-weighted resource utilisation.

        Args:
            run_length: Total simulation time.
            capacity: Resource capacity.

        Returns:
            Utilisation as a fraction (0-1).
        """
        if not self.resource_log or capacity == 0 or run_length == 0:
            return 0.0

        # Sort by time
        log = sorted(self.resource_log)

        total_busy_time = 0.0
        for i in range(len(log) - 1):
            t_start, n_busy = log[i]
            t_end = log[i + 1][0]
            total_busy_time += n_busy * (t_end - t_start)

        # Final segment to run_length
        if log:
            t_last, n_last = log[-1]
            total_busy_time += n_last * (run_length - t_last)

        return total_busy_time / (capacity * run_length)

    # ============== Phase 12: Scaling Event Methods ==============

    def record_scaling_event(
        self,
        time: float,
        rule_name: str,
        action_type: str,
        resource: str,
        old_capacity: int,
        new_capacity: int,
        trigger_value: float,
        direction: str
    ) -> None:
        """
        Record a capacity scaling event.

        Args:
            time: Simulation time of the event.
            rule_name: Name of the rule that triggered.
            action_type: Type of action taken.
            resource: Resource affected.
            old_capacity: Capacity before change.
            new_capacity: Capacity after change.
            trigger_value: Value that triggered the event.
            direction: "scale_up" or "scale_down".
        """
        event = ScalingEventRecord(
            time=time,
            rule_name=rule_name,
            action_type=action_type,
            resource=resource,
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            trigger_value=trigger_value,
            direction=direction
        )
        self.scaling_events.append(event)

        # Update aggregated metrics
        if direction == "scale_up":
            self.scaling_metrics.total_scale_up_events += 1
        else:
            self.scaling_metrics.total_scale_down_events += 1

        # Track rule activations
        if rule_name not in self.scaling_metrics.rule_activations:
            self.scaling_metrics.rule_activations[rule_name] = 0
        self.scaling_metrics.rule_activations[rule_name] += 1

    def record_diversion(self) -> None:
        """Record a diverted patient."""
        self.patients_diverted += 1
        self.scaling_metrics.patients_diverted += 1

    def finalize_scaling_metrics(
        self,
        run_length: float,
        baseline_capacities: Dict[str, int],
        opel_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize scaling metrics after simulation completes.

        Args:
            run_length: Total simulation run length in minutes.
            baseline_capacities: Dictionary of baseline capacities per resource.
            opel_metrics: Optional OPEL-specific metrics from scaling monitor.
        """
        # Calculate additional bed-hours from scaling events
        # This is a simplified calculation - tracks surge capacity usage
        resource_timelines: Dict[str, List[Tuple[float, int]]] = {}

        for event in self.scaling_events:
            if event.resource not in resource_timelines:
                resource_timelines[event.resource] = []
            resource_timelines[event.resource].append(
                (event.time, event.new_capacity)
            )

        total_additional_bed_mins = 0.0
        time_at_surge_mins = 0.0

        for resource, timeline in resource_timelines.items():
            baseline = baseline_capacities.get(resource, 0)
            sorted_timeline = sorted(timeline, key=lambda x: x[0])

            for i, (time, capacity) in enumerate(sorted_timeline):
                if i + 1 < len(sorted_timeline):
                    duration = sorted_timeline[i + 1][0] - time
                else:
                    duration = run_length - time

                if duration > 0:
                    additional = max(0, capacity - baseline)
                    total_additional_bed_mins += additional * duration

                    if capacity > baseline:
                        time_at_surge_mins += duration

        # Convert to hours
        self.scaling_metrics.total_additional_bed_hours = total_additional_bed_mins / 60.0

        # Calculate percentage of time at surge
        if run_length > 0:
            self.scaling_metrics.pct_time_at_surge = (time_at_surge_mins / run_length) * 100.0

        # Incorporate OPEL metrics if provided
        if opel_metrics:
            self.scaling_metrics.opel_peak_level = opel_metrics.get("opel_peak_level", 1)
            self.scaling_metrics.opel_transitions = opel_metrics.get("opel_transitions", 0)
            if "opel_time_at_level" in opel_metrics:
                for level, time_val in opel_metrics["opel_time_at_level"].items():
                    level_int = level.value if hasattr(level, 'value') else int(level)
                    self.scaling_metrics.opel_time_at_level[level_int] = time_val

    def get_scaling_summary(self) -> Dict[str, Any]:
        """
        Get a summary of scaling metrics for display.

        Returns:
            Dictionary with scaling summary data.
        """
        return {
            "scale_up_events": self.scaling_metrics.total_scale_up_events,
            "scale_down_events": self.scaling_metrics.total_scale_down_events,
            "total_events": len(self.scaling_events),
            "additional_bed_hours": self.scaling_metrics.total_additional_bed_hours,
            "pct_time_at_surge": self.scaling_metrics.pct_time_at_surge,
            "opel_peak_level": self.scaling_metrics.opel_peak_level,
            "opel_transitions": self.scaling_metrics.opel_transitions,
            "patients_diverted": self.scaling_metrics.patients_diverted,
            "rule_activations": dict(self.scaling_metrics.rule_activations),
            "events": [
                {
                    "time": e.time,
                    "rule": e.rule_name,
                    "action": e.action_type,
                    "resource": e.resource,
                    "old_capacity": e.old_capacity,
                    "new_capacity": e.new_capacity,
                    "trigger_value": e.trigger_value,
                    "direction": e.direction,
                }
                for e in self.scaling_events
            ]
        }
