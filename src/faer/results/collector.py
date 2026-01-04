"""Event logging during simulation runs."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


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
    """

    arrivals: int = 0
    departures: int = 0
    queue_times: List[float] = field(default_factory=list)
    system_times: List[float] = field(default_factory=list)
    resource_log: List[Tuple[float, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize with starting state."""
        if not self.resource_log:
            self.resource_log = [(0.0, 0)]

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
