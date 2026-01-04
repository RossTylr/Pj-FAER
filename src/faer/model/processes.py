"""SimPy process logic for patient flow."""

from typing import Dict, Generator, List, Any

import numpy as np
import simpy

from faer.core.scenario import Scenario


def sample_lognormal(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Sample from lognormal distribution given mean and CV.

    Args:
        rng: NumPy random generator.
        mean: Desired mean of the distribution.
        cv: Coefficient of variation (std/mean).

    Returns:
        A sample from the lognormal distribution.
    """
    if cv <= 0:
        return mean
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - sigma**2 / 2
    return float(rng.lognormal(mu, sigma))


def patient_process(
    env: simpy.Environment,
    patient_id: int,
    resus_bays: simpy.Resource,
    scenario: Scenario,
    results: Dict[str, Any],
) -> Generator[simpy.Event, None, None]:
    """Single patient journey: arrive -> queue -> service -> depart.

    Args:
        env: SimPy environment.
        patient_id: Unique identifier for this patient.
        resus_bays: SimPy Resource representing Resus bays.
        scenario: Scenario configuration with parameters and RNGs.
        results: Dictionary to record metrics.

    Yields:
        SimPy events for resource requests and timeouts.
    """
    arrival_time = env.now

    # Request Resus bay
    with resus_bays.request() as req:
        yield req

        # Record queue time (time waiting for resource)
        queue_time = env.now - arrival_time
        results["queue_times"].append(queue_time)

        # Record resource state for utilisation calculation
        results["resource_log"].append((env.now, resus_bays.count))

        # Service time (lognormal for realistic LoS)
        service_time = sample_lognormal(
            scenario.rng_service, scenario.resus_mean, scenario.resus_cv
        )
        yield env.timeout(service_time)

    # Record resource state after departure
    results["resource_log"].append((env.now, resus_bays.count))

    # Patient departed
    system_time = env.now - arrival_time
    results["system_times"].append(system_time)
    results["departures"] += 1


def arrival_generator(
    env: simpy.Environment,
    resus_bays: simpy.Resource,
    scenario: Scenario,
    results: Dict[str, Any],
) -> Generator[simpy.Event, None, None]:
    """Generate patient arrivals at constant rate (Phase 1).

    Args:
        env: SimPy environment.
        resus_bays: SimPy Resource representing Resus bays.
        scenario: Scenario configuration with parameters and RNGs.
        results: Dictionary to record metrics.

    Yields:
        SimPy timeout events for inter-arrival times.
    """
    patient_id = 0

    while True:
        # Exponential inter-arrival time
        iat = scenario.rng_arrivals.exponential(scenario.mean_iat)
        yield env.timeout(iat)

        # New arrival
        patient_id += 1
        results["arrivals"] += 1

        # Spawn patient process
        env.process(patient_process(env, patient_id, resus_bays, scenario, results))


def run_simulation(scenario: Scenario) -> Dict[str, Any]:
    """Execute a single simulation run.

    Args:
        scenario: Scenario configuration with all parameters.

    Returns:
        Dictionary containing simulation results:
        - arrivals: Total number of arrivals
        - departures: Total number of departures
        - queue_times: List of queue times (minutes)
        - system_times: List of total system times (minutes)
        - resource_log: List of (time, n_busy) tuples
    """
    # Initialize SimPy environment
    env = simpy.Environment()

    # Create Resus resource
    resus_bays = simpy.Resource(env, capacity=scenario.n_resus_bays)

    # Initialize results collection
    results: Dict[str, Any] = {
        "arrivals": 0,
        "departures": 0,
        "queue_times": [],
        "system_times": [],
        "resource_log": [(0.0, 0)],  # Initial state
    }

    # Start arrival generator
    env.process(arrival_generator(env, resus_bays, scenario, results))

    # Run simulation
    env.run(until=scenario.run_length)

    # Compute summary metrics
    results["run_length"] = scenario.run_length
    results["n_resus_bays"] = scenario.n_resus_bays

    if results["queue_times"]:
        queue_times = np.array(results["queue_times"])
        results["mean_queue_time"] = float(np.mean(queue_times))
        results["p_delay"] = float(np.mean(queue_times > 0))
    else:
        results["mean_queue_time"] = 0.0
        results["p_delay"] = 0.0

    if results["system_times"]:
        results["mean_system_time"] = float(np.mean(results["system_times"]))
    else:
        results["mean_system_time"] = 0.0

    # Compute utilisation from resource log
    results["utilisation"] = _compute_utilisation(
        results["resource_log"], scenario.run_length, scenario.n_resus_bays
    )

    return results


def _compute_utilisation(
    resource_log: List[tuple], run_length: float, capacity: int
) -> float:
    """Compute time-weighted resource utilisation.

    Args:
        resource_log: List of (time, n_busy) tuples.
        run_length: Total simulation time.
        capacity: Resource capacity.

    Returns:
        Utilisation as a fraction (0-1).
    """
    if not resource_log or capacity == 0:
        return 0.0

    # Sort by time
    log = sorted(resource_log)

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
