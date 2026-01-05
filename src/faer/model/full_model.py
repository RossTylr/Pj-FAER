"""Full A&E pathway simulation model.

This module implements the complete patient flow:
Arrival → Triage → Treatment (Resus/Majors/Minors) → Disposition → Departure

For Resus patients: bypass triage, go directly to Resus.
For admitted patients: boarding wait after treatment.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Any, Optional
import itertools

import numpy as np
import simpy

from faer.core.scenario import FullScenario, ArrivalConfig
from faer.core.entities import ArrivalMode, Priority
from faer.model.patient import Patient, Acuity, Disposition


def sample_lognormal(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Sample from lognormal distribution given mean and CV."""
    if cv <= 0 or mean <= 0:
        return mean
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - sigma**2 / 2
    return float(rng.lognormal(mu, sigma))


@dataclass
class AEResources:
    """Container for A&E resources.

    Uses PriorityResource to enable priority-based queuing where
    lower priority values (P1) are served before higher values (P4).
    """

    triage: simpy.PriorityResource
    resus: simpy.PriorityResource
    majors: simpy.PriorityResource
    minors: simpy.PriorityResource


@dataclass
class FullResultsCollector:
    """Collect results for full A&E model."""

    # Patient tracking
    patients: List[Patient] = field(default_factory=list)

    # Counts by acuity
    arrivals_resus: int = 0
    arrivals_majors: int = 0
    arrivals_minors: int = 0

    departures_resus: int = 0
    departures_majors: int = 0
    departures_minors: int = 0

    # Disposition counts
    discharged: int = 0
    admitted: int = 0

    # Resource logs for utilisation: {resource_name: [(time, count), ...]}
    resource_logs: Dict[str, List[tuple]] = field(default_factory=dict)

    # Boarding events: [(patient_id, from_node, duration), ...]
    boarding_events: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        for resource in ["triage", "resus", "majors", "minors", "surgery", "itu", "ward"]:
            self.resource_logs[resource] = [(0.0, 0)]

    def record_arrival(self, patient: Patient) -> None:
        """Record a patient arrival."""
        self.patients.append(patient)
        if patient.acuity == Acuity.RESUS:
            self.arrivals_resus += 1
        elif patient.acuity == Acuity.MAJORS:
            self.arrivals_majors += 1
        else:
            self.arrivals_minors += 1

    def record_departure(self, patient: Patient) -> None:
        """Record a patient departure."""
        if patient.acuity == Acuity.RESUS:
            self.departures_resus += 1
        elif patient.acuity == Acuity.MAJORS:
            self.departures_majors += 1
        else:
            self.departures_minors += 1

        if patient.is_admitted:
            self.admitted += 1
        else:
            self.discharged += 1

    def record_resource_state(self, resource_name: str, time: float, count: int) -> None:
        """Record resource utilisation state change."""
        self.resource_logs[resource_name].append((time, count))

    def record_boarding(self, patient_id: int, from_node: str, duration: float) -> None:
        """Record a boarding event."""
        if duration > 0:
            self.boarding_events.append((patient_id, from_node, duration))

    def compute_metrics(self, run_length: float, scenario: FullScenario) -> Dict:
        """Compute all KPIs from collected data."""
        # Filter out warm-up patients
        valid_patients = [p for p in self.patients
                         if p.arrival_time >= scenario.warm_up
                         and p.departure_time is not None]

        if not valid_patients:
            return self._empty_metrics()

        # Overall metrics
        total_arrivals = len(valid_patients)
        total_departures = len([p for p in valid_patients if p.departure_time])

        # Wait times
        triage_waits = [p.triage_wait for p in valid_patients if p.triage_start]
        treatment_waits = [p.treatment_wait for p in valid_patients if p.treatment_start]
        system_times = [p.system_time for p in valid_patients if p.departure_time]

        # By acuity
        resus_patients = [p for p in valid_patients if p.acuity == Acuity.RESUS]
        majors_patients = [p for p in valid_patients if p.acuity == Acuity.MAJORS]
        minors_patients = [p for p in valid_patients if p.acuity == Acuity.MINORS]

        # P(delay) - proportion who waited for treatment
        p_delay = np.mean([p.treatment_wait > 0 for p in valid_patients]) if valid_patients else 0.0

        # Utilisation
        effective_run = run_length - scenario.warm_up

        metrics = {
            # Counts
            "arrivals": total_arrivals,
            "departures": total_departures,
            "arrivals_resus": len(resus_patients),
            "arrivals_majors": len(majors_patients),
            "arrivals_minors": len(minors_patients),

            # Delays
            "p_delay": float(p_delay),
            "mean_triage_wait": float(np.mean(triage_waits)) if triage_waits else 0.0,
            "mean_treatment_wait": float(np.mean(treatment_waits)) if treatment_waits else 0.0,
            "p95_treatment_wait": float(np.percentile(treatment_waits, 95)) if treatment_waits else 0.0,

            # System time
            "mean_system_time": float(np.mean(system_times)) if system_times else 0.0,
            "p95_system_time": float(np.percentile(system_times, 95)) if system_times else 0.0,

            # Disposition
            "admission_rate": self.admitted / total_departures if total_departures > 0 else 0.0,
            "discharged": self.discharged,
            "admitted": self.admitted,

            # Utilisation
            "util_triage": self._compute_utilisation("triage", effective_run, scenario.n_triage, scenario.warm_up),
            "util_resus": self._compute_utilisation("resus", effective_run, scenario.n_resus_bays, scenario.warm_up),
            "util_majors": self._compute_utilisation("majors", effective_run, scenario.n_majors_bays, scenario.warm_up),
            "util_minors": self._compute_utilisation("minors", effective_run, scenario.n_minors_bays, scenario.warm_up),

            # By acuity - mean treatment wait
            "resus_mean_wait": float(np.mean([p.treatment_wait for p in resus_patients])) if resus_patients else 0.0,
            "majors_mean_wait": float(np.mean([p.treatment_wait for p in majors_patients])) if majors_patients else 0.0,
            "minors_mean_wait": float(np.mean([p.treatment_wait for p in minors_patients])) if minors_patients else 0.0,
        }

        # Boarding metrics
        if self.boarding_events:
            boarding_times = [e[2] for e in self.boarding_events]
            metrics["mean_boarding_time"] = float(np.mean(boarding_times))
            metrics["max_boarding_time"] = float(np.max(boarding_times))
            metrics["total_boarding_events"] = len(self.boarding_events)
            metrics["p_boarding"] = len(self.boarding_events) / total_arrivals if total_arrivals > 0 else 0.0
        else:
            metrics["mean_boarding_time"] = 0.0
            metrics["max_boarding_time"] = 0.0
            metrics["total_boarding_events"] = 0
            metrics["p_boarding"] = 0.0

        return metrics

    def _compute_utilisation(self, resource: str, run_length: float, capacity: int, warm_up: float) -> float:
        """Compute time-weighted utilisation for a resource."""
        if capacity == 0 or run_length <= 0:
            return 0.0

        log = sorted(self.resource_logs.get(resource, []))
        if not log:
            return 0.0

        total_busy_time = 0.0
        for i in range(len(log) - 1):
            t_start, n_busy = log[i]
            t_end = log[i + 1][0]

            # Only count time after warm-up
            if t_end > warm_up:
                effective_start = max(t_start, warm_up)
                total_busy_time += n_busy * (t_end - effective_start)

        # Final segment
        if log:
            t_last, n_last = log[-1]
            if t_last < warm_up + run_length:
                effective_start = max(t_last, warm_up)
                total_busy_time += n_last * (warm_up + run_length - effective_start)

        return total_busy_time / (capacity * run_length)

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            "arrivals": 0, "departures": 0,
            "arrivals_resus": 0, "arrivals_majors": 0, "arrivals_minors": 0,
            "p_delay": 0.0,
            "mean_triage_wait": 0.0, "mean_treatment_wait": 0.0, "p95_treatment_wait": 0.0,
            "mean_system_time": 0.0, "p95_system_time": 0.0,
            "admission_rate": 0.0, "discharged": 0, "admitted": 0,
            "util_triage": 0.0, "util_resus": 0.0, "util_majors": 0.0, "util_minors": 0.0,
            "resus_mean_wait": 0.0, "majors_mean_wait": 0.0, "minors_mean_wait": 0.0,
            "mean_boarding_time": 0.0, "max_boarding_time": 0.0,
            "total_boarding_events": 0, "p_boarding": 0.0,
        }


def assign_acuity(scenario: FullScenario) -> Acuity:
    """Randomly assign patient acuity based on scenario probabilities."""
    r = scenario.rng_acuity.random()
    if r < scenario.p_resus:
        return Acuity.RESUS
    elif r < scenario.p_resus + scenario.p_majors:
        return Acuity.MAJORS
    else:
        return Acuity.MINORS


def assign_priority(acuity: Acuity, rng: np.random.Generator) -> Priority:
    """Assign priority based on acuity with some variability.

    Resus patients are always P1 (immediate).
    Majors patients are P2 (70%) or P3 (30%).
    Minors patients are P3 (60%) or P4 (40%).
    """
    if acuity == Acuity.RESUS:
        return Priority.P1_IMMEDIATE
    elif acuity == Acuity.MAJORS:
        return Priority.P2_VERY_URGENT if rng.random() < 0.7 else Priority.P3_URGENT
    else:  # MINORS
        return Priority.P3_URGENT if rng.random() < 0.6 else Priority.P4_STANDARD


def determine_disposition(patient: Patient, scenario: FullScenario) -> Disposition:
    """Determine patient disposition based on acuity and probabilities."""
    acuity_name = patient.acuity.name.lower()
    p_admit = scenario.get_admission_prob(acuity_name)

    if scenario.rng_disposition.random() < p_admit:
        # For Resus, higher chance of ICU
        if patient.acuity == Acuity.RESUS and scenario.rng_disposition.random() < 0.3:
            return Disposition.ADMIT_ICU
        return Disposition.ADMIT_WARD
    else:
        return Disposition.DISCHARGE


def triage_process(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Triage process for non-Resus patients."""
    with resources.triage.request(priority=patient.priority.value) as req:
        yield req

        triage_start = env.now
        results.record_resource_state("triage", env.now, resources.triage.count)

        # Triage duration
        duration = sample_lognormal(scenario.rng_triage, scenario.triage_mean, scenario.triage_cv)
        yield env.timeout(duration)

        patient.record_triage(triage_start, env.now)

    results.record_resource_state("triage", env.now, resources.triage.count)


def treatment_process(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Treatment process based on acuity."""
    # Select appropriate resource
    if patient.acuity == Acuity.RESUS:
        resource = resources.resus
        resource_name = "resus"
    elif patient.acuity == Acuity.MAJORS:
        resource = resources.majors
        resource_name = "majors"
    else:
        resource = resources.minors
        resource_name = "minors"

    with resource.request(priority=patient.priority.value) as req:
        yield req

        treatment_start = env.now
        results.record_resource_state(resource_name, env.now, resource.count)
        patient.resources_used.append(resource_name)

        # Treatment duration
        mean, cv = scenario.get_treatment_params(resource_name)
        duration = sample_lognormal(scenario.rng_treatment, mean, cv)
        yield env.timeout(duration)

        patient.record_treatment(treatment_start, env.now)

    results.record_resource_state(resource_name, env.now, resource.count)


def boarding_process(
    env: simpy.Environment,
    patient: Patient,
    scenario: FullScenario,
) -> Generator[simpy.Event, None, None]:
    """Boarding process for admitted patients awaiting bed."""
    boarding_start = env.now

    duration = sample_lognormal(scenario.rng_boarding, scenario.boarding_mean, scenario.boarding_cv)
    yield env.timeout(duration)

    patient.record_boarding(boarding_start, env.now)


def patient_journey(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Complete patient journey through A&E."""
    # Record arrival
    results.record_arrival(patient)

    # Resus patients bypass triage
    if patient.acuity != Acuity.RESUS:
        yield from triage_process(env, patient, resources, scenario, results)

    # Treatment
    yield from treatment_process(env, patient, resources, scenario, results)

    # Determine disposition
    disposition = determine_disposition(patient, scenario)

    # Boarding for admitted patients
    if disposition in (Disposition.ADMIT_WARD, Disposition.ADMIT_ICU):
        yield from boarding_process(env, patient, scenario)

    # Departure
    patient.record_departure(env.now, disposition)
    results.record_departure(patient)


def arrival_generator_single(
    env: simpy.Environment,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
    patient_counter: itertools.count,
) -> Generator[simpy.Event, None, None]:
    """Generate patient arrivals (single stream, legacy mode)."""
    # Use first arrival mode's RNG for backwards compatibility
    rng = scenario.rng_arrivals[ArrivalMode.AMBULANCE]

    while True:
        # Exponential inter-arrival time
        iat = rng.exponential(scenario.mean_iat)
        yield env.timeout(iat)

        # Create patient with assigned acuity and priority
        acuity = assign_acuity(scenario)
        priority = assign_priority(acuity, scenario.rng_acuity)
        patient = Patient(
            id=next(patient_counter),
            arrival_time=env.now,
            acuity=acuity,
            priority=priority,
            mode=ArrivalMode.AMBULANCE,
        )

        # Start patient journey
        env.process(patient_journey(env, patient, resources, scenario, results))


def arrival_generator_multistream(
    env: simpy.Environment,
    resources: AEResources,
    config: ArrivalConfig,
    scenario: FullScenario,
    results: FullResultsCollector,
    patient_counter: itertools.count,
) -> Generator[simpy.Event, None, None]:
    """Generate patient arrivals for a single stream (multi-stream mode)."""
    rng = scenario.rng_arrivals[config.mode]

    while True:
        # Get current hour for time-varying rate
        current_hour = int(env.now / 60) % 24
        rate = config.hourly_rates[current_hour]

        if rate <= 0:
            # No arrivals this hour, wait until next hour
            yield env.timeout(60)
            continue

        # Exponential inter-arrival time based on hourly rate
        mean_iat = 60.0 / rate  # minutes
        iat = rng.exponential(mean_iat)
        yield env.timeout(iat)

        # Sample priority from triage mix
        priorities = list(config.triage_mix.keys())
        probs = list(config.triage_mix.values())
        priority_idx = rng.choice(len(priorities), p=probs)
        priority = priorities[priority_idx]

        # Assign acuity based on priority
        if priority == Priority.P1_IMMEDIATE:
            acuity = Acuity.RESUS
        elif priority in (Priority.P2_VERY_URGENT, Priority.P3_URGENT):
            acuity = Acuity.MAJORS if rng.random() < 0.6 else Acuity.MINORS
        else:
            acuity = Acuity.MINORS

        # Create patient
        patient = Patient(
            id=next(patient_counter),
            arrival_time=env.now,
            acuity=acuity,
            priority=priority,
            mode=config.mode,
        )

        results.record_arrival(patient)
        env.process(patient_journey(env, patient, resources, scenario, results))


def run_full_simulation(scenario: FullScenario, use_multistream: bool = False) -> Dict[str, Any]:
    """Execute full A&E simulation.

    Args:
        scenario: FullScenario configuration.
        use_multistream: If True, use multi-stream arrivals from arrival_configs.
                         If False, use single-stream based on arrival_rate.

    Returns:
        Dictionary containing simulation results and metrics.
    """
    # Initialize environment
    env = simpy.Environment()

    # Create resources (PriorityResource for priority-based queuing)
    resources = AEResources(
        triage=simpy.PriorityResource(env, capacity=scenario.n_triage),
        resus=simpy.PriorityResource(env, capacity=scenario.n_resus_bays),
        majors=simpy.PriorityResource(env, capacity=scenario.n_majors_bays),
        minors=simpy.PriorityResource(env, capacity=scenario.n_minors_bays),
    )

    # Initialize results collector
    results = FullResultsCollector()

    # Shared patient counter for unique IDs across all streams
    patient_counter = itertools.count(1)

    # Start arrival generator(s)
    if use_multistream and scenario.arrival_configs:
        # Multi-stream mode: start a generator for each arrival config
        for config in scenario.arrival_configs:
            env.process(arrival_generator_multistream(
                env, resources, config, scenario, results, patient_counter
            ))
    else:
        # Single-stream mode (legacy)
        env.process(arrival_generator_single(
            env, resources, scenario, results, patient_counter
        ))

    # Run simulation
    total_time = scenario.warm_up + scenario.run_length
    env.run(until=total_time)

    # Compute metrics
    metrics = results.compute_metrics(scenario.run_length, scenario)

    # Add scenario info
    metrics["run_length"] = scenario.run_length
    metrics["warm_up"] = scenario.warm_up

    return metrics
