"""Full A&E pathway simulation model.

This module implements the complete patient flow:
Arrival → Triage → ED Bays (priority queuing) → Disposition → Departure

Phase 5: Single ED bay pool with priority queuing (P1-P4).
P1 patients bypass triage, go directly to ED bays with highest priority.
For admitted patients: boarding wait after treatment.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Any, Optional
import itertools

import numpy as np
import simpy

from faer.core.scenario import FullScenario, ArrivalConfig
from faer.core.entities import ArrivalMode, Priority, NodeType, BedState
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

    Phase 5: Single ED bay pool with priority-based queuing.
    Lower priority values (P1) are served before higher values (P4).
    Phase 5b: Handover bays for ambulance/helicopter arrivals.
    Phase 5c: Fleet resources for ambulances and helicopters.
    """

    triage: simpy.PriorityResource
    ed_bays: simpy.PriorityResource  # Single pool replaces resus/majors/minors
    handover_bays: simpy.Resource  # Phase 5b: FIFO for ambulances (not priority)
    ambulance_fleet: simpy.Resource  # Phase 5c: Ambulance vehicles
    helicopter_fleet: simpy.Resource  # Phase 5c: Helicopter vehicles


@dataclass
class FullResultsCollector:
    """Collect results for full A&E model.

    Phase 5: Tracks single ED bay pool and priority-based metrics.
    """

    # Patient tracking
    patients: List[Patient] = field(default_factory=list)

    # Counts by priority
    arrivals_by_priority: Dict[Priority, int] = field(default_factory=dict)
    departures_by_priority: Dict[Priority, int] = field(default_factory=dict)

    # Legacy acuity counts (for backwards compatibility)
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

    # Bed state log (Phase 5e): [(node, bed_id, state, time), ...]
    bed_state_log: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        for resource in ["triage", "ed_bays", "handover", "surgery", "itu", "ward",
                        "ambulance_fleet", "helicopter_fleet"]:
            self.resource_logs[resource] = [(0.0, 0)]
        for p in Priority:
            self.arrivals_by_priority[p] = 0
            self.departures_by_priority[p] = 0

    def record_arrival(self, patient: Patient) -> None:
        """Record a patient arrival."""
        self.patients.append(patient)
        # Track by priority
        self.arrivals_by_priority[patient.priority] = self.arrivals_by_priority.get(patient.priority, 0) + 1
        # Legacy acuity tracking for backwards compatibility
        if patient.acuity == Acuity.RESUS:
            self.arrivals_resus += 1
        elif patient.acuity == Acuity.MAJORS:
            self.arrivals_majors += 1
        else:
            self.arrivals_minors += 1

    def record_departure(self, patient: Patient) -> None:
        """Record a patient departure."""
        # Track by priority
        self.departures_by_priority[patient.priority] = self.departures_by_priority.get(patient.priority, 0) + 1
        # Legacy acuity tracking
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

    def record_bed_state_change(self, node: NodeType, bed_id: int, state: BedState, time: float) -> None:
        """Record a bed state change (Phase 5e)."""
        self.bed_state_log.append((node, bed_id, state, time))

    def compute_metrics(self, run_length: float, scenario: FullScenario) -> Dict:
        """Compute all KPIs from collected data.

        Phase 5: Metrics for single ED bay pool with priority-based queuing.
        """
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

        # By priority (Phase 5)
        p1_patients = [p for p in valid_patients if p.priority == Priority.P1_IMMEDIATE]
        p2_patients = [p for p in valid_patients if p.priority == Priority.P2_VERY_URGENT]
        p3_patients = [p for p in valid_patients if p.priority == Priority.P3_URGENT]
        p4_patients = [p for p in valid_patients if p.priority == Priority.P4_STANDARD]

        # By acuity (legacy)
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

            # Priority counts
            "arrivals_P1": len(p1_patients),
            "arrivals_P2": len(p2_patients),
            "arrivals_P3": len(p3_patients),
            "arrivals_P4": len(p4_patients),

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

            # Utilisation - Phase 5: single ED bay pool
            "util_triage": self._compute_utilisation("triage", effective_run, scenario.n_triage, scenario.warm_up),
            "util_ed_bays": self._compute_utilisation("ed_bays", effective_run, scenario.n_ed_bays, scenario.warm_up),

            # By priority - mean treatment wait (Phase 5)
            "P1_mean_wait": float(np.mean([p.treatment_wait for p in p1_patients])) if p1_patients else 0.0,
            "P2_mean_wait": float(np.mean([p.treatment_wait for p in p2_patients])) if p2_patients else 0.0,
            "P3_mean_wait": float(np.mean([p.treatment_wait for p in p3_patients])) if p3_patients else 0.0,
            "P4_mean_wait": float(np.mean([p.treatment_wait for p in p4_patients])) if p4_patients else 0.0,

            # By acuity - mean treatment wait (legacy)
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

        # Handover metrics (Phase 5b)
        handover_delays = [p.handover_delay for p in valid_patients
                         if p.handover_queue_start is not None]
        if handover_delays:
            metrics["mean_handover_delay"] = float(np.mean(handover_delays))
            metrics["max_handover_delay"] = float(np.max(handover_delays))
            metrics["p95_handover_delay"] = float(np.percentile(handover_delays, 95))
            metrics["handover_arrivals"] = len(handover_delays)
        else:
            metrics["mean_handover_delay"] = 0.0
            metrics["max_handover_delay"] = 0.0
            metrics["p95_handover_delay"] = 0.0
            metrics["handover_arrivals"] = 0

        # Handover utilisation
        effective_run = run_length - scenario.warm_up
        metrics["util_handover"] = self._compute_utilisation(
            "handover", effective_run, scenario.n_handover_bays, scenario.warm_up
        )

        # Fleet utilisation (Phase 5c)
        metrics["util_ambulance_fleet"] = self._compute_utilisation(
            "ambulance_fleet", effective_run, scenario.n_ambulances, scenario.warm_up
        )
        metrics["util_helicopter_fleet"] = self._compute_utilisation(
            "helicopter_fleet", effective_run, scenario.n_helicopters, scenario.warm_up
        )

        # Bed state metrics (Phase 5e)
        bed_state_metrics = self._compute_bed_state_metrics(
            scenario.warm_up + run_length, scenario.warm_up
        )
        metrics.update(bed_state_metrics)

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

    def _compute_bed_state_metrics(self, end_time: float, warm_up: float) -> Dict:
        """Compute time-weighted bed state percentages (Phase 5e)."""
        metrics = {}

        # Group state log by node
        from collections import defaultdict
        node_states = defaultdict(list)
        for node, bed_id, state, time in self.bed_state_log:
            node_states[node].append((time, bed_id, state))

        # Compute state times for each tracked node
        for node in [NodeType.ED_BAYS]:  # Can extend to ITU, WARD later
            state_log = sorted(node_states.get(node, []))

            if not state_log:
                metrics[f"{node.name}_pct_occupied"] = 0.0
                metrics[f"{node.name}_pct_blocked"] = 0.0
                metrics[f"{node.name}_pct_cleaning"] = 0.0
                continue

            # Track cumulative time per state
            state_times = {BedState.EMPTY: 0.0, BedState.OCCUPIED: 0.0,
                          BedState.BLOCKED: 0.0, BedState.CLEANING: 0.0}

            # Track current state per bed
            bed_states = {}  # bed_id -> (state, start_time)

            for time, bed_id, state in state_log:
                if bed_id in bed_states:
                    prev_state, prev_time = bed_states[bed_id]
                    # Count time in previous state (only after warm-up)
                    if time > warm_up:
                        effective_start = max(prev_time, warm_up)
                        state_times[prev_state] += (time - effective_start)
                bed_states[bed_id] = (state, time)

            # Final segment for each bed
            for bed_id, (state, start_time) in bed_states.items():
                if start_time < end_time and end_time > warm_up:
                    effective_start = max(start_time, warm_up)
                    state_times[state] += (end_time - effective_start)

            # Compute percentages
            total_time = sum(state_times.values())
            if total_time > 0:
                metrics[f"{node.name}_pct_occupied"] = state_times[BedState.OCCUPIED] / total_time
                metrics[f"{node.name}_pct_blocked"] = state_times[BedState.BLOCKED] / total_time
                metrics[f"{node.name}_pct_cleaning"] = state_times[BedState.CLEANING] / total_time
            else:
                metrics[f"{node.name}_pct_occupied"] = 0.0
                metrics[f"{node.name}_pct_blocked"] = 0.0
                metrics[f"{node.name}_pct_cleaning"] = 0.0

        return metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            "arrivals": 0, "departures": 0,
            "arrivals_resus": 0, "arrivals_majors": 0, "arrivals_minors": 0,
            "arrivals_P1": 0, "arrivals_P2": 0, "arrivals_P3": 0, "arrivals_P4": 0,
            "p_delay": 0.0,
            "mean_triage_wait": 0.0, "mean_treatment_wait": 0.0, "p95_treatment_wait": 0.0,
            "mean_system_time": 0.0, "p95_system_time": 0.0,
            "admission_rate": 0.0, "discharged": 0, "admitted": 0,
            "util_triage": 0.0, "util_ed_bays": 0.0, "util_handover": 0.0,
            "util_ambulance_fleet": 0.0, "util_helicopter_fleet": 0.0,
            "P1_mean_wait": 0.0, "P2_mean_wait": 0.0, "P3_mean_wait": 0.0, "P4_mean_wait": 0.0,
            "resus_mean_wait": 0.0, "majors_mean_wait": 0.0, "minors_mean_wait": 0.0,
            "mean_boarding_time": 0.0, "max_boarding_time": 0.0,
            "total_boarding_events": 0, "p_boarding": 0.0,
            "mean_handover_delay": 0.0, "max_handover_delay": 0.0,
            "p95_handover_delay": 0.0, "handover_arrivals": 0,
            "ED_BAYS_pct_occupied": 0.0, "ED_BAYS_pct_blocked": 0.0, "ED_BAYS_pct_cleaning": 0.0,
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
    """Treatment process in single ED bay pool.

    Phase 5: All patients use same ED bay pool with priority queuing.
    P1 patients are served before P4 patients.
    """
    resource = resources.ed_bays
    resource_name = "ed_bays"

    with resource.request(priority=patient.priority.value) as req:
        yield req

        treatment_start = env.now
        patient.current_node = NodeType.ED_BAYS
        results.record_resource_state(resource_name, env.now, resource.count)
        patient.resources_used.append(resource_name)

        # Treatment duration - single service time for all priorities
        mean, cv = scenario.get_treatment_params()
        duration = sample_lognormal(scenario.rng_treatment, mean, cv)
        yield env.timeout(duration)

        patient.record_treatment(treatment_start, env.now)

    results.record_resource_state(resource_name, env.now, resource.count)


def handover_process(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, Any, Any]:
    """Handover process for ambulance/helicopter arrivals.

    Phase 5b: Returns the handover request object (NOT released until ED bay acquired).
    This creates the feedback loop: full ED → slow handover release → ambulance queues.

    Returns:
        The handover request object to be released after ED bay acquisition.
    """
    handover_queue_start = env.now

    # Request handover bay (FIFO queue)
    handover_req = resources.handover_bays.request()
    yield handover_req

    handover_start = env.now
    results.record_resource_state("handover", env.now, resources.handover_bays.count)

    # Handover process duration
    duration = sample_lognormal(scenario.rng_handover, scenario.handover_time_mean, scenario.handover_time_cv)
    yield env.timeout(duration)

    handover_end = env.now
    patient.record_handover(handover_queue_start, handover_start, handover_end)

    # Return the request - it will be released AFTER ED bay acquisition
    return handover_req


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


def vehicle_mission(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
    fleet: simpy.Resource,
    fleet_name: str,
    turnaround_time: float,
) -> Generator[simpy.Event, None, None]:
    """Vehicle mission process for ambulance/helicopter transport.

    Phase 5c: Vehicle delivers patient then becomes unavailable for turnaround.

    The vehicle:
    1. Is occupied during patient transport (starts at mission start)
    2. Delivers patient to hospital (patient journey starts)
    3. Remains unavailable during turnaround period
    4. Becomes available for next mission

    Args:
        env: SimPy environment.
        patient: Patient being transported.
        resources: A&E resources.
        scenario: Scenario configuration.
        results: Results collector.
        fleet: The fleet resource (ambulance_fleet or helicopter_fleet).
        fleet_name: Name for logging ("ambulance_fleet" or "helicopter_fleet").
        turnaround_time: Time vehicle is unavailable after delivery.
    """
    with fleet.request() as fleet_req:
        yield fleet_req

        # Vehicle is now on mission
        results.record_resource_state(fleet_name, env.now, fleet.count)

        # Start patient journey (runs concurrently)
        # Note: patient_journey calls record_arrival internally
        env.process(patient_journey(env, patient, resources, scenario, results))

        # Vehicle turnaround (unavailable after delivery)
        yield env.timeout(turnaround_time)

    # Vehicle now available for next mission
    results.record_resource_state(fleet_name, env.now, fleet.count)


def patient_journey(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Complete patient journey through A&E.

    Phase 5: P1 (immediate) patients bypass triage and go directly to ED bays.
    All other patients go through triage first.

    Phase 5b: Ambulance/helicopter arrivals go through handover first.
    Handover bay is held until ED bay is acquired (feedback mechanism).
    Walk-ins bypass handover entirely.
    """
    # Record arrival
    results.record_arrival(patient)

    # Phase 5b: Handover process for ambulance/helicopter arrivals
    handover_req = None
    if patient.mode in (ArrivalMode.AMBULANCE, ArrivalMode.HELICOPTER):
        # Handover process - returns the request to be released later
        handover_gen = handover_process(env, patient, resources, scenario, results)
        handover_req = yield from handover_gen

    # P1 patients bypass triage (immediate resuscitation)
    if patient.priority != Priority.P1_IMMEDIATE:
        yield from triage_process(env, patient, resources, scenario, results)

    # Treatment in ED bays (priority queuing)
    # NOTE: Handover bay is still held at this point!
    resource = resources.ed_bays
    resource_name = "ed_bays"

    with resource.request(priority=patient.priority.value) as req:
        yield req

        # CRITICAL: NOW release handover bay (feedback mechanism)
        if handover_req is not None:
            resources.handover_bays.release(handover_req)
            patient.record_handover_release(env.now)
            results.record_resource_state("handover", env.now, resources.handover_bays.count)

        treatment_start = env.now
        patient.current_node = NodeType.ED_BAYS
        results.record_resource_state(resource_name, env.now, resource.count)
        patient.resources_used.append(resource_name)

        # Phase 5e: Track bed state - OCCUPIED during treatment
        bed_id = patient.id  # Use patient ID as proxy for bed ID
        results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.OCCUPIED, env.now)

        # Treatment duration - single service time for all priorities
        mean, cv = scenario.get_treatment_params()
        duration = sample_lognormal(scenario.rng_treatment, mean, cv)
        yield env.timeout(duration)

        patient.record_treatment(treatment_start, env.now)

        # Determine disposition using routing matrix
        next_node = scenario.get_next_node(NodeType.ED_BAYS, patient.priority)

        # Convert node to disposition
        if next_node == NodeType.EXIT:
            disposition = Disposition.DISCHARGE
        elif next_node == NodeType.ITU:
            disposition = Disposition.ADMIT_ICU
        else:
            disposition = Disposition.ADMIT_WARD

        # Boarding for admitted patients (Phase 5e: bed is BLOCKED during boarding)
        if disposition in (Disposition.ADMIT_WARD, Disposition.ADMIT_ICU):
            results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.BLOCKED, env.now)
            yield from boarding_process(env, patient, scenario)

        # Phase 5e: Bed goes to CLEANING state
        results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.CLEANING, env.now)

        # Bed turnaround/cleaning time (patient departs but bed still cleaning)
        yield env.timeout(scenario.bed_turnaround_mins)

        # Bed now EMPTY (ready for next patient)
        results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.EMPTY, env.now)

    results.record_resource_state(resource_name, env.now, resource.count)

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
    """Generate patient arrivals for a single stream (multi-stream mode).

    Phase 5c: Ambulance/helicopter arrivals use fleet resources with turnaround.
    Walk-ins go directly to patient journey.
    """
    rng = scenario.rng_arrivals[config.mode]

    # Determine fleet resource and turnaround for this mode (Phase 5c)
    if config.mode == ArrivalMode.AMBULANCE:
        fleet = resources.ambulance_fleet
        fleet_name = "ambulance_fleet"
        turnaround = scenario.ambulance_turnaround_mins
    elif config.mode == ArrivalMode.HELICOPTER:
        fleet = resources.helicopter_fleet
        fleet_name = "helicopter_fleet"
        turnaround = scenario.helicopter_turnaround_mins
    else:
        fleet = None
        fleet_name = None
        turnaround = 0

    while True:
        # Get current hour for time-varying rate
        current_hour = int(env.now / 60) % 24
        base_rate = config.hourly_rates[current_hour]

        # Phase 5d: Apply demand multipliers
        rate_multiplier = scenario.get_rate_multiplier(config.mode)
        rate = base_rate * rate_multiplier

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

        # Phase 5c: Use fleet resources for ambulance/helicopter
        if fleet is not None:
            # Start vehicle mission (includes patient journey)
            env.process(vehicle_mission(
                env, patient, resources, scenario, results,
                fleet, fleet_name, turnaround
            ))
        else:
            # Walk-ins go directly
            results.record_arrival(patient)
            env.process(patient_journey(env, patient, resources, scenario, results))


def run_full_simulation(scenario: FullScenario, use_multistream: bool = False) -> Dict[str, Any]:
    """Execute full A&E simulation.

    Phase 5: Single ED bay pool with priority queuing.

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
    # Phase 5: Single ED bay pool
    # Phase 5b: Handover bays (FIFO, not priority)
    # Phase 5c: Fleet resources for ambulances and helicopters
    resources = AEResources(
        triage=simpy.PriorityResource(env, capacity=scenario.n_triage),
        ed_bays=simpy.PriorityResource(env, capacity=scenario.n_ed_bays),
        handover_bays=simpy.Resource(env, capacity=scenario.n_handover_bays),
        ambulance_fleet=simpy.Resource(env, capacity=scenario.n_ambulances),
        helicopter_fleet=simpy.Resource(env, capacity=scenario.n_helicopters),
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
