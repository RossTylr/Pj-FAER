"""Full A&E pathway simulation model.

This module implements the complete patient flow:
Arrival → Triage → ED Bays (priority queuing) → Disposition → Departure

Phase 5: Single ED bay pool with priority queuing (P1-P4).
P1 patients bypass triage, go directly to ED bays with highest priority.
For admitted patients: boarding wait after treatment.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Any, Optional, TYPE_CHECKING
import itertools

import numpy as np
import simpy

from faer.core.scenario import FullScenario, ArrivalConfig
from faer.core.entities import (
    ArrivalMode, Priority, NodeType, BedState, DiagnosticType,
    TransferType, TransferDestination,
)
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
    Phase 7: Diagnostic resources (CT, X-ray, Bloods).
    Phase 7: Transfer resources (land ambulance, helicopter).
    """

    triage: simpy.PriorityResource
    ed_bays: simpy.PriorityResource  # Single pool replaces resus/majors/minors
    handover_bays: simpy.Resource  # Phase 5b: FIFO for ambulances (not priority)
    ambulance_fleet: simpy.Resource  # Phase 5c: Ambulance vehicles
    helicopter_fleet: simpy.Resource  # Phase 5c: Helicopter vehicles

    # Phase 7: Diagnostic resources (priority queuing for CT/X-ray/Bloods)
    diagnostics: Dict[DiagnosticType, simpy.PriorityResource] = field(default_factory=dict)

    # Phase 7: Transfer resources
    transfer_ambulances: Optional[simpy.Resource] = None
    transfer_helicopters: Optional[simpy.Resource] = None


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

    # Phase 7: Diagnostic tracking
    # diagnostic_waits[diag_type] = [wait_time, ...]
    diagnostic_waits: Dict[DiagnosticType, List[float]] = field(default_factory=dict)
    # diagnostic_turnarounds[diag_type] = [total_turnaround_time, ...]
    diagnostic_turnarounds: Dict[DiagnosticType, List[float]] = field(default_factory=dict)

    # Phase 7: Transfer tracking
    # transfer_waits[transfer_type] = [wait_time, ...]
    transfer_waits: Dict[TransferType, List[float]] = field(default_factory=dict)
    # transfer_total_times[transfer_type] = [total_time, ...]
    transfer_total_times: Dict[TransferType, List[float]] = field(default_factory=dict)
    # Count of transfers by destination
    transfers_by_destination: Dict[TransferDestination, int] = field(default_factory=dict)

    def __post_init__(self):
        for resource in ["triage", "ed_bays", "handover", "surgery", "itu", "ward",
                        "ambulance_fleet", "helicopter_fleet"]:
            self.resource_logs[resource] = [(0.0, 0)]
        for p in Priority:
            self.arrivals_by_priority[p] = 0
            self.departures_by_priority[p] = 0
        # Initialize diagnostic tracking (Phase 7)
        for diag_type in DiagnosticType:
            self.diagnostic_waits[diag_type] = []
            self.diagnostic_turnarounds[diag_type] = []
            self.resource_logs[f"diag_{diag_type.name}"] = [(0.0, 0)]
        # Initialize transfer tracking (Phase 7)
        for transfer_type in TransferType:
            self.transfer_waits[transfer_type] = []
            self.transfer_total_times[transfer_type] = []
            self.resource_logs[f"transfer_{transfer_type.name}"] = [(0.0, 0)]
        for dest in TransferDestination:
            self.transfers_by_destination[dest] = 0

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

    def record_diagnostic_wait(self, diag_type: DiagnosticType, wait_time: float) -> None:
        """Record diagnostic queue wait time (Phase 7)."""
        self.diagnostic_waits[diag_type].append(wait_time)

    def record_diagnostic_turnaround(self, diag_type: DiagnosticType, turnaround: float) -> None:
        """Record total diagnostic turnaround time (Phase 7)."""
        self.diagnostic_turnarounds[diag_type].append(turnaround)

    def record_transfer_wait(self, transfer_type: TransferType, wait_time: float) -> None:
        """Record transfer vehicle wait time (Phase 7)."""
        self.transfer_waits[transfer_type].append(wait_time)

    def record_transfer_total(self, transfer_type: TransferType, total_time: float) -> None:
        """Record total transfer process time (Phase 7)."""
        self.transfer_total_times[transfer_type].append(total_time)

    def record_transfer_destination(self, destination: TransferDestination) -> None:
        """Record transfer destination (Phase 7)."""
        self.transfers_by_destination[destination] = self.transfers_by_destination.get(destination, 0) + 1

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

        # Diagnostic metrics (Phase 7)
        for diag_type in DiagnosticType:
            waits = self.diagnostic_waits.get(diag_type, [])
            turnarounds = self.diagnostic_turnarounds.get(diag_type, [])

            if waits:
                metrics[f"mean_wait_{diag_type.name}"] = float(np.mean(waits))
                metrics[f"p95_wait_{diag_type.name}"] = float(np.percentile(waits, 95))
            else:
                metrics[f"mean_wait_{diag_type.name}"] = 0.0
                metrics[f"p95_wait_{diag_type.name}"] = 0.0

            if turnarounds:
                metrics[f"mean_turnaround_{diag_type.name}"] = float(np.mean(turnarounds))
            else:
                metrics[f"mean_turnaround_{diag_type.name}"] = 0.0

            # Diagnostic utilisation
            diag_config = scenario.diagnostic_configs.get(diag_type)
            if diag_config and diag_config.enabled:
                metrics[f"util_{diag_type.name}"] = self._compute_utilisation(
                    f"diag_{diag_type.name}", effective_run, diag_config.capacity, scenario.warm_up
                )
            else:
                metrics[f"util_{diag_type.name}"] = 0.0

            # Count of patients who had this diagnostic
            metrics[f"count_{diag_type.name}"] = len(waits)

        # Transfer metrics (Phase 7)
        total_transfers = 0
        for transfer_type in TransferType:
            waits = self.transfer_waits.get(transfer_type, [])
            totals = self.transfer_total_times.get(transfer_type, [])

            if waits:
                metrics[f"mean_transfer_wait_{transfer_type.name}"] = float(np.mean(waits))
                metrics[f"p95_transfer_wait_{transfer_type.name}"] = float(np.percentile(waits, 95))
            else:
                metrics[f"mean_transfer_wait_{transfer_type.name}"] = 0.0
                metrics[f"p95_transfer_wait_{transfer_type.name}"] = 0.0

            if totals:
                metrics[f"mean_transfer_total_{transfer_type.name}"] = float(np.mean(totals))
            else:
                metrics[f"mean_transfer_total_{transfer_type.name}"] = 0.0

            metrics[f"count_transfer_{transfer_type.name}"] = len(waits)
            total_transfers += len(waits)

        # Transfer resource utilisation
        if scenario.transfer_config.enabled:
            metrics["util_transfer_ambulances"] = self._compute_utilisation(
                f"transfer_{TransferType.LAND_AMBULANCE.name}",
                effective_run,
                scenario.transfer_config.n_transfer_ambulances,
                scenario.warm_up
            )
            metrics["util_transfer_helicopters"] = self._compute_utilisation(
                f"transfer_{TransferType.HELICOPTER.name}",
                effective_run,
                scenario.transfer_config.n_transfer_helicopters,
                scenario.warm_up
            )
        else:
            metrics["util_transfer_ambulances"] = 0.0
            metrics["util_transfer_helicopters"] = 0.0

        # Transfer counts by destination
        metrics["total_transfers"] = total_transfers
        for dest in TransferDestination:
            metrics[f"transfers_to_{dest.name}"] = self.transfers_by_destination.get(dest, 0)

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
        metrics = {
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
        # Add diagnostic metrics (Phase 7)
        for diag_type in DiagnosticType:
            metrics[f"mean_wait_{diag_type.name}"] = 0.0
            metrics[f"p95_wait_{diag_type.name}"] = 0.0
            metrics[f"mean_turnaround_{diag_type.name}"] = 0.0
            metrics[f"util_{diag_type.name}"] = 0.0
            metrics[f"count_{diag_type.name}"] = 0
        # Add transfer metrics (Phase 7)
        for transfer_type in TransferType:
            metrics[f"mean_transfer_wait_{transfer_type.name}"] = 0.0
            metrics[f"p95_transfer_wait_{transfer_type.name}"] = 0.0
            metrics[f"mean_transfer_total_{transfer_type.name}"] = 0.0
            metrics[f"count_transfer_{transfer_type.name}"] = 0
        metrics["util_transfer_ambulances"] = 0.0
        metrics["util_transfer_helicopters"] = 0.0
        metrics["total_transfers"] = 0
        for dest in TransferDestination:
            metrics[f"transfers_to_{dest.name}"] = 0
        return metrics


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


def determine_required_diagnostics(
    patient: Patient,
    scenario: FullScenario,
) -> List[DiagnosticType]:
    """Determine which diagnostics this patient needs based on priority (Phase 7).

    Args:
        patient: The patient to assess.
        scenario: The scenario configuration.

    Returns:
        List of DiagnosticType enums for required diagnostics.
    """
    required = []

    for diag_type, config in scenario.diagnostic_configs.items():
        if not config.enabled:
            continue

        prob = config.probability_by_priority.get(patient.priority, 0)
        if scenario.rng_diagnostics.random() < prob:
            required.append(diag_type)

    return required


def diagnostic_process(
    env: simpy.Environment,
    patient: Patient,
    diag_type: DiagnosticType,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Process a single diagnostic journey (Phase 7).

    Patient KEEPS their ED bay while going for diagnostic.
    This is the key realism: bay is blocked while patient is in CT.

    Args:
        env: SimPy environment.
        patient: The patient needing diagnostic.
        diag_type: Type of diagnostic (CT, X-ray, Bloods).
        resources: A&E resources including diagnostics.
        scenario: Scenario configuration.
        results: Results collector.
    """
    if diag_type not in resources.diagnostics:
        return

    diag_resource = resources.diagnostics[diag_type]
    diag_config = scenario.diagnostic_configs[diag_type]
    resource_name = f"diag_{diag_type.name}"

    # Record leaving bay for diagnostic
    patient.record_diagnostic_event(diag_type, 'journey_start', env.now)

    # Queue for diagnostic (patient still "owns" ED bay)
    patient.record_diagnostic_event(diag_type, 'queue_start', env.now)

    with diag_resource.request(priority=patient.priority.value) as diag_req:
        yield diag_req

        # Got the diagnostic resource
        patient.record_diagnostic_event(diag_type, 'start', env.now)
        results.record_resource_state(resource_name, env.now, diag_resource.count)

        # Diagnostic process time
        process_time = sample_lognormal(
            scenario.rng_diagnostics,
            diag_config.process_time_mean,
            diag_config.process_time_cv
        )
        yield env.timeout(process_time)

        patient.record_diagnostic_event(diag_type, 'end', env.now)

    # Released diagnostic resource
    results.record_resource_state(resource_name, env.now, diag_resource.count)

    # Patient returns to bay
    patient.record_diagnostic_event(diag_type, 'return_to_bay', env.now)

    # Wait for results (turnaround time) - patient back in bay
    if diag_config.turnaround_time_mean > 0:
        turnaround = sample_lognormal(
            scenario.rng_diagnostics,
            diag_config.turnaround_time_mean,
            diag_config.turnaround_time_cv
        )
        yield env.timeout(turnaround)
        patient.record_diagnostic_event(diag_type, 'results_available', env.now)

    # Mark diagnostic complete
    patient.complete_diagnostic(diag_type)

    # Record metrics
    wait_time = patient.get_diagnostic_wait(diag_type)
    results.record_diagnostic_wait(diag_type, wait_time)

    # Total turnaround = journey_start to results_available (or end if no turnaround)
    journey_start = patient.diagnostic_timestamps.get(f'{diag_type.name}_journey_start', env.now)
    total_turnaround = env.now - journey_start
    results.record_diagnostic_turnaround(diag_type, total_turnaround)


def determine_transfer_required(
    patient: Patient,
    scenario: FullScenario,
) -> bool:
    """Determine if patient requires inter-facility transfer (Phase 7).

    Args:
        patient: The patient to assess.
        scenario: The scenario configuration.

    Returns:
        True if patient requires transfer, False otherwise.
    """
    if not scenario.transfer_config.enabled:
        return False

    prob = scenario.transfer_config.probability_by_priority.get(patient.priority, 0)
    return scenario.rng_transfer.random() < prob


def select_transfer_destination(
    patient: Patient,
    scenario: FullScenario,
) -> TransferDestination:
    """Select transfer destination based on patient acuity (Phase 7).

    More acute patients tend to go to specialist centres.

    Args:
        patient: The patient being transferred.
        scenario: The scenario configuration.

    Returns:
        The TransferDestination for this patient.
    """
    rng = scenario.rng_transfer

    if patient.acuity == Acuity.RESUS:
        # P1/Resus patients typically go to specialist centres
        destinations = [
            TransferDestination.MAJOR_TRAUMA_CENTRE,
            TransferDestination.NEUROSURGERY,
            TransferDestination.CARDIAC_CENTRE,
            TransferDestination.PAEDIATRIC_ICU,
        ]
        probs = [0.4, 0.25, 0.25, 0.1]
    elif patient.acuity == Acuity.MAJORS:
        # Majors typically need regional ICU or specialist
        destinations = [
            TransferDestination.REGIONAL_ICU,
            TransferDestination.CARDIAC_CENTRE,
            TransferDestination.BURNS_UNIT,
            TransferDestination.MAJOR_TRAUMA_CENTRE,
        ]
        probs = [0.5, 0.2, 0.15, 0.15]
    else:
        # Minors - less common transfers, usually regional
        destinations = [
            TransferDestination.REGIONAL_ICU,
            TransferDestination.BURNS_UNIT,
        ]
        probs = [0.7, 0.3]

    idx = rng.choice(len(destinations), p=probs)
    return destinations[idx]


def select_transfer_type(
    patient: Patient,
    scenario: FullScenario,
) -> TransferType:
    """Select transfer vehicle type based on acuity and configuration (Phase 7).

    P1 patients have higher chance of helicopter transfer.

    Args:
        patient: The patient being transferred.
        scenario: The scenario configuration.

    Returns:
        The TransferType for this patient.
    """
    rng = scenario.rng_transfer
    config = scenario.transfer_config

    # P1 patients may get helicopter
    if patient.priority == Priority.P1_IMMEDIATE:
        if rng.random() < config.helicopter_proportion_p1:
            return TransferType.HELICOPTER

    # Otherwise, decide between land ambulance and critical care
    # P1/P2 more likely to get critical care ambulance
    if patient.priority in (Priority.P1_IMMEDIATE, Priority.P2_VERY_URGENT):
        if rng.random() < 0.4:
            return TransferType.CRITICAL_CARE

    return TransferType.LAND_AMBULANCE


def transfer_process(
    env: simpy.Environment,
    patient: Patient,
    resources: AEResources,
    scenario: FullScenario,
    results: FullResultsCollector,
) -> Generator[simpy.Event, None, None]:
    """Process inter-facility transfer (Phase 7).

    Transfer sequence:
    1. Decision made - determine destination and vehicle type
    2. Request vehicle (wait for availability)
    3. Vehicle arrives
    4. Patient departs (releases ED bay)

    Patient keeps ED bay until vehicle arrives and transfer begins.

    Args:
        env: SimPy environment.
        patient: The patient being transferred.
        resources: A&E resources including transfer fleet.
        scenario: Scenario configuration.
        results: Results collector.
    """
    config = scenario.transfer_config
    decision_time = env.now

    # Select destination and transfer type
    destination = select_transfer_destination(patient, scenario)
    transfer_type = select_transfer_type(patient, scenario)

    patient.requires_transfer = True
    patient.transfer_type = transfer_type
    patient.transfer_destination = destination
    patient.transfer_decision_time = decision_time

    # Time from decision to actually requesting vehicle
    decision_to_request = sample_lognormal(
        scenario.rng_transfer,
        config.decision_to_request_mean,
        0.3  # CV
    )
    yield env.timeout(decision_to_request)

    requested_time = env.now
    patient.transfer_requested_time = requested_time

    # Select appropriate fleet resource
    if transfer_type == TransferType.HELICOPTER:
        fleet = resources.transfer_helicopters
        fleet_name = f"transfer_{TransferType.HELICOPTER.name}"
        wait_mean = config.helicopter_wait_mean
        transfer_duration_mean = config.helicopter_transfer_time_mean
    else:
        # Both LAND_AMBULANCE and CRITICAL_CARE use ambulance fleet
        fleet = resources.transfer_ambulances
        fleet_name = f"transfer_{TransferType.LAND_AMBULANCE.name}"
        wait_mean = config.land_ambulance_wait_mean
        transfer_duration_mean = config.land_transfer_time_mean

    if fleet is None:
        # Transfer not available, patient stays
        return

    # Wait for transfer vehicle
    with fleet.request() as req:
        yield req

        vehicle_arrived_time = env.now
        patient.transfer_vehicle_arrived_time = vehicle_arrived_time
        results.record_resource_state(fleet_name, env.now, fleet.count)

        # Vehicle response time (already waited in queue, add any additional time)
        # The wait in queue IS the response time
        wait_time = vehicle_arrived_time - requested_time
        results.record_transfer_wait(transfer_type, wait_time)

        # Transfer process (loading + travel)
        transfer_duration = sample_lognormal(
            scenario.rng_transfer,
            transfer_duration_mean,
            0.3  # CV
        )
        yield env.timeout(transfer_duration)

        departed_time = env.now
        patient.transfer_departed_time = departed_time

    # Vehicle released
    results.record_resource_state(fleet_name, env.now, fleet.count)

    # Record full transfer
    patient.record_transfer(
        transfer_type, destination, decision_time,
        requested_time, vehicle_arrived_time, departed_time
    )

    # Record metrics
    total_time = departed_time - decision_time
    results.record_transfer_total(transfer_type, total_time)
    results.record_transfer_destination(destination)


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

    Phase 7: Diagnostics loop - patient leaves bay for CT/X-ray/Bloods but
    keeps the bay (blocking). Bay remains occupied during diagnostic journey.
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

        # Phase 7: Determine required diagnostics AFTER initial assessment
        # Typically done early in treatment when clinician assesses patient
        patient.diagnostics_required = determine_required_diagnostics(patient, scenario)

        # Phase 7: Process all required diagnostics
        # Patient keeps their bay while going for diagnostics
        for diag_type in patient.diagnostics_required:
            yield from diagnostic_process(
                env, patient, diag_type, resources, scenario, results
            )

        # Treatment duration - single service time for all priorities
        # This represents the remaining treatment after diagnostics
        mean, cv = scenario.get_treatment_params()
        duration = sample_lognormal(scenario.rng_treatment, mean, cv)
        yield env.timeout(duration)

        patient.record_treatment(treatment_start, env.now)

        # Phase 7: Check if patient requires inter-facility transfer
        if determine_transfer_required(patient, scenario):
            # Transfer process - patient keeps bay until vehicle arrives
            disposition = Disposition.TRANSFER
            results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.BLOCKED, env.now)
            yield from transfer_process(env, patient, resources, scenario, results)
        else:
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
    # Phase 7: Diagnostic resources
    diagnostic_resources = {}
    for diag_type, config in scenario.diagnostic_configs.items():
        if config.enabled:
            diagnostic_resources[diag_type] = simpy.PriorityResource(
                env, capacity=config.capacity
            )

    # Phase 7: Transfer resources
    transfer_ambulances = None
    transfer_helicopters = None
    if scenario.transfer_config.enabled:
        transfer_ambulances = simpy.Resource(
            env, capacity=scenario.transfer_config.n_transfer_ambulances
        )
        transfer_helicopters = simpy.Resource(
            env, capacity=scenario.transfer_config.n_transfer_helicopters
        )

    resources = AEResources(
        triage=simpy.PriorityResource(env, capacity=scenario.n_triage),
        ed_bays=simpy.PriorityResource(env, capacity=scenario.n_ed_bays),
        handover_bays=simpy.Resource(env, capacity=scenario.n_handover_bays),
        ambulance_fleet=simpy.Resource(env, capacity=scenario.n_ambulances),
        helicopter_fleet=simpy.Resource(env, capacity=scenario.n_helicopters),
        diagnostics=diagnostic_resources,
        transfer_ambulances=transfer_ambulances,
        transfer_helicopters=transfer_helicopters,
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
