"""Full A&E pathway simulation model.

This module implements the complete patient flow:
Arrival → Triage → ED Bays (priority queuing) → Disposition → Departure

Phase 5: Single ED bay pool with priority queuing (P1-P4).
P1 patients bypass triage, go directly to ED bays with highest priority.
For admitted patients: boarding wait after treatment.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Any, Optional, Union, TYPE_CHECKING
import itertools

import numpy as np
import simpy

from faer.core.scenario import FullScenario, ArrivalConfig
from faer.core.entities import (
    ArrivalMode, Priority, NodeType, BedState, DiagnosticType,
    TransferType, TransferDestination,
)
from faer.core.incident import MajorIncidentConfig, CASUALTY_PROFILES
from faer.model.patient import Patient, Acuity, Disposition
from faer.model.dynamic_resource import DynamicCapacityResource
from faer.model.scaling_monitor import ScalingMonitor, capacity_scaling_monitor


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
    Phase 9: Downstream resources (Theatre, ITU, Ward).
    Phase 12: Dynamic capacity resources for scaling.

    Resources can be either simpy.Resource/PriorityResource or DynamicCapacityResource.
    The DynamicCapacityResource provides compatible request/release interface.
    """

    triage: Any  # simpy.PriorityResource or DynamicCapacityResource
    ed_bays: Any  # simpy.PriorityResource or DynamicCapacityResource
    handover_bays: Any  # simpy.Resource or DynamicCapacityResource
    ambulance_fleet: simpy.Resource  # Phase 5c: Ambulance vehicles
    helicopter_fleet: simpy.Resource  # Phase 5c: Helicopter vehicles

    # Phase 7: Diagnostic resources (priority queuing for CT/X-ray/Bloods)
    diagnostics: Dict[DiagnosticType, Any] = field(default_factory=dict)

    # Phase 7: Transfer resources
    transfer_ambulances: Optional[simpy.Resource] = None
    transfer_helicopters: Optional[simpy.Resource] = None

    # Phase 9: Downstream resources
    theatre_tables: Optional[Any] = None  # simpy.PriorityResource or DynamicCapacityResource
    itu_beds: Optional[Any] = None  # simpy.Resource or DynamicCapacityResource
    ward_beds: Optional[Any] = None  # simpy.Resource or DynamicCapacityResource

    # Phase 10: Aeromedical resources
    hems_slots: Optional[simpy.Resource] = None

    # Phase 12: Dynamic resources for scaling monitor
    dynamic_resources: Dict[str, DynamicCapacityResource] = field(default_factory=dict)
    scaling_monitor: Optional[ScalingMonitor] = None


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

    # Phase 9: Downstream tracking
    # Wait times for downstream resources
    theatre_waits: List[float] = field(default_factory=list)
    itu_waits: List[float] = field(default_factory=list)
    ward_waits: List[float] = field(default_factory=list)
    # Durations (length of stay)
    theatre_durations: List[float] = field(default_factory=list)
    itu_durations: List[float] = field(default_factory=list)
    ward_durations: List[float] = field(default_factory=list)
    # Counts
    theatre_admissions: int = 0
    itu_admissions: int = 0
    ward_admissions: int = 0
    # ED bay early release count
    ed_bay_early_releases: int = 0

    # Phase 10: Aeromedical tracking
    aeromed_hems_count: int = 0
    aeromed_fixedwing_count: int = 0
    aeromed_slot_waits: List[float] = field(default_factory=list)
    aeromed_slots_missed: int = 0

    # Phase 11: Major Incident tracking
    incident_arrivals_total: int = 0
    incident_arrivals_by_profile: Dict[str, int] = field(default_factory=dict)
    incident_decon_waits: List[float] = field(default_factory=list)

    # Phase 12: Capacity Scaling tracking
    scaling_metrics: Dict[str, Any] = field(default_factory=dict)

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

    def record_theatre_admission(self, wait_time: float, duration: float) -> None:
        """Record theatre/surgery admission (Phase 9)."""
        self.theatre_admissions += 1
        self.theatre_waits.append(wait_time)
        self.theatre_durations.append(duration)

    def record_itu_admission(self, wait_time: float, duration: float) -> None:
        """Record ITU admission (Phase 9)."""
        self.itu_admissions += 1
        self.itu_waits.append(wait_time)
        self.itu_durations.append(duration)

    def record_ward_admission(self, wait_time: float, duration: float) -> None:
        """Record ward admission (Phase 9)."""
        self.ward_admissions += 1
        self.ward_waits.append(wait_time)
        self.ward_durations.append(duration)

    def record_ed_bay_early_release(self) -> None:
        """Record early ED bay release (Phase 9)."""
        self.ed_bay_early_releases += 1

    def record_aeromed_evacuation(
        self,
        patient: Patient,
        aeromed_type: str,
        slot_wait: float,
        slot_missed: bool = False,
    ) -> None:
        """Record aeromedical evacuation (Phase 10)."""
        if aeromed_type == "HEMS":
            self.aeromed_hems_count += 1
        else:
            self.aeromed_fixedwing_count += 1
        self.aeromed_slot_waits.append(slot_wait)
        if slot_missed:
            self.aeromed_slots_missed += 1

    def record_missed_aeromed_slot(self, patient: Patient, time: float) -> None:
        """Record a missed aeromed slot (Phase 10)."""
        self.aeromed_slots_missed += 1

    def record_incident_arrival(self, profile: str) -> None:
        """Record an incident casualty arrival (Phase 11).

        Args:
            profile: The casualty profile value (e.g., "rta", "cbrn").
        """
        self.incident_arrivals_total += 1
        self.incident_arrivals_by_profile[profile] = (
            self.incident_arrivals_by_profile.get(profile, 0) + 1
        )

    def record_decon_wait(self, wait_time: float) -> None:
        """Record CBRN decontamination wait time (Phase 11).

        Args:
            wait_time: Time spent in decontamination in minutes.
        """
        self.incident_decon_waits.append(wait_time)

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

        # P1-P4 Extended Statistics (NHS breach targets)
        # Targets: P1=0min, P2=10min, P3=60min, P4=120min
        priority_targets = {
            "P1": (p1_patients, 0),
            "P2": (p2_patients, 10),
            "P3": (p3_patients, 60),
            "P4": (p4_patients, 120),
        }

        for priority_name, (patients, target_mins) in priority_targets.items():
            waits = [p.treatment_wait for p in patients]
            if waits:
                metrics[f"{priority_name}_p95_wait"] = float(np.percentile(waits, 95))
                metrics[f"{priority_name}_max_wait"] = float(np.max(waits))
                metrics[f"{priority_name}_breach_rate"] = float(np.mean([w > target_mins for w in waits]))
                sys_times = [p.system_time for p in patients if p.departure_time]
                metrics[f"{priority_name}_mean_system_time"] = float(np.mean(sys_times)) if sys_times else 0.0
            else:
                metrics[f"{priority_name}_p95_wait"] = 0.0
                metrics[f"{priority_name}_max_wait"] = 0.0
                metrics[f"{priority_name}_breach_rate"] = 0.0
                metrics[f"{priority_name}_mean_system_time"] = 0.0

            # Departure counts by priority
            metrics[f"departures_{priority_name}"] = len([p for p in patients if p.departure_time])

        # Acuity Extended Statistics
        # Acuity maps to typical priorities: Resus→P1, Majors→P2/P3, Minors→P3/P4
        acuity_data = {
            "resus": resus_patients,
            "majors": majors_patients,
            "minors": minors_patients,
        }

        for acuity_name, patients in acuity_data.items():
            waits = [p.treatment_wait for p in patients]
            if waits:
                metrics[f"{acuity_name}_p95_wait"] = float(np.percentile(waits, 95))
                metrics[f"{acuity_name}_max_wait"] = float(np.max(waits))
                sys_times = [p.system_time for p in patients if p.departure_time]
                metrics[f"{acuity_name}_mean_system_time"] = float(np.mean(sys_times)) if sys_times else 0.0
            else:
                metrics[f"{acuity_name}_p95_wait"] = 0.0
                metrics[f"{acuity_name}_max_wait"] = 0.0
                metrics[f"{acuity_name}_mean_system_time"] = 0.0

            # Departure counts by acuity
            metrics[f"departures_{acuity_name}"] = len([p for p in patients if p.departure_time])

        # Detailed disposition breakdown
        discharged_patients = [p for p in valid_patients if p.disposition == Disposition.DISCHARGE]
        ward_patients = [p for p in valid_patients if p.disposition == Disposition.ADMIT_WARD]
        icu_patients = [p for p in valid_patients if p.disposition == Disposition.ADMIT_ICU]
        transfer_patients = [p for p in valid_patients if p.disposition == Disposition.TRANSFER]
        left_patients = [p for p in valid_patients if p.disposition == Disposition.LEFT]

        metrics["discharged_count"] = len(discharged_patients)
        metrics["admitted_ward_count"] = len(ward_patients)
        metrics["admitted_icu_count"] = len(icu_patients)
        metrics["transfer_count"] = len(transfer_patients)
        metrics["left_count"] = len(left_patients)

        # Mean LoS by disposition type
        metrics["mean_los_discharged"] = float(np.mean([p.system_time for p in discharged_patients])) if discharged_patients else 0.0
        metrics["mean_los_ward"] = float(np.mean([p.system_time for p in ward_patients])) if ward_patients else 0.0
        metrics["mean_los_icu"] = float(np.mean([p.system_time for p in icu_patients])) if icu_patients else 0.0
        metrics["mean_los_transfer"] = float(np.mean([p.system_time for p in transfer_patients])) if transfer_patients else 0.0

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

        # Downstream resource utilisation (Phase 8)
        # Theatre
        if hasattr(scenario, 'theatre_config') and scenario.theatre_config:
            metrics["util_theatre"] = self._compute_utilisation(
                "surgery", effective_run, scenario.theatre_config.n_tables, scenario.warm_up
            )
        else:
            metrics["util_theatre"] = 0.0

        # ITU
        if hasattr(scenario, 'itu_config') and scenario.itu_config:
            metrics["util_itu"] = self._compute_utilisation(
                "itu", effective_run, scenario.itu_config.capacity, scenario.warm_up
            )
        else:
            metrics["util_itu"] = 0.0

        # Ward
        if hasattr(scenario, 'ward_config') and scenario.ward_config:
            metrics["util_ward"] = self._compute_utilisation(
                "ward", effective_run, scenario.ward_config.capacity, scenario.warm_up
            )
        else:
            metrics["util_ward"] = 0.0

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

        # Phase 9: Downstream process metrics
        # Theatre metrics
        metrics["theatre_admissions"] = self.theatre_admissions
        if self.theatre_waits:
            metrics["mean_theatre_wait"] = float(np.mean(self.theatre_waits))
            metrics["p95_theatre_wait"] = float(np.percentile(self.theatre_waits, 95))
            metrics["max_theatre_wait"] = float(np.max(self.theatre_waits))
        else:
            metrics["mean_theatre_wait"] = 0.0
            metrics["p95_theatre_wait"] = 0.0
            metrics["max_theatre_wait"] = 0.0

        if self.theatre_durations:
            metrics["mean_theatre_duration"] = float(np.mean(self.theatre_durations))
        else:
            metrics["mean_theatre_duration"] = 0.0

        # ITU metrics
        metrics["itu_admissions"] = self.itu_admissions
        if self.itu_waits:
            metrics["mean_itu_wait"] = float(np.mean(self.itu_waits))
            metrics["p95_itu_wait"] = float(np.percentile(self.itu_waits, 95))
            metrics["max_itu_wait"] = float(np.max(self.itu_waits))
        else:
            metrics["mean_itu_wait"] = 0.0
            metrics["p95_itu_wait"] = 0.0
            metrics["max_itu_wait"] = 0.0

        if self.itu_durations:
            metrics["mean_itu_duration"] = float(np.mean(self.itu_durations))
        else:
            metrics["mean_itu_duration"] = 0.0

        # Ward metrics
        metrics["ward_admissions"] = self.ward_admissions
        if self.ward_waits:
            metrics["mean_ward_wait"] = float(np.mean(self.ward_waits))
            metrics["p95_ward_wait"] = float(np.percentile(self.ward_waits, 95))
            metrics["max_ward_wait"] = float(np.max(self.ward_waits))
        else:
            metrics["mean_ward_wait"] = 0.0
            metrics["p95_ward_wait"] = 0.0
            metrics["max_ward_wait"] = 0.0

        if self.ward_durations:
            metrics["mean_ward_duration"] = float(np.mean(self.ward_durations))
        else:
            metrics["mean_ward_duration"] = 0.0

        # ED bay early release tracking
        metrics["ed_bay_early_releases"] = self.ed_bay_early_releases
        if total_arrivals > 0:
            metrics["p_early_release"] = self.ed_bay_early_releases / total_arrivals
        else:
            metrics["p_early_release"] = 0.0

        # Phase 10: Aeromedical metrics
        metrics["aeromed_hems_count"] = self.aeromed_hems_count
        metrics["aeromed_fixedwing_count"] = self.aeromed_fixedwing_count
        metrics["aeromed_total"] = self.aeromed_hems_count + self.aeromed_fixedwing_count
        metrics["aeromed_slots_missed"] = self.aeromed_slots_missed

        if self.aeromed_slot_waits:
            metrics["mean_aeromed_slot_wait"] = float(np.mean(self.aeromed_slot_waits))
            metrics["max_aeromed_slot_wait"] = float(np.max(self.aeromed_slot_waits))
        else:
            metrics["mean_aeromed_slot_wait"] = 0.0
            metrics["max_aeromed_slot_wait"] = 0.0

        # Ward bed-days blocked by aeromed wait
        if self.aeromed_slot_waits:
            metrics["ward_bed_days_blocked_aeromed"] = sum(self.aeromed_slot_waits) / 1440
        else:
            metrics["ward_bed_days_blocked_aeromed"] = 0.0

        # Phase 11: Major Incident metrics
        metrics["incident_arrivals_total"] = self.incident_arrivals_total
        metrics["incident_arrivals_by_profile"] = dict(self.incident_arrivals_by_profile)

        # Incident-specific patient metrics (filtered from valid_patients)
        incident_patients = [p for p in valid_patients if p.is_incident_casualty]
        metrics["incident_patients_count"] = len(incident_patients)

        if incident_patients:
            # Wait times for incident casualties
            incident_waits = [p.treatment_wait for p in incident_patients if p.treatment_wait > 0]
            if incident_waits:
                metrics["incident_mean_wait"] = float(np.mean(incident_waits))
                metrics["incident_p95_wait"] = float(np.percentile(incident_waits, 95))
                metrics["incident_max_wait"] = float(np.max(incident_waits))
            else:
                metrics["incident_mean_wait"] = 0.0
                metrics["incident_p95_wait"] = 0.0
                metrics["incident_max_wait"] = 0.0

            # System times for incident casualties
            incident_system_times = [p.system_time for p in incident_patients if p.departure_time is not None]
            if incident_system_times:
                metrics["incident_mean_system_time"] = float(np.mean(incident_system_times))
                metrics["incident_p95_system_time"] = float(np.percentile(incident_system_times, 95))
            else:
                metrics["incident_mean_system_time"] = 0.0
                metrics["incident_p95_system_time"] = 0.0

            # Admission rate for incident casualties
            incident_admitted = sum(1 for p in incident_patients if p.is_admitted)
            metrics["incident_admission_rate"] = incident_admitted / len(incident_patients)

            # Priority breakdown for incident casualties
            for priority in Priority:
                p_patients = [p for p in incident_patients if p.priority == priority]
                metrics[f"incident_{priority.name}_count"] = len(p_patients)
        else:
            metrics["incident_mean_wait"] = 0.0
            metrics["incident_p95_wait"] = 0.0
            metrics["incident_max_wait"] = 0.0
            metrics["incident_mean_system_time"] = 0.0
            metrics["incident_p95_system_time"] = 0.0
            metrics["incident_admission_rate"] = 0.0
            for priority in Priority:
                metrics[f"incident_{priority.name}_count"] = 0

        # Decontamination metrics (CBRN incidents)
        if self.incident_decon_waits:
            metrics["incident_decon_count"] = len(self.incident_decon_waits)
            metrics["incident_mean_decon_time"] = float(np.mean(self.incident_decon_waits))
            metrics["incident_max_decon_time"] = float(np.max(self.incident_decon_waits))
        else:
            metrics["incident_decon_count"] = 0
            metrics["incident_mean_decon_time"] = 0.0
            metrics["incident_max_decon_time"] = 0.0

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
            # P1-P4 extended statistics
            "P1_p95_wait": 0.0, "P2_p95_wait": 0.0, "P3_p95_wait": 0.0, "P4_p95_wait": 0.0,
            "P1_max_wait": 0.0, "P2_max_wait": 0.0, "P3_max_wait": 0.0, "P4_max_wait": 0.0,
            "P1_breach_rate": 0.0, "P2_breach_rate": 0.0, "P3_breach_rate": 0.0, "P4_breach_rate": 0.0,
            "P1_mean_system_time": 0.0, "P2_mean_system_time": 0.0, "P3_mean_system_time": 0.0, "P4_mean_system_time": 0.0,
            "departures_P1": 0, "departures_P2": 0, "departures_P3": 0, "departures_P4": 0,
            # Acuity extended statistics
            "resus_p95_wait": 0.0, "majors_p95_wait": 0.0, "minors_p95_wait": 0.0,
            "resus_max_wait": 0.0, "majors_max_wait": 0.0, "minors_max_wait": 0.0,
            "resus_mean_system_time": 0.0, "majors_mean_system_time": 0.0, "minors_mean_system_time": 0.0,
            "departures_resus": 0, "departures_majors": 0, "departures_minors": 0,
            # Detailed disposition breakdown
            "discharged_count": 0, "admitted_ward_count": 0, "admitted_icu_count": 0,
            "transfer_count": 0, "left_count": 0,
            "mean_los_discharged": 0.0, "mean_los_ward": 0.0, "mean_los_icu": 0.0, "mean_los_transfer": 0.0,
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
    """Boarding process for admitted patients awaiting bed (legacy)."""
    boarding_start = env.now

    duration = sample_lognormal(scenario.rng_boarding, scenario.boarding_mean, scenario.boarding_cv)
    yield env.timeout(duration)

    patient.record_boarding(boarding_start, env.now)


# =============================================================================
# Phase 9: Downstream Processes (Theatre, ITU, Ward)
# =============================================================================


def can_release_ed_bay_early(patient: Patient, next_node: NodeType) -> bool:
    """Determine if patient can release ED bay early (Phase 9).

    Only stable patients (P3/P4) going to Ward can release early.
    P1/P2 patients and those needing ITU/Surgery always hold ED bay.

    Args:
        patient: The patient being routed.
        next_node: The downstream destination.

    Returns:
        True if patient can release ED bay before downstream bed acquired.
    """
    # Discharge always releases
    if next_node == NodeType.EXIT:
        return True

    # Only P3/P4 eligible for early release
    if patient.priority in (Priority.P1_IMMEDIATE, Priority.P2_VERY_URGENT):
        return False

    # Only Ward destination eligible (not ITU, not Surgery)
    if next_node != NodeType.WARD:
        return False

    return True


def theatre_process(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    ed_bay_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """Theatre/surgery process (Phase 9).

    Patient HOLDS upstream resource (ED bay) until theatre table acquired.
    After theatre, routes to ITU or Ward based on post-op probabilities.

    Args:
        env: SimPy environment.
        patient: The patient undergoing surgery.
        resources: Container of simulation resources.
        scenario: Scenario configuration.
        results: Results collector for metrics.
        ed_bay_req: The ED bay request to release once theatre acquired.

    Yields:
        SimPy events for the theatre process.
    """
    config = scenario.theatre_config
    wait_start = env.now

    # Request theatre table (priority-based)
    theatre_req = resources.theatre_tables.request(priority=patient.priority.value)
    yield theatre_req

    # Log theatre resource acquisition
    results.record_resource_state("surgery", env.now, resources.theatre_tables.count)

    surgery_start = env.now
    wait_time = surgery_start - wait_start

    # Release ED bay now that we have theatre
    if ed_bay_req is not None:
        resources.ed_bays.release(ed_bay_req)
        patient.record_ed_bay_release(env.now, early=False)

    # Procedure time (lognormal distribution)
    procedure_mins = sample_lognormal(
        scenario.rng_theatre,
        config.procedure_time_mean,
        config.procedure_time_cv,
    )
    yield env.timeout(procedure_mins)

    surgery_end = env.now
    patient.record_surgery(wait_start, surgery_start, surgery_end)
    results.record_theatre_admission(wait_time, surgery_end - surgery_start)

    # Determine post-op destination
    if scenario.rng_theatre.random() < config.to_itu_prob:
        # Route to ITU - hold theatre until ITU acquired
        yield from itu_process(env, patient, resources, scenario, results, theatre_req)
    else:
        # Route to Ward - hold theatre until ward acquired
        yield from ward_process(env, patient, resources, scenario, results, theatre_req)

    # Theatre table released after downstream routing


def itu_process(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    upstream_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """ITU process with step-down routing (Phase 9).

    Patient HOLDS upstream resource until ITU bed acquired.
    After ITU stay, routes to Ward (step-down) or discharge.

    Args:
        env: SimPy environment.
        patient: The patient needing ITU.
        resources: Container of simulation resources.
        scenario: Scenario configuration.
        results: Results collector for metrics.
        upstream_req: The upstream resource request to release once ITU acquired.
            Could be ED bay, theatre table, etc.

    Yields:
        SimPy events for the ITU process.
    """
    config = scenario.itu_config
    wait_start = env.now

    # Request ITU bed
    itu_req = resources.itu_beds.request()
    yield itu_req

    # Log ITU resource acquisition
    results.record_resource_state("itu", env.now, resources.itu_beds.count)

    itu_start = env.now
    wait_time = itu_start - wait_start

    # Release upstream resource now that we have ITU bed
    if upstream_req is not None:
        # Release the upstream resource (theatre table) and log state change
        resources.theatre_tables.release(upstream_req)
        results.record_resource_state("surgery", env.now, resources.theatre_tables.count)

    # ITU length of stay
    los_mins = sample_lognormal(
        scenario.rng_itu,
        config.los_mean_mins,
        config.los_cv,
    )
    yield env.timeout(los_mins)

    itu_end = env.now
    patient.record_itu(wait_start, itu_start, itu_end)
    results.record_itu_admission(wait_time, itu_end - itu_start)

    # Determine outcome (step-down, discharge, or death)
    outcome_roll = scenario.rng_itu.random()

    if outcome_roll < config.death_prob:
        # Patient died - bed released, no further routing
        resources.itu_beds.release(itu_req)
        results.record_resource_state("itu", env.now, resources.itu_beds.count)
        patient.disposition = Disposition.LEFT  # Or create DEATH disposition
        return

    if outcome_roll < config.death_prob + config.direct_discharge_prob:
        # Direct discharge from ITU (rare)
        resources.itu_beds.release(itu_req)
        results.record_resource_state("itu", env.now, resources.itu_beds.count)
        patient.disposition = Disposition.DISCHARGE
        return

    # Step-down to Ward - HOLD ITU bed until ward bed acquired
    yield from ward_process(env, patient, resources, scenario, results, itu_req)

    # ITU bed released after ward_process acquires ward bed


def _complete_theatre_stay(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    theatre_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """Complete theatre stay after resource already acquired (Phase 9).

    Used when theatre_req was acquired in patient_journey to enable blocking cascade.
    This function handles the actual procedure and downstream routing.

    Args:
        theatre_req: The theatre table request to release after downstream routing.
    """
    config = scenario.theatre_config
    surgery_start = env.now

    # Procedure time
    procedure_mins = sample_lognormal(
        scenario.rng_theatre,
        config.procedure_time_mean,
        config.procedure_time_cv,
    )
    yield env.timeout(procedure_mins)

    surgery_end = env.now
    # Note: wait_start was when we requested the resource, which we don't have here
    # So we record with surgery_start as wait_start (no wait recorded)
    patient.record_surgery(surgery_start, surgery_start, surgery_end)
    results.record_theatre_admission(0.0, surgery_end - surgery_start)

    # Determine post-op destination and route - pass theatre_req as upstream
    if scenario.rng_theatre.random() < config.to_itu_prob:
        yield from itu_process(env, patient, resources, scenario, results, upstream_req=theatre_req)
    else:
        yield from ward_process(env, patient, resources, scenario, results, upstream_req=theatre_req)


def _complete_itu_stay(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    itu_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """Complete ITU stay after resource already acquired (Phase 9).

    Args:
        itu_req: The ITU bed request to release after ward routing or on exit.
    """
    config = scenario.itu_config
    itu_start = env.now

    # ITU length of stay
    los_mins = sample_lognormal(
        scenario.rng_itu,
        config.los_mean_mins,
        config.los_cv,
    )
    yield env.timeout(los_mins)

    itu_end = env.now
    patient.record_itu(itu_start, itu_start, itu_end)
    results.record_itu_admission(0.0, itu_end - itu_start)

    # Determine outcome
    outcome_roll = scenario.rng_itu.random()

    if outcome_roll < config.death_prob:
        # Release ITU bed on death
        if itu_req is not None:
            resources.itu_beds.release(itu_req)
            results.record_resource_state("itu", env.now, resources.itu_beds.count)
        patient.disposition = Disposition.LEFT
        return

    if outcome_roll < config.death_prob + config.direct_discharge_prob:
        # Release ITU bed on direct discharge
        if itu_req is not None:
            resources.itu_beds.release(itu_req)
            results.record_resource_state("itu", env.now, resources.itu_beds.count)
        patient.disposition = Disposition.DISCHARGE
        return

    # Step-down to Ward - pass itu_req as upstream to be released when ward acquired
    yield from ward_process(env, patient, resources, scenario, results, upstream_req=itu_req)


def _complete_ward_stay(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    ward_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """Complete ward stay after resource already acquired (Phase 9).

    Args:
        ward_req: The ward bed request to release on discharge.
    """
    config = scenario.ward_config
    ward_start = env.now

    # Ward length of stay
    los_mins = sample_lognormal(
        scenario.rng_ward,
        config.los_mean_mins,
        config.los_cv,
    )
    yield env.timeout(los_mins)

    ward_end = env.now
    patient.record_ward(ward_start, ward_start, ward_end)
    results.record_ward_admission(0.0, ward_end - ward_start)

    # Release ward bed
    if ward_req is not None:
        resources.ward_beds.release(ward_req)
        results.record_resource_state("ward", env.now, resources.ward_beds.count)

    patient.disposition = Disposition.ADMIT_WARD


def ward_process(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
    upstream_req: Optional[Any] = None,
) -> Generator[simpy.Event, None, None]:
    """Ward process (Phase 9).

    Patient HOLDS upstream resource until ward bed acquired.
    After ward stay, patient is discharged.

    Args:
        env: SimPy environment.
        patient: The patient needing ward bed.
        resources: Container of simulation resources.
        scenario: Scenario configuration.
        results: Results collector for metrics.
        upstream_req: The upstream resource request to release once ward acquired.

    Yields:
        SimPy events for the ward process.
    """
    config = scenario.ward_config
    wait_start = env.now

    # Request ward bed
    ward_req = resources.ward_beds.request()
    yield ward_req

    # Log ward resource acquisition
    results.record_resource_state("ward", env.now, resources.ward_beds.count)

    ward_start = env.now
    wait_time = ward_start - wait_start

    # Release upstream resource now that we have ward bed
    if upstream_req is not None:
        # Determine which resource to release and log appropriately
        if upstream_req.resource is resources.theatre_tables:
            resources.theatre_tables.release(upstream_req)
            results.record_resource_state("surgery", env.now, resources.theatre_tables.count)
        elif upstream_req.resource is resources.itu_beds:
            resources.itu_beds.release(upstream_req)
            results.record_resource_state("itu", env.now, resources.itu_beds.count)

    # Ward length of stay
    los_mins = sample_lognormal(
        scenario.rng_ward,
        config.los_mean_mins,
        config.los_cv,
    )
    yield env.timeout(los_mins)

    ward_end = env.now
    patient.record_ward(wait_start, ward_start, ward_end)
    results.record_ward_admission(wait_time, ward_end - ward_start)

    # Phase 10: Determine discharge pathway
    # Patient HOLDS ward bed during aeromed process (BLOCKED state)
    discharge_pathway = determine_discharge_pathway(patient, scenario)

    if discharge_pathway == "AEROMED_HEMS":
        # Patient needs HEMS evacuation - holds bed throughout
        yield from hems_evacuation_process(env, patient, resources, scenario, results)

    elif discharge_pathway == "AEROMED_FIXED_WING":
        # Patient needs fixed-wing evacuation - holds bed throughout
        yield from fixedwing_evacuation_process(env, patient, resources, scenario, results)

    # Release ward bed
    resources.ward_beds.release(ward_req)
    results.record_resource_state("ward", env.now, resources.ward_beds.count)

    # Patient discharged
    patient.disposition = Disposition.ADMIT_WARD


# =============================================================================
# Phase 10: Aeromedical Evacuation Processes
# =============================================================================


def determine_discharge_pathway(patient: Patient, scenario: FullScenario) -> str:
    """Determine discharge pathway based on patient characteristics (Phase 10).

    Only P1 (Immediate/Resus) patients are eligible for aeromed evacuation.
    Of those selected, a proportion go fixed-wing vs HEMS.

    Args:
        patient: The patient being discharged.
        scenario: Scenario configuration with aeromed settings.

    Returns:
        Discharge pathway: "STANDARD", "AEROMED_HEMS", or "AEROMED_FIXED_WING".
    """
    aeromed_config = scenario.aeromed_config

    if not aeromed_config or not aeromed_config.enabled:
        return "STANDARD"

    # Only P1 patients eligible for aeromed
    if patient.priority != Priority.P1_IMMEDIATE:
        return "STANDARD"

    # Roll for aeromed requirement
    if scenario.rng_aeromed.random() < aeromed_config.p1_aeromed_probability:
        patient.requires_aeromed = True

        # Split between fixed-wing and HEMS
        if scenario.rng_aeromed.random() < aeromed_config.fixedwing_proportion:
            return "AEROMED_FIXED_WING"
        else:
            return "AEROMED_HEMS"

    return "STANDARD"


def hems_evacuation_process(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
) -> Generator[simpy.Event, None, None]:
    """HEMS evacuation process (Phase 10).

    HEMS provides flexible, on-demand evacuation within operating hours.
    Patient HOLDS their ward bed throughout this process (BLOCKED state).

    Timeline:
    1. Clinical stabilisation (30-120 mins)
    2. Wait for HEMS slot (if outside operating hours or slots full)
    3. Transfer to helipad (15-45 mins)
    4. Flight (15-60 mins)
    5. Ward bed released on departure

    Args:
        env: SimPy environment.
        patient: The patient being evacuated.
        resources: Container of simulation resources.
        scenario: Scenario configuration.
        results: Results collector for metrics.

    Yields:
        SimPy events for the evacuation process.
    """
    config = scenario.aeromed_config.hems

    # Stabilisation
    stabilisation_start = env.now
    stab_mins = scenario.rng_aeromed.uniform(*config.stabilisation_mins)
    yield env.timeout(stab_mins)
    stabilisation_end = env.now

    total_slot_wait = 0.0

    # Check if within operating hours
    current_hour = (env.now / 60) % 24

    if current_hour < config.operating_start_hour or current_hour >= config.operating_end_hour:
        # Wait until operating hours
        if current_hour >= config.operating_end_hour:
            # Wait until next morning
            day_start_mins = (env.now // 1440) * 1440  # Start of current day
            next_morning = day_start_mins + 1440 + config.operating_start_hour * 60
            wait_mins = next_morning - env.now
        else:
            # Wait until start hour today
            day_start_mins = (env.now // 1440) * 1440
            today_start = day_start_mins + config.operating_start_hour * 60
            wait_mins = today_start - env.now

        total_slot_wait += wait_mins
        yield env.timeout(wait_mins)

    # Request HEMS slot (if resource exists)
    if resources.hems_slots is not None:
        slot_wait_start = env.now
        with resources.hems_slots.request() as slot_req:
            yield slot_req
            total_slot_wait += env.now - slot_wait_start

            # Transfer to helipad
            transfer_mins = scenario.rng_aeromed.uniform(*config.transfer_to_helipad_mins)
            yield env.timeout(transfer_mins)

            # Flight
            flight_mins = scenario.rng_aeromed.uniform(*config.flight_duration_mins)
            yield env.timeout(flight_mins)
    else:
        # No slot resource - just do transfer and flight
        transfer_mins = scenario.rng_aeromed.uniform(*config.transfer_to_helipad_mins)
        yield env.timeout(transfer_mins)

        flight_mins = scenario.rng_aeromed.uniform(*config.flight_duration_mins)
        yield env.timeout(flight_mins)

    departure_time = env.now

    # Record evacuation
    patient.record_aeromed_evacuation(
        aeromed_type="HEMS",
        stabilisation_start=stabilisation_start,
        stabilisation_end=stabilisation_end,
        wait_for_slot=total_slot_wait,
        departure=departure_time,
        slot_missed=False,
    )

    results.record_aeromed_evacuation(patient, "HEMS", total_slot_wait)


def find_next_fixedwing_slot(current_time: float, config) -> float:
    """Find the next available fixed-wing slot (Phase 10).

    Slots are configured per 12-hour segment. Pattern repeats after
    configured segments.

    Args:
        current_time: Current simulation time in minutes.
        config: FixedWingConfig with slots_per_segment and departure times.

    Returns:
        Time (in simulation minutes) of next available slot.
    """
    segment_duration_mins = 12 * 60  # 720 mins per segment
    current_segment = int(current_time // segment_duration_mins)
    pattern_length = len(config.slots_per_segment)

    # Search for next slot (up to 2 pattern cycles ahead)
    for offset in range(pattern_length * 2):
        segment = current_segment + offset
        pattern_index = segment % pattern_length

        slots_in_segment = config.slots_per_segment[pattern_index]

        if slots_in_segment > 0:
            # Calculate departure time for this segment
            segment_start = segment * segment_duration_mins
            departure_time = segment_start + (config.departure_hour_in_segment * 60)

            # Check if slot is in the future
            if departure_time > current_time:
                return departure_time

    # Fallback: return next day's first slot
    next_day_segment = ((current_segment // 2) + 1) * 2
    segment_start = next_day_segment * segment_duration_mins
    return segment_start + (config.departure_hour_in_segment * 60)


def fixedwing_evacuation_process(
    env: simpy.Environment,
    patient: Patient,
    resources: 'AEResources',
    scenario: FullScenario,
    results: 'FullResultsCollector',
) -> Generator[simpy.Event, None, None]:
    """Fixed-wing evacuation process with scheduled slots (Phase 10).

    Fixed-wing has rigid departure times based on 12-hour segments.
    Patient must be ready by cutoff time to make their slot.

    Timeline:
    1. Clinical stabilisation (2-4 hours)
    2. Find next slot and wait (may miss slot if not ready in time)
    3. Transport to airfield (30-90 mins)
    4. Flight (1-3 hours)
    5. Ward bed released on departure

    Args:
        env: SimPy environment.
        patient: The patient being evacuated.
        resources: Container of simulation resources.
        scenario: Scenario configuration.
        results: Results collector for metrics.

    Yields:
        SimPy events for the evacuation process.
    """
    config = scenario.aeromed_config.fixedwing
    missed_config = scenario.aeromed_config.missed_slot

    # Stabilisation
    stabilisation_start = env.now
    stab_mins = scenario.rng_aeromed.uniform(*config.stabilisation_mins)
    yield env.timeout(stab_mins)
    stabilisation_end = env.now

    slot_missed = False
    total_slot_wait = 0.0

    # Find next available slot
    while True:
        next_slot = find_next_fixedwing_slot(env.now, config)
        cutoff_time = next_slot - (config.cutoff_hours_before * 60)

        if env.now <= cutoff_time:
            # Can make this slot - wait until departure
            wait_mins = next_slot - env.now
            total_slot_wait += wait_mins
            yield env.timeout(wait_mins)
            break
        else:
            # Missed this slot
            slot_missed = True
            results.record_missed_aeromed_slot(patient, env.now)

            # Optional re-stabilisation
            if missed_config.requires_restabilisation:
                restab_mins = stab_mins * missed_config.restabilisation_factor
                yield env.timeout(restab_mins)

            # Continue loop to find next slot

    # Transport to airfield
    transport_mins = scenario.rng_aeromed.uniform(*config.transport_to_airfield_mins)
    yield env.timeout(transport_mins)

    # Flight
    flight_mins = scenario.rng_aeromed.uniform(*config.flight_duration_mins)
    yield env.timeout(flight_mins)

    departure_time = env.now

    # Record evacuation
    patient.record_aeromed_evacuation(
        aeromed_type="FIXED_WING",
        stabilisation_start=stabilisation_start,
        stabilisation_end=stabilisation_end,
        wait_for_slot=total_slot_wait,
        departure=departure_time,
        slot_missed=slot_missed,
    )

    results.record_aeromed_evacuation(patient, "FIXED_WING", total_slot_wait, slot_missed)


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

        # Phase 9: Initialize request variables for downstream resources
        # Must be outside if/else blocks to be accessible after the with block
        theatre_req = None
        itu_req = None
        ward_req = None

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
            elif next_node == NodeType.SURGERY:
                disposition = Disposition.ADMIT_WARD  # Surgery leads to ward eventually
            else:
                disposition = Disposition.ADMIT_WARD

            # Phase 9: Downstream processing (if enabled)
            if scenario.downstream_enabled and disposition != Disposition.DISCHARGE:
                results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.BLOCKED, env.now)

                # Check if patient can release ED bay early (P3/P4 to Ward only)
                can_release_early = (
                    scenario.release_stable_to_wait and
                    can_release_ed_bay_early(patient, next_node)
                )

                # Handle downstream routing WITH blocking cascade
                # Patient holds ED bay until downstream resource acquired
                if next_node == NodeType.SURGERY and resources.theatre_tables is not None:
                    # Request theatre - blocks until available
                    theatre_req = resources.theatre_tables.request(priority=patient.priority.value)
                    yield theatre_req
                    results.record_resource_state("surgery", env.now, resources.theatre_tables.count)
                    # Now have theatre - release ED bay
                    # (context manager will release ED bay when we exit)

                elif next_node == NodeType.ITU and resources.itu_beds is not None:
                    # Request ITU - blocks until available
                    itu_req = resources.itu_beds.request()
                    yield itu_req
                    results.record_resource_state("itu", env.now, resources.itu_beds.count)
                    # Now have ITU - ED bay released when context exits

                elif next_node == NodeType.WARD and resources.ward_beds is not None:
                    if can_release_early:
                        # P3/P4 patients can release ED bay and wait elsewhere
                        patient.record_ed_bay_release(env.now, early=True)
                        results.record_ed_bay_early_release()
                        # ED bay released when context exits, patient will wait for ward
                    else:
                        # Request ward - blocks until available
                        ward_req = resources.ward_beds.request()
                        yield ward_req
                        results.record_resource_state("ward", env.now, resources.ward_beds.count)
                        # Now have ward - ED bay released when context exits
            else:
                # Legacy behaviour: boarding for admitted patients
                if disposition in (Disposition.ADMIT_WARD, Disposition.ADMIT_ICU):
                    results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.BLOCKED, env.now)
                    yield from boarding_process(env, patient, scenario)

        # Phase 5e: Bed goes to CLEANING state (for all paths)
        results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.CLEANING, env.now)
        yield env.timeout(scenario.bed_turnaround_mins)
        results.record_bed_state_change(NodeType.ED_BAYS, bed_id, BedState.EMPTY, env.now)

    results.record_resource_state(resource_name, env.now, resource.count)

    # Phase 9: Continue downstream processing AFTER ED bay released
    # The patient now holds their downstream resource (if acquired above)
    if (scenario.downstream_enabled and
        disposition != Disposition.DISCHARGE and
        not determine_transfer_required(patient, scenario)):

        next_node = scenario.get_next_node(NodeType.ED_BAYS, patient.priority)

        if next_node == NodeType.SURGERY and resources.theatre_tables is not None:
            # Patient already has theatre_req from above
            # Continue with surgery and downstream routing
            yield from _complete_theatre_stay(env, patient, resources, scenario, results, theatre_req)

        elif next_node == NodeType.ITU and resources.itu_beds is not None:
            # Patient already has itu_req from above
            yield from _complete_itu_stay(env, patient, resources, scenario, results, itu_req)

        elif next_node == NodeType.WARD and resources.ward_beds is not None:
            # For early release patients, need to acquire ward now
            if scenario.release_stable_to_wait and can_release_ed_bay_early(patient, next_node):
                # Patient released ED bay early, now wait for ward
                yield from ward_process(env, patient, resources, scenario, results, upstream_req=None)
            else:
                # Patient already has ward_req from above
                yield from _complete_ward_stay(env, patient, resources, scenario, results, ward_req)

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


# ============== Phase 11: Major Incident Arrival Generator ==============

def incident_arrival_generator(
    env: simpy.Environment,
    resources: AEResources,
    incident_times: List[float],
    scenario: FullScenario,
    results: 'FullResultsCollector',
    patient_counter: itertools.count,
) -> Generator[simpy.Event, None, None]:
    """Generate incident casualty arrivals at pre-scheduled times.

    Phase 11: Incident arrivals are pre-calculated at simulation start for
    reproducibility. This generator yields until each scheduled arrival time,
    then creates and processes the incident casualty.

    Args:
        env: SimPy environment.
        resources: Hospital resources.
        incident_times: Pre-calculated absolute arrival times (sorted).
        scenario: Full scenario configuration.
        results: Results collector.
        patient_counter: Shared counter for unique patient IDs.

    Yields:
        SimPy events for timeouts between arrivals.
    """
    config = scenario.major_incident_config
    if not config:
        return

    rng = scenario.rng_incident
    profile_data = CASUALTY_PROFILES[config.casualty_profile]

    for arrival_time in incident_times:
        # Wait until scheduled arrival time
        if env.now < arrival_time:
            yield env.timeout(arrival_time - env.now)

        # Sample priority from incident profile's casualty mix
        priority = config.sample_priority(rng)

        # Map priority to acuity (same logic as normal arrivals)
        if priority == Priority.P1_IMMEDIATE:
            acuity = Acuity.RESUS
        elif priority in (Priority.P2_VERY_URGENT, Priority.P3_URGENT):
            acuity = Acuity.MAJORS if rng.random() < 0.6 else Acuity.MINORS
        else:
            acuity = Acuity.MINORS

        # Create incident casualty patient
        patient = Patient(
            id=next(patient_counter),
            arrival_time=env.now,
            acuity=acuity,
            priority=priority,
            mode=ArrivalMode.AMBULANCE,  # Incident casualties arrive by ambulance
        )

        # Mark as incident casualty with profile info
        patient.mark_as_incident_casualty(
            incident_profile=config.casualty_profile.value,
            requires_decon=profile_data["requires_decon"],
        )

        # Handle CBRN decontamination delay
        if patient.requires_decon:
            decon_start = env.now
            decon_duration = config.sample_decon_time(rng)
            yield env.timeout(decon_duration)
            patient.record_decontamination(decon_start, env.now)
            results.record_decon_wait(decon_duration)

        # Record arrival and start patient journey
        results.record_arrival(patient)
        results.record_incident_arrival(config.casualty_profile.value)

        # Incident casualties go directly to patient journey (no vehicle turnaround)
        # They are assumed to have been delivered and vehicle departed
        env.process(patient_journey(env, patient, resources, scenario, results))


def run_full_simulation(scenario: FullScenario, use_multistream: bool = False) -> Dict[str, Any]:
    """Execute full A&E simulation.

    Phase 5: Single ED bay pool with priority queuing.
    Phase 12: Capacity scaling with dynamic resources.

    Args:
        scenario: FullScenario configuration.
        use_multistream: If True, use multi-stream arrivals from arrival_configs.
                         If False, use single-stream based on arrival_rate.

    Returns:
        Dictionary containing simulation results and metrics.
    """
    # Initialize environment
    env = simpy.Environment()

    # Phase 12: Check if capacity scaling is enabled
    scaling_enabled = (
        scenario.capacity_scaling is not None and
        scenario.capacity_scaling.enabled
    )
    scaling_config = scenario.capacity_scaling

    # Dictionary to hold dynamic resources for scaling monitor
    dynamic_resources: Dict[str, DynamicCapacityResource] = {}

    # Create resources - use DynamicCapacityResource when scaling is enabled
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

    # Phase 9: Downstream resources (Theatre, ITU, Ward)
    # When scaling enabled, use DynamicCapacityResource for ED and Ward
    theatre_tables = None
    itu_beds = None
    ward_beds = None

    if scenario.downstream_enabled:
        if scenario.theatre_config and scenario.theatre_config.enabled:
            theatre_tables = simpy.PriorityResource(
                env, capacity=scenario.theatre_config.n_tables
            )
        if scenario.itu_config and scenario.itu_config.enabled:
            itu_beds = simpy.Resource(env, capacity=scenario.itu_config.capacity)
        if scenario.ward_config and scenario.ward_config.enabled:
            if scaling_enabled:
                # Dynamic ward beds - max capacity is baseline + max surge
                max_ward = scenario.ward_config.capacity + scaling_config.opel_config.opel_4_surge_beds * 2
                ward_beds = DynamicCapacityResource(
                    env=env,
                    name="ward_beds",
                    initial_capacity=scenario.ward_config.capacity,
                    max_capacity=max_ward,
                    is_priority=False
                )
                dynamic_resources["ward_beds"] = ward_beds
            else:
                ward_beds = simpy.Resource(env, capacity=scenario.ward_config.capacity)

    # Phase 10: Aeromedical resources
    hems_slots = None
    if scenario.aeromed_config and scenario.aeromed_config.enabled:
        if scenario.aeromed_config.hems.enabled:
            hems_slots = simpy.Resource(
                env, capacity=scenario.aeromed_config.hems.slots_per_day
            )

    # Create ED bays - dynamic when scaling enabled
    if scaling_enabled:
        # Max ED capacity = baseline + max surge from OPEL 4
        max_ed = scenario.n_ed_bays + scaling_config.opel_config.opel_4_surge_beds * 2
        ed_bays = DynamicCapacityResource(
            env=env,
            name="ed_bays",
            initial_capacity=scenario.n_ed_bays,
            max_capacity=max_ed,
            is_priority=True
        )
        dynamic_resources["ed_bays"] = ed_bays
    else:
        ed_bays = simpy.PriorityResource(env, capacity=scenario.n_ed_bays)

    # Create AEResources container
    resources = AEResources(
        triage=simpy.PriorityResource(env, capacity=scenario.n_triage),
        ed_bays=ed_bays,
        handover_bays=simpy.Resource(env, capacity=scenario.n_handover_bays),
        ambulance_fleet=simpy.Resource(env, capacity=scenario.n_ambulances),
        helicopter_fleet=simpy.Resource(env, capacity=scenario.n_helicopters),
        diagnostics=diagnostic_resources,
        transfer_ambulances=transfer_ambulances,
        transfer_helicopters=transfer_helicopters,
        theatre_tables=theatre_tables,
        itu_beds=itu_beds,
        ward_beds=ward_beds,
        hems_slots=hems_slots,
        dynamic_resources=dynamic_resources,
    )

    # Initialize results collector
    results = FullResultsCollector()

    # Phase 12: Start capacity scaling monitor if enabled
    scaling_monitor = None
    if scaling_enabled and dynamic_resources:
        scaling_monitor = capacity_scaling_monitor(
            env=env,
            resources=dynamic_resources,
            config=scaling_config,
            results_collector=results,
            discharge_manager=None  # Future: add discharge manager
        )
        resources.scaling_monitor = scaling_monitor

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

    # Phase 11: Start incident arrival generator if configured
    if (scenario.major_incident_config and
        scenario.major_incident_config.enabled):
        # Pre-generate incident arrival times for reproducibility
        # Use average rate across configured arrival profiles for baseline
        total_hourly_rate = sum(
            sum(cfg.hourly_rates) / 24 for cfg in scenario.arrival_configs
        ) if scenario.arrival_configs else scenario.arrival_rate

        incident_times = scenario.major_incident_config.generate_arrival_times(
            scenario.rng_incident,
            total_hourly_rate,
        )

        if incident_times:
            env.process(incident_arrival_generator(
                env, resources, incident_times, scenario, results, patient_counter
            ))

    # Run simulation
    total_time = scenario.warm_up + scenario.run_length
    env.run(until=total_time)

    # Phase 12: Collect scaling metrics before computing other metrics
    if scaling_monitor is not None:
        scaling_metrics = scaling_monitor.get_metrics()

        # Calculate additional derived metrics
        run_length_mins = scenario.run_length

        # Time at surge (when any dynamic resource is above baseline)
        total_surge_time = 0.0
        for resource_name, resource in dynamic_resources.items():
            for event in resource.capacity_log:
                if event.new_capacity > resource.capacity_log[0].new_capacity:
                    # This is a surge state - count time until next event
                    pass  # Simplified - use events from monitor instead

        # Use OPEL time breakdown for surge percentage
        opel_times = scaling_metrics.get('opel_time_at_level', {})
        from faer.core.scaling import OPELLevel
        opel_3_4_time = opel_times.get(OPELLevel.OPEL_3, 0) + opel_times.get(OPELLevel.OPEL_4, 0)
        total_opel_time = sum(opel_times.values()) if opel_times else run_length_mins
        pct_time_at_surge = (opel_3_4_time / total_opel_time * 100) if total_opel_time > 0 else 0

        # Calculate additional bed-hours from surge
        total_additional_bed_hours = 0.0
        for resource_name, resource in dynamic_resources.items():
            baseline = resource.capacity_log[0].new_capacity if resource.capacity_log else 0
            timeline = resource.get_capacity_timeline()
            for i in range(len(timeline) - 1):
                t_start, cap = timeline[i]
                t_end = timeline[i + 1][0]
                if cap > baseline:
                    additional_beds = cap - baseline
                    duration_hours = (t_end - t_start) / 60
                    total_additional_bed_hours += additional_beds * duration_hours
            # Add final segment to end of run
            if timeline:
                t_start, cap = timeline[-1]
                if cap > baseline:
                    additional_beds = cap - baseline
                    duration_hours = (total_time - t_start) / 60
                    total_additional_bed_hours += additional_beds * duration_hours

        # Convert events to serializable format
        events_serializable = []
        for event in scaling_metrics.get('events', []):
            events_serializable.append({
                'time': event.time,
                'rule': event.rule_name,
                'action': event.action_type,
                'resource': event.resource,
                'old_capacity': event.old_capacity,
                'new_capacity': event.new_capacity,
                'trigger_value': event.trigger_value,
                'direction': event.direction,
            })

        # Convert OPEL times to serializable format
        opel_time_serializable = {level.value: time for level, time in opel_times.items()} if opel_times else {}

        # Build rule activations dict
        rule_activations = {}
        for rule_name, rule_data in scaling_metrics.get('rule_metrics', {}).items():
            rule_activations[rule_name] = rule_data.get('activations', 0)

        results.scaling_metrics = {
            'total_scale_up_events': scaling_metrics.get('total_scale_up_events', 0),
            'total_scale_down_events': scaling_metrics.get('total_scale_down_events', 0),
            'total_scaling_events': scaling_metrics.get('total_scaling_events', 0),
            'pct_time_at_surge': pct_time_at_surge,
            'total_additional_bed_hours': total_additional_bed_hours,
            'opel_transitions': scaling_metrics.get('opel_transitions', 0),
            'opel_peak_level': scaling_metrics.get('opel_peak_level', 1),
            'opel_time_at_level': opel_time_serializable,
            'patients_diverted': 0,  # Future: track from monitor
            'rule_activations': rule_activations,
            'events': events_serializable,
        }

    # Compute metrics
    metrics = results.compute_metrics(scenario.run_length, scenario)

    # Add scenario info
    metrics["run_length"] = scenario.run_length
    metrics["warm_up"] = scenario.warm_up

    # Phase 12: Add scaling metrics to output
    if results.scaling_metrics:
        metrics["scaling_metrics"] = results.scaling_metrics

    return metrics
