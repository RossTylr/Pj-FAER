"""Patient entity definition."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

# Re-export from entities to maintain backwards compatibility
from faer.core.entities import (
    Priority, NodeType, ArrivalMode, DiagnosticType,
    TransferType, TransferDestination,
)


class Acuity(Enum):
    """Patient acuity levels (Manchester Triage Scale inspired)."""

    RESUS = auto()  # Immediate - life-threatening
    MAJORS = auto()  # Urgent/Very urgent - serious conditions
    MINORS = auto()  # Standard/Less urgent - ambulatory


class Disposition(Enum):
    """Patient disposition outcomes."""

    DISCHARGE = auto()  # Discharged home
    ADMIT_WARD = auto()  # Admitted to general ward
    ADMIT_ICU = auto()  # Admitted to ICU/HDU
    TRANSFER = auto()  # Transferred to another facility
    LEFT = auto()  # Left before treatment complete


@dataclass
class Patient:
    """Patient entity tracking journey through A&E.

    Attributes:
        id: Unique patient identifier.
        arrival_time: Simulation time of arrival.
        acuity: Patient acuity level (Resus/Majors/Minors).
        priority: Triage priority (P1-P4, lower = more urgent).
        timestamps: Dictionary of event timestamps.
        disposition: Final outcome (set at departure).
    """

    id: int
    arrival_time: float
    acuity: Acuity
    priority: Priority = Priority.P3_URGENT  # Default to P3
    mode: ArrivalMode = ArrivalMode.AMBULANCE  # Default arrival mode
    current_node: Optional[NodeType] = None  # Current location in hospital

    # Timestamps (filled during simulation)
    handover_queue_start: Optional[float] = None  # Phase 5b
    handover_start: Optional[float] = None  # Phase 5b
    handover_end: Optional[float] = None  # Phase 5b
    handover_released: Optional[float] = None  # Phase 5b - when handover bay released (after ED acquisition)
    triage_start: Optional[float] = None
    triage_end: Optional[float] = None
    treatment_start: Optional[float] = None
    treatment_end: Optional[float] = None
    boarding_start: Optional[float] = None
    boarding_end: Optional[float] = None
    departure_time: Optional[float] = None

    # Outcome
    disposition: Optional[Disposition] = None

    # For tracking which resources were used
    resources_used: List[str] = field(default_factory=list)

    # Phase 7: Diagnostic tracking
    diagnostics_required: List[DiagnosticType] = field(default_factory=list)
    diagnostics_completed: List[DiagnosticType] = field(default_factory=list)
    # Timestamps dict: stores {f'{diag_type.name}_queue_start': time, ...}
    diagnostic_timestamps: dict = field(default_factory=dict)

    # Phase 7: Transfer tracking
    requires_transfer: bool = False
    transfer_type: Optional[TransferType] = None
    transfer_destination: Optional[TransferDestination] = None
    transfer_decision_time: Optional[float] = None
    transfer_requested_time: Optional[float] = None
    transfer_vehicle_arrived_time: Optional[float] = None
    transfer_departed_time: Optional[float] = None

    # Phase 9: Downstream timing
    surgery_wait_start: Optional[float] = None
    surgery_start: Optional[float] = None
    surgery_end: Optional[float] = None

    itu_wait_start: Optional[float] = None
    itu_start: Optional[float] = None
    itu_end: Optional[float] = None

    ward_wait_start: Optional[float] = None
    ward_start: Optional[float] = None
    ward_end: Optional[float] = None

    # ED bay release tracking
    released_ed_bay_early: bool = False
    ed_bay_release_time: Optional[float] = None

    # Phase 10: Aeromedical evacuation
    requires_aeromed: bool = False
    aeromed_type: Optional[str] = None  # "HEMS" or "FIXED_WING"
    aeromed_stabilisation_start: Optional[float] = None
    aeromed_stabilisation_end: Optional[float] = None
    aeromed_wait_for_slot: Optional[float] = None  # Time waiting for transport
    aeromed_slot_missed: bool = False
    aeromed_departure: Optional[float] = None

    # Phase 11: Major Incident
    is_incident_casualty: bool = False
    incident_profile: Optional[str] = None  # CasualtyProfile value (e.g., "rta", "cbrn")
    requires_decon: bool = False  # CBRN casualties need decontamination
    decon_start: Optional[float] = None
    decon_end: Optional[float] = None

    @property
    def handover_delay(self) -> float:
        """Time waiting for handover bay (Phase 5b)."""
        if self.handover_queue_start is not None and self.handover_start is not None:
            return self.handover_start - self.handover_queue_start
        return 0.0

    @property
    def handover_duration(self) -> float:
        """Time in handover process (Phase 5b)."""
        if self.handover_start is not None and self.handover_end is not None:
            return self.handover_end - self.handover_start
        return 0.0

    @property
    def triage_wait(self) -> float:
        """Time waiting for triage (arrival to triage start)."""
        if self.triage_start is not None:
            return self.triage_start - self.arrival_time
        return 0.0

    @property
    def triage_duration(self) -> float:
        """Time in triage."""
        if self.triage_start is not None and self.triage_end is not None:
            return self.triage_end - self.triage_start
        return 0.0

    @property
    def treatment_wait(self) -> float:
        """Time waiting for treatment (post-triage to treatment start)."""
        if self.treatment_start is not None and self.triage_end is not None:
            return self.treatment_start - self.triage_end
        return 0.0

    @property
    def treatment_duration(self) -> float:
        """Time in treatment."""
        if self.treatment_start is not None and self.treatment_end is not None:
            return self.treatment_end - self.treatment_start
        return 0.0

    @property
    def boarding_duration(self) -> float:
        """Time boarding (waiting for bed after decision to admit)."""
        if self.boarding_start is not None and self.boarding_end is not None:
            return self.boarding_end - self.boarding_start
        return 0.0

    @property
    def total_wait(self) -> float:
        """Total waiting time (triage wait + treatment wait)."""
        return self.triage_wait + self.treatment_wait

    @property
    def system_time(self) -> float:
        """Total time in system (arrival to departure)."""
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return 0.0

    @property
    def is_admitted(self) -> bool:
        """Whether patient was admitted."""
        return self.disposition in (Disposition.ADMIT_WARD, Disposition.ADMIT_ICU)

    def record_handover(self, queue_start: float, start: float, end: float) -> None:
        """Record handover timestamps (Phase 5b)."""
        self.handover_queue_start = queue_start
        self.handover_start = start
        self.handover_end = end
        self.resources_used.append("handover")

    def record_handover_release(self, time: float) -> None:
        """Record when handover bay was released (Phase 5b)."""
        self.handover_released = time

    def record_triage(self, start: float, end: float) -> None:
        """Record triage timestamps."""
        self.triage_start = start
        self.triage_end = end
        self.resources_used.append("triage")

    def record_treatment(self, start: float, end: float) -> None:
        """Record treatment timestamps."""
        self.treatment_start = start
        self.treatment_end = end

    def record_boarding(self, start: float, end: float) -> None:
        """Record boarding timestamps."""
        self.boarding_start = start
        self.boarding_end = end

    def record_departure(self, time: float, disposition: Disposition) -> None:
        """Record departure."""
        self.departure_time = time
        self.disposition = disposition

    def record_diagnostic_event(self, diag_type: DiagnosticType, event: str, time: float) -> None:
        """Record a diagnostic journey event (Phase 7).

        Args:
            diag_type: The diagnostic type (CT_SCAN, XRAY, BLOODS).
            event: The event name (queue_start, start, end, return_to_bay, results_available).
            time: The simulation time.
        """
        key = f"{diag_type.name}_{event}"
        self.diagnostic_timestamps[key] = time

    def complete_diagnostic(self, diag_type: DiagnosticType) -> None:
        """Mark a diagnostic as completed (Phase 7)."""
        if diag_type not in self.diagnostics_completed:
            self.diagnostics_completed.append(diag_type)

    @property
    def total_diagnostic_time(self) -> float:
        """Total time spent on diagnostics journey (Phase 7)."""
        total = 0.0
        for diag in self.diagnostics_completed:
            start_key = f'{diag.name}_journey_start'
            end_key = f'{diag.name}_results_available'
            if start_key in self.diagnostic_timestamps:
                end_time = self.diagnostic_timestamps.get(
                    end_key,
                    self.diagnostic_timestamps.get(f'{diag.name}_end', 0)
                )
                total += end_time - self.diagnostic_timestamps[start_key]
        return total

    def get_diagnostic_wait(self, diag_type: DiagnosticType) -> float:
        """Get wait time for a specific diagnostic (Phase 7)."""
        queue_key = f"{diag_type.name}_queue_start"
        start_key = f"{diag_type.name}_start"
        if queue_key in self.diagnostic_timestamps and start_key in self.diagnostic_timestamps:
            return self.diagnostic_timestamps[start_key] - self.diagnostic_timestamps[queue_key]
        return 0.0

    @property
    def transfer_wait_time(self) -> float:
        """Time waiting for transfer vehicle (Phase 7)."""
        if self.transfer_requested_time and self.transfer_vehicle_arrived_time:
            return self.transfer_vehicle_arrived_time - self.transfer_requested_time
        return 0.0

    @property
    def transfer_total_time(self) -> float:
        """Total time for transfer process from decision to departure (Phase 7)."""
        if self.transfer_decision_time and self.transfer_departed_time:
            return self.transfer_departed_time - self.transfer_decision_time
        return 0.0

    def record_transfer(
        self,
        transfer_type: TransferType,
        destination: TransferDestination,
        decision_time: float,
        requested_time: float,
        vehicle_arrived_time: float,
        departed_time: float,
    ) -> None:
        """Record transfer journey timestamps (Phase 7)."""
        self.requires_transfer = True
        self.transfer_type = transfer_type
        self.transfer_destination = destination
        self.transfer_decision_time = decision_time
        self.transfer_requested_time = requested_time
        self.transfer_vehicle_arrived_time = vehicle_arrived_time
        self.transfer_departed_time = departed_time

    # Phase 9: Downstream timing properties and methods

    @property
    def surgery_wait(self) -> float:
        """Time waiting for theatre table (Phase 9)."""
        if self.surgery_wait_start is not None and self.surgery_start is not None:
            return self.surgery_start - self.surgery_wait_start
        return 0.0

    @property
    def surgery_duration(self) -> float:
        """Time in surgery (Phase 9)."""
        if self.surgery_start is not None and self.surgery_end is not None:
            return self.surgery_end - self.surgery_start
        return 0.0

    @property
    def itu_wait(self) -> float:
        """Time waiting for ITU bed (Phase 9)."""
        if self.itu_wait_start is not None and self.itu_start is not None:
            return self.itu_start - self.itu_wait_start
        return 0.0

    @property
    def itu_duration(self) -> float:
        """Time in ITU (Phase 9)."""
        if self.itu_start is not None and self.itu_end is not None:
            return self.itu_end - self.itu_start
        return 0.0

    @property
    def ward_wait(self) -> float:
        """Time waiting for ward bed (Phase 9)."""
        if self.ward_wait_start is not None and self.ward_start is not None:
            return self.ward_start - self.ward_wait_start
        return 0.0

    @property
    def ward_duration(self) -> float:
        """Time in ward (Phase 9)."""
        if self.ward_start is not None and self.ward_end is not None:
            return self.ward_end - self.ward_start
        return 0.0

    def record_surgery(self, wait_start: float, start: float, end: float) -> None:
        """Record surgery timestamps (Phase 9)."""
        self.surgery_wait_start = wait_start
        self.surgery_start = start
        self.surgery_end = end
        self.resources_used.append("theatre")

    def record_itu(self, wait_start: float, start: float, end: float) -> None:
        """Record ITU timestamps (Phase 9)."""
        self.itu_wait_start = wait_start
        self.itu_start = start
        self.itu_end = end
        self.resources_used.append("itu")

    def record_ward(self, wait_start: float, start: float, end: float) -> None:
        """Record ward timestamps (Phase 9)."""
        self.ward_wait_start = wait_start
        self.ward_start = start
        self.ward_end = end
        self.resources_used.append("ward")

    def record_ed_bay_release(self, time: float, early: bool = False) -> None:
        """Record when ED bay was released (Phase 9)."""
        self.ed_bay_release_time = time
        self.released_ed_bay_early = early

    # Phase 10: Aeromedical evacuation properties and methods

    @property
    def aeromed_stabilisation_duration(self) -> float:
        """Time for aeromed clinical stabilisation (Phase 10)."""
        if self.aeromed_stabilisation_start is not None and self.aeromed_stabilisation_end is not None:
            return self.aeromed_stabilisation_end - self.aeromed_stabilisation_start
        return 0.0

    @property
    def aeromed_total_time(self) -> float:
        """Total aeromed process time from stabilisation start to departure (Phase 10)."""
        if self.aeromed_stabilisation_start is not None and self.aeromed_departure is not None:
            return self.aeromed_departure - self.aeromed_stabilisation_start
        return 0.0

    def record_aeromed_evacuation(
        self,
        aeromed_type: str,
        stabilisation_start: float,
        stabilisation_end: float,
        wait_for_slot: float,
        departure: float,
        slot_missed: bool = False,
    ) -> None:
        """Record aeromedical evacuation timestamps (Phase 10).

        Args:
            aeromed_type: "HEMS" or "FIXED_WING".
            stabilisation_start: Time when clinical stabilisation began.
            stabilisation_end: Time when stabilisation completed.
            wait_for_slot: Total time waiting for transport slot.
            departure: Time when patient departed via aeromed.
            slot_missed: Whether patient missed a scheduled slot.
        """
        self.requires_aeromed = True
        self.aeromed_type = aeromed_type
        self.aeromed_stabilisation_start = stabilisation_start
        self.aeromed_stabilisation_end = stabilisation_end
        self.aeromed_wait_for_slot = wait_for_slot
        self.aeromed_departure = departure
        self.aeromed_slot_missed = slot_missed
        self.resources_used.append(f"aeromed_{aeromed_type.lower()}")

    # Phase 11: Major Incident properties and methods

    @property
    def decon_duration(self) -> float:
        """Time for decontamination (Phase 11, CBRN only)."""
        if self.decon_start is not None and self.decon_end is not None:
            return self.decon_end - self.decon_start
        return 0.0

    def record_decontamination(self, start: float, end: float) -> None:
        """Record decontamination timestamps (Phase 11, CBRN).

        Args:
            start: Time when decontamination began.
            end: Time when decontamination completed.
        """
        self.decon_start = start
        self.decon_end = end
        self.resources_used.append("decon")

    def mark_as_incident_casualty(
        self,
        incident_profile: str,
        requires_decon: bool = False,
    ) -> None:
        """Mark patient as casualty from major incident (Phase 11).

        Args:
            incident_profile: The casualty profile value (e.g., "rta", "cbrn").
            requires_decon: Whether CBRN decontamination is needed.
        """
        self.is_incident_casualty = True
        self.incident_profile = incident_profile
        self.requires_decon = requires_decon
