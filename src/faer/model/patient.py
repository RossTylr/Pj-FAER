"""Patient entity definition."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

# Re-export from entities to maintain backwards compatibility
from faer.core.entities import Priority, NodeType, ArrivalMode


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
