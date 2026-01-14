"""Scenario configuration dataclass."""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict

import numpy as np

from faer.core.entities import (
    NodeType, Priority, ArrivalMode, ArrivalModel, DayType,
    DiagnosticType, TransferType,
)
from faer.core.incident import MajorIncidentConfig


# ============== Phase 8: Run Length Presets ==============

class RunLengthPreset(Enum):
    """Standard run length options (Phase 8).

    Provides preset durations from 12 hours to 72 hours (3 days).
    Longer runs show compounding effects of capacity constraints.
    """
    HALF_DAY = 12 * 60      # 720 mins = 12 hours
    ONE_DAY = 24 * 60       # 1440 mins = 24 hours
    THIRTY_SIX_H = 36 * 60  # 2160 mins = 36 hours
    TWO_DAYS = 48 * 60      # 2880 mins = 48 hours
    SIXTY_H = 60 * 60       # 3600 mins = 60 hours
    THREE_DAYS = 72 * 60    # 4320 mins = 72 hours

    @property
    def hours(self) -> int:
        """Return duration in hours."""
        return self.value // 60

    @property
    def minutes(self) -> int:
        """Return duration in minutes."""
        return self.value

    @property
    def label(self) -> str:
        """Return human-readable label."""
        h = self.hours
        if h <= 24:
            return f"{h} hours"
        else:
            return f"{h} hours ({h/24:.1f} days)"


# ============== Phase 8: ITU Configuration ==============

@dataclass
class ITUConfig:
    """Intensive Care Unit configuration (Phase 8).

    Models ITU beds with step-down pathway to Ward.
    Key metrics: occupancy, refused admissions, step-down delays.

    Attributes:
        capacity: Number of ITU beds.
        los_mean_hours: Average length of stay in hours.
        los_cv: Coefficient of variation for LoS (high variability typical).
        step_down_to_ward_prob: Probability patient steps down to Ward.
        direct_discharge_prob: Probability of rare direct discharge from ITU.
        death_prob: In-ITU mortality rate.
        min_step_down_ready_hours: Minimum time before patient ready for step-down.
        enabled: Whether ITU is modelled.
    """
    capacity: int = 6
    los_mean_hours: float = 48.0
    los_cv: float = 0.8

    # Step-down routing probabilities (must sum to 1.0)
    step_down_to_ward_prob: float = 0.85
    direct_discharge_prob: float = 0.05
    death_prob: float = 0.10

    # Step-down timing
    min_step_down_ready_hours: float = 12.0

    enabled: bool = True

    def __post_init__(self):
        # Validate probabilities sum to 1.0
        total = self.step_down_to_ward_prob + self.direct_discharge_prob + self.death_prob
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"ITU outcome probabilities must sum to 1.0, got {total}"
            )

    @property
    def los_mean_mins(self) -> float:
        """Return LoS in minutes for simulation."""
        return self.los_mean_hours * 60


# ============== Phase 8: Ward Configuration ==============

@dataclass
class WardConfig:
    """Ward bed configuration (Phase 8).

    Models ward beds with configurable LoS and turnaround.
    Ward is often the system bottleneck affecting upstream flow.

    Attributes:
        capacity: Number of ward beds.
        los_mean_hours: Average length of stay in hours (typically 72h = 3 days).
        los_cv: Coefficient of variation (high due to mix of simple/complex cases).
        turnaround_mins: Bed cleaning/prep time between patients.
        discharge_bias_hours: Hours when discharge is more likely (default 10-14).
        enabled: Whether Ward is modelled.
    """
    capacity: int = 30
    los_mean_hours: float = 72.0  # 3 days average
    los_cv: float = 1.0  # High variability

    turnaround_mins: float = 30.0

    # Discharge pattern (optional - hours when discharge more likely)
    discharge_bias_hours: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14])

    enabled: bool = True

    @property
    def los_mean_mins(self) -> float:
        """Return LoS in minutes for simulation."""
        return self.los_mean_hours * 60


# ============== Phase 8: Theatre Configuration ==============

@dataclass
class TheatreConfig:
    """Operating theatre configuration (Phase 8).

    Models theatre capacity with session-based scheduling.
    Emergency surgery competes with elective lists.

    Attributes:
        n_tables: Number of operating tables.
        session_duration_mins: Duration of a theatre session (list).
        sessions_per_day: Number of sessions per day (AM + PM).
        procedure_time_mean: Average surgery duration in minutes.
        procedure_time_cv: Variability in procedure time.
        to_itu_prob: Probability patient goes to ITU post-op.
        to_ward_prob: Probability patient goes to Ward post-op.
        emergency_priority_boost: Priority adjustment for emergency cases.
        enabled: Whether Theatre is modelled.
    """
    n_tables: int = 2
    session_duration_mins: int = 240  # 4 hours per session
    sessions_per_day: int = 2  # AM and PM lists

    procedure_time_mean: float = 90.0  # Average surgery duration
    procedure_time_cv: float = 0.5

    # Post-op destination probabilities (must sum to 1.0)
    to_itu_prob: float = 0.25
    to_ward_prob: float = 0.75

    # Emergency priority
    emergency_priority_boost: int = 10

    enabled: bool = True

    def __post_init__(self):
        # Validate probabilities sum to 1.0
        total = self.to_itu_prob + self.to_ward_prob
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Theatre post-op probabilities must sum to 1.0, got {total}"
            )

    @property
    def daily_capacity_hours(self) -> float:
        """Total theatre hours available per day."""
        return self.n_tables * self.session_duration_mins * self.sessions_per_day / 60

    @property
    def estimated_daily_cases(self) -> float:
        """Estimated number of cases per day."""
        total_mins = self.n_tables * self.session_duration_mins * self.sessions_per_day
        return total_mins / self.procedure_time_mean


# Day type multipliers for arrival pattern adjustments (Phase 6)
DAY_TYPE_MULTIPLIERS: Dict[DayType, Dict] = {
    DayType.WEEKDAY: {
        'overall': 1.0,
        'hourly_adjustments': {}  # No adjustments
    },
    DayType.MONDAY: {
        'overall': 1.0,
        'hourly_adjustments': {
            7: 1.2, 8: 1.3, 9: 1.3, 10: 1.2, 11: 1.1  # Morning surge
        }
    },
    DayType.FRIDAY_EVE: {
        'overall': 1.0,
        'hourly_adjustments': {
            18: 1.2, 19: 1.3, 20: 1.4, 21: 1.4, 22: 1.3, 23: 1.2
        }
    },
    DayType.SATURDAY_NIGHT: {
        'overall': 1.0,
        'hourly_adjustments': {
            20: 1.3, 21: 1.4, 22: 1.5, 23: 1.5, 0: 1.4, 1: 1.3, 2: 1.2
        }
    },
    DayType.SUNDAY: {
        'overall': 0.85,
        'hourly_adjustments': {
            14: 1.1, 15: 1.15, 16: 1.1  # Afternoon family visit discoveries
        }
    },
    DayType.BANK_HOLIDAY: {
        'overall': 0.95,  # Slightly lower than weekend
        'hourly_adjustments': {
            20: 1.3, 21: 1.4, 22: 1.4, 23: 1.3  # Evening surge
        }
    },
}


# Simple demand level presets (Phase 6)
DEMAND_LEVEL_MULTIPLIERS: Dict[str, float] = {
    'Low': 0.75,
    'Normal': 1.0,
    'Busy': 1.25,
    'Surge': 1.5,
    'Major Incident': 2.0,
}


@dataclass
class DetailedArrivalConfig:
    """Per-hour, per-mode arrival counts (Phase 6, extended Phase 8).

    Used with ArrivalModel.DETAILED for full control over arrivals.
    Allows specifying exact numbers of ambulances, helicopters, and
    walk-ins for each hour of the configured period.

    Phase 8: Supports extended hours (12-72h) with pattern repetition
    for runs longer than configured period.
    """
    # Number of hours configured (12, 24, 36, 48, 60, 72)
    hours_configured: int = 24

    # hourly_counts[hour][mode] = count
    hourly_counts: Dict[int, Dict[ArrivalMode, int]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize with zeros if empty
        if not self.hourly_counts:
            for hour in range(self.hours_configured):
                self.hourly_counts[hour] = {
                    ArrivalMode.AMBULANCE: 0,
                    ArrivalMode.HELICOPTER: 0,
                    ArrivalMode.SELF_PRESENTATION: 0,
                }

    def get_rate(self, hour: int, mode: ArrivalMode) -> float:
        """Get arrival rate for specific hour and mode, with wraparound.

        For hours beyond hours_configured, the pattern repeats from hour 0.

        Args:
            hour: Hour of simulation (can exceed hours_configured).
            mode: Arrival mode (AMBULANCE, HELICOPTER, SELF_PRESENTATION).

        Returns:
            Arrival rate for that hour and mode.
        """
        effective_hour = hour % self.hours_configured
        return float(self.hourly_counts.get(effective_hour, {}).get(mode, 0))

    def set_rate(self, hour: int, mode: ArrivalMode, count: int) -> None:
        """Set arrival count for specific hour and mode."""
        if hour not in self.hourly_counts:
            self.hourly_counts[hour] = {
                ArrivalMode.AMBULANCE: 0,
                ArrivalMode.HELICOPTER: 0,
                ArrivalMode.SELF_PRESENTATION: 0,
            }
        self.hourly_counts[hour][mode] = count

    def expand_to_hours(self, target_hours: int) -> 'DetailedArrivalConfig':
        """Create new config expanded/repeated to target hours.

        Args:
            target_hours: Target number of hours for the new config.

        Returns:
            New DetailedArrivalConfig with pattern repeated to fill target_hours.
        """
        new_config = DetailedArrivalConfig(hours_configured=target_hours)

        for hour in range(target_hours):
            source_hour = hour % self.hours_configured
            new_config.hourly_counts[hour] = self.hourly_counts.get(source_hour, {
                ArrivalMode.AMBULANCE: 0,
                ArrivalMode.HELICOPTER: 0,
                ArrivalMode.SELF_PRESENTATION: 0,
            }).copy()

        return new_config

    def total_arrivals(self) -> int:
        """Calculate total arrivals across all hours and modes."""
        total = 0
        for hour_counts in self.hourly_counts.values():
            total += sum(hour_counts.values())
        return total


@dataclass
class RoutingRule:
    """Single routing probability rule.

    Defines the probability of transitioning from one node to another
    for patients of a specific priority level.
    """
    from_node: NodeType
    to_node: NodeType
    priority: Priority
    probability: float


@dataclass
class ArrivalConfig:
    """Configuration for a single arrival stream.

    Each stream has its own hourly arrival rates and triage priority mix.
    """
    mode: ArrivalMode
    hourly_rates: List[float]  # 24 values, one per hour
    triage_mix: dict  # Dict[Priority, float] - must sum to 1.0

    def __post_init__(self):
        if len(self.hourly_rates) != 24:
            raise ValueError("hourly_rates must have 24 values")
        total = sum(self.triage_mix.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"triage_mix must sum to 1.0, got {total}")


@dataclass
class NodeConfig:
    """Configuration for a service node."""
    node_type: NodeType
    capacity: int
    service_time_mean: float  # minutes
    service_time_cv: float = 0.5
    enabled: bool = True


@dataclass
class DiagnosticConfig:
    """Configuration for a diagnostic service (Phase 7).

    Models CT scanners, X-ray rooms, and phlebotomy/lab services.
    Each diagnostic has:
    - capacity: Number of scanners/machines/staff
    - process_time: Time for the actual scan/test
    - turnaround_time: Additional wait for results (especially bloods)
    - probability_by_priority: Likelihood patient needs this diagnostic

    Attributes:
        diagnostic_type: Type of diagnostic (CT, X-ray, Bloods).
        capacity: Number of scanners/rooms/phlebotomists.
        process_time_mean: Mean time for the scan/test in minutes.
        process_time_cv: Coefficient of variation for process time.
        turnaround_time_mean: Mean wait for results in minutes (e.g., lab turnaround).
        turnaround_time_cv: CV for turnaround time.
        enabled: Whether this diagnostic is available.
        probability_by_priority: Dict mapping Priority to probability patient needs this.
    """
    diagnostic_type: DiagnosticType
    capacity: int
    process_time_mean: float  # Time for actual scan/test (mins)
    process_time_cv: float = 0.3
    turnaround_time_mean: float = 0.0  # Additional wait for results (bloods)
    turnaround_time_cv: float = 0.3
    enabled: bool = True
    probability_by_priority: Dict[Priority, float] = field(default_factory=dict)


# ============== Phase 10: Aeromedical Evacuation ==============

@dataclass
class HEMSConfig:
    """HEMS (Helicopter Emergency Medical Service) configuration (Phase 10).

    HEMS provides flexible, rapid evacuation for critically ill patients.
    Unlike fixed-wing, HEMS can typically operate on-demand within
    daylight hours (weather permitting).

    Attributes:
        enabled: Whether HEMS evacuation is available.
        slots_per_day: Number of HEMS evacuations available per day.
        operating_start_hour: First hour HEMS can operate (e.g., 7 = 07:00).
        operating_end_hour: Last hour HEMS can launch (e.g., 21 = 21:00).
        stabilisation_mins: (min, max) clinical preparation time.
        transfer_to_helipad_mins: (min, max) transfer time to helipad.
        flight_duration_mins: (min, max) flight duration.
    """
    enabled: bool = True
    slots_per_day: int = 6
    operating_start_hour: int = 7   # 07:00
    operating_end_hour: int = 21    # 21:00 (last launch)
    stabilisation_mins: tuple = (30.0, 120.0)  # 30min - 2h
    transfer_to_helipad_mins: tuple = (15.0, 45.0)
    flight_duration_mins: tuple = (15.0, 60.0)


@dataclass
class FixedWingConfig:
    """Fixed-wing aeromedical evacuation configuration (Phase 10).

    Slots are configured per 12-hour segment, allowing modelling of:
    - Single daily departure (1 slot in segment 0, 0 in segment 1)
    - Twice daily (1 slot each segment)
    - Variable patterns (more slots on weekdays)

    Pattern repeats after configured segments (default: 2 segments = 1 day).
    Maximum 10 segments = 5 days.

    Attributes:
        enabled: Whether fixed-wing evacuation is available.
        slots_per_segment: List of slots per 12h segment (max 10 segments = 5 days).
        departure_hour_in_segment: Hour within segment for departure (0-11).
        cutoff_hours_before: Hours before departure patient must be ready.
        stabilisation_mins: (min, max) clinical preparation time.
        transport_to_airfield_mins: (min, max) transfer to airfield.
        flight_duration_mins: (min, max) flight duration.
    """
    enabled: bool = False
    slots_per_segment: List[int] = field(default_factory=lambda: [1, 0])
    departure_hour_in_segment: int = 2  # 2h into segment (02:00 or 14:00)
    cutoff_hours_before: int = 4  # Must be ready 4h before departure
    stabilisation_mins: tuple = (120.0, 240.0)  # 2-4 hours
    transport_to_airfield_mins: tuple = (30.0, 90.0)  # 30-90 min
    flight_duration_mins: tuple = (60.0, 180.0)  # 1-3 hours


@dataclass
class MissedSlotConfig:
    """Configuration for handling missed aeromedical slots (Phase 10).

    CLINICAL CONTEXT:
    When a patient misses their scheduled evacuation slot (e.g., aircraft
    departed before patient ready, or no slots remaining today), they must
    wait for the next available slot.

    During this wait:
    - Patient HOLDS their ward bed (cannot be discharged, cannot free bed)
    - Bed state is BLOCKED (not OCCUPIED) - blocked by downstream constraint
    - This creates upstream blocking (new admissions wait for ward beds)
    - The "missed slot cascade" can significantly impact hospital flow

    RE-STABILISATION:
    In most cases, a patient who was stabilised for evacuation remains stable.
    Re-stabilisation is typically only needed if:
    - Significant time has passed (>24h)
    - Patient's condition has changed
    - Different aircraft type with different requirements

    Default: No re-stabilisation (patient waits in stable condition)

    Attributes:
        requires_restabilisation: Whether patient needs re-prep after missing slot.
        restabilisation_factor: Fraction of original stabilisation time if needed.
        max_wait_before_restab_hours: Auto re-stabilise after this time (None to disable).
    """
    requires_restabilisation: bool = False
    restabilisation_factor: float = 0.3  # 30% of original stabilisation time
    max_wait_before_restab_hours: Optional[float] = 24.0


@dataclass
class AeromedConfig:
    """Aeromedical evacuation configuration (Phase 10).

    Models patient evacuation via HEMS (helicopter) or fixed-wing aircraft
    to specialist receiving facilities.

    Attributes:
        enabled: Whether aeromedical evacuation is modelled.
        p1_aeromed_probability: Probability that a P1/Resus patient requires aeromed.
        fixedwing_proportion: Of aeromed patients, proportion going fixed-wing (vs HEMS).
        hems: HEMS sub-configuration.
        fixedwing: Fixed-wing sub-configuration.
        missed_slot: Missed slot behaviour configuration.
    """
    enabled: bool = False
    p1_aeromed_probability: float = 0.05  # 5% of P1s need aeromed
    fixedwing_proportion: float = 0.3  # 30% fixed-wing, 70% HEMS
    hems: HEMSConfig = field(default_factory=HEMSConfig)
    fixedwing: FixedWingConfig = field(default_factory=FixedWingConfig)
    missed_slot: MissedSlotConfig = field(default_factory=MissedSlotConfig)


@dataclass
class TransferConfig:
    """Configuration for inter-facility transfers (Phase 7).

    Models transfers to specialist centres (Major Trauma, Neuro, Cardiac, etc.).
    Includes land ambulance and helicopter transfer options.

    Attributes:
        probability_by_priority: Probability of transfer by patient priority.
        n_transfer_ambulances: Dedicated transfer vehicle count.
        n_transfer_helicopters: Air ambulance availability.
        decision_to_request_mean: Time to arrange transfer (mins).
        land_ambulance_wait_mean: Wait for land vehicle (mins).
        helicopter_wait_mean: Wait for air ambulance (mins).
        land_transfer_time_mean: Journey time by road (mins).
        helicopter_transfer_time_mean: Journey time by air (mins).
        helicopter_proportion_p1: Proportion of P1 transfers by helicopter.
        enabled: Whether transfers are modelled.
    """
    probability_by_priority: Dict[Priority, float] = field(default_factory=lambda: {
        Priority.P1_IMMEDIATE: 0.05,     # 5% of P1s need specialist transfer
        Priority.P2_VERY_URGENT: 0.02,
        Priority.P3_URGENT: 0.005,
        Priority.P4_STANDARD: 0.001,
    })

    # Transfer resources
    n_transfer_ambulances: int = 2
    n_transfer_helicopters: int = 1

    # Times (minutes)
    decision_to_request_mean: float = 30.0   # Time to arrange transfer
    land_ambulance_wait_mean: float = 45.0   # Wait for vehicle
    helicopter_wait_mean: float = 30.0       # Air ambulance response
    land_transfer_time_mean: float = 60.0    # Journey time by road
    helicopter_transfer_time_mean: float = 25.0  # Journey time by air

    # Proportion by transfer type (for P1)
    helicopter_proportion_p1: float = 0.3  # 30% of P1 transfers by air

    enabled: bool = True


@dataclass
class Scenario:
    """Configuration for a simulation scenario.

    Contains all parameters needed to run a simulation, including
    horizon settings, resource counts, service time parameters,
    and random seed for reproducibility.

    Attributes:
        run_length: Simulation duration in minutes (default 480 = 8 hours).
        warm_up: Warm-up period in minutes (results discarded).
        arrival_rate: Patient arrivals per hour.
        n_resus_bays: Number of Resus bays available.
        resus_mean: Mean Resus length of stay in minutes.
        resus_cv: Coefficient of variation for Resus LoS.
        random_seed: Master seed for reproducibility.
    """

    # Horizon settings
    run_length: float = 480.0  # 8 hours in minutes
    warm_up: float = 0.0

    # Arrivals (constant rate for Phase 1)
    arrival_rate: float = 4.0  # patients per hour

    # Resources
    n_resus_bays: int = 2

    # Service times (minutes)
    resus_mean: float = 45.0
    resus_cv: float = 0.5  # Coefficient of variation

    # Reproducibility
    random_seed: int = 42

    # RNG streams (created in __post_init__)
    rng_arrivals: Optional[np.random.Generator] = None
    rng_service: Optional[np.random.Generator] = None
    rng_routing: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        """Initialize separate RNG streams for each stochastic element."""
        self.rng_arrivals = np.random.default_rng(self.random_seed)
        self.rng_service = np.random.default_rng(self.random_seed + 1)
        self.rng_routing = np.random.default_rng(self.random_seed + 2)

    @property
    def mean_iat(self) -> float:
        """Mean inter-arrival time in minutes."""
        return 60.0 / self.arrival_rate

    def clone_with_seed(self, new_seed: int) -> "Scenario":
        """Create a copy of this scenario with a different seed.

        Args:
            new_seed: The new random seed to use.

        Returns:
            A new Scenario instance with updated seed and fresh RNGs.
        """
        return Scenario(
            run_length=self.run_length,
            warm_up=self.warm_up,
            arrival_rate=self.arrival_rate,
            n_resus_bays=self.n_resus_bays,
            resus_mean=self.resus_mean,
            resus_cv=self.resus_cv,
            random_seed=new_seed,
        )


@dataclass
class FullScenario:
    """Extended scenario for full A&E pathway simulation.

    Includes multiple acuity streams, triage, and disposition routing.

    Phase 5: Simplified ED with single bay pool and priority queuing.
    Priority (P1-P4) determines service order, not destination.

    Attributes:
        run_length: Simulation duration in minutes.
        warm_up: Warm-up period in minutes.
        arrival_rate: Total patient arrivals per hour.
        random_seed: Master seed for reproducibility.

        Acuity mix (must sum to 1.0):
        p_resus: Proportion of Resus-level patients.
        p_majors: Proportion of Majors-level patients.
        p_minors: Proportion of Minors-level patients.

        Resources:
        n_triage: Number of triage clinicians (nurses, ANPs, PAs, or doctors).
        n_ed_bays: Number of ED bays (single pool with priority queuing).

        Service times (mean, cv) in minutes:
        triage_mean, triage_cv: Triage duration.
        ed_service_mean, ed_service_cv: ED treatment duration.
        boarding_mean, boarding_cv: Boarding time for admits.
    """

    # Horizon settings
    run_length: float = 480.0  # 8 hours
    warm_up: float = 60.0  # 1 hour warm-up

    # Arrivals
    arrival_rate: float = 6.0  # patients per hour (total)

    # Acuity mix (must sum to 1.0)
    p_resus: float = 0.05
    p_majors: float = 0.55
    p_minors: float = 0.40

    # Resources - Phase 5 simplified ED
    n_triage: int = 2
    n_ed_bays: int = 20  # Single pool replaces resus/majors/minors

    # Handover bays (Phase 5b)
    n_handover_bays: int = 4
    handover_time_mean: float = 15.0  # minutes
    handover_time_cv: float = 0.3

    # Fleet controls (Phase 5c)
    n_ambulances: int = 10
    n_helicopters: int = 2
    ambulance_turnaround_mins: float = 45.0
    helicopter_turnaround_mins: float = 90.0
    litters_per_ambulance: int = 1  # Future: could be 2 for MCIs

    # Demand scaling (Phase 5d)
    demand_multiplier: float = 1.0  # Global scaling for all streams
    ambulance_rate_multiplier: float = 1.0  # Per-stream scaling
    helicopter_rate_multiplier: float = 1.0
    walkin_rate_multiplier: float = 1.0

    # Arrival model configuration (Phase 6)
    arrival_model: ArrivalModel = ArrivalModel.PROFILE_24H
    day_type: DayType = DayType.WEEKDAY
    detailed_arrivals: Optional[DetailedArrivalConfig] = None

    # Bed management (Phase 5e)
    bed_turnaround_mins: float = 10.0  # Cleaning time after patient leaves

    # Service times - Triage
    triage_mean: float = 5.0
    triage_cv: float = 0.3

    # Service times - ED (unified, varies by priority in practice)
    ed_service_mean: float = 60.0
    ed_service_cv: float = 0.6

    # Boarding time (for admitted patients awaiting bed)
    boarding_mean: float = 120.0
    boarding_cv: float = 1.0

    # Reproducibility
    random_seed: int = 42

    # RNG streams (created in __post_init__)
    rng_arrivals: Optional[dict] = None  # Dict[ArrivalMode, Generator]
    rng_acuity: Optional[np.random.Generator] = None
    rng_triage: Optional[np.random.Generator] = None
    rng_treatment: Optional[np.random.Generator] = None
    rng_boarding: Optional[np.random.Generator] = None
    rng_disposition: Optional[np.random.Generator] = None
    rng_routing: Optional[np.random.Generator] = None
    rng_handover: Optional[np.random.Generator] = None  # Phase 5b

    # Routing configuration (created in __post_init__ if not provided)
    routing: List[RoutingRule] = field(default_factory=list)

    # Arrival stream configurations (created in __post_init__ if not provided)
    arrival_configs: List[ArrivalConfig] = field(default_factory=list)

    # Node configurations (created in __post_init__ if not provided)
    node_configs: dict = field(default_factory=dict)  # Dict[NodeType, NodeConfig]

    # Diagnostic configurations (Phase 7)
    diagnostic_configs: dict = field(default_factory=dict)  # Dict[DiagnosticType, DiagnosticConfig]

    # Transfer configuration (Phase 7)
    transfer_config: Optional[TransferConfig] = None

    # Phase 8: Detailed downstream configs (optional - override node_configs)
    itu_config: Optional[ITUConfig] = None
    ward_config: Optional[WardConfig] = None
    theatre_config: Optional[TheatreConfig] = None

    # Phase 9: Downstream simulation controls
    downstream_enabled: bool = False  # Enable Theatre/ITU/Ward resource consumption
    release_stable_to_wait: bool = False  # P3/P4 ward-bound patients release ED bay early

    # Phase 10: Aeromedical evacuation configuration
    aeromed_config: Optional[AeromedConfig] = None

    # Phase 11: Major Incident configuration
    major_incident_config: Optional[MajorIncidentConfig] = None

    # RNG for diagnostics (Phase 7) - created in __post_init__
    rng_diagnostics: Optional[np.random.Generator] = None
    rng_transfer: Optional[np.random.Generator] = None
    rng_itu: Optional[np.random.Generator] = None  # Phase 8
    rng_ward: Optional[np.random.Generator] = None  # Phase 8
    rng_theatre: Optional[np.random.Generator] = None  # Phase 8
    rng_aeromed: Optional[np.random.Generator] = None  # Phase 10
    rng_incident: Optional[np.random.Generator] = None  # Phase 11

    def __post_init__(self) -> None:
        """Initialize separate RNG streams and validate parameters."""
        # Validate acuity mix
        total = self.p_resus + self.p_majors + self.p_minors
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Acuity proportions must sum to 1.0, got {total}")

        # Create separate RNG streams per arrival mode
        self.rng_arrivals = {
            mode: np.random.default_rng(self.random_seed + i)
            for i, mode in enumerate(ArrivalMode)
        }
        self.rng_acuity = np.random.default_rng(self.random_seed + 10)
        self.rng_triage = np.random.default_rng(self.random_seed + 11)
        self.rng_treatment = np.random.default_rng(self.random_seed + 12)
        self.rng_boarding = np.random.default_rng(self.random_seed + 13)
        self.rng_disposition = np.random.default_rng(self.random_seed + 14)
        self.rng_routing = np.random.default_rng(self.random_seed + 15)
        self.rng_handover = np.random.default_rng(self.random_seed + 16)  # Phase 5b
        self.rng_diagnostics = np.random.default_rng(self.random_seed + 17)  # Phase 7
        self.rng_transfer = np.random.default_rng(self.random_seed + 18)  # Phase 7
        self.rng_itu = np.random.default_rng(self.random_seed + 19)  # Phase 8
        self.rng_ward = np.random.default_rng(self.random_seed + 20)  # Phase 8
        self.rng_theatre = np.random.default_rng(self.random_seed + 21)  # Phase 8
        self.rng_aeromed = np.random.default_rng(self.random_seed + 22)  # Phase 10
        self.rng_incident = np.random.default_rng(self.random_seed + 23)  # Phase 11

        # Initialize default Phase 8 configs if not provided
        if self.itu_config is None:
            self.itu_config = ITUConfig()
        if self.ward_config is None:
            self.ward_config = WardConfig()
        if self.theatre_config is None:
            self.theatre_config = TheatreConfig()

        # Initialize default routing if not provided
        if not self.routing:
            self.routing = self._default_routing()
        self._validate_routing()

        # Initialize default arrival configs if not provided
        if not self.arrival_configs:
            self.arrival_configs = self._default_arrival_configs()

        # Initialize default node configs if not provided
        if not self.node_configs:
            self.node_configs = self._default_node_configs()

        # Initialize default diagnostic configs if not provided (Phase 7)
        if not self.diagnostic_configs:
            self.diagnostic_configs = self._default_diagnostic_configs()

        # Initialize default transfer config if not provided (Phase 7)
        if self.transfer_config is None:
            self.transfer_config = TransferConfig()

        # Initialize default aeromed config if not provided (Phase 10)
        if self.aeromed_config is None:
            self.aeromed_config = AeromedConfig()

    def _default_node_configs(self) -> dict:
        """Default node configurations.

        Phase 5: Single ED_BAYS pool with priority queuing.
        """
        return {
            NodeType.TRIAGE: NodeConfig(NodeType.TRIAGE, capacity=self.n_triage, service_time_mean=self.triage_mean),
            NodeType.ED_BAYS: NodeConfig(NodeType.ED_BAYS, capacity=self.n_ed_bays, service_time_mean=self.ed_service_mean),
            NodeType.SURGERY: NodeConfig(NodeType.SURGERY, capacity=2, service_time_mean=150),
            NodeType.ITU: NodeConfig(NodeType.ITU, capacity=6, service_time_mean=720),
            NodeType.WARD: NodeConfig(NodeType.WARD, capacity=30, service_time_mean=480),
        }

    def _default_diagnostic_configs(self) -> dict:
        """Default diagnostic configurations (Phase 7).

        CT scanner: 2 scanners, 20 min scan, 30 min radiologist report
        X-ray: 3 rooms, 10 min, 15 min for report
        Bloods: 5 phlebotomists, 5 min draw, 45 min lab turnaround
        """
        return {
            DiagnosticType.CT_SCAN: DiagnosticConfig(
                diagnostic_type=DiagnosticType.CT_SCAN,
                capacity=2,                    # 2 CT scanners
                process_time_mean=20.0,        # 20 min scan time
                turnaround_time_mean=30.0,     # 30 min for radiologist report
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.70,    # Most P1s need CT
                    Priority.P2_VERY_URGENT: 0.40,
                    Priority.P3_URGENT: 0.15,
                    Priority.P4_STANDARD: 0.05,
                }
            ),
            DiagnosticType.XRAY: DiagnosticConfig(
                diagnostic_type=DiagnosticType.XRAY,
                capacity=3,                    # 3 X-ray rooms
                process_time_mean=10.0,        # 10 min
                turnaround_time_mean=15.0,     # 15 min for report
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.30,
                    Priority.P2_VERY_URGENT: 0.35,
                    Priority.P3_URGENT: 0.40,
                    Priority.P4_STANDARD: 0.25,
                }
            ),
            DiagnosticType.BLOODS: DiagnosticConfig(
                diagnostic_type=DiagnosticType.BLOODS,
                capacity=5,                    # Phlebotomy capacity
                process_time_mean=5.0,         # 5 min to take bloods
                turnaround_time_mean=45.0,     # 45 min lab turnaround
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.90,
                    Priority.P2_VERY_URGENT: 0.80,
                    Priority.P3_URGENT: 0.50,
                    Priority.P4_STANDARD: 0.20,
                }
            ),
        }

    def _default_routing(self) -> List[RoutingRule]:
        """Default ED disposition routing.

        Phase 5: All patients go through single ED_BAYS pool.
        Routing from ED_BAYS depends on priority level.
        """
        rules = []

        # ED_BAYS routing by priority (all routes from single ED pool)
        # P1: Surgery 30%, ITU 40%, Ward 20%, Exit 10%
        rules.extend([
            RoutingRule(NodeType.ED_BAYS, NodeType.SURGERY, Priority.P1_IMMEDIATE, 0.30),
            RoutingRule(NodeType.ED_BAYS, NodeType.ITU, Priority.P1_IMMEDIATE, 0.40),
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P1_IMMEDIATE, 0.20),
            RoutingRule(NodeType.ED_BAYS, NodeType.EXIT, Priority.P1_IMMEDIATE, 0.10),
        ])

        # P2: Surgery 15%, ITU 10%, Ward 45%, Exit 30%
        rules.extend([
            RoutingRule(NodeType.ED_BAYS, NodeType.SURGERY, Priority.P2_VERY_URGENT, 0.15),
            RoutingRule(NodeType.ED_BAYS, NodeType.ITU, Priority.P2_VERY_URGENT, 0.10),
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P2_VERY_URGENT, 0.45),
            RoutingRule(NodeType.ED_BAYS, NodeType.EXIT, Priority.P2_VERY_URGENT, 0.30),
        ])

        # P3: Surgery 5%, ITU 2%, Ward 25%, Exit 68%
        rules.extend([
            RoutingRule(NodeType.ED_BAYS, NodeType.SURGERY, Priority.P3_URGENT, 0.05),
            RoutingRule(NodeType.ED_BAYS, NodeType.ITU, Priority.P3_URGENT, 0.02),
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P3_URGENT, 0.25),
            RoutingRule(NodeType.ED_BAYS, NodeType.EXIT, Priority.P3_URGENT, 0.68),
        ])

        # P4: Surgery 2%, ITU 0%, Ward 5%, Exit 93%
        rules.extend([
            RoutingRule(NodeType.ED_BAYS, NodeType.SURGERY, Priority.P4_STANDARD, 0.02),
            RoutingRule(NodeType.ED_BAYS, NodeType.ITU, Priority.P4_STANDARD, 0.0),
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P4_STANDARD, 0.05),
            RoutingRule(NodeType.ED_BAYS, NodeType.EXIT, Priority.P4_STANDARD, 0.93),
        ])

        # Downstream routing (same for all priorities)
        for p in Priority:
            # Surgery → ITU 40%, Ward 60%
            rules.append(RoutingRule(NodeType.SURGERY, NodeType.ITU, p, 0.40))
            rules.append(RoutingRule(NodeType.SURGERY, NodeType.WARD, p, 0.60))
            # ITU → Ward 100%
            rules.append(RoutingRule(NodeType.ITU, NodeType.WARD, p, 1.0))
            # Ward → Exit 100%
            rules.append(RoutingRule(NodeType.WARD, NodeType.EXIT, p, 1.0))

        return rules

    def _default_arrival_configs(self) -> List[ArrivalConfig]:
        """Default arrival configurations for 3 streams."""
        return [
            # Ambulance: moderate volume, higher acuity
            ArrivalConfig(
                mode=ArrivalMode.AMBULANCE,
                hourly_rates=[2, 1.5, 1, 1, 1.5, 2, 3, 4, 5, 5.5, 5, 4.5,
                             4, 4, 4, 4.5, 5, 5.5, 5, 4, 3, 2.5, 2, 2],
                triage_mix={
                    Priority.P1_IMMEDIATE: 0.15,
                    Priority.P2_VERY_URGENT: 0.40,
                    Priority.P3_URGENT: 0.35,
                    Priority.P4_STANDARD: 0.10,
                }
            ),
            # Helicopter: low volume, high acuity
            ArrivalConfig(
                mode=ArrivalMode.HELICOPTER,
                hourly_rates=[0.1] * 24,
                triage_mix={
                    Priority.P1_IMMEDIATE: 0.70,
                    Priority.P2_VERY_URGENT: 0.25,
                    Priority.P3_URGENT: 0.05,
                    Priority.P4_STANDARD: 0.0,
                }
            ),
            # Self-presentation (walk-in): high volume, low acuity
            ArrivalConfig(
                mode=ArrivalMode.SELF_PRESENTATION,
                hourly_rates=[1, 0.5, 0.3, 0.2, 0.3, 0.5, 1, 2, 3, 4, 4.5, 4,
                             3.5, 3, 3, 3.5, 4, 4.5, 4, 3, 2, 1.5, 1, 1],
                triage_mix={
                    Priority.P1_IMMEDIATE: 0.02,
                    Priority.P2_VERY_URGENT: 0.15,
                    Priority.P3_URGENT: 0.45,
                    Priority.P4_STANDARD: 0.38,
                }
            ),
        ]

    def _validate_routing(self) -> None:
        """Ensure routing probabilities sum to 1.0 for each source/priority."""
        sums = defaultdict(float)
        for rule in self.routing:
            sums[(rule.from_node, rule.priority)] += rule.probability

        for (node, priority), total in sums.items():
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"Routing from {node.name} for {priority.name} "
                    f"sums to {total:.3f}, expected 1.0"
                )

    def get_next_node(self, current_node: NodeType, priority: Priority) -> NodeType:
        """Sample next node based on routing probabilities."""
        applicable = [r for r in self.routing
                     if r.from_node == current_node and r.priority == priority]

        if not applicable:
            return NodeType.EXIT  # Fallback

        probs = [r.probability for r in applicable]
        nodes = [r.to_node for r in applicable]

        # Use scenario's routing RNG
        idx = self.rng_routing.choice(len(nodes), p=probs)
        return nodes[idx]

    @property
    def mean_iat(self) -> float:
        """Mean inter-arrival time in minutes."""
        return 60.0 / self.arrival_rate

    def get_treatment_params(self) -> tuple:
        """Get ED treatment time parameters.

        Phase 5: Single ED pool uses unified service time parameters.

        Returns:
            Tuple of (mean, cv) for treatment time.
        """
        return (self.ed_service_mean, self.ed_service_cv)

    def get_rate_multiplier(self, mode: ArrivalMode) -> float:
        """Get effective rate multiplier for an arrival mode.

        Phase 5d: Combines global demand_multiplier with per-stream multiplier.

        Args:
            mode: The arrival mode.

        Returns:
            Combined multiplier for this mode's arrival rate.
        """
        stream_multiplier = {
            ArrivalMode.AMBULANCE: self.ambulance_rate_multiplier,
            ArrivalMode.HELICOPTER: self.helicopter_rate_multiplier,
            ArrivalMode.SELF_PRESENTATION: self.walkin_rate_multiplier,
        }.get(mode, 1.0)

        return self.demand_multiplier * stream_multiplier

    def get_effective_arrival_rate(self, config: ArrivalConfig, hour: int) -> float:
        """Get arrival rate based on selected model and day type (Phase 6).

        Computes the effective arrival rate considering:
        - Simple model: Average rate across all hours * demand multiplier
        - Profile 24h: Hourly rates with day type adjustments
        - Detailed model: Exact per-mode, per-hour counts

        Args:
            config: ArrivalConfig for the stream (mode, hourly_rates, triage_mix).
            hour: Hour of day (0-23).

        Returns:
            Effective arrival rate (patients per hour) for this config at this hour.
        """
        if self.arrival_model == ArrivalModel.SIMPLE:
            # Use base rate * demand multiplier only
            base_rate = sum(config.hourly_rates) / 24  # Average
            return base_rate * self.demand_multiplier

        elif self.arrival_model == ArrivalModel.PROFILE_24H:
            # Use 24-hour profile with day type adjustments
            base_rate = config.hourly_rates[hour]
            day_config = DAY_TYPE_MULTIPLIERS[self.day_type]

            rate = base_rate * day_config['overall']
            rate *= day_config['hourly_adjustments'].get(hour, 1.0)
            rate *= self.demand_multiplier

            # Apply per-stream multiplier
            stream_mult = {
                ArrivalMode.AMBULANCE: self.ambulance_rate_multiplier,
                ArrivalMode.HELICOPTER: self.helicopter_rate_multiplier,
                ArrivalMode.SELF_PRESENTATION: self.walkin_rate_multiplier,
            }
            rate *= stream_mult.get(config.mode, 1.0)

            return rate

        elif self.arrival_model == ArrivalModel.DETAILED:
            # Use detailed per-mode, per-hour counts
            if self.detailed_arrivals:
                return self.detailed_arrivals.get_rate(hour, config.mode)
            return 0.0

        return 0.0

    def clone_with_seed(self, new_seed: int) -> "FullScenario":
        """Create a copy with a different seed."""
        return FullScenario(
            run_length=self.run_length,
            warm_up=self.warm_up,
            arrival_rate=self.arrival_rate,
            p_resus=self.p_resus,
            p_majors=self.p_majors,
            p_minors=self.p_minors,
            n_triage=self.n_triage,
            n_ed_bays=self.n_ed_bays,
            n_handover_bays=self.n_handover_bays,
            handover_time_mean=self.handover_time_mean,
            handover_time_cv=self.handover_time_cv,
            n_ambulances=self.n_ambulances,
            n_helicopters=self.n_helicopters,
            ambulance_turnaround_mins=self.ambulance_turnaround_mins,
            helicopter_turnaround_mins=self.helicopter_turnaround_mins,
            litters_per_ambulance=self.litters_per_ambulance,
            demand_multiplier=self.demand_multiplier,
            ambulance_rate_multiplier=self.ambulance_rate_multiplier,
            helicopter_rate_multiplier=self.helicopter_rate_multiplier,
            walkin_rate_multiplier=self.walkin_rate_multiplier,
            arrival_model=self.arrival_model,
            day_type=self.day_type,
            detailed_arrivals=self.detailed_arrivals,
            bed_turnaround_mins=self.bed_turnaround_mins,
            triage_mean=self.triage_mean,
            triage_cv=self.triage_cv,
            ed_service_mean=self.ed_service_mean,
            ed_service_cv=self.ed_service_cv,
            boarding_mean=self.boarding_mean,
            boarding_cv=self.boarding_cv,
            random_seed=new_seed,
            # Phase 7: Copy diagnostic and transfer configs
            diagnostic_configs=self.diagnostic_configs,
            transfer_config=self.transfer_config,
            # Phase 8: Copy downstream configs
            itu_config=self.itu_config,
            ward_config=self.ward_config,
            theatre_config=self.theatre_config,
            # Phase 9: Copy downstream controls
            downstream_enabled=self.downstream_enabled,
            release_stable_to_wait=self.release_stable_to_wait,
            # Phase 10: Copy aeromed config
            aeromed_config=self.aeromed_config,
            # Phase 11: Copy major incident config
            major_incident_config=self.major_incident_config,
        )
