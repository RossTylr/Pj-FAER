"""Core entity definitions for the simulation.

This module contains enums and basic types that are used across
the codebase, placed here to avoid circular imports.
"""

from enum import Enum, IntEnum


class Priority(IntEnum):
    """Triage priority levels. Lower value = more urgent.

    Based on standard ED triage categories with time-to-treatment targets.
    """
    P1_IMMEDIATE = 1      # Immediate resuscitation (0 min target)
    P2_VERY_URGENT = 2    # Very urgent (10 min target)
    P3_URGENT = 3         # Urgent (60 min target)
    P4_STANDARD = 4       # Standard (120 min target)


class NodeType(IntEnum):
    """Service node types in the hospital network."""
    TRIAGE = 1
    ED_BAYS = 2      # Single ED bay pool (replaces Resus/Majors/Minors)
    SURGERY = 3
    ITU = 4
    WARD = 5
    EXIT = 6


class ArrivalMode(IntEnum):
    """Patient arrival stream type."""
    AMBULANCE = 1
    HELICOPTER = 2
    SELF_PRESENTATION = 3  # Walk-in


class BedState(IntEnum):
    """Bed states for operational tracking (Phase 5e).

    Used to track the operational state of individual beds/bays
    for capacity planning and visualization.
    """
    EMPTY = 1        # Available for new patient
    OCCUPIED = 2     # Patient receiving active care
    BLOCKED = 3      # Patient boarding (waiting for downstream bed)
    CLEANING = 4     # Turnaround after patient leaves


class ArrivalModel(Enum):
    """Arrival configuration complexity levels (Phase 6).

    Three tiers of control over arrival patterns:
    - SIMPLE: Single demand multiplier for quick scenario testing
    - PROFILE_24H: Hourly rates with day type presets
    - DETAILED: Per-mode, per-hour explicit control
    """
    SIMPLE = "simple"           # Single demand multiplier
    PROFILE_24H = "profile_24h" # Hourly rates, same for all modes
    DETAILED = "detailed"       # Per-mode, per-hour control


class DayType(Enum):
    """Day type presets affecting arrival patterns (Phase 6).

    NHS patterns vary significantly by day of week:
    - Monday: Morning surge from weekend backlog
    - Friday/Saturday: Evening/night peaks (alcohol, accidents)
    - Sunday: Quieter overall with afternoon family visit discoveries
    """
    WEEKDAY = "weekday"           # Tue-Thu baseline
    MONDAY = "monday"             # +20% morning surge
    FRIDAY_EVE = "friday_eve"     # +30% evening
    SATURDAY_NIGHT = "sat_night"  # +40% night
    SUNDAY = "sunday"             # -15% overall
    BANK_HOLIDAY = "bank_holiday" # Weekend + 10%


class DiagnosticType(IntEnum):
    """Diagnostic service types (Phase 7).

    Models key diagnostic resources that can become bottlenecks:
    - CT_SCAN: CT scanner (high priority for P1/P2, slower)
    - XRAY: X-ray rooms (moderate priority, faster)
    - BLOODS: Phlebotomy + lab (high volume, has turnaround time)
    """
    CT_SCAN = 1
    XRAY = 2
    BLOODS = 3
    # Future: MRI = 4, ULTRASOUND = 5


class TransferType(IntEnum):
    """Inter-facility transfer types (Phase 7).

    Transfer mode affects wait time and journey duration.
    """
    LAND_AMBULANCE = 1      # Standard ambulance transfer
    CRITICAL_CARE = 2       # With doctor/nurse escort
    HELICOPTER = 3          # Air ambulance / coastguard


class TransferDestination(IntEnum):
    """Specialist receiving facilities (Phase 7).

    Tracks where patients are transferred for planning/reporting.
    """
    MAJOR_TRAUMA_CENTRE = 1
    NEUROSURGERY = 2
    CARDIAC_CENTRE = 3
    BURNS_UNIT = 4
    PAEDIATRIC_ICU = 5
    REGIONAL_ICU = 6
