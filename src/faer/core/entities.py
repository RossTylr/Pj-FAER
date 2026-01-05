"""Core entity definitions for the simulation.

This module contains enums and basic types that are used across
the codebase, placed here to avoid circular imports.
"""

from enum import IntEnum


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
    RESUS = 1
    MAJORS = 2
    MINORS = 3
    SURGERY = 4
    ITU = 5
    WARD = 6
    EXIT = 7


class ArrivalMode(IntEnum):
    """Patient arrival stream type."""
    AMBULANCE = 1
    HELICOPTER = 2
    SELF_PRESENTATION = 3  # Walk-in
