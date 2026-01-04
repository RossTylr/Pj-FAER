"""Core foundation layer: scenario configuration, arrivals, distributions."""

from faer.core.scenario import Scenario, FullScenario
from faer.core.arrivals import (
    ArrivalProfile,
    NSPPThinning,
    load_default_profile,
    create_constant_profile,
)

__all__ = [
    "Scenario",
    "FullScenario",
    "ArrivalProfile",
    "NSPPThinning",
    "load_default_profile",
    "create_constant_profile",
]
