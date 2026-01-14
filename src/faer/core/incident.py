"""Major Incident configuration and casualty profiles.

This module provides the overlay system for simulating major incidents
(mass casualty events) on top of normal ED operations. The incident
acts as an additional arrival stream that injects surge casualties
at a configurable time with specified severity patterns.

Key concepts:
- OVERLAY: Normal arrivals continue; incident adds surge on top
- PRE-CALCULATED: Arrival times generated at simulation start for reproducibility
- PROFILE-BASED: Casualty mix determined by incident type (RTA, CBRN, etc.)

Phase 11: Major Incident Module
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np

from faer.core.entities import Priority


class IncidentArrivalPattern(Enum):
    """Temporal distribution of incident casualty arrivals.

    BOLUS: Front-loaded surge (60% in first 20% of duration)
           Models initial wave from scene with rapid transport.

    WAVES: Multiple peaks at regular intervals
           Models phased evacuation or multiple vehicle arrivals.

    SUSTAINED: Uniform elevated rate throughout duration
               Models prolonged incident with steady casualty flow.
    """
    BOLUS = "bolus"
    WAVES = "waves"
    SUSTAINED = "sustained"


class CasualtyProfile(Enum):
    """Casualty mix profiles based on incident type.

    Each profile defines priority distribution and special requirements.
    Priority mix based on published MCI data and clinical experience.
    """
    GENERIC = "generic"    # Standard MCI - balanced distribution
    BLAST = "blast"        # Explosion/IED - high P1, blast lung injuries
    RTA = "rta"            # Road traffic accident - moderate trauma
    CBRN = "cbrn"          # Chemical/biological/radiological/nuclear - requires decon
    BURNS = "burns"        # Fire/industrial - burns unit referral likely
    COMBAT = "combat"      # Penetrating trauma focus - military/stabbing


# Priority distributions for each casualty profile
# Based on published MCI data and UK Major Incident Medical Management guidelines
CASUALTY_PROFILES: Dict[CasualtyProfile, Dict] = {
    CasualtyProfile.GENERIC: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.20,
            Priority.P2_VERY_URGENT: 0.35,
            Priority.P3_URGENT: 0.35,
            Priority.P4_STANDARD: 0.10,
        },
        "requires_decon": False,
        "description": "Standard MCI distribution - balanced severity mix",
        "typical_scenarios": ["Building collapse", "Stadium incident", "Train derailment"],
    },
    CasualtyProfile.BLAST: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.30,
            Priority.P2_VERY_URGENT: 0.40,
            Priority.P3_URGENT: 0.25,
            Priority.P4_STANDARD: 0.05,
        },
        "requires_decon": False,
        "description": "Explosion/IED - high immediate priority, blast lung/fragmentation",
        "typical_scenarios": ["Terrorist attack", "Gas explosion", "Industrial blast"],
    },
    CasualtyProfile.RTA: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.15,
            Priority.P2_VERY_URGENT: 0.45,
            Priority.P3_URGENT: 0.30,
            Priority.P4_STANDARD: 0.10,
        },
        "requires_decon": False,
        "description": "Road traffic collision - blunt trauma, fractures predominate",
        "typical_scenarios": ["Multi-vehicle pileup", "Bus crash", "Coach accident"],
    },
    CasualtyProfile.CBRN: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.25,
            Priority.P2_VERY_URGENT: 0.35,
            Priority.P3_URGENT: 0.30,
            Priority.P4_STANDARD: 0.10,
        },
        "requires_decon": True,  # Critical: decontamination required before triage
        "description": "Chemical/biological - decontamination delay before ED entry",
        "typical_scenarios": ["Chemical spill", "Industrial leak", "Deliberate release"],
    },
    CasualtyProfile.BURNS: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.20,
            Priority.P2_VERY_URGENT: 0.50,
            Priority.P3_URGENT: 0.25,
            Priority.P4_STANDARD: 0.05,
        },
        "requires_decon": False,
        "description": "Burns casualties - high P2 load, specialist referral likely",
        "typical_scenarios": ["Building fire", "Industrial fire", "Tanker explosion"],
    },
    CasualtyProfile.COMBAT: {
        "priority_mix": {
            Priority.P1_IMMEDIATE: 0.35,
            Priority.P2_VERY_URGENT: 0.40,
            Priority.P3_URGENT: 0.20,
            Priority.P4_STANDARD: 0.05,
        },
        "requires_decon": False,
        "description": "Penetrating trauma - highest P1 rate, hemorrhage control critical",
        "typical_scenarios": ["Mass stabbing", "Shooting incident", "Military repatriation"],
    },
}


@dataclass
class MajorIncidentConfig:
    """Configuration for major incident overlay.

    The incident injects additional casualties on top of normal arrivals
    starting at trigger_time and lasting for duration minutes.

    Attributes:
        enabled: Whether incident simulation is active
        trigger_time: Minutes into simulation when incident starts
        duration: Duration of incident arrival window in minutes
        overload_percentage: Extra load as percentage of normal rate
                            (50 = 50% more casualties than normal in window)
        arrival_pattern: Temporal distribution of arrivals within window
        casualty_profile: Type of incident determining priority mix
        wave_count: Number of peaks for WAVES pattern (ignored for others)
        decon_time_range: Min/max decontamination time in minutes (CBRN only)

    Example:
        50% overload with 20/hr baseline rate over 2 hours:
        - Normal window arrivals: 20 * 2 = 40
        - Additional casualties: 40 * 0.50 = 20
        - Total incident casualties: 20
    """
    enabled: bool = False
    trigger_time: float = 120.0  # 2 hours into simulation
    duration: float = 120.0  # 2-hour incident window
    overload_percentage: float = 50.0  # 50% extra load
    arrival_pattern: IncidentArrivalPattern = IncidentArrivalPattern.BOLUS
    casualty_profile: CasualtyProfile = CasualtyProfile.GENERIC
    wave_count: int = 3  # For WAVES pattern
    decon_time_range: Tuple[float, float] = (15.0, 45.0)  # min, max minutes

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.trigger_time < 0:
            raise ValueError("trigger_time must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.overload_percentage <= 0:
            raise ValueError("overload_percentage must be positive")
        if self.wave_count < 1:
            raise ValueError("wave_count must be at least 1")
        if self.decon_time_range[0] < 0 or self.decon_time_range[1] < self.decon_time_range[0]:
            raise ValueError("decon_time_range must be (min, max) with min >= 0 and max >= min")

    def calculate_additional_casualties(self, baseline_hourly_rate: float) -> int:
        """Calculate number of extra casualties to inject.

        Args:
            baseline_hourly_rate: Normal arrival rate per hour

        Returns:
            Number of additional casualties (rounded to nearest integer)
        """
        window_hours = self.duration / 60.0
        window_arrivals = baseline_hourly_rate * window_hours
        additional = window_arrivals * (self.overload_percentage / 100.0)
        return max(1, round(additional))  # At least 1 casualty

    def generate_arrival_times(
        self,
        rng: np.random.Generator,
        baseline_hourly_rate: float,
    ) -> List[float]:
        """Pre-generate all incident casualty arrival times.

        Pre-calculation ensures reproducibility with the same seed.
        Times are absolute (simulation time, not relative to trigger).

        Args:
            rng: NumPy random generator for this incident
            baseline_hourly_rate: Normal arrival rate per hour

        Returns:
            Sorted list of arrival times in minutes from simulation start
        """
        n_casualties = self.calculate_additional_casualties(baseline_hourly_rate)

        if self.arrival_pattern == IncidentArrivalPattern.BOLUS:
            times = self._generate_bolus_arrivals(rng, n_casualties)
        elif self.arrival_pattern == IncidentArrivalPattern.WAVES:
            times = self._generate_wave_arrivals(rng, n_casualties)
        else:  # SUSTAINED
            times = self._generate_sustained_arrivals(rng, n_casualties)

        # Convert relative times to absolute and sort
        absolute_times = [self.trigger_time + t for t in times]
        return sorted(absolute_times)

    def _generate_bolus_arrivals(
        self,
        rng: np.random.Generator,
        n_casualties: int,
    ) -> List[float]:
        """Generate front-loaded bolus arrival pattern.

        Distribution: 60% in first 20%, 30% in next 40%, 10% in final 40%
        Uses beta distribution for smooth realistic curve.
        """
        # Beta(2, 5) gives strong front-loading
        relative_positions = rng.beta(2.0, 5.0, size=n_casualties)
        return [p * self.duration for p in relative_positions]

    def _generate_wave_arrivals(
        self,
        rng: np.random.Generator,
        n_casualties: int,
    ) -> List[float]:
        """Generate multi-peak wave arrival pattern.

        Creates wave_count peaks at regular intervals with Gaussian spread.
        """
        wave_interval = self.duration / self.wave_count
        wave_std = wave_interval * 0.2  # 20% of interval as spread

        times = []
        casualties_per_wave = n_casualties // self.wave_count
        remainder = n_casualties % self.wave_count

        for wave_idx in range(self.wave_count):
            wave_center = wave_interval * (wave_idx + 0.5)
            n_this_wave = casualties_per_wave + (1 if wave_idx < remainder else 0)

            # Generate times around wave center
            wave_times = rng.normal(wave_center, wave_std, size=n_this_wave)
            # Clip to valid range
            wave_times = np.clip(wave_times, 0, self.duration)
            times.extend(wave_times.tolist())

        return times

    def _generate_sustained_arrivals(
        self,
        rng: np.random.Generator,
        n_casualties: int,
    ) -> List[float]:
        """Generate uniform sustained arrival pattern.

        Evenly distributed across duration with small random jitter.
        """
        # Uniform distribution across duration
        return rng.uniform(0, self.duration, size=n_casualties).tolist()

    def sample_priority(self, rng: np.random.Generator) -> Priority:
        """Sample a priority level based on casualty profile.

        Args:
            rng: NumPy random generator

        Returns:
            Priority enum value based on profile's priority mix
        """
        profile_data = CASUALTY_PROFILES[self.casualty_profile]
        priority_mix = profile_data["priority_mix"]

        priorities = list(priority_mix.keys())
        probabilities = list(priority_mix.values())

        # Use index-based choice to ensure we return the actual enum
        idx = rng.choice(len(priorities), p=probabilities)
        return priorities[idx]

    def sample_decon_time(self, rng: np.random.Generator) -> float:
        """Sample decontamination time for CBRN casualties.

        Args:
            rng: NumPy random generator

        Returns:
            Decontamination duration in minutes (0 if not CBRN)
        """
        profile_data = CASUALTY_PROFILES[self.casualty_profile]
        if not profile_data["requires_decon"]:
            return 0.0

        min_time, max_time = self.decon_time_range
        # Use triangular distribution - mode at 25% point (faster decon more common)
        mode = min_time + (max_time - min_time) * 0.25
        return rng.triangular(min_time, mode, max_time)

    @property
    def requires_decon(self) -> bool:
        """Check if this incident profile requires decontamination."""
        return CASUALTY_PROFILES[self.casualty_profile]["requires_decon"]

    @property
    def profile_description(self) -> str:
        """Get human-readable description of casualty profile."""
        return CASUALTY_PROFILES[self.casualty_profile]["description"]

    def get_priority_breakdown(self) -> Dict[str, float]:
        """Get priority percentages for display.

        Returns:
            Dict mapping priority names to percentages (0-100)
        """
        profile_data = CASUALTY_PROFILES[self.casualty_profile]
        return {
            p.name: prob * 100
            for p, prob in profile_data["priority_mix"].items()
        }


def get_profile_info(profile: CasualtyProfile) -> Dict:
    """Get full information about a casualty profile.

    Args:
        profile: The casualty profile enum value

    Returns:
        Dict with priority_mix, requires_decon, description, typical_scenarios
    """
    return CASUALTY_PROFILES[profile].copy()


def list_profiles() -> List[Dict]:
    """Get summary information for all casualty profiles.

    Useful for UI dropdowns and documentation.

    Returns:
        List of dicts with name, description, requires_decon for each profile
    """
    return [
        {
            "name": profile.value,
            "enum": profile,
            "description": data["description"],
            "requires_decon": data["requires_decon"],
            "typical_scenarios": data["typical_scenarios"],
        }
        for profile, data in CASUALTY_PROFILES.items()
    ]
