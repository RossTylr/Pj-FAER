"""NSPP thinning arrival generator."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class ArrivalProfile:
    """Piecewise constant arrival rate schedule.

    Represents a time-varying arrival rate over a day (or other period).
    Each entry in the schedule is (end_time, rate) where rate applies
    from the previous end_time (or 0) up to end_time.

    Attributes:
        schedule: List of (end_time, rate) tuples in minutes.
        max_rate: Maximum rate across all periods (computed automatically).
    """

    schedule: List[Tuple[float, float]]
    max_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate schedule and compute max rate."""
        self._validate()
        self.max_rate = max(rate for _, rate in self.schedule) if self.schedule else 0.0

    def _validate(self) -> None:
        """Ensure schedule is sorted and non-overlapping."""
        if not self.schedule:
            return
        times = [t for t, _ in self.schedule]
        if times != sorted(times):
            raise ValueError("Schedule must be sorted by time")
        if any(t < 0 for t in times):
            raise ValueError("Schedule times must be non-negative")
        if any(r < 0 for _, r in self.schedule):
            raise ValueError("Arrival rates must be non-negative")

    def get_rate(self, t: float) -> float:
        """Get arrival rate at time t.

        Args:
            t: Time in minutes.

        Returns:
            Arrival rate (patients per hour) at time t.
        """
        # Handle cyclic time (wrap around 24 hours)
        if self.schedule:
            period_length = self.schedule[-1][0]
            if period_length > 0:
                t = t % period_length

        for end_time, rate in self.schedule:
            if t < end_time:
                return rate

        # Beyond schedule: use last rate
        return self.schedule[-1][1] if self.schedule else 0.0


class NSPPThinning:
    """Non-stationary Poisson Process via thinning algorithm.

    Uses the thinning (acceptance-rejection) method to generate
    inter-arrival times from a time-varying arrival rate.

    Attributes:
        profile: ArrivalProfile with time-varying rates.
        rng: NumPy random generator for reproducibility.
        max_rate: Maximum arrival rate (for thinning efficiency).
    """

    def __init__(self, profile: ArrivalProfile, rng: np.random.Generator) -> None:
        """Initialize thinning sampler.

        Args:
            profile: Arrival rate profile.
            rng: NumPy random generator.
        """
        self.profile = profile
        self.rng = rng
        self.max_rate = profile.max_rate

    def sample_iat(self, current_time: float) -> float:
        """Sample inter-arrival time from current_time.

        Uses thinning algorithm: generate candidate from homogeneous
        Poisson at max rate, then accept with probability proportional
        to actual rate at candidate time.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            Inter-arrival time in minutes.
        """
        if self.max_rate <= 0:
            return float("inf")

        t = current_time

        while True:
            # Candidate IAT from homogeneous Poisson at max rate
            # Rate is per hour, so convert to minutes
            mean_iat = 60.0 / self.max_rate
            candidate_iat = self.rng.exponential(mean_iat)
            t += candidate_iat

            # Accept with probability λ(t) / λ_max
            current_rate = self.profile.get_rate(t)
            acceptance_prob = current_rate / self.max_rate

            if self.rng.random() <= acceptance_prob:
                return t - current_time

            # Rejected: continue from t (already advanced)


def load_default_profile() -> ArrivalProfile:
    """Load default 24-hour A&E arrival profile.

    Returns a typical ED arrival pattern with:
    - Low overnight (1-2/hr)
    - Peak mid-morning (7-7.5/hr)
    - Evening peak (6.5-7/hr)

    Returns:
        ArrivalProfile with 24 hourly periods (in minutes).
    """
    # Each tuple is (end_time_minutes, rate_per_hour)
    return ArrivalProfile([
        (60, 2.0),    # 00:00-01:00
        (120, 1.5),   # 01:00-02:00
        (180, 1.0),   # 02:00-03:00
        (240, 1.0),   # 03:00-04:00
        (300, 1.5),   # 04:00-05:00
        (360, 2.0),   # 05:00-06:00
        (420, 3.0),   # 06:00-07:00
        (480, 4.5),   # 07:00-08:00
        (540, 6.0),   # 08:00-09:00
        (600, 7.0),   # 09:00-10:00
        (660, 7.5),   # 10:00-11:00
        (720, 7.0),   # 11:00-12:00
        (780, 6.5),   # 12:00-13:00
        (840, 6.0),   # 13:00-14:00
        (900, 5.5),   # 14:00-15:00
        (960, 5.5),   # 15:00-16:00
        (1020, 6.0),  # 16:00-17:00
        (1080, 6.5),  # 17:00-18:00
        (1140, 7.0),  # 18:00-19:00
        (1200, 6.5),  # 19:00-20:00
        (1260, 5.5),  # 20:00-21:00
        (1320, 4.5),  # 21:00-22:00
        (1380, 3.5),  # 22:00-23:00
        (1440, 2.5),  # 23:00-24:00
    ])


def create_constant_profile(rate: float, duration: float = 1440.0) -> ArrivalProfile:
    """Create a constant arrival rate profile.

    Useful for testing and comparison with stationary arrivals.

    Args:
        rate: Constant arrival rate (patients per hour).
        duration: Profile duration in minutes (default 24 hours).

    Returns:
        ArrivalProfile with constant rate.
    """
    return ArrivalProfile([(duration, rate)])
