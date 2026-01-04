"""Scenario configuration dataclass."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
