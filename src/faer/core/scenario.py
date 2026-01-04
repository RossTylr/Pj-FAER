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


@dataclass
class FullScenario:
    """Extended scenario for full A&E pathway simulation.

    Includes multiple acuity streams, triage, and disposition routing.

    Attributes:
        run_length: Simulation duration in minutes.
        warm_up: Warm-up period in minutes.
        arrival_rate: Total patient arrivals per hour.
        random_seed: Master seed for reproducibility.

        Acuity mix (must sum to 1.0):
        p_resus: Proportion of Resus patients.
        p_majors: Proportion of Majors patients.
        p_minors: Proportion of Minors patients.

        Resources:
        n_triage: Number of triage nurses.
        n_resus_bays: Number of Resus bays.
        n_majors_bays: Number of Majors cubicles.
        n_minors_bays: Number of Minors cubicles.

        Service times (mean, cv) in minutes:
        triage_mean, triage_cv: Triage duration.
        resus_mean, resus_cv: Resus treatment duration.
        majors_mean, majors_cv: Majors treatment duration.
        minors_mean, minors_cv: Minors treatment duration.
        boarding_mean, boarding_cv: Boarding time for admits.

        Disposition probabilities:
        resus_p_admit: Probability Resus patient is admitted.
        majors_p_admit: Probability Majors patient is admitted.
        minors_p_admit: Probability Minors patient is admitted.
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

    # Resources
    n_triage: int = 2
    n_resus_bays: int = 2
    n_majors_bays: int = 10
    n_minors_bays: int = 6

    # Service times - Triage
    triage_mean: float = 5.0
    triage_cv: float = 0.3

    # Service times - Resus
    resus_mean: float = 90.0
    resus_cv: float = 0.6

    # Service times - Majors
    majors_mean: float = 120.0
    majors_cv: float = 0.7

    # Service times - Minors
    minors_mean: float = 45.0
    minors_cv: float = 0.5

    # Boarding time (for admitted patients awaiting bed)
    boarding_mean: float = 120.0
    boarding_cv: float = 1.0

    # Disposition probabilities
    resus_p_admit: float = 0.85
    majors_p_admit: float = 0.35
    minors_p_admit: float = 0.05

    # Reproducibility
    random_seed: int = 42

    # RNG streams (created in __post_init__)
    rng_arrivals: Optional[np.random.Generator] = None
    rng_acuity: Optional[np.random.Generator] = None
    rng_triage: Optional[np.random.Generator] = None
    rng_treatment: Optional[np.random.Generator] = None
    rng_boarding: Optional[np.random.Generator] = None
    rng_disposition: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        """Initialize separate RNG streams and validate parameters."""
        # Validate acuity mix
        total = self.p_resus + self.p_majors + self.p_minors
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Acuity proportions must sum to 1.0, got {total}")

        # Create separate RNG streams
        self.rng_arrivals = np.random.default_rng(self.random_seed)
        self.rng_acuity = np.random.default_rng(self.random_seed + 1)
        self.rng_triage = np.random.default_rng(self.random_seed + 2)
        self.rng_treatment = np.random.default_rng(self.random_seed + 3)
        self.rng_boarding = np.random.default_rng(self.random_seed + 4)
        self.rng_disposition = np.random.default_rng(self.random_seed + 5)

    @property
    def mean_iat(self) -> float:
        """Mean inter-arrival time in minutes."""
        return 60.0 / self.arrival_rate

    def get_admission_prob(self, acuity: str) -> float:
        """Get admission probability for an acuity level.

        Args:
            acuity: One of 'resus', 'majors', 'minors'.

        Returns:
            Probability of admission.
        """
        probs = {
            "resus": self.resus_p_admit,
            "majors": self.majors_p_admit,
            "minors": self.minors_p_admit,
        }
        return probs.get(acuity.lower(), 0.0)

    def get_treatment_params(self, acuity: str) -> tuple:
        """Get treatment time parameters for an acuity level.

        Args:
            acuity: One of 'resus', 'majors', 'minors'.

        Returns:
            Tuple of (mean, cv) for treatment time.
        """
        params = {
            "resus": (self.resus_mean, self.resus_cv),
            "majors": (self.majors_mean, self.majors_cv),
            "minors": (self.minors_mean, self.minors_cv),
        }
        return params.get(acuity.lower(), (60.0, 0.5))

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
            n_resus_bays=self.n_resus_bays,
            n_majors_bays=self.n_majors_bays,
            n_minors_bays=self.n_minors_bays,
            triage_mean=self.triage_mean,
            triage_cv=self.triage_cv,
            resus_mean=self.resus_mean,
            resus_cv=self.resus_cv,
            majors_mean=self.majors_mean,
            majors_cv=self.majors_cv,
            minors_mean=self.minors_mean,
            minors_cv=self.minors_cv,
            boarding_mean=self.boarding_mean,
            boarding_cv=self.boarding_cv,
            resus_p_admit=self.resus_p_admit,
            majors_p_admit=self.majors_p_admit,
            minors_p_admit=self.minors_p_admit,
            random_seed=new_seed,
        )
