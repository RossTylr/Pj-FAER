"""Scenario configuration dataclass."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from faer.core.entities import NodeType, Priority, ArrivalMode


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
    rng_arrivals: Optional[dict] = None  # Dict[ArrivalMode, Generator]
    rng_acuity: Optional[np.random.Generator] = None
    rng_triage: Optional[np.random.Generator] = None
    rng_treatment: Optional[np.random.Generator] = None
    rng_boarding: Optional[np.random.Generator] = None
    rng_disposition: Optional[np.random.Generator] = None
    rng_routing: Optional[np.random.Generator] = None

    # Routing configuration (created in __post_init__ if not provided)
    routing: List[RoutingRule] = field(default_factory=list)

    # Arrival stream configurations (created in __post_init__ if not provided)
    arrival_configs: List[ArrivalConfig] = field(default_factory=list)

    # Node configurations (created in __post_init__ if not provided)
    node_configs: dict = field(default_factory=dict)  # Dict[NodeType, NodeConfig]

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

    def _default_node_configs(self) -> dict:
        """Default node configurations."""
        return {
            NodeType.RESUS: NodeConfig(NodeType.RESUS, capacity=3, service_time_mean=60),
            NodeType.MAJORS: NodeConfig(NodeType.MAJORS, capacity=12, service_time_mean=45),
            NodeType.MINORS: NodeConfig(NodeType.MINORS, capacity=8, service_time_mean=30),
            NodeType.SURGERY: NodeConfig(NodeType.SURGERY, capacity=2, service_time_mean=150),
            NodeType.ITU: NodeConfig(NodeType.ITU, capacity=6, service_time_mean=720),
            NodeType.WARD: NodeConfig(NodeType.WARD, capacity=30, service_time_mean=480),
        }

    def _default_routing(self) -> List[RoutingRule]:
        """Default ED disposition routing."""
        rules = []

        # P1 from Resus: Surgery 30%, ITU 40%, Ward 20%, Exit 10%
        rules.extend([
            RoutingRule(NodeType.RESUS, NodeType.SURGERY, Priority.P1_IMMEDIATE, 0.30),
            RoutingRule(NodeType.RESUS, NodeType.ITU, Priority.P1_IMMEDIATE, 0.40),
            RoutingRule(NodeType.RESUS, NodeType.WARD, Priority.P1_IMMEDIATE, 0.20),
            RoutingRule(NodeType.RESUS, NodeType.EXIT, Priority.P1_IMMEDIATE, 0.10),
        ])

        # P2 from Resus/Majors: Surgery 15%, ITU 10%, Ward 45%, Exit 30%
        for src in [NodeType.RESUS, NodeType.MAJORS]:
            rules.extend([
                RoutingRule(src, NodeType.SURGERY, Priority.P2_VERY_URGENT, 0.15),
                RoutingRule(src, NodeType.ITU, Priority.P2_VERY_URGENT, 0.10),
                RoutingRule(src, NodeType.WARD, Priority.P2_VERY_URGENT, 0.45),
                RoutingRule(src, NodeType.EXIT, Priority.P2_VERY_URGENT, 0.30),
            ])

        # P3 from Majors: Ward 25%, Exit 75%
        rules.extend([
            RoutingRule(NodeType.MAJORS, NodeType.WARD, Priority.P3_URGENT, 0.25),
            RoutingRule(NodeType.MAJORS, NodeType.EXIT, Priority.P3_URGENT, 0.75),
        ])

        # P3 from Minors: Ward 10%, Exit 90%
        rules.extend([
            RoutingRule(NodeType.MINORS, NodeType.WARD, Priority.P3_URGENT, 0.10),
            RoutingRule(NodeType.MINORS, NodeType.EXIT, Priority.P3_URGENT, 0.90),
        ])

        # P4 from Minors: Ward 5%, Exit 95%
        rules.extend([
            RoutingRule(NodeType.MINORS, NodeType.WARD, Priority.P4_STANDARD, 0.05),
            RoutingRule(NodeType.MINORS, NodeType.EXIT, Priority.P4_STANDARD, 0.95),
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
