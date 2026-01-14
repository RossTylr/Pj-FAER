"""Data structures and transformation for schematic visualization.

Transforms simulation results and scenario configuration into the
SchematicData format expected by the React component.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class NodeState:
    """State of a single resource node."""

    id: str
    label: str
    node_type: str  # "entry" | "resource" | "process" | "exit"

    # Capacity (None for entry/exit nodes)
    capacity: Optional[int]
    occupied: int

    # Performance metrics
    throughput_per_hour: float
    mean_wait_mins: float

    @property
    def utilisation(self) -> float:
        """Calculate utilisation ratio."""
        if self.capacity is None or self.capacity == 0:
            return 0.0
        return min(1.0, self.occupied / self.capacity)

    @property
    def status(self) -> str:
        """Determine status based on utilisation."""
        if self.utilisation >= 0.90:
            return "critical"
        elif self.utilisation >= 0.70:
            return "warning"
        return "normal"


@dataclass
class FlowEdge:
    """Flow connection between two nodes."""

    source: str  # Source node ID
    target: str  # Target node ID
    volume_per_hour: float
    is_blocked: bool = False


@dataclass
class SchematicData:
    """Complete schematic state for rendering."""

    timestamp: str  # Display label (e.g., "Hour 14.5" or "Configuration")
    nodes: Dict[str, NodeState]
    edges: List[FlowEdge]

    # Summary metrics
    total_in_system: int
    total_throughput_24h: int
    overall_status: str  # "normal" | "warning" | "critical"


def to_dict(data: SchematicData) -> dict:
    """Convert SchematicData to JSON-serializable dict for React component."""
    return {
        "timestamp": data.timestamp,
        "nodes": {
            k: {
                "id": v.id,
                "label": v.label,
                "node_type": v.node_type,
                "capacity": v.capacity,
                "occupied": v.occupied,
                "throughput_per_hour": v.throughput_per_hour,
                "mean_wait_mins": v.mean_wait_mins,
                "utilisation": v.utilisation,
                "status": v.status,
            }
            for k, v in data.nodes.items()
        },
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "volume_per_hour": e.volume_per_hour,
                "is_blocked": e.is_blocked,
            }
            for e in data.edges
        ],
        "total_in_system": data.total_in_system,
        "total_throughput_24h": data.total_throughput_24h,
        "overall_status": data.overall_status,
    }


def build_schematic_from_config(session_state: Dict[str, Any]) -> SchematicData:
    """Build schematic data from session state configuration only (no results).

    Shows configured capacities with 0 occupied (no simulation run yet).

    Args:
        session_state: Streamlit session state dict with resource configurations

    Returns:
        SchematicData for rendering the configuration schematic
    """
    # Get capacities from session state with defaults
    n_ambulances = session_state.get("n_ambulances", 10)
    n_helicopters = session_state.get("n_helicopters", 2)
    n_handover = session_state.get("n_handover_bays", 4)
    n_triage = session_state.get("n_triage", 3)
    n_ed_bays = session_state.get("n_ed_bays", 20)
    n_theatre = session_state.get("n_theatre_tables", 2)
    n_itu = session_state.get("n_itu_beds", 6)
    n_ward = session_state.get("n_ward_beds", 30)

    # Build nodes with 0 occupied (configuration mode)
    nodes = {
        # Entry nodes
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=n_ambulances,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "walkin": NodeState(
            id="walkin",
            label="Walk-in",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "hems": NodeState(
            id="hems",
            label="HEMS",
            node_type="entry",
            capacity=n_helicopters,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        # Process nodes
        "handover": NodeState(
            id="handover",
            label="Handover",
            node_type="process",
            capacity=n_handover,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "triage": NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=n_triage,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        # Resource nodes
        "ed_bays": NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=n_ed_bays,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "theatre": NodeState(
            id="theatre",
            label="Theatre",
            node_type="resource",
            capacity=n_theatre,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "itu": NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=n_itu,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "ward": NodeState(
            id="ward",
            label="Ward",
            node_type="resource",
            capacity=n_ward,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        # Exit node
        "discharge": NodeState(
            id="discharge",
            label="Discharge",
            node_type="exit",
            capacity=None,
            occupied=0,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
    }

    # Build edges (no flow rates in config mode)
    edges = [
        # Arrivals -> Handover/Triage
        FlowEdge(source="ambulance", target="handover", volume_per_hour=0.0),
        FlowEdge(source="hems", target="handover", volume_per_hour=0.0),
        FlowEdge(source="walkin", target="triage", volume_per_hour=0.0),
        # Handover -> Triage
        FlowEdge(source="handover", target="triage", volume_per_hour=0.0),
        # Triage -> ED
        FlowEdge(source="triage", target="ed_bays", volume_per_hour=0.0),
        # ED -> Downstream
        FlowEdge(source="ed_bays", target="theatre", volume_per_hour=0.0),
        FlowEdge(source="ed_bays", target="itu", volume_per_hour=0.0),
        FlowEdge(source="ed_bays", target="ward", volume_per_hour=0.0),
        FlowEdge(source="ed_bays", target="discharge", volume_per_hour=0.0),
        # Theatre -> Post-op
        FlowEdge(source="theatre", target="itu", volume_per_hour=0.0),
        FlowEdge(source="theatre", target="ward", volume_per_hour=0.0),
        # ITU -> Step-down/Discharge
        FlowEdge(source="itu", target="ward", volume_per_hour=0.0),
        FlowEdge(source="itu", target="discharge", volume_per_hour=0.0),
        # Ward -> Discharge
        FlowEdge(source="ward", target="discharge", volume_per_hour=0.0),
    ]

    return SchematicData(
        timestamp="Configuration",
        nodes=nodes,
        edges=edges,
        total_in_system=0,
        total_throughput_24h=0,
        overall_status="normal",
    )


def build_schematic_from_results(
    results: Dict[str, List[float]],
    scenario: Any,
    run_length_hours: float,
) -> SchematicData:
    """Build schematic data from simulation results.

    Uses mean values from replications to compute node states and flow rates.

    Args:
        results: Dict of metric name to list of values from replications
        scenario: FullScenario with capacity configuration
        run_length_hours: Simulation run length in hours

    Returns:
        SchematicData for rendering the results schematic
    """
    # Helper to safely get mean
    def safe_mean(key: str, default: float = 0.0) -> float:
        values = results.get(key, [default])
        if not values:
            return default
        return float(np.mean(values))

    # Get utilisation means
    util_handover = safe_mean("util_handover")
    util_triage = safe_mean("util_triage")
    util_ed = safe_mean("util_ed_bays")
    util_theatre = safe_mean("util_theatre")
    util_itu = safe_mean("util_itu")
    util_ward = safe_mean("util_ward")

    # Get capacities from scenario
    n_ambulances = getattr(scenario, "n_ambulances", 10)
    n_helicopters = getattr(scenario, "n_helicopters", 2)
    n_handover = getattr(scenario, "n_handover_bays", 4)
    n_triage = getattr(scenario, "n_triage", 3)
    n_ed_bays = getattr(scenario, "n_ed_bays", 20)

    # Theatre config
    n_theatre = 2
    if hasattr(scenario, "theatre_config") and scenario.theatre_config:
        n_theatre = getattr(scenario.theatre_config, "n_tables", 2)

    # ITU config
    n_itu = 6
    if hasattr(scenario, "itu_config") and scenario.itu_config:
        n_itu = getattr(scenario.itu_config, "capacity", 6)

    # Ward config
    n_ward = 30
    if hasattr(scenario, "ward_config") and scenario.ward_config:
        n_ward = getattr(scenario.ward_config, "capacity", 30)

    # Calculate occupied counts from utilisation
    occupied_handover = int(util_handover * n_handover)
    occupied_triage = int(util_triage * n_triage)
    occupied_ed = int(util_ed * n_ed_bays)
    occupied_theatre = int(util_theatre * n_theatre)
    occupied_itu = int(util_itu * n_itu)
    occupied_ward = int(util_ward * n_ward)

    # Get throughput metrics
    mean_arrivals = safe_mean("arrivals")
    mean_departures = safe_mean("departures")
    throughput_per_hour = mean_departures / run_length_hours if run_length_hours > 0 else 0.0

    # Get wait times
    mean_treatment_wait = safe_mean("mean_treatment_wait")
    mean_triage_wait = safe_mean("mean_triage_wait", 5.0)

    # Get counts by disposition for flow rates
    discharge_count = safe_mean("discharge_count")
    admit_ward_count = safe_mean("admit_ward_count")
    admit_icu_count = safe_mean("admit_icu_count")

    # Calculate flow rates (patients per hour)
    flow_to_discharge = discharge_count / run_length_hours if run_length_hours > 0 else 0.0
    flow_to_ward = admit_ward_count / run_length_hours if run_length_hours > 0 else 0.0
    flow_to_itu = admit_icu_count / run_length_hours if run_length_hours > 0 else 0.0

    # Estimate theatre flow (roughly 10% of admissions need surgery)
    surgery_rate = (flow_to_ward + flow_to_itu) * 0.1

    # Build nodes
    nodes = {
        # Entry nodes
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=n_ambulances,
            occupied=0,
            throughput_per_hour=throughput_per_hour * 0.5,  # ~50% arrive by ambulance
            mean_wait_mins=0.0,
        ),
        "walkin": NodeState(
            id="walkin",
            label="Walk-in",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=throughput_per_hour * 0.45,  # ~45% walk-in
            mean_wait_mins=0.0,
        ),
        "hems": NodeState(
            id="hems",
            label="HEMS",
            node_type="entry",
            capacity=n_helicopters,
            occupied=0,
            throughput_per_hour=throughput_per_hour * 0.05,  # ~5% HEMS
            mean_wait_mins=0.0,
        ),
        # Process nodes
        "handover": NodeState(
            id="handover",
            label="Handover",
            node_type="process",
            capacity=n_handover,
            occupied=occupied_handover,
            throughput_per_hour=throughput_per_hour * 0.55,
            mean_wait_mins=safe_mean("mean_handover_wait", 10.0),
        ),
        "triage": NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=n_triage,
            occupied=occupied_triage,
            throughput_per_hour=throughput_per_hour,
            mean_wait_mins=mean_triage_wait,
        ),
        # Resource nodes
        "ed_bays": NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=n_ed_bays,
            occupied=occupied_ed,
            throughput_per_hour=throughput_per_hour,
            mean_wait_mins=mean_treatment_wait,
        ),
        "theatre": NodeState(
            id="theatre",
            label="Theatre",
            node_type="resource",
            capacity=n_theatre,
            occupied=occupied_theatre,
            throughput_per_hour=surgery_rate,
            mean_wait_mins=safe_mean("mean_theatre_wait", 45.0),
        ),
        "itu": NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=n_itu,
            occupied=occupied_itu,
            throughput_per_hour=flow_to_itu,
            mean_wait_mins=safe_mean("mean_itu_wait", 60.0),
        ),
        "ward": NodeState(
            id="ward",
            label="Ward",
            node_type="resource",
            capacity=n_ward,
            occupied=occupied_ward,
            throughput_per_hour=flow_to_ward,
            mean_wait_mins=safe_mean("mean_ward_wait", 30.0),
        ),
        # Exit node
        "discharge": NodeState(
            id="discharge",
            label="Discharge",
            node_type="exit",
            capacity=None,
            occupied=0,
            throughput_per_hour=flow_to_discharge,
            mean_wait_mins=0.0,
        ),
    }

    # Determine blocked edges (utilisation > 95%)
    itu_blocked = util_itu > 0.95
    ward_blocked = util_ward > 0.95
    ed_blocked = util_ed > 0.95

    # Build edges with flow rates
    edges = [
        # Arrivals -> Handover/Triage
        FlowEdge(
            source="ambulance",
            target="handover",
            volume_per_hour=throughput_per_hour * 0.5,
        ),
        FlowEdge(
            source="hems",
            target="handover",
            volume_per_hour=throughput_per_hour * 0.05,
        ),
        FlowEdge(
            source="walkin",
            target="triage",
            volume_per_hour=throughput_per_hour * 0.45,
        ),
        # Handover -> Triage
        FlowEdge(
            source="handover",
            target="triage",
            volume_per_hour=throughput_per_hour * 0.55,
            is_blocked=ed_blocked,  # Handover blocked when ED full
        ),
        # Triage -> ED
        FlowEdge(
            source="triage",
            target="ed_bays",
            volume_per_hour=throughput_per_hour,
            is_blocked=ed_blocked,
        ),
        # ED -> Downstream
        FlowEdge(
            source="ed_bays",
            target="theatre",
            volume_per_hour=surgery_rate,
        ),
        FlowEdge(
            source="ed_bays",
            target="itu",
            volume_per_hour=flow_to_itu * 0.5,
            is_blocked=itu_blocked,
        ),
        FlowEdge(
            source="ed_bays",
            target="ward",
            volume_per_hour=flow_to_ward * 0.7,
            is_blocked=ward_blocked,
        ),
        FlowEdge(
            source="ed_bays",
            target="discharge",
            volume_per_hour=flow_to_discharge,
        ),
        # Theatre -> Post-op
        FlowEdge(
            source="theatre",
            target="itu",
            volume_per_hour=surgery_rate * 0.3,
            is_blocked=itu_blocked,
        ),
        FlowEdge(
            source="theatre",
            target="ward",
            volume_per_hour=surgery_rate * 0.7,
            is_blocked=ward_blocked,
        ),
        # ITU -> Step-down/Discharge
        FlowEdge(
            source="itu",
            target="ward",
            volume_per_hour=flow_to_itu * 0.85,
            is_blocked=ward_blocked,
        ),
        FlowEdge(
            source="itu",
            target="discharge",
            volume_per_hour=flow_to_itu * 0.05,
        ),
        # Ward -> Discharge
        FlowEdge(
            source="ward",
            target="discharge",
            volume_per_hour=flow_to_ward,
        ),
    ]

    # Calculate total in system
    total_in_system = (
        occupied_handover
        + occupied_triage
        + occupied_ed
        + occupied_theatre
        + occupied_itu
        + occupied_ward
    )

    # 24h throughput estimate
    total_throughput_24h = int(throughput_per_hour * 24)

    # Determine overall status
    max_util = max(util_ed, util_itu, util_ward, util_theatre)
    if max_util >= 0.90:
        overall_status = "critical"
    elif max_util >= 0.70:
        overall_status = "warning"
    else:
        overall_status = "normal"

    return SchematicData(
        timestamp=f"Post-{run_length_hours:.0f}h Simulation",
        nodes=nodes,
        edges=edges,
        total_in_system=total_in_system,
        total_throughput_24h=total_throughput_24h,
        overall_status=overall_status,
    )
