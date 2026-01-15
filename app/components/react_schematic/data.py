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
    # Entry nodes use capacity=None so React schematic displays throughput_per_hour
    nodes = {
        # Entry nodes (capacity=None to show throughput instead of occupancy)
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=None,  # Entry nodes show throughput, not occupancy
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
            capacity=None,  # Entry nodes show throughput, not occupancy
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


def build_schematic_at_time(
    patients: List[Any],
    resource_logs: Dict[str, List[tuple]],
    timestamp: float,
    scenario: Any,
) -> SchematicData:
    """Build schematic showing state at a specific simulation time.

    Filters patients active at the timestamp and calculates occupancy
    from resource_logs for that time snapshot.

    Args:
        patients: List of Patient objects from simulation
        resource_logs: Dict {resource_name: [(time, count), ...]}
        timestamp: Simulation time to show snapshot for
        scenario: FullScenario with capacity configuration

    Returns:
        SchematicData for rendering the snapshot schematic
    """

    def get_occupancy_at_time(resource_name: str, time: float) -> int:
        """Get resource occupancy at a specific time from logs."""
        logs = resource_logs.get(resource_name, [(0.0, 0)])
        # Find the most recent log entry at or before the given time
        occupancy = 0
        for log_time, count in logs:
            if log_time <= time:
                occupancy = count
            else:
                break
        return occupancy

    # Get capacities from scenario
    n_ambulances = getattr(scenario, "n_ambulances", 10)
    n_helicopters = getattr(scenario, "n_helicopters", 2)
    n_handover = getattr(scenario, "n_handover_bays", 4)
    n_triage = getattr(scenario, "n_triage", 3)
    n_ed_bays = getattr(scenario, "n_ed_bays", 20)

    n_theatre = 2
    if hasattr(scenario, "theatre_config") and scenario.theatre_config:
        n_theatre = getattr(scenario.theatre_config, "n_tables", 2)

    n_itu = 6
    if hasattr(scenario, "itu_config") and scenario.itu_config:
        n_itu = getattr(scenario.itu_config, "capacity", 6)

    n_ward = 30
    if hasattr(scenario, "ward_config") and scenario.ward_config:
        n_ward = getattr(scenario.ward_config, "capacity", 30)

    # Get occupancy at this time from resource logs
    occupied_handover = get_occupancy_at_time("handover", timestamp)
    occupied_triage = get_occupancy_at_time("triage", timestamp)
    occupied_ed = get_occupancy_at_time("ed_bays", timestamp)
    occupied_theatre = get_occupancy_at_time("surgery", timestamp)
    occupied_itu = get_occupancy_at_time("itu", timestamp)
    occupied_ward = get_occupancy_at_time("ward", timestamp)

    # Calculate utilisation
    util_handover = occupied_handover / n_handover if n_handover > 0 else 0
    util_triage = occupied_triage / n_triage if n_triage > 0 else 0
    util_ed = occupied_ed / n_ed_bays if n_ed_bays > 0 else 0
    util_theatre = occupied_theatre / n_theatre if n_theatre > 0 else 0
    util_itu = occupied_itu / n_itu if n_itu > 0 else 0
    util_ward = occupied_ward / n_ward if n_ward > 0 else 0

    # Count patients at each location at this time
    # Patient is "in system" if arrival_time <= timestamp and (departure_time is None or departure_time > timestamp)
    patients_in_system = [
        p for p in patients
        if p.arrival_time <= timestamp
        and (p.departure_time is None or p.departure_time > timestamp)
    ]

    # Calculate throughput - count departures in the hour preceding this timestamp
    hour_window_start = max(0, timestamp - 60)  # 60 minutes = 1 hour
    recent_departures = [
        p for p in patients
        if p.departure_time is not None
        and hour_window_start <= p.departure_time <= timestamp
    ]
    throughput_per_hour = len(recent_departures)  # Already per hour (60-min window)

    # Calculate recent arrivals for entry nodes (same hour window)
    recent_arrivals = [
        p for p in patients
        if hour_window_start <= p.arrival_time <= timestamp
    ]

    # Count by arrival mode
    from faer.core.entities import ArrivalMode
    ambulance_arrivals = sum(1 for p in recent_arrivals if p.mode == ArrivalMode.AMBULANCE)
    walkin_arrivals = sum(1 for p in recent_arrivals if p.mode == ArrivalMode.SELF_PRESENTATION)
    hems_arrivals = sum(1 for p in recent_arrivals if p.mode == ArrivalMode.HELICOPTER)

    # Calculate mean waits for patients currently waiting or recently processed
    triage_waits = [
        p.triage_wait for p in patients_in_system
        if p.triage_start is not None and p.triage_start <= timestamp
    ]
    treatment_waits = [
        p.treatment_wait for p in patients_in_system
        if p.treatment_start is not None and p.treatment_start <= timestamp
    ]

    mean_triage_wait = float(np.mean(triage_waits)) if triage_waits else 0.0
    mean_treatment_wait = float(np.mean(treatment_waits)) if treatment_waits else 0.0

    # Build nodes
    # Entry nodes use capacity=None so React schematic displays throughput_per_hour
    nodes = {
        # Entry nodes (capacity=None to show throughput instead of occupancy)
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=None,  # Entry nodes show throughput, not occupancy
            occupied=0,
            throughput_per_hour=float(ambulance_arrivals),
            mean_wait_mins=0.0,
        ),
        "walkin": NodeState(
            id="walkin",
            label="Walk-in",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=float(walkin_arrivals),
            mean_wait_mins=0.0,
        ),
        "hems": NodeState(
            id="hems",
            label="HEMS",
            node_type="entry",
            capacity=None,  # Entry nodes show throughput, not occupancy
            occupied=0,
            throughput_per_hour=float(hems_arrivals),
            mean_wait_mins=0.0,
        ),
        # Process nodes
        "handover": NodeState(
            id="handover",
            label="Handover",
            node_type="process",
            capacity=n_handover,
            occupied=occupied_handover,
            throughput_per_hour=float(ambulance_arrivals + hems_arrivals),
            mean_wait_mins=0.0,
        ),
        "triage": NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=n_triage,
            occupied=occupied_triage,
            throughput_per_hour=float(len(recent_arrivals)),
            mean_wait_mins=mean_triage_wait,
        ),
        # Resource nodes
        "ed_bays": NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=n_ed_bays,
            occupied=occupied_ed,
            throughput_per_hour=float(throughput_per_hour),
            mean_wait_mins=mean_treatment_wait,
        ),
        "theatre": NodeState(
            id="theatre",
            label="Theatre",
            node_type="resource",
            capacity=n_theatre,
            occupied=occupied_theatre,
            throughput_per_hour=0.0,  # Would need surgery logs
            mean_wait_mins=0.0,
        ),
        "itu": NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=n_itu,
            occupied=occupied_itu,
            throughput_per_hour=0.0,
            mean_wait_mins=0.0,
        ),
        "ward": NodeState(
            id="ward",
            label="Ward",
            node_type="resource",
            capacity=n_ward,
            occupied=occupied_ward,
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
            throughput_per_hour=float(throughput_per_hour),
            mean_wait_mins=0.0,
        ),
    }

    # Determine blocked edges (utilisation > 95%)
    itu_blocked = util_itu > 0.95
    ward_blocked = util_ward > 0.95
    ed_blocked = util_ed > 0.95

    # Build edges with flow rates
    edges = [
        FlowEdge(source="ambulance", target="handover", volume_per_hour=float(ambulance_arrivals)),
        FlowEdge(source="hems", target="handover", volume_per_hour=float(hems_arrivals)),
        FlowEdge(source="walkin", target="triage", volume_per_hour=float(walkin_arrivals)),
        FlowEdge(source="handover", target="triage", volume_per_hour=float(ambulance_arrivals + hems_arrivals), is_blocked=ed_blocked),
        FlowEdge(source="triage", target="ed_bays", volume_per_hour=float(len(recent_arrivals)), is_blocked=ed_blocked),
        FlowEdge(source="ed_bays", target="theatre", volume_per_hour=0.0),
        FlowEdge(source="ed_bays", target="itu", volume_per_hour=0.0, is_blocked=itu_blocked),
        FlowEdge(source="ed_bays", target="ward", volume_per_hour=0.0, is_blocked=ward_blocked),
        FlowEdge(source="ed_bays", target="discharge", volume_per_hour=float(throughput_per_hour)),
        FlowEdge(source="theatre", target="itu", volume_per_hour=0.0, is_blocked=itu_blocked),
        FlowEdge(source="theatre", target="ward", volume_per_hour=0.0, is_blocked=ward_blocked),
        FlowEdge(source="itu", target="ward", volume_per_hour=0.0, is_blocked=ward_blocked),
        FlowEdge(source="itu", target="discharge", volume_per_hour=0.0),
        FlowEdge(source="ward", target="discharge", volume_per_hour=0.0),
    ]

    # Calculate total in system
    total_in_system = len(patients_in_system)

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

    # Format timestamp as hours:minutes
    hours = int(timestamp // 60)
    minutes = int(timestamp % 60)
    timestamp_label = f"T+{hours}h{minutes:02d}m"

    return SchematicData(
        timestamp=timestamp_label,
        nodes=nodes,
        edges=edges,
        total_in_system=total_in_system,
        total_throughput_24h=total_throughput_24h,
        overall_status=overall_status,
    )


def build_timeline_data(
    patients: List[Any],
    resource_logs: Dict[str, List[tuple]],
    run_length: float,
    warm_up: float = 0.0,
    interval_mins: float = 15.0,
) -> Dict[str, List]:
    """Build time-series data for timeline chart.

    Creates binned data for arrivals, departures, and patients-in-system
    at regular intervals throughout the simulation.

    Args:
        patients: List of Patient objects from simulation
        resource_logs: Dict {resource_name: [(time, count), ...]}
        run_length: Total simulation run length in minutes
        warm_up: Warm-up period in minutes (excluded from analysis)
        interval_mins: Time interval for binning (default 15 minutes)

    Returns:
        Dict with keys: 'time', 'arrivals', 'departures', 'in_system', 'ed_occupancy'
    """
    # Generate time points
    start_time = warm_up
    end_time = run_length
    time_points = []
    current = start_time
    while current <= end_time:
        time_points.append(current)
        current += interval_mins

    arrivals = []
    departures = []
    in_system = []
    ed_occupancy = []
    itu_occupancy = []
    theatre_occupancy = []

    # Get resource logs
    ed_logs = resource_logs.get("ed_bays", [(0.0, 0)])
    itu_logs = resource_logs.get("itu", [(0.0, 0)])
    theatre_logs = resource_logs.get("surgery", [(0.0, 0)])

    def get_occ_at_time(logs: list, t: float) -> int:
        """Get occupancy at a specific time from logs."""
        occ = 0
        for log_time, count in logs:
            if log_time <= t:
                occ = count
            else:
                break
        return occ

    for t in time_points:
        # Count arrivals in this interval
        interval_start = t - interval_mins if t > start_time else start_time
        arr_count = sum(
            1 for p in patients
            if interval_start < p.arrival_time <= t
        )
        arrivals.append(arr_count)

        # Count departures in this interval
        dep_count = sum(
            1 for p in patients
            if p.departure_time is not None and interval_start < p.departure_time <= t
        )
        departures.append(dep_count)

        # Count patients in system at this time
        in_sys = sum(
            1 for p in patients
            if p.arrival_time <= t and (p.departure_time is None or p.departure_time > t)
        )
        in_system.append(in_sys)

        # Get resource occupancies at this time
        ed_occupancy.append(get_occ_at_time(ed_logs, t))
        itu_occupancy.append(get_occ_at_time(itu_logs, t))
        theatre_occupancy.append(get_occ_at_time(theatre_logs, t))

    return {
        "time": time_points,
        "arrivals": arrivals,
        "departures": departures,
        "in_system": in_system,
        "ed_occupancy": ed_occupancy,
        "itu_occupancy": itu_occupancy,
        "theatre_occupancy": theatre_occupancy,
    }


# ============ UTILISATION SCHEMATIC ============


@dataclass
class UtilisationNode:
    """Resource utilisation with confidence interval."""

    id: str
    label: str
    category: str  # "emergency" | "triage_ed" | "diagnostics" | "downstream" | "exit"
    utilisation: float  # Mean utilisation (0-1)
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    # Exit node metrics (throughput instead of utilisation)
    throughput_per_hour: Optional[float] = None
    throughput_ci_lower: Optional[float] = None
    throughput_ci_upper: Optional[float] = None

    @property
    def status(self) -> str:
        """Determine status based on utilisation."""
        if self.utilisation >= 0.90:
            return "critical"
        elif self.utilisation >= 0.70:
            return "warning"
        return "normal"

    @property
    def utilisation_pct(self) -> str:
        """Format utilisation as percentage string."""
        return f"{self.utilisation * 100:.1f}%"

    @property
    def ci_str(self) -> str:
        """Format CI as string."""
        return f"[{self.ci_lower * 100:.1f}%, {self.ci_upper * 100:.1f}%]"

    @property
    def throughput_str(self) -> str:
        """Format throughput as string."""
        if self.throughput_per_hour is not None:
            return f"{self.throughput_per_hour:.1f}/hr"
        return ""

    @property
    def throughput_ci_str(self) -> str:
        """Format throughput CI as string."""
        if self.throughput_ci_lower is not None and self.throughput_ci_upper is not None:
            return f"[{self.throughput_ci_lower:.1f}, {self.throughput_ci_upper:.1f}]"
        return ""


@dataclass
class UtilisationSchematicData:
    """Complete utilisation schematic data for rendering."""

    title: str
    n_reps: int
    nodes: Dict[str, UtilisationNode]
    categories: List[str]  # Ordered list of category names


def build_utilisation_schematic(
    results: Dict[str, List[float]],
    compute_ci_func,
    run_length_hours: float = 8.0,
) -> UtilisationSchematicData:
    """Build utilisation schematic data from simulation results.

    Args:
        results: Dict of metric name to list of values from replications
        compute_ci_func: Function to compute confidence intervals (from faer.experiment.analysis)
        run_length_hours: Simulation run length in hours (for throughput calculation)

    Returns:
        UtilisationSchematicData for rendering
    """
    n_reps = len(results.get("arrivals", [1]))

    def get_util(key: str, default: float = 0.0) -> List[float]:
        """Get utilisation from results with fallback."""
        if key in results and results[key]:
            return results[key]
        return [default] * n_reps

    def make_node(node_id: str, label: str, category: str, util_key: str) -> UtilisationNode:
        """Create a UtilisationNode from results."""
        util_values = get_util(util_key)
        ci = compute_ci_func(util_values)
        return UtilisationNode(
            id=node_id,
            label=label,
            category=category,
            utilisation=ci["mean"],
            ci_lower=ci["ci_lower"],
            ci_upper=ci["ci_upper"],
        )

    # Calculate discharge throughput per hour from departures
    departures = results.get("departures", [0])
    throughput_per_hour_list = [d / run_length_hours if run_length_hours > 0 else 0 for d in departures]
    throughput_ci = compute_ci_func(throughput_per_hour_list)

    nodes = {
        # Emergency Services (left column)
        "ambulance_fleet": make_node(
            "ambulance_fleet", "Ambulance", "emergency", "util_ambulance_fleet"
        ),
        "hems_fleet": make_node(
            "hems_fleet", "HEMS", "emergency", "util_helicopter_fleet"
        ),
        "handover_bays": make_node(
            "handover_bays", "Handover", "emergency", "util_handover"
        ),
        # Main lane (Triage -> ED -> Theatre -> Discharge)
        "triage": make_node("triage", "Triage", "triage_ed", "util_triage"),
        "ed_bays": make_node("ed_bays", "ED Bays", "triage_ed", "util_ed_bays"),
        "theatre": make_node("theatre", "Theatre", "downstream", "util_theatre"),
        # Crucifix arms (ITU above, Ward below Theatre)
        "itu": make_node("itu", "ITU", "downstream", "util_itu"),
        "ward": make_node("ward", "Ward", "downstream", "util_ward"),
        # Exit node - shows throughput instead of utilisation
        "discharge": UtilisationNode(
            id="discharge",
            label="Discharge",
            category="exit",
            utilisation=0.0,  # Exit nodes don't have utilisation
            ci_lower=0.0,
            ci_upper=0.0,
            throughput_per_hour=throughput_ci["mean"],
            throughput_ci_lower=throughput_ci["ci_lower"],
            throughput_ci_upper=throughput_ci["ci_upper"],
        ),
    }

    categories = ["emergency", "triage_ed", "downstream", "exit"]

    return UtilisationSchematicData(
        title="Resource Utilisation",
        n_reps=n_reps,
        nodes=nodes,
        categories=categories,
    )


def render_utilisation_schematic_svg(data: UtilisationSchematicData) -> str:
    """Render utilisation schematic as inline SVG in crucifix layout.

    Matches the exact layout from the patient flow React schematic (1400x1000 viewbox):
    - Left column: Entry nodes (Ambulance, Walk-in/HEMS, Handover) stacked vertically
    - Main lane (horizontal at y=500): Triage → ED Bays → Theatre → Discharge
    - Top arm: ITU (above Theatre)
    - Bottom arm: Ward (below Theatre)

    Args:
        data: UtilisationSchematicData from build_utilisation_schematic

    Returns:
        SVG string for rendering with st.components.v1.html
    """

    def get_color(status: str) -> tuple:
        """Get fill and stroke colors for a status."""
        colors = {
            "normal": ("#f0fff4", "#28a745", "#166534"),  # fill, stroke, text
            "warning": ("#fffbf0", "#ffc107", "#92400e"),
            "critical": ("#fff5f5", "#dc3545", "#991b1b"),
        }
        return colors.get(status, ("#f5f5f5", "#9e9e9e", "#424242"))

    def render_node(node: UtilisationNode, x: int, y: int, width: int = 156, height: int = 108) -> str:
        """Render a single utilisation node box (matching React schematic dimensions)."""
        fill, stroke, text_color = get_color(node.status)
        bar_width = int((width - 24) * min(node.utilisation, 1.0))

        # Exit nodes (discharge) use different colors and show throughput instead of utilisation
        if node.category == "exit":
            return f"""
            <g transform="translate({x - width // 2}, {y - height // 2})">
                <!-- Drop shadow -->
                <rect x="3" y="3" width="{width}" height="{height}" rx="10" fill="black" fill-opacity="0.1"/>

                <!-- Box (exit node purple style) -->
                <rect width="{width}" height="{height}" rx="10" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>

                <!-- Label -->
                <text x="{width // 2}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#6a1b9a">{node.label}</text>

                <!-- Throughput metric -->
                <text x="{width // 2}" y="58" text-anchor="middle" font-size="18" font-weight="bold" fill="#4a148c">{node.throughput_str}</text>

                <!-- CI -->
                <text x="{width // 2}" y="82" text-anchor="middle" font-size="12" fill="#7b1fa2">{node.throughput_ci_str}</text>
            </g>
            """

        return f"""
        <g transform="translate({x - width // 2}, {y - height // 2})">
            <!-- Drop shadow -->
            <rect x="3" y="3" width="{width}" height="{height}" rx="10" fill="black" fill-opacity="0.1"/>

            <!-- Box -->
            <rect width="{width}" height="{height}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>

            <!-- Status dot -->
            <circle cx="{width - 18}" cy="18" r="7" fill="{stroke}"/>

            <!-- Label -->
            <text x="{width // 2}" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="{text_color}">{node.label}</text>

            <!-- Utilisation bar background -->
            <rect x="12" y="42" width="{width - 24}" height="12" rx="6" fill="#e9ecef"/>

            <!-- Utilisation bar fill -->
            <rect x="12" y="42" width="{bar_width}" height="12" rx="6" fill="{stroke}"/>

            <!-- Utilisation percentage -->
            <text x="{width // 2}" y="72" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{node.utilisation_pct}</text>

            <!-- CI -->
            <text x="{width // 2}" y="94" text-anchor="middle" font-size="12" fill="#666">{node.ci_str}</text>
        </g>
        """

    def render_arrow(x1: int, y1: int, x2: int, y2: int, curved: bool = False) -> str:
        """Render a flow arrow between nodes."""
        if curved:
            mid_x = (x1 + x2) // 2
            return f"""
            <path d="M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}"
                  fill="none" stroke="#adb5bd" stroke-width="2" marker-end="url(#arrowhead)"/>
            """
        return f"""
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#adb5bd" stroke-width="2" marker-end="url(#arrowhead)"/>
        """

    # Layout constants (matching React schematic EXACTLY)
    VIEWBOX_WIDTH = 1400
    VIEWBOX_HEIGHT = 1000
    SPACING = 270  # Horizontal spacing between nodes (matches React)
    MAIN_Y = 500   # Main lane Y position (matches React)

    # Node dimensions (matching React schematic: 156x108)
    NODE_W = 156
    NODE_H = 108

    # Node positions (crucifix layout - matching React Schematic.tsx exactly)
    # Entry column at x=120, main lane flows right
    positions = {
        # Entry column (left, stacked vertically) - x=120
        "ambulance_fleet": (120, MAIN_Y - 120),
        "hems_fleet": (120, MAIN_Y),
        "handover_bays": (120, MAIN_Y + 120),
        # Main lane (horizontal flow)
        "triage": (120 + SPACING, MAIN_Y),           # x=390
        "ed_bays": (120 + SPACING * 2, MAIN_Y),      # x=660
        "theatre": (120 + SPACING * 3, MAIN_Y),      # x=930
        # ITU (top arm of crucifix - above Theatre)
        "itu": (120 + SPACING * 3, MAIN_Y - SPACING),  # x=930, y=230
        # Ward (bottom arm of crucifix - below Theatre)
        "ward": (120 + SPACING * 3, MAIN_Y + SPACING),  # x=930, y=770
        # Discharge (exit - right end of main lane)
        "discharge": (120 + SPACING * 4, MAIN_Y),    # x=1200
    }

    svg_parts = [
        f"""
        <svg viewBox="0 0 {VIEWBOX_WIDTH} {VIEWBOX_HEIGHT}" width="100%" height="100%"
             preserveAspectRatio="xMidYMid meet"
             style="font-family: system-ui, -apple-system, sans-serif; background: #fafafa;">
            <!-- Arrow marker definition -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#adb5bd"/>
                </marker>
            </defs>

            <!-- Header -->
            <g transform="translate(24, 22)">
                <text font-size="14" fill="#666">
                    <tspan font-weight="bold" fill="#333">{data.title}</tspan>
                    <tspan dx="18">|</tspan>
                    <tspan dx="18">{data.n_reps} replications</tspan>
                    <tspan dx="18">|</tspan>
                    <tspan dx="18">95% Confidence Intervals</tspan>
                </text>
            </g>
        """
    ]

    # === RENDER FLOW ARROWS (behind nodes) ===
    # Entry -> Triage (curved arrows from entry column to triage)
    for entry_id in ["ambulance_fleet", "hems_fleet", "handover_bays"]:
        if entry_id in positions:
            ex, ey = positions[entry_id]
            tx, ty = positions["triage"]
            svg_parts.append(render_arrow(ex + NODE_W // 2, ey, tx - NODE_W // 2, ty, curved=True))

    # Triage -> ED (horizontal on main lane)
    tx, ty = positions["triage"]
    edx, edy = positions["ed_bays"]
    svg_parts.append(render_arrow(tx + NODE_W // 2, ty, edx - NODE_W // 2, edy))

    # ED -> Theatre (horizontal on main lane)
    thx, thy = positions["theatre"]
    svg_parts.append(render_arrow(edx + NODE_W // 2, edy, thx - NODE_W // 2, thy))

    # Theatre -> ITU (vertical up)
    itux, ituy = positions["itu"]
    svg_parts.append(render_arrow(thx, thy - NODE_H // 2, itux, ituy + NODE_H // 2))

    # Theatre -> Ward (vertical down)
    wardx, wardy = positions["ward"]
    svg_parts.append(render_arrow(thx, thy + NODE_H // 2, wardx, wardy - NODE_H // 2))

    # Discharge position
    dischx, dischy = positions["discharge"]

    # Note: No Theatre -> Discharge arrow (Theatre patients go to ITU/Ward first)

    # ITU -> Discharge (curved down from ITU to discharge)
    svg_parts.append(render_arrow(itux + NODE_W // 2, ituy, dischx - NODE_W // 2, dischy, curved=True))

    # Ward -> Discharge (curved up from Ward to discharge)
    svg_parts.append(render_arrow(wardx + NODE_W // 2, wardy, dischx - NODE_W // 2, dischy, curved=True))

    # === RENDER NODES ===
    for node_id, (x, y) in positions.items():
        if node_id in data.nodes:
            svg_parts.append(render_node(data.nodes[node_id], x, y))

    # === LEGEND ===
    svg_parts.append(f"""
        <g transform="translate({VIEWBOX_WIDTH - 145}, 60)">
            <rect width="126" height="140" fill="white" stroke="#dee2e6" rx="8" fill-opacity="0.95"/>
            <text x="12" y="22" font-size="12" font-weight="bold" fill="#333">Legend</text>
            <circle cx="22" cy="44" r="5" fill="#28a745"/>
            <text x="34" y="48" font-size="11" fill="#333">Normal (&lt;70%)</text>
            <circle cx="22" cy="66" r="5" fill="#ffc107"/>
            <text x="34" y="70" font-size="11" fill="#333">Warning (70-90%)</text>
            <circle cx="22" cy="88" r="5" fill="#dc3545"/>
            <text x="34" y="92" font-size="11" fill="#333">Critical (&gt;90%)</text>
            <line x1="14" y1="112" x2="38" y2="112" stroke="#adb5bd" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="46" y="116" font-size="11" fill="#333">Flow</text>
        </g>
    """)

    svg_parts.append("</svg>")

    return "".join(svg_parts)
