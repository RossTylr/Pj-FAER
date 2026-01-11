"""
Shared data structures for schematic demos.

Both Streamlit and React demos consume this same data format.
This module is COMPLETELY ISOLATED - no imports from main application.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


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
    source: str      # Source node ID
    target: str      # Target node ID
    volume_per_hour: float
    is_blocked: bool = False


@dataclass
class SchematicData:
    """Complete schematic state for rendering."""
    timestamp: str   # Display label (e.g., "Hour 14.5")
    nodes: Dict[str, NodeState]
    edges: List[FlowEdge]

    # Summary metrics
    total_in_system: int
    total_throughput_24h: int
    overall_status: str  # "normal" | "warning" | "critical"


def create_sample_data() -> SchematicData:
    """
    Create sample schematic data for demos.

    This represents a hospital mid-simulation with:
    - ED at 80% capacity (warning)
    - ITU at 100% capacity (critical, blocking)
    - Some downstream blocking occurring
    """
    nodes = {
        # Entry nodes
        "ambulance": NodeState(
            id="ambulance",
            label="Ambulance",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=12.0,
            mean_wait_mins=0.0,
        ),
        "walkin": NodeState(
            id="walkin",
            label="Walk-in",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=8.0,
            mean_wait_mins=0.0,
        ),
        "hems": NodeState(
            id="hems",
            label="HEMS",
            node_type="entry",
            capacity=None,
            occupied=0,
            throughput_per_hour=0.5,
            mean_wait_mins=0.0,
        ),

        # Process nodes
        "triage": NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=2,
            occupied=1,
            throughput_per_hour=20.5,
            mean_wait_mins=3.2,
        ),

        # Resource nodes
        "ed_bays": NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=30,
            occupied=24,
            throughput_per_hour=18.0,
            mean_wait_mins=23.0,
        ),
        "theatre": NodeState(
            id="theatre",
            label="Theatre",
            node_type="resource",
            capacity=2,
            occupied=1,
            throughput_per_hour=1.5,
            mean_wait_mins=45.0,
        ),
        "itu": NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=10,
            occupied=10,  # Full!
            throughput_per_hour=0.8,
            mean_wait_mins=120.0,
        ),
        "ward": NodeState(
            id="ward",
            label="Ward",
            node_type="resource",
            capacity=50,
            occupied=38,
            throughput_per_hour=3.2,
            mean_wait_mins=35.0,
        ),

        # Exit node
        "discharge": NodeState(
            id="discharge",
            label="Discharge",
            node_type="exit",
            capacity=None,
            occupied=0,
            throughput_per_hour=15.0,
            mean_wait_mins=0.0,
        ),
    }

    edges = [
        # Arrivals -> Triage
        FlowEdge(source="ambulance", target="triage", volume_per_hour=12.0),
        FlowEdge(source="walkin", target="triage", volume_per_hour=8.0),
        FlowEdge(source="hems", target="triage", volume_per_hour=0.5),

        # Triage -> ED
        FlowEdge(source="triage", target="ed_bays", volume_per_hour=20.5),

        # ED -> Downstream
        FlowEdge(source="ed_bays", target="theatre", volume_per_hour=1.5),
        FlowEdge(source="ed_bays", target="itu", volume_per_hour=0.8, is_blocked=True),
        FlowEdge(source="ed_bays", target="ward", volume_per_hour=5.0),
        FlowEdge(source="ed_bays", target="discharge", volume_per_hour=10.0),

        # Theatre -> Post-op
        FlowEdge(source="theatre", target="itu", volume_per_hour=0.5, is_blocked=True),
        FlowEdge(source="theatre", target="ward", volume_per_hour=0.8),

        # ITU -> Step-down/Discharge
        FlowEdge(source="itu", target="ward", volume_per_hour=0.6, is_blocked=True),
        FlowEdge(source="itu", target="discharge", volume_per_hour=0.2),

        # Ward -> Discharge
        FlowEdge(source="ward", target="discharge", volume_per_hour=4.0),
    ]

    return SchematicData(
        timestamp="Hour 14.5",
        nodes=nodes,
        edges=edges,
        total_in_system=73,
        total_throughput_24h=432,
        overall_status="warning",
    )


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
