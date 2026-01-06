"""Capacity schematic diagram generator (Phase 5f).

Generates Graphviz DOT strings for visualizing hospital capacity
and patient flow.
"""

from faer.core.scenario import FullScenario
from faer.core.entities import NodeType


def build_capacity_graph(scenario: FullScenario) -> str:
    """Generate Graphviz DOT string for capacity schematic.

    Args:
        scenario: Current scenario configuration

    Returns:
        DOT string for st.graphviz_chart()
    """
    # Get capacities from scenario
    n_amb = scenario.n_ambulances
    n_heli = scenario.n_helicopters
    n_handover = scenario.n_handover_bays
    n_triage = scenario.n_triage
    n_ed = scenario.n_ed_bays

    # Get downstream capacities from node_configs
    n_surgery = scenario.node_configs.get(NodeType.SURGERY, type('', (), {'capacity': 2})()).capacity
    n_itu = scenario.node_configs.get(NodeType.ITU, type('', (), {'capacity': 6})()).capacity
    n_ward = scenario.node_configs.get(NodeType.WARD, type('', (), {'capacity': 30})()).capacity

    # Build DOT string
    dot = f'''
    digraph HospitalFlow {{
        rankdir=LR;
        bgcolor="transparent";

        // Node styling
        node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11];

        // Fleet (yellow)
        subgraph cluster_fleet {{
            label="Fleet";
            style="dashed";
            color="gray";

            Ambulances [label="Ambulances\\n{n_amb} vehicles", fillcolor="#FFE4B5"];
            Helicopters [label="Helicopters\\n{n_heli} vehicles", fillcolor="#FFE4B5"];
        }}

        // Walk-in (no resource constraint)
        WalkIn [label="Walk-in\\nUnlimited", fillcolor="#E8E8E8"];

        // Handover (orange - bottleneck potential)
        Handover [label="Handover\\n{n_handover} bays", fillcolor="#FFDAB9"];

        // ED (green)
        subgraph cluster_ed {{
            label="Emergency Dept";
            style="dashed";
            color="gray";

            Triage [label="Triage\\n{n_triage} staff", fillcolor="#90EE90"];
            ED [label="ED Bays\\n{n_ed} beds", fillcolor="#98FB98"];
        }}

        // Downstream (blue)
        subgraph cluster_downstream {{
            label="Inpatient";
            style="dashed";
            color="gray";

            Surgery [label="Surgery\\n{n_surgery} tables", fillcolor="#ADD8E6"];
            ITU [label="ITU\\n{n_itu} beds", fillcolor="#87CEEB"];
            Ward [label="Ward\\n{n_ward} beds", fillcolor="#B0E0E6"];
        }}

        // Exit
        Exit [label="Exit", shape=oval, fillcolor="#D3D3D3"];

        // Edges with labels
        Ambulances -> Handover [label="P1-P4"];
        Helicopters -> Handover [label="P1-P2"];
        WalkIn -> Triage [label="P3-P4"];
        Handover -> Triage;
        Triage -> ED [label="All"];
        ED -> Surgery [label="Surgical", style="dashed"];
        ED -> ITU [label="Critical", style="dashed"];
        ED -> Ward [label="Admit"];
        ED -> Exit [label="Discharge"];
        Surgery -> ITU [style="dotted"];
        Surgery -> Ward;
        ITU -> Ward [label="Step-down"];
        Ward -> Exit;
    }}
    '''

    return dot


def build_simple_schematic(
    n_ambulances: int = 10,
    n_helicopters: int = 2,
    n_handover: int = 4,
    n_triage: int = 2,
    n_ed_bays: int = 20,
) -> str:
    """Generate simplified schematic with just key parameters.

    Useful when full scenario is not yet created.

    Args:
        n_ambulances: Number of ambulance vehicles
        n_helicopters: Number of helicopter vehicles
        n_handover: Number of handover bays
        n_triage: Number of triage staff
        n_ed_bays: Number of ED bays

    Returns:
        DOT string for st.graphviz_chart()
    """
    dot = f'''
    digraph HospitalFlow {{
        rankdir=LR;
        bgcolor="transparent";

        node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11];

        // Fleet
        Ambulances [label="Ambulances\\n{n_ambulances}", fillcolor="#FFE4B5"];
        Helicopters [label="Helicopters\\n{n_helicopters}", fillcolor="#FFE4B5"];
        WalkIn [label="Walk-in\\n(unlimited)", fillcolor="#E8E8E8"];

        // Hospital
        Handover [label="Handover\\n{n_handover} bays", fillcolor="#FFDAB9"];
        Triage [label="Triage\\n{n_triage}", fillcolor="#90EE90"];
        ED [label="ED Bays\\n{n_ed_bays}", fillcolor="#98FB98"];
        Exit [label="Exit", shape=oval, fillcolor="#D3D3D3"];

        // Flow
        Ambulances -> Handover;
        Helicopters -> Handover;
        WalkIn -> Triage;
        Handover -> Triage;
        Triage -> ED;
        ED -> Exit;
    }}
    '''

    return dot
