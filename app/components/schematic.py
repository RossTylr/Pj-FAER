"""Capacity schematic diagram generator (Phase 5f, Enhanced Phase 8).

Generates Graphviz DOT strings for visualizing hospital capacity
and patient flow, including feedback loops showing blocking cascades.
"""

from faer.core.scenario import FullScenario
from faer.core.entities import NodeType, DiagnosticType


def build_capacity_graph(scenario: FullScenario) -> str:
    """Generate comprehensive Graphviz DOT string for capacity schematic.

    Phase 8: Enhanced schematic showing:
    - All resources with capacities
    - Patient flow paths
    - Feedback loops (blocking cascades)
    - Diagnostics loop
    - Transfer pathway

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
    surgery_config = scenario.node_configs.get(NodeType.SURGERY)
    n_surgery = surgery_config.capacity if surgery_config else 2

    itu_config = scenario.node_configs.get(NodeType.ITU)
    n_itu = itu_config.capacity if itu_config else 6

    ward_config = scenario.node_configs.get(NodeType.WARD)
    n_ward = ward_config.capacity if ward_config else 30

    # Get diagnostic capacities (Phase 7)
    ct_config = scenario.diagnostic_configs.get(DiagnosticType.CT_SCAN)
    n_ct = ct_config.capacity if ct_config and ct_config.enabled else 2

    xray_config = scenario.diagnostic_configs.get(DiagnosticType.XRAY)
    n_xray = xray_config.capacity if xray_config and xray_config.enabled else 3

    bloods_config = scenario.diagnostic_configs.get(DiagnosticType.BLOODS)
    n_bloods = bloods_config.capacity if bloods_config and bloods_config.enabled else 5

    # Get transfer capacities (Phase 7)
    transfer_config = scenario.transfer_config
    n_transfer_amb = transfer_config.n_transfer_ambulances if transfer_config else 2
    n_transfer_heli = transfer_config.n_transfer_helicopters if transfer_config else 1

    # Build DOT string
    dot = f'''
    digraph HospitalSystem {{
        rankdir=LR;
        bgcolor="white";
        fontname="Arial";
        label="Hospital Patient Flow & Feedback Loops";
        labelloc="t";
        fontsize=16;

        // Graph settings
        node [fontname="Arial", fontsize=10];
        edge [fontname="Arial", fontsize=9];

        // ==================== ARRIVAL ====================
        subgraph cluster_arrival {{
            label="Arrival";
            style="rounded,filled";
            fillcolor="#FFF8DC";

            Ambulances [label="Ambulances\\n{n_amb} vehicles", shape=box, style="rounded,filled", fillcolor="#FFE4B5"];
            Helicopters [label="Helicopters\\n{n_heli} vehicles", shape=box, style="rounded,filled", fillcolor="#FFE4B5"];
            WalkIn [label="Walk-in\\nUnlimited", shape=box, style="rounded,filled", fillcolor="#E8E8E8"];
        }}

        // ==================== HANDOVER ====================
        Handover [label="Handover Bay\\n{n_handover} bays", shape=box, style="rounded,filled", fillcolor="#FFDAB9"];

        // ==================== EMERGENCY DEPT ====================
        subgraph cluster_ed {{
            label="Emergency Department";
            style="rounded,filled";
            fillcolor="#F0FFF0";

            Triage [label="Triage\\n{n_triage} clinicians", shape=box, style="rounded,filled", fillcolor="#90EE90"];
            ED [label="ED Bays\\n{n_ed} beds", shape=box, style="rounded,filled", fillcolor="#98FB98"];
        }}

        // ==================== DIAGNOSTICS ====================
        subgraph cluster_diag {{
            label="Diagnostics";
            style="rounded,filled";
            fillcolor="#F5F0FF";

            CT [label="CT Scanner\\n{n_ct} scanners", shape=box, style="rounded,filled", fillcolor="#DDA0DD"];
            Xray [label="X-ray\\n{n_xray} rooms", shape=box, style="rounded,filled", fillcolor="#DDA0DD"];
            Bloods [label="Bloods\\n{n_bloods} phleb", shape=box, style="rounded,filled", fillcolor="#DDA0DD"];
        }}

        // ==================== TRANSFER ====================
        subgraph cluster_transfer {{
            label="Specialist Transfer";
            style="rounded,filled";
            fillcolor="#FFF0F5";

            TransferAmb [label="Transfer Amb\\n{n_transfer_amb} vehicles", shape=box, style="rounded,filled", fillcolor="#FFB6C1"];
            TransferHeli [label="Air Ambulance\\n{n_transfer_heli} aircraft", shape=box, style="rounded,filled", fillcolor="#FFB6C1"];
            SpecialistCentre [label="Specialist\\nCentre", shape=oval, style="filled", fillcolor="#FF69B4"];
        }}

        // ==================== SURGICAL ====================
        subgraph cluster_surgical {{
            label="Surgical";
            style="rounded,filled";
            fillcolor="#F0F8FF";

            Theatre [label="Theatre\\n{n_surgery} tables", shape=box, style="rounded,filled", fillcolor="#87CEEB"];
        }}

        // ==================== INPATIENT ====================
        subgraph cluster_inpatient {{
            label="Inpatient";
            style="rounded,filled";
            fillcolor="#F0FFFF";

            ITU [label="ITU\\n{n_itu} beds", shape=box, style="rounded,filled", fillcolor="#ADD8E6"];
            Ward [label="Ward\\n{n_ward} beds", shape=box, style="rounded,filled", fillcolor="#B0E0E6"];
        }}

        // ==================== EXIT ====================
        Exit [label="Exit /\\nDischarge", shape=oval, style="filled", fillcolor="#D3D3D3"];

        // ==================== FORWARD FLOW (solid blue) ====================
        Ambulances -> Handover [color="blue", penwidth=2];
        Helicopters -> Handover [color="blue", penwidth=2];
        WalkIn -> Triage [color="blue", penwidth=2];
        Handover -> Triage [color="blue", penwidth=2];
        Triage -> ED [color="blue", penwidth=2];

        // ED to downstream
        ED -> Theatre [label="Surgical", color="blue"];
        ED -> ITU [label="Critical", color="blue"];
        ED -> Ward [label="Admit", color="blue"];
        ED -> Exit [label="Discharge", color="blue"];

        // Surgical pathway
        Theatre -> ITU [label="Post-op critical", color="blue"];
        Theatre -> Ward [label="Post-op", color="blue"];

        // ITU step-down
        ITU -> Ward [label="Step-down", color="blue"];

        // Ward discharge
        Ward -> Exit [label="Discharge", color="blue", penwidth=2];

        // ==================== DIAGNOSTICS LOOP (purple dashed) ====================
        ED -> CT [label="scan", style="dashed", color="purple"];
        CT -> ED [label="return", style="dashed", color="purple"];
        ED -> Xray [label="scan", style="dashed", color="purple"];
        Xray -> ED [label="return", style="dashed", color="purple"];
        ED -> Bloods [label="test", style="dashed", color="purple"];
        Bloods -> ED [label="return", style="dashed", color="purple"];

        // ==================== TRANSFER PATHWAY (pink) ====================
        ED -> TransferAmb [label="critical\\ntransfer", color="deeppink", style="dashed"];
        ED -> TransferHeli [label="air\\ntransfer", color="deeppink", style="dashed"];
        TransferAmb -> SpecialistCentre [color="deeppink"];
        TransferHeli -> SpecialistCentre [color="deeppink"];

        // ==================== FEEDBACK LOOPS (red, thick, labeled) ====================
        // These show blocking/backpressure
        Ward -> ED [label="BLOCKED:\\nED boarding", color="red", style="bold", penwidth=2, constraint=false];
        ITU -> Theatre [label="BLOCKED:\\nSurgery delay", color="red", style="bold", penwidth=2, constraint=false];
        ED -> Handover [label="BLOCKED:\\nHandover delay", color="red", style="bold", penwidth=2, constraint=false];

        // ==================== LEGEND ====================
        subgraph cluster_legend {{
            label="Legend";
            style="rounded";
            fontsize=10;

            leg1 [label="Blue: Patient Flow", shape=plaintext];
            leg2 [label="Purple: Diagnostics Loop", shape=plaintext];
            leg3 [label="Pink: Transfer Out", shape=plaintext];
            leg4 [label="Red: BLOCKING", shape=plaintext];
        }}
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
        n_triage: Number of triage clinicians
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


def build_capacity_graph_from_params(
    n_ambulances: int,
    n_helicopters: int,
    n_handover: int,
    n_triage: int,
    n_ed_bays: int,
    n_ct: int,
    n_xray: int,
    n_bloods: int,
    n_theatre: int,
    n_itu: int,
    n_ward: int,
    ct_enabled: bool = True,
    xray_enabled: bool = True,
    bloods_enabled: bool = True
) -> str:
    """Generate Graphviz DOT with all resources and feedback loops.

    Used by the Schematic page when scenario is not yet built.

    Args:
        n_ambulances: Number of ambulances
        n_helicopters: Number of helicopters
        n_handover: Number of handover bays
        n_triage: Number of triage clinicians
        n_ed_bays: Number of ED bays
        n_ct: Number of CT scanners
        n_xray: Number of X-ray rooms
        n_bloods: Number of phlebotomists
        n_theatre: Number of theatre tables
        n_itu: Number of ITU beds
        n_ward: Number of ward beds
        ct_enabled: Whether CT is enabled
        xray_enabled: Whether X-ray is enabled
        bloods_enabled: Whether bloods is enabled

    Returns:
        DOT string for st.graphviz_chart()
    """
    # Build diagnostic nodes conditionally
    diag_nodes = ""
    diag_edges = ""

    if ct_enabled:
        diag_nodes += f'CT [label="CT\\n{n_ct}", fillcolor="#DDA0DD"];\n'
        diag_edges += 'ED -> CT [style="dashed", color="purple"];\n'
        diag_edges += 'CT -> ED [style="dashed", color="purple"];\n'

    if xray_enabled:
        diag_nodes += f'Xray [label="X-ray\\n{n_xray}", fillcolor="#DDA0DD"];\n'
        diag_edges += 'ED -> Xray [style="dashed", color="purple"];\n'
        diag_edges += 'Xray -> ED [style="dashed", color="purple"];\n'

    if bloods_enabled:
        diag_nodes += f'Bloods [label="Bloods\\n{n_bloods}", fillcolor="#DDA0DD"];\n'
        diag_edges += 'ED -> Bloods [style="dashed", color="purple"];\n'
        diag_edges += 'Bloods -> ED [style="dashed", color="purple"];\n'

    return f'''
    digraph Hospital {{
        rankdir=LR; bgcolor="white";
        node [shape=box, style="rounded,filled", fontsize=10];

        // Arrival
        Amb [label="Ambulances\\n{n_ambulances}", fillcolor="#FFE4B5"];
        Heli [label="HEMS\\n{n_helicopters}", fillcolor="#FFE4B5"];
        Walk [label="Walk-in", fillcolor="#E8E8E8"];

        // Front door
        Handover [label="Handover\\n{n_handover} bays", fillcolor="#FFDAB9"];
        Triage [label="Triage\\n{n_triage}", fillcolor="#90EE90"];
        ED [label="ED Bays\\n{n_ed_bays}", fillcolor="#98FB98"];

        // Diagnostics
        {diag_nodes}

        // Downstream
        Theatre [label="Theatre\\n{n_theatre}", fillcolor="#87CEEB"];
        ITU [label="ITU\\n{n_itu}", fillcolor="#ADD8E6"];
        Ward [label="Ward\\n{n_ward}", fillcolor="#B0E0E6"];
        Exit [label="Exit", shape=oval, fillcolor="#D3D3D3"];

        // Forward flow (blue)
        Amb -> Handover [color="blue", penwidth=2];
        Heli -> Handover [color="blue", penwidth=2];
        Walk -> Triage [color="blue", penwidth=2];
        Handover -> Triage [color="blue", penwidth=2];
        Triage -> ED [color="blue", penwidth=2];
        ED -> Theatre [color="blue"]; ED -> ITU [color="blue"];
        ED -> Ward [color="blue"]; ED -> Exit [color="blue"];
        Theatre -> ITU [color="blue"]; Theatre -> Ward [color="blue"];
        ITU -> Ward [color="blue"]; Ward -> Exit [color="blue", penwidth=2];

        // Diagnostics loop (purple)
        {diag_edges}

        // Feedback (red)
        Ward -> ED [label="BLOCKED", color="red", penwidth=2, constraint=false];
        ED -> Handover [label="BLOCKED", color="red", penwidth=2, constraint=false];
    }}'''


def build_feedback_diagram() -> str:
    """Focused feedback cascade diagram.

    Shows the blocking cascade that occurs when downstream beds fill up.

    Returns:
        DOT string for st.graphviz_chart()
    """
    return '''
    digraph Cascade {
        rankdir=LR; node [shape=box, style="rounded,filled"];
        Ward [label="Ward FULL", fillcolor="#FFB6C1"];
        ITU [label="ITU Blocked", fillcolor="#FFB6C1"];
        ED [label="ED Boarding", fillcolor="#FFB6C1"];
        Handover [label="Handover Delayed", fillcolor="#FFB6C1"];
        Ambulance [label="Ambulances Queuing", fillcolor="#FFB6C1"];
        Community [label="999 Delayed", fillcolor="#FF6B6B"];

        Ward -> ITU [label="can\\'t step down", color="red", penwidth=2];
        Ward -> ED [label="can\\'t admit", color="red", penwidth=2];
        ED -> Handover [label="can\\'t accept", color="red", penwidth=2];
        Handover -> Ambulance [label="crews wait", color="red", penwidth=2];
        Ambulance -> Community [label="unavailable", color="red", penwidth=2];
    }'''


def build_results_schematic(
    util_ed: float,
    util_itu: float,
    util_ward: float,
    util_theatre: float,
    util_ct: float,
    util_handover: float,
    n_ed: int,
    n_itu: int,
    n_ward: int,
    n_theatre: int
) -> str:
    """Schematic colored by utilisation results.

    Visualizes the system state based on simulation results, with
    color-coding to highlight bottlenecks.

    Args:
        util_ed: ED bay utilisation (0-1)
        util_itu: ITU utilisation (0-1)
        util_ward: Ward utilisation (0-1)
        util_theatre: Theatre utilisation (0-1)
        util_ct: CT scanner utilisation (0-1)
        util_handover: Handover bay utilisation (0-1)
        n_ed: Number of ED bays
        n_itu: Number of ITU beds
        n_ward: Number of ward beds
        n_theatre: Number of theatre tables

    Returns:
        DOT string for st.graphviz_chart()
    """
    def color(u: float) -> str:
        """Get color based on utilisation level."""
        if u > 0.85:
            return "#FF6B6B"  # Red - critical
        elif u > 0.70:
            return "#FFD93D"  # Yellow - warning
        else:
            return "#6BCB77"  # Green - ok

    def pct(u: float) -> str:
        """Format utilisation as percentage."""
        return f"{u:.0%}" if u > 0 else "-"

    return f'''
    digraph Results {{
        rankdir=LR; node [shape=box, style="rounded,filled"];
        Handover [label="Handover\\n{pct(util_handover)}", fillcolor="{color(util_handover)}"];
        ED [label="ED ({n_ed})\\n{pct(util_ed)}", fillcolor="{color(util_ed)}"];
        CT [label="CT\\n{pct(util_ct)}", fillcolor="{color(util_ct)}"];
        Theatre [label="Theatre ({n_theatre})\\n{pct(util_theatre)}", fillcolor="{color(util_theatre)}"];
        ITU [label="ITU ({n_itu})\\n{pct(util_itu)}", fillcolor="{color(util_itu)}"];
        Ward [label="Ward ({n_ward})\\n{pct(util_ward)}", fillcolor="{color(util_ward)}"];

        Handover -> ED; ED -> CT [style="dashed"]; CT -> ED [style="dashed"];
        ED -> Theatre; ED -> ITU; ED -> Ward;
        Theatre -> ITU; Theatre -> Ward; ITU -> Ward;
    }}'''
