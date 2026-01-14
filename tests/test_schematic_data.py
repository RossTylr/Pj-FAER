"""Tests for schematic data transformation functions.

Tests the transformation of simulation results and configuration
into SchematicData format for the React component.
"""

import json
import pytest
from unittest.mock import Mock

from app.components.react_schematic.data import (
    NodeState,
    FlowEdge,
    SchematicData,
    to_dict,
    build_schematic_from_config,
    build_schematic_from_results,
)


class TestNodeState:
    """Tests for NodeState dataclass."""

    def test_utilisation_normal(self):
        """Utilisation calculated correctly for normal capacity."""
        node = NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=20,
            occupied=10,
            throughput_per_hour=5.0,
            mean_wait_mins=15.0,
        )
        assert node.utilisation == 0.5

    def test_utilisation_zero_capacity(self):
        """Utilisation is 0 when capacity is 0."""
        node = NodeState(
            id="entry",
            label="Entry",
            node_type="entry",
            capacity=0,
            occupied=0,
            throughput_per_hour=5.0,
            mean_wait_mins=0.0,
        )
        assert node.utilisation == 0.0

    def test_utilisation_none_capacity(self):
        """Utilisation is 0 when capacity is None (entry/exit nodes)."""
        node = NodeState(
            id="discharge",
            label="Discharge",
            node_type="exit",
            capacity=None,
            occupied=0,
            throughput_per_hour=10.0,
            mean_wait_mins=0.0,
        )
        assert node.utilisation == 0.0

    def test_utilisation_capped_at_100(self):
        """Utilisation capped at 1.0 even when over capacity."""
        node = NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=10,
            occupied=15,  # Over capacity
            throughput_per_hour=1.0,
            mean_wait_mins=60.0,
        )
        assert node.utilisation == 1.0

    def test_status_normal(self):
        """Status is normal when utilisation < 70%."""
        node = NodeState(
            id="triage",
            label="Triage",
            node_type="process",
            capacity=3,
            occupied=1,  # 33%
            throughput_per_hour=10.0,
            mean_wait_mins=5.0,
        )
        assert node.status == "normal"

    def test_status_warning(self):
        """Status is warning when 70% <= utilisation < 90%."""
        node = NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=20,
            occupied=16,  # 80%
            throughput_per_hour=8.0,
            mean_wait_mins=20.0,
        )
        assert node.status == "warning"

    def test_status_critical(self):
        """Status is critical when utilisation >= 90%."""
        node = NodeState(
            id="itu",
            label="ITU",
            node_type="resource",
            capacity=10,
            occupied=10,  # 100%
            throughput_per_hour=0.5,
            mean_wait_mins=120.0,
        )
        assert node.status == "critical"


class TestFlowEdge:
    """Tests for FlowEdge dataclass."""

    def test_default_not_blocked(self):
        """Edge is not blocked by default."""
        edge = FlowEdge(source="ed_bays", target="ward", volume_per_hour=5.0)
        assert edge.is_blocked is False

    def test_blocked_edge(self):
        """Edge can be marked as blocked."""
        edge = FlowEdge(source="ed_bays", target="itu", volume_per_hour=0.5, is_blocked=True)
        assert edge.is_blocked is True


class TestToDict:
    """Tests for to_dict serialization."""

    def test_json_serializable(self):
        """to_dict output is JSON serializable."""
        data = SchematicData(
            timestamp="Test",
            nodes={
                "test": NodeState(
                    id="test",
                    label="Test",
                    node_type="resource",
                    capacity=10,
                    occupied=5,
                    throughput_per_hour=2.0,
                    mean_wait_mins=10.0,
                )
            },
            edges=[FlowEdge(source="a", target="b", volume_per_hour=1.0)],
            total_in_system=5,
            total_throughput_24h=48,
            overall_status="normal",
        )

        result = to_dict(data)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_includes_computed_properties(self):
        """to_dict includes computed utilisation and status."""
        node = NodeState(
            id="ed_bays",
            label="ED Bays",
            node_type="resource",
            capacity=20,
            occupied=18,  # 90% - critical
            throughput_per_hour=8.0,
            mean_wait_mins=30.0,
        )
        data = SchematicData(
            timestamp="Test",
            nodes={"ed_bays": node},
            edges=[],
            total_in_system=18,
            total_throughput_24h=192,
            overall_status="critical",
        )

        result = to_dict(data)

        assert result["nodes"]["ed_bays"]["utilisation"] == 0.9
        assert result["nodes"]["ed_bays"]["status"] == "critical"


class TestBuildSchematicFromConfig:
    """Tests for build_schematic_from_config."""

    def test_config_mode_zero_occupied(self):
        """Config mode has all nodes at 0 occupied."""
        session_state = {
            "n_ed_bays": 20,
            "n_itu_beds": 6,
            "n_ward_beds": 30,
            "n_triage": 3,
        }

        data = build_schematic_from_config(session_state)

        for node in data.nodes.values():
            assert node.occupied == 0

    def test_config_mode_uses_capacities(self):
        """Config mode uses capacities from session state."""
        session_state = {
            "n_ed_bays": 25,
            "n_itu_beds": 8,
            "n_ward_beds": 40,
            "n_triage": 4,
        }

        data = build_schematic_from_config(session_state)

        assert data.nodes["ed_bays"].capacity == 25
        assert data.nodes["itu"].capacity == 8
        assert data.nodes["ward"].capacity == 40
        assert data.nodes["triage"].capacity == 4

    def test_config_mode_normal_status(self):
        """Config mode has normal overall status (0% utilisation)."""
        data = build_schematic_from_config({})

        assert data.overall_status == "normal"

    def test_config_mode_has_all_nodes(self):
        """Config mode creates all expected nodes."""
        data = build_schematic_from_config({})

        expected_nodes = [
            "ambulance",
            "walkin",
            "hems",
            "handover",
            "triage",
            "ed_bays",
            "theatre",
            "itu",
            "ward",
            "discharge",
        ]
        for node_id in expected_nodes:
            assert node_id in data.nodes

    def test_config_mode_timestamp(self):
        """Config mode has 'Configuration' timestamp."""
        data = build_schematic_from_config({})

        assert data.timestamp == "Configuration"


class TestBuildSchematicFromResults:
    """Tests for build_schematic_from_results."""

    @pytest.fixture
    def mock_scenario(self):
        """Create a mock scenario with nested configs."""
        scenario = Mock()
        scenario.run_length = 480  # 8 hours in minutes
        scenario.n_ambulances = 10
        scenario.n_helicopters = 2
        scenario.n_handover_bays = 4
        scenario.n_triage = 3
        scenario.n_ed_bays = 20

        # Nested configs
        scenario.theatre_config = Mock()
        scenario.theatre_config.n_tables = 2

        scenario.itu_config = Mock()
        scenario.itu_config.capacity = 6

        scenario.ward_config = Mock()
        scenario.ward_config.capacity = 30

        return scenario

    @pytest.fixture
    def sample_results(self):
        """Sample results dict from multiple replications."""
        return {
            "util_ed_bays": [0.75, 0.80, 0.78],
            "util_itu": [0.95, 0.98, 0.96],
            "util_ward": [0.70, 0.72, 0.71],
            "util_theatre": [0.50, 0.55, 0.52],
            "util_triage": [0.40, 0.45, 0.42],
            "util_handover": [0.30, 0.35, 0.32],
            "arrivals": [160, 165, 162],
            "departures": [155, 160, 158],
            "mean_treatment_wait": [25.0, 28.0, 26.5],
            "discharge_count": [100, 105, 102],
            "admit_ward_count": [40, 42, 41],
            "admit_icu_count": [15, 17, 16],
        }

    def test_results_mode_utilisation_to_occupied(self, mock_scenario, sample_results):
        """Results mode converts utilisation to occupied counts."""
        data = build_schematic_from_results(sample_results, mock_scenario, 8.0)

        # ED: mean util ~0.78, capacity 20, occupied = int(0.78 * 20) = 15
        assert data.nodes["ed_bays"].occupied == 15

    def test_results_mode_blocked_edges_high_utilisation(
        self, mock_scenario, sample_results
    ):
        """Edges to high-utilisation nodes are marked blocked."""
        data = build_schematic_from_results(sample_results, mock_scenario, 8.0)

        # ITU at 96% mean utilisation should have blocked incoming edges
        itu_edges = [e for e in data.edges if e.target == "itu"]
        assert any(e.is_blocked for e in itu_edges)

    def test_results_mode_overall_status_critical(self, mock_scenario, sample_results):
        """Overall status is critical when max utilisation >= 90%."""
        data = build_schematic_from_results(sample_results, mock_scenario, 8.0)

        # ITU at 96% should make overall status critical
        assert data.overall_status == "critical"

    def test_results_mode_timestamp_includes_hours(self, mock_scenario, sample_results):
        """Results mode timestamp includes run length."""
        data = build_schematic_from_results(sample_results, mock_scenario, 8.0)

        assert "8" in data.timestamp
        assert "Simulation" in data.timestamp

    def test_results_mode_throughput_calculation(self, mock_scenario, sample_results):
        """Results mode calculates throughput per hour from departures."""
        data = build_schematic_from_results(sample_results, mock_scenario, 8.0)

        # Mean departures ~158, run length 8 hours = ~19.75/hr
        # 24h throughput = ~474
        assert data.total_throughput_24h > 0

    def test_results_mode_handles_missing_metrics(self, mock_scenario):
        """Results mode handles missing metrics gracefully."""
        # Minimal results dict
        results = {
            "arrivals": [100],
            "departures": [95],
        }

        data = build_schematic_from_results(results, mock_scenario, 8.0)

        # Should not raise, should have default values
        assert data.nodes["ed_bays"].occupied >= 0


class TestSchematicIntegration:
    """Integration tests for the full schematic data flow."""

    def test_config_to_results_transition(self):
        """Schematic transitions cleanly from config to results mode."""
        # Config mode
        config_data = build_schematic_from_config({"n_ed_bays": 20})
        assert config_data.overall_status == "normal"
        assert config_data.total_in_system == 0

        # Results mode
        scenario = Mock()
        scenario.run_length = 480
        scenario.n_ambulances = 10
        scenario.n_helicopters = 2
        scenario.n_handover_bays = 4
        scenario.n_triage = 3
        scenario.n_ed_bays = 20
        scenario.theatre_config = None
        scenario.itu_config = None
        scenario.ward_config = None

        results = {
            "util_ed_bays": [0.85],
            "arrivals": [160],
            "departures": [155],
        }

        results_data = build_schematic_from_results(results, scenario, 8.0)

        # Should have different state
        assert results_data.total_in_system > 0
        assert results_data.nodes["ed_bays"].occupied > 0
