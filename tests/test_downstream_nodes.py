"""Tests for downstream nodes (Phase 4)."""

import pytest

from faer.core.entities import NodeType
from faer.core.scenario import FullScenario, NodeConfig


class TestNodeConfig:
    """Test NodeConfig dataclass."""

    def test_create_node_config(self):
        """Can create a node config."""
        config = NodeConfig(
            node_type=NodeType.SURGERY,
            capacity=3,
            service_time_mean=120.0,
            service_time_cv=0.6,
        )
        assert config.node_type == NodeType.SURGERY
        assert config.capacity == 3
        assert config.service_time_mean == 120.0
        assert config.enabled is True

    def test_node_config_defaults(self):
        """NodeConfig has sensible defaults."""
        config = NodeConfig(
            node_type=NodeType.WARD,
            capacity=20,
            service_time_mean=480.0,
        )
        assert config.service_time_cv == 0.5
        assert config.enabled is True


class TestDefaultNodeConfigs:
    """Test default node configurations."""

    def test_default_configs_created(self):
        """Default node configs are created automatically."""
        scenario = FullScenario()
        assert len(scenario.node_configs) == 6

    def test_all_nodes_configured(self):
        """All nodes have configs."""
        scenario = FullScenario()
        expected_nodes = [
            NodeType.RESUS, NodeType.MAJORS, NodeType.MINORS,
            NodeType.SURGERY, NodeType.ITU, NodeType.WARD
        ]
        for node in expected_nodes:
            assert node in scenario.node_configs

    def test_exit_not_in_configs(self):
        """EXIT node doesn't need a config (no resources)."""
        scenario = FullScenario()
        assert NodeType.EXIT not in scenario.node_configs

    def test_downstream_nodes_have_larger_capacity(self):
        """Ward has larger capacity than ED nodes."""
        scenario = FullScenario()
        ward_cap = scenario.node_configs[NodeType.WARD].capacity
        resus_cap = scenario.node_configs[NodeType.RESUS].capacity
        assert ward_cap > resus_cap

    def test_surgery_has_long_service_time(self):
        """Surgery has longer service time than ED."""
        scenario = FullScenario()
        surgery_time = scenario.node_configs[NodeType.SURGERY].service_time_mean
        majors_time = scenario.node_configs[NodeType.MAJORS].service_time_mean
        assert surgery_time > majors_time


class TestNodeConfigModification:
    """Test modifying node configs."""

    def test_can_modify_capacity(self):
        """Can modify node capacity after creation."""
        scenario = FullScenario()
        original_cap = scenario.node_configs[NodeType.WARD].capacity

        # Modify capacity
        scenario.node_configs[NodeType.WARD].capacity = 5

        assert scenario.node_configs[NodeType.WARD].capacity == 5
        assert scenario.node_configs[NodeType.WARD].capacity != original_cap

    def test_can_disable_node(self):
        """Can disable a node."""
        scenario = FullScenario()
        scenario.node_configs[NodeType.SURGERY].enabled = False

        assert scenario.node_configs[NodeType.SURGERY].enabled is False
