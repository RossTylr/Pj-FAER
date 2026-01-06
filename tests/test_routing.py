"""Tests for routing matrix (Phase 5 - simplified ED)."""

import pytest
from collections import Counter

from faer.core.entities import NodeType, Priority
from faer.core.scenario import FullScenario, RoutingRule


class TestRoutingRule:
    """Test RoutingRule dataclass."""

    def test_create_routing_rule(self):
        """Can create a routing rule."""
        rule = RoutingRule(
            from_node=NodeType.ED_BAYS,
            to_node=NodeType.SURGERY,
            priority=Priority.P1_IMMEDIATE,
            probability=0.3,
        )
        assert rule.from_node == NodeType.ED_BAYS
        assert rule.to_node == NodeType.SURGERY
        assert rule.priority == Priority.P1_IMMEDIATE
        assert rule.probability == 0.3


class TestRoutingValidation:
    """Test routing validation."""

    def test_default_routing_valid(self):
        """Default routing is valid (sums to 1.0)."""
        scenario = FullScenario()
        assert len(scenario.routing) > 0

    def test_routing_validation_catches_bad_sum(self):
        """Routing that doesn't sum to 1.0 raises ValueError."""
        bad_rules = [
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P1_IMMEDIATE, 0.5),
            # Missing 0.5 probability
        ]
        with pytest.raises(ValueError, match="sums to"):
            FullScenario(routing=bad_rules)

    def test_routing_validation_catches_over_sum(self):
        """Routing that sums to more than 1.0 raises ValueError."""
        bad_rules = [
            RoutingRule(NodeType.ED_BAYS, NodeType.WARD, Priority.P1_IMMEDIATE, 0.6),
            RoutingRule(NodeType.ED_BAYS, NodeType.EXIT, Priority.P1_IMMEDIATE, 0.6),
        ]
        with pytest.raises(ValueError, match="sums to"):
            FullScenario(routing=bad_rules)


class TestGetNextNode:
    """Test get_next_node sampling."""

    def test_get_next_node_returns_valid_node(self):
        """get_next_node returns a valid NodeType."""
        scenario = FullScenario(random_seed=42)
        next_node = scenario.get_next_node(NodeType.ED_BAYS, Priority.P1_IMMEDIATE)
        assert isinstance(next_node, NodeType)
        assert next_node in [NodeType.SURGERY, NodeType.ITU, NodeType.WARD, NodeType.EXIT]

    def test_get_next_node_fallback_to_exit(self):
        """get_next_node returns EXIT for undefined routes."""
        scenario = FullScenario(random_seed=42)
        # EXIT has no outgoing routes
        next_node = scenario.get_next_node(NodeType.EXIT, Priority.P1_IMMEDIATE)
        assert next_node == NodeType.EXIT

    def test_get_next_node_samples_correctly(self):
        """Over many samples, distribution matches probabilities."""
        scenario = FullScenario(random_seed=42)

        # Sample 1000 times from ED_BAYS P1
        # Expected: Surgery 30%, ITU 40%, Ward 20%, Exit 10%
        results = [scenario.get_next_node(NodeType.ED_BAYS, Priority.P1_IMMEDIATE)
                   for _ in range(1000)]

        counts = Counter(results)

        # Check roughly matches expected (±5%)
        assert 0.25 < counts[NodeType.SURGERY] / 1000 < 0.35
        assert 0.35 < counts[NodeType.ITU] / 1000 < 0.45
        assert 0.15 < counts[NodeType.WARD] / 1000 < 0.25
        assert 0.05 < counts[NodeType.EXIT] / 1000 < 0.15

    def test_ward_always_goes_to_exit(self):
        """Ward patients always exit."""
        scenario = FullScenario(random_seed=42)

        for priority in Priority:
            for _ in range(100):
                next_node = scenario.get_next_node(NodeType.WARD, priority)
                assert next_node == NodeType.EXIT

    def test_routing_reproducibility(self):
        """Same seed produces same routing decisions."""
        scenario1 = FullScenario(random_seed=42)
        scenario2 = FullScenario(random_seed=42)

        results1 = [scenario1.get_next_node(NodeType.ED_BAYS, Priority.P1_IMMEDIATE)
                    for _ in range(100)]
        results2 = [scenario2.get_next_node(NodeType.ED_BAYS, Priority.P1_IMMEDIATE)
                    for _ in range(100)]

        assert results1 == results2


class TestNodeType:
    """Test NodeType enum - Phase 5 simplified."""

    def test_node_types_exist(self):
        """All expected node types exist (Phase 5)."""
        assert NodeType.TRIAGE.value == 1
        assert NodeType.ED_BAYS.value == 2
        assert NodeType.SURGERY.value == 3
        assert NodeType.ITU.value == 4
        assert NodeType.WARD.value == 5
        assert NodeType.EXIT.value == 6

    def test_ed_nodes(self):
        """ED nodes are distinct from downstream nodes (Phase 5)."""
        ed_nodes = {NodeType.TRIAGE, NodeType.ED_BAYS}
        downstream_nodes = {NodeType.SURGERY, NodeType.ITU, NodeType.WARD}

        assert ed_nodes.isdisjoint(downstream_nodes)
