"""Tests for capacity schematic diagram (Phase 5f)."""

import sys
from pathlib import Path

import pytest

# Add app directory to path
app_dir = Path(__file__).parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from faer.core.scenario import FullScenario


class TestSchematicGeneration:
    """Test schematic DOT string generation."""

    def test_simple_schematic_generates_valid_dot(self):
        """Simple schematic generates parseable DOT string."""
        from components.schematic import build_simple_schematic

        dot = build_simple_schematic()

        assert "digraph" in dot
        assert "ED Bays" in dot
        assert "Triage" in dot
        assert "Handover" in dot

    def test_simple_schematic_with_custom_values(self):
        """Simple schematic uses custom values."""
        from components.schematic import build_simple_schematic

        dot = build_simple_schematic(
            n_ambulances=15,
            n_helicopters=5,
            n_handover=8,
            n_triage=4,
            n_ed_bays=25,
        )

        assert "15" in dot  # n_ambulances
        assert "5" in dot   # n_helicopters
        assert "8" in dot   # n_handover
        assert "4" in dot   # n_triage
        assert "25" in dot  # n_ed_bays

    def test_full_schematic_generates_valid_dot(self):
        """Full schematic generates parseable DOT string."""
        from components.schematic import build_capacity_graph

        scenario = FullScenario()
        dot = build_capacity_graph(scenario)

        assert "digraph" in dot
        assert "ED Bays" in dot
        assert str(scenario.n_ed_bays) in dot
        assert str(scenario.n_ambulances) in dot

    def test_full_schematic_includes_downstream(self):
        """Full schematic includes downstream nodes."""
        from components.schematic import build_capacity_graph

        scenario = FullScenario()
        dot = build_capacity_graph(scenario)

        assert "Surgery" in dot
        assert "ITU" in dot
        assert "Ward" in dot
        assert "Exit" in dot

    def test_schematic_has_edges(self):
        """Schematic includes flow edges."""
        from components.schematic import build_simple_schematic

        dot = build_simple_schematic()

        assert "->" in dot  # Graphviz edge syntax
        assert "Ambulances -> Handover" in dot
        assert "Triage -> ED" in dot
