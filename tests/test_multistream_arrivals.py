"""Tests for multi-stream arrivals (Phase 4)."""

import pytest
from collections import defaultdict

from faer.core.entities import ArrivalMode, Priority
from faer.core.scenario import FullScenario, ArrivalConfig
from faer.model.full_model import run_full_simulation


class TestArrivalMode:
    """Test ArrivalMode enum."""

    def test_arrival_modes_exist(self):
        """All expected arrival modes exist."""
        assert ArrivalMode.AMBULANCE.value == 1
        assert ArrivalMode.HELICOPTER.value == 2
        assert ArrivalMode.SELF_PRESENTATION.value == 3


class TestArrivalConfig:
    """Test ArrivalConfig dataclass."""

    def test_create_arrival_config(self):
        """Can create an arrival config."""
        config = ArrivalConfig(
            mode=ArrivalMode.AMBULANCE,
            hourly_rates=[1.0] * 24,
            triage_mix={
                Priority.P1_IMMEDIATE: 0.2,
                Priority.P2_VERY_URGENT: 0.3,
                Priority.P3_URGENT: 0.3,
                Priority.P4_STANDARD: 0.2,
            }
        )
        assert config.mode == ArrivalMode.AMBULANCE
        assert len(config.hourly_rates) == 24

    def test_arrival_config_validates_hourly_rates(self):
        """hourly_rates must have 24 values."""
        with pytest.raises(ValueError, match="24 values"):
            ArrivalConfig(
                mode=ArrivalMode.AMBULANCE,
                hourly_rates=[1.0] * 12,  # Only 12 values
                triage_mix={Priority.P1_IMMEDIATE: 1.0}
            )

    def test_arrival_config_validates_triage_mix(self):
        """triage_mix must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ArrivalConfig(
                mode=ArrivalMode.AMBULANCE,
                hourly_rates=[1.0] * 24,
                triage_mix={
                    Priority.P1_IMMEDIATE: 0.5,
                    Priority.P2_VERY_URGENT: 0.3,
                    # Missing 0.2
                }
            )


class TestDefaultArrivalConfigs:
    """Test default arrival configurations."""

    def test_default_configs_created(self):
        """Default configs are created automatically."""
        scenario = FullScenario()
        assert len(scenario.arrival_configs) == 3

    def test_default_configs_modes(self):
        """Default configs cover all modes."""
        scenario = FullScenario()
        modes = {c.mode for c in scenario.arrival_configs}
        assert ArrivalMode.AMBULANCE in modes
        assert ArrivalMode.HELICOPTER in modes
        assert ArrivalMode.SELF_PRESENTATION in modes


class TestMultiStreamSimulation:
    """Test multi-stream simulation."""

    def test_multistream_runs(self):
        """Multi-stream simulation runs without error."""
        scenario = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario, use_multistream=True)

        assert results["arrivals"] > 0
        assert results["departures"] > 0

    def test_single_stream_still_works(self):
        """Single-stream mode still works (backwards compatibility)."""
        scenario = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)
        results = run_full_simulation(scenario, use_multistream=False)

        assert results["arrivals"] > 0
        assert results["departures"] > 0

    def test_multistream_reproducibility(self):
        """Same seed produces same results in multistream mode."""
        scenario1 = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)
        scenario2 = FullScenario(run_length=120.0, warm_up=0.0, random_seed=42)

        results1 = run_full_simulation(scenario1, use_multistream=True)
        results2 = run_full_simulation(scenario2, use_multistream=True)

        assert results1["arrivals"] == results2["arrivals"]
        assert results1["departures"] == results2["departures"]


class TestPatientModes:
    """Test patient arrival modes are tracked."""

    def test_patients_have_modes(self):
        """Patients have arrival mode attribute."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(
            id=1,
            arrival_time=0.0,
            acuity=Acuity.MAJORS,
            mode=ArrivalMode.HELICOPTER,
        )
        assert patient.mode == ArrivalMode.HELICOPTER

    def test_default_mode_is_ambulance(self):
        """Default arrival mode is ambulance."""
        from faer.model.patient import Patient, Acuity

        patient = Patient(id=1, arrival_time=0.0, acuity=Acuity.MAJORS)
        assert patient.mode == ArrivalMode.AMBULANCE


class TestRNGStreams:
    """Test RNG streams for arrivals."""

    def test_per_mode_rng_streams(self):
        """Each arrival mode has its own RNG stream."""
        scenario = FullScenario(random_seed=42)

        assert ArrivalMode.AMBULANCE in scenario.rng_arrivals
        assert ArrivalMode.HELICOPTER in scenario.rng_arrivals
        assert ArrivalMode.SELF_PRESENTATION in scenario.rng_arrivals

    def test_rng_streams_are_different(self):
        """RNG streams for different modes are independent."""
        scenario = FullScenario(random_seed=42)

        # Sample from each - they should produce different sequences
        amb_vals = [scenario.rng_arrivals[ArrivalMode.AMBULANCE].random()
                    for _ in range(10)]
        heli_vals = [scenario.rng_arrivals[ArrivalMode.HELICOPTER].random()
                     for _ in range(10)]

        assert amb_vals != heli_vals
