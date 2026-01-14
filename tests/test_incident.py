"""Tests for Major Incident configuration and arrival generation (Phase 11)."""

import numpy as np
import pytest

from faer.core.incident import (
    MajorIncidentConfig,
    CasualtyProfile,
    IncidentArrivalPattern,
    CASUALTY_PROFILES,
    get_profile_info,
    list_profiles,
)
from faer.core.entities import Priority


class TestMajorIncidentConfig:
    """Test MajorIncidentConfig dataclass."""

    def test_create_default_config(self):
        """Can create default config with sensible defaults."""
        config = MajorIncidentConfig()

        assert config.enabled is False
        assert config.trigger_time == 120.0
        assert config.duration == 120.0
        assert config.overload_percentage == 50.0
        assert config.arrival_pattern == IncidentArrivalPattern.BOLUS
        assert config.casualty_profile == CasualtyProfile.GENERIC

    def test_create_enabled_config(self):
        """Can create enabled config with custom parameters."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=60.0,
            duration=90.0,
            overload_percentage=100.0,
            arrival_pattern=IncidentArrivalPattern.WAVES,
            casualty_profile=CasualtyProfile.RTA,
            wave_count=5,
        )

        assert config.enabled is True
        assert config.trigger_time == 60.0
        assert config.duration == 90.0
        assert config.overload_percentage == 100.0
        assert config.arrival_pattern == IncidentArrivalPattern.WAVES
        assert config.casualty_profile == CasualtyProfile.RTA
        assert config.wave_count == 5

    def test_validation_negative_trigger_time(self):
        """Rejects negative trigger time."""
        with pytest.raises(ValueError, match="trigger_time"):
            MajorIncidentConfig(trigger_time=-10.0)

    def test_validation_zero_duration(self):
        """Rejects zero or negative duration."""
        with pytest.raises(ValueError, match="duration"):
            MajorIncidentConfig(duration=0.0)

        with pytest.raises(ValueError, match="duration"):
            MajorIncidentConfig(duration=-30.0)

    def test_validation_zero_overload(self):
        """Rejects zero or negative overload percentage."""
        with pytest.raises(ValueError, match="overload_percentage"):
            MajorIncidentConfig(overload_percentage=0.0)

        with pytest.raises(ValueError, match="overload_percentage"):
            MajorIncidentConfig(overload_percentage=-50.0)

    def test_validation_invalid_wave_count(self):
        """Rejects wave count less than 1."""
        with pytest.raises(ValueError, match="wave_count"):
            MajorIncidentConfig(wave_count=0)

    def test_validation_invalid_decon_range(self):
        """Rejects invalid decontamination time range."""
        with pytest.raises(ValueError, match="decon_time_range"):
            MajorIncidentConfig(decon_time_range=(-5.0, 30.0))

        with pytest.raises(ValueError, match="decon_time_range"):
            MajorIncidentConfig(decon_time_range=(30.0, 15.0))  # min > max


class TestCasualtyCalculations:
    """Test casualty count calculations."""

    def test_calculate_additional_casualties_basic(self):
        """Basic casualty count calculation."""
        config = MajorIncidentConfig(
            duration=120.0,  # 2 hours
            overload_percentage=50.0,  # 50% extra
        )

        # 20/hr baseline * 2hr window = 40 normal arrivals
        # 40 * 0.5 = 20 additional casualties
        additional = config.calculate_additional_casualties(baseline_hourly_rate=20.0)
        assert additional == 20

    def test_calculate_additional_casualties_100_percent(self):
        """100% overload doubles the normal arrivals."""
        config = MajorIncidentConfig(
            duration=60.0,  # 1 hour
            overload_percentage=100.0,
        )

        # 10/hr * 1hr = 10 normal arrivals
        # 10 * 1.0 = 10 additional
        additional = config.calculate_additional_casualties(baseline_hourly_rate=10.0)
        assert additional == 10

    def test_calculate_additional_casualties_minimum_one(self):
        """Always generates at least 1 casualty."""
        config = MajorIncidentConfig(
            duration=10.0,  # 10 minutes
            overload_percentage=1.0,  # 1% (very low)
        )

        # 2/hr * (10/60)hr = 0.33 arrivals, 0.33 * 0.01 = 0.003
        # Rounded to 0, but minimum is 1
        additional = config.calculate_additional_casualties(baseline_hourly_rate=2.0)
        assert additional >= 1

    def test_calculate_additional_casualties_rounding(self):
        """Casualties are rounded to nearest integer."""
        config = MajorIncidentConfig(
            duration=60.0,
            overload_percentage=75.0,
        )

        # 10/hr * 1hr = 10 normal arrivals
        # 10 * 0.75 = 7.5 → rounds to 8
        additional = config.calculate_additional_casualties(baseline_hourly_rate=10.0)
        assert additional == 8


class TestArrivalTimeGeneration:
    """Test arrival time generation for different patterns."""

    def test_generate_bolus_arrivals_sorted(self):
        """Bolus arrivals are returned sorted."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=100.0,
            duration=60.0,
            overload_percentage=50.0,
            arrival_pattern=IncidentArrivalPattern.BOLUS,
        )

        rng = np.random.default_rng(42)
        times = config.generate_arrival_times(rng, baseline_hourly_rate=20.0)

        assert times == sorted(times)

    def test_generate_bolus_arrivals_within_window(self):
        """Bolus arrivals are within the incident window."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=100.0,
            duration=60.0,
            overload_percentage=50.0,
            arrival_pattern=IncidentArrivalPattern.BOLUS,
        )

        rng = np.random.default_rng(42)
        times = config.generate_arrival_times(rng, baseline_hourly_rate=20.0)

        # All times should be >= trigger_time and <= trigger_time + duration
        for t in times:
            assert t >= config.trigger_time
            assert t <= config.trigger_time + config.duration

    def test_generate_bolus_front_loaded(self):
        """Bolus pattern is front-loaded (more arrivals early)."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=0.0,
            duration=100.0,
            overload_percentage=500.0,  # Many casualties for statistical test
            arrival_pattern=IncidentArrivalPattern.BOLUS,
        )

        rng = np.random.default_rng(42)
        times = config.generate_arrival_times(rng, baseline_hourly_rate=20.0)

        # Count arrivals in first half vs second half
        midpoint = config.trigger_time + config.duration / 2
        first_half = sum(1 for t in times if t < midpoint)
        second_half = sum(1 for t in times if t >= midpoint)

        # Bolus should have significantly more in first half
        assert first_half > second_half

    def test_generate_waves_arrivals_count(self):
        """Waves pattern generates expected number of arrivals."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=60.0,
            duration=120.0,
            overload_percentage=50.0,
            arrival_pattern=IncidentArrivalPattern.WAVES,
            wave_count=3,
        )

        rng = np.random.default_rng(42)
        times = config.generate_arrival_times(rng, baseline_hourly_rate=20.0)

        expected = config.calculate_additional_casualties(20.0)
        assert len(times) == expected

    def test_generate_sustained_arrivals_even_distribution(self):
        """Sustained pattern has relatively even distribution."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=0.0,
            duration=100.0,
            overload_percentage=500.0,  # Many for statistical test
            arrival_pattern=IncidentArrivalPattern.SUSTAINED,
        )

        rng = np.random.default_rng(42)
        times = config.generate_arrival_times(rng, baseline_hourly_rate=20.0)

        # Split into quarters and check distribution is roughly even
        quartiles = [25.0, 50.0, 75.0, 100.0]
        counts = []
        prev = 0.0
        for q in quartiles:
            count = sum(1 for t in times if prev <= t < q)
            counts.append(count)
            prev = q

        # No quartile should be more than 2x any other (rough uniformity check)
        max_count = max(counts)
        min_count = min(counts) if min(counts) > 0 else 1
        assert max_count / min_count < 2.5

    def test_generate_arrivals_reproducible(self):
        """Same seed produces same arrival times."""
        config = MajorIncidentConfig(
            enabled=True,
            trigger_time=120.0,
            duration=60.0,
            overload_percentage=50.0,
        )

        rng1 = np.random.default_rng(42)
        times1 = config.generate_arrival_times(rng1, baseline_hourly_rate=10.0)

        rng2 = np.random.default_rng(42)
        times2 = config.generate_arrival_times(rng2, baseline_hourly_rate=10.0)

        assert times1 == times2


class TestPrioritySampling:
    """Test casualty priority sampling from profiles."""

    def test_sample_priority_returns_valid_priority(self):
        """Sampled priority is a valid Priority enum."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.GENERIC)
        rng = np.random.default_rng(42)

        for _ in range(100):
            priority = config.sample_priority(rng)
            assert isinstance(priority, Priority)

    def test_sample_priority_distribution_generic(self):
        """Generic profile produces expected priority distribution."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.GENERIC)
        rng = np.random.default_rng(42)

        # Sample many times
        samples = [config.sample_priority(rng) for _ in range(1000)]

        # Count each priority
        counts = {p: samples.count(p) for p in Priority}

        # Check proportions are roughly as expected (within tolerance)
        total = sum(counts.values())
        p1_prop = counts[Priority.P1_IMMEDIATE] / total
        p2_prop = counts[Priority.P2_VERY_URGENT] / total
        p3_prop = counts[Priority.P3_URGENT] / total
        p4_prop = counts[Priority.P4_STANDARD] / total

        # Allow 5% tolerance
        assert abs(p1_prop - 0.20) < 0.05
        assert abs(p2_prop - 0.35) < 0.05
        assert abs(p3_prop - 0.35) < 0.05
        assert abs(p4_prop - 0.10) < 0.05

    def test_sample_priority_combat_more_p1(self):
        """Combat profile produces more P1 casualties than generic."""
        generic_config = MajorIncidentConfig(casualty_profile=CasualtyProfile.GENERIC)
        combat_config = MajorIncidentConfig(casualty_profile=CasualtyProfile.COMBAT)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        generic_samples = [generic_config.sample_priority(rng1) for _ in range(1000)]
        combat_samples = [combat_config.sample_priority(rng2) for _ in range(1000)]

        generic_p1 = generic_samples.count(Priority.P1_IMMEDIATE)
        combat_p1 = combat_samples.count(Priority.P1_IMMEDIATE)

        # Combat should have significantly more P1s (35% vs 20%)
        assert combat_p1 > generic_p1


class TestDecontamination:
    """Test CBRN decontamination logic."""

    def test_requires_decon_cbrn(self):
        """CBRN profile requires decontamination."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.CBRN)
        assert config.requires_decon is True

    def test_requires_decon_non_cbrn(self):
        """Non-CBRN profiles do not require decontamination."""
        for profile in [CasualtyProfile.GENERIC, CasualtyProfile.RTA,
                       CasualtyProfile.BLAST, CasualtyProfile.BURNS, CasualtyProfile.COMBAT]:
            config = MajorIncidentConfig(casualty_profile=profile)
            assert config.requires_decon is False

    def test_sample_decon_time_cbrn(self):
        """CBRN profile samples decon time within range."""
        config = MajorIncidentConfig(
            casualty_profile=CasualtyProfile.CBRN,
            decon_time_range=(15.0, 45.0),
        )
        rng = np.random.default_rng(42)

        for _ in range(100):
            decon_time = config.sample_decon_time(rng)
            assert decon_time >= 15.0
            assert decon_time <= 45.0

    def test_sample_decon_time_non_cbrn_zero(self):
        """Non-CBRN profiles return 0 decon time."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.RTA)
        rng = np.random.default_rng(42)

        decon_time = config.sample_decon_time(rng)
        assert decon_time == 0.0


class TestCasualtyProfiles:
    """Test casualty profile definitions."""

    def test_all_profiles_have_required_keys(self):
        """All profiles have required keys."""
        required_keys = ["priority_mix", "requires_decon", "description", "typical_scenarios"]

        for profile, data in CASUALTY_PROFILES.items():
            for key in required_keys:
                assert key in data, f"Profile {profile.name} missing key {key}"

    def test_all_profiles_priority_mix_sums_to_one(self):
        """All profile priority mixes sum to 1.0."""
        for profile, data in CASUALTY_PROFILES.items():
            total = sum(data["priority_mix"].values())
            assert abs(total - 1.0) < 0.001, f"Profile {profile.name} priority_mix sums to {total}"

    def test_all_profiles_have_all_priorities(self):
        """All profiles specify all priority levels."""
        for profile, data in CASUALTY_PROFILES.items():
            for priority in Priority:
                assert priority in data["priority_mix"], \
                    f"Profile {profile.name} missing {priority.name}"

    def test_only_cbrn_requires_decon(self):
        """Only CBRN profile requires decontamination."""
        for profile, data in CASUALTY_PROFILES.items():
            if profile == CasualtyProfile.CBRN:
                assert data["requires_decon"] is True
            else:
                assert data["requires_decon"] is False


class TestProfileHelpers:
    """Test helper functions for profiles."""

    def test_get_profile_info(self):
        """get_profile_info returns correct data."""
        info = get_profile_info(CasualtyProfile.RTA)

        assert "priority_mix" in info
        assert info["requires_decon"] is False
        assert "traffic" in info["description"].lower() or "collision" in info["description"].lower()

    def test_list_profiles_returns_all(self):
        """list_profiles returns info for all profiles."""
        profiles = list_profiles()

        assert len(profiles) == len(CasualtyProfile)

        for profile_info in profiles:
            assert "name" in profile_info
            assert "enum" in profile_info
            assert "description" in profile_info
            assert "requires_decon" in profile_info
            assert "typical_scenarios" in profile_info

    def test_profile_description_property(self):
        """profile_description property returns correct string."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.BLAST)
        assert "blast" in config.profile_description.lower() or "explosion" in config.profile_description.lower()

    def test_get_priority_breakdown(self):
        """get_priority_breakdown returns percentages."""
        config = MajorIncidentConfig(casualty_profile=CasualtyProfile.GENERIC)
        breakdown = config.get_priority_breakdown()

        # Should have all priorities
        assert len(breakdown) == len(Priority)

        # Values should be percentages (0-100)
        for name, value in breakdown.items():
            assert 0 <= value <= 100

        # Should sum to 100
        assert abs(sum(breakdown.values()) - 100.0) < 0.1
