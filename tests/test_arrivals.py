"""Tests for arrival profile and NSPP thinning."""

import numpy as np
import pytest

from faer.core.arrivals import (
    ArrivalProfile,
    NSPPThinning,
    load_default_profile,
    create_constant_profile,
)


class TestArrivalProfile:
    """Test ArrivalProfile dataclass."""

    def test_create_profile(self):
        """Can create a simple arrival profile."""
        profile = ArrivalProfile([(60, 4.0), (120, 6.0)])
        assert profile.max_rate == 6.0

    def test_get_rate(self):
        """Get rate returns correct value for time."""
        profile = ArrivalProfile([(60, 4.0), (120, 6.0), (180, 2.0)])

        assert profile.get_rate(30) == 4.0   # First period
        assert profile.get_rate(90) == 6.0   # Second period
        assert profile.get_rate(150) == 2.0  # Third period

    def test_cyclic_time(self):
        """Profile wraps around at end of period."""
        profile = ArrivalProfile([(60, 4.0), (120, 6.0)])

        # At t=150, wraps to t=30 (mod 120)
        assert profile.get_rate(150) == 4.0

    def test_validation_sorted(self):
        """Rejects unsorted schedule."""
        with pytest.raises(ValueError, match="sorted"):
            ArrivalProfile([(120, 4.0), (60, 6.0)])

    def test_validation_negative_time(self):
        """Rejects negative times."""
        with pytest.raises(ValueError, match="non-negative"):
            ArrivalProfile([(-60, 4.0), (60, 6.0)])

    def test_validation_negative_rate(self):
        """Rejects negative rates."""
        with pytest.raises(ValueError, match="non-negative"):
            ArrivalProfile([(60, -4.0)])


class TestDefaultProfile:
    """Test default profile loading."""

    def test_load_default_profile(self):
        """Default profile loads with 24 periods."""
        profile = load_default_profile()

        assert len(profile.schedule) == 24
        assert profile.schedule[-1][0] == 1440  # 24 hours in minutes
        assert profile.max_rate == 7.5  # Peak rate

    def test_default_profile_rates(self):
        """Default profile has expected rate pattern."""
        profile = load_default_profile()

        # Overnight should be low
        assert profile.get_rate(120) <= 2.0  # 2am

        # Peak should be high
        assert profile.get_rate(660) >= 7.0  # 11am


class TestConstantProfile:
    """Test constant profile creation."""

    def test_create_constant(self):
        """Can create constant rate profile."""
        profile = create_constant_profile(4.0)

        assert profile.max_rate == 4.0
        assert profile.get_rate(0) == 4.0
        assert profile.get_rate(720) == 4.0

    def test_constant_custom_duration(self):
        """Constant profile respects custom duration."""
        profile = create_constant_profile(4.0, duration=480.0)

        assert profile.schedule[-1][0] == 480.0


class TestNSPPThinning:
    """Test NSPP thinning algorithm."""

    def test_sample_iat_positive(self):
        """Sampled IAT is positive."""
        profile = load_default_profile()
        rng = np.random.default_rng(42)
        thinning = NSPPThinning(profile, rng)

        iat = thinning.sample_iat(0.0)
        assert iat > 0

    def test_reproducibility(self):
        """Same seed produces same IAT sequence."""
        profile = load_default_profile()

        rng1 = np.random.default_rng(42)
        thinning1 = NSPPThinning(profile, rng1)

        rng2 = np.random.default_rng(42)
        thinning2 = NSPPThinning(profile, rng2)

        iats1 = [thinning1.sample_iat(i * 10) for i in range(10)]
        iats2 = [thinning2.sample_iat(i * 10) for i in range(10)]

        assert iats1 == iats2

    def test_constant_rate_matches_exponential(self):
        """Constant rate should approximate exponential IAT."""
        profile = create_constant_profile(6.0)  # 6 per hour
        rng = np.random.default_rng(42)
        thinning = NSPPThinning(profile, rng)

        # Sample many IATs
        iats = []
        t = 0.0
        for _ in range(1000):
            iat = thinning.sample_iat(t)
            iats.append(iat)
            t += iat

        mean_iat = np.mean(iats)
        expected_mean = 60.0 / 6.0  # 10 minutes

        # Should be close to expected (within 15%)
        assert abs(mean_iat - expected_mean) / expected_mean < 0.15

    def test_hourly_counts_match_rates(self):
        """Arrival counts per hour should match expected rates."""
        profile = load_default_profile()
        rng = np.random.default_rng(42)
        thinning = NSPPThinning(profile, rng)

        # Generate arrivals for 24 hours
        arrivals_per_hour = [0] * 24
        t = 0.0
        while t < 1440:  # 24 hours
            iat = thinning.sample_iat(t)
            t += iat
            if t < 1440:
                hour = int(t // 60)
                arrivals_per_hour[hour] += 1

        # Check peak hours have more arrivals than off-peak
        # Morning peak (10-11am, hour 10) vs overnight (2-3am, hour 2)
        assert arrivals_per_hour[10] > arrivals_per_hour[2]


class TestThinningEfficiency:
    """Test thinning algorithm efficiency."""

    def test_efficiency_reasonable(self):
        """Thinning efficiency should be reasonable for ED profile."""
        profile = load_default_profile()

        # Expected efficiency = average_rate / max_rate
        rates = [rate for _, rate in profile.schedule]
        avg_rate = np.mean(rates)
        expected_efficiency = avg_rate / profile.max_rate

        # Should be at least 50% efficient for typical ED profile
        assert expected_efficiency > 0.5
