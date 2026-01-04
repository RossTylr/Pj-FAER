"""Tests for ResultsCollector."""

import pytest

from faer.results.collector import ResultsCollector


class TestResultsCollector:
    """Test ResultsCollector dataclass."""

    def test_create_collector(self):
        """Can create empty collector."""
        collector = ResultsCollector()

        assert collector.arrivals == 0
        assert collector.departures == 0
        assert len(collector.queue_times) == 0
        assert len(collector.system_times) == 0

    def test_record_arrival(self):
        """Record arrival increments count."""
        collector = ResultsCollector()

        collector.record_arrival()
        collector.record_arrival()

        assert collector.arrivals == 2

    def test_record_queue_time(self):
        """Record queue time appends to list."""
        collector = ResultsCollector()

        collector.record_queue_time(5.0)
        collector.record_queue_time(10.0)

        assert collector.queue_times == [5.0, 10.0]

    def test_record_system_time(self):
        """Record system time increments departures."""
        collector = ResultsCollector()

        collector.record_system_time(30.0)
        collector.record_system_time(45.0)

        assert collector.departures == 2
        assert collector.system_times == [30.0, 45.0]

    def test_record_resource_state(self):
        """Record resource state appends to log."""
        collector = ResultsCollector()

        collector.record_resource_state(10.0, 1)
        collector.record_resource_state(20.0, 2)

        # Initial state + 2 recorded
        assert len(collector.resource_log) == 3


class TestMetricsComputation:
    """Test metrics computation."""

    def test_compute_p_delay(self):
        """P(delay) is proportion with queue_time > 0."""
        collector = ResultsCollector()

        collector.record_queue_time(0.0)
        collector.record_queue_time(5.0)
        collector.record_queue_time(0.0)
        collector.record_queue_time(10.0)

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["p_delay"] == 0.5  # 2 of 4 delayed

    def test_compute_queue_time_stats(self):
        """Queue time statistics computed correctly."""
        collector = ResultsCollector()

        for qt in [0, 5, 10, 15, 20]:
            collector.record_queue_time(float(qt))

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["mean_queue_time"] == 10.0
        assert metrics["median_queue_time"] == 10.0

    def test_compute_throughput(self):
        """Throughput per hour computed correctly."""
        collector = ResultsCollector()

        for _ in range(12):
            collector.record_system_time(30.0)

        # 12 departures in 120 minutes = 6 per hour
        metrics = collector.compute_metrics(run_length=120, capacity=2)

        assert metrics["throughput_per_hour"] == 6.0

    def test_empty_collector_metrics(self):
        """Empty collector returns zero metrics."""
        collector = ResultsCollector()

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["arrivals"] == 0
        assert metrics["departures"] == 0
        assert metrics["p_delay"] == 0.0
        assert metrics["mean_queue_time"] == 0.0


class TestUtilisationComputation:
    """Test utilisation calculation."""

    def test_utilisation_simple(self):
        """Simple utilisation calculation."""
        collector = ResultsCollector()

        # Resource busy from t=0 to t=50 (1 resource)
        collector.resource_log = [(0.0, 1), (50.0, 0)]

        metrics = collector.compute_metrics(run_length=100, capacity=1)

        assert metrics["utilisation"] == 0.5

    def test_utilisation_full(self):
        """Full utilisation (always busy)."""
        collector = ResultsCollector()

        collector.resource_log = [(0.0, 2)]

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["utilisation"] == 1.0

    def test_utilisation_empty(self):
        """Zero utilisation (never busy)."""
        collector = ResultsCollector()

        collector.resource_log = [(0.0, 0)]

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["utilisation"] == 0.0

    def test_utilisation_partial_capacity(self):
        """Partial capacity utilisation."""
        collector = ResultsCollector()

        # 1 of 2 bays busy for entire run
        collector.resource_log = [(0.0, 1)]

        metrics = collector.compute_metrics(run_length=100, capacity=2)

        assert metrics["utilisation"] == 0.5
