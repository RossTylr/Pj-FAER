"""Single and batch simulation runners."""

from typing import Dict, List, Optional, Callable, Any, Union

from faer.core.scenario import Scenario, FullScenario
from faer.model.processes import run_simulation
from faer.model.full_model import run_full_simulation


def multiple_replications(
    scenario: Union[Scenario, FullScenario],
    n_reps: int = 30,
    metric_names: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_multistream: bool = False,
) -> Dict[str, List[float]]:
    """Run multiple replications and collect specified metrics.

    Each replication uses a different random seed (base_seed + rep_number)
    to ensure independent samples.

    Args:
        scenario: Base scenario configuration (Scenario or FullScenario).
        n_reps: Number of replications to run.
        metric_names: List of metric names to collect. If None, collects
            defaults based on scenario type.
        progress_callback: Optional callback(current_rep, total_reps) for
            progress reporting.
        use_multistream: If True, use multi-stream arrivals for FullScenario.

    Returns:
        Dictionary mapping metric names to lists of values across replications.
    """
    is_full_model = isinstance(scenario, FullScenario)

    if metric_names is None:
        if is_full_model:
            metric_names = [
                "arrivals", "departures",
                "arrivals_resus", "arrivals_majors", "arrivals_minors",
                "p_delay", "mean_triage_wait", "mean_treatment_wait",
                "mean_system_time", "p95_system_time",
                "admission_rate", "admitted", "discharged",
                # Resource utilisation
                "util_triage", "util_ed_bays", "util_handover",
                "util_ambulance_fleet", "util_helicopter_fleet",
                # Downstream utilisation (Phase 8)
                "util_theatre", "util_itu", "util_ward",
                # Diagnostic utilisation
                "util_CT_SCAN", "util_XRAY", "util_BLOODS",
                # Aeromed metrics
                "aeromed_hems_count", "aeromed_fixedwing_count",
                "aeromed_total", "aeromed_slots_missed",
                "mean_aeromed_slot_wait", "max_aeromed_slot_wait",
                "ward_bed_days_blocked_aeromed",
                # P1-P4 Priority Statistics (Phase 4 enhancement)
                "arrivals_P1", "arrivals_P2", "arrivals_P3", "arrivals_P4",
                "departures_P1", "departures_P2", "departures_P3", "departures_P4",
                "P1_mean_wait", "P2_mean_wait", "P3_mean_wait", "P4_mean_wait",
                "P1_p95_wait", "P2_p95_wait", "P3_p95_wait", "P4_p95_wait",
                "P1_max_wait", "P2_max_wait", "P3_max_wait", "P4_max_wait",
                "P1_breach_rate", "P2_breach_rate", "P3_breach_rate", "P4_breach_rate",
                "P1_mean_system_time", "P2_mean_system_time", "P3_mean_system_time", "P4_mean_system_time",
                # Acuity Statistics (Phase 4 enhancement)
                "departures_resus", "departures_majors", "departures_minors",
                "resus_mean_wait", "majors_mean_wait", "minors_mean_wait",
                "resus_p95_wait", "majors_p95_wait", "minors_p95_wait",
                "resus_max_wait", "majors_max_wait", "minors_max_wait",
                "resus_mean_system_time", "majors_mean_system_time", "minors_mean_system_time",
                # Disposition breakdown (Phase 4 enhancement)
                "discharged_count", "admitted_ward_count", "admitted_icu_count",
                "transfer_count", "left_count",
                "mean_los_discharged", "mean_los_ward", "mean_los_icu",
            ]
        else:
            metric_names = ["p_delay", "mean_queue_time", "utilisation"]

    results: Dict[str, List[float]] = {name: [] for name in metric_names}

    for rep in range(n_reps):
        # Create scenario with different seed for this rep
        rep_scenario = scenario.clone_with_seed(scenario.random_seed + rep)

        # Run appropriate simulation
        if is_full_model:
            run_results = run_full_simulation(rep_scenario, use_multistream=use_multistream)
        else:
            run_results = run_simulation(rep_scenario)

        # Collect specified metrics
        for name in metric_names:
            if name in run_results:
                results[name].append(run_results[name])

        # Report progress if callback provided
        if progress_callback is not None:
            progress_callback(rep + 1, n_reps)

    return results


def run_scenario_comparison(
    scenarios: Dict[str, Scenario],
    n_reps: int = 30,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Run multiple scenarios for comparison.

    Args:
        scenarios: Dictionary mapping scenario names to Scenario objects.
        n_reps: Number of replications per scenario.
        metric_names: Metrics to collect.

    Returns:
        Nested dictionary: {scenario_name: {metric_name: [values]}}.
    """
    all_results = {}

    for name, scenario in scenarios.items():
        all_results[name] = multiple_replications(
            scenario, n_reps=n_reps, metric_names=metric_names
        )

    return all_results
