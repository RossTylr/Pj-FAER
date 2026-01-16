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

    # Phase 12: Track scaling metrics across replications
    scaling_metrics_list: List[Dict[str, Any]] = []

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

        # Phase 12: Collect scaling metrics if present
        if 'scaling_metrics' in run_results:
            scaling_metrics_list.append(run_results['scaling_metrics'])

        # Report progress if callback provided
        if progress_callback is not None:
            progress_callback(rep + 1, n_reps)

    # Phase 12: Aggregate scaling metrics across replications
    if scaling_metrics_list:
        results['scaling_metrics'] = _aggregate_scaling_metrics(scaling_metrics_list)

    return results


def _aggregate_scaling_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scaling metrics across multiple replications.

    For numeric metrics, computes mean across replications.
    For events, concatenates all events with replication index.

    Args:
        metrics_list: List of scaling metrics dictionaries from each replication.

    Returns:
        Aggregated scaling metrics dictionary.
    """
    if not metrics_list:
        return {}

    n_reps = len(metrics_list)

    # Aggregate numeric metrics by taking mean
    numeric_keys = [
        'total_scale_up_events', 'total_scale_down_events', 'total_scaling_events',
        'pct_time_at_surge', 'total_additional_bed_hours', 'opel_transitions',
        'patients_diverted'
    ]

    aggregated = {}

    for key in numeric_keys:
        values = [m.get(key, 0) for m in metrics_list]
        aggregated[key] = sum(values) / n_reps if values else 0

    # Peak OPEL level - take maximum across replications
    peak_levels = [m.get('opel_peak_level', 1) for m in metrics_list]
    aggregated['opel_peak_level'] = max(peak_levels) if peak_levels else 1

    # OPEL time at level - average across replications
    opel_times = {}
    for level in [1, 2, 3, 4]:
        times = [m.get('opel_time_at_level', {}).get(level, 0) for m in metrics_list]
        opel_times[level] = sum(times) / n_reps if times else 0
    aggregated['opel_time_at_level'] = opel_times

    # Rule activations - sum across replications then divide by n_reps
    all_rules = set()
    for m in metrics_list:
        all_rules.update(m.get('rule_activations', {}).keys())

    rule_activations = {}
    for rule in all_rules:
        counts = [m.get('rule_activations', {}).get(rule, 0) for m in metrics_list]
        rule_activations[rule] = sum(counts) / n_reps if counts else 0
    aggregated['rule_activations'] = rule_activations

    # Events - take from first replication only (representative sample)
    # Including all events from all replications would be too verbose
    if metrics_list and 'events' in metrics_list[0]:
        aggregated['events'] = metrics_list[0]['events']

    return aggregated


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
