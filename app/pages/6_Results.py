"""Results display page for full A&E model."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.experiment.analysis import compute_ci

st.set_page_config(page_title="Results - FAER", page_icon="", layout="wide")

st.title("Results")

# Check for results
if not st.session_state.get("run_complete"):
    st.warning("Please run a simulation first.")
    st.page_link("pages/5_Run.py", label="Go to Run Simulation")
    st.stop()

# Support both old key (results) and new key (run_results)
results = st.session_state.get('run_results') or st.session_state.get('results')
scenario = st.session_state.get('run_scenario') or st.session_state.get('scenario')

# Check for stale results (from before Phase 5 update)
if "util_ed_bays" not in results:
    st.warning("Results are from an older version. Please re-run the simulation to see updated metrics.")
    st.session_state.run_complete = False
    st.page_link("pages/5_Run.py", label="Go to Run Simulation")
    st.stop()

n_reps = len(results["arrivals"])

# ===== KEY PERFORMANCE INDICATORS =====
st.header("Key Performance Indicators")
st.caption(f"Based on {n_reps} replications | 95% Confidence Intervals")

kpi_cols = st.columns(4)

# P(Delay)
with kpi_cols[0]:
    ci = compute_ci(results["p_delay"])
    st.metric(
        "P(Delay)",
        f"{ci['mean']:.1%}",
        help="Probability a patient waits for treatment",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# Mean Treatment Wait
with kpi_cols[1]:
    ci = compute_ci(results["mean_treatment_wait"])
    st.metric(
        "Mean Treatment Wait",
        f"{ci['mean']:.1f} min",
        help="Average time waiting for treatment bay",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

# Mean System Time
with kpi_cols[2]:
    ci = compute_ci(results["mean_system_time"])
    st.metric(
        "Mean System Time",
        f"{ci['mean']:.1f} min",
        help="Average time from arrival to departure",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

# Admission Rate
with kpi_cols[3]:
    ci = compute_ci(results["admission_rate"])
    st.metric(
        "Admission Rate",
        f"{ci['mean']:.1%}",
        help="Proportion of patients admitted",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

st.divider()

# ===== RESOURCE UTILISATION - EXPANDED =====
st.header("Resource Utilisation")

# Helper to safely get utilisation with fallback
def get_util(key: str, default: float = 0.0):
    """Get utilisation from results with fallback."""
    if key in results and results[key]:
        return results[key]
    return [default] * n_reps

# --- Emergency Services ---
st.subheader("Emergency Services")
es_cols = st.columns(3)

with es_cols[0]:
    ci = compute_ci(get_util("util_ambulance_fleet"))
    st.metric("Ambulance Fleet", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with es_cols[1]:
    ci = compute_ci(get_util("util_helicopter_fleet"))
    st.metric("HEMS Fleet", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with es_cols[2]:
    ci = compute_ci(get_util("util_handover"))
    st.metric("Handover Bays", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# --- Triage & ED ---
st.subheader("Triage & ED")
ed_cols = st.columns(2)

with ed_cols[0]:
    ci = compute_ci(get_util("util_triage"))
    st.metric("Triage", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with ed_cols[1]:
    ci = compute_ci(get_util("util_ed_bays"))
    st.metric("ED Bays", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# --- Diagnostics ---
st.subheader("Diagnostics")
diag_cols = st.columns(3)

with diag_cols[0]:
    ci = compute_ci(get_util("util_CT_SCAN"))
    st.metric("CT Scanner", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with diag_cols[1]:
    ci = compute_ci(get_util("util_XRAY"))
    st.metric("X-ray", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with diag_cols[2]:
    ci = compute_ci(get_util("util_BLOODS"))
    st.metric("Bloods", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# --- Downstream: Theatre, ITU, Ward ---
st.subheader("Downstream (Theatre, ITU, Ward)")
ds_cols = st.columns(3)

with ds_cols[0]:
    ci = compute_ci(get_util("util_theatre"))
    st.metric("Theatre", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with ds_cols[1]:
    ci = compute_ci(get_util("util_itu"))
    st.metric("ITU", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with ds_cols[2]:
    ci = compute_ci(get_util("util_ward"))
    st.metric("Ward", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# --- Aeromed (if enabled and toggle is checked) ---
aeromed_total = results.get("aeromed_total", [0] * n_reps)
has_aeromed_data = sum(aeromed_total) > 0

# Check if aeromed was enabled in scenario
aeromed_was_enabled = (
    hasattr(scenario, 'aeromed_config')
    and scenario.aeromed_config
    and scenario.aeromed_config.enabled
)

# Show toggle if aeromed was enabled (even if no evacuations occurred)
if aeromed_was_enabled or has_aeromed_data:
    show_aeromed = st.checkbox(
        "Show Aeromed Evacuation Metrics",
        value=has_aeromed_data,  # Default to checked if there's data
        help="Toggle to show/hide aeromed evacuation statistics"
    )
else:
    show_aeromed = False

if show_aeromed:
    st.subheader("Aeromed Evacuation")
    if not has_aeromed_data:
        st.info("Aeromed was enabled but no evacuations occurred during this simulation run.")
    else:
        aeromed_cols = st.columns(4)

        with aeromed_cols[0]:
            ci = compute_ci(aeromed_total)
            st.metric("Total Aeromed", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[1]:
            hems_count = results.get("aeromed_hems_count", [0] * n_reps)
            ci = compute_ci(hems_count)
            st.metric("HEMS Evacuations", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[2]:
            fw_count = results.get("aeromed_fixedwing_count", [0] * n_reps)
            ci = compute_ci(fw_count)
            st.metric("Fixed-Wing Evacuations", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[3]:
            slots_missed = results.get("aeromed_slots_missed", [0] * n_reps)
            ci = compute_ci(slots_missed)
            st.metric("Slots Missed", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        # Additional aeromed metrics row
        aeromed_cols2 = st.columns(3)

        with aeromed_cols2[0]:
            slot_wait = results.get("mean_aeromed_slot_wait", [0] * n_reps)
            ci = compute_ci(slot_wait)
            st.metric("Mean Slot Wait", f"{ci['mean']:.1f} min")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols2[1]:
            blocked_days = results.get("ward_bed_days_blocked_aeromed", [0] * n_reps)
            ci = compute_ci(blocked_days)
            st.metric("Ward Bed-Days Blocked", f"{ci['mean']:.2f}")
            st.caption(f"By aeromed patients waiting for slots")

        with aeromed_cols2[2]:
            # Show aeromed as % of total departures
            departures = results.get("departures", [1] * n_reps)
            aeromed_pct = [a / d * 100 if d > 0 else 0 for a, d in zip(aeromed_total, departures)]
            ci = compute_ci(aeromed_pct)
            st.metric("Aeromed % of Departures", f"{ci['mean']:.1f}%")
            st.caption(f"CI: [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")

# Utilisation bar chart - all resources
st.markdown("---")
st.subheader("Utilisation Overview")

util_data = pd.DataFrame({
    "Resource": [
        "Ambulance", "HEMS", "Handover",
        "Triage", "ED Bays",
        "CT", "X-ray", "Bloods",
        "Theatre", "ITU", "Ward"
    ],
    "Category": [
        "Emergency", "Emergency", "Emergency",
        "ED", "ED",
        "Diagnostics", "Diagnostics", "Diagnostics",
        "Downstream", "Downstream", "Downstream"
    ],
    "Utilisation": [
        np.mean(get_util("util_ambulance_fleet")),
        np.mean(get_util("util_helicopter_fleet")),
        np.mean(get_util("util_handover")),
        np.mean(get_util("util_triage")),
        np.mean(get_util("util_ed_bays")),
        np.mean(get_util("util_CT_SCAN")),
        np.mean(get_util("util_XRAY")),
        np.mean(get_util("util_BLOODS")),
        np.mean(get_util("util_theatre")),
        np.mean(get_util("util_itu")),
        np.mean(get_util("util_ward")),
    ]
})

fig = px.bar(
    util_data,
    x="Resource",
    y="Utilisation",
    title="Mean Resource Utilisation Across System",
    color="Category",
    color_discrete_map={
        "Emergency": "#ffa62b",
        "ED": "#ff4b4b",
        "Diagnostics": "#ab63fa",
        "Downstream": "#636efa"
    },
)
fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45)
fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Target (85%)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== ARRIVALS BY ACUITY =====
st.header("Patient Flow")

flow_cols = st.columns(2)

with flow_cols[0]:
    st.subheader("Arrivals by Acuity")
    arrivals_data = pd.DataFrame({
        "Acuity": ["Resus", "Majors", "Minors"],
        "Arrivals": [
            np.mean(results["arrivals_resus"]),
            np.mean(results["arrivals_majors"]),
            np.mean(results["arrivals_minors"]),
        ]
    })

    fig = px.pie(
        arrivals_data,
        values="Arrivals",
        names="Acuity",
        title="Mean Arrivals by Acuity",
        color="Acuity",
        color_discrete_map={"Resus": "#ff4b4b", "Majors": "#ffa62b", "Minors": "#29b09d"},
    )
    st.plotly_chart(fig, use_container_width=True)

with flow_cols[1]:
    st.subheader("Disposition")
    ci_admit = compute_ci(results["admitted"])
    ci_disch = compute_ci(results["discharged"])

    disp_data = pd.DataFrame({
        "Outcome": ["Admitted", "Discharged"],
        "Count": [ci_admit["mean"], ci_disch["mean"]],
    })

    fig = px.pie(
        disp_data,
        values="Count",
        names="Outcome",
        title="Mean Departures by Outcome",
        color="Outcome",
        color_discrete_map={"Admitted": "#636efa", "Discharged": "#00cc96"},
    )
    st.plotly_chart(fig, use_container_width=True)

# Throughput metrics
throughput_cols = st.columns(3)

with throughput_cols[0]:
    ci = compute_ci(results["arrivals"])
    st.metric("Total Arrivals", f"{ci['mean']:.0f}")
    st.caption(f"Range: [{min(results['arrivals']):.0f}, {max(results['arrivals']):.0f}]")

with throughput_cols[1]:
    ci = compute_ci(results["departures"])
    st.metric("Total Departures", f"{ci['mean']:.0f}")
    st.caption(f"Range: [{min(results['departures']):.0f}, {max(results['departures']):.0f}]")

with throughput_cols[2]:
    ci = compute_ci(results["p95_system_time"])
    st.metric("95th Percentile System Time", f"{ci['mean']:.1f} min")
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

st.divider()

# ===== DISTRIBUTION PLOTS =====
st.header("Metric Distributions Across Replications")

# Tabs for different metric groups
dist_tab1, dist_tab2, dist_tab3 = st.tabs(["Wait Times", "System Time", "Utilisation"])

with dist_tab1:
    wait_cols = st.columns(2)

    with wait_cols[0]:
        fig = px.histogram(
            results["mean_triage_wait"],
            nbins=20,
            labels={"value": "Triage Wait (min)", "count": "Frequency"},
            title="Distribution of Mean Triage Wait",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with wait_cols[1]:
        fig = px.histogram(
            results["mean_treatment_wait"],
            nbins=20,
            labels={"value": "Treatment Wait (min)", "count": "Frequency"},
            title="Distribution of Mean Treatment Wait",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with dist_tab2:
    sys_cols = st.columns(2)

    with sys_cols[0]:
        fig = px.histogram(
            results["mean_system_time"],
            nbins=20,
            labels={"value": "System Time (min)", "count": "Frequency"},
            title="Distribution of Mean System Time",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with sys_cols[1]:
        fig = px.histogram(
            results["p95_system_time"],
            nbins=20,
            labels={"value": "P95 System Time (min)", "count": "Frequency"},
            title="Distribution of 95th Percentile System Time",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with dist_tab3:
    util_hist_cols = st.columns(2)

    with util_hist_cols[0]:
        fig = px.histogram(
            results["util_ed_bays"],
            nbins=20,
            labels={"value": "Utilisation", "count": "Frequency"},
            title="Distribution of ED Bays Utilisation",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with util_hist_cols[1]:
        fig = px.histogram(
            results["util_handover"],
            nbins=20,
            labels={"value": "Utilisation", "count": "Frequency"},
            title="Distribution of Handover Utilisation",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== SCENARIO CONFIGURATION =====
st.header("Scenario Configuration")

config_cols = st.columns(4)

with config_cols[0]:
    st.write("**Timing**")
    st.write(f"Run: {scenario.run_length / 60:.0f}h")
    st.write(f"Warm-up: {scenario.warm_up / 60:.0f}h")

with config_cols[1]:
    st.write("**Resources**")
    st.write(f"Triage: {scenario.n_triage}")
    st.write(f"ED Bays: {scenario.n_ed_bays}")
    st.write(f"Handover: {scenario.n_handover_bays}")
    st.write(f"Ambulances: {scenario.n_ambulances}")
    st.write(f"Helicopters: {scenario.n_helicopters}")

with config_cols[2]:
    st.write("**Acuity Mix**")
    st.write(f"Resus: {scenario.p_resus:.0%}")
    st.write(f"Majors: {scenario.p_majors:.0%}")
    st.write(f"Minors: {scenario.p_minors:.0%}")

with config_cols[3]:
    st.write("**Experiment**")
    st.write(f"Replications: {n_reps}")
    st.write(f"Seed: {scenario.random_seed}")

# Second row: Diagnostics and Downstream
config_cols2 = st.columns(3)

with config_cols2[0]:
    st.write("**Diagnostics**")
    if hasattr(scenario, 'diagnostic_configs') and scenario.diagnostic_configs:
        from faer.core.entities import DiagnosticType
        ct_config = scenario.diagnostic_configs.get(DiagnosticType.CT_SCAN)
        xray_config = scenario.diagnostic_configs.get(DiagnosticType.XRAY)
        bloods_config = scenario.diagnostic_configs.get(DiagnosticType.BLOODS)
        st.write(f"CT Scanners: {ct_config.capacity if ct_config else 'N/A'}")
        st.write(f"X-ray Rooms: {xray_config.capacity if xray_config else 'N/A'}")
        st.write(f"Phlebotomy: {bloods_config.capacity if bloods_config else 'N/A'}")
    else:
        st.write("Not configured")

with config_cols2[1]:
    st.write("**Downstream**")
    if hasattr(scenario, 'theatre_config') and scenario.theatre_config:
        st.write(f"Theatre Tables: {scenario.theatre_config.n_tables}")
    else:
        st.write("Theatre: N/A")
    if hasattr(scenario, 'itu_config') and scenario.itu_config:
        st.write(f"ITU Beds: {scenario.itu_config.capacity}")
    else:
        st.write("ITU: N/A")
    if hasattr(scenario, 'ward_config') and scenario.ward_config:
        st.write(f"Ward Beds: {scenario.ward_config.capacity}")
    else:
        st.write("Ward: N/A")

with config_cols2[2]:
    st.write("**Downstream Status**")
    if hasattr(scenario, 'downstream_enabled'):
        st.write(f"Enabled: {'Yes' if scenario.downstream_enabled else 'No'}")
    else:
        st.write("Enabled: No")
    if hasattr(scenario, 'aeromed_config') and scenario.aeromed_config:
        st.write(f"Aeromed: {'Yes' if scenario.aeromed_config.enabled else 'No'}")
    else:
        st.write("Aeromed: No")

st.divider()

# ===== COST MODELLING SECTION =====
st.header("Cost Modelling")

with st.expander("Calculate Financial Costs", expanded=False):
    st.markdown("""
    Calculate financial costs from simulation results using configurable bed-day rates.
    This is a **post-hoc analysis** that does not affect simulation behaviour.

    **Note:** Cost calculation runs a single replication to access detailed patient data.
    """)

    # Import cost modelling module
    from faer.results.costs import (
        CostConfig, calculate_costs, calculate_costs_by_priority,
        format_currency, CURRENCY_SYMBOLS
    )
    from faer.model.full_model import run_full_simulation, FullResultsCollector

    # Cost configuration in columns
    st.subheader("Configuration")

    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:
        currency = st.selectbox(
            "Currency",
            options=["GBP", "USD", "EUR"],
            index=0,
            help="Select currency for cost display"
        )
        currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)

    with cost_col2:
        apply_priority_multipliers = st.checkbox(
            "Apply priority multipliers",
            value=True,
            help="Higher acuity patients (P1/P2) cost more due to resource intensity"
        )

    # Bed-day rates
    st.markdown("**Bed-Day Rates** (per 24 hours)")
    rate_cols = st.columns(4)

    with rate_cols[0]:
        ed_bay_rate = st.number_input(
            f"ED Bay ({currency_symbol}/day)",
            min_value=0.0, max_value=10000.0, value=500.0, step=50.0
        )
    with rate_cols[1]:
        itu_rate = st.number_input(
            f"ITU ({currency_symbol}/day)",
            min_value=0.0, max_value=20000.0, value=2000.0, step=100.0
        )
    with rate_cols[2]:
        ward_rate = st.number_input(
            f"Ward ({currency_symbol}/day)",
            min_value=0.0, max_value=5000.0, value=400.0, step=50.0
        )
    with rate_cols[3]:
        theatre_rate = st.number_input(
            f"Theatre ({currency_symbol}/hr)",
            min_value=0.0, max_value=10000.0, value=2000.0, step=100.0
        )

    # Per-episode and transport costs
    st.markdown("**Per-Episode & Transport Costs**")
    episode_cols = st.columns(4)

    with episode_cols[0]:
        triage_cost = st.number_input(
            f"Triage ({currency_symbol})",
            min_value=0.0, max_value=500.0, value=20.0, step=5.0
        )
    with episode_cols[1]:
        diagnostics_cost = st.number_input(
            f"Diagnostics avg ({currency_symbol})",
            min_value=0.0, max_value=1000.0, value=75.0, step=10.0
        )
    with episode_cols[2]:
        ambulance_cost = st.number_input(
            f"Ambulance ({currency_symbol})",
            min_value=0.0, max_value=2000.0, value=275.0, step=25.0
        )
    with episode_cols[3]:
        hems_cost = st.number_input(
            f"HEMS flight ({currency_symbol})",
            min_value=0.0, max_value=20000.0, value=3500.0, step=100.0
        )

    st.markdown("---")

    # Run button
    if st.button("Calculate Costs", type="primary", use_container_width=True):
        with st.spinner("Running cost calculation (single replication)..."):
            # Create cost config
            config = CostConfig(
                enabled=True,
                currency=currency,
                ed_bay_per_day=ed_bay_rate,
                itu_bed_per_day=itu_rate,
                ward_bed_per_day=ward_rate,
                theatre_per_hour=theatre_rate,
                triage_cost=triage_cost,
                diagnostics_base_cost=diagnostics_cost,
                ambulance_per_journey=ambulance_cost,
                hems_per_flight=hems_cost,
            )

            if not apply_priority_multipliers:
                config.apply_priority_to = []

            # Run a single replication to get patient-level data
            # Use the scenario from session state
            try:
                import simpy
                from faer.core.entities import DiagnosticType

                # Create fresh environment for cost calculation
                env = simpy.Environment()

                # Run simulation to get patient data
                # We need to access the FullResultsCollector directly
                from faer.model.full_model import (
                    AEResources, FullResultsCollector,
                    arrival_generator_multistream, arrival_generator_single,
                    sample_lognormal
                )
                import itertools

                cost_scenario = scenario.clone_with_seed(scenario.random_seed)

                # Create resources
                diagnostic_resources = {}
                for diag_type, diag_config in cost_scenario.diagnostic_configs.items():
                    if diag_config.enabled:
                        diagnostic_resources[diag_type] = simpy.PriorityResource(
                            env, capacity=diag_config.capacity
                        )

                transfer_ambulances = None
                transfer_helicopters = None
                if cost_scenario.transfer_config.enabled:
                    transfer_ambulances = simpy.Resource(
                        env, capacity=cost_scenario.transfer_config.n_transfer_ambulances
                    )
                    transfer_helicopters = simpy.Resource(
                        env, capacity=cost_scenario.transfer_config.n_transfer_helicopters
                    )

                theatre_tables = None
                itu_beds = None
                ward_beds = None
                if cost_scenario.downstream_enabled:
                    if cost_scenario.theatre_config and cost_scenario.theatre_config.enabled:
                        theatre_tables = simpy.PriorityResource(
                            env, capacity=cost_scenario.theatre_config.n_tables
                        )
                    if cost_scenario.itu_config and cost_scenario.itu_config.enabled:
                        itu_beds = simpy.Resource(env, capacity=cost_scenario.itu_config.capacity)
                    if cost_scenario.ward_config and cost_scenario.ward_config.enabled:
                        ward_beds = simpy.Resource(env, capacity=cost_scenario.ward_config.capacity)

                hems_slots = None
                if cost_scenario.aeromed_config and cost_scenario.aeromed_config.enabled:
                    if cost_scenario.aeromed_config.hems.enabled:
                        hems_slots = simpy.Resource(
                            env, capacity=cost_scenario.aeromed_config.hems.slots_per_day
                        )

                resources = AEResources(
                    triage=simpy.PriorityResource(env, capacity=cost_scenario.n_triage),
                    ed_bays=simpy.PriorityResource(env, capacity=cost_scenario.n_ed_bays),
                    handover_bays=simpy.Resource(env, capacity=cost_scenario.n_handover_bays),
                    ambulance_fleet=simpy.Resource(env, capacity=cost_scenario.n_ambulances),
                    helicopter_fleet=simpy.Resource(env, capacity=cost_scenario.n_helicopters),
                    diagnostics=diagnostic_resources,
                    transfer_ambulances=transfer_ambulances,
                    transfer_helicopters=transfer_helicopters,
                    theatre_tables=theatre_tables,
                    itu_beds=itu_beds,
                    ward_beds=ward_beds,
                    hems_slots=hems_slots,
                )

                collector = FullResultsCollector()
                patient_counter = itertools.count(1)

                # Start arrival generators
                if cost_scenario.arrival_configs:
                    for arr_config in cost_scenario.arrival_configs:
                        env.process(arrival_generator_multistream(
                            env, resources, arr_config, cost_scenario, collector, patient_counter
                        ))
                else:
                    env.process(arrival_generator_single(
                        env, resources, cost_scenario, collector, patient_counter
                    ))

                # Run simulation
                total_time = cost_scenario.warm_up + cost_scenario.run_length
                env.run(until=total_time)

                # Filter to post-warmup patients who have departed
                valid_patients = [
                    p for p in collector.patients
                    if p.arrival_time >= cost_scenario.warm_up and p.departure_time is not None
                ]

                # Calculate costs
                breakdown = calculate_costs(valid_patients, config)
                by_priority = calculate_costs_by_priority(valid_patients, config)

                # Store results in session state
                st.session_state['cost_breakdown'] = breakdown
                st.session_state['cost_by_priority'] = by_priority
                st.session_state['cost_config'] = config
                st.session_state['cost_patients_count'] = len(valid_patients)

                st.success(f"Cost calculation complete for {len(valid_patients)} patients.")

            except Exception as e:
                st.error(f"Error calculating costs: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results if available
    if 'cost_breakdown' in st.session_state:
        breakdown = st.session_state['cost_breakdown']
        by_priority = st.session_state['cost_by_priority']
        sym = breakdown.get_currency_symbol()

        st.markdown("---")
        st.subheader("Cost Summary")

        # Key metrics
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric(
                "Grand Total",
                format_currency(breakdown.grand_total, sym),
                help="Total cost across all categories"
            )
        with summary_cols[1]:
            st.metric(
                "Cost per Patient",
                format_currency(breakdown.cost_per_patient, sym),
                help="Average cost per patient"
            )
        with summary_cols[2]:
            st.metric(
                "Patients",
                f"{breakdown.total_patients:,}",
                help="Number of patients included in calculation"
            )

        # Cost breakdown by category
        st.subheader("Breakdown by Category")

        cat_cols = st.columns(4)
        with cat_cols[0]:
            st.markdown("**Bed Costs**")
            st.write(f"ED Bays: {format_currency(breakdown.ed_bay_costs, sym)}")
            st.write(f"ITU: {format_currency(breakdown.itu_bed_costs, sym)}")
            st.write(f"Ward: {format_currency(breakdown.ward_bed_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_bed_costs, sym)}**")

        with cat_cols[1]:
            st.markdown("**Theatre**")
            st.write(f"Theatre: {format_currency(breakdown.theatre_costs, sym)}")

        with cat_cols[2]:
            st.markdown("**Per-Episode**")
            st.write(f"Triage: {format_currency(breakdown.triage_costs, sym)}")
            st.write(f"Diagnostics: {format_currency(breakdown.diagnostics_costs, sym)}")
            st.write(f"Discharge: {format_currency(breakdown.discharge_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_episode_costs, sym)}**")

        with cat_cols[3]:
            st.markdown("**Transport**")
            st.write(f"Ambulance: {format_currency(breakdown.ambulance_costs, sym)}")
            st.write(f"HEMS: {format_currency(breakdown.hems_costs, sym)}")
            st.write(f"Fixed-wing: {format_currency(breakdown.fixedwing_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_transport_costs, sym)}**")

        # Cost by priority
        st.subheader("Breakdown by Priority")

        priority_data = pd.DataFrame({
            "Priority": ["P1 (Immediate)", "P2 (Very Urgent)", "P3 (Urgent)", "P4 (Standard)"],
            "Patients": [by_priority.p1_count, by_priority.p2_count,
                        by_priority.p3_count, by_priority.p4_count],
            "Total Cost": [
                format_currency(by_priority.p1_total, sym),
                format_currency(by_priority.p2_total, sym),
                format_currency(by_priority.p3_total, sym),
                format_currency(by_priority.p4_total, sym),
            ],
            "Per Patient": [
                format_currency(by_priority.p1_per_patient, sym),
                format_currency(by_priority.p2_per_patient, sym),
                format_currency(by_priority.p3_per_patient, sym),
                format_currency(by_priority.p4_per_patient, sym),
            ],
        })
        st.dataframe(priority_data, use_container_width=True, hide_index=True)

        # Cost distribution chart
        cost_data = pd.DataFrame({
            "Category": ["ED Bays", "Theatre", "ITU", "Ward", "Transport", "Per-Episode"],
            "Cost": [
                breakdown.ed_bay_costs,
                breakdown.theatre_costs,
                breakdown.itu_bed_costs,
                breakdown.ward_bed_costs,
                breakdown.total_transport_costs,
                breakdown.total_episode_costs,
            ]
        })

        fig = px.pie(
            cost_data,
            values="Cost",
            names="Category",
            title="Cost Distribution by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export cost data
        st.markdown("---")
        cost_export = pd.DataFrame([breakdown.to_dict()])
        csv_costs = cost_export.to_csv(index=False)
        st.download_button(
            label="Download Cost Breakdown (CSV)",
            data=csv_costs,
            file_name="faer_costs.csv",
            mime="text/csv",
        )

st.divider()

# ===== EXPORT SECTION =====
st.header("Export Results")

# Create DataFrame for export
export_df = pd.DataFrame({
    "replication": range(1, n_reps + 1),
    "arrivals": results["arrivals"],
    "departures": results["departures"],
    "arrivals_resus": results["arrivals_resus"],
    "arrivals_majors": results["arrivals_majors"],
    "arrivals_minors": results["arrivals_minors"],
    "p_delay": results["p_delay"],
    "mean_triage_wait": results["mean_triage_wait"],
    "mean_treatment_wait": results["mean_treatment_wait"],
    "mean_system_time": results["mean_system_time"],
    "p95_system_time": results["p95_system_time"],
    "admission_rate": results["admission_rate"],
    "admitted": results["admitted"],
    "discharged": results["discharged"],
    # Emergency Services
    "util_ambulance_fleet": get_util("util_ambulance_fleet"),
    "util_helicopter_fleet": get_util("util_helicopter_fleet"),
    "util_handover": get_util("util_handover"),
    # Triage & ED
    "util_triage": get_util("util_triage"),
    "util_ed_bays": get_util("util_ed_bays"),
    # Diagnostics
    "util_ct": get_util("util_CT_SCAN"),
    "util_xray": get_util("util_XRAY"),
    "util_bloods": get_util("util_BLOODS"),
    # Downstream
    "util_theatre": get_util("util_theatre"),
    "util_itu": get_util("util_itu"),
    "util_ward": get_util("util_ward"),
    # Aeromed
    "aeromed_total": results.get("aeromed_total", [0] * n_reps),
    "aeromed_hems_count": results.get("aeromed_hems_count", [0] * n_reps),
    "aeromed_fixedwing_count": results.get("aeromed_fixedwing_count", [0] * n_reps),
    "aeromed_slots_missed": results.get("aeromed_slots_missed", [0] * n_reps),
    "mean_aeromed_slot_wait": results.get("mean_aeromed_slot_wait", [0] * n_reps),
    "ward_bed_days_blocked_aeromed": results.get("ward_bed_days_blocked_aeromed", [0] * n_reps),
})

# Show preview
with st.expander("Preview data"):
    st.dataframe(export_df, use_container_width=True)

# Summary statistics
with st.expander("Summary statistics"):
    summary_stats = export_df.describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)

# Download buttons
col1, col2 = st.columns(2)

with col1:
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download Raw Results (CSV)",
        data=csv,
        file_name="faer_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    # Create summary with CIs from export dataframe
    summary_data = []
    for metric in export_df.columns[1:]:  # Skip replication column
        values = export_df[metric].tolist()
        ci = compute_ci(values)
        summary_data.append({
            "metric": metric,
            "mean": ci["mean"],
            "std": ci["std"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "ci_half_width": ci["ci_half_width"],
        })
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False)

    st.download_button(
        label="Download Summary with CIs (CSV)",
        data=summary_csv,
        file_name="faer_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
