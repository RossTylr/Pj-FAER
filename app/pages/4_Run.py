"""
Run Simulation Page (Enhanced Phase 8a).

Shows configuration summary, runs simulation, displays results with schematic.
This page combines settings from Arrivals and Resources into a FullScenario
and executes the simulation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict

from faer.core.scenario import (
    FullScenario,
    ArrivalModel,
    DayType,
    DiagnosticConfig,
    ITUConfig,
    WardConfig,
    TheatreConfig,
)
from faer.core.entities import DiagnosticType, Priority, ArrivalMode
from faer.experiment.runner import multiple_replications
from app.components.schematic import (
    build_capacity_graph_from_params,
    build_results_schematic
)

st.set_page_config(page_title="Run Simulation", page_icon="▶️", layout="wide")
st.title("▶️ Run Simulation")

# ============================================================
# HELPER: Initialize defaults if not set
# ============================================================
def init_defaults():
    """Ensure all session state defaults are set."""
    defaults = {
        # Fleet
        'n_ambulances': 10,
        'ambulance_turnaround': 45.0,
        'n_helicopters': 2,
        'helicopter_turnaround': 90.0,

        # Handover
        'n_handover_bays': 4,
        'handover_time_mean': 15.0,
        'handover_time_cv': 0.3,

        # Arrival model
        'arrival_model': 'profile_24h',
        'day_type': 'weekday',
        'demand_multiplier': 1.0,
        'ambulance_rate_multiplier': 1.0,
        'helicopter_rate_multiplier': 1.0,
        'walkin_rate_multiplier': 1.0,

        # Triage
        'n_triage': 3,
        'triage_time_mean': 5.0,
        'triage_time_cv': 0.3,

        # ED
        'n_ed_bays': 20,
        'ed_assessment_time_mean': 15.0,
        'ed_treatment_time_mean': 45.0,
        'ed_service_time_cv': 0.4,

        # Diagnostics
        'ct_capacity': 2,
        'ct_scan_time_mean': 20.0,
        'ct_report_time_mean': 30.0,
        'ct_enabled': True,
        'xray_capacity': 3,
        'xray_time_mean': 10.0,
        'xray_report_time_mean': 15.0,
        'xray_enabled': True,
        'bloods_capacity': 5,
        'bloods_draw_time_mean': 5.0,
        'bloods_lab_time_mean': 45.0,
        'bloods_enabled': True,

        # Theatre
        'n_theatre_tables': 2,
        'theatre_procedure_time_mean': 90.0,
        'postop_itu_probability': 0.25,

        # ITU
        'n_itu_beds': 6,
        'itu_los_hours_mean': 48.0,
        'itu_los_cv': 0.8,
        'itu_stepdown_probability': 0.85,
        'itu_mortality_probability': 0.10,

        # Ward
        'n_ward_beds': 30,
        'ward_los_hours_mean': 72.0,
        'ward_los_cv': 1.0,
        'ward_turnaround_mins': 30.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_defaults()


# ============================================================
# HELPER: Build FullScenario from session state
# ============================================================
def build_scenario_from_session(run_length: float, warm_up: float, random_seed: int) -> FullScenario:
    """Build a FullScenario from current session state values."""

    # Map string arrival_model to enum
    arrival_model_str = st.session_state.get('arrival_model', 'profile_24h')
    arrival_model_enum = {
        'simple': ArrivalModel.SIMPLE,
        'profile_24h': ArrivalModel.PROFILE_24H,
        'detailed': ArrivalModel.DETAILED,
    }.get(arrival_model_str, ArrivalModel.PROFILE_24H)

    # Map string day_type to enum
    day_type_str = st.session_state.get('day_type', 'weekday')
    day_type_enum = {
        'weekday': DayType.WEEKDAY,
        'monday': DayType.MONDAY,
        'friday_eve': DayType.FRIDAY_EVE,
        'sat_night': DayType.SATURDAY_NIGHT,
        'sunday': DayType.SUNDAY,
        'bank_holiday': DayType.BANK_HOLIDAY,
    }.get(day_type_str, DayType.WEEKDAY)

    # Build diagnostic configs
    diagnostic_configs = {}

    if st.session_state.get('ct_enabled', True):
        diagnostic_configs[DiagnosticType.CT_SCAN] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.CT_SCAN,
            capacity=st.session_state.get('ct_capacity', 2),
            process_time_mean=st.session_state.get('ct_scan_time_mean', 20.0),
            turnaround_time_mean=st.session_state.get('ct_report_time_mean', 30.0),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.70,
                Priority.P2_VERY_URGENT: 0.40,
                Priority.P3_URGENT: 0.15,
                Priority.P4_STANDARD: 0.05,
            }
        )

    if st.session_state.get('xray_enabled', True):
        diagnostic_configs[DiagnosticType.XRAY] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.XRAY,
            capacity=st.session_state.get('xray_capacity', 3),
            process_time_mean=st.session_state.get('xray_time_mean', 10.0),
            turnaround_time_mean=st.session_state.get('xray_report_time_mean', 15.0),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.30,
                Priority.P2_VERY_URGENT: 0.35,
                Priority.P3_URGENT: 0.40,
                Priority.P4_STANDARD: 0.25,
            }
        )

    if st.session_state.get('bloods_enabled', True):
        diagnostic_configs[DiagnosticType.BLOODS] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.BLOODS,
            capacity=st.session_state.get('bloods_capacity', 5),
            process_time_mean=st.session_state.get('bloods_draw_time_mean', 5.0),
            turnaround_time_mean=st.session_state.get('bloods_lab_time_mean', 45.0),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.90,
                Priority.P2_VERY_URGENT: 0.80,
                Priority.P3_URGENT: 0.50,
                Priority.P4_STANDARD: 0.20,
            }
        )

    # Build downstream configs
    itu_config = ITUConfig(
        capacity=st.session_state.get('n_itu_beds', 6),
        los_mean_hours=st.session_state.get('itu_los_hours_mean', 48.0),
        los_cv=st.session_state.get('itu_los_cv', 0.8),
        step_down_to_ward_prob=st.session_state.get('itu_stepdown_probability', 0.85),
        death_prob=st.session_state.get('itu_mortality_probability', 0.10),
        direct_discharge_prob=1.0 - st.session_state.get('itu_stepdown_probability', 0.85) - st.session_state.get('itu_mortality_probability', 0.10),
    )

    ward_config = WardConfig(
        capacity=st.session_state.get('n_ward_beds', 30),
        los_mean_hours=st.session_state.get('ward_los_hours_mean', 72.0),
        los_cv=st.session_state.get('ward_los_cv', 1.0),
        turnaround_mins=st.session_state.get('ward_turnaround_mins', 30.0),
    )

    theatre_config = TheatreConfig(
        n_tables=st.session_state.get('n_theatre_tables', 2),
        procedure_time_mean=st.session_state.get('theatre_procedure_time_mean', 90.0),
        to_itu_prob=st.session_state.get('postop_itu_probability', 0.25),
        to_ward_prob=1.0 - st.session_state.get('postop_itu_probability', 0.25),
    )

    # Build FullScenario
    scenario = FullScenario(
        # Run config
        run_length=run_length,
        warm_up=warm_up,
        random_seed=random_seed,

        # Fleet
        n_ambulances=st.session_state.get('n_ambulances', 10),
        n_helicopters=st.session_state.get('n_helicopters', 2),
        ambulance_turnaround_mins=st.session_state.get('ambulance_turnaround', 45.0),
        helicopter_turnaround_mins=st.session_state.get('helicopter_turnaround', 90.0),

        # Handover
        n_handover_bays=st.session_state.get('n_handover_bays', 4),
        handover_time_mean=st.session_state.get('handover_time_mean', 15.0),
        handover_time_cv=st.session_state.get('handover_time_cv', 0.3),

        # Triage
        n_triage=st.session_state.get('n_triage', 3),
        triage_mean=st.session_state.get('triage_time_mean', 5.0),
        triage_cv=st.session_state.get('triage_time_cv', 0.3),

        # ED
        n_ed_bays=st.session_state.get('n_ed_bays', 20),
        ed_service_mean=st.session_state.get('ed_treatment_time_mean', 45.0),
        ed_service_cv=st.session_state.get('ed_service_time_cv', 0.4),

        # Arrivals
        arrival_model=arrival_model_enum,
        day_type=day_type_enum,
        demand_multiplier=st.session_state.get('demand_multiplier', 1.0),
        ambulance_rate_multiplier=st.session_state.get('ambulance_rate_multiplier', 1.0),
        helicopter_rate_multiplier=st.session_state.get('helicopter_rate_multiplier', 1.0),
        walkin_rate_multiplier=st.session_state.get('walkin_rate_multiplier', 1.0),

        # Detailed arrivals (if applicable)
        detailed_arrivals=st.session_state.get('detailed_arrivals', None),

        # Diagnostics
        diagnostic_configs=diagnostic_configs,

        # Downstream configs
        itu_config=itu_config,
        ward_config=ward_config,
        theatre_config=theatre_config,
    )

    return scenario


# ============================================================
# SECTION 1: PRE-RUN CONFIGURATION SUMMARY
# ============================================================
st.header("📋 Configuration Summary")
st.markdown("Review your configuration before running.")

tab_arrivals, tab_resources, tab_schematic = st.tabs(["Arrivals", "Resources", "Schematic"])

with tab_arrivals:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ambulance**")
        st.write(f"Vehicles: {st.session_state.get('n_ambulances', 10)}")
        st.write(f"Turnaround: {st.session_state.get('ambulance_turnaround', 45):.0f} min")

    with col2:
        st.markdown("**HEMS**")
        st.write(f"Aircraft: {st.session_state.get('n_helicopters', 2)}")
        st.write(f"Turnaround: {st.session_state.get('helicopter_turnaround', 90):.0f} min")

    with col3:
        st.markdown("**Handover**")
        st.write(f"Bays: {st.session_state.get('n_handover_bays', 4)}")
        st.write(f"Time: {st.session_state.get('handover_time_mean', 15):.0f} min")

    st.markdown("---")
    model = st.session_state.get('arrival_model', 'profile_24h')
    st.write(f"**Arrival Model**: {model}")
    if model == 'profile_24h':
        st.write(f"**Day Type**: {st.session_state.get('day_type', 'weekday')}")
        st.write(f"**Demand Scale**: {st.session_state.get('demand_multiplier', 1.0):.1f}x")

with tab_resources:
    # Bed capacity - ED, Theatre, ITU, Ward only
    st.markdown("**Bed Capacity**")
    bed_resources = {
        'Resource': ['ED Bays', 'Theatre', 'ITU', 'Ward'],
        'Capacity': [
            st.session_state.get('n_ed_bays', 20),
            st.session_state.get('n_theatre_tables', 2),
            st.session_state.get('n_itu_beds', 6),
            st.session_state.get('n_ward_beds', 30),
        ],
        'Key Timing': [
            f"{st.session_state.get('ed_assessment_time_mean', 15):.0f}+{st.session_state.get('ed_treatment_time_mean', 45):.0f}m",
            f"{st.session_state.get('theatre_procedure_time_mean', 90):.0f}m",
            f"{st.session_state.get('itu_los_hours_mean', 48):.0f}h LOS",
            f"{st.session_state.get('ward_los_hours_mean', 72):.0f}h LOS",
        ]
    }
    st.dataframe(pd.DataFrame(bed_resources), use_container_width=True, hide_index=True)

    # Total bed capacity
    total_beds = (st.session_state.get('n_ed_bays', 20) +
                  st.session_state.get('n_theatre_tables', 2) +
                  st.session_state.get('n_itu_beds', 6) +
                  st.session_state.get('n_ward_beds', 30))
    st.metric("Total Bed Capacity", total_beds)

    # Other resources (triage, diagnostics) - compact display
    st.markdown("---")
    st.markdown("**Other Resources**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"Triage: {st.session_state.get('n_triage', 3)}")
    with col2:
        ct_val = st.session_state.get('ct_capacity', 2) if st.session_state.get('ct_enabled', True) else '-'
        st.write(f"CT: {ct_val}")
    with col3:
        xray_val = st.session_state.get('xray_capacity', 3) if st.session_state.get('xray_enabled', True) else '-'
        st.write(f"X-ray: {xray_val}")
    with col4:
        bloods_val = st.session_state.get('bloods_capacity', 5) if st.session_state.get('bloods_enabled', True) else '-'
        st.write(f"Bloods: {bloods_val}")

with tab_schematic:
    schematic = build_capacity_graph_from_params(
        n_ambulances=st.session_state.get('n_ambulances', 10),
        n_helicopters=st.session_state.get('n_helicopters', 2),
        n_handover=st.session_state.get('n_handover_bays', 4),
        n_triage=st.session_state.get('n_triage', 3),
        n_ed_bays=st.session_state.get('n_ed_bays', 20),
        n_ct=st.session_state.get('ct_capacity', 2),
        n_xray=st.session_state.get('xray_capacity', 3),
        n_bloods=st.session_state.get('bloods_capacity', 5),
        n_theatre=st.session_state.get('n_theatre_tables', 2),
        n_itu=st.session_state.get('n_itu_beds', 6),
        n_ward=st.session_state.get('n_ward_beds', 30),
        ct_enabled=st.session_state.get('ct_enabled', True),
        xray_enabled=st.session_state.get('xray_enabled', True),
        bloods_enabled=st.session_state.get('bloods_enabled', True),
    )
    st.graphviz_chart(schematic, use_container_width=True)

# ============================================================
# SECTION 2: RUN CONTROLS
# ============================================================
st.markdown("---")
st.header("▶️ Run Controls")

# Check arrival model settings from Arrivals page
arrival_model = st.session_state.get('arrival_model', 'profile_24h')
detailed_config_hours = st.session_state.get('detailed_config_hours', 24)

# Determine default run length based on arrival model
run_length_options = [720, 1440, 2160, 2880, 3600, 4320]  # 12, 24, 36, 48, 60, 72 hours
run_length_labels = {
    720: "12 hours",
    1440: "24 hours (1 day)",
    2160: "36 hours",
    2880: "48 hours (2 days)",
    3600: "60 hours",
    4320: "72 hours (3 days)",
}

# Set default index based on arrival model
if arrival_model == 'detailed':
    # Match the detailed config period
    detailed_config_mins = detailed_config_hours * 60
    if detailed_config_mins in run_length_options:
        default_index = run_length_options.index(detailed_config_mins)
    else:
        default_index = 1  # Default to 24h
else:
    default_index = 1  # Default to 24h for simple/profile modes

col1, col2, col3, col4 = st.columns(4)

with col1:
    run_length = st.selectbox(
        "Run Length",
        options=run_length_options,
        format_func=lambda x: run_length_labels[x],
        index=default_index
    )

with col2:
    random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)

with col3:
    warm_up = st.number_input("Warm-up (mins)", min_value=0, max_value=480, value=60)

with col4:
    n_reps = st.number_input("Replications", min_value=1, max_value=100, value=30)

# Show arrival model info and any mismatch warnings
run_length_hours = run_length / 60

if arrival_model == 'detailed':
    if run_length_hours == detailed_config_hours:
        st.success(f"Run length matches detailed arrival config ({detailed_config_hours}h)")
    elif run_length_hours > detailed_config_hours:
        repeats = run_length_hours / detailed_config_hours
        st.info(f"Detailed arrival pattern ({detailed_config_hours}h) will repeat {repeats:.1f}x for {run_length_hours:.0f}h run")
    else:
        st.warning(f"Run length ({run_length_hours:.0f}h) shorter than arrival config ({detailed_config_hours}h) - only first {run_length_hours:.0f}h of pattern used")
elif arrival_model == 'profile_24h':
    day_type = st.session_state.get('day_type', 'weekday')
    if run_length_hours >= 24:
        st.info(f"Using 24h profile ({day_type}) - pattern repeats for {run_length_hours:.0f}h run")
elif arrival_model == 'simple':
    st.info(f"Using simple daily totals distributed across {run_length_hours:.0f}h")

# Build scenario button
if st.button("▶️ Run Simulation", type="primary", use_container_width=True):

    # Build FullScenario from session state
    try:
        scenario = build_scenario_from_session(run_length, warm_up, random_seed)

        # Store scenario
        st.session_state.scenario = scenario

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        def progress_callback(current: int, total: int) -> None:
            progress_bar.progress(current / total)
            status_text.text(f"Running replication {current}/{total}...")

        # Run replications with full model metrics
        results = multiple_replications(
            scenario,
            n_reps=n_reps,
            progress_callback=progress_callback,
            use_multistream=True,
        )

        elapsed = time.time() - start_time

        # Store results
        st.session_state.run_results = results
        st.session_state.run_complete = True
        st.session_state.run_time = elapsed
        st.session_state.run_scenario = scenario

        # Complete
        progress_bar.progress(1.0)
        status_text.empty()

        st.success(f"Completed {n_reps} replications in {elapsed:.1f} seconds")
        st.rerun()

    except Exception as e:
        st.error(f"Error running simulation: {e}")

# ============================================================
# SECTION 3: RESULTS SUMMARY (if available)
# ============================================================
if st.session_state.get('run_complete') and 'run_results' in st.session_state:
    results = st.session_state.run_results

    st.markdown("---")
    st.header("📊 Results Summary")

    # Headline KPIs only
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Arrivals", f"{np.mean(results.get('arrivals', [0])):.0f}")
    with col2:
        st.metric("Mean System Time", f"{np.mean(results.get('mean_system_time', [0])):.0f} min")
    with col3:
        p_delay = np.mean(results.get('p_delay', [0]))
        st.metric("P(Delay)", f"{p_delay:.1%}")
    with col4:
        admission = np.mean(results.get('admission_rate', [0]))
        st.metric("Admission Rate", f"{admission:.1%}")

    # Quick bottleneck check
    util_ed = np.mean(results.get('util_ed_bays', [0]))
    util_handover = np.mean(results.get('util_handover', [0]))
    max_util = max(util_ed, util_handover)

    if max_util > 0.85:
        st.warning(f"High utilisation detected ({max_util:.0%}). Check Results page for details.")
    else:
        st.success("Simulation complete. View detailed results below.")

    # Prominent link to Results page
    st.page_link("pages/5_Results.py", label="View Full Results & Analysis", icon="📈", use_container_width=True)

# No results yet
else:
    st.markdown("---")
    st.info("Configure settings above and click **Run Simulation** to see results.")
