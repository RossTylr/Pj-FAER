"""
Arrivals Configuration Page.

Configures emergency service fleet and patient arrival patterns.
Extracted from original 1_Scenario.py for operational clarity.

Integration:
- Stores values to st.session_state for FullScenario construction
- Compatible with existing run_full_simulation() flow
- Links to ArrivalMode enum from entities.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Optional

# Import from existing codebase
from faer.core.entities import ArrivalMode, Priority, ArrivalModel, DayType
from faer.core.scenario import (
    FullScenario,
    DetailedArrivalConfig,
    DAY_TYPE_MULTIPLIERS,
)

st.set_page_config(
    page_title="Arrivals",
    page_icon="🚑",
    layout="wide"
)

st.title("🚑 Arrivals Configuration")

st.info("""
Configure how patients arrive at the Emergency Department.
This page sets fleet sizes, arrival patterns, and day type profiles.

**Data Flow**: Settings here are stored in session state and combined
with Resources settings when you run the simulation.
""")

# ============================================================
# HELPER: Initialize session state with defaults from FullScenario
# ============================================================
def init_session_defaults():
    """Initialize session state with FullScenario defaults if not set."""
    defaults = {
        # Fleet - matches FullScenario defaults
        'n_ambulances': 10,
        'ambulance_turnaround': 45.0,
        'litters_per_ambulance': 1,
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

        # Per-stream multipliers
        'ambulance_rate_multiplier': 1.0,
        'helicopter_rate_multiplier': 1.0,
        'walkin_rate_multiplier': 1.0,

        # Walk-in
        'walkin_enabled': True,

        # Simple mode daily totals
        'daily_ambulance': 80,
        'daily_helicopter': 3,
        'daily_walkin': 50,

        # Detailed config hours
        'detailed_config_hours': 24,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize defaults
init_session_defaults()

# ============================================================
# SECTION 1: EMERGENCY SERVICES
# ============================================================
st.header("🚨 Emergency Services")

st.markdown("""
Configure the emergency service fleet that brings patients to the ED.
These resources determine arrival capacity and turnaround constraints.

**Simulation linkage**: Creates `simpy.Resource` for fleet management.
""")

# Three columns for the three arrival modes
col_amb, col_heli, col_walk = st.columns(3)

# ---------------- AMBULANCE ----------------
with col_amb:
    st.subheader("🚑 Ambulance Service")

    with st.container(border=True):
        # Fleet size - maps to FullScenario.n_ambulances
        n_ambulances = st.number_input(
            "Available Ambulances",
            min_value=1,
            max_value=50,
            value=st.session_state.n_ambulances,
            key="input_n_ambulances",
            help="Maps to FullScenario.n_ambulances -> simpy.Resource"
        )
        st.session_state.n_ambulances = n_ambulances

        # Turnaround - maps to FullScenario.ambulance_turnaround_mins
        amb_turnaround = st.number_input(
            "Turnaround Time (mins)",
            min_value=15.0,
            max_value=120.0,
            value=float(st.session_state.ambulance_turnaround),
            step=5.0,
            key="input_amb_turnaround",
            help="Time from patient delivery to vehicle available. Maps to FullScenario.ambulance_turnaround_mins"
        )
        st.session_state.ambulance_turnaround = amb_turnaround

        # Litters per ambulance (extension)
        litters = st.number_input(
            "Litters per Ambulance",
            min_value=1,
            max_value=2,
            value=st.session_state.litters_per_ambulance,
            key="input_litters",
            help="Patient capacity per vehicle (1 standard, 2 for MCI config)"
        )
        st.session_state.litters_per_ambulance = litters

        # Calculated capacity metric
        hourly_capacity = n_ambulances * (60 / amb_turnaround) * litters
        st.metric(
            "Max Arrivals/Hour",
            f"{hourly_capacity:.1f}",
            help="Theoretical max: n_ambulances x (60/turnaround) x litters"
        )

        # Acuity distribution (read-only, from typical ambulance data)
        st.markdown("**Typical Acuity Mix** *(read-only)*")
        st.caption("Based on ambulance conveyance patterns")

        # These could be made configurable if needed
        amb_acuity = {
            Priority.P1_IMMEDIATE: 0.15,
            Priority.P2_VERY_URGENT: 0.40,
            Priority.P3_URGENT: 0.35,
            Priority.P4_STANDARD: 0.10
        }

        for priority, proportion in amb_acuity.items():
            label = priority.name.replace('_', ' ').title()
            st.progress(proportion, text=f"{label}: {proportion:.0%}")

# ---------------- HELICOPTER (HEMS) ----------------
with col_heli:
    st.subheader("🚁 HEMS / Air Ambulance")

    with st.container(border=True):
        # Fleet size - maps to FullScenario.n_helicopters
        n_helicopters = st.number_input(
            "Aircraft Available",
            min_value=0,
            max_value=5,
            value=st.session_state.n_helicopters,
            key="input_n_helicopters",
            help="Maps to FullScenario.n_helicopters -> simpy.Resource"
        )
        st.session_state.n_helicopters = n_helicopters

        # Turnaround - maps to FullScenario.helicopter_turnaround_mins
        heli_turnaround = st.number_input(
            "Turnaround Time (mins)",
            min_value=30.0,
            max_value=180.0,
            value=float(st.session_state.helicopter_turnaround),
            step=10.0,
            key="input_heli_turnaround",
            help="Refuel, crew change, mission reset. Maps to FullScenario.helicopter_turnaround_mins"
        )
        st.session_state.helicopter_turnaround = heli_turnaround

        # Operating hours note
        st.info("HEMS typically operates 07:00-02:00 (19 hrs)")

        # Calculated capacity
        if n_helicopters > 0:
            # 19 hours of operation
            heli_daily = n_helicopters * 19 * (60 / heli_turnaround)
            st.metric("Max Daily Missions", f"{heli_daily:.0f}")
        else:
            st.metric("Max Daily Missions", "0")
            st.warning("No HEMS coverage configured")

        # Acuity distribution (HEMS = highest acuity)
        st.markdown("**Typical Acuity Mix** *(read-only)*")
        st.caption("HEMS carries highest acuity patients")

        hems_acuity = {
            Priority.P1_IMMEDIATE: 0.70,
            Priority.P2_VERY_URGENT: 0.25,
            Priority.P3_URGENT: 0.05,
            Priority.P4_STANDARD: 0.00
        }

        for priority, proportion in hems_acuity.items():
            if proportion > 0:
                label = priority.name.replace('_', ' ').title()
                st.progress(proportion, text=f"{label}: {proportion:.0%}")

# ---------------- WALK-IN ----------------
with col_walk:
    st.subheader("🚶 Self-Presentation (Walk-in)")

    with st.container(border=True):
        st.markdown("**No fleet constraint**")
        st.caption("Walk-in patients not limited by vehicle availability")

        walkin_enabled = st.checkbox(
            "Enable Walk-in Arrivals",
            value=st.session_state.walkin_enabled,
            key="input_walkin_enabled",
            help="Uncheck to model ambulance-only scenarios (e.g., MCI)"
        )
        st.session_state.walkin_enabled = walkin_enabled

        if walkin_enabled:
            # Walk-in scaling - maps to FullScenario.walkin_rate_multiplier
            walkin_scale = st.slider(
                "Walk-in Volume Scale",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.walkin_rate_multiplier),
                step=0.1,
                key="input_walkin_scale",
                help="Maps to FullScenario.walkin_rate_multiplier"
            )
            st.session_state.walkin_rate_multiplier = walkin_scale

            # Acuity distribution (walk-ins = lower acuity)
            st.markdown("**Typical Acuity Mix** *(read-only)*")
            st.caption("Walk-ins typically lower acuity")

            walkin_acuity = {
                Priority.P1_IMMEDIATE: 0.02,
                Priority.P2_VERY_URGENT: 0.15,
                Priority.P3_URGENT: 0.45,
                Priority.P4_STANDARD: 0.38
            }

            for priority, proportion in walkin_acuity.items():
                if proportion > 0.05:  # Only show significant
                    label = priority.name.replace('_', ' ').title()
                    st.progress(proportion, text=f"{label}: {proportion:.0%}")
        else:
            st.warning("Walk-in arrivals disabled")
            st.session_state.walkin_rate_multiplier = 0.0

# ============================================================
# SECTION 2: HANDOVER GATE
# ============================================================
st.markdown("---")
st.header("🚪 Handover Gate")

st.markdown("""
Ambulance and HEMS crews hand over patients at the handover bay.
When the ED is congested, handover delays occur (crews wait with patients).

**Critical feedback mechanism**: Full ED -> delayed handover release ->
ambulance crews wait -> 999 response delays in community.

**Simulation linkage**: Creates `simpy.Resource(n_handover_bays)`.
Handover bay is held until patient acquires ED bay.
""")

col1, col2, col3 = st.columns(3)

with col1:
    # Maps to FullScenario.n_handover_bays
    n_handover = st.number_input(
        "Handover Bays",
        min_value=1,
        max_value=20,
        value=st.session_state.n_handover_bays,
        key="input_n_handover",
        help="Physical spaces for ambulance handover. Maps to FullScenario.n_handover_bays"
    )
    st.session_state.n_handover_bays = n_handover

with col2:
    # Maps to FullScenario.handover_time_mean
    handover_time = st.number_input(
        "Handover Time Mean (mins)",
        min_value=5.0,
        max_value=60.0,
        value=float(st.session_state.handover_time_mean),
        step=1.0,
        key="input_handover_time",
        help="Average time for clinical handover process. Maps to FullScenario.handover_time_mean"
    )
    st.session_state.handover_time_mean = handover_time

with col3:
    # Maps to FullScenario.handover_time_cv
    handover_cv = st.slider(
        "Handover Time Variability (CV)",
        min_value=0.1,
        max_value=0.8,
        value=float(st.session_state.handover_time_cv),
        step=0.05,
        key="input_handover_cv",
        help="Coefficient of variation. Higher = more variable handover times."
    )
    st.session_state.handover_time_cv = handover_cv

# Capacity check
st.markdown("**Capacity Validation**")

handover_capacity_per_hour = n_handover * (60 / handover_time)
ambulance_rate_per_hour = n_ambulances * (60 / amb_turnaround)
total_conveyed_rate = ambulance_rate_per_hour + (n_helicopters * (60 / heli_turnaround) if n_helicopters > 0 else 0)

col1, col2 = st.columns(2)
with col1:
    st.metric("Handover Throughput", f"{handover_capacity_per_hour:.1f}/hr")
with col2:
    st.metric("Max Conveyed Arrivals", f"{total_conveyed_rate:.1f}/hr")

if total_conveyed_rate > handover_capacity_per_hour * 0.9:
    st.warning(f"Conveyed arrivals ({total_conveyed_rate:.1f}/hr) approaching handover capacity ({handover_capacity_per_hour:.1f}/hr). Risk of handover delays.")
elif total_conveyed_rate > handover_capacity_per_hour:
    st.error(f"Conveyed arrivals ({total_conveyed_rate:.1f}/hr) EXCEED handover capacity ({handover_capacity_per_hour:.1f}/hr). Handover delays guaranteed.")
else:
    st.success(f"Handover capacity ({handover_capacity_per_hour:.1f}/hr) sufficient for conveyed rate ({total_conveyed_rate:.1f}/hr)")

# ============================================================
# SECTION 3: SESSION STATE SUMMARY
# ============================================================
st.markdown("---")
st.header("📋 Emergency Services Summary")

with st.container(border=True):
    summary_data = {
        'Parameter': [
            'Ambulances',
            'Amb Turnaround (min)',
            'Helicopters',
            'Heli Turnaround (min)',
            'Handover Bays',
            'Handover Time (min)',
            'Walk-in Enabled',
            'Walk-in Scale'
        ],
        'Value': [
            st.session_state.n_ambulances,
            st.session_state.ambulance_turnaround,
            st.session_state.n_helicopters,
            st.session_state.helicopter_turnaround,
            st.session_state.n_handover_bays,
            st.session_state.handover_time_mean,
            'Yes' if st.session_state.walkin_enabled else 'No',
            f"{st.session_state.walkin_rate_multiplier:.1f}x"
        ],
        'FullScenario Attribute': [
            'n_ambulances',
            'ambulance_turnaround_mins',
            'n_helicopters',
            'helicopter_turnaround_mins',
            'n_handover_bays',
            'handover_time_mean',
            'walkin_rate_multiplier > 0',
            'walkin_rate_multiplier'
        ]
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    st.caption("These values are stored in st.session_state and used when building FullScenario for simulation.")

# ============================================================
# NAVIGATION HINT
# ============================================================
st.markdown("---")
st.info("""
**Next Steps:**
- Continue to **Arrival Profiles** section below to configure timing patterns
- Or go to **2_Resources** page to configure ED, diagnostics, and downstream capacity
- **Run** page will combine all settings into a FullScenario for simulation
""")

# ============================================================
# SECTION 4: ARRIVAL PROFILES (Phase 8a-2 content)
# ============================================================
st.markdown("---")
st.header("📊 Arrival Profiles")

st.markdown("""
Configure **when** patients arrive throughout the day.
Choose from three models based on your analysis needs.

**Simulation linkage**: Controls `arrival_generator_multistream()` in full_model.py
""")

# Define base rates - typical A&E arrival pattern (patients/hour)
BASE_HOURLY_RATES = [
    2.5, 1.8, 1.2, 1.2, 1.5, 2.0,  # 00:00 - 05:00 (night trough)
    3.0, 4.5, 6.0, 6.5, 6.0, 5.5,  # 06:00 - 11:00 (morning rise)
    5.0, 5.0, 5.0, 5.0, 5.5, 6.0,  # 12:00 - 17:00 (afternoon plateau)
    5.5, 4.5, 3.5, 3.0, 2.8, 2.8   # 18:00 - 23:00 (evening decline)
]

# Define day type multipliers (local version for UI)
UI_DAY_TYPE_MULTIPLIERS = {
    'weekday': {
        'overall': 1.0,
        'hourly_adjustments': {},
        'description': 'Standard Tue-Thu pattern'
    },
    'monday': {
        'overall': 1.0,
        'hourly_adjustments': {7: 1.2, 8: 1.3, 9: 1.3, 10: 1.2, 11: 1.1},
        'description': 'Morning surge from weekend backlog'
    },
    'friday_eve': {
        'overall': 1.0,
        'hourly_adjustments': {18: 1.2, 19: 1.3, 20: 1.4, 21: 1.4, 22: 1.3, 23: 1.2},
        'description': 'Evening surge - start of weekend'
    },
    'sat_night': {
        'overall': 1.0,
        'hourly_adjustments': {20: 1.3, 21: 1.4, 22: 1.5, 23: 1.5, 0: 1.4, 1: 1.3, 2: 1.2},
        'description': 'Night surge - alcohol, violence, accidents'
    },
    'sunday': {
        'overall': 0.85,
        'hourly_adjustments': {14: 1.1, 15: 1.15, 16: 1.1},
        'description': 'Quieter overall, afternoon family visit discoveries'
    },
    'bank_holiday': {
        'overall': 0.95,
        'hourly_adjustments': {20: 1.3, 21: 1.4, 22: 1.4, 23: 1.3},
        'description': 'Weekend pattern with evening surge'
    },
}

# Profile type selector - maps to FullScenario.arrival_model
arrival_model = st.radio(
    "Select Arrival Model",
    options=['simple', 'profile_24h', 'detailed'],
    format_func=lambda x: {
        'simple': 'Simple - Set daily totals, auto-distribute across 24h',
        'profile_24h': '24-Hour Profile - Standard A&E pattern with day type',
        'detailed': 'Detailed - Hour-by-hour control (12-72h)'
    }[x],
    horizontal=True,
    index=['simple', 'profile_24h', 'detailed'].index(st.session_state.arrival_model),
    key="input_arrival_model"
)
st.session_state.arrival_model = arrival_model

st.markdown("---")

# ============ SIMPLE MODE ============
if arrival_model == 'simple':
    st.subheader("Simple Daily Totals")

    st.info("""
    **How it works**: Set total patients per day for each arrival mode.
    The system applies the standard A&E 24-hour pattern automatically.

    **Best for**: Quick scenario testing, board presentations, sensitivity analysis.

    **Simulation linkage**: Daily totals are distributed across hours using BASE_HOURLY_RATES,
    then scaled by stream multipliers.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        daily_ambulance = st.number_input(
            "🚑 Ambulance Arrivals/Day",
            min_value=0,
            max_value=300,
            value=st.session_state.daily_ambulance,
            key="input_daily_ambulance",
            help="Total ambulance conveyances per 24 hours"
        )
        st.session_state.daily_ambulance = daily_ambulance

        # Validation against fleet capacity
        fleet_capacity = st.session_state.n_ambulances * 24 * (60 / st.session_state.ambulance_turnaround)
        if daily_ambulance > fleet_capacity:
            st.warning(f"Exceeds fleet capacity ({fleet_capacity:.0f}/day)")

    with col2:
        daily_helicopter = st.number_input(
            "🚁 HEMS Arrivals/Day",
            min_value=0,
            max_value=30,
            value=st.session_state.daily_helicopter,
            key="input_daily_helicopter",
            help="Total helicopter arrivals per 24 hours"
        )
        st.session_state.daily_helicopter = daily_helicopter

        # Validation
        if st.session_state.n_helicopters > 0:
            heli_capacity = st.session_state.n_helicopters * 19 * (60 / st.session_state.helicopter_turnaround)
            if daily_helicopter > heli_capacity:
                st.warning(f"Exceeds HEMS capacity ({heli_capacity:.0f}/day)")
        elif daily_helicopter > 0:
            st.error("No HEMS aircraft configured!")

    with col3:
        daily_walkin = st.number_input(
            "🚶 Walk-in Arrivals/Day",
            min_value=0,
            max_value=400,
            value=st.session_state.daily_walkin,
            key="input_daily_walkin",
            help="Total self-presentations per 24 hours"
        )
        st.session_state.daily_walkin = daily_walkin

        if not st.session_state.walkin_enabled and daily_walkin > 0:
            st.warning("Walk-ins disabled in Emergency Services")

    # Total and hourly distribution
    total_daily = daily_ambulance + daily_helicopter + daily_walkin

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Daily Arrivals", total_daily)
    with col2:
        st.metric("Average per Hour", f"{total_daily/24:.1f}")

    # Show hourly distribution preview
    with st.expander("Preview: Hourly Distribution", expanded=False):
        st.caption("Shows how daily totals are distributed using standard A&E pattern")

        # Calculate hourly distribution
        total_base = sum(BASE_HOURLY_RATES)
        hourly_proportions = [r / total_base for r in BASE_HOURLY_RATES]
        hourly_arrivals = [total_daily * p for p in hourly_proportions]

        # Create chart
        chart_df = pd.DataFrame({
            'Hour': [f"{h:02d}:00" for h in range(24)],
            'Expected Arrivals': hourly_arrivals
        })

        fig = px.bar(
            chart_df,
            x='Hour',
            y='Expected Arrivals',
            title='Daily Totals Distributed by Standard A&E Pattern',
            color='Expected Arrivals',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show peak info
        peak_hour = hourly_arrivals.index(max(hourly_arrivals))
        st.caption(f"Peak hour: {peak_hour:02d}:00 with ~{max(hourly_arrivals):.1f} arrivals")

# ============ 24-HOUR PROFILE MODE ============
elif arrival_model == 'profile_24h':
    st.subheader("24-Hour Profile with Day Type")

    st.info("""
    **How it works**: Uses realistic hourly patterns that vary through the day.
    Select a day type to model known NHS patterns (Monday surge, weekend nights).

    **Best for**: Typical day modelling, shift planning, capacity analysis.

    **Simulation linkage**:
    - `scenario.day_type` selects multiplier pattern
    - `scenario.demand_multiplier` scales all arrivals
    - Per-stream multipliers for fine-tuning
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Day type selector - maps to FullScenario.day_type
        day_type_options = list(UI_DAY_TYPE_MULTIPLIERS.keys())
        day_type = st.selectbox(
            "Day Type Preset",
            options=day_type_options,
            format_func=lambda x: {
                'weekday': 'Weekday (Tue-Thu)',
                'monday': 'Monday (+morning surge)',
                'friday_eve': 'Friday Evening (+30% 18:00-23:00)',
                'sat_night': 'Saturday Night (+40% 20:00-02:00)',
                'sunday': 'Sunday (-15% overall)',
                'bank_holiday': 'Bank Holiday'
            }[x],
            index=day_type_options.index(st.session_state.day_type) if st.session_state.day_type in day_type_options else 0,
            key="input_day_type"
        )
        st.session_state.day_type = day_type

        # Show day type description
        st.caption(UI_DAY_TYPE_MULTIPLIERS[day_type]['description'])

    with col2:
        # Demand scale - maps to FullScenario.demand_multiplier
        demand_scale = st.slider(
            "Overall Demand Scale",
            min_value=0.5,
            max_value=2.0,
            value=float(st.session_state.demand_multiplier),
            step=0.1,
            key="input_demand_scale",
            help="Maps to FullScenario.demand_multiplier. Scales all arrivals."
        )
        st.session_state.demand_multiplier = demand_scale

    # Calculate effective hourly rates
    config = UI_DAY_TYPE_MULTIPLIERS[day_type]
    effective_rates = []
    for h, base in enumerate(BASE_HOURLY_RATES):
        rate = base * config['overall']
        rate *= config['hourly_adjustments'].get(h, 1.0)
        rate *= demand_scale
        effective_rates.append(rate)

    # Visualization
    chart_df = pd.DataFrame({
        'Hour': [f"{h:02d}:00" for h in range(24)],
        'Arrivals/hr': effective_rates
    })

    fig = px.bar(
        chart_df,
        x='Hour',
        y='Arrivals/hr',
        title=f'Arrival Profile: {day_type.replace("_", " ").title()} (x{demand_scale})',
        color='Arrivals/hr',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Daily Total", f"{sum(effective_rates):.0f}")
    col2.metric("Peak Hour Rate", f"{max(effective_rates):.1f}/hr")
    col3.metric("Trough Hour Rate", f"{min(effective_rates):.1f}/hr")

    # Per-stream fine tuning
    with st.expander("Per-Stream Scaling (Advanced)", expanded=False):
        st.caption("""
        Adjust individual arrival streams. These multiply with the pattern above.
        Maps to `FullScenario.[ambulance|helicopter|walkin]_rate_multiplier`
        """)

        c1, c2, c3 = st.columns(3)

        with c1:
            amb_mult = st.slider(
                "🚑 Ambulance Scale",
                0.0, 2.0,
                float(st.session_state.ambulance_rate_multiplier),
                0.1,
                key="input_amb_mult_24h"
            )
            st.session_state.ambulance_rate_multiplier = amb_mult

        with c2:
            heli_mult = st.slider(
                "🚁 HEMS Scale",
                0.0, 2.0,
                float(st.session_state.helicopter_rate_multiplier),
                0.1,
                key="input_heli_mult_24h"
            )
            st.session_state.helicopter_rate_multiplier = heli_mult

        with c3:
            walk_mult = st.slider(
                "🚶 Walk-in Scale",
                0.0, 2.0,
                float(st.session_state.walkin_rate_multiplier),
                0.1,
                key="input_walk_mult_24h"
            )
            st.session_state.walkin_rate_multiplier = walk_mult

        if amb_mult != 1.0 or heli_mult != 1.0 or walk_mult != 1.0:
            st.info(f"Stream adjustments active: Amb x{amb_mult}, HEMS x{heli_mult}, Walk-in x{walk_mult}")

# ============ DETAILED MODE ============
elif arrival_model == 'detailed':
    st.subheader("Detailed Hour-by-Hour Control")

    st.info("""
    **How it works**: Set exact arrival numbers for each hour and each arrival type.
    Choose configuration period (12-72 hours). Pattern repeats if run exceeds config.

    **Best for**: Replaying historical data, specific scenario planning, MCI exercises.

    **Simulation linkage**: Creates `DetailedArrivalConfig` stored in
    `FullScenario.detailed_arrivals`. Used when `arrival_model == 'detailed'`.
    """)

    # Configuration period selection
    config_hours = st.select_slider(
        "Configuration Period",
        options=[12, 24, 36, 48, 60, 72],
        value=st.session_state.detailed_config_hours,
        key="input_config_hours",
        help="Number of hours to configure. Pattern repeats for longer runs."
    )
    st.session_state.detailed_config_hours = config_hours

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Configuring {config_hours} hours ({config_hours/24:.1f} days)")
    with col2:
        run_length_hours = st.session_state.get('run_length', 1440) / 60
        if config_hours < run_length_hours:
            repeats = run_length_hours / config_hours
            st.caption(f"Pattern will repeat {repeats:.1f}x for {run_length_hours:.0f}h run")

    # Initialize from default pattern button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Default A&E Pattern", key="btn_apply_default"):
            st.session_state.detailed_use_default = True
            st.rerun()
    with col2:
        if st.button("Reset to Zero", key="btn_reset_zero"):
            st.session_state.detailed_use_default = False
            st.session_state.detailed_arrivals_df = None
            st.rerun()

    # Generate data for table
    def get_default_arrivals(hour: int) -> tuple:
        """Get default arrivals for an hour based on standard pattern."""
        h = hour % 24
        # Proportional split: ~60% ambulance, ~5% HEMS (daytime), ~35% walk-in
        base = BASE_HOURLY_RATES[h]
        amb = int(base * 0.60)
        heli = 1 if 7 <= h <= 18 else 0  # HEMS daytime only
        walk = int(base * 0.35)
        return amb, heli, walk

    # Build dataframe
    hours = list(range(config_hours))

    # Check if we have existing data or need defaults
    use_defaults = st.session_state.get('detailed_use_default', True)
    existing_df = st.session_state.get('detailed_arrivals_df', None)

    data = {
        'Hour': [],
        'Ambulance': [],
        'HEMS': [],
        'Walk-in': []
    }

    for h in hours:
        # Hour label
        if h < 24:
            data['Hour'].append(f"{h:02d}:00")
        else:
            day = h // 24 + 1
            hour_of_day = h % 24
            data['Hour'].append(f"D{day} {hour_of_day:02d}:00")

        # Values
        if existing_df is not None and h < len(existing_df):
            data['Ambulance'].append(int(existing_df.iloc[h]['Ambulance']))
            data['HEMS'].append(int(existing_df.iloc[h]['HEMS']))
            data['Walk-in'].append(int(existing_df.iloc[h]['Walk-in']))
        elif use_defaults:
            amb, heli, walk = get_default_arrivals(h)
            data['Ambulance'].append(amb)
            data['HEMS'].append(heli)
            data['Walk-in'].append(walk)
        else:
            data['Ambulance'].append(0)
            data['HEMS'].append(0)
            data['Walk-in'].append(0)

    df = pd.DataFrame(data)

    # Editable table
    edited_df = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        height=min(500, 35 * min(len(hours), 15) + 50),
        column_config={
            'Hour': st.column_config.TextColumn('Hour', disabled=True, width='small'),
            'Ambulance': st.column_config.NumberColumn('🚑 Amb', min_value=0, max_value=50, width='small'),
            'HEMS': st.column_config.NumberColumn('🚁 HEMS', min_value=0, max_value=10, width='small'),
            'Walk-in': st.column_config.NumberColumn('🚶 Walk', min_value=0, max_value=50, width='small'),
        }
    )

    # Store edited dataframe
    st.session_state.detailed_arrivals_df = edited_df

    # Build DetailedArrivalConfig for session state
    detailed_config = DetailedArrivalConfig(hours_configured=config_hours)
    for idx, row in edited_df.iterrows():
        detailed_config.hourly_counts[idx] = {
            ArrivalMode.AMBULANCE: int(row['Ambulance']),
            ArrivalMode.HELICOPTER: int(row['HEMS']),
            ArrivalMode.SELF_PRESENTATION: int(row['Walk-in']),
        }
    st.session_state.detailed_arrivals = detailed_config

    # Totals
    total_amb = edited_df['Ambulance'].sum()
    total_heli = edited_df['HEMS'].sum()
    total_walk = edited_df['Walk-in'].sum()
    total_all = total_amb + total_heli + total_walk

    st.markdown("**Totals**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚑 Ambulance", int(total_amb))
    col2.metric("🚁 HEMS", int(total_heli))
    col3.metric("🚶 Walk-in", int(total_walk))
    col4.metric("Total", int(total_all))

    # Projected totals for longer runs
    if config_hours < 72:
        with st.expander("Projected Totals for Extended Runs"):
            for target in [24, 48, 72]:
                if target > config_hours:
                    repeats = target / config_hours
                    projected = int(total_all * repeats)
                    st.write(f"**{target}h run**: ~{projected} arrivals ({repeats:.1f}x pattern)")

    # Visualization
    st.markdown("**Arrival Pattern Visualization**")

    plot_df = edited_df.melt(id_vars=['Hour'], var_name='Type', value_name='Count')

    fig = px.bar(
        plot_df,
        x='Hour',
        y='Count',
        color='Type',
        barmode='stack',
        title=f'Arrivals by Hour ({config_hours}h configuration)',
        color_discrete_map={
            'Ambulance': '#FF6B6B',
            'HEMS': '#4ECDC4',
            'Walk-in': '#95E1D3'
        }
    )
    fig.update_layout(xaxis_tickangle=-45 if config_hours > 24 else 0)
    st.plotly_chart(fig, use_container_width=True)

    # Import/Export
    with st.expander("Import/Export Data"):
        col1, col2 = st.columns(2)

        with col1:
            csv_data = edited_df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv_data,
                f"arrivals_{config_hours}h.csv",
                "text/csv",
                key="btn_download_csv"
            )

        with col2:
            uploaded = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                key="upload_arrivals_csv"
            )
            if uploaded:
                try:
                    uploaded_df = pd.read_csv(uploaded)
                    # Validate columns
                    required_cols = {'Hour', 'Ambulance', 'HEMS', 'Walk-in'}
                    if required_cols.issubset(set(uploaded_df.columns)):
                        st.session_state.detailed_arrivals_df = uploaded_df
                        st.session_state.detailed_config_hours = len(uploaded_df)
                        st.success(f"Loaded {len(uploaded_df)} hours of data")
                        st.rerun()
                    else:
                        st.error(f"CSV must have columns: {required_cols}")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")

# ============================================================
# SECTION 5: CONFIGURATION VALIDATION
# ============================================================
st.markdown("---")
st.header("Configuration Validation")

with st.container(border=True):
    st.markdown("**Linking Emergency Services <-> Arrival Profiles**")

    validation_messages = []
    all_valid = True

    # Get current config
    n_amb = st.session_state.n_ambulances
    amb_turn = st.session_state.ambulance_turnaround
    n_heli = st.session_state.n_helicopters
    heli_turn = st.session_state.helicopter_turnaround

    if arrival_model == 'simple':
        # Check ambulance capacity
        daily_amb_capacity = n_amb * 24 * (60 / amb_turn)
        if st.session_state.daily_ambulance > daily_amb_capacity:
            validation_messages.append(
                f"Ambulance arrivals ({st.session_state.daily_ambulance}/day) exceed fleet capacity ({daily_amb_capacity:.0f}/day)"
            )
            all_valid = False
        else:
            validation_messages.append(
                f"Ambulance: {st.session_state.daily_ambulance}/day within fleet capacity ({daily_amb_capacity:.0f}/day)"
            )

        # Check HEMS capacity
        if n_heli > 0:
            daily_heli_capacity = n_heli * 19 * (60 / heli_turn)
            if st.session_state.daily_helicopter > daily_heli_capacity:
                validation_messages.append(
                    f"HEMS arrivals ({st.session_state.daily_helicopter}/day) exceed capacity ({daily_heli_capacity:.0f}/day)"
                )
                all_valid = False
            else:
                validation_messages.append(
                    f"HEMS: {st.session_state.daily_helicopter}/day within capacity ({daily_heli_capacity:.0f}/day)"
                )
        elif st.session_state.daily_helicopter > 0:
            validation_messages.append(
                f"HEMS arrivals configured ({st.session_state.daily_helicopter}/day) but no aircraft available"
            )
            all_valid = False

        # Check walk-in
        if not st.session_state.walkin_enabled and st.session_state.daily_walkin > 0:
            validation_messages.append(
                f"Walk-in arrivals ({st.session_state.daily_walkin}/day) but walk-ins disabled"
            )

    elif arrival_model == 'profile_24h':
        # Check peak hour vs handover capacity
        peak_rate = max(effective_rates)
        handover_capacity = st.session_state.n_handover_bays * (60 / st.session_state.handover_time_mean)

        # Estimate conveyed proportion (ambulance + HEMS typically ~60%)
        conveyed_peak = peak_rate * 0.6 * st.session_state.ambulance_rate_multiplier

        if conveyed_peak > handover_capacity:
            validation_messages.append(
                f"Peak conveyed rate (~{conveyed_peak:.1f}/hr) may exceed handover capacity ({handover_capacity:.1f}/hr)"
            )
        else:
            validation_messages.append(
                f"Peak conveyed rate (~{conveyed_peak:.1f}/hr) within handover capacity ({handover_capacity:.1f}/hr)"
            )

        validation_messages.append(
            f"Day type: {day_type}, Demand scale: x{demand_scale}"
        )

    elif arrival_model == 'detailed':
        # Check hourly peaks
        max_amb_hour = edited_df['Ambulance'].max()
        max_heli_hour = edited_df['HEMS'].max()

        amb_hourly_capacity = n_amb * (60 / amb_turn)
        if max_amb_hour > amb_hourly_capacity:
            validation_messages.append(
                f"Peak ambulance hour ({max_amb_hour}) exceeds hourly fleet capacity ({amb_hourly_capacity:.0f})"
            )
        else:
            validation_messages.append(
                f"Peak ambulance hour ({max_amb_hour}) within fleet capacity ({amb_hourly_capacity:.0f}/hr)"
            )

        if n_heli > 0:
            heli_hourly_capacity = n_heli * (60 / heli_turn)
            if max_heli_hour > heli_hourly_capacity:
                validation_messages.append(
                    f"Peak HEMS hour ({max_heli_hour}) exceeds capacity ({heli_hourly_capacity:.0f}/hr)"
                )

        validation_messages.append(
            f"Configured {config_hours}h with {int(total_all)} total arrivals"
        )

    # Display messages
    for msg in validation_messages:
        if 'exceed' in msg.lower() or 'no aircraft' in msg.lower():
            st.error(msg)
        elif 'may exceed' in msg.lower() or 'disabled' in msg.lower():
            st.warning(msg)
        else:
            st.success(msg)

    if all_valid:
        st.success("All configuration checks passed")
    else:
        st.warning("Review warnings above before running simulation")

# ============================================================
# SECTION 6: ARRIVAL PROFILE SUMMARY
# ============================================================
st.markdown("---")
st.header("📋 Arrival Profile Summary")

with st.container(border=True):
    summary_data = {
        'Parameter': [
            'Arrival Model',
            'Day Type',
            'Demand Multiplier',
            'Ambulance Rate Mult',
            'Helicopter Rate Mult',
            'Walk-in Rate Mult',
        ],
        'Value': [
            st.session_state.arrival_model,
            st.session_state.day_type if st.session_state.arrival_model == 'profile_24h' else 'N/A',
            f"{st.session_state.demand_multiplier:.1f}x",
            f"{st.session_state.ambulance_rate_multiplier:.1f}x",
            f"{st.session_state.helicopter_rate_multiplier:.1f}x",
            f"{st.session_state.walkin_rate_multiplier:.1f}x",
        ],
        'FullScenario Attribute': [
            'arrival_model',
            'day_type',
            'demand_multiplier',
            'ambulance_rate_multiplier',
            'helicopter_rate_multiplier',
            'walkin_rate_multiplier',
        ]
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    if arrival_model == 'detailed':
        st.caption(f"Detailed config: {st.session_state.detailed_config_hours}h, stored in FullScenario.detailed_arrivals")

# ============================================================
# NAVIGATION
# ============================================================
st.markdown("---")
st.info("""
**Configuration Complete!**

**Next Steps:**
- Go to **2_Resources** to configure ED bays, diagnostics, theatre, ITU, and ward capacity
- Go to **3_Schematic** to visualize the full system with your configuration
- Go to **Run** to execute the simulation

All settings are stored in session state and will be combined into a `FullScenario` when you run.
""")
