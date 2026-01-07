"""Scenario configuration page for full A&E model."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add app directory to path for component imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from faer.core.scenario import (
    FullScenario, DiagnosticConfig, DAY_TYPE_MULTIPLIERS, DetailedArrivalConfig,
)
from faer.core.arrivals import load_default_profile
from faer.core.entities import (
    DiagnosticType, Priority, ArrivalModel, DayType, ArrivalMode,
)
import plotly.express as px
from components.schematic import build_simple_schematic

st.set_page_config(page_title="Scenario - FAER", page_icon="📊", layout="wide")

st.title("📊 Scenario Configuration")

# Initialize session state
if "scenario" not in st.session_state:
    st.session_state.scenario = FullScenario()
if "n_reps" not in st.session_state:
    st.session_state.n_reps = 30

scenario = st.session_state.scenario

# Tabs for different configuration sections
tab_time, tab_resources, tab_acuity, tab_arrivals, tab_service, tab_experiment = st.tabs([
    "Timing", "Resources", "Acuity Mix", "Arrivals", "Service Times", "Experiment"
])

# ===== TIMING TAB =====
with tab_time:
    st.header("Simulation Timing")

    col1, col2 = st.columns(2)

    with col1:
        run_hours = st.slider(
            "Run length (hours)",
            min_value=1,
            max_value=36,
            value=int(scenario.run_length / 60),
            help="How long to simulate (excluding warm-up)",
        )

    with col2:
        warm_up_hours = st.slider(
            "Warm-up period (hours)",
            min_value=0,
            max_value=4,
            value=int(scenario.warm_up / 60),
            help="Initial period excluded from statistics",
        )

    # Arrival profile
    st.subheader("Arrival Profile")
    st.markdown("Default 24-hour A&E arrival pattern (patients per hour):")

    profile = load_default_profile()
    profile_data = []
    for i, (end_time, rate) in enumerate(profile.schedule):
        hour = int(end_time / 60) - 1
        profile_data.append({"Hour": f"{hour:02d}:00", "Rate (per hour)": rate})

    profile_df = pd.DataFrame(profile_data)

    chart_col, data_col = st.columns([2, 1])

    with chart_col:
        st.line_chart(
            profile_df.set_index("Hour")["Rate (per hour)"],
            use_container_width=True,
        )

    with data_col:
        st.dataframe(profile_df, height=300, use_container_width=True)

# ===== RESOURCES TAB =====
with tab_resources:
    st.header("Resource Configuration")

    st.markdown("""
    **Phase 5: Simplified ED**
    - Single ED bay pool with priority queuing (P1-P4)
    - P1 patients are served before P4 patients
    - All patients use the same physical bays
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Triage")
        n_triage = st.slider(
            "Triage clinicians",
            min_value=1,
            max_value=6,
            value=scenario.n_triage,
            help="Includes nurses, ANPs, PAs, or doctors staffing triage",
        )

    with col2:
        st.subheader("ED Bays")
        n_ed_bays = st.slider(
            "ED Bays (total)",
            min_value=5,
            max_value=50,
            value=scenario.n_ed_bays,
            help="Total treatment bays (single pool with priority queuing)",
        )

    # Handover bays (Phase 5b)
    st.subheader("Ambulance Handover")
    st.markdown("""
    Ambulance/helicopter arrivals use handover bays.
    Handover bay is held until ED bay is acquired (feedback mechanism).
    """)

    ho_col1, ho_col2 = st.columns(2)
    with ho_col1:
        n_handover_bays = st.slider(
            "Handover Bays",
            min_value=1,
            max_value=10,
            value=scenario.n_handover_bays,
            help="Number of ambulance handover bays",
        )
    with ho_col2:
        handover_time_mean = st.number_input(
            "Handover Time (min)",
            min_value=5,
            max_value=60,
            value=int(scenario.handover_time_mean),
            help="Mean handover time in minutes",
        )

    # Fleet controls (Phase 5c)
    st.subheader("Fleet")
    st.markdown("""
    Ambulance/helicopter arrivals require available vehicles.
    Vehicles are unavailable during turnaround after patient delivery.
    """)

    fleet_col1, fleet_col2 = st.columns(2)
    with fleet_col1:
        n_ambulances = st.number_input(
            "Ambulances",
            min_value=1,
            max_value=50,
            value=scenario.n_ambulances,
            help="Number of ambulance vehicles",
        )
        ambulance_turnaround = st.number_input(
            "Ambulance Turnaround (min)",
            min_value=15,
            max_value=120,
            value=int(scenario.ambulance_turnaround_mins),
            help="Time ambulance unavailable after delivery",
        )
    with fleet_col2:
        n_helicopters = st.number_input(
            "Helicopters",
            min_value=0,
            max_value=10,
            value=scenario.n_helicopters,
            help="Number of helicopter vehicles",
        )
        helicopter_turnaround = st.number_input(
            "Helicopter Turnaround (min)",
            min_value=30,
            max_value=180,
            value=int(scenario.helicopter_turnaround_mins),
            help="Time helicopter unavailable after delivery",
        )

    # Diagnostics (Phase 7)
    st.subheader("Diagnostics")
    st.markdown("""
    Diagnostic services (CT, X-ray, Bloods) are shared resources that can become bottlenecks.
    Patients queue for diagnostics while holding their ED bay.
    """)

    diag_col1, diag_col2, diag_col3 = st.columns(3)

    with diag_col1:
        st.markdown("**CT Scanner**")
        ct_capacity = st.number_input(
            "CT Scanners",
            min_value=1,
            max_value=5,
            value=scenario.diagnostic_configs[DiagnosticType.CT_SCAN].capacity,
            key="ct_cap",
            help="Number of CT scanners available",
        )
        ct_time = st.number_input(
            "CT Scan time (min)",
            min_value=10,
            max_value=60,
            value=int(scenario.diagnostic_configs[DiagnosticType.CT_SCAN].process_time_mean),
            key="ct_time",
            help="Mean time for CT scan procedure",
        )
        ct_turnaround = st.number_input(
            "CT Report time (min)",
            min_value=10,
            max_value=90,
            value=int(scenario.diagnostic_configs[DiagnosticType.CT_SCAN].turnaround_time_mean),
            key="ct_report",
            help="Mean time for radiologist report",
        )

    with diag_col2:
        st.markdown("**X-ray**")
        xray_capacity = st.number_input(
            "X-ray Rooms",
            min_value=1,
            max_value=10,
            value=scenario.diagnostic_configs[DiagnosticType.XRAY].capacity,
            key="xray_cap",
            help="Number of X-ray rooms available",
        )
        xray_time = st.number_input(
            "X-ray time (min)",
            min_value=5,
            max_value=30,
            value=int(scenario.diagnostic_configs[DiagnosticType.XRAY].process_time_mean),
            key="xray_time",
            help="Mean time for X-ray procedure",
        )
        xray_turnaround = st.number_input(
            "X-ray Report time (min)",
            min_value=5,
            max_value=60,
            value=int(scenario.diagnostic_configs[DiagnosticType.XRAY].turnaround_time_mean),
            key="xray_report",
            help="Mean time for radiologist report",
        )

    with diag_col3:
        st.markdown("**Bloods**")
        bloods_capacity = st.number_input(
            "Phlebotomists",
            min_value=1,
            max_value=10,
            value=scenario.diagnostic_configs[DiagnosticType.BLOODS].capacity,
            key="bloods_cap",
            help="Number of phlebotomists available",
        )
        bloods_time = st.number_input(
            "Blood draw time (min)",
            min_value=2,
            max_value=15,
            value=int(scenario.diagnostic_configs[DiagnosticType.BLOODS].process_time_mean),
            key="bloods_time",
            help="Mean time to take blood sample",
        )
        bloods_turnaround = st.number_input(
            "Lab turnaround (min)",
            min_value=15,
            max_value=90,
            value=int(scenario.diagnostic_configs[DiagnosticType.BLOODS].turnaround_time_mean),
            key="bloods_tat",
            help="Mean time for lab to process bloods",
        )

    # Visual summary
    st.subheader("Resource Summary")
    resource_df = pd.DataFrame({
        "Resource": ["Triage", "ED Bays", "Handover Bays", "Ambulances", "Helicopters"],
        "Capacity": [n_triage, n_ed_bays, n_handover_bays, n_ambulances, n_helicopters],
    })
    st.bar_chart(resource_df.set_index("Resource"), use_container_width=True)

    # System schematic (Phase 5f)
    st.subheader("System Schematic")
    st.caption("Live diagram updates as you adjust capacities")
    try:
        schematic_dot = build_simple_schematic(
            n_ambulances=n_ambulances,
            n_helicopters=n_helicopters,
            n_handover=n_handover_bays,
            n_triage=n_triage,
            n_ed_bays=n_ed_bays,
        )
        st.graphviz_chart(schematic_dot, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render schematic: {e}")

# ===== ACUITY MIX TAB =====
with tab_acuity:
    st.header("Patient Acuity Mix")
    st.markdown("""
    Configure the proportion of patients arriving in each clinical acuity category.
    Acuity determines the **priority level (P1-P4)** used for queuing:

    | Acuity | Description | Priority Assignment |
    |--------|-------------|---------------------|
    | **Resus** | Life-threatening, immediate | Always P1 |
    | **Majors** | Serious, urgent treatment | 70% P2, 30% P3 |
    | **Minors** | Less urgent, ambulatory | 60% P3, 40% P4 |
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        p_resus = st.slider(
            "% Resus",
            min_value=1,
            max_value=20,
            value=int(scenario.p_resus * 100),
            help="Typically 3-8% of ED arrivals",
        )

    with col2:
        p_majors = st.slider(
            "% Majors",
            min_value=20,
            max_value=80,
            value=int(scenario.p_majors * 100),
            help="Typically 40-60% of ED arrivals",
        )

    with col3:
        p_minors = 100 - p_resus - p_majors
        st.metric("% Minors", f"{p_minors}%")
        st.caption("(Calculated from remainder)")

    # Validate
    if p_minors < 0:
        st.error("Resus + Majors cannot exceed 100%!")
        p_minors = 0

    # Pie chart
    st.subheader("Acuity Distribution")
    acuity_df = pd.DataFrame({
        "Acuity": ["Resus", "Majors", "Minors"],
        "Percentage": [p_resus, p_majors, p_minors],
    })

    fig = px.pie(acuity_df, values="Percentage", names="Acuity",
                 color="Acuity",
                 color_discrete_map={"Resus": "#ff4b4b", "Majors": "#ffa62b", "Minors": "#29b09d"})
    st.plotly_chart(fig, use_container_width=True)

    # Priority distribution info
    st.subheader("Priority Queue Behaviour")
    st.markdown("""
    **P1 patients bypass triage** and go directly to ED bays with highest priority.

    When multiple patients are waiting, **lower priority numbers are served first**:
    - P1 (Immediate) → P2 (Very Urgent) → P3 (Urgent) → P4 (Standard)
    """)

    # Disposition routing info
    st.subheader("Disposition Routing (Phase 5)")
    st.markdown("""
    Patients are routed based on **priority level** after ED treatment:

    | Priority | Surgery | ITU | Ward | Exit |
    |----------|---------|-----|------|------|
    | P1 (Immediate) | 30% | 40% | 20% | 10% |
    | P2 (Very Urgent) | 15% | 10% | 45% | 30% |
    | P3 (Urgent) | 5% | 2% | 25% | 68% |
    | P4 (Standard) | 2% | 0% | 5% | 93% |

    *Routing probabilities are configured in the scenario's routing matrix.*
    """)

# ===== ARRIVALS TAB =====
with tab_arrivals:
    st.header("Arrival Configuration")

    # Model selector with plain language explanations
    st.subheader("Choose Arrival Model")

    arrival_model = st.radio(
        "How do you want to configure arrivals?",
        options=[m.value for m in ArrivalModel],
        format_func=lambda x: {
            'simple': 'Simple - Just set a demand level',
            'profile_24h': '24-Hour Profile - Hourly patterns with day type',
            'detailed': 'Detailed - Set exact numbers per hour per mode'
        }[x],
        horizontal=True,
        index=['simple', 'profile_24h', 'detailed'].index(
            st.session_state.get('arrival_model', 'profile_24h')
        )
    )
    st.session_state.arrival_model = arrival_model

    st.markdown("---")

    # ============ SIMPLE MODEL ============
    if arrival_model == 'simple':
        st.info("""
        **Simple Mode**

        Set an overall demand level. The system uses typical arrival patterns
        scaled up or down.

        **Best for**: Quick "what-if" testing, board presentations
        """)

        demand_preset = st.select_slider(
            "Demand Level",
            options=['Low', 'Normal', 'Busy', 'Surge', 'Major Incident'],
            value=st.session_state.get('demand_preset', 'Normal'),
            help="Low=0.75x, Normal=1.0x, Busy=1.25x, Surge=1.5x, Major Incident=2.0x"
        )
        st.session_state.demand_preset = demand_preset

        demand_map = {'Low': 0.75, 'Normal': 1.0, 'Busy': 1.25, 'Surge': 1.5, 'Major Incident': 2.0}
        st.session_state.demand_mult = demand_map[demand_preset]

        col1, col2 = st.columns(2)
        col1.metric("Demand Multiplier", f"{st.session_state.demand_mult}x")

        # Show expected arrivals
        base_daily = 120  # Approximate baseline
        col2.metric("Expected Daily Arrivals", f"~{int(base_daily * st.session_state.demand_mult)}")

    # ============ 24-HOUR PROFILE MODEL ============
    elif arrival_model == 'profile_24h':
        st.info("""
        **24-Hour Profile Mode**

        Uses realistic hourly patterns that vary through the day.
        Choose a day type to model known patterns.

        **Best for**: Typical day modelling, shift planning, capacity analysis
        """)

        col1, col2 = st.columns(2)

        with col1:
            day_type = st.selectbox(
                "Day Type",
                options=[d.value for d in DayType],
                format_func=lambda x: {
                    'weekday': 'Weekday (Tue-Thu) - Standard pattern',
                    'monday': 'Monday - Morning surge (+20% 7-11am)',
                    'friday_eve': 'Friday Evening - +30% after 6pm',
                    'sat_night': 'Saturday Night - +40% 8pm-2am',
                    'sunday': 'Sunday - Quieter (-15%), afternoon peak',
                    'bank_holiday': 'Bank Holiday - Weekend + evening surge'
                }[x],
                index=[d.value for d in DayType].index(
                    st.session_state.get('day_type', 'weekday')
                )
            )
            st.session_state.day_type = day_type

        with col2:
            demand_fine_tune = st.slider(
                "Additional scaling",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.get('demand_mult', 1.0),
                step=0.1,
                help="Extra multiplier on top of day type pattern"
            )
            st.session_state.demand_mult = demand_fine_tune

        # Show the resulting hourly profile
        st.subheader("Resulting Arrival Profile")

        # Calculate effective rates
        base_rates = [2, 1.5, 1, 1, 1.5, 2, 3, 4, 5, 5.5, 5, 4.5,
                    4, 4, 4, 4.5, 5, 5.5, 5, 4, 3, 2.5, 2, 2]
        day_config = DAY_TYPE_MULTIPLIERS[DayType(day_type)]

        effective_rates = []
        for hour, base in enumerate(base_rates):
            rate = base * day_config['overall']
            rate *= day_config['hourly_adjustments'].get(hour, 1.0)
            rate *= demand_fine_tune
            effective_rates.append(rate)

        # Create chart
        chart_df = pd.DataFrame({
            'Hour': [f"{h:02d}:00" for h in range(24)],
            'Arrivals/hr': effective_rates
        })

        fig = px.bar(
            chart_df,
            x='Hour',
            y='Arrivals/hr',
            title=f'Expected Arrivals by Hour ({day_type})',
            color='Arrivals/hr',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Daily total
        st.metric("Expected Daily Total", f"~{int(sum(effective_rates))}")

        # Per-stream fine tuning
        with st.expander("Fine-tune by arrival type"):
            st.caption("Adjust individual streams (multiplied with pattern above)")

            ft_col1, ft_col2, ft_col3 = st.columns(3)
            with ft_col1:
                ambulance_rate_mult = st.slider(
                    "Ambulance", 0.0, 2.0,
                    st.session_state.get('ambulance_rate_mult', 1.0), 0.1,
                    key="amb_mult_24h"
                )
                st.session_state.ambulance_rate_mult = ambulance_rate_mult
            with ft_col2:
                helicopter_rate_mult = st.slider(
                    "Helicopter", 0.0, 2.0,
                    st.session_state.get('helicopter_rate_mult', 1.0), 0.1,
                    key="heli_mult_24h"
                )
                st.session_state.helicopter_rate_mult = helicopter_rate_mult
            with ft_col3:
                walkin_rate_mult = st.slider(
                    "Walk-in", 0.0, 2.0,
                    st.session_state.get('walkin_rate_mult', 1.0), 0.1,
                    key="walk_mult_24h"
                )
                st.session_state.walkin_rate_mult = walkin_rate_mult

    # ============ DETAILED MODEL ============
    elif arrival_model == 'detailed':
        st.info("""
        **Detailed Mode**

        Set exact arrival numbers for each hour and each arrival type.

        **Best for**: Replaying historical data, modelling specific scenarios,
        major incident planning
        """)

        st.subheader("Enter Arrivals per Hour")

        # Create editable dataframe with sensible defaults
        hours = list(range(24))

        default_amb = [2, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 4, 4, 4, 4, 4, 5, 5, 5, 4, 3, 3, 2, 2]
        default_heli = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        default_walk = [1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 3, 2, 2, 1, 1]

        # Load from session state if exists
        if 'detailed_arrivals_df' in st.session_state:
            df = st.session_state.detailed_arrivals_df
        else:
            df = pd.DataFrame({
                'Hour': [f"{h:02d}:00" for h in hours],
                'Ambulances': default_amb,
                'Helicopters': default_heli,
                'Walk-ins': default_walk
            })

        edited_df = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Hour': st.column_config.TextColumn('Hour', disabled=True, width='small'),
                'Ambulances': st.column_config.NumberColumn('Ambulances', min_value=0, max_value=30, width='medium'),
                'Helicopters': st.column_config.NumberColumn('Helicopters', min_value=0, max_value=10, width='medium'),
                'Walk-ins': st.column_config.NumberColumn('Walk-ins', min_value=0, max_value=30, width='medium'),
            },
            height=400
        )
        st.session_state.detailed_arrivals_df = edited_df

        # Store in session state as DetailedArrivalConfig
        detailed_config = DetailedArrivalConfig()
        for idx, row in edited_df.iterrows():
            hour = idx
            detailed_config.hourly_counts[hour] = {
                ArrivalMode.AMBULANCE: int(row['Ambulances']),
                ArrivalMode.HELICOPTER: int(row['Helicopters']),
                ArrivalMode.SELF_PRESENTATION: int(row['Walk-ins']),
            }
        st.session_state.detailed_arrivals = detailed_config

        # Show totals
        total_amb = edited_df['Ambulances'].sum()
        total_heli = edited_df['Helicopters'].sum()
        total_walk = edited_df['Walk-ins'].sum()

        st.subheader("Summary")

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Ambulances", int(total_amb))
        sum_col2.metric("Helicopters", int(total_heli))
        sum_col3.metric("Walk-ins", int(total_walk))
        sum_col4.metric("Total", int(total_amb + total_heli + total_walk))

        # Stacked bar visualization
        st.subheader("Arrival Pattern")

        plot_df = edited_df.melt(
            id_vars=['Hour'],
            var_name='Type',
            value_name='Count'
        )

        fig = px.bar(
            plot_df,
            x='Hour',
            y='Count',
            color='Type',
            title='Arrivals by Hour and Type',
            barmode='stack',
            color_discrete_map={
                'Ambulances': '#FF6B6B',
                'Helicopters': '#4ECDC4',
                'Walk-ins': '#95E1D3'
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Import/Export
        with st.expander("Import/Export Data"):
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.download_button(
                    "Download Current as CSV",
                    edited_df.to_csv(index=False),
                    "arrival_config.csv",
                    "text/csv"
                )

            with exp_col2:
                uploaded = st.file_uploader("Upload CSV", type=['csv'])
                if uploaded:
                    try:
                        uploaded_df = pd.read_csv(uploaded)
                        st.success("File loaded! Data will be applied on next refresh.")
                        st.dataframe(uploaded_df, height=150)
                        st.session_state.detailed_arrivals_df = uploaded_df
                    except Exception as e:
                        st.error(f"Error loading file: {e}")

    # Set defaults for stream multipliers if not in detailed mode
    if arrival_model != 'profile_24h':
        # Ensure defaults exist
        if 'ambulance_rate_mult' not in st.session_state:
            st.session_state.ambulance_rate_mult = 1.0
        if 'helicopter_rate_mult' not in st.session_state:
            st.session_state.helicopter_rate_mult = 1.0
        if 'walkin_rate_mult' not in st.session_state:
            st.session_state.walkin_rate_mult = 1.0
        ambulance_rate_mult = st.session_state.ambulance_rate_mult
        helicopter_rate_mult = st.session_state.helicopter_rate_mult
        walkin_rate_mult = st.session_state.walkin_rate_mult

    # Multi-stream toggle (keep for backwards compatibility)
    st.markdown("---")
    use_multistream = st.toggle(
        "Use multi-stream arrivals (ambulance, helicopter, walk-in)",
        value=st.session_state.get("use_multistream", True),
        help="Enable separate arrival streams for Ambulance, Helicopter, and Walk-in"
    )
    st.session_state.use_multistream = use_multistream

    if not use_multistream:
        st.caption("""
        **Single-stream mode**: All patients arrive via a single Poisson process.
        Useful for simpler scenarios or legacy compatibility.
        """)

# ===== SERVICE TIMES TAB =====
with tab_service:
    st.header("Service Time Parameters")
    st.markdown("All times in minutes. CV = Coefficient of Variation (higher = more variable).")

    st.subheader("Triage")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        triage_mean = st.number_input("Mean triage time", min_value=1, max_value=30, value=int(scenario.triage_mean))
    with t_col2:
        triage_cv = st.slider("Triage CV", min_value=0.1, max_value=1.0, value=scenario.triage_cv, step=0.1)

    st.subheader("ED Treatment Time")
    st.markdown("""
    **Phase 5: Single ED pool**
    All patients use the same ED bays. Priority determines queue position, not service time.
    """)

    ed_col1, ed_col2 = st.columns(2)
    with ed_col1:
        ed_service_mean = st.number_input("Mean ED treatment time", min_value=20, max_value=240, value=int(scenario.ed_service_mean))
    with ed_col2:
        ed_service_cv = st.slider("ED service CV", min_value=0.1, max_value=1.5, value=scenario.ed_service_cv, step=0.1)

    st.subheader("Boarding Time (for admitted patients)")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        boarding_mean = st.number_input("Mean boarding time", min_value=0, max_value=480, value=int(scenario.boarding_mean))
    with b_col2:
        boarding_cv = st.slider("Boarding CV", min_value=0.1, max_value=2.0, value=scenario.boarding_cv, step=0.1)

    st.subheader("Bed Turnaround (Phase 5e)")
    st.markdown("""
    Time to clean and prepare a bed after patient departure.
    Bed is unavailable during turnaround.
    """)
    bed_turnaround = st.slider(
        "Bed turnaround time (min)",
        min_value=5,
        max_value=30,
        value=int(scenario.bed_turnaround_mins),
        help="Cleaning time after patient leaves",
    )

# ===== EXPERIMENT TAB =====
with tab_experiment:
    st.header("Experiment Settings")

    col1, col2 = st.columns(2)

    with col1:
        n_reps = st.slider(
            "Number of replications",
            min_value=5,
            max_value=100,
            value=st.session_state.n_reps,
            help="More replications = tighter confidence intervals",
        )

    with col2:
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=scenario.random_seed,
            help="For reproducibility - same seed gives same results",
        )

# ===== SCENARIO SUMMARY =====
st.divider()
st.header("Scenario Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Run Duration", f"{run_hours}h")
    st.metric("Warm-up", f"{warm_up_hours}h")

with summary_col2:
    st.metric("Triage Clinicians", n_triage)
    st.metric("ED Bays", n_ed_bays)

with summary_col3:
    st.metric("ED Treatment", f"{ed_service_mean} min")
    st.metric("Boarding Time", f"{boarding_mean} min")

with summary_col4:
    st.metric("Replications", n_reps)
    st.metric("Random Seed", random_seed)

# ===== SAVE BUTTON =====
st.divider()
if st.button("Save Scenario", type="primary", use_container_width=True):
    try:
        # Build diagnostic configs from UI values (Phase 7)
        diag_configs = {
            DiagnosticType.CT_SCAN: DiagnosticConfig(
                diagnostic_type=DiagnosticType.CT_SCAN,
                capacity=ct_capacity,
                process_time_mean=float(ct_time),
                turnaround_time_mean=float(ct_turnaround),
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.70,
                    Priority.P2_VERY_URGENT: 0.40,
                    Priority.P3_URGENT: 0.15,
                    Priority.P4_STANDARD: 0.05,
                }
            ),
            DiagnosticType.XRAY: DiagnosticConfig(
                diagnostic_type=DiagnosticType.XRAY,
                capacity=xray_capacity,
                process_time_mean=float(xray_time),
                turnaround_time_mean=float(xray_turnaround),
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.30,
                    Priority.P2_VERY_URGENT: 0.35,
                    Priority.P3_URGENT: 0.40,
                    Priority.P4_STANDARD: 0.25,
                }
            ),
            DiagnosticType.BLOODS: DiagnosticConfig(
                diagnostic_type=DiagnosticType.BLOODS,
                capacity=bloods_capacity,
                process_time_mean=float(bloods_time),
                turnaround_time_mean=float(bloods_turnaround),
                probability_by_priority={
                    Priority.P1_IMMEDIATE: 0.90,
                    Priority.P2_VERY_URGENT: 0.80,
                    Priority.P3_URGENT: 0.50,
                    Priority.P4_STANDARD: 0.20,
                }
            ),
        }

        # Get arrival model settings from session state (Phase 7d)
        arrival_model_value = st.session_state.get('arrival_model', 'profile_24h')
        day_type_value = st.session_state.get('day_type', 'weekday')
        demand_mult_value = st.session_state.get('demand_mult', 1.0)

        st.session_state.scenario = FullScenario(
            run_length=run_hours * 60.0,
            warm_up=warm_up_hours * 60.0,
            arrival_rate=6.0,  # Base rate for single-stream
            p_resus=p_resus / 100.0,
            p_majors=p_majors / 100.0,
            p_minors=p_minors / 100.0,
            n_triage=n_triage,
            n_ed_bays=n_ed_bays,
            n_handover_bays=n_handover_bays,
            handover_time_mean=float(handover_time_mean),
            n_ambulances=n_ambulances,
            n_helicopters=n_helicopters,
            ambulance_turnaround_mins=float(ambulance_turnaround),
            helicopter_turnaround_mins=float(helicopter_turnaround),
            demand_multiplier=demand_mult_value,  # From arrivals tab
            ambulance_rate_multiplier=st.session_state.get('ambulance_rate_mult', 1.0),
            helicopter_rate_multiplier=st.session_state.get('helicopter_rate_mult', 1.0),
            walkin_rate_multiplier=st.session_state.get('walkin_rate_mult', 1.0),
            bed_turnaround_mins=float(bed_turnaround),
            triage_mean=float(triage_mean),
            triage_cv=triage_cv,
            ed_service_mean=float(ed_service_mean),
            ed_service_cv=ed_service_cv,
            boarding_mean=float(boarding_mean),
            boarding_cv=boarding_cv,
            random_seed=random_seed,
            diagnostic_configs=diag_configs,  # Phase 7
            arrival_model=ArrivalModel(arrival_model_value),  # Phase 7d
            day_type=DayType(day_type_value),  # Phase 7d
            detailed_arrivals=st.session_state.get('detailed_arrivals'),  # Phase 7d
        )
        st.session_state.n_reps = n_reps
        st.session_state.run_complete = False
        st.success("Scenario saved! Go to **Run** page to execute.")
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
