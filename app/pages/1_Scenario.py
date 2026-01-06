"""Scenario configuration page for full A&E model."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add app directory to path for component imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from faer.core.scenario import FullScenario
from faer.core.arrivals import load_default_profile
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
            "Triage nurses",
            min_value=1,
            max_value=6,
            value=scenario.n_triage,
            help="Number of triage assessment stations",
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

    import plotly.express as px
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

    # Multi-stream toggle
    use_multistream = st.toggle(
        "Use multi-stream arrivals",
        value=st.session_state.get("use_multistream", False),
        help="Enable separate arrival streams for Ambulance, Helicopter, and Walk-in"
    )
    st.session_state.use_multistream = use_multistream

    if use_multistream:
        st.markdown("""
        **Multi-stream arrivals enabled.** Patients arrive via three channels:

        | Stream | Volume | Typical Acuity |
        |--------|--------|----------------|
        | 🚑 Ambulance | Moderate | Higher (P1/P2) |
        | 🚁 Helicopter | Low | Critical (P1) |
        | 🚶 Walk-in | High | Lower (P3/P4) |

        Each stream has its own time-varying arrival pattern and priority mix.
        """)

        # Show default arrival patterns
        st.subheader("Default Arrival Patterns (per hour)")

        amb_rates = [2, 1.5, 1, 1, 1.5, 2, 3, 4, 5, 5.5, 5, 4.5,
                    4, 4, 4, 4.5, 5, 5.5, 5, 4, 3, 2.5, 2, 2]
        heli_rates = [0.1] * 24
        walk_rates = [1, 0.5, 0.3, 0.2, 0.3, 0.5, 1, 2, 3, 4, 4.5, 4,
                     3.5, 3, 3, 3.5, 4, 4.5, 4, 3, 2, 1.5, 1, 1]

        arrival_df = pd.DataFrame({
            "Hour": list(range(24)),
            "Ambulance": amb_rates,
            "Helicopter": heli_rates,
            "Walk-in": walk_rates,
        }).set_index("Hour")

        st.line_chart(arrival_df)

    else:
        st.markdown("""
        **Single-stream arrivals** (legacy mode).
        All patients arrive via a single Poisson process with the configured arrival rate.
        """)

        arrival_rate = st.slider(
            "Arrival rate (patients/hour)",
            min_value=1.0,
            max_value=20.0,
            value=scenario.arrival_rate,
            step=0.5,
            help="Average number of patients arriving per hour",
        )
        st.session_state.arrival_rate = arrival_rate

    # Demand Scaling (Phase 5d)
    st.subheader("Demand Scaling")
    st.markdown("""
    Scale arrival rates to simulate surge scenarios or quiet periods.
    """)

    # Presets
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("Normal (1.0x)", use_container_width=True):
            st.session_state.demand_mult = 1.0
    with preset_col2:
        if st.button("Busy (+25%)", use_container_width=True):
            st.session_state.demand_mult = 1.25
    with preset_col3:
        if st.button("Surge (+50%)", use_container_width=True):
            st.session_state.demand_mult = 1.5

    demand_multiplier = st.slider(
        "Overall Demand Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.get("demand_mult", scenario.demand_multiplier),
        step=0.1,
        help="Scales ALL arrival streams",
    )
    st.session_state.demand_mult = demand_multiplier

    st.caption("Per-stream fine tuning (applied on top of overall multiplier):")
    scale_col1, scale_col2, scale_col3 = st.columns(3)
    with scale_col1:
        ambulance_rate_mult = st.slider(
            "Ambulance",
            min_value=0.0,
            max_value=2.0,
            value=scenario.ambulance_rate_multiplier,
            step=0.1,
        )
    with scale_col2:
        helicopter_rate_mult = st.slider(
            "Helicopter",
            min_value=0.0,
            max_value=2.0,
            value=scenario.helicopter_rate_multiplier,
            step=0.1,
        )
    with scale_col3:
        walkin_rate_mult = st.slider(
            "Walk-in",
            min_value=0.0,
            max_value=2.0,
            value=scenario.walkin_rate_multiplier,
            step=0.1,
        )

    # Routing information
    st.subheader("Patient Routing")
    st.markdown("""
    After ED treatment, patients are routed based on priority:

    | From | P1 Routing | P2/P3/P4 Routing |
    |------|------------|------------------|
    | Resus | Surgery 30%, ITU 40%, Ward 20%, Exit 10% | Ward 45%, Exit 30%, Surgery 15%, ITU 10% |
    | Majors | - | Ward 25-45%, Exit 30-75% |
    | Minors | - | Ward 5-10%, Exit 90-95% |

    Downstream nodes: Surgery -> ITU/Ward, ITU -> Ward, Ward -> Exit
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
    st.metric("Triage Nurses", n_triage)
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
        st.session_state.scenario = FullScenario(
            run_length=run_hours * 60.0,
            warm_up=warm_up_hours * 60.0,
            arrival_rate=6.0,  # Could make configurable
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
            demand_multiplier=demand_multiplier,
            ambulance_rate_multiplier=ambulance_rate_mult,
            helicopter_rate_multiplier=helicopter_rate_mult,
            walkin_rate_multiplier=walkin_rate_mult,
            bed_turnaround_mins=float(bed_turnaround),
            triage_mean=float(triage_mean),
            triage_cv=triage_cv,
            ed_service_mean=float(ed_service_mean),
            ed_service_cv=ed_service_cv,
            boarding_mean=float(boarding_mean),
            boarding_cv=boarding_cv,
            random_seed=random_seed,
        )
        st.session_state.n_reps = n_reps
        st.session_state.run_complete = False
        st.success("Scenario saved! Go to **Run** page to execute.")
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
