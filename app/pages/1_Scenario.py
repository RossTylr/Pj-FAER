"""Scenario configuration page for full A&E model."""

import streamlit as st
import pandas as pd

from faer.core.scenario import FullScenario
from faer.core.arrivals import load_default_profile

st.set_page_config(page_title="Scenario - FAER", page_icon="📊", layout="wide")

st.title("📊 Scenario Configuration")

# Initialize session state
if "scenario" not in st.session_state:
    st.session_state.scenario = FullScenario()
if "n_reps" not in st.session_state:
    st.session_state.n_reps = 30

scenario = st.session_state.scenario

# Tabs for different configuration sections
tab_time, tab_resources, tab_acuity, tab_service, tab_experiment = st.tabs([
    "Timing", "Resources", "Acuity Mix", "Service Times", "Experiment"
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

        st.subheader("Resus")
        n_resus = st.slider(
            "Resus bays",
            min_value=1,
            max_value=6,
            value=scenario.n_resus_bays,
            help="Critical care resuscitation bays",
        )

    with col2:
        st.subheader("Majors")
        n_majors = st.slider(
            "Majors bays",
            min_value=2,
            max_value=20,
            value=scenario.n_majors_bays,
            help="Treatment bays for serious conditions",
        )

        st.subheader("Minors")
        n_minors = st.slider(
            "Minors bays",
            min_value=2,
            max_value=15,
            value=scenario.n_minors_bays,
            help="Treatment bays for less urgent cases",
        )

    # Visual summary
    st.subheader("Resource Summary")
    resource_df = pd.DataFrame({
        "Resource": ["Triage", "Resus", "Majors", "Minors"],
        "Capacity": [n_triage, n_resus, n_majors, n_minors],
    })
    st.bar_chart(resource_df.set_index("Resource"), use_container_width=True)

# ===== ACUITY MIX TAB =====
with tab_acuity:
    st.header("Patient Acuity Mix")
    st.markdown("""
    Configure the proportion of patients arriving in each acuity category:
    - **Resus**: Life-threatening, immediate attention required
    - **Majors**: Serious conditions requiring urgent treatment
    - **Minors**: Less urgent, ambulatory patients
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

    # Disposition probabilities
    st.subheader("Admission Probabilities by Acuity")
    st.markdown("Probability a patient is admitted (vs discharged) after treatment:")

    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        p_admit_resus = st.slider(
            "P(Admit | Resus)",
            min_value=0.0,
            max_value=1.0,
            value=scenario.resus_p_admit,
            step=0.05,
            help="~70-90% typically admitted",
        )

    with d_col2:
        p_admit_majors = st.slider(
            "P(Admit | Majors)",
            min_value=0.0,
            max_value=1.0,
            value=scenario.majors_p_admit,
            step=0.05,
            help="~20-40% typically admitted",
        )

    with d_col3:
        p_admit_minors = st.slider(
            "P(Admit | Minors)",
            min_value=0.0,
            max_value=1.0,
            value=scenario.minors_p_admit,
            step=0.05,
            help="~5-15% typically admitted",
        )

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

    st.subheader("Treatment Times by Acuity")

    st.markdown("**Resus** (critical care)")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        resus_mean = st.number_input("Mean Resus time", min_value=30, max_value=300, value=int(scenario.resus_mean))
    with r_col2:
        resus_cv = st.slider("Resus CV", min_value=0.1, max_value=1.5, value=scenario.resus_cv, step=0.1)

    st.markdown("**Majors** (serious conditions)")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        majors_mean = st.number_input("Mean Majors time", min_value=20, max_value=240, value=int(scenario.majors_mean))
    with m_col2:
        majors_cv = st.slider("Majors CV", min_value=0.1, max_value=1.5, value=scenario.majors_cv, step=0.1)

    st.markdown("**Minors** (less urgent)")
    mi_col1, mi_col2 = st.columns(2)
    with mi_col1:
        minors_mean = st.number_input("Mean Minors time", min_value=10, max_value=120, value=int(scenario.minors_mean))
    with mi_col2:
        minors_cv = st.slider("Minors CV", min_value=0.1, max_value=1.5, value=scenario.minors_cv, step=0.1)

    st.subheader("Boarding Time (for admitted patients)")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        boarding_mean = st.number_input("Mean boarding time", min_value=0, max_value=480, value=int(scenario.boarding_mean))
    with b_col2:
        boarding_cv = st.slider("Boarding CV", min_value=0.1, max_value=2.0, value=scenario.boarding_cv, step=0.1)

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
    st.metric("Resus Bays", n_resus)

with summary_col3:
    st.metric("Majors Bays", n_majors)
    st.metric("Minors Bays", n_minors)

with summary_col4:
    st.metric("Replications", n_reps)
    st.metric("Random Seed", random_seed)

# ===== SAVE BUTTON =====
st.divider()
if st.button("💾 Save Scenario", type="primary", use_container_width=True):
    try:
        st.session_state.scenario = FullScenario(
            run_length=run_hours * 60.0,
            warm_up=warm_up_hours * 60.0,
            arrival_rate=6.0,  # Could make configurable
            p_resus=p_resus / 100.0,
            p_majors=p_majors / 100.0,
            p_minors=p_minors / 100.0,
            n_triage=n_triage,
            n_resus_bays=n_resus,
            n_majors_bays=n_majors,
            n_minors_bays=n_minors,
            triage_mean=float(triage_mean),
            triage_cv=triage_cv,
            resus_mean=float(resus_mean),
            resus_cv=resus_cv,
            majors_mean=float(majors_mean),
            majors_cv=majors_cv,
            minors_mean=float(minors_mean),
            minors_cv=minors_cv,
            boarding_mean=float(boarding_mean),
            boarding_cv=boarding_cv,
            resus_p_admit=p_admit_resus,
            majors_p_admit=p_admit_majors,
            minors_p_admit=p_admit_minors,
            random_seed=random_seed,
        )
        st.session_state.n_reps = n_reps
        st.session_state.run_complete = False
        st.success("✅ Scenario saved! Go to **Run** page to execute.")
    except ValueError as e:
        st.error(f"Invalid configuration: {e}")
