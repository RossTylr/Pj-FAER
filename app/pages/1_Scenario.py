"""Scenario configuration page."""

import streamlit as st
import pandas as pd

from faer.core.scenario import Scenario
from faer.core.arrivals import load_default_profile

st.set_page_config(page_title="Scenario - FAER", page_icon="📊", layout="wide")

st.title("📊 Scenario Configuration")

# Initialize session state
if "scenario" not in st.session_state:
    st.session_state.scenario = Scenario()
if "n_reps" not in st.session_state:
    st.session_state.n_reps = 30

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.header("Simulation Horizon")
    run_hours = st.slider(
        "Run length (hours)",
        min_value=1,
        max_value=36,
        value=int(st.session_state.scenario.run_length / 60),
        help="How long to simulate (1-36 hours)",
    )

    st.header("Resources")
    n_resus = st.slider(
        "Resus bays",
        min_value=1,
        max_value=10,
        value=st.session_state.scenario.n_resus_bays,
        help="Number of Resuscitation bays available",
    )

with col2:
    st.header("Service Times")
    resus_mean = st.number_input(
        "Mean Resus LoS (minutes)",
        min_value=10,
        max_value=300,
        value=int(st.session_state.scenario.resus_mean),
        help="Average length of stay in Resus",
    )

    resus_cv = st.slider(
        "Resus LoS variability (CV)",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.scenario.resus_cv,
        step=0.1,
        help="Coefficient of variation - higher means more variable LoS",
    )

# Arrival profile section
st.header("Arrival Profile")
st.markdown("Default 24-hour A&E arrival pattern (patients per hour):")

# Load and display default profile
profile = load_default_profile()
profile_data = []
for i, (end_time, rate) in enumerate(profile.schedule):
    hour = int(end_time / 60) - 1
    profile_data.append({"Hour": f"{hour:02d}:00", "Rate (per hour)": rate})

profile_df = pd.DataFrame(profile_data)

# Create two columns for chart and data
chart_col, data_col = st.columns([2, 1])

with chart_col:
    st.line_chart(
        profile_df.set_index("Hour")["Rate (per hour)"],
        use_container_width=True,
    )

with data_col:
    st.dataframe(profile_df, height=300, use_container_width=True)

# Experiment settings
st.header("Experiment Settings")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    n_reps = st.slider(
        "Number of replications",
        min_value=5,
        max_value=100,
        value=st.session_state.n_reps,
        help="More replications = tighter confidence intervals",
    )

with exp_col2:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=st.session_state.scenario.random_seed,
        help="For reproducibility - same seed gives same results",
    )

# Summary panel
st.header("Scenario Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Simulation Duration", f"{run_hours} hours")
    st.metric("Resus Bays", n_resus)

with summary_col2:
    st.metric("Mean LoS", f"{resus_mean} min")
    st.metric("LoS Variability (CV)", f"{resus_cv:.1f}")

with summary_col3:
    st.metric("Replications", n_reps)
    st.metric("Random Seed", random_seed)

# Save button
st.divider()
if st.button("💾 Save Scenario", type="primary", use_container_width=True):
    st.session_state.scenario = Scenario(
        run_length=run_hours * 60.0,  # Convert to minutes
        n_resus_bays=n_resus,
        resus_mean=float(resus_mean),
        resus_cv=resus_cv,
        random_seed=random_seed,
    )
    st.session_state.n_reps = n_reps
    st.session_state.run_complete = False  # Reset results
    st.success("✅ Scenario saved! Go to **Run** page to execute.")
