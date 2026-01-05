"""Run simulation page for full A&E model."""

import time
import streamlit as st
import numpy as np

from faer.core.scenario import FullScenario
from faer.experiment.runner import multiple_replications

st.set_page_config(page_title="Run - FAER", page_icon="▶️", layout="wide")

st.title("▶️ Run Simulation")

# Check for scenario
if "scenario" not in st.session_state:
    st.warning("⚠️ Please configure a scenario first.")
    st.page_link("pages/1_Scenario.py", label="Go to Scenario Configuration", icon="📊")
    st.stop()

scenario = st.session_state.scenario
n_reps = st.session_state.get("n_reps", 30)
use_multistream = st.session_state.get("use_multistream", False)

# Scenario summary
st.header("Current Scenario")

# First row - timing and resources
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Duration", f"{scenario.run_length / 60:.0f}h")
    st.metric("Warm-up", f"{scenario.warm_up / 60:.0f}h")

with col2:
    st.metric("Triage Nurses", scenario.n_triage)
    st.metric("Resus Bays", scenario.n_resus_bays)

with col3:
    st.metric("Majors Bays", scenario.n_majors_bays)
    st.metric("Minors Bays", scenario.n_minors_bays)

with col4:
    st.metric("Replications", n_reps)
    st.metric("Random Seed", scenario.random_seed)

# Arrival mode
if use_multistream:
    st.info("🚑 Multi-stream arrivals enabled (Ambulance, Helicopter, Walk-in)")
else:
    st.info(f"📊 Single-stream arrivals at {scenario.arrival_rate:.1f} patients/hour")

# Second row - acuity mix
st.subheader("Acuity Mix")
acuity_cols = st.columns(3)

with acuity_cols[0]:
    st.metric("Resus", f"{scenario.p_resus:.0%}")

with acuity_cols[1]:
    st.metric("Majors", f"{scenario.p_majors:.0%}")

with acuity_cols[2]:
    st.metric("Minors", f"{scenario.p_minors:.0%}")

st.divider()

# Run button
if st.button("🚀 Run Experiment", type="primary", use_container_width=True):
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
        use_multistream=use_multistream,
    )

    elapsed = time.time() - start_time

    # Store results
    st.session_state.results = results
    st.session_state.run_complete = True
    st.session_state.run_time = elapsed

    # Complete
    progress_bar.progress(1.0)
    status_text.empty()

    st.success(f"✅ Completed {n_reps} replications in {elapsed:.1f} seconds")

    # Show quick summary
    st.header("Quick Summary")

    # Row 1: Key metrics
    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric("Mean Arrivals", f"{np.mean(results['arrivals']):.0f}")

    with summary_cols[1]:
        st.metric("Mean System Time", f"{np.mean(results['mean_system_time']):.1f} min")

    with summary_cols[2]:
        st.metric("P(Delay)", f"{np.mean(results['p_delay']):.1%}")

    with summary_cols[3]:
        st.metric("Admission Rate", f"{np.mean(results['admission_rate']):.1%}")

    # Row 2: Utilisation
    st.subheader("Resource Utilisation")
    util_cols = st.columns(4)

    with util_cols[0]:
        st.metric("Triage", f"{np.mean(results['util_triage']):.1%}")

    with util_cols[1]:
        st.metric("Resus", f"{np.mean(results['util_resus']):.1%}")

    with util_cols[2]:
        st.metric("Majors", f"{np.mean(results['util_majors']):.1%}")

    with util_cols[3]:
        st.metric("Minors", f"{np.mean(results['util_minors']):.1%}")

    st.info("📈 Go to **Results** page for detailed analysis with confidence intervals.")

# Show previous results if they exist
elif st.session_state.get("run_complete"):
    st.info(f"✅ Previous run completed in {st.session_state.get('run_time', 0):.1f}s. "
            "Click **Run Experiment** to re-run, or view **Results**.")

    results = st.session_state.results

    # Row 1: Key metrics
    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric("Mean Arrivals", f"{np.mean(results['arrivals']):.0f}")

    with summary_cols[1]:
        st.metric("Mean System Time", f"{np.mean(results['mean_system_time']):.1f} min")

    with summary_cols[2]:
        st.metric("P(Delay)", f"{np.mean(results['p_delay']):.1%}")

    with summary_cols[3]:
        st.metric("Admission Rate", f"{np.mean(results['admission_rate']):.1%}")

    # Row 2: Utilisation
    st.subheader("Resource Utilisation")
    util_cols = st.columns(4)

    with util_cols[0]:
        st.metric("Triage", f"{np.mean(results['util_triage']):.1%}")

    with util_cols[1]:
        st.metric("Resus", f"{np.mean(results['util_resus']):.1%}")

    with util_cols[2]:
        st.metric("Majors", f"{np.mean(results['util_majors']):.1%}")

    with util_cols[3]:
        st.metric("Minors", f"{np.mean(results['util_minors']):.1%}")
