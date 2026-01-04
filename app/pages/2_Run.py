"""Run simulation page."""

import time
import streamlit as st

from faer.core.scenario import Scenario
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

# Scenario summary
st.header("Current Scenario")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Duration", f"{scenario.run_length / 60:.0f} hours")

with col2:
    st.metric("Resus Bays", scenario.n_resus_bays)

with col3:
    st.metric("Mean LoS", f"{scenario.resus_mean:.0f} min")

with col4:
    st.metric("Replications", n_reps)

st.divider()

# Run button
if st.button("🚀 Run Experiment", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()

    start_time = time.time()

    def progress_callback(current: int, total: int) -> None:
        progress_bar.progress(current / total)
        status_text.text(f"Running replication {current}/{total}...")

    # Run replications
    results = multiple_replications(
        scenario,
        n_reps=n_reps,
        metric_names=[
            "arrivals",
            "departures",
            "p_delay",
            "mean_queue_time",
            "mean_system_time",
            "utilisation",
        ],
        progress_callback=progress_callback,
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
    import numpy as np

    summary_cols = st.columns(3)

    with summary_cols[0]:
        p_delay_mean = np.mean(results["p_delay"])
        st.metric("Mean P(Delay)", f"{p_delay_mean:.1%}")

    with summary_cols[1]:
        queue_mean = np.mean(results["mean_queue_time"])
        st.metric("Mean Queue Time", f"{queue_mean:.1f} min")

    with summary_cols[2]:
        util_mean = np.mean(results["utilisation"])
        st.metric("Mean Utilisation", f"{util_mean:.1%}")

    st.info("📈 Go to **Results** page for detailed analysis with confidence intervals.")

# Show previous results if they exist
elif st.session_state.get("run_complete"):
    st.info(f"✅ Previous run completed in {st.session_state.get('run_time', 0):.1f}s. "
            "Click **Run Experiment** to re-run, or view **Results**.")

    import numpy as np

    results = st.session_state.results
    summary_cols = st.columns(3)

    with summary_cols[0]:
        st.metric("Mean P(Delay)", f"{np.mean(results['p_delay']):.1%}")

    with summary_cols[1]:
        st.metric("Mean Queue Time", f"{np.mean(results['mean_queue_time']):.1f} min")

    with summary_cols[2]:
        st.metric("Mean Utilisation", f"{np.mean(results['utilisation']):.1%}")
