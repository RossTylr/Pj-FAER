"""Results display page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.experiment.analysis import compute_ci

st.set_page_config(page_title="Results - FAER", page_icon="📈", layout="wide")

st.title("📈 Results")

# Check for results
if not st.session_state.get("run_complete"):
    st.warning("⚠️ Please run a simulation first.")
    st.page_link("pages/2_Run.py", label="Go to Run Simulation", icon="▶️")
    st.stop()

results = st.session_state.results
scenario = st.session_state.scenario
n_reps = len(results["p_delay"])

# KPI Cards with CIs
st.header("Key Performance Indicators")
st.caption(f"Based on {n_reps} replications | 95% Confidence Intervals")

kpi_cols = st.columns(3)

# P(Delay)
with kpi_cols[0]:
    ci = compute_ci(results["p_delay"])
    st.metric(
        "P(Delay)",
        f"{ci['mean']:.1%}",
        help="Probability a patient waits for a Resus bay",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
    st.caption(f"Half-width: ±{ci['ci_half_width']:.1%}")

# Mean Queue Time
with kpi_cols[1]:
    ci = compute_ci(results["mean_queue_time"])
    st.metric(
        "Mean Queue Time",
        f"{ci['mean']:.1f} min",
        help="Average time waiting for Resus bay",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")
    st.caption(f"Half-width: ±{ci['ci_half_width']:.1f}")

# Utilisation
with kpi_cols[2]:
    ci = compute_ci(results["utilisation"])
    st.metric(
        "Resus Utilisation",
        f"{ci['mean']:.1%}",
        help="Time-weighted proportion of bays in use",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
    st.caption(f"Half-width: ±{ci['ci_half_width']:.1%}")

st.divider()

# Secondary metrics
st.header("Additional Metrics")
sec_cols = st.columns(3)

with sec_cols[0]:
    ci = compute_ci(results["arrivals"])
    st.metric("Arrivals", f"{ci['mean']:.0f}")
    st.caption(f"Range: [{min(results['arrivals']):.0f}, {max(results['arrivals']):.0f}]")

with sec_cols[1]:
    ci = compute_ci(results["departures"])
    st.metric("Departures", f"{ci['mean']:.0f}")
    st.caption(f"Range: [{min(results['departures']):.0f}, {max(results['departures']):.0f}]")

with sec_cols[2]:
    ci = compute_ci(results["mean_system_time"])
    st.metric("Mean System Time", f"{ci['mean']:.1f} min")
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

st.divider()

# Distribution plots
st.header("Metric Distributions Across Replications")

plot_cols = st.columns(2)

with plot_cols[0]:
    fig = px.histogram(
        results["p_delay"],
        nbins=20,
        labels={"value": "P(Delay)", "count": "Frequency"},
        title="Distribution of P(Delay)",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with plot_cols[1]:
    fig = px.histogram(
        results["mean_queue_time"],
        nbins=20,
        labels={"value": "Queue Time (min)", "count": "Frequency"},
        title="Distribution of Mean Queue Time",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Utilisation histogram
fig = px.histogram(
    results["utilisation"],
    nbins=20,
    labels={"value": "Utilisation", "count": "Frequency"},
    title="Distribution of Utilisation",
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Scenario summary
st.header("Scenario Configuration")
config_cols = st.columns(4)

with config_cols[0]:
    st.write("**Duration**")
    st.write(f"{scenario.run_length / 60:.0f} hours")

with config_cols[1]:
    st.write("**Resus Bays**")
    st.write(f"{scenario.n_resus_bays}")

with config_cols[2]:
    st.write("**Mean LoS**")
    st.write(f"{scenario.resus_mean:.0f} min")

with config_cols[3]:
    st.write("**Random Seed**")
    st.write(f"{scenario.random_seed}")

st.divider()

# Export section
st.header("Export Results")

# Create DataFrame for export
export_df = pd.DataFrame({
    "replication": range(1, n_reps + 1),
    "arrivals": results["arrivals"],
    "departures": results["departures"],
    "p_delay": results["p_delay"],
    "mean_queue_time": results["mean_queue_time"],
    "mean_system_time": results["mean_system_time"],
    "utilisation": results["utilisation"],
})

# Show preview
with st.expander("Preview data"):
    st.dataframe(export_df, use_container_width=True)

# Download button
csv = export_df.to_csv(index=False)
st.download_button(
    label="📥 Download Results CSV",
    data=csv,
    file_name="faer_results.csv",
    mime="text/csv",
    use_container_width=True,
)
