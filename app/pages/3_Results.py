"""Results display page for full A&E model."""

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

# Check for stale results (from before Phase 5 update)
if "util_ed_bays" not in results:
    st.warning("⚠️ Results are from an older version. Please re-run the simulation to see updated metrics.")
    st.session_state.run_complete = False
    st.page_link("pages/2_Run.py", label="Go to Run Simulation", icon="▶️")
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

# ===== RESOURCE UTILISATION =====
st.header("Resource Utilisation")

util_cols = st.columns(4)

with util_cols[0]:
    ci = compute_ci(results["util_triage"])
    st.metric("Triage", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with util_cols[1]:
    ci = compute_ci(results["util_ed_bays"])
    st.metric("ED Bays", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with util_cols[2]:
    ci = compute_ci(results["util_handover"])
    st.metric("Handover", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with util_cols[3]:
    ci = compute_ci(results["util_ambulance_fleet"])
    st.metric("Fleet", f"{ci['mean']:.1%}")
    st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# Utilisation bar chart
util_data = pd.DataFrame({
    "Resource": ["Triage", "ED Bays", "Handover", "Fleet"],
    "Utilisation": [
        np.mean(results["util_triage"]),
        np.mean(results["util_ed_bays"]),
        np.mean(results["util_handover"]),
        np.mean(results["util_ambulance_fleet"]),
    ]
})

fig = px.bar(
    util_data,
    x="Resource",
    y="Utilisation",
    title="Mean Resource Utilisation",
    color="Resource",
    color_discrete_map={
        "Triage": "#636efa",
        "ED Bays": "#ff4b4b",
        "Handover": "#ffa62b",
        "Fleet": "#29b09d"
    },
)
fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
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

with config_cols[2]:
    st.write("**Acuity Mix**")
    st.write(f"Resus: {scenario.p_resus:.0%}")
    st.write(f"Majors: {scenario.p_majors:.0%}")
    st.write(f"Minors: {scenario.p_minors:.0%}")

with config_cols[3]:
    st.write("**Experiment**")
    st.write(f"Replications: {n_reps}")
    st.write(f"Seed: {scenario.random_seed}")

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
    "util_triage": results["util_triage"],
    "util_ed_bays": results["util_ed_bays"],
    "util_handover": results["util_handover"],
    "util_ambulance_fleet": results["util_ambulance_fleet"],
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
        label="📥 Download Raw Results (CSV)",
        data=csv,
        file_name="faer_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    # Create summary with CIs
    summary_data = []
    for metric in export_df.columns[1:]:  # Skip replication column
        ci = compute_ci(results[metric])
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
        label="📥 Download Summary with CIs (CSV)",
        data=summary_csv,
        file_name="faer_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
