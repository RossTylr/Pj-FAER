"""Results display page for full A&E model."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.experiment.analysis import compute_ci

st.set_page_config(page_title="Results - FAER", page_icon="", layout="wide")

st.title("Results")

# Check for results
if not st.session_state.get("run_complete"):
    st.warning("Please run a simulation first.")
    st.page_link("pages/5_Run.py", label="Go to Run Simulation")
    st.stop()

# Support both old key (results) and new key (run_results)
results = st.session_state.get('run_results') or st.session_state.get('results')
scenario = st.session_state.get('run_scenario') or st.session_state.get('scenario')

# Check for stale results (from before Phase 5 update)
if "util_ed_bays" not in results:
    st.warning("Results are from an older version. Please re-run the simulation to see updated metrics.")
    st.session_state.run_complete = False
    st.page_link("pages/5_Run.py", label="Go to Run Simulation")
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

# ===== RESOURCE UTILISATION - EXPANDED =====
st.header("Resource Utilisation")

# Helper to safely get utilisation with fallback
def get_util(key: str, default: float = 0.0):
    """Get utilisation from results with fallback."""
    if key in results and results[key]:
        return results[key]
    return [default] * n_reps

# Toggle between schematic and detailed view
view_mode = st.radio(
    "View mode",
    ["Schematic", "Detailed"],
    horizontal=True,
    help="Toggle between visual schematic and detailed metrics view"
)

if view_mode == "Schematic":
    # Render the utilisation schematic (matches crucifix layout from patient flow schematic)
    from app.components.react_schematic.data import (
        build_utilisation_schematic,
        render_utilisation_schematic_svg,
    )

    # Get run length in hours for throughput calculation
    run_length_hours = scenario.run_length / 60 if scenario else 8.0

    util_data = build_utilisation_schematic(results, compute_ci, run_length_hours)
    svg_content = render_utilisation_schematic_svg(util_data)
    st.components.v1.html(svg_content, height=650)

else:
    # --- Emergency Services ---
    st.subheader("Emergency Services")
    es_cols = st.columns(3)

    with es_cols[0]:
        ci = compute_ci(get_util("util_ambulance_fleet"))
        st.metric("Ambulance Fleet", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with es_cols[1]:
        ci = compute_ci(get_util("util_helicopter_fleet"))
        st.metric("HEMS Fleet", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with es_cols[2]:
        ci = compute_ci(get_util("util_handover"))
        st.metric("Handover Bays", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    # --- Triage & ED ---
    st.subheader("Triage & ED")
    ed_cols = st.columns(2)

    with ed_cols[0]:
        ci = compute_ci(get_util("util_triage"))
        st.metric("Triage", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with ed_cols[1]:
        ci = compute_ci(get_util("util_ed_bays"))
        st.metric("ED Bays", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    # --- Diagnostics ---
    st.subheader("Diagnostics")
    diag_cols = st.columns(3)

    with diag_cols[0]:
        ci = compute_ci(get_util("util_CT_SCAN"))
        st.metric("CT Scanner", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with diag_cols[1]:
        ci = compute_ci(get_util("util_XRAY"))
        st.metric("X-ray", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with diag_cols[2]:
        ci = compute_ci(get_util("util_BLOODS"))
        st.metric("Bloods", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    # --- Downstream: Theatre, ITU, Ward ---
    st.subheader("Downstream (Theatre, ITU, Ward)")
    ds_cols = st.columns(3)

    with ds_cols[0]:
        ci = compute_ci(get_util("util_theatre"))
        st.metric("Theatre", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with ds_cols[1]:
        ci = compute_ci(get_util("util_itu"))
        st.metric("ITU", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

    with ds_cols[2]:
        ci = compute_ci(get_util("util_ward"))
        st.metric("Ward", f"{ci['mean']:.1%}")
        st.caption(f"CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# --- Aeromed (if enabled and toggle is checked) ---
aeromed_total = results.get("aeromed_total", [0] * n_reps)
has_aeromed_data = sum(aeromed_total) > 0

# Check if aeromed was enabled in scenario
aeromed_was_enabled = (
    hasattr(scenario, 'aeromed_config')
    and scenario.aeromed_config
    and scenario.aeromed_config.enabled
)

# Show toggle if aeromed was enabled (even if no evacuations occurred)
if aeromed_was_enabled or has_aeromed_data:
    show_aeromed = st.checkbox(
        "Show Aeromed Evacuation Metrics",
        value=has_aeromed_data,  # Default to checked if there's data
        help="Toggle to show/hide aeromed evacuation statistics"
    )
else:
    show_aeromed = False

if show_aeromed:
    st.subheader("Aeromed Evacuation")
    if not has_aeromed_data:
        st.info("Aeromed was enabled but no evacuations occurred during this simulation run.")
    else:
        aeromed_cols = st.columns(4)

        with aeromed_cols[0]:
            ci = compute_ci(aeromed_total)
            st.metric("Total Aeromed", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[1]:
            hems_count = results.get("aeromed_hems_count", [0] * n_reps)
            ci = compute_ci(hems_count)
            st.metric("HEMS Evacuations", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[2]:
            fw_count = results.get("aeromed_fixedwing_count", [0] * n_reps)
            ci = compute_ci(fw_count)
            st.metric("Fixed-Wing Evacuations", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols[3]:
            slots_missed = results.get("aeromed_slots_missed", [0] * n_reps)
            ci = compute_ci(slots_missed)
            st.metric("Slots Missed", f"{ci['mean']:.1f}")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        # Additional aeromed metrics row
        aeromed_cols2 = st.columns(3)

        with aeromed_cols2[0]:
            slot_wait = results.get("mean_aeromed_slot_wait", [0] * n_reps)
            ci = compute_ci(slot_wait)
            st.metric("Mean Slot Wait", f"{ci['mean']:.1f} min")
            st.caption(f"CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

        with aeromed_cols2[1]:
            blocked_days = results.get("ward_bed_days_blocked_aeromed", [0] * n_reps)
            ci = compute_ci(blocked_days)
            st.metric("Ward Bed-Days Blocked", f"{ci['mean']:.2f}")
            st.caption(f"By aeromed patients waiting for slots")

        with aeromed_cols2[2]:
            # Show aeromed as % of total departures
            departures = results.get("departures", [1] * n_reps)
            aeromed_pct = [a / d * 100 if d > 0 else 0 for a, d in zip(aeromed_total, departures)]
            ci = compute_ci(aeromed_pct)
            st.metric("Aeromed % of Departures", f"{ci['mean']:.1f}%")
            st.caption(f"CI: [{ci['ci_lower']:.1f}%, {ci['ci_upper']:.1f}%]")

# Utilisation bar chart - all resources
st.markdown("---")
st.subheader("Utilisation Overview")

util_data = pd.DataFrame({
    "Resource": [
        "Ambulance", "HEMS", "Handover",
        "Triage", "ED Bays",
        "CT", "X-ray", "Bloods",
        "Theatre", "ITU", "Ward"
    ],
    "Category": [
        "Emergency", "Emergency", "Emergency",
        "ED", "ED",
        "Diagnostics", "Diagnostics", "Diagnostics",
        "Downstream", "Downstream", "Downstream"
    ],
    "Utilisation": [
        np.mean(get_util("util_ambulance_fleet")),
        np.mean(get_util("util_helicopter_fleet")),
        np.mean(get_util("util_handover")),
        np.mean(get_util("util_triage")),
        np.mean(get_util("util_ed_bays")),
        np.mean(get_util("util_CT_SCAN")),
        np.mean(get_util("util_XRAY")),
        np.mean(get_util("util_BLOODS")),
        np.mean(get_util("util_theatre")),
        np.mean(get_util("util_itu")),
        np.mean(get_util("util_ward")),
    ]
})

fig = px.bar(
    util_data,
    x="Resource",
    y="Utilisation",
    title="Mean Resource Utilisation Across System",
    color="Category",
    color_discrete_map={
        "Emergency": "#ffa62b",
        "ED": "#ff4b4b",
        "Diagnostics": "#ab63fa",
        "Downstream": "#636efa"
    },
)
fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45)
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
    # Expanded 5-way disposition breakdown
    disp_data = pd.DataFrame({
        "Outcome": ["Discharged", "Ward", "ICU", "Transfer", "Left"],
        "Count": [
            np.mean(results.get("discharged_count", results.get("discharged", [0]*n_reps))),
            np.mean(results.get("admitted_ward_count", [0]*n_reps)),
            np.mean(results.get("admitted_icu_count", [0]*n_reps)),
            np.mean(results.get("transfer_count", [0]*n_reps)),
            np.mean(results.get("left_count", [0]*n_reps)),
        ]
    })
    # Filter out zero counts for cleaner display
    disp_data = disp_data[disp_data["Count"] > 0]

    fig = px.pie(
        disp_data,
        values="Count",
        names="Outcome",
        title="Mean Departures by Disposition",
        color="Outcome",
        color_discrete_map={
            "Discharged": "#00cc96",
            "Ward": "#636efa",
            "ICU": "#ff4b4b",
            "Transfer": "#ffa62b",
            "Left": "#ab63fa"
        },
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

# Length of Stay by Disposition (if data available)
if "mean_los_discharged" in results:
    st.markdown("---")
    st.subheader("Length of Stay by Disposition")
    los_cols = st.columns(4)

    los_metrics = [
        ("Discharged", "mean_los_discharged"),
        ("Ward", "mean_los_ward"),
        ("ICU", "mean_los_icu"),
        ("Transfer", "mean_los_transfer"),
    ]

    for i, (label, key) in enumerate(los_metrics):
        with los_cols[i]:
            los_values = results.get(key, [0]*n_reps)
            if np.mean(los_values) > 0:
                ci = compute_ci(los_values)
                st.metric(f"{label} LoS", f"{ci['mean']:.0f} min")
                st.caption(f"({ci['mean']/60:.1f} hrs)")

st.divider()

# ===== ACUITY STATISTICS =====
st.header("Acuity Statistics")
st.caption("Performance by clinical acuity level (assigned at arrival)")

# Acuity labels and descriptions
ACUITY_INFO = {
    "resus": ("Resus", "Immediate life-threatening", "#ff4b4b"),
    "majors": ("Majors", "Urgent/serious conditions", "#ffa62b"),
    "minors": ("Minors", "Standard/ambulatory", "#29b09d"),
}

# Build acuity statistics table
acuity_stats = []
for acuity_key in ["resus", "majors", "minors"]:
    label, desc, _ = ACUITY_INFO[acuity_key]
    arrivals = np.mean(results.get(f"arrivals_{acuity_key}", [0]*n_reps))
    departures = np.mean(results.get(f"departures_{acuity_key}", [0]*n_reps))
    mean_wait_data = results.get(f"{acuity_key}_mean_wait", [0]*n_reps)
    p95_wait_data = results.get(f"{acuity_key}_p95_wait", [0]*n_reps)
    sys_time_data = results.get(f"{acuity_key}_mean_system_time", [0]*n_reps)

    acuity_stats.append({
        "Acuity": label,
        "Description": desc,
        "Arrivals": f"{arrivals:.0f}",
        "Departures": f"{departures:.0f}",
        "Mean Wait (min)": f"{np.mean(mean_wait_data):.1f}",
        "P95 Wait (min)": f"{np.mean(p95_wait_data):.1f}",
        "Mean System Time (min)": f"{np.mean(sys_time_data):.0f}",
    })

acuity_df = pd.DataFrame(acuity_stats)
st.dataframe(acuity_df, use_container_width=True, hide_index=True)

# Acuity wait time comparison chart
acuity_wait_data = pd.DataFrame({
    "Acuity": ["Resus", "Majors", "Minors"],
    "Mean Wait": [
        np.mean(results.get("resus_mean_wait", [0]*n_reps)),
        np.mean(results.get("majors_mean_wait", [0]*n_reps)),
        np.mean(results.get("minors_mean_wait", [0]*n_reps)),
    ],
    "P95 Wait": [
        np.mean(results.get("resus_p95_wait", [0]*n_reps)),
        np.mean(results.get("majors_p95_wait", [0]*n_reps)),
        np.mean(results.get("minors_p95_wait", [0]*n_reps)),
    ],
})

fig = px.bar(
    acuity_wait_data,
    x="Acuity",
    y=["Mean Wait", "P95 Wait"],
    barmode="group",
    title="Wait Times by Acuity Level",
    color_discrete_map={"Mean Wait": "#636efa", "P95 Wait": "#ffa62b"},
    labels={"value": "Wait Time (min)", "variable": "Metric"},
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== PRIORITY STATISTICS =====
st.header("Priority Statistics")
st.caption("Performance by triage priority level | NHS breach targets (P1=0min, P2=10min, P3=60min, P4=120min)")

# NHS Standard Targets
PRIORITY_TARGETS = {"P1": 0, "P2": 10, "P3": 60, "P4": 120}
PRIORITY_LABELS = {
    "P1": "P1 - Immediate",
    "P2": "P2 - Very Urgent",
    "P3": "P3 - Urgent",
    "P4": "P4 - Standard",
}

# Build priority statistics table
priority_stats = []
for p in ["P1", "P2", "P3", "P4"]:
    arrivals = np.mean(results.get(f"arrivals_{p}", [0]*n_reps))
    departures = np.mean(results.get(f"departures_{p}", [0]*n_reps))
    mean_wait_data = results.get(f"{p}_mean_wait", [0]*n_reps)
    p95_wait_data = results.get(f"{p}_p95_wait", [0]*n_reps)
    breach_data = results.get(f"{p}_breach_rate", [0]*n_reps)
    sys_time_data = results.get(f"{p}_mean_system_time", [0]*n_reps)

    priority_stats.append({
        "Priority": PRIORITY_LABELS[p],
        "Target (min)": PRIORITY_TARGETS[p],
        "Arrivals": f"{arrivals:.0f}",
        "Departures": f"{departures:.0f}",
        "Mean Wait (min)": f"{np.mean(mean_wait_data):.1f}",
        "P95 Wait (min)": f"{np.mean(p95_wait_data):.1f}",
        "Breach Rate": f"{np.mean(breach_data):.1%}",
        "Mean System Time (min)": f"{np.mean(sys_time_data):.0f}",
    })

priority_df = pd.DataFrame(priority_stats)
st.dataframe(priority_df, use_container_width=True, hide_index=True)

# Breach rate visualization
breach_data_chart = pd.DataFrame({
    "Priority": ["P1", "P2", "P3", "P4"],
    "Breach Rate": [
        np.mean(results.get("P1_breach_rate", [0]*n_reps)),
        np.mean(results.get("P2_breach_rate", [0]*n_reps)),
        np.mean(results.get("P3_breach_rate", [0]*n_reps)),
        np.mean(results.get("P4_breach_rate", [0]*n_reps)),
    ],
    "Target Wait (min)": [0, 10, 60, 120],
})

fig = px.bar(
    breach_data_chart,
    x="Priority",
    y="Breach Rate",
    title="Breach Rates by Priority (% exceeding NHS target)",
    color="Breach Rate",
    color_continuous_scale=["#00cc96", "#ffa62b", "#ff4b4b"],
)
fig.update_layout(yaxis_tickformat=".0%")
fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="5% target")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== DISTRIBUTION PLOTS =====
st.header("Metric Distributions Across Replications")

# Build dynamic tab list based on active departments
tab_names = ["Wait Times", "System Time", "Utilisation"]

# Check if downstream has data (hide empty departments)
has_downstream = any(
    np.mean(get_util(k)) > 0 for k in ["util_theatre", "util_itu", "util_ward"]
)
if has_downstream:
    tab_names.append("Downstream")

# Check if diagnostics have data
has_diagnostics = any(
    np.mean(get_util(f"util_{dt}")) > 0
    for dt in ["CT_SCAN", "XRAY", "BLOODS"]
)
if has_diagnostics:
    tab_names.append("Diagnostics")

# Priority tab always shown
tab_names.append("Priority")

dist_tabs = st.tabs(tab_names)
tab_idx = 0

# Wait Times tab
with dist_tabs[tab_idx]:
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
tab_idx += 1

# System Time tab
with dist_tabs[tab_idx]:
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
tab_idx += 1

# Utilisation tab (core resources always shown)
with dist_tabs[tab_idx]:
    st.subheader("Core Resource Utilisation")
    util_cols = st.columns(3)

    for i, (name, key) in enumerate([
        ("ED Bays", "util_ed_bays"),
        ("Triage", "util_triage"),
        ("Handover", "util_handover"),
    ]):
        with util_cols[i]:
            fig = px.histogram(
                get_util(key),
                nbins=20,
                labels={"value": "Utilisation", "count": "Frequency"},
                title=f"{name} Utilisation Distribution",
            )
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
tab_idx += 1

# Downstream tab (conditional)
if has_downstream:
    with dist_tabs[tab_idx]:
        st.subheader("Downstream Resource Distributions")
        # Only show departments with data
        active_downstream = []
        for name, key in [("Theatre", "util_theatre"), ("ITU", "util_itu"), ("Ward", "util_ward")]:
            if np.mean(get_util(key)) > 0:
                active_downstream.append((name, key))

        if active_downstream:
            ds_cols = st.columns(len(active_downstream))
            for i, (name, key) in enumerate(active_downstream):
                with ds_cols[i]:
                    fig = px.histogram(
                        get_util(key),
                        nbins=20,
                        labels={"value": "Utilisation", "count": "Frequency"},
                        title=f"{name} Utilisation Distribution",
                    )
                    fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
    tab_idx += 1

# Diagnostics tab (conditional)
if has_diagnostics:
    with dist_tabs[tab_idx]:
        st.subheader("Diagnostics Resource Distributions")
        active_diag = []
        for name, key in [("CT Scanner", "util_CT_SCAN"), ("X-ray", "util_XRAY"), ("Bloods", "util_BLOODS")]:
            if np.mean(get_util(key)) > 0:
                active_diag.append((name, key))

        if active_diag:
            diag_cols = st.columns(len(active_diag))
            for i, (name, key) in enumerate(active_diag):
                with diag_cols[i]:
                    fig = px.histogram(
                        get_util(key),
                        nbins=20,
                        labels={"value": "Utilisation", "count": "Frequency"},
                        title=f"{name} Utilisation Distribution",
                    )
                    fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
    tab_idx += 1

# Priority tab (always shown)
with dist_tabs[tab_idx]:
    st.subheader("Wait Time Distributions by Priority")
    p_cols = st.columns(4)
    for i, p in enumerate(["P1", "P2", "P3", "P4"]):
        with p_cols[i]:
            wait_data = results.get(f"{p}_mean_wait", [0]*n_reps)
            fig = px.histogram(
                wait_data,
                nbins=15,
                labels={"value": "Wait (min)", "count": "Frequency"},
                title=f"{p} Mean Wait Distribution",
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
    st.write(f"Helicopters: {scenario.n_helicopters}")

with config_cols[2]:
    st.write("**Acuity Mix**")
    st.write(f"Resus: {scenario.p_resus:.0%}")
    st.write(f"Majors: {scenario.p_majors:.0%}")
    st.write(f"Minors: {scenario.p_minors:.0%}")

with config_cols[3]:
    st.write("**Experiment**")
    st.write(f"Replications: {n_reps}")
    st.write(f"Seed: {scenario.random_seed}")

# Second row: Diagnostics and Downstream
config_cols2 = st.columns(3)

with config_cols2[0]:
    st.write("**Diagnostics**")
    if hasattr(scenario, 'diagnostic_configs') and scenario.diagnostic_configs:
        from faer.core.entities import DiagnosticType
        ct_config = scenario.diagnostic_configs.get(DiagnosticType.CT_SCAN)
        xray_config = scenario.diagnostic_configs.get(DiagnosticType.XRAY)
        bloods_config = scenario.diagnostic_configs.get(DiagnosticType.BLOODS)
        st.write(f"CT Scanners: {ct_config.capacity if ct_config else 'N/A'}")
        st.write(f"X-ray Rooms: {xray_config.capacity if xray_config else 'N/A'}")
        st.write(f"Phlebotomy: {bloods_config.capacity if bloods_config else 'N/A'}")
    else:
        st.write("Not configured")

with config_cols2[1]:
    st.write("**Downstream**")
    if hasattr(scenario, 'theatre_config') and scenario.theatre_config:
        st.write(f"Theatre Tables: {scenario.theatre_config.n_tables}")
    else:
        st.write("Theatre: N/A")
    if hasattr(scenario, 'itu_config') and scenario.itu_config:
        st.write(f"ITU Beds: {scenario.itu_config.capacity}")
    else:
        st.write("ITU: N/A")
    if hasattr(scenario, 'ward_config') and scenario.ward_config:
        st.write(f"Ward Beds: {scenario.ward_config.capacity}")
    else:
        st.write("Ward: N/A")

with config_cols2[2]:
    st.write("**Downstream Status**")
    if hasattr(scenario, 'downstream_enabled'):
        st.write(f"Enabled: {'Yes' if scenario.downstream_enabled else 'No'}")
    else:
        st.write("Enabled: No")
    if hasattr(scenario, 'aeromed_config') and scenario.aeromed_config:
        st.write(f"Aeromed: {'Yes' if scenario.aeromed_config.enabled else 'No'}")
    else:
        st.write("Aeromed: No")

st.divider()

# ===== COST MODELLING SECTION =====
st.header("Cost Modelling")

with st.expander("Calculate Financial Costs", expanded=False):
    st.markdown("""
    Calculate financial costs from simulation results using configurable bed-day rates.
    This is a **post-hoc analysis** that does not affect simulation behaviour.

    **Note:** Cost calculation runs a single replication to access detailed patient data.
    """)

    # Import cost modelling module
    from faer.results.costs import (
        CostConfig, calculate_costs, calculate_costs_by_priority,
        format_currency, CURRENCY_SYMBOLS
    )
    from faer.model.full_model import run_full_simulation, FullResultsCollector

    # Cost configuration in columns
    st.subheader("Configuration")

    cost_col1, cost_col2 = st.columns(2)

    with cost_col1:
        currency = st.selectbox(
            "Currency",
            options=["GBP", "USD", "EUR"],
            index=0,
            help="Select currency for cost display"
        )
        currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)

    with cost_col2:
        apply_priority_multipliers = st.checkbox(
            "Apply priority multipliers",
            value=True,
            help="Higher acuity patients (P1/P2) cost more due to resource intensity"
        )

    # Bed-day rates
    st.markdown("**Bed-Day Rates** (per 24 hours)")
    rate_cols = st.columns(4)

    with rate_cols[0]:
        ed_bay_rate = st.number_input(
            f"ED Bay ({currency_symbol}/day)",
            min_value=0.0, max_value=10000.0, value=500.0, step=50.0
        )
    with rate_cols[1]:
        itu_rate = st.number_input(
            f"ITU ({currency_symbol}/day)",
            min_value=0.0, max_value=20000.0, value=2000.0, step=100.0
        )
    with rate_cols[2]:
        ward_rate = st.number_input(
            f"Ward ({currency_symbol}/day)",
            min_value=0.0, max_value=5000.0, value=400.0, step=50.0
        )
    with rate_cols[3]:
        theatre_rate = st.number_input(
            f"Theatre ({currency_symbol}/hr)",
            min_value=0.0, max_value=10000.0, value=2000.0, step=100.0
        )

    # Per-episode and transport costs
    st.markdown("**Per-Episode & Transport Costs**")
    episode_cols = st.columns(4)

    with episode_cols[0]:
        triage_cost = st.number_input(
            f"Triage ({currency_symbol})",
            min_value=0.0, max_value=500.0, value=20.0, step=5.0
        )
    with episode_cols[1]:
        diagnostics_cost = st.number_input(
            f"Diagnostics avg ({currency_symbol})",
            min_value=0.0, max_value=1000.0, value=75.0, step=10.0
        )
    with episode_cols[2]:
        ambulance_cost = st.number_input(
            f"Ambulance ({currency_symbol})",
            min_value=0.0, max_value=2000.0, value=275.0, step=25.0
        )
    with episode_cols[3]:
        hems_cost = st.number_input(
            f"HEMS flight ({currency_symbol})",
            min_value=0.0, max_value=20000.0, value=3500.0, step=100.0
        )

    st.markdown("---")

    # Run button
    if st.button("Calculate Costs", type="primary", use_container_width=True):
        with st.spinner("Running cost calculation (single replication)..."):
            # Create cost config
            config = CostConfig(
                enabled=True,
                currency=currency,
                ed_bay_per_day=ed_bay_rate,
                itu_bed_per_day=itu_rate,
                ward_bed_per_day=ward_rate,
                theatre_per_hour=theatre_rate,
                triage_cost=triage_cost,
                diagnostics_base_cost=diagnostics_cost,
                ambulance_per_journey=ambulance_cost,
                hems_per_flight=hems_cost,
            )

            if not apply_priority_multipliers:
                config.apply_priority_to = []

            # Run a single replication to get patient-level data
            # Use the scenario from session state
            try:
                import simpy
                from faer.core.entities import DiagnosticType

                # Create fresh environment for cost calculation
                env = simpy.Environment()

                # Run simulation to get patient data
                # We need to access the FullResultsCollector directly
                from faer.model.full_model import (
                    AEResources, FullResultsCollector,
                    arrival_generator_multistream, arrival_generator_single,
                    sample_lognormal
                )
                import itertools

                cost_scenario = scenario.clone_with_seed(scenario.random_seed)

                # Create resources
                diagnostic_resources = {}
                for diag_type, diag_config in cost_scenario.diagnostic_configs.items():
                    if diag_config.enabled:
                        diagnostic_resources[diag_type] = simpy.PriorityResource(
                            env, capacity=diag_config.capacity
                        )

                transfer_ambulances = None
                transfer_helicopters = None
                if cost_scenario.transfer_config.enabled:
                    transfer_ambulances = simpy.Resource(
                        env, capacity=cost_scenario.transfer_config.n_transfer_ambulances
                    )
                    transfer_helicopters = simpy.Resource(
                        env, capacity=cost_scenario.transfer_config.n_transfer_helicopters
                    )

                theatre_tables = None
                itu_beds = None
                ward_beds = None
                if cost_scenario.downstream_enabled:
                    if cost_scenario.theatre_config and cost_scenario.theatre_config.enabled:
                        theatre_tables = simpy.PriorityResource(
                            env, capacity=cost_scenario.theatre_config.n_tables
                        )
                    if cost_scenario.itu_config and cost_scenario.itu_config.enabled:
                        itu_beds = simpy.Resource(env, capacity=cost_scenario.itu_config.capacity)
                    if cost_scenario.ward_config and cost_scenario.ward_config.enabled:
                        ward_beds = simpy.Resource(env, capacity=cost_scenario.ward_config.capacity)

                hems_slots = None
                if cost_scenario.aeromed_config and cost_scenario.aeromed_config.enabled:
                    if cost_scenario.aeromed_config.hems.enabled:
                        hems_slots = simpy.Resource(
                            env, capacity=cost_scenario.aeromed_config.hems.slots_per_day
                        )

                resources = AEResources(
                    triage=simpy.PriorityResource(env, capacity=cost_scenario.n_triage),
                    ed_bays=simpy.PriorityResource(env, capacity=cost_scenario.n_ed_bays),
                    handover_bays=simpy.Resource(env, capacity=cost_scenario.n_handover_bays),
                    ambulance_fleet=simpy.Resource(env, capacity=cost_scenario.n_ambulances),
                    helicopter_fleet=simpy.Resource(env, capacity=cost_scenario.n_helicopters),
                    diagnostics=diagnostic_resources,
                    transfer_ambulances=transfer_ambulances,
                    transfer_helicopters=transfer_helicopters,
                    theatre_tables=theatre_tables,
                    itu_beds=itu_beds,
                    ward_beds=ward_beds,
                    hems_slots=hems_slots,
                )

                collector = FullResultsCollector()
                patient_counter = itertools.count(1)

                # Start arrival generators
                if cost_scenario.arrival_configs:
                    for arr_config in cost_scenario.arrival_configs:
                        env.process(arrival_generator_multistream(
                            env, resources, arr_config, cost_scenario, collector, patient_counter
                        ))
                else:
                    env.process(arrival_generator_single(
                        env, resources, cost_scenario, collector, patient_counter
                    ))

                # Run simulation
                total_time = cost_scenario.warm_up + cost_scenario.run_length
                env.run(until=total_time)

                # Filter to post-warmup patients who have departed
                valid_patients = [
                    p for p in collector.patients
                    if p.arrival_time >= cost_scenario.warm_up and p.departure_time is not None
                ]

                # Calculate costs
                breakdown = calculate_costs(valid_patients, config)
                by_priority = calculate_costs_by_priority(valid_patients, config)

                # Store results in session state
                st.session_state['cost_breakdown'] = breakdown
                st.session_state['cost_by_priority'] = by_priority
                st.session_state['cost_config'] = config
                st.session_state['cost_patients_count'] = len(valid_patients)

                st.success(f"Cost calculation complete for {len(valid_patients)} patients.")

            except Exception as e:
                st.error(f"Error calculating costs: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results if available
    if 'cost_breakdown' in st.session_state:
        breakdown = st.session_state['cost_breakdown']
        by_priority = st.session_state['cost_by_priority']
        sym = breakdown.get_currency_symbol()

        st.markdown("---")
        st.subheader("Cost Summary")

        # Key metrics
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric(
                "Grand Total",
                format_currency(breakdown.grand_total, sym),
                help="Total cost across all categories"
            )
        with summary_cols[1]:
            st.metric(
                "Cost per Patient",
                format_currency(breakdown.cost_per_patient, sym),
                help="Average cost per patient"
            )
        with summary_cols[2]:
            st.metric(
                "Patients",
                f"{breakdown.total_patients:,}",
                help="Number of patients included in calculation"
            )

        # Cost breakdown by category
        st.subheader("Breakdown by Category")

        cat_cols = st.columns(4)
        with cat_cols[0]:
            st.markdown("**Bed Costs**")
            st.write(f"ED Bays: {format_currency(breakdown.ed_bay_costs, sym)}")
            st.write(f"ITU: {format_currency(breakdown.itu_bed_costs, sym)}")
            st.write(f"Ward: {format_currency(breakdown.ward_bed_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_bed_costs, sym)}**")

        with cat_cols[1]:
            st.markdown("**Theatre**")
            st.write(f"Theatre: {format_currency(breakdown.theatre_costs, sym)}")

        with cat_cols[2]:
            st.markdown("**Per-Episode**")
            st.write(f"Triage: {format_currency(breakdown.triage_costs, sym)}")
            st.write(f"Diagnostics: {format_currency(breakdown.diagnostics_costs, sym)}")
            st.write(f"Discharge: {format_currency(breakdown.discharge_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_episode_costs, sym)}**")

        with cat_cols[3]:
            st.markdown("**Transport**")
            st.write(f"Ambulance: {format_currency(breakdown.ambulance_costs, sym)}")
            st.write(f"HEMS: {format_currency(breakdown.hems_costs, sym)}")
            st.write(f"Fixed-wing: {format_currency(breakdown.fixedwing_costs, sym)}")
            st.write(f"**Total: {format_currency(breakdown.total_transport_costs, sym)}**")

        # Cost by priority
        st.subheader("Breakdown by Priority")

        priority_data = pd.DataFrame({
            "Priority": ["P1 (Immediate)", "P2 (Very Urgent)", "P3 (Urgent)", "P4 (Standard)"],
            "Patients": [by_priority.p1_count, by_priority.p2_count,
                        by_priority.p3_count, by_priority.p4_count],
            "Total Cost": [
                format_currency(by_priority.p1_total, sym),
                format_currency(by_priority.p2_total, sym),
                format_currency(by_priority.p3_total, sym),
                format_currency(by_priority.p4_total, sym),
            ],
            "Per Patient": [
                format_currency(by_priority.p1_per_patient, sym),
                format_currency(by_priority.p2_per_patient, sym),
                format_currency(by_priority.p3_per_patient, sym),
                format_currency(by_priority.p4_per_patient, sym),
            ],
        })
        st.dataframe(priority_data, use_container_width=True, hide_index=True)

        # Cost distribution chart
        cost_data = pd.DataFrame({
            "Category": ["ED Bays", "Theatre", "ITU", "Ward", "Transport", "Per-Episode"],
            "Cost": [
                breakdown.ed_bay_costs,
                breakdown.theatre_costs,
                breakdown.itu_bed_costs,
                breakdown.ward_bed_costs,
                breakdown.total_transport_costs,
                breakdown.total_episode_costs,
            ]
        })

        fig = px.pie(
            cost_data,
            values="Cost",
            names="Category",
            title="Cost Distribution by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export cost data
        st.markdown("---")
        cost_export = pd.DataFrame([breakdown.to_dict()])
        csv_costs = cost_export.to_csv(index=False)
        st.download_button(
            label="Download Cost Breakdown (CSV)",
            data=csv_costs,
            file_name="faer_costs.csv",
            mime="text/csv",
        )

st.divider()

# ===== RESULTS PLAYBACK SCHEMATIC =====
st.header("Results Playback Schematic")
st.caption("Click on the timeline to see system state at that point in time")

# Check if we have patient-level data for playback
playback_available = False
playback_patients = None
playback_resource_logs = None

# Try to get patient data from session state (from a previous cost calculation run)
# or run a quick simulation to get it
if 'playback_collector' in st.session_state:
    playback_available = True
    playback_patients = st.session_state['playback_patients']
    playback_resource_logs = st.session_state['playback_resource_logs']

with st.expander("Simulation Playback", expanded=True):
    if not playback_available:
        st.info("""
        **Playback requires patient-level data.** Click the button below to run a single
        replication and enable timeline playback of simulation state.
        """)

        if st.button("Enable Playback (Run Single Replication)", type="primary"):
            with st.spinner("Running simulation for playback data..."):
                try:
                    import simpy
                    from faer.core.entities import DiagnosticType
                    from faer.model.full_model import (
                        AEResources, FullResultsCollector,
                        arrival_generator_multistream, arrival_generator_single,
                    )
                    import itertools

                    # Clone scenario for playback
                    playback_scenario = scenario.clone_with_seed(scenario.random_seed)

                    # Create fresh environment
                    env = simpy.Environment()

                    # Create resources
                    diagnostic_resources = {}
                    for diag_type, diag_config in playback_scenario.diagnostic_configs.items():
                        if diag_config.enabled:
                            diagnostic_resources[diag_type] = simpy.PriorityResource(
                                env, capacity=diag_config.capacity
                            )

                    transfer_ambulances = None
                    transfer_helicopters = None
                    if playback_scenario.transfer_config.enabled:
                        transfer_ambulances = simpy.Resource(
                            env, capacity=playback_scenario.transfer_config.n_transfer_ambulances
                        )
                        transfer_helicopters = simpy.Resource(
                            env, capacity=playback_scenario.transfer_config.n_transfer_helicopters
                        )

                    theatre_tables = None
                    itu_beds = None
                    ward_beds = None
                    if playback_scenario.downstream_enabled:
                        if playback_scenario.theatre_config and playback_scenario.theatre_config.enabled:
                            theatre_tables = simpy.PriorityResource(
                                env, capacity=playback_scenario.theatre_config.n_tables
                            )
                        if playback_scenario.itu_config and playback_scenario.itu_config.enabled:
                            itu_beds = simpy.Resource(env, capacity=playback_scenario.itu_config.capacity)
                        if playback_scenario.ward_config and playback_scenario.ward_config.enabled:
                            ward_beds = simpy.Resource(env, capacity=playback_scenario.ward_config.capacity)

                    hems_slots = None
                    if playback_scenario.aeromed_config and playback_scenario.aeromed_config.enabled:
                        if playback_scenario.aeromed_config.hems.enabled:
                            hems_slots = simpy.Resource(
                                env, capacity=playback_scenario.aeromed_config.hems.slots_per_day
                            )

                    resources = AEResources(
                        triage=simpy.PriorityResource(env, capacity=playback_scenario.n_triage),
                        ed_bays=simpy.PriorityResource(env, capacity=playback_scenario.n_ed_bays),
                        handover_bays=simpy.Resource(env, capacity=playback_scenario.n_handover_bays),
                        ambulance_fleet=simpy.Resource(env, capacity=playback_scenario.n_ambulances),
                        helicopter_fleet=simpy.Resource(env, capacity=playback_scenario.n_helicopters),
                        diagnostics=diagnostic_resources,
                        transfer_ambulances=transfer_ambulances,
                        transfer_helicopters=transfer_helicopters,
                        theatre_tables=theatre_tables,
                        itu_beds=itu_beds,
                        ward_beds=ward_beds,
                        hems_slots=hems_slots,
                    )

                    collector = FullResultsCollector()
                    patient_counter = itertools.count(1)

                    # Start arrival generators
                    if playback_scenario.arrival_configs:
                        for arr_config in playback_scenario.arrival_configs:
                            env.process(arrival_generator_multistream(
                                env, resources, arr_config, playback_scenario, collector, patient_counter
                            ))
                    else:
                        env.process(arrival_generator_single(
                            env, resources, playback_scenario, collector, patient_counter
                        ))

                    # Run simulation
                    total_time = playback_scenario.warm_up + playback_scenario.run_length
                    env.run(until=total_time)

                    # Store in session state
                    st.session_state['playback_collector'] = True
                    st.session_state['playback_patients'] = collector.patients
                    st.session_state['playback_resource_logs'] = collector.resource_logs
                    st.session_state['playback_scenario'] = playback_scenario

                    playback_available = True
                    playback_patients = collector.patients
                    playback_resource_logs = collector.resource_logs

                    st.success(f"Playback data ready! {len(collector.patients)} patients simulated.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error running playback simulation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    if playback_available and playback_patients:
        # Import playback functions
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from app.components.react_schematic.component import is_built, react_schematic
        from app.components.react_schematic.data import (
            build_schematic_at_time,
            build_timeline_data,
            to_dict,
        )

        # Get playback scenario
        playback_scenario = st.session_state.get('playback_scenario', scenario)

        # Show arrival mode summary (clean display)
        from collections import Counter
        from faer.core.entities import ArrivalMode

        mode_counts = Counter(p.mode for p in playback_patients if hasattr(p, 'mode'))
        ambulance_count = mode_counts.get(ArrivalMode.AMBULANCE, 0)
        hems_count = mode_counts.get(ArrivalMode.HELICOPTER, 0)
        walkin_count = mode_counts.get(ArrivalMode.SELF_PRESENTATION, 0)

        # Clean summary metrics
        summary_cols = st.columns(4)
        summary_cols[0].metric("Total Patients", len(playback_patients))
        summary_cols[1].metric("Ambulance", ambulance_count)
        summary_cols[2].metric("HEMS", hems_count)
        summary_cols[3].metric("Walk-in", walkin_count)

        # Debug: Show time range info
        if playback_patients:
            arrival_times = [p.arrival_time for p in playback_patients]
            departure_times = [p.departure_time for p in playback_patients if p.departure_time]
            st.caption(f"Patient arrival range: {min(arrival_times):.0f} - {max(arrival_times):.0f} min | "
                      f"Departures: {len(departure_times)} | "
                      f"Warm-up: {playback_scenario.warm_up:.0f}min, Run: {playback_scenario.run_length:.0f}min")

        # Build timeline data
        timeline_data = build_timeline_data(
            patients=playback_patients,
            resource_logs=playback_resource_logs,
            run_length=playback_scenario.warm_up + playback_scenario.run_length,
            warm_up=playback_scenario.warm_up,
            interval_mins=15.0,
        )

        # Debug: Show timeline data summary
        st.caption(f"Timeline: {len(timeline_data['time'])} points | "
                  f"Max in_system: {max(timeline_data['in_system']) if timeline_data['in_system'] else 0} | "
                  f"Max ED occ: {max(timeline_data['ed_occupancy']) if timeline_data['ed_occupancy'] else 0} | "
                  f"Total arrivals: {sum(timeline_data['arrivals']) if timeline_data['arrivals'] else 0}")

        # Initialize selected time if not set
        if 'playback_selected_time' not in st.session_state:
            # Default to end of simulation
            st.session_state['playback_selected_time'] = timeline_data['time'][-1] if timeline_data['time'] else playback_scenario.warm_up

        # Create timeline chart using Plotly
        timeline_df = pd.DataFrame(timeline_data)

        # Convert time to hours for display
        timeline_df['time_hours'] = timeline_df['time'] / 60

        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(
            x=timeline_df['time_hours'],
            y=timeline_df['in_system'],
            mode='lines',
            name='Patients in System',
            line=dict(color='#636efa', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.2)',
        ))

        fig.add_trace(go.Scatter(
            x=timeline_df['time_hours'],
            y=timeline_df['ed_occupancy'],
            mode='lines',
            name='ED Occupancy',
            line=dict(color='#ff4b4b', width=2),
        ))

        # Add ITU and Theatre occupancy traces
        if 'itu_occupancy' in timeline_df.columns:
            fig.add_trace(go.Scatter(
                x=timeline_df['time_hours'],
                y=timeline_df['itu_occupancy'],
                mode='lines',
                name='ITU Occupancy',
                line=dict(color='#ab63fa', width=2, dash='dot'),
            ))

        if 'theatre_occupancy' in timeline_df.columns:
            fig.add_trace(go.Scatter(
                x=timeline_df['time_hours'],
                y=timeline_df['theatre_occupancy'],
                mode='lines',
                name='Theatre Occupancy',
                line=dict(color='#ffa62b', width=2, dash='dash'),
            ))

        fig.add_trace(go.Bar(
            x=timeline_df['time_hours'],
            y=timeline_df['arrivals'],
            name='Arrivals (per 15min)',
            marker_color='rgba(0, 204, 150, 0.6)',
            yaxis='y2',
        ))

        # Add vertical line for selected time
        selected_time_hours = st.session_state['playback_selected_time'] / 60
        fig.add_vline(
            x=selected_time_hours,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"T={selected_time_hours:.1f}h",
            annotation_position="top",
        )

        # Get x-axis range from data
        min_time_hours = timeline_df['time_hours'].min()
        max_time_hours = timeline_df['time_hours'].max()

        # Update layout with explicit x-axis range
        fig.update_layout(
            title="Simulation Timeline (Click to Select Time)",
            xaxis_title="Simulation Time (hours)",
            yaxis_title="Patient Count",
            xaxis=dict(
                range=[min_time_hours, max_time_hours],
                dtick=2,  # Tick every 2 hours
            ),
            yaxis2=dict(
                title="Arrivals",
                overlaying='y',
                side='right',
                showgrid=False,
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=350,
            margin=dict(l=60, r=60, t=60, b=40),
        )

        # Render the timeline chart (standard plotly_chart for reliable rendering)
        st.plotly_chart(fig, use_container_width=True, key="timeline_chart")

        # Time selection slider (full width)
        min_time_hrs = (timeline_data['time'][0] if timeline_data['time'] else playback_scenario.warm_up) / 60
        max_time_hrs = (timeline_data['time'][-1] if timeline_data['time'] else playback_scenario.warm_up + playback_scenario.run_length) / 60

        # Format current selection for display in label
        current_hrs = int(st.session_state['playback_selected_time'] // 60)
        current_mins = int(st.session_state['playback_selected_time'] % 60)

        selected_time_hrs = st.slider(
            f"Select Time — Currently: T+{current_hrs}h {current_mins:02d}m",
            min_value=float(min_time_hrs),
            max_value=float(max_time_hrs),
            value=float(st.session_state['playback_selected_time'] / 60),
            step=0.25,  # 15-minute steps
            format="%.2f h",
            key="time_slider",
        )

        # Convert back to minutes for storage
        selected_time_mins = selected_time_hrs * 60
        if abs(selected_time_mins - st.session_state['playback_selected_time']) > 1:  # Allow small float tolerance
            st.session_state['playback_selected_time'] = selected_time_mins
            st.rerun()

        # Build and render schematic at selected time
        st.subheader("System State at Selected Time")

        schematic_data = build_schematic_at_time(
            patients=playback_patients,
            resource_logs=playback_resource_logs,
            timestamp=st.session_state['playback_selected_time'],
            scenario=playback_scenario,
        )

        # Summary metrics for this time point
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Patients in System", schematic_data.total_in_system)
        with metric_cols[1]:
            ed_node = schematic_data.nodes.get('ed_bays')
            if ed_node:
                st.metric("ED Occupancy", f"{ed_node.occupied}/{ed_node.capacity}")
        with metric_cols[2]:
            st.metric("Est. 24h Throughput", schematic_data.total_throughput_24h)
        with metric_cols[3]:
            status_emoji = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}.get(schematic_data.overall_status, "⚪")
            st.metric("Status", f"{status_emoji} {schematic_data.overall_status.title()}")

        # Render the schematic
        if is_built():
            clicked = react_schematic(
                data=to_dict(schematic_data),
                width=1200,
                height=650,
                key="playback_schematic",
            )

            if clicked and clicked in schematic_data.nodes:
                node = schematic_data.nodes[clicked]
                st.info(f"**{node.label}**: {node.occupied}/{node.capacity or '∞'} occupied, Util: {node.utilisation:.0%}, Wait: {node.mean_wait_mins:.1f}min")
        else:
            st.warning("""
            **React schematic not built.** To enable interactive visualization:
            ```bash
            cd app/components/react_schematic
            npm install && npm run build
            ```
            """)

            # Show basic text-based summary instead
            # Entry nodes (arrival streams)
            st.markdown("**Entry Nodes (Arrivals in Last Hour):**")
            entry_data = []
            for node_id in ['ambulance', 'hems', 'walkin']:
                node = schematic_data.nodes.get(node_id)
                if node:
                    entry_data.append({
                        "Source": node.label,
                        "Arrivals/hr": f"{node.throughput_per_hour:.0f}",
                    })
            if entry_data:
                st.dataframe(pd.DataFrame(entry_data), use_container_width=True, hide_index=True)

            # Resource nodes
            st.markdown("**Resource Occupancy at Selected Time:**")
            resource_data = []
            for node_id, node in schematic_data.nodes.items():
                if node.capacity and node.node_type in ['process', 'resource']:
                    resource_data.append({
                        "Resource": node.label,
                        "Occupied": node.occupied,
                        "Capacity": node.capacity,
                        "Utilisation": f"{node.utilisation:.0%}",
                        "Status": node.status.title(),
                    })
            if resource_data:
                st.dataframe(pd.DataFrame(resource_data), use_container_width=True, hide_index=True)

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
    # Priority arrivals and departures
    "arrivals_P1": results.get("arrivals_P1", [0] * n_reps),
    "arrivals_P2": results.get("arrivals_P2", [0] * n_reps),
    "arrivals_P3": results.get("arrivals_P3", [0] * n_reps),
    "arrivals_P4": results.get("arrivals_P4", [0] * n_reps),
    "departures_P1": results.get("departures_P1", [0] * n_reps),
    "departures_P2": results.get("departures_P2", [0] * n_reps),
    "departures_P3": results.get("departures_P3", [0] * n_reps),
    "departures_P4": results.get("departures_P4", [0] * n_reps),
    # Core metrics
    "p_delay": results["p_delay"],
    "mean_triage_wait": results["mean_triage_wait"],
    "mean_treatment_wait": results["mean_treatment_wait"],
    "mean_system_time": results["mean_system_time"],
    "p95_system_time": results["p95_system_time"],
    "admission_rate": results["admission_rate"],
    "admitted": results["admitted"],
    "discharged": results["discharged"],
    # Detailed disposition
    "discharged_count": results.get("discharged_count", results.get("discharged", [0] * n_reps)),
    "admitted_ward_count": results.get("admitted_ward_count", [0] * n_reps),
    "admitted_icu_count": results.get("admitted_icu_count", [0] * n_reps),
    "transfer_count": results.get("transfer_count", [0] * n_reps),
    "left_count": results.get("left_count", [0] * n_reps),
    # LoS by disposition
    "mean_los_discharged": results.get("mean_los_discharged", [0] * n_reps),
    "mean_los_ward": results.get("mean_los_ward", [0] * n_reps),
    "mean_los_icu": results.get("mean_los_icu", [0] * n_reps),
    "mean_los_transfer": results.get("mean_los_transfer", [0] * n_reps),
    # P1-P4 extended statistics
    "P1_mean_wait": results.get("P1_mean_wait", [0] * n_reps),
    "P2_mean_wait": results.get("P2_mean_wait", [0] * n_reps),
    "P3_mean_wait": results.get("P3_mean_wait", [0] * n_reps),
    "P4_mean_wait": results.get("P4_mean_wait", [0] * n_reps),
    "P1_p95_wait": results.get("P1_p95_wait", [0] * n_reps),
    "P2_p95_wait": results.get("P2_p95_wait", [0] * n_reps),
    "P3_p95_wait": results.get("P3_p95_wait", [0] * n_reps),
    "P4_p95_wait": results.get("P4_p95_wait", [0] * n_reps),
    "P1_breach_rate": results.get("P1_breach_rate", [0] * n_reps),
    "P2_breach_rate": results.get("P2_breach_rate", [0] * n_reps),
    "P3_breach_rate": results.get("P3_breach_rate", [0] * n_reps),
    "P4_breach_rate": results.get("P4_breach_rate", [0] * n_reps),
    "P1_mean_system_time": results.get("P1_mean_system_time", [0] * n_reps),
    "P2_mean_system_time": results.get("P2_mean_system_time", [0] * n_reps),
    "P3_mean_system_time": results.get("P3_mean_system_time", [0] * n_reps),
    "P4_mean_system_time": results.get("P4_mean_system_time", [0] * n_reps),
    # Acuity extended statistics
    "resus_mean_wait": results.get("resus_mean_wait", [0] * n_reps),
    "majors_mean_wait": results.get("majors_mean_wait", [0] * n_reps),
    "minors_mean_wait": results.get("minors_mean_wait", [0] * n_reps),
    "resus_p95_wait": results.get("resus_p95_wait", [0] * n_reps),
    "majors_p95_wait": results.get("majors_p95_wait", [0] * n_reps),
    "minors_p95_wait": results.get("minors_p95_wait", [0] * n_reps),
    "resus_mean_system_time": results.get("resus_mean_system_time", [0] * n_reps),
    "majors_mean_system_time": results.get("majors_mean_system_time", [0] * n_reps),
    "minors_mean_system_time": results.get("minors_mean_system_time", [0] * n_reps),
    "departures_resus": results.get("departures_resus", [0] * n_reps),
    "departures_majors": results.get("departures_majors", [0] * n_reps),
    "departures_minors": results.get("departures_minors", [0] * n_reps),
    # Emergency Services
    "util_ambulance_fleet": get_util("util_ambulance_fleet"),
    "util_helicopter_fleet": get_util("util_helicopter_fleet"),
    "util_handover": get_util("util_handover"),
    # Triage & ED
    "util_triage": get_util("util_triage"),
    "util_ed_bays": get_util("util_ed_bays"),
    # Diagnostics
    "util_ct": get_util("util_CT_SCAN"),
    "util_xray": get_util("util_XRAY"),
    "util_bloods": get_util("util_BLOODS"),
    # Downstream
    "util_theatre": get_util("util_theatre"),
    "util_itu": get_util("util_itu"),
    "util_ward": get_util("util_ward"),
    # Aeromed
    "aeromed_total": results.get("aeromed_total", [0] * n_reps),
    "aeromed_hems_count": results.get("aeromed_hems_count", [0] * n_reps),
    "aeromed_fixedwing_count": results.get("aeromed_fixedwing_count", [0] * n_reps),
    "aeromed_slots_missed": results.get("aeromed_slots_missed", [0] * n_reps),
    "mean_aeromed_slot_wait": results.get("mean_aeromed_slot_wait", [0] * n_reps),
    "ward_bed_days_blocked_aeromed": results.get("ward_bed_days_blocked_aeromed", [0] * n_reps),
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
        label="Download Raw Results (CSV)",
        data=csv,
        file_name="faer_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col2:
    # Create summary with CIs from export dataframe
    summary_data = []
    for metric in export_df.columns[1:]:  # Skip replication column
        values = export_df[metric].tolist()
        ci = compute_ci(values)
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
        label="Download Summary with CIs (CSV)",
        data=summary_csv,
        file_name="faer_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
