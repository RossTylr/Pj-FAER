"""Enhanced Scenario Comparison Page (Phase 8).

Provides side-by-side comparison of two scenarios with all resources visible,
including difference highlighting and statistical analysis.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add app directory to path for component imports
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from faer.core.scenario import (
    FullScenario, RunLengthPreset, ITUConfig, WardConfig, TheatreConfig,
)
from faer.core.entities import NodeType, DiagnosticType
from faer.experiment.comparison import compare_scenarios
from components.scenario_summary import get_scenario_diff

st.set_page_config(page_title="Compare - FAER", page_icon="⚖️", layout="wide")

st.title("Compare Scenarios")

st.info("""
**Compare two scenarios side-by-side**

Configure resources for each scenario, then run the comparison
to see which performs better across key metrics.
""")

# Initialize session state for comparison
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

# ============ SCENARIO CONFIGURATION ============
col_a, col_divider, col_b = st.columns([5, 1, 5])

# ===== Scenario A (Baseline) =====
with col_a:
    st.header("Scenario A (Baseline)")

    with st.expander("Run Settings", expanded=True):
        a_run = st.selectbox(
            "Run Length",
            options=[p.value for p in RunLengthPreset],
            format_func=lambda x: f"{x//60}h ({x//60/24:.1f} days)" if x > 1440 else f"{x//60}h",
            index=1,  # Default to 24h
            key="a_run"
        )
        a_demand = st.slider("Demand", 0.5, 2.0, 1.0, 0.1, key="a_demand")

    with st.expander("Front Door", expanded=True):
        a_amb = st.number_input("Ambulances", 1, 20, 10, key="a_amb")
        a_heli = st.number_input("Helicopters", 0, 5, 2, key="a_heli")
        a_handover = st.number_input("Handover Bays", 1, 10, 4, key="a_handover")
        a_triage = st.number_input("Triage Clinicians", 1, 10, 2, key="a_triage")
        a_ed = st.number_input("ED Bays", 5, 50, 20, key="a_ed")

    with st.expander("Diagnostics"):
        a_ct = st.number_input("CT Scanners", 1, 5, 2, key="a_ct")
        a_xray = st.number_input("X-ray Rooms", 1, 10, 3, key="a_xray")
        a_bloods = st.number_input("Phlebotomists", 1, 10, 5, key="a_bloods")

    with st.expander("Downstream"):
        a_theatre = st.number_input("Theatre Tables", 1, 10, 2, key="a_theatre")
        a_theatre_session = st.number_input("Session Time (min)", 60, 480, 240, key="a_theatre_session")
        a_itu = st.number_input("ITU Beds", 1, 20, 6, key="a_itu")
        a_itu_los = st.number_input("ITU LoS (hours)", 12, 168, 48, key="a_itu_los")
        a_ward = st.number_input("Ward Beds", 10, 100, 30, key="a_ward")
        a_ward_los = st.number_input("Ward LoS (hours)", 24, 336, 72, key="a_ward_los")

# ===== Divider =====
with col_divider:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("### vs")

# ===== Scenario B (Proposed) =====
with col_b:
    st.header("Scenario B (Proposed)")

    with st.expander("Run Settings", expanded=True):
        b_run = st.selectbox(
            "Run Length",
            options=[p.value for p in RunLengthPreset],
            format_func=lambda x: f"{x//60}h ({x//60/24:.1f} days)" if x > 1440 else f"{x//60}h",
            index=1,  # Default to 24h
            key="b_run"
        )
        b_demand = st.slider("Demand", 0.5, 2.0, 1.0, 0.1, key="b_demand")

    with st.expander("Front Door", expanded=True):
        b_amb = st.number_input("Ambulances", 1, 20, 10, key="b_amb")
        b_heli = st.number_input("Helicopters", 0, 5, 2, key="b_heli")
        b_handover = st.number_input("Handover Bays", 1, 10, 4, key="b_handover")
        b_triage = st.number_input("Triage Clinicians", 1, 10, 2, key="b_triage")
        b_ed = st.number_input("ED Bays", 5, 50, 25, key="b_ed")  # Default higher

    with st.expander("Diagnostics"):
        b_ct = st.number_input("CT Scanners", 1, 5, 2, key="b_ct")
        b_xray = st.number_input("X-ray Rooms", 1, 10, 3, key="b_xray")
        b_bloods = st.number_input("Phlebotomists", 1, 10, 5, key="b_bloods")

    with st.expander("Downstream"):
        b_theatre = st.number_input("Theatre Tables", 1, 10, 2, key="b_theatre")
        b_theatre_session = st.number_input("Session Time (min)", 60, 480, 240, key="b_theatre_session")
        b_itu = st.number_input("ITU Beds", 1, 20, 6, key="b_itu")
        b_itu_los = st.number_input("ITU LoS (hours)", 12, 168, 48, key="b_itu_los")
        b_ward = st.number_input("Ward Beds", 10, 100, 30, key="b_ward")
        b_ward_los = st.number_input("Ward LoS (hours)", 24, 336, 72, key="b_ward_los")

# ============ DIFFERENCES SUMMARY ============
st.markdown("---")
st.subheader("Configuration Differences")

# Build comparison table
comparisons = [
    ("Run Length (h)", a_run // 60, b_run // 60),
    ("Demand", a_demand, b_demand),
    ("Ambulances", a_amb, b_amb),
    ("Helicopters", a_heli, b_heli),
    ("Handover Bays", a_handover, b_handover),
    ("Triage Clinicians", a_triage, b_triage),
    ("ED Bays", a_ed, b_ed),
    ("CT Scanners", a_ct, b_ct),
    ("X-ray Rooms", a_xray, b_xray),
    ("Phlebotomists", a_bloods, b_bloods),
    ("Theatre Tables", a_theatre, b_theatre),
    ("Theatre Session (min)", a_theatre_session, b_theatre_session),
    ("ITU Beds", a_itu, b_itu),
    ("ITU LoS (h)", a_itu_los, b_itu_los),
    ("Ward Beds", a_ward, b_ward),
    ("Ward LoS (h)", a_ward_los, b_ward_los),
]

diff_data = []
has_differences = False
for name, val_a, val_b in comparisons:
    diff = val_b - val_a
    pct = ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
    changed = diff != 0
    if changed:
        has_differences = True
    diff_data.append({
        "Resource": name,
        "Scenario A": val_a,
        "Scenario B": val_b,
        "Difference": f"{diff:+.1f}" if diff != 0 else "-",
        "% Change": f"{pct:+.1f}%" if diff != 0 else "-",
        "Changed": "Yes" if changed else ""
    })

diff_df = pd.DataFrame(diff_data)


# Highlight changed rows
def highlight_changes(row):
    if row['Changed'] == "Yes":
        return ['background-color: #FFFACD'] * len(row)
    return [''] * len(row)


st.dataframe(
    diff_df.style.apply(highlight_changes, axis=1),
    use_container_width=True,
    hide_index=True
)

if not has_differences:
    st.info("Both scenarios are identical. Change some parameters to compare.")

# ============ RUN COMPARISON ============
st.markdown("---")
st.subheader("Run Comparison")

sim_col1, sim_col2, sim_col3 = st.columns(3)

with sim_col1:
    n_reps = st.number_input("Replications", 10, 100, 30, help="More = more reliable but slower")

with sim_col2:
    warm_up = st.number_input("Warm-up (hours)", 0, 4, 1)

# Metrics to compare
available_metrics = [
    'arrivals',
    'p_delay',
    'mean_triage_wait',
    'mean_treatment_wait',
    'mean_system_time',
    'p95_system_time',
    'util_triage',
    'util_ed_bays',
    'util_handover',
    'mean_handover_delay',
    'mean_boarding_time',
    'p_boarding',
    'admission_rate',
]

with sim_col3:
    st.write("")  # Spacer

selected_metrics = st.multiselect(
    "Metrics to Compare",
    available_metrics,
    default=['p_delay', 'mean_treatment_wait', 'mean_system_time', 'util_ed_bays'],
    help="Select which metrics to analyze"
)

# Run button
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_comparison = st.button("Run Comparison", type="primary", use_container_width=True)

if run_comparison:
    if not selected_metrics:
        st.error("Please select at least one metric to compare.")
    else:
        # Build scenarios from inputs
        scenario_a = FullScenario(
            run_length=float(a_run),
            warm_up=warm_up * 60.0,
            demand_multiplier=a_demand,
            n_ambulances=a_amb,
            n_helicopters=a_heli,
            n_handover_bays=a_handover,
            n_triage=a_triage,
            n_ed_bays=a_ed,
            itu_config=ITUConfig(capacity=a_itu, los_mean_hours=float(a_itu_los)),
            ward_config=WardConfig(capacity=a_ward, los_mean_hours=float(a_ward_los)),
            theatre_config=TheatreConfig(
                n_tables=a_theatre,
                session_duration_mins=a_theatre_session,
            ),
        )

        scenario_b = FullScenario(
            run_length=float(b_run),
            warm_up=warm_up * 60.0,
            demand_multiplier=b_demand,
            n_ambulances=b_amb,
            n_helicopters=b_heli,
            n_handover_bays=b_handover,
            n_triage=b_triage,
            n_ed_bays=b_ed,
            itu_config=ITUConfig(capacity=b_itu, los_mean_hours=float(b_itu_los)),
            ward_config=WardConfig(capacity=b_ward, los_mean_hours=float(b_ward_los)),
            theatre_config=TheatreConfig(
                n_tables=b_theatre,
                session_duration_mins=b_theatre_session,
            ),
        )

        with st.spinner(f"Running {n_reps * 2} simulations..."):
            try:
                result = compare_scenarios(
                    scenario_a,
                    scenario_b,
                    metrics=selected_metrics,
                    n_reps=n_reps,
                    scenario_a_name="Baseline",
                    scenario_b_name="Proposed"
                )
                st.session_state.compare_results = result
                st.success("Comparison complete!")
            except Exception as e:
                st.error(f"Error running comparison: {e}")
                st.session_state.compare_results = None

# ============ DISPLAY RESULTS ============
if st.session_state.compare_results is not None:
    result = st.session_state.compare_results

    st.markdown("---")
    st.header("Results")

    # Summary in markdown
    st.markdown(result.summary)

    st.divider()

    # Detailed table
    st.subheader("Detailed Results")

    # Style the DataFrame
    df = result.metrics.copy()

    # Format numeric columns
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            if 'pct' in col.lower():
                df[col] = df[col].round(1)
            elif 'p_value' in col.lower():
                df[col] = df[col].round(4)
            else:
                df[col] = df[col].round(3)

    st.dataframe(df, use_container_width=True)

    # Visualization
    st.subheader("Comparison Chart")

    # Create bar chart comparing means
    chart_data = []
    for _, row in result.metrics.iterrows():
        chart_data.append({
            'Metric': row['metric'],
            'Scenario': 'Baseline',
            'Value': row['Baseline_mean']
        })
        chart_data.append({
            'Metric': row['metric'],
            'Scenario': 'Proposed',
            'Value': row['Proposed_mean']
        })

    chart_df = pd.DataFrame(chart_data)

    fig = px.bar(
        chart_df,
        x='Metric',
        y='Value',
        color='Scenario',
        barmode='group',
        title="Scenario Comparison",
        color_discrete_map={'Baseline': '#636EFA', 'Proposed': '#00CC96'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Effect size visualization
    st.subheader("Effect Sizes")

    effect_data = []
    for _, row in result.metrics.iterrows():
        effect = row.get('effect_size', 0)
        p_val = row.get('p_value', 1)
        sig = "*" if p_val < 0.05 else ""
        effect_data.append({
            'Metric': f"{row['metric']}{sig}",
            'Effect Size': effect,
            'Significant': p_val < 0.05
        })

    effect_df = pd.DataFrame(effect_data)

    fig2 = px.bar(
        effect_df,
        x='Metric',
        y='Effect Size',
        color='Significant',
        title="Effect Sizes (Cohen's d) - * = statistically significant",
        color_discrete_map={True: '#00CC96', False: '#CCCCCC'}
    )
    # Add reference lines for effect size interpretation
    fig2.add_hline(y=0.2, line_dash="dash", line_color="gray", annotation_text="Small")
    fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Medium")
    fig2.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="Large")
    fig2.add_hline(y=-0.2, line_dash="dash", line_color="gray")
    fig2.add_hline(y=-0.5, line_dash="dash", line_color="gray")
    fig2.add_hline(y=-0.8, line_dash="dash", line_color="gray")
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # Interpretation guide
    with st.expander("How to Interpret"):
        st.markdown("""
        **Statistical Significance (p-value)**:
        - p < 0.05: Difference is statistically significant (likely real)
        - p >= 0.05: Difference may be due to random variation

        **Effect Size (Cohen's d)**:
        - |d| < 0.2: Negligible - difference is trivial
        - 0.2 <= |d| < 0.5: Small - noticeable but modest
        - 0.5 <= |d| < 0.8: Medium - meaningful difference
        - |d| >= 0.8: Large - substantial difference

        **Positive vs Negative**:
        - Positive effect: Proposed (B) has higher values
        - Negative effect: Baseline (A) has higher values

        **Practical Significance**:
        Even if statistically significant, consider whether the difference
        matters in practice. A 1-minute improvement in wait time might be
        significant statistically but not clinically meaningful.
        """)

    # Download button
    st.divider()
    csv = result.metrics.to_csv(index=False)
    st.download_button(
        "Download Results (CSV)",
        csv,
        "comparison_results.csv",
        "text/csv",
        use_container_width=True
    )
