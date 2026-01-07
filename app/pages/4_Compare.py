"""Scenario Comparison Page (Phase 6)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.core.scenario import FullScenario
from faer.experiment.comparison import compare_scenarios

st.set_page_config(page_title="Compare - FAER", page_icon="⚖️", layout="wide")

st.title("⚖️ Compare Scenarios")

st.info("""
**What this does**: Run two scenarios side-by-side and see which performs better.

**Use it for**:
- Testing proposed changes before implementing
- Comparing different capacity configurations
- Evaluating staffing options
""")

# Initialize session state for comparison
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

# Two-column layout for scenario configuration
col_a, col_b = st.columns(2)

# ===== Scenario A (Baseline) =====
with col_a:
    st.subheader("Scenario A: Baseline")
    with st.expander("Configure Baseline", expanded=True):
        a_ed_bays = st.number_input("ED Bays (A)", min_value=5, max_value=50, value=20, key="a_ed")
        a_triage = st.number_input("Triage Clinicians (A)", min_value=1, max_value=10, value=2, key="a_triage")
        a_handover = st.number_input("Handover Bays (A)", min_value=1, max_value=10, value=4, key="a_handover")
        a_demand = st.slider("Demand Multiplier (A)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="a_demand")

# ===== Scenario B (Proposed) =====
with col_b:
    st.subheader("Scenario B: Proposed Change")
    with st.expander("Configure Proposed", expanded=True):
        b_ed_bays = st.number_input("ED Bays (B)", min_value=5, max_value=50, value=25, key="b_ed")
        b_triage = st.number_input("Triage Clinicians (B)", min_value=1, max_value=10, value=2, key="b_triage")
        b_handover = st.number_input("Handover Bays (B)", min_value=1, max_value=10, value=4, key="b_handover")
        b_demand = st.slider("Demand Multiplier (B)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="b_demand")

st.divider()

# Simulation parameters
st.subheader("Simulation Settings")
sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    run_hours = st.slider("Run Length (hours)", min_value=2, max_value=24, value=8)
with sim_col2:
    n_reps = st.slider("Replications (more = more reliable)", min_value=10, max_value=100, value=30)

# Metrics to compare
st.subheader("Metrics to Compare")
available_metrics = [
    'arrivals',
    'p_delay',
    'mean_triage_wait',
    'mean_treatment_wait',
    'mean_system_time',
    'util_triage',
    'util_ed_bays',
    'util_handover',
    'admission_rate',
]
selected_metrics = st.multiselect(
    "Select metrics",
    available_metrics,
    default=['p_delay', 'mean_treatment_wait', 'util_ed_bays']
)

st.divider()

# Run comparison button
if st.button("🔬 Run Comparison", type="primary", use_container_width=True):
    if not selected_metrics:
        st.error("Please select at least one metric to compare.")
    else:
        with st.spinner("Running simulations... This may take a few minutes."):
            # Build scenarios
            scenario_a = FullScenario(
                run_length=run_hours * 60.0,
                warm_up=60.0,
                n_ed_bays=a_ed_bays,
                n_triage=a_triage,
                n_handover_bays=a_handover,
                demand_multiplier=a_demand,
            )
            scenario_b = FullScenario(
                run_length=run_hours * 60.0,
                warm_up=60.0,
                n_ed_bays=b_ed_bays,
                n_triage=b_triage,
                n_handover_bays=b_handover,
                demand_multiplier=b_demand,
            )

            # Run comparison
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

# Display results if available
if st.session_state.compare_results is not None:
    result = st.session_state.compare_results

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
        if 'mean' in col or 'std' in col or 'difference' in col:
            if col in df.columns:
                df[col] = df[col].round(3)
        if 'pct_difference' in col:
            df[col] = df[col].round(1)
        if 'p_value' in col:
            df[col] = df[col].round(4)
        if 'effect_size' in col:
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
            'Value': row[f'Baseline_mean']
        })
        chart_data.append({
            'Metric': row['metric'],
            'Scenario': 'Proposed',
            'Value': row[f'Proposed_mean']
        })

    chart_df = pd.DataFrame(chart_data)

    fig = px.bar(
        chart_df,
        x='Metric',
        y='Value',
        color='Scenario',
        barmode='group',
        title="Scenario Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    st.subheader("How to Interpret")
    st.markdown("""
    **Statistical Significance (p-value)**:
    - p < 0.05: Difference is statistically significant (likely real)
    - p >= 0.05: Difference may be due to random variation

    **Effect Size**:
    - Negligible (< 0.2): Difference is trivial
    - Small (0.2 - 0.5): Noticeable but modest difference
    - Medium (0.5 - 0.8): Meaningful difference
    - Large (> 0.8): Substantial difference

    **Practical Significance**:
    Even if statistically significant, consider whether the difference
    matters in practice. A 1-minute improvement in wait time might be
    significant statistically but not clinically meaningful.
    """)

    # Download button
    st.divider()
    csv = result.metrics.to_csv(index=False)
    st.download_button(
        "📥 Download Results (CSV)",
        csv,
        "comparison_results.csv",
        "text/csv",
        use_container_width=True
    )
