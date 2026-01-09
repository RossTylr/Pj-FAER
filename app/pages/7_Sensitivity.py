"""Sensitivity Analysis Page (Phase 6)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.core.scenario import FullScenario
from faer.experiment.analysis import sensitivity_sweep, find_breaking_point, identify_bottlenecks
from faer.model.full_model import run_full_simulation

st.set_page_config(page_title="Sensitivity - FAER", page_icon="📈", layout="wide")

st.title("📈 Sensitivity Analysis")

st.info("""
**What this does**: See how changing one thing affects performance.

**Use it for**:
- Finding optimal capacity levels
- Understanding diminishing returns
- Identifying safety margins
""")

# Mode selection
mode = st.radio(
    "Analysis Type",
    ["Sensitivity Sweep", "Find Breaking Point", "Bottleneck Analysis"],
    help="Sweep: test a range of values. Breaking Point: find exact threshold. Bottleneck: identify constraints."
)

st.divider()

# Initialize session state
if "sweep_results" not in st.session_state:
    st.session_state.sweep_results = None
if "breaking_point_result" not in st.session_state:
    st.session_state.breaking_point_result = None
if "bottleneck_result" not in st.session_state:
    st.session_state.bottleneck_result = None

# ===== SENSITIVITY SWEEP =====
if mode == "Sensitivity Sweep":
    st.header("Parameter Sweep")

    st.markdown("""
    **How to read results**:
    - Line shows average result at each value
    - Error bars show variation (uncertainty)
    - Flat line = no effect, steep line = big effect
    - Look for "diminishing returns" where line flattens
    """)

    # Parameter selection
    st.subheader("Parameter to Test")

    param_options = {
        'n_ed_bays': 'ED Bays',
        'n_triage': 'Triage Clinicians',
        'n_handover_bays': 'Handover Bays',
        'demand_multiplier': 'Demand Level',
        'n_ambulances': 'Ambulances'
    }

    param = st.selectbox(
        "Parameter",
        options=list(param_options.keys()),
        format_func=lambda x: param_options[x]
    )

    # Value range based on parameter
    col1, col2, col3 = st.columns(3)
    with col1:
        if param == 'demand_multiplier':
            min_val = st.number_input("Minimum", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
        else:
            min_val = st.number_input("Minimum", value=10, min_value=1, max_value=100)
    with col2:
        if param == 'demand_multiplier':
            max_val = st.number_input("Maximum", value=2.0, min_value=0.1, max_value=5.0, step=0.1)
        else:
            max_val = st.number_input("Maximum", value=30, min_value=1, max_value=100)
    with col3:
        steps = st.number_input("Steps", value=5, min_value=3, max_value=10)

    # Metric selection
    metric_options = {
        'p_delay': 'P(Delay)',
        'mean_treatment_wait': 'Mean Treatment Wait',
        'mean_system_time': 'Mean System Time',
        'util_ed_bays': 'ED Bay Utilisation',
        'util_triage': 'Triage Utilisation',
    }

    metric = st.selectbox(
        "Metric to Measure",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x]
    )

    # Simulation settings
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        run_hours = st.slider("Run Length (hours)", min_value=2, max_value=24, value=8, key="sweep_hours")
    with sim_col2:
        n_reps = st.slider("Replications per value", min_value=5, max_value=50, value=20, key="sweep_reps")

    if st.button("🔬 Run Sweep", type="primary", use_container_width=True):
        values = list(np.linspace(min_val, max_val, int(steps)))

        with st.spinner(f"Running sweep ({int(steps)} values × {n_reps} reps)..."):
            base_scenario = FullScenario(run_length=run_hours * 60.0, warm_up=60.0)

            result = sensitivity_sweep(
                base_scenario,
                param,
                values,
                metric,
                n_reps=n_reps
            )

            st.session_state.sweep_results = result

        st.success("Sweep complete!")

    # Display sweep results
    if st.session_state.sweep_results is not None:
        result = st.session_state.sweep_results

        st.subheader("Results")

        # Plot with confidence intervals
        df = result.to_dataframe()

        fig = go.Figure()

        # Add error bars
        fig.add_trace(go.Scatter(
            x=df['value'],
            y=df['mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=df['ci_upper'] - df['mean'],
                arrayminus=df['mean'] - df['ci_lower']
            ),
            mode='lines+markers',
            name=metric_options.get(result.metric, result.metric),
            line=dict(color='#636efa')
        ))

        # Add baseline reference
        fig.add_hline(
            y=df['mean'].iloc[0],
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline"
        )

        fig.update_layout(
            title=f"Effect of {param_options.get(result.parameter, result.parameter)} on {metric_options.get(result.metric, result.metric)}",
            xaxis_title=param_options.get(result.parameter, result.parameter),
            yaxis_title=metric_options.get(result.metric, result.metric),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("View Data"):
            st.dataframe(df.round(3), use_container_width=True)


# ===== BREAKING POINT =====
elif mode == "Find Breaking Point":
    st.header("Find Breaking Point")

    st.markdown("""
    **Example questions this answers**:
    - "At what demand level do we breach 50% delay rate?"
    - "How many beds can we close before wait times exceed 30 mins?"
    """)

    # Parameter to vary
    param_options = {
        'demand_multiplier': 'Demand Level',
        'n_ed_bays': 'ED Bays',
        'n_triage': 'Triage Clinicians',
    }

    param = st.selectbox(
        "Parameter to Vary",
        options=list(param_options.keys()),
        format_func=lambda x: param_options[x],
        key="bp_param"
    )

    # Search range
    col1, col2 = st.columns(2)
    with col1:
        if param == 'demand_multiplier':
            search_min = st.number_input("Search Min", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
        else:
            search_min = st.number_input("Search Min", value=5, min_value=1, max_value=50)
    with col2:
        if param == 'demand_multiplier':
            search_max = st.number_input("Search Max", value=3.0, min_value=0.1, max_value=5.0, step=0.1)
        else:
            search_max = st.number_input("Search Max", value=40, min_value=1, max_value=100)

    # Metric and threshold
    metric_options = {
        'p_delay': 'P(Delay)',
        'mean_treatment_wait': 'Mean Treatment Wait (min)',
        'util_ed_bays': 'ED Bay Utilisation',
    }

    metric = st.selectbox(
        "Metric to Monitor",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x],
        key="bp_metric"
    )

    threshold = st.number_input(
        "Threshold Value",
        value=0.5 if metric == 'p_delay' else 30.0 if metric == 'mean_treatment_wait' else 0.85,
        help="The value you want to find the crossing point for"
    )

    direction = st.radio(
        "Direction",
        ["above", "below"],
        format_func=lambda x: f"Find where metric goes {x} threshold"
    )

    # Simulation settings
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        run_hours = st.slider("Run Length (hours)", min_value=2, max_value=24, value=8, key="bp_hours")
    with sim_col2:
        n_reps = st.slider("Replications", min_value=5, max_value=30, value=10, key="bp_reps")

    if st.button("🔍 Find Breaking Point", type="primary", use_container_width=True):
        with st.spinner("Searching for breaking point..."):
            base_scenario = FullScenario(run_length=run_hours * 60.0, warm_up=60.0)

            result = find_breaking_point(
                base_scenario,
                param,
                metric,
                threshold=threshold,
                search_range=(search_min, search_max),
                direction=direction,
                n_reps=n_reps,
                max_iterations=10
            )

            st.session_state.breaking_point_result = result

        st.success("Search complete!")

    # Display breaking point results
    if st.session_state.breaking_point_result is not None:
        result = st.session_state.breaking_point_result

        st.subheader("Result")

        # Key finding
        st.markdown(f"""
        ### Breaking Point: **{result.breaking_point:.2f}**

        When **{param_options.get(result.parameter, result.parameter)}** reaches **{result.breaking_point:.2f}**,
        the **{metric_options.get(result.metric, result.metric)}** crosses **{result.threshold}**.

        At this point:
        - **{result.metric}** = {result.metric_at_break:.3f}
        - 95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]
        """)

        # Search history visualization
        if result.search_history:
            st.subheader("Search Progress")

            history_df = pd.DataFrame([
                {'Iteration': h['iteration'], 'Value Tested': h['value'], 'Metric Mean': h['mean']}
                for h in result.search_history if h['mean'] is not None
            ])

            if len(history_df) > 0:
                fig = px.line(
                    history_df,
                    x='Iteration',
                    y='Value Tested',
                    markers=True,
                    title="Binary Search Progress"
                )
                fig.add_hline(y=result.breaking_point, line_dash="dash", annotation_text="Final")
                st.plotly_chart(fig, use_container_width=True)


# ===== BOTTLENECK ANALYSIS =====
else:  # Bottleneck Analysis
    st.header("Bottleneck Analysis")

    st.markdown("""
    **What this does**: Identifies which part of the system is limiting performance.

    **Why it matters**: Adding beds to the wrong place wastes money.
    If Ward is the bottleneck, adding ED beds won't help.
    """)

    # Scenario configuration
    st.subheader("Scenario Configuration")

    col1, col2 = st.columns(2)
    with col1:
        n_ed_bays = st.number_input("ED Bays", min_value=5, max_value=50, value=20, key="bn_ed")
        n_triage = st.number_input("Triage Clinicians", min_value=1, max_value=10, value=2, key="bn_triage")
    with col2:
        n_handover = st.number_input("Handover Bays", min_value=1, max_value=10, value=4, key="bn_handover")
        demand_mult = st.slider("Demand Multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key="bn_demand")

    run_hours = st.slider("Run Length (hours)", min_value=4, max_value=24, value=8, key="bn_hours")

    if st.button("🔍 Analyse Bottlenecks", type="primary", use_container_width=True):
        with st.spinner("Running simulation and analysing bottlenecks..."):
            scenario = FullScenario(
                run_length=run_hours * 60.0,
                warm_up=60.0,
                n_ed_bays=n_ed_bays,
                n_triage=n_triage,
                n_handover_bays=n_handover,
                demand_multiplier=demand_mult
            )

            # Run a single simulation to get metrics
            results = run_full_simulation(scenario)

            # Identify bottlenecks
            bottleneck = identify_bottlenecks(results, scenario)

            st.session_state.bottleneck_result = bottleneck

        st.success("Analysis complete!")

    # Display bottleneck results
    if st.session_state.bottleneck_result is not None:
        result = st.session_state.bottleneck_result

        st.subheader("Results")

        # Primary bottleneck
        if result.primary_bottleneck != "None":
            st.error(f"**Primary Bottleneck**: {result.primary_bottleneck.replace('_', ' ').title()}")
        else:
            st.success("**No critical bottleneck identified.** System has adequate capacity.")

        # Recommendation
        st.markdown("### Recommendation")
        st.markdown(result.recommendation)

        # Utilisation chart
        st.subheader("Resource Utilisation")

        util_df = pd.DataFrame([
            {'Resource': name.replace('_', ' ').title(), 'Utilisation': util}
            for name, util in result.utilisation_ranking
        ])

        fig = px.bar(
            util_df,
            x='Resource',
            y='Utilisation',
            title="Resource Utilisation (Higher = More Constrained)",
            color='Utilisation',
            color_continuous_scale=['green', 'yellow', 'red'],
            range_color=[0, 1]
        )
        fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Critical (85%)")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.markdown("""
        ### How to Read This

        **Utilisation > 85%**: Resource is critically constrained (red flag)
        **Utilisation 70-85%**: Resource is busy but not blocking (yellow flag)
        **Utilisation < 70%**: Resource has spare capacity (green)

        **Common NHS Patterns**:
        1. Ward beds full → ED boarding → Ambulance delays
        2. ED bays full → Handover delays → Ambulance queues
        3. Triage constrained → Front door delays
        """)
