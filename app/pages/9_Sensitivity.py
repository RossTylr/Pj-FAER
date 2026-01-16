"""Sensitivity Analysis Page (Phase 6)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.core.scenario import FullScenario
from faer.experiment.analysis import sensitivity_sweep, find_breaking_point, identify_bottlenecks
from faer.model.full_model import run_full_simulation

st.set_page_config(page_title="Sensitivity - FAER", page_icon="", layout="wide")

st.title("Sensitivity Analysis")

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
        # Front Door
        'n_ambulances': 'Ambulances',
        'n_helicopters': 'Helicopters',
        'n_handover_bays': 'Handover Bays',
        'n_triage': 'Triage Clinicians',
        'n_ed_bays': 'ED Bays',
        # Demand
        'demand_multiplier': 'Demand Level',
        # Acuity
        'p_resus': 'Resus Proportion',
        'p_majors': 'Majors Proportion',
        # Downstream (use dot notation for nested configs)
        'theatre_config.n_tables': 'Theatre Tables',
        'itu_config.capacity': 'ITU Beds',
        'ward_config.capacity': 'Ward Beds',
        # Aeromed
        'aeromed_config.hems.slots_per_day': 'HEMS Slots/Day',
    }

    param = st.selectbox(
        "Parameter",
        options=list(param_options.keys()),
        format_func=lambda x: param_options[x]
    )

    # Value range based on parameter type
    col1, col2, col3 = st.columns(3)

    # Define defaults for each parameter type (min_default, max_default, abs_min, abs_max)
    param_defaults = {
        # Front Door
        'n_ambulances': (5, 20, 1, 50),
        'n_helicopters': (1, 5, 0, 10),
        'n_handover_bays': (2, 8, 1, 20),
        'n_triage': (1, 5, 1, 10),
        'n_ed_bays': (10, 30, 5, 50),
        # Demand (float)
        'demand_multiplier': (0.5, 2.0, 0.1, 5.0),
        # Acuity (float, proportions)
        'p_resus': (0.02, 0.15, 0.01, 0.30),
        'p_majors': (0.40, 0.70, 0.20, 0.80),
        # Downstream (nested config paths)
        'theatre_config.n_tables': (1, 4, 1, 10),
        'itu_config.capacity': (3, 12, 1, 30),
        'ward_config.capacity': (15, 50, 10, 100),
        # Aeromed
        'aeromed_config.hems.slots_per_day': (2, 10, 0, 20),
    }

    defaults = param_defaults.get(param, (1, 10, 1, 50))
    is_float_param = param in ['demand_multiplier', 'p_resus', 'p_majors']

    with col1:
        if is_float_param:
            min_val = st.number_input("Minimum", value=defaults[0], min_value=defaults[2], max_value=defaults[3], step=0.05)
        else:
            min_val = st.number_input("Minimum", value=defaults[0], min_value=defaults[2], max_value=defaults[3])
    with col2:
        if is_float_param:
            max_val = st.number_input("Maximum", value=defaults[1], min_value=defaults[2], max_value=defaults[3], step=0.05)
        else:
            max_val = st.number_input("Maximum", value=defaults[1], min_value=defaults[2], max_value=defaults[3])
    with col3:
        steps = st.number_input("Steps", value=5, min_value=3, max_value=10)

    # Metric selection
    metric_options = {
        # Core flow metrics
        'p_delay': 'P(Delay)',
        'mean_triage_wait': 'Mean Triage Wait',
        'mean_treatment_wait': 'Mean Treatment Wait',
        'mean_system_time': 'Mean System Time',
        'p95_system_time': '95th Percentile System Time',
        'admission_rate': 'Admission Rate',
        # Emergency services
        'util_ambulance_fleet': 'Ambulance Fleet Utilisation',
        'util_helicopter_fleet': 'Helicopter Fleet Utilisation',
        'util_handover': 'Handover Utilisation',
        'mean_handover_delay': 'Mean Handover Delay',
        # Triage & ED
        'util_triage': 'Triage Utilisation',
        'util_ed_bays': 'ED Bay Utilisation',
        'mean_boarding_time': 'Mean Boarding Time',
        'p_boarding': 'P(Boarding)',
        # Diagnostics
        'util_CT_SCAN': 'CT Scanner Utilisation',
        'util_XRAY': 'X-ray Utilisation',
        'util_BLOODS': 'Bloods Utilisation',
        # Downstream
        'util_theatre': 'Theatre Utilisation',
        'util_itu': 'ITU Utilisation',
        'util_ward': 'Ward Utilisation',
        # Aeromed
        'aeromed_total': 'Total Aeromed Evacuations',
        'aeromed_slots_missed': 'Aeromed Slots Missed',
        'mean_aeromed_slot_wait': 'Mean Aeromed Slot Wait',
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

    if st.button("Run Sweep", type="primary", use_container_width=True):
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
    - "How many HEMS slots do we need to avoid missed evacuations?"
    """)

    # Parameter to vary
    bp_param_options = {
        # Front Door
        'n_ambulances': 'Ambulances',
        'n_helicopters': 'Helicopters',
        'n_handover_bays': 'Handover Bays',
        'n_triage': 'Triage Clinicians',
        'n_ed_bays': 'ED Bays',
        # Demand
        'demand_multiplier': 'Demand Level',
        # Acuity
        'p_resus': 'Resus Proportion',
        # Downstream (use dot notation for nested configs)
        'theatre_config.n_tables': 'Theatre Tables',
        'itu_config.capacity': 'ITU Beds',
        'ward_config.capacity': 'Ward Beds',
        # Aeromed
        'aeromed_config.hems.slots_per_day': 'HEMS Slots/Day',
    }

    param = st.selectbox(
        "Parameter to Vary",
        options=list(bp_param_options.keys()),
        format_func=lambda x: bp_param_options[x],
        key="bp_param"
    )

    # Define defaults for search range
    bp_param_defaults = {
        'n_ambulances': (5, 25, 1, 50),
        'n_helicopters': (1, 8, 0, 15),
        'n_handover_bays': (2, 10, 1, 20),
        'n_triage': (1, 8, 1, 15),
        'n_ed_bays': (5, 40, 1, 60),
        'demand_multiplier': (0.5, 3.0, 0.1, 5.0),
        'p_resus': (0.02, 0.20, 0.01, 0.30),
        'theatre_config.n_tables': (1, 6, 1, 10),
        'itu_config.capacity': (2, 20, 1, 40),
        'ward_config.capacity': (10, 80, 5, 150),
        'aeromed_config.hems.slots_per_day': (1, 15, 0, 24),
    }

    bp_defaults = bp_param_defaults.get(param, (1, 50, 1, 100))
    is_bp_float = param in ['demand_multiplier', 'p_resus']

    # Search range
    col1, col2 = st.columns(2)
    with col1:
        if is_bp_float:
            search_min = st.number_input("Search Min", value=bp_defaults[0], min_value=bp_defaults[2], max_value=bp_defaults[3], step=0.1)
        else:
            search_min = st.number_input("Search Min", value=bp_defaults[0], min_value=bp_defaults[2], max_value=bp_defaults[3])
    with col2:
        if is_bp_float:
            search_max = st.number_input("Search Max", value=bp_defaults[1], min_value=bp_defaults[2], max_value=bp_defaults[3], step=0.1)
        else:
            search_max = st.number_input("Search Max", value=bp_defaults[1], min_value=bp_defaults[2], max_value=bp_defaults[3])

    # Metric and threshold
    bp_metric_options = {
        # Core flow
        'p_delay': 'P(Delay)',
        'mean_triage_wait': 'Mean Triage Wait (min)',
        'mean_treatment_wait': 'Mean Treatment Wait (min)',
        'mean_system_time': 'Mean System Time (min)',
        'p95_system_time': '95th Percentile System Time (min)',
        # Utilisation
        'util_ambulance_fleet': 'Ambulance Fleet Utilisation',
        'util_helicopter_fleet': 'Helicopter Fleet Utilisation',
        'util_handover': 'Handover Utilisation',
        'util_triage': 'Triage Utilisation',
        'util_ed_bays': 'ED Bay Utilisation',
        'util_CT_SCAN': 'CT Scanner Utilisation',
        'util_XRAY': 'X-ray Utilisation',
        'util_theatre': 'Theatre Utilisation',
        'util_itu': 'ITU Utilisation',
        'util_ward': 'Ward Utilisation',
        # Aeromed
        'aeromed_slots_missed': 'Aeromed Slots Missed',
    }

    metric = st.selectbox(
        "Metric to Monitor",
        options=list(bp_metric_options.keys()),
        format_func=lambda x: bp_metric_options[x],
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

    if st.button("Find Breaking Point", type="primary", use_container_width=True):
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

        When **{bp_param_options.get(result.parameter, result.parameter)}** reaches **{result.breaking_point:.2f}**,
        the **{bp_metric_options.get(result.metric, result.metric)}** crosses **{result.threshold}**.

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

    with st.expander("Front Door & ED", expanded=True):
        bn_col1, bn_col2 = st.columns(2)
        with bn_col1:
            n_ambulances = st.number_input("Ambulances", min_value=1, max_value=30, value=10, key="bn_amb")
            n_helicopters = st.number_input("Helicopters", min_value=0, max_value=10, value=2, key="bn_heli")
            n_handover = st.number_input("Handover Bays", min_value=1, max_value=15, value=4, key="bn_handover")
        with bn_col2:
            n_triage = st.number_input("Triage Clinicians", min_value=1, max_value=10, value=2, key="bn_triage")
            n_ed_bays = st.number_input("ED Bays", min_value=5, max_value=50, value=20, key="bn_ed")
            demand_mult = st.slider("Demand Multiplier", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key="bn_demand")

    with st.expander("Diagnostics"):
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        with diag_col1:
            n_ct = st.number_input("CT Scanners", min_value=1, max_value=10, value=2, key="bn_ct")
        with diag_col2:
            n_xray = st.number_input("X-ray Rooms", min_value=1, max_value=10, value=3, key="bn_xray")
        with diag_col3:
            n_bloods = st.number_input("Phlebotomists", min_value=1, max_value=15, value=5, key="bn_bloods")

    with st.expander("Downstream"):
        ds_col1, ds_col2, ds_col3 = st.columns(3)
        with ds_col1:
            n_theatre = st.number_input("Theatre Tables", min_value=1, max_value=10, value=2, key="bn_theatre")
        with ds_col2:
            n_itu = st.number_input("ITU Beds", min_value=1, max_value=30, value=6, key="bn_itu")
        with ds_col3:
            n_ward = st.number_input("Ward Beds", min_value=10, max_value=100, value=30, key="bn_ward")
        downstream_enabled = st.checkbox("Enable Downstream Simulation", value=False, key="bn_downstream")

    run_hours = st.slider("Run Length (hours)", min_value=4, max_value=24, value=8, key="bn_hours")

    if st.button("Analyse Bottlenecks", type="primary", use_container_width=True):
        with st.spinner("Running simulation and analysing bottlenecks..."):
            from faer.core.scenario import ITUConfig, WardConfig, TheatreConfig

            scenario = FullScenario(
                run_length=run_hours * 60.0,
                warm_up=60.0,
                n_ambulances=n_ambulances,
                n_helicopters=n_helicopters,
                n_handover_bays=n_handover,
                n_triage=n_triage,
                n_ed_bays=n_ed_bays,
                demand_multiplier=demand_mult,
                downstream_enabled=downstream_enabled,
                itu_config=ITUConfig(capacity=n_itu),
                ward_config=WardConfig(capacity=n_ward),
                theatre_config=TheatreConfig(n_tables=n_theatre),
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
