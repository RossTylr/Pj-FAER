"""Sensitivity Analysis Page (Phase 6).

Enhanced with AI Agent insights and parameter-metric linkage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from faer.core.scenario import FullScenario, ITUConfig, WardConfig, TheatreConfig
from faer.core.scaling import CapacityScalingConfig, OPELConfig
from faer.experiment.analysis import sensitivity_sweep, find_breaking_point, identify_bottlenecks
from faer.experiment.runner import multiple_replications
from faer.model.full_model import run_full_simulation

# AI Agent imports
from faer.agents.interface import MetricsSummary, Severity
from faer.agents.shadow import HeuristicShadowAgent
from faer.agents.orchestrator import AgentOrchestrator, OrchestratorConfig
from faer.agents.parameter_metrics import (
    get_affected_metrics,
    get_influencing_parameters,
    get_parameter_guidance,
    get_metric_info,
    get_sweep_suggestions,
    PARAMETER_GUIDANCE,
    METRIC_INFO,
)

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
    ["Sensitivity Sweep", "Find Breaking Point", "Bottleneck Analysis", "OPEL Optimization", "AI Agent Explainer"],
    help="Sweep: test a range of values. Breaking Point: find exact threshold. Bottleneck: identify constraints. OPEL: optimize scaling thresholds. AI Agent: understand how insights are generated."
)

st.divider()

# Initialize session state
if "sweep_results" not in st.session_state:
    st.session_state.sweep_results = None
if "breaking_point_result" not in st.session_state:
    st.session_state.breaking_point_result = None
if "bottleneck_result" not in st.session_state:
    st.session_state.bottleneck_result = None
if "sweep_insights" not in st.session_state:
    st.session_state.sweep_insights = None


# ============ HELPER FUNCTION: RUN AI INSIGHTS ============
def run_ai_insights(results_dict: dict, scenario_name: str = "Sensitivity Run") -> list:
    """Run AI agent analysis on simulation results.

    Args:
        results_dict: Dict with metric names as keys and values (or lists)
        scenario_name: Name for the scenario

    Returns:
        List of ClinicalInsight objects
    """
    # Convert to MetricsSummary format
    # If results are single values, wrap in lists for from_run_results
    wrapped_results = {}
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            wrapped_results[key] = [value]
        else:
            wrapped_results[key] = value

    metrics = MetricsSummary.from_run_results(
        wrapped_results,
        scenario_name=scenario_name,
        compute_confidence_intervals=True,
    )

    # Run the heuristic agent
    agent = HeuristicShadowAgent()
    insights = agent.analyze(metrics)

    return insights


def display_insights(insights: list, title: str = "AI Clinical Insights"):
    """Display AI insights with proper formatting.

    Args:
        insights: List of ClinicalInsight objects
        title: Section title
    """
    if not insights:
        st.info("No clinical alerts triggered for these results.")
        return

    st.subheader(title)

    # Count by severity
    severity_counts = {}
    for insight in insights:
        sev = insight.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Summary badges
    badge_cols = st.columns(5)
    severity_colors = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "yellow",
        "LOW": "blue",
        "INFO": "gray",
    }
    severity_icons = {
        "CRITICAL": "!!",
        "HIGH": "!",
        "MEDIUM": "~",
        "LOW": "i",
        "INFO": "-",
    }

    for i, sev in enumerate(["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]):
        count = severity_counts.get(sev, 0)
        with badge_cols[i]:
            if count > 0:
                st.metric(sev, count)
            else:
                st.metric(sev, "-")

    st.divider()

    # Display each insight
    for insight in insights:
        sev = insight.severity.value
        icon = severity_icons.get(sev, "?")

        # Determine container color based on severity
        if sev == "CRITICAL":
            st.error(f"**{icon} {insight.title}**")
        elif sev == "HIGH":
            st.warning(f"**{icon} {insight.title}**")
        elif sev == "MEDIUM":
            st.warning(f"**{icon} {insight.title}**")
        else:
            st.info(f"**{icon} {insight.title}**")

        st.markdown(insight.message)

        if insight.recommendation:
            st.markdown(f"**Recommendation:** {insight.recommendation}")

        # Show confidence info if available
        if insight.uncertainty_note:
            st.caption(insight.uncertainty_note)

        # Show evidence metrics
        if insight.evidence:
            with st.expander("Evidence"):
                evidence_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.3f}" if isinstance(v, float) else str(v)}
                    for k, v in insight.evidence.items()
                ])
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)

        st.markdown("---")

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
        # Capacity Scaling (Phase 12)
        'capacity_scaling.opel_config.opel_3_ed_threshold': 'OPEL 3 ED Trigger',
        'capacity_scaling.opel_config.opel_3_ed_surge_beds': 'OPEL 3 ED Surge Beds',
        'capacity_scaling.opel_config.opel_3_ward_surge_beds': 'OPEL 3 Ward Surge Beds',
        'capacity_scaling.opel_config.opel_3_itu_surge_beds': 'OPEL 3 ITU Surge Beds',
        'capacity_scaling.opel_config.opel_4_ed_threshold': 'OPEL 4 ED Trigger',
        'capacity_scaling.opel_config.opel_4_ed_surge_beds': 'OPEL 4 ED Surge Beds',
        'capacity_scaling.opel_config.opel_4_ward_surge_beds': 'OPEL 4 Ward Surge Beds',
        'capacity_scaling.opel_config.opel_4_itu_surge_beds': 'OPEL 4 ITU Surge Beds',
        'capacity_scaling.opel_config.opel_3_los_reduction_pct': 'Discharge Acceleration %',
        'capacity_scaling.discharge_lounge_capacity': 'Discharge Lounge Capacity',
    }

    param = st.selectbox(
        "Parameter",
        options=list(param_options.keys()),
        format_func=lambda x: param_options[x]
    )

    # === PARAMETER INTELLIGENCE PANEL ===
    # Show guidance for selected parameter
    guidance = get_parameter_guidance(param)
    if guidance:
        with st.expander("Parameter Intelligence (AI-Powered)", expanded=True):
            st.markdown(f"**{guidance.display_name}**: {guidance.description}")

            if guidance.clinical_context:
                st.markdown(f"*Clinical Context*: {guidance.clinical_context}")

            # Show affected metrics
            if guidance.primary_metrics:
                st.markdown("**Primary metrics affected:**")
                for effect in guidance.affected_metrics:
                    if effect.metric in guidance.primary_metrics:
                        direction_icon = "+" if effect.direction.value == "increase" else "-" if effect.direction.value == "decrease" else "~"
                        mag_label = f"({effect.magnitude.value})"
                        metric_info = get_metric_info(effect.metric)
                        display_name = metric_info.display_name if metric_info else effect.metric
                        st.markdown(f"- **{display_name}** {direction_icon} {mag_label}: {effect.explanation}")

            # Show interaction notes
            if guidance.interaction_notes:
                st.markdown("**Watch out for:**")
                for note in guidance.interaction_notes:
                    st.markdown(f"- {note}")

            # Recommended range
            st.markdown(f"**Recommended sweep range**: {guidance.typical_range[0]} - {guidance.typical_range[1]}")
            if guidance.diminishing_returns_threshold:
                st.markdown(f"**Diminishing returns above**: {guidance.diminishing_returns_threshold}")
    else:
        # Fallback for parameters without detailed guidance
        with st.expander("Parameter Info"):
            st.markdown(f"No detailed guidance available for `{param}`. Consider testing with the suggested defaults.")

    st.divider()

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
        # Capacity Scaling (Phase 12)
        'capacity_scaling.opel_config.opel_3_ed_threshold': (0.80, 0.95, 0.70, 1.0),
        'capacity_scaling.opel_config.opel_3_ed_surge_beds': (2, 10, 0, 20),
        'capacity_scaling.opel_config.opel_3_ward_surge_beds': (1, 6, 0, 15),
        'capacity_scaling.opel_config.opel_3_itu_surge_beds': (0, 3, 0, 6),
        'capacity_scaling.opel_config.opel_4_ed_threshold': (0.90, 1.0, 0.80, 1.0),
        'capacity_scaling.opel_config.opel_4_ed_surge_beds': (5, 15, 0, 30),
        'capacity_scaling.opel_config.opel_4_ward_surge_beds': (3, 10, 0, 20),
        'capacity_scaling.opel_config.opel_4_itu_surge_beds': (1, 4, 0, 10),
        'capacity_scaling.opel_config.opel_3_los_reduction_pct': (5.0, 20.0, 0.0, 30.0),
        'capacity_scaling.discharge_lounge_capacity': (5, 15, 0, 30),
    }

    defaults = param_defaults.get(param, (1, 10, 1, 50))
    is_float_param = param in [
        'demand_multiplier', 'p_resus', 'p_majors',
        'capacity_scaling.opel_config.opel_3_ed_threshold',
        'capacity_scaling.opel_config.opel_4_ed_threshold',
        'capacity_scaling.opel_config.opel_3_los_reduction_pct',
    ]

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
        # Capacity Scaling (Phase 12)
        'opel_peak_level': 'Peak OPEL Level',
        'opel_transitions': 'OPEL Transitions',
        'pct_time_at_surge': '% Time at Surge Capacity',
        'total_additional_bed_hours': 'Additional Surge Bed-Hours',
        'patients_diverted': 'Patients Diverted',
        'scale_up_events': 'Scale-Up Events',
        'scale_down_events': 'Scale-Down Events',
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
elif mode == "Bottleneck Analysis":
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

        # OPEL-based recommendations
        if result.primary_bottleneck in ['ed_bays', 'ward_beds']:
            util_level = dict(result.utilisation_ranking).get(result.primary_bottleneck, 0)
            if util_level > 0.85:
                st.markdown("""
                ### OPEL-Based Response Options

                **At this utilisation level, OPEL escalation would typically trigger:**

                **OPEL 3 (Severe Pressure) - 85-90% utilisation:**
                - Activate surge capacity (+5 beds)
                - Enable discharge acceleration (10% LoS reduction)
                - Open discharge lounge to free beds faster
                - Focus on flow initiatives

                **OPEL 4 (Critical) - >90% utilisation:**
                - Full surge capacity (+10 beds)
                - Aggressive discharge acceleration (15% LoS reduction)
                - Consider ambulance diversion for P3/P4 patients
                - Escalate to system-level response

                *Enable Capacity Scaling in the Compare page to simulate these responses.*
                """)


# ===== OPEL OPTIMIZATION =====
elif mode == "OPEL Optimization":
    st.header("OPEL Threshold Optimization")

    st.markdown("""
    **Find optimal OPEL trigger thresholds** that balance:
    - Patient flow (minimize delays)
    - Resource efficiency (minimize surge capacity usage)
    - Safety (avoid reaching OPEL 4)

    This runs a grid search over OPEL threshold combinations and shows
    how different settings affect key metrics.
    """)

    # Initialize session state
    if "opel_optimization_results" not in st.session_state:
        st.session_state.opel_optimization_results = None

    # Base scenario configuration
    st.subheader("Base Scenario")

    with st.expander("Resource Configuration", expanded=True):
        opel_col1, opel_col2 = st.columns(2)
        with opel_col1:
            opel_n_ed = st.number_input("ED Bays", 10, 50, 20, key="opel_ed")
            opel_n_triage = st.number_input("Triage Clinicians", 1, 10, 3, key="opel_triage")
            opel_n_ward = st.number_input("Ward Beds", 10, 100, 30, key="opel_ward")
        with opel_col2:
            opel_demand = st.slider("Demand Multiplier", 0.5, 3.0, 1.5, 0.1, key="opel_demand")
            opel_run_hours = st.slider("Run Length (hours)", 4, 24, 8, key="opel_run_hours")
            opel_n_reps = st.slider("Replications per combo", 3, 15, 5, key="opel_reps")

    # Optimization grid settings
    st.subheader("OPEL Threshold Grid")

    grid_col1, grid_col2 = st.columns(2)
    with grid_col1:
        st.markdown("**OPEL 3 Thresholds to Test**")
        opel_3_min = st.slider("Min", 0.75, 0.90, 0.80, 0.05, key="opel3_min")
        opel_3_max = st.slider("Max", 0.85, 0.95, 0.95, 0.05, key="opel3_max")
    with grid_col2:
        st.markdown("**OPEL 4 Thresholds to Test**")
        opel_4_min = st.slider("Min", 0.85, 0.95, 0.90, 0.05, key="opel4_min")
        opel_4_max = st.slider("Max", 0.90, 1.00, 1.00, 0.05, key="opel4_max")

    # Metric to optimize
    opel_metric = st.selectbox(
        "Primary Metric to Optimize",
        ['mean_system_time', 'pct_time_at_surge', 'patients_diverted', 'opel_peak_level'],
        format_func=lambda x: {
            'mean_system_time': 'Mean System Time (minimize)',
            'pct_time_at_surge': '% Time at Surge (minimize)',
            'patients_diverted': 'Patients Diverted (minimize)',
            'opel_peak_level': 'Peak OPEL Level (minimize)',
        }.get(x, x)
    )

    if st.button("Run OPEL Optimization", type="primary", use_container_width=True):
        # Generate threshold combinations
        opel_3_thresholds = np.arange(opel_3_min, opel_3_max + 0.01, 0.05)
        opel_4_thresholds = np.arange(opel_4_min, opel_4_max + 0.01, 0.05)

        valid_combinations = [
            (t3, t4) for t3 in opel_3_thresholds for t4 in opel_4_thresholds if t4 > t3
        ]

        total_runs = len(valid_combinations) * opel_n_reps
        st.info(f"Running {len(valid_combinations)} threshold combinations × {opel_n_reps} reps = {total_runs} simulations")

        results_data = []
        progress_bar = st.progress(0)

        for i, (t3, t4) in enumerate(valid_combinations):
            # Create scenario with this OPEL configuration
            scenario = FullScenario(
                run_length=opel_run_hours * 60.0,
                warm_up=60.0,
                n_ed_bays=opel_n_ed,
                n_triage=opel_n_triage,
                demand_multiplier=opel_demand,
                downstream_enabled=True,
                ward_config=WardConfig(capacity=opel_n_ward),
                capacity_scaling=CapacityScalingConfig(
                    enabled=True,
                    opel_config=OPELConfig(
                        enabled=True,
                        opel_3_ed_threshold=t3,
                        opel_3_ward_threshold=t3,
                        opel_3_itu_threshold=t3,
                        opel_3_ed_surge_beds=5,
                        opel_3_ward_surge_beds=3,
                        opel_3_itu_surge_beds=1,
                        opel_3_los_reduction_pct=10.0,
                        opel_4_ed_threshold=t4,
                        opel_4_ward_threshold=t4,
                        opel_4_itu_threshold=t4,
                        opel_4_ed_surge_beds=10,
                        opel_4_ward_surge_beds=6,
                        opel_4_itu_surge_beds=2,
                        opel_4_enable_divert=True,
                    ),
                    discharge_lounge_capacity=10,
                ),
            )

            # Run replications
            metrics_to_collect = [
                'mean_system_time', 'pct_time_at_surge', 'patients_diverted',
                'opel_peak_level', 'scale_up_events', 'util_ed_bays'
            ]

            try:
                rep_results = multiple_replications(scenario, opel_n_reps, metrics_to_collect)

                results_data.append({
                    'opel_3_threshold': t3,
                    'opel_4_threshold': t4,
                    'mean_system_time': np.mean(rep_results.get('mean_system_time', [0])),
                    'pct_time_at_surge': np.mean(rep_results.get('pct_time_at_surge', [0])),
                    'patients_diverted': np.mean(rep_results.get('patients_diverted', [0])),
                    'opel_peak_level': np.mean(rep_results.get('opel_peak_level', [1])),
                    'scale_up_events': np.mean(rep_results.get('scale_up_events', [0])),
                    'util_ed_bays': np.mean(rep_results.get('util_ed_bays', [0])),
                })
            except Exception as e:
                st.warning(f"Error running combination ({t3:.2f}, {t4:.2f}): {e}")

            progress_bar.progress((i + 1) / len(valid_combinations))

        st.session_state.opel_optimization_results = pd.DataFrame(results_data)
        st.success("Optimization complete!")

    # Display results
    if st.session_state.opel_optimization_results is not None:
        df = st.session_state.opel_optimization_results

        st.subheader("Results")

        # Heatmap
        st.markdown("### Heatmap: OPEL Thresholds vs " + opel_metric.replace('_', ' ').title())

        # Pivot for heatmap
        pivot_df = df.pivot(index='opel_4_threshold', columns='opel_3_threshold', values=opel_metric)

        fig = px.imshow(
            pivot_df,
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='RdYlGn_r',  # Red=high (bad), Green=low (good)
            labels={'x': 'OPEL 3 Threshold', 'y': 'OPEL 4 Threshold', 'color': opel_metric},
            title=f"Effect of OPEL Thresholds on {opel_metric.replace('_', ' ').title()}"
        )
        fig.update_layout(
            xaxis_title="OPEL 3 Threshold",
            yaxis_title="OPEL 4 Threshold"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Find optimal combination
        optimal_idx = df[opel_metric].idxmin()
        optimal = df.loc[optimal_idx]

        st.markdown(f"""
        ### Optimal Configuration

        **Best OPEL 3 Threshold**: {optimal['opel_3_threshold']:.0%}
        **Best OPEL 4 Threshold**: {optimal['opel_4_threshold']:.0%}

        At these settings:
        - Mean System Time: {optimal['mean_system_time']:.1f} minutes
        - % Time at Surge: {optimal['pct_time_at_surge']:.1f}%
        - Patients Diverted: {optimal['patients_diverted']:.1f}
        - Peak OPEL Level: {optimal['opel_peak_level']:.1f}
        """)

        # Full results table
        with st.expander("View All Results"):
            st.dataframe(df.round(3), use_container_width=True)

# ===== AI AGENT EXPLAINER =====
elif mode == "AI Agent Explainer":
    st.header("AI Agent Explainer")

    st.info("""
    **How FAER's AI Clinical Agent Works**

    The AI agent analyzes simulation results and generates clinical insights
    based on NHS standards, clinical guidelines, and queuing theory principles.
    This page explains how insights are generated and how to interpret them.
    """)

    # Overview
    st.subheader("Agent Architecture")

    st.markdown("""
    The **Heuristic Shadow Agent** applies rule-based clinical thresholds to simulation metrics.
    It serves as a deterministic, testable baseline that can later be augmented with LLM-based analysis.

    **Key Features:**
    - **Single-metric thresholds**: NHS standards (4-hour wait, handover targets)
    - **Compound rules**: Multi-metric risk patterns (e.g., high acuity + long waits)
    - **Uncertainty-aware**: Considers confidence intervals when alerting
    - **Expert perspectives**: EM consultant, anaesthetist, surgeon, paramedic viewpoints
    """)

    st.divider()

    # Threshold Rules Explanation
    st.subheader("Clinical Threshold Rules")

    st.markdown("""
    The agent monitors these key metrics against clinical thresholds:
    """)

    # Import threshold rules from shadow agent
    from faer.agents.shadow import NHS_THRESHOLDS

    threshold_data = []
    for rule in NHS_THRESHOLDS:
        threshold_data.append({
            "Metric": rule.metric,
            "Threshold": f"{rule.threshold:.2f}" if rule.threshold < 10 else f"{rule.threshold:.0f}",
            "Operator": rule.operator,
            "Severity": rule.severity.value,
            "Title": rule.title,
        })

    threshold_df = pd.DataFrame(threshold_data)
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    st.divider()

    # Parameter-Metric Relationships
    st.subheader("Parameter-Metric Relationships")

    st.markdown("""
    Understanding which parameters affect which metrics helps you design effective experiments.
    Select a metric to see which parameters influence it most:
    """)

    # Metric selector
    selected_metric = st.selectbox(
        "Select a metric to investigate",
        options=list(METRIC_INFO.keys()),
        format_func=lambda x: METRIC_INFO.get(x).display_name if x in METRIC_INFO else x
    )

    metric_info = get_metric_info(selected_metric)
    if metric_info:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{metric_info.display_name}**")
            st.markdown(metric_info.description)
            st.markdown(f"*Unit*: {metric_info.unit}")
            st.markdown(f"*Good direction*: {metric_info.good_direction}")

            if metric_info.nhs_target:
                st.markdown(f"*NHS Target*: {metric_info.nhs_target}")

            if metric_info.warning_threshold:
                st.markdown(f"*Warning at*: {metric_info.warning_threshold}")
            if metric_info.critical_threshold:
                st.markdown(f"*Critical at*: {metric_info.critical_threshold}")

        with col2:
            st.markdown("**Influenced by these parameters:**")
            influencers = get_influencing_parameters(selected_metric)
            if influencers:
                for param in influencers[:5]:  # Top 5
                    guidance = get_parameter_guidance(param)
                    if guidance:
                        # Find the effect for this metric
                        for effect in guidance.affected_metrics:
                            if effect.metric == selected_metric:
                                direction = "increases" if effect.direction.value == "increase" else "decreases"
                                st.markdown(f"- **{guidance.display_name}**: {direction} it ({effect.magnitude.value})")
                                break
            else:
                st.markdown("No parameter guidance available for this metric.")

    st.divider()

    # Interactive Demo
    st.subheader("Try the Agent")

    st.markdown("""
    Enter hypothetical metric values to see what insights the agent would generate:
    """)

    demo_col1, demo_col2, demo_col3 = st.columns(3)

    with demo_col1:
        demo_p_delay = st.slider("P(Delay)", 0.0, 1.0, 0.3, 0.05, key="demo_p_delay")
        demo_treatment_wait = st.slider("Mean Treatment Wait (min)", 0, 300, 45, 5, key="demo_treatment_wait")
        demo_p95_wait = st.slider("P95 Treatment Wait (min)", 0, 400, 120, 10, key="demo_p95_wait")

    with demo_col2:
        demo_util_ed = st.slider("ED Bay Utilisation", 0.0, 1.0, 0.75, 0.05, key="demo_util_ed")
        demo_util_itu = st.slider("ITU Utilisation", 0.0, 1.0, 0.60, 0.05, key="demo_util_itu")
        demo_util_ward = st.slider("Ward Utilisation", 0.0, 1.0, 0.85, 0.05, key="demo_util_ward")

    with demo_col3:
        demo_handover = st.slider("Mean Handover Delay (min)", 0, 60, 10, 2, key="demo_handover")
        demo_boarding = st.slider("Mean Boarding Time (min)", 0, 120, 20, 5, key="demo_boarding")
        demo_opel = st.slider("Peak OPEL Level", 1, 4, 2, 1, key="demo_opel")

    if st.button("Run Agent Analysis", type="primary", use_container_width=True):
        # Build demo metrics
        demo_results = {
            "p_delay": demo_p_delay,
            "mean_treatment_wait": demo_treatment_wait,
            "p95_treatment_wait": demo_p95_wait,
            "util_ed_bays": demo_util_ed,
            "util_itu": demo_util_itu,
            "util_ward": demo_util_ward,
            "util_triage": 0.5,  # Defaults
            "util_theatre": 0.6,
            "mean_handover_delay": demo_handover,
            "max_handover_delay": demo_handover * 2,
            "mean_boarding_time": demo_boarding,
            "opel_peak_level": demo_opel,
            "arrivals": 100,
            "mean_triage_wait": 5,
            "mean_system_time": 180,
            "p95_system_time": 280,
            "mean_itu_wait": 30,
            "mean_ward_wait": 45,
            "mean_theatre_wait": 60,
            "p_boarding": 0.3,
            "aeromed_total": 5,
            "aeromed_slots_missed": 0,
        }

        with st.spinner("Running AI analysis..."):
            insights = run_ai_insights(demo_results, "Demo Scenario")

        display_insights(insights, "Generated Insights")

    st.divider()

    # Expert Perspectives
    st.subheader("Expert Perspectives")

    st.markdown("""
    The agent provides analysis from multiple clinical viewpoints:

    | Expert | Focus Area | Key Metrics |
    |--------|------------|-------------|
    | **EM Consultant** | Triage priorities, ED flow, 4-hour standard | P95 wait, P(delay), boarding |
    | **Anaesthetist** | ITU pathways, P1 stabilisation, critical care | ITU utilisation, ITU wait |
    | **Surgeon** | Theatre capacity, trauma cases, time-critical procedures | Theatre utilisation, theatre wait |
    | **CBRN Specialist** | Decontamination, MCI surge, contamination protocols | ED capacity, surge headroom |
    | **Data Scientist** | Statistical confidence, CI interpretation | Replication count, CI widths |
    | **Paramedic** | Ambulance turnaround, handover delays, bolus patterns | Handover delay, arrival patterns |
    """)

    st.divider()

    # How to Use Insights
    st.subheader("Interpreting Insights")

    st.markdown("""
    **Severity Levels:**

    | Level | Meaning | Action |
    |-------|---------|--------|
    | **CRITICAL** | Immediate patient safety risk | Stop, review, escalate |
    | **HIGH** | Urgent attention needed | Address within 1-2 hours |
    | **MEDIUM** | Monitor closely | Review in daily huddle |
    | **LOW** | Awareness item | Note for planning |
    | **INFO** | Informational | No action needed |

    **Confidence Levels:**

    - **High confidence**: Threshold clearly breached, CI doesn't overlap
    - **Medium confidence**: Threshold breached but CI is close
    - **Low confidence**: Uncertain - more replications recommended

    **Best Practices:**
    1. Run 20+ replications for reliable insights
    2. Focus on CRITICAL and HIGH severity first
    3. Check the evidence section to understand what triggered the alert
    4. Use recommendations as starting points, not prescriptions
    5. Consider compound rules - they catch patterns single metrics miss
    """)
