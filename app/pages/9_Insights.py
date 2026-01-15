"""Clinical Shadow Insights Page.

This page displays agent-generated clinical insights from simulation runs.
It integrates with the agent layer to provide actionable intelligence
from simulation metrics.

Features:
- System-level evaluation with structural strengths/weaknesses
- Peak load and overloading analysis
- Multi-expert perspective analysis (EM, Anaesthetist, Surgeon, CBRN, Paramedic, Data Scientist)
- Threshold-based clinical alerts with confidence intervals
- Resilience scoring and bottleneck chain identification

Usage:
1. Run a simulation on the Run page
2. Navigate to this page to see detected issues
3. Filter by severity to focus on critical items
4. Review recommendations from multiple expert perspectives
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Clinical Insights",
    page_icon="CI",
    layout="wide",
)

st.title("Clinical Shadow Analysis")
st.caption("AI-powered multi-expert analysis of simulation results")

# Import agent modules
try:
    from faer.agents import (
        HeuristicShadowAgent,
        AgentOrchestrator,
        MetricsSummary,
        Severity,
        StructuralAssessment,
        PeakLoadAnalysis,
        ExpertPerspective,
        SystemEvaluation,
    )

    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    st.error(f"Agent modules not available: {e}")
    st.info("Install the agent layer with: `pip install -e .`")
    st.stop()


def get_severity_label(severity: Severity) -> str:
    """Map severity to text label."""
    return {
        Severity.CRITICAL: "[CRITICAL]",
        Severity.HIGH: "[HIGH]",
        Severity.MEDIUM: "[MEDIUM]",
        Severity.LOW: "[LOW]",
        Severity.INFO: "[INFO]",
    }.get(severity, "[UNKNOWN]")


def get_severity_color(severity: Severity) -> str:
    """Map severity to Streamlit color name."""
    return {
        Severity.CRITICAL: "red",
        Severity.HIGH: "orange",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "green",
        Severity.INFO: "blue",
    }.get(severity, "gray")


# Check for simulation results
if "run_results" not in st.session_state:
    st.warning("No simulation results available.")
    st.markdown("""
    **To generate insights:**
    1. Configure your scenario on the **Arrivals** and **Resources** pages
    2. Run a simulation on the **Run** page
    3. Return here to see clinical insights

    The Clinical Shadow agent analyzes simulation outputs to identify:
    - Clinical safety risks (4-hour breaches, high acuity delays)
    - Capacity bottlenecks (ED saturation, ITU pressure)
    - Flow issues (boarding, handover delays)
    - Compound risk patterns (multi-factor alerts)
    """)

    if st.button("Go to Run Page"):
        st.switch_page("pages/5_Run.py")

    st.stop()

# Convert results to MetricsSummary
scenario_name = st.session_state.get("scenario_name", "Current Scenario")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def analyze_results(results_dict: dict, name: str) -> dict:
    """Run agent analysis on results (cached).

    Args:
        results_dict: Raw simulation results
        name: Scenario name

    Returns:
        Dict with insights, system evaluation, and summary from orchestrator
    """
    metrics = MetricsSummary.from_run_results(results_dict, scenario_name=name)

    # Create agent and run analysis
    agent = HeuristicShadowAgent()

    orchestrator = AgentOrchestrator()
    orchestrator.register(agent)
    # Future: orchestrator.register(CapacityAdvisorAgent())

    result = orchestrator.run_all(metrics)

    # Run comprehensive system evaluation
    system_eval = agent.evaluate_system(metrics)

    return {
        "insights": result.get_all_insights(),
        "summary": result.summary,
        "metrics": metrics,
        "raw_result": result.to_dict(),
        # New system evaluation data
        "system_evaluation": system_eval,
        "structural": system_eval.structural,
        "peak_load": system_eval.peak_load,
        "expert_perspectives": system_eval.expert_perspectives,
    }


# Run analysis
with st.spinner("Analyzing simulation results..."):
    analysis = analyze_results(st.session_state["run_results"], scenario_name)

insights = analysis["insights"]
summary = analysis["summary"]
metrics = analysis["metrics"]
system_eval = analysis["system_evaluation"]
structural = analysis["structural"]
peak_load = analysis["peak_load"]
expert_perspectives = analysis["expert_perspectives"]

# ===== EXECUTIVE SUMMARY =====
st.subheader("Executive Summary")

# Display system status and executive summary
status_colors = {"critical": "red", "warning": "orange", "caution": "yellow", "normal": "green"}
status = system_eval.overall_system_status
st.markdown(system_eval.summary_text)

st.divider()

# ===== TOP-LEVEL METRICS =====
st.subheader("System Health Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Resilience Score",
        f"{structural.resilience_score:.0f}/100",
        help="System's ability to absorb surge demand (0-100)",
    )

with col2:
    critical_count = summary["critical_count"]
    st.metric(
        "Critical Alerts",
        critical_count,
        delta=None,
        delta_color="inverse" if critical_count > 0 else "normal",
    )

with col3:
    high_count = summary["high_count"]
    st.metric(
        "High Priority",
        high_count,
        delta=None,
        delta_color="inverse" if high_count > 0 else "normal",
    )

with col4:
    st.metric(
        "Peak/Mean Ratio",
        f"{peak_load.peak_to_mean_ratio:.1f}x",
        help="Arrival burstiness indicator",
    )

with col5:
    st.metric(
        "Time Over Capacity",
        f"{peak_load.time_above_capacity_pct:.0f}%",
        help="Percentage of time system was above threshold",
    )

# Alert banners
medium_count = summary["medium_count"]
low_count = summary["low_count"]
total_count = summary["total_insights"]

if critical_count > 0:
    st.error(
        f"**{critical_count} CRITICAL issue(s) detected** - "
        "Immediate attention required!"
    )
elif high_count > 0:
    st.warning(
        f"**{high_count} HIGH priority issue(s) detected** - "
        "Review recommended."
    )
elif total_count == 0:
    st.success("**No significant issues detected** - System performing within thresholds.")

st.divider()

# Severity filter
st.subheader("Detected Issues")

severity_options = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
default_selection = ["CRITICAL", "HIGH", "MEDIUM"]

severity_filter = st.multiselect(
    "Filter by severity",
    options=severity_options,
    default=default_selection,
    help="Select which severity levels to display",
)

# Filter insights
filtered_insights = [
    i for i in insights if i.severity.value in severity_filter
]

def get_confidence_badge(confidence_level: str) -> str:
    """Generate confidence badge with color coding."""
    colors = {
        "high": "green",
        "medium": "orange",
        "low": "red",
    }
    color = colors.get(confidence_level, "gray")
    return f":{color}[{confidence_level.upper()}]"


if not filtered_insights:
    if total_count == 0:
        st.info("No issues detected in this simulation run. The system is operating within clinical thresholds.")
    else:
        st.info(f"No issues at selected severity levels. Try expanding the filter to see {total_count} total issues.")
else:
    # Display insights
    for idx, insight in enumerate(filtered_insights):
        label = get_severity_label(insight.severity)

        # Build expander title with confidence badge
        confidence_badge = ""
        if hasattr(insight, "confidence_level") and insight.confidence_level:
            confidence_badge = f" | Confidence: {get_confidence_badge(insight.confidence_level)}"

        # Expandable card for each insight
        with st.expander(
            f"{label} **{insight.severity.value}**: {insight.title}{confidence_badge}",
            expanded=(insight.severity == Severity.CRITICAL),
        ):
            # Main message
            st.markdown(insight.message)

            # Show uncertainty note if present
            if hasattr(insight, "uncertainty_note") and insight.uncertainty_note:
                st.warning(f"**Statistical Note**: {insight.uncertainty_note}")

            # Show CI bounds if available
            if hasattr(insight, "ci_lower") and insight.ci_lower is not None:
                threshold_val = insight.evidence.get("threshold", "N/A")
                ci_text = (
                    f"95% CI: [{insight.ci_lower:.2f}, {insight.ci_upper:.2f}] | "
                    f"Threshold: {threshold_val}"
                )
                if hasattr(insight, "threshold_overlap") and insight.threshold_overlap:
                    ci_text += " | CI overlaps threshold"
                st.caption(ci_text)

            # Evidence section
            st.caption("**Evidence**")
            # Filter out CI bounds from evidence display (shown separately above)
            evidence_items = [
                f"`{k}` = `{v:.2f}`" if isinstance(v, float) else f"`{k}` = `{v}`"
                for k, v in insight.evidence.items()
                if k not in ("ci_lower", "ci_upper")
            ]
            st.code(" | ".join(evidence_items), language=None)

            # Recommendation
            if insight.recommendation:
                st.info(f"**Recommendation**: {insight.recommendation}")

            # Metadata
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption(f"Category: `{insight.category.value}`")
            with col_b:
                st.caption(f"Source: `{insight.source_agent}`")
            with col_c:
                if hasattr(insight, "confidence_level"):
                    st.caption(f"Confidence: `{insight.confidence_level}`")

# ===== TABBED DEEP-DIVE ANALYSIS =====
st.divider()
st.header("Deep-Dive Analysis")

analysis_tabs = st.tabs([
    "Expert Perspectives",
    "Structural Assessment",
    "Peak Load Analysis",
    "Key Metrics",
])

# ===== TAB 1: EXPERT PERSPECTIVES =====
with analysis_tabs[0]:
    st.subheader("Multi-Expert Panel Analysis")
    st.caption(
        "Simulation results analyzed from multiple clinical and operational perspectives. "
        "Each expert provides domain-specific insights."
    )

    def get_severity_label(severity: Severity) -> str:
        """Get text severity label."""
        return f"[{severity.value}]"

    # Display each expert perspective
    for perspective in expert_perspectives:
        severity_label = get_severity_label(perspective.severity)

        with st.expander(
            f"**{perspective.expert_title}** - {severity_label}",
            expanded=(perspective.severity in [Severity.CRITICAL, Severity.HIGH]),
        ):
            st.markdown(f"**Focus Area**: {perspective.focus_area}")
            st.markdown(f"**Assessment**: {perspective.assessment}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Concerns:**")
                for concern in perspective.concerns:
                    st.markdown(f"- {concern}")

            with col2:
                st.markdown("**Recommendations:**")
                for rec in perspective.recommendations:
                    st.markdown(f"- {rec}")

            # Key metrics for this expert
            st.markdown("**Key Metrics:**")
            metric_items = [
                f"`{k}` = `{v:.2f}`" if isinstance(v, float) else f"`{k}` = `{v}`"
                for k, v in perspective.key_metrics.items()
            ]
            st.code(" | ".join(metric_items), language=None)

# ===== TAB 2: STRUCTURAL ASSESSMENT =====
with analysis_tabs[1]:
    st.subheader("Structural Strengths & Weaknesses")
    st.caption(
        "System-level analysis of configuration and performance characteristics."
    )

    # Resilience score gauge
    col1, col2 = st.columns([1, 2])

    with col1:
        score = structural.resilience_score
        if score >= 70:
            score_color = "green"
            score_label = "Good"
        elif score >= 40:
            score_color = "orange"
            score_label = "Moderate"
        else:
            score_color = "red"
            score_label = "Low"

        st.metric(
            "Resilience Score",
            f"{score:.0f}/100",
            help="Higher scores indicate better ability to absorb surge demand",
        )
        st.markdown(f"**Status**: :{score_color}[{score_label}]")

    with col2:
        st.markdown("**Resource Headroom (Remaining Capacity)**")
        # Create a simple bar chart for headroom
        import pandas as pd

        headroom_df = pd.DataFrame({
            "Resource": list(structural.headroom_by_resource.keys()),
            "Headroom": [v * 100 for v in structural.headroom_by_resource.values()],
        })
        headroom_df = headroom_df.sort_values("Headroom")

        st.bar_chart(headroom_df.set_index("Resource"), height=200)

    # Bottleneck chain
    if structural.bottleneck_chain:
        st.markdown("---")
        st.markdown("**Bottleneck Chain** (resources with <30% headroom)")
        chain_display = " → ".join([f"**{r.upper()}**" for r in structural.bottleneck_chain])
        st.warning(f"🔗 {chain_display}")
        st.caption(
            "Bottleneck cascade: congestion at these resources propagates through the system."
        )

    # Strengths and weaknesses
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Structural Strengths**")
        if structural.strengths:
            for strength in structural.strengths:
                st.markdown(f"✅ {strength}")
        else:
            st.markdown("_No significant strengths identified_")

    with col2:
        st.markdown("**Structural Weaknesses**")
        if structural.weaknesses:
            for weakness in structural.weaknesses:
                st.markdown(f"⚠️ {weakness}")
        else:
            st.markdown("✅ _No significant weaknesses identified_")

# ===== TAB 3: PEAK LOAD ANALYSIS =====
with analysis_tabs[2]:
    st.subheader("Peak Load & Overloading Analysis")
    st.caption(
        "Analysis of arrival patterns, surge capacity, and system stress under peak conditions."
    )

    # Key peak load metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Arrival Rate",
            f"{peak_load.mean_arrival_rate:.1f}/hr",
            help="Average patients per hour",
        )

    with col2:
        st.metric(
            "Peak Arrival Rate",
            f"{peak_load.peak_arrival_rate:.1f}/hr",
            help="Estimated peak patients per hour",
        )

    with col3:
        st.metric(
            "Peak/Mean Ratio",
            f"{peak_load.peak_to_mean_ratio:.1f}x",
            help="Higher ratio = more bursty arrivals",
        )

    with col4:
        st.metric(
            "Surge Headroom",
            f"{peak_load.estimated_surge_headroom:.0f} min",
            help="Estimated time system can absorb surge before saturation",
        )

    st.markdown("---")

    # Capacity stress indicators
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capacity Stress Indicators**")
        st.write(f"- Time above capacity: **{peak_load.time_above_capacity_pct:.0f}%**")
        st.write(f"- Queue buildup rate: **{peak_load.queue_buildup_rate:.1f}** pts/hr during peak")

        if peak_load.bolus_pattern_detected:
            st.warning("⚠️ **Bolus arrival pattern detected** - multi-wave or convoy arrivals likely")
        else:
            st.success("✅ No significant bolus patterns detected")

    with col2:
        st.markdown("**Interpretation**")
        if peak_load.peak_to_mean_ratio > 2.0:
            st.markdown(
                "🔴 **High variability**: Arrival pattern is highly bursty. "
                "System experiences significant peaks above average load."
            )
        elif peak_load.peak_to_mean_ratio > 1.5:
            st.markdown(
                "🟠 **Moderate variability**: Some arrival clustering present. "
                "System may struggle during peak periods."
            )
        else:
            st.markdown(
                "🟢 **Stable arrivals**: Relatively uniform arrival pattern. "
                "System operates consistently throughout the period."
            )

        if peak_load.time_above_capacity_pct > 50:
            st.markdown(
                "🔴 **Over capacity frequently**: System spending majority of time "
                "operating above sustainable threshold."
            )
        elif peak_load.time_above_capacity_pct > 20:
            st.markdown(
                "🟠 **Periodic overload**: System experiences regular periods of stress. "
                "May need capacity increase or demand management."
            )
        else:
            st.markdown(
                "🟢 **Generally within capacity**: System operating within sustainable "
                "parameters most of the time."
            )

# ===== TAB 4: KEY METRICS =====
with analysis_tabs[3]:
    st.subheader("Key Metrics Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demand**")
        st.write(f"- Total Arrivals: {metrics.arrivals:.0f}")
        st.write(f"- P1 (Resus): {metrics.arrivals_by_priority.get('P1', 0):.0f}")
        st.write(f"- P2 (Very Urgent): {metrics.arrivals_by_priority.get('P2', 0):.0f}")
        st.write(f"- P3 (Urgent): {metrics.arrivals_by_priority.get('P3', 0):.0f}")
        st.write(f"- P4 (Standard): {metrics.arrivals_by_priority.get('P4', 0):.0f}")

    with col2:
        st.markdown("**Wait Times**")
        st.write(f"- Mean Triage Wait: {metrics.mean_triage_wait:.1f} min")
        st.write(f"- Mean Treatment Wait: {metrics.mean_treatment_wait:.1f} min")
        st.write(f"- P95 Treatment Wait: {metrics.p95_treatment_wait:.1f} min")
        st.write(f"- Mean System Time: {metrics.mean_system_time:.1f} min")
        st.write(f"- P(Delay): {metrics.p_delay:.1%}")

    with col3:
        st.markdown("**Utilization**")
        st.write(f"- ED Bays: {metrics.util_ed_bays:.1%}")
        st.write(f"- Triage: {metrics.util_triage:.1%}")
        st.write(f"- ITU: {metrics.util_itu:.1%}")
        st.write(f"- Ward: {metrics.util_ward:.1%}")
        st.write(f"- Theatre: {metrics.util_theatre:.1%}")

    # Additional downstream metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Downstream (Boarding/Blocking)**")
        st.write(f"- Mean Boarding Time: {metrics.mean_boarding_time:.1f} min")
        st.write(f"- P(Boarding): {metrics.p_boarding:.1%}")

    with col2:
        st.markdown("**Handover**")
        st.write(f"- Mean Handover Delay: {metrics.mean_handover_delay:.1f} min")
        st.write(f"- Max Handover Delay: {metrics.max_handover_delay:.1f} min")

    with col3:
        st.markdown("**Aeromedical**")
        st.write(f"- Total Aeromed: {metrics.aeromed_total:.0f}")
        st.write(f"- Slots Missed: {metrics.aeromed_slots_missed:.0f}")
        st.write(f"- Mean Slot Wait: {metrics.mean_aeromed_slot_wait:.1f} min")

# Export options
st.divider()
with st.expander("Export Options", expanded=False):
    st.markdown("Export analysis results for reporting or further analysis.")

    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        import json

        # Build comprehensive export dict
        export_data = {
            "scenario_name": scenario_name,
            "system_status": system_eval.overall_system_status,
            "resilience_score": structural.resilience_score,
            "summary": {
                "total_insights": total_count,
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
            },
            "structural_assessment": {
                "strengths": structural.strengths,
                "weaknesses": structural.weaknesses,
                "bottleneck_chain": structural.bottleneck_chain,
                "headroom_by_resource": structural.headroom_by_resource,
            },
            "peak_load_analysis": {
                "mean_arrival_rate": peak_load.mean_arrival_rate,
                "peak_arrival_rate": peak_load.peak_arrival_rate,
                "peak_to_mean_ratio": peak_load.peak_to_mean_ratio,
                "time_above_capacity_pct": peak_load.time_above_capacity_pct,
                "bolus_pattern_detected": peak_load.bolus_pattern_detected,
                "surge_headroom_mins": peak_load.estimated_surge_headroom,
            },
            "expert_perspectives": [
                {
                    "role": p.expert_role,
                    "title": p.expert_title,
                    "severity": p.severity.value,
                    "assessment": p.assessment,
                    "concerns": p.concerns,
                    "recommendations": p.recommendations,
                }
                for p in expert_perspectives
            ],
            "raw_orchestrator_result": analysis["raw_result"],
        }

        json_export = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download Full Analysis (JSON)",
            data=json_export,
            file_name=f"insights_{scenario_name.replace(' ', '_')}.json",
            mime="application/json",
        )

    with col2:
        # Export insights as markdown with full system evaluation
        md_lines = [
            f"# Clinical Shadow Analysis: {scenario_name}",
            f"",
            f"## Executive Summary",
            f"",
            system_eval.summary_text,
            f"",
            f"## System Health",
            f"- **Overall Status**: {system_eval.overall_system_status.upper()}",
            f"- **Resilience Score**: {structural.resilience_score:.0f}/100",
            f"- **Peak/Mean Ratio**: {peak_load.peak_to_mean_ratio:.1f}x",
            f"- **Time Over Capacity**: {peak_load.time_above_capacity_pct:.0f}%",
            f"",
            f"## Alert Summary",
            f"- Total Issues: {total_count}",
            f"- Critical: {critical_count}",
            f"- High: {high_count}",
            f"- Medium: {medium_count}",
            f"",
            f"## Structural Assessment",
            f"",
            f"### Strengths",
            "",
        ]
        for s in structural.strengths:
            md_lines.append(f"- {s}")
        md_lines.extend([
            f"",
            f"### Weaknesses",
            "",
        ])
        for w in structural.weaknesses:
            md_lines.append(f"- {w}")
        if structural.bottleneck_chain:
            md_lines.extend([
                f"",
                f"### Bottleneck Chain",
                f"",
                f"{' → '.join(structural.bottleneck_chain)}",
            ])

        md_lines.extend([
            f"",
            f"## Expert Perspectives",
            f"",
        ])

        for p in expert_perspectives:
            md_lines.extend([
                f"### {p.expert_title} ({p.severity.value})",
                f"",
                f"**Focus**: {p.focus_area}",
                f"",
                f"**Assessment**: {p.assessment}",
                f"",
                f"**Concerns**:",
            ])
            for c in p.concerns:
                md_lines.append(f"- {c}")
            md_lines.extend([
                f"",
                f"**Recommendations**:",
            ])
            for r in p.recommendations:
                md_lines.append(f"- {r}")
            md_lines.append(f"")
            md_lines.append(f"---")
            md_lines.append(f"")

        md_lines.extend([
            f"## Detected Issues",
            f"",
        ])

        for insight in insights:
            md_lines.extend(
                [
                    f"### [{insight.severity.value}] {insight.title}",
                    f"",
                    insight.message,
                    f"",
                    f"**Evidence**: {', '.join(f'{k}={v}' for k, v in insight.evidence.items())}",
                    f"",
                    f"**Recommendation**: {insight.recommendation or 'N/A'}",
                    f"",
                    f"---",
                    f"",
                ]
            )

        md_export = "\n".join(md_lines)
        st.download_button(
            label="Download Full Report (Markdown)",
            data=md_export,
            file_name=f"insights_{scenario_name.replace(' ', '_')}.md",
            mime="text/markdown",
        )

# Footer
st.divider()
st.caption(
    "Clinical Shadow Agent v2.0 | "
    "Multi-expert analysis with 6 clinical/operational perspectives | "
    "Powered by heuristic rules based on NHS clinical thresholds | "
    "For research and planning purposes only"
)
