"""Clinical Shadow Insights Page.

This page displays agent-generated clinical insights from simulation runs.
It integrates with the agent layer to provide actionable intelligence
from simulation metrics.

Usage:
1. Run a simulation on the Run page
2. Navigate to this page to see detected issues
3. Filter by severity to focus on critical items
4. Review recommendations for each insight
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Clinical Insights",
    page_icon="CI",
    layout="wide",
)

st.title("Clinical Shadow Analysis")
st.caption("AI-powered analysis of simulation results")

# Import agent modules
try:
    from faer.agents import (
        HeuristicShadowAgent,
        AgentOrchestrator,
        MetricsSummary,
        Severity,
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
        Dict with insights and summary from orchestrator
    """
    metrics = MetricsSummary.from_run_results(results_dict, scenario_name=name)

    orchestrator = AgentOrchestrator()
    orchestrator.register(HeuristicShadowAgent())
    # Future: orchestrator.register(CapacityAdvisorAgent())

    result = orchestrator.run_all(metrics)

    return {
        "insights": result.get_all_insights(),
        "summary": result.summary,
        "metrics": metrics,
        "raw_result": result.to_dict(),
    }


# Run analysis
with st.spinner("Analyzing simulation results..."):
    analysis = analyze_results(st.session_state["run_results"], scenario_name)

insights = analysis["insights"]
summary = analysis["summary"]
metrics = analysis["metrics"]

# Summary metrics row
st.subheader("Analysis Summary")

col1, col2, col3, col4, col5 = st.columns(5)

critical_count = summary["critical_count"]
high_count = summary["high_count"]
medium_count = summary["medium_count"]
low_count = summary["low_count"]
total_count = summary["total_insights"]

with col1:
    st.metric(
        "Critical",
        critical_count,
        delta=None,
        delta_color="inverse" if critical_count > 0 else "normal",
    )

with col2:
    st.metric(
        "High",
        high_count,
        delta=None,
        delta_color="inverse" if high_count > 0 else "normal",
    )

with col3:
    st.metric("Medium", medium_count)

with col4:
    st.metric("Low", low_count)

with col5:
    st.metric(
        "Analysis Time",
        f"{summary['execution_time_ms']:.0f}ms",
    )

# Alert banner for critical issues
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

# Expandable section for key metrics
st.divider()
with st.expander("Key Metrics Summary", expanded=False):
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

# Export options
st.divider()
with st.expander("Export Options", expanded=False):
    st.markdown("Export analysis results for reporting or further analysis.")

    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        import json

        json_export = json.dumps(analysis["raw_result"], indent=2)
        st.download_button(
            label="Download JSON",
            data=json_export,
            file_name=f"insights_{scenario_name.replace(' ', '_')}.json",
            mime="application/json",
        )

    with col2:
        # Export insights as markdown
        md_lines = [
            f"# Clinical Shadow Analysis: {scenario_name}",
            f"",
            f"## Summary",
            f"- Total Issues: {total_count}",
            f"- Critical: {critical_count}",
            f"- High: {high_count}",
            f"- Medium: {medium_count}",
            f"",
            f"## Insights",
            f"",
        ]

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
            label="Download Markdown",
            data=md_export,
            file_name=f"insights_{scenario_name.replace(' ', '_')}.md",
            mime="text/markdown",
        )

# Footer
st.divider()
st.caption(
    "Clinical Shadow Agent v1.0 | "
    "Powered by heuristic rules based on NHS clinical thresholds | "
    "For research and planning purposes only"
)
