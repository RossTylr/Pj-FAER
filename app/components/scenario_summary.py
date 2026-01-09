"""Scenario summary component (Phase 8).

Displays comprehensive scenario configuration summary showing all resources
grouped by category: Arrivals, Front Door, Diagnostics, Downstream, Transfer.
"""

import streamlit as st
from faer.core.scenario import FullScenario
from faer.core.entities import DiagnosticType, NodeType


def render_scenario_summary(scenario: FullScenario):
    """Render comprehensive scenario summary showing all resources.

    Groups:
    - Arrivals & Fleet
    - Front Door (Handover, Triage, ED)
    - Diagnostics
    - Downstream (Theatre, ITU, Ward)
    - Transfer & Run Settings

    Args:
        scenario: The scenario configuration to display.
    """
    st.subheader("Scenario Summary")

    # ============ ROW 1: ARRIVALS & FRONT DOOR ============
    st.markdown("#### Arrivals & Front Door")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Ambulances", scenario.n_ambulances)
        st.caption(f"Turnaround: {scenario.ambulance_turnaround_mins:.0f}m")

    with col2:
        st.metric("Helicopters", scenario.n_helicopters)
        st.caption(f"Turnaround: {scenario.helicopter_turnaround_mins:.0f}m")

    with col3:
        st.metric("Handover Bays", scenario.n_handover_bays)
        st.caption(f"Time: {scenario.handover_time_mean:.0f}m")

    with col4:
        st.metric("Triage Clinicians", scenario.n_triage)
        st.caption(f"Time: {scenario.triage_mean:.0f}m")

    with col5:
        st.metric("ED Bays", scenario.n_ed_bays)
        st.caption("Priority queued")

    st.markdown("---")

    # ============ ROW 2: DIAGNOSTICS ============
    st.markdown("#### Diagnostics")

    col1, col2, col3 = st.columns(3)

    ct_config = scenario.diagnostic_configs.get(DiagnosticType.CT_SCAN)
    xray_config = scenario.diagnostic_configs.get(DiagnosticType.XRAY)
    bloods_config = scenario.diagnostic_configs.get(DiagnosticType.BLOODS)

    with col1:
        if ct_config and ct_config.enabled:
            st.metric("CT Scanners", ct_config.capacity)
            st.caption(f"Scan: {ct_config.process_time_mean:.0f}m + Report: {ct_config.turnaround_time_mean:.0f}m")
        else:
            st.metric("CT Scanners", "N/A")

    with col2:
        if xray_config and xray_config.enabled:
            st.metric("X-ray Rooms", xray_config.capacity)
            st.caption(f"Time: {xray_config.process_time_mean:.0f}m + Report: {xray_config.turnaround_time_mean:.0f}m")
        else:
            st.metric("X-ray Rooms", "N/A")

    with col3:
        if bloods_config and bloods_config.enabled:
            st.metric("Phlebotomists", bloods_config.capacity)
            st.caption(f"Draw: {bloods_config.process_time_mean:.0f}m + Lab: {bloods_config.turnaround_time_mean:.0f}m")
        else:
            st.metric("Phlebotomists", "N/A")

    st.markdown("---")

    # ============ ROW 3: DOWNSTREAM ============
    st.markdown("#### Downstream Capacity")

    col1, col2, col3, col4 = st.columns(4)

    surgery_config = scenario.node_configs.get(NodeType.SURGERY)
    itu_config = scenario.node_configs.get(NodeType.ITU)
    ward_config = scenario.node_configs.get(NodeType.WARD)

    with col1:
        if surgery_config:
            st.metric("Theatre Tables", surgery_config.capacity)
            st.caption(f"Session: {surgery_config.service_time_mean:.0f}m")
        else:
            st.metric("Theatre Tables", "N/A")

    with col2:
        if itu_config:
            st.metric("ITU Beds", itu_config.capacity)
            los_hours = itu_config.service_time_mean / 60
            st.caption(f"Avg LoS: {los_hours:.1f}h")
        else:
            st.metric("ITU Beds", "N/A")

    with col3:
        if ward_config:
            st.metric("Ward Beds", ward_config.capacity)
            los_hours = ward_config.service_time_mean / 60
            st.caption(f"Avg LoS: {los_hours:.1f}h")
        else:
            st.metric("Ward Beds", "N/A")

    with col4:
        # Total inpatient beds
        total_beds = (
            (itu_config.capacity if itu_config else 0) +
            (ward_config.capacity if ward_config else 0)
        )
        st.metric("Total Inpatient", total_beds)
        st.caption("ITU + Ward")

    st.markdown("---")

    # ============ ROW 4: TRANSFER & RUN CONFIG ============
    st.markdown("#### Transfer & Run Settings")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if scenario.transfer_config and scenario.transfer_config.enabled:
            st.metric("Transfer Ambulances", scenario.transfer_config.n_transfer_ambulances)
        else:
            st.metric("Transfer Ambulances", "N/A")

    with col2:
        if scenario.transfer_config and scenario.transfer_config.enabled:
            st.metric("Air Ambulance", scenario.transfer_config.n_transfer_helicopters)
        else:
            st.metric("Air Ambulance", "N/A")

    with col3:
        run_hours = scenario.run_length / 60
        st.metric("Run Length", f"{run_hours:.0f}h")
        if run_hours >= 24:
            st.caption(f"{run_hours/24:.1f} days")
        else:
            st.caption(f"{run_hours:.0f} hours")

    with col4:
        st.metric("Demand Multiplier", f"{scenario.demand_multiplier:.2f}x")

    # ============ QUICK STATS (collapsible) ============
    with st.expander("Quick Capacity Check"):
        # Calculate rough capacity indicators
        expected_daily_arrivals = 120 * scenario.demand_multiplier  # Rough estimate
        ed_turnover_per_day = scenario.n_ed_bays * (24 * 60 / 90)  # Assume 90min avg LoS

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rough Daily Capacity**")
            st.write(f"- Expected arrivals: ~{expected_daily_arrivals:.0f}")
            st.write(f"- ED throughput capacity: ~{ed_turnover_per_day:.0f}")

            if expected_daily_arrivals > ed_turnover_per_day:
                st.warning("Arrivals may exceed ED capacity")
            else:
                st.success("ED capacity looks adequate")

        with col2:
            st.markdown("**Bed Ratios**")
            if surgery_config and itu_config:
                ratio = itu_config.capacity / surgery_config.capacity
                st.write(f"- Theatre:ITU ratio: 1:{ratio:.1f}")
            if itu_config and ward_config:
                ratio = ward_config.capacity / itu_config.capacity
                st.write(f"- ITU:Ward ratio: 1:{ratio:.1f}")


def render_scenario_summary_compact(scenario: FullScenario):
    """Compact version for sidebar or comparison view.

    Args:
        scenario: The scenario configuration to display.
    """
    st.markdown("**Resources**")

    # Get config values safely
    ct_config = scenario.diagnostic_configs.get(DiagnosticType.CT_SCAN)
    surgery_config = scenario.node_configs.get(NodeType.SURGERY)
    itu_config = scenario.node_configs.get(NodeType.ITU)
    ward_config = scenario.node_configs.get(NodeType.WARD)

    resources = {
        "Ambulances": scenario.n_ambulances,
        "Helicopters": scenario.n_helicopters,
        "Handover": scenario.n_handover_bays,
        "Triage": scenario.n_triage,
        "ED Bays": scenario.n_ed_bays,
        "CT": ct_config.capacity if ct_config else "?",
        "Theatre": surgery_config.capacity if surgery_config else "?",
        "ITU": itu_config.capacity if itu_config else "?",
        "Ward": ward_config.capacity if ward_config else "?",
    }

    # Display as 2-column table
    cols = st.columns(2)
    items = list(resources.items())

    for i, (key, val) in enumerate(items):
        with cols[i % 2]:
            st.write(f"**{key}**: {val}")


def get_scenario_diff(scenario_a: FullScenario, scenario_b: FullScenario) -> list:
    """Get list of differences between two scenarios.

    Args:
        scenario_a: First scenario (baseline).
        scenario_b: Second scenario (proposed).

    Returns:
        List of tuples: (parameter_name, value_a, value_b, diff, pct_change)
    """
    # Get config values safely
    a_surgery = scenario_a.node_configs.get(NodeType.SURGERY)
    b_surgery = scenario_b.node_configs.get(NodeType.SURGERY)
    a_itu = scenario_a.node_configs.get(NodeType.ITU)
    b_itu = scenario_b.node_configs.get(NodeType.ITU)
    a_ward = scenario_a.node_configs.get(NodeType.WARD)
    b_ward = scenario_b.node_configs.get(NodeType.WARD)
    a_ct = scenario_a.diagnostic_configs.get(DiagnosticType.CT_SCAN)
    b_ct = scenario_b.diagnostic_configs.get(DiagnosticType.CT_SCAN)
    a_xray = scenario_a.diagnostic_configs.get(DiagnosticType.XRAY)
    b_xray = scenario_b.diagnostic_configs.get(DiagnosticType.XRAY)

    comparisons = [
        ("Run Length (h)", scenario_a.run_length / 60, scenario_b.run_length / 60),
        ("Demand", scenario_a.demand_multiplier, scenario_b.demand_multiplier),
        ("Ambulances", scenario_a.n_ambulances, scenario_b.n_ambulances),
        ("Helicopters", scenario_a.n_helicopters, scenario_b.n_helicopters),
        ("Handover Bays", scenario_a.n_handover_bays, scenario_b.n_handover_bays),
        ("Triage Clinicians", scenario_a.n_triage, scenario_b.n_triage),
        ("ED Bays", scenario_a.n_ed_bays, scenario_b.n_ed_bays),
        ("CT Scanners", a_ct.capacity if a_ct else 0, b_ct.capacity if b_ct else 0),
        ("X-ray Rooms", a_xray.capacity if a_xray else 0, b_xray.capacity if b_xray else 0),
        ("Theatre Tables", a_surgery.capacity if a_surgery else 0, b_surgery.capacity if b_surgery else 0),
        ("ITU Beds", a_itu.capacity if a_itu else 0, b_itu.capacity if b_itu else 0),
        ("Ward Beds", a_ward.capacity if a_ward else 0, b_ward.capacity if b_ward else 0),
    ]

    results = []
    for name, val_a, val_b in comparisons:
        diff = val_b - val_a
        pct = ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
        results.append((name, val_a, val_b, diff, pct))

    return results
