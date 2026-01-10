"""
Resources Configuration Page.

Configures hospital resources with integrated capacity and service times.
Each section shows both HOW MANY resources and HOW LONG patients spend there.

Extracted and reorganized from original 1_Scenario.py for operational clarity.

Integration:
- Stores values to st.session_state for FullScenario construction
- Compatible with existing run_full_simulation() flow
- Some resources (ITU, Ward, Theatre) may not be fully simulated yet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict

# Import from existing codebase
from faer.core.entities import DiagnosticType, Priority, NodeType
from faer.core.scenario import (
    FullScenario,
    DiagnosticConfig,
    ITUConfig,
    WardConfig,
    TheatreConfig,
)

st.set_page_config(
    page_title="Resources",
    page_icon="",
    layout="wide"
)

st.title("Resources Configuration")

st.info("""
Configure all hospital resources - **capacity AND service times together**.
Each section shows both how many resources AND how long patients spend there.

**Simulation linkage**: Creates `simpy.PriorityResource` objects in full_model.py
""")


# ============================================================
# HELPER: Initialize session state with defaults
# ============================================================
def init_resource_defaults():
    """Initialize resource-related session state with FullScenario defaults."""
    defaults = {
        # Triage
        'n_triage': 3,
        'triage_time_mean': 5.0,
        'triage_time_cv': 0.3,

        # ED Bays
        'n_ed_bays': 20,
        'ed_assessment_time_mean': 15.0,
        'ed_treatment_time_mean': 45.0,
        'ed_service_time_cv': 0.4,

        # Diagnostics - CT
        'ct_capacity': 2,
        'ct_scan_time_mean': 20.0,
        'ct_report_time_mean': 30.0,
        'ct_time_cv': 0.3,
        'ct_enabled': True,

        # Diagnostics - X-ray
        'xray_capacity': 3,
        'xray_time_mean': 10.0,
        'xray_report_time_mean': 15.0,
        'xray_time_cv': 0.3,
        'xray_enabled': True,

        # Diagnostics - Bloods
        'bloods_capacity': 5,
        'bloods_draw_time_mean': 5.0,
        'bloods_lab_time_mean': 45.0,
        'bloods_time_cv': 0.3,
        'bloods_enabled': True,

        # Theatre
        'n_theatre_tables': 2,
        'theatre_session_duration': 240.0,  # 4 hours
        'theatre_procedure_time_mean': 90.0,
        'theatre_procedure_time_cv': 0.5,
        'theatre_sessions_per_day': 2,
        'postop_itu_probability': 0.25,

        # ITU
        'n_itu_beds': 6,
        'itu_los_hours_mean': 48.0,
        'itu_los_cv': 0.8,
        'itu_stepdown_probability': 0.85,
        'itu_mortality_probability': 0.10,

        # Ward
        'n_ward_beds': 30,
        'ward_los_hours_mean': 72.0,  # 3 days
        'ward_los_cv': 1.0,
        'ward_turnaround_mins': 30.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize defaults
init_resource_defaults()

# ============================================================
# SECTION 1: TRIAGE
# ============================================================
st.header("1. Triage")

with st.container(border=True):
    st.markdown("""
    Initial patient assessment to assign priority (P1-P4).
    P1 patients bypass triage and go directly to ED bays.

    **Simulation**: `simpy.PriorityResource(n_triage)` -> `triage_process()`
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Capacity**")
        n_triage = st.number_input(
            "Triage Clinicians",
            min_value=1,
            max_value=20,
            value=st.session_state.n_triage,
            key="input_n_triage",
            help="ANPs, PAs, or doctors staffing triage. Maps to FullScenario.n_triage"
        )
        st.session_state.n_triage = n_triage

    with col2:
        st.markdown("**Assessment Time**")
        triage_time = st.number_input(
            "Mean Time (mins)",
            min_value=2.0,
            max_value=30.0,
            value=float(st.session_state.triage_time_mean),
            step=1.0,
            key="input_triage_time",
            help="Average triage assessment duration. Maps to FullScenario.triage_mean"
        )
        st.session_state.triage_time_mean = triage_time

    with col3:
        st.markdown("**Variability**")
        triage_cv = st.slider(
            "CV",
            min_value=0.1,
            max_value=0.8,
            value=float(st.session_state.triage_time_cv),
            step=0.05,
            key="input_triage_cv",
            help="Coefficient of variation. Higher = more variable."
        )
        st.session_state.triage_time_cv = triage_cv

    # Capacity indicator
    throughput = n_triage * (60 / triage_time)
    st.metric("Triage Throughput", f"~{throughput:.0f} patients/hour")

# ============================================================
# SECTION 2: ED BAYS
# ============================================================
st.header("2. Emergency Department Bays")

with st.container(border=True):
    st.markdown("""
    Main treatment bays/cubicles in ED. Patients queue by priority (P1 first).
    Patients **keep their bay** during diagnostic journeys.

    **Simulation**: `simpy.PriorityResource(n_ed_bays)` -> `patient_journey()`

    **Critical feedback**: When ED bays are full, handover bays block -> ambulance delays.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capacity**")
        n_ed_bays = st.number_input(
            "ED Bays",
            min_value=5,
            max_value=100,
            value=st.session_state.n_ed_bays,
            key="input_n_ed_bays",
            help="Total treatment bays/cubicles. Maps to FullScenario.n_ed_bays"
        )
        st.session_state.n_ed_bays = n_ed_bays

        st.info("Priority queuing: P1 patients served before P2, P2 before P3, etc.")

    with col2:
        st.markdown("**Service Times**")

        ed_assessment = st.number_input(
            "Initial Assessment (mins)",
            min_value=5.0,
            max_value=60.0,
            value=float(st.session_state.ed_assessment_time_mean),
            step=5.0,
            key="input_ed_assessment",
            help="Time before diagnostics ordered"
        )
        st.session_state.ed_assessment_time_mean = ed_assessment

        ed_treatment = st.number_input(
            "Treatment Time (mins)",
            min_value=15.0,
            max_value=180.0,
            value=float(st.session_state.ed_treatment_time_mean),
            step=5.0,
            key="input_ed_treatment",
            help="Average treatment after diagnostics complete"
        )
        st.session_state.ed_treatment_time_mean = ed_treatment

        ed_cv = st.slider(
            "Time Variability (CV)",
            min_value=0.2,
            max_value=0.8,
            value=float(st.session_state.ed_service_time_cv),
            step=0.05,
            key="input_ed_cv"
        )
        st.session_state.ed_service_time_cv = ed_cv

    # Summary
    total_ed_time = ed_assessment + ed_treatment
    st.caption(f"Minimum ED time (excl. diagnostics): ~{total_ed_time:.0f} mins")

    # Capacity check against arrivals
    if 'demand_multiplier' in st.session_state:
        # Rough estimate of hourly arrivals
        expected_hourly = 5.0 * st.session_state.get('demand_multiplier', 1.0)
        ed_turnover = n_ed_bays * (60 / total_ed_time)

        if expected_hourly > ed_turnover * 0.85:
            st.warning(f"Expected arrivals (~{expected_hourly:.0f}/hr) approaching ED turnover capacity (~{ed_turnover:.0f}/hr)")

# ============================================================
# SECTION 3: DIAGNOSTICS
# ============================================================
st.header("3. Diagnostics")

st.markdown("""
Diagnostic services requested during ED stay.
**Critical**: Patients **KEEP their ED bay** during diagnostic journeys.

**Simulation**: Each creates `simpy.PriorityResource` -> `diagnostic_process()`
""")

# CT Scanner
with st.container(border=True):
    st.subheader("CT Scanner")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ct_enabled = st.checkbox(
            "Enabled",
            value=st.session_state.ct_enabled,
            key="input_ct_enabled"
        )
        st.session_state.ct_enabled = ct_enabled

    with col2:
        ct_capacity = st.number_input(
            "Scanners",
            min_value=1,
            max_value=10,
            value=st.session_state.ct_capacity,
            key="input_ct_capacity",
            disabled=not ct_enabled
        )
        st.session_state.ct_capacity = ct_capacity

    with col3:
        ct_scan = st.number_input(
            "Scan Time (mins)",
            min_value=10.0,
            max_value=60.0,
            value=float(st.session_state.ct_scan_time_mean),
            key="input_ct_scan",
            disabled=not ct_enabled
        )
        st.session_state.ct_scan_time_mean = ct_scan

    with col4:
        ct_report = st.number_input(
            "Report Time (mins)",
            min_value=10.0,
            max_value=120.0,
            value=float(st.session_state.ct_report_time_mean),
            key="input_ct_report",
            disabled=not ct_enabled,
            help="Radiologist reporting time"
        )
        st.session_state.ct_report_time_mean = ct_report

    if ct_enabled:
        st.caption(f"Total CT turnaround: ~{ct_scan + ct_report:.0f} mins")

# X-ray
with st.container(border=True):
    st.subheader("X-ray")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        xray_enabled = st.checkbox(
            "Enabled",
            value=st.session_state.xray_enabled,
            key="input_xray_enabled"
        )
        st.session_state.xray_enabled = xray_enabled

    with col2:
        xray_capacity = st.number_input(
            "X-ray Rooms",
            min_value=1,
            max_value=10,
            value=st.session_state.xray_capacity,
            key="input_xray_capacity",
            disabled=not xray_enabled
        )
        st.session_state.xray_capacity = xray_capacity

    with col3:
        xray_time = st.number_input(
            "X-ray Time (mins)",
            min_value=5.0,
            max_value=30.0,
            value=float(st.session_state.xray_time_mean),
            key="input_xray_time",
            disabled=not xray_enabled
        )
        st.session_state.xray_time_mean = xray_time

    with col4:
        xray_report = st.number_input(
            "Report Time (mins)",
            min_value=5.0,
            max_value=60.0,
            value=float(st.session_state.xray_report_time_mean),
            key="input_xray_report",
            disabled=not xray_enabled
        )
        st.session_state.xray_report_time_mean = xray_report

    if xray_enabled:
        st.caption(f"Total X-ray turnaround: ~{xray_time + xray_report:.0f} mins")

# Bloods
with st.container(border=True):
    st.subheader("Bloods / Pathology")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        bloods_enabled = st.checkbox(
            "Enabled",
            value=st.session_state.bloods_enabled,
            key="input_bloods_enabled"
        )
        st.session_state.bloods_enabled = bloods_enabled

    with col2:
        bloods_capacity = st.number_input(
            "Phlebotomists",
            min_value=1,
            max_value=15,
            value=st.session_state.bloods_capacity,
            key="input_bloods_capacity",
            disabled=not bloods_enabled
        )
        st.session_state.bloods_capacity = bloods_capacity

    with col3:
        bloods_draw = st.number_input(
            "Blood Draw (mins)",
            min_value=2.0,
            max_value=15.0,
            value=float(st.session_state.bloods_draw_time_mean),
            key="input_bloods_draw",
            disabled=not bloods_enabled
        )
        st.session_state.bloods_draw_time_mean = bloods_draw

    with col4:
        bloods_lab = st.number_input(
            "Lab Turnaround (mins)",
            min_value=15.0,
            max_value=120.0,
            value=float(st.session_state.bloods_lab_time_mean),
            key="input_bloods_lab",
            disabled=not bloods_enabled,
            help="Time for lab processing and reporting"
        )
        st.session_state.bloods_lab_time_mean = bloods_lab

    if bloods_enabled:
        st.caption(f"Total bloods turnaround: ~{bloods_draw + bloods_lab:.0f} mins")

# ============================================================
# SECTION 4: THEATRE / SURGERY
# ============================================================
st.header("4. Theatre / Surgery")

with st.container(border=True):
    st.markdown("""
    Operating theatres for patients requiring surgery.
    Post-op patients route to ITU or Ward based on procedure complexity.

    **Simulation status**: UI ready, simulation wiring in Phase 8b+
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capacity**")

        n_tables = st.number_input(
            "Operating Tables",
            min_value=1,
            max_value=20,
            value=st.session_state.n_theatre_tables,
            key="input_n_tables"
        )
        st.session_state.n_theatre_tables = n_tables

        sessions_per_day = st.number_input(
            "Sessions per Day",
            min_value=1,
            max_value=4,
            value=st.session_state.theatre_sessions_per_day,
            key="input_sessions_per_day",
            help="AM and PM lists = 2 sessions"
        )
        st.session_state.theatre_sessions_per_day = sessions_per_day

    with col2:
        st.markdown("**Timing**")

        session_duration = st.number_input(
            "Session Duration (mins)",
            min_value=60.0,
            max_value=480.0,
            value=float(st.session_state.theatre_session_duration),
            step=30.0,
            key="input_session_duration",
            help="Length of each theatre list"
        )
        st.session_state.theatre_session_duration = session_duration

        procedure_time = st.number_input(
            "Avg Procedure (mins)",
            min_value=30.0,
            max_value=300.0,
            value=float(st.session_state.theatre_procedure_time_mean),
            step=10.0,
            key="input_procedure_time"
        )
        st.session_state.theatre_procedure_time_mean = procedure_time

    # Post-op routing
    st.markdown("**Post-op Destination**")
    col1, col2 = st.columns(2)

    with col1:
        postop_itu = st.slider(
            "To ITU (%)",
            min_value=0,
            max_value=50,
            value=int(st.session_state.postop_itu_probability * 100),
            key="input_postop_itu"
        )
        st.session_state.postop_itu_probability = postop_itu / 100

    with col2:
        st.metric("To Ward (%)", 100 - postop_itu)

    # Capacity calculation
    daily_hours = n_tables * session_duration * sessions_per_day / 60
    cases_per_day = daily_hours * 60 / procedure_time
    st.caption(f"Daily capacity: ~{daily_hours:.0f} theatre-hours, ~{cases_per_day:.0f} cases")

# ============================================================
# SECTION 5: ITU (Intensive Care)
# ============================================================
st.header("5. Intensive Care Unit (ITU)")

with st.container(border=True):
    st.markdown("""
    Critical care beds for patients requiring intensive monitoring.
    Patients step-down to Ward when stable (two-phase blocking if Ward full).

    **Simulation status**: UI ready, simulation wiring in Phase 8b+
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capacity**")

        n_itu = st.number_input(
            "ITU Beds",
            min_value=1,
            max_value=50,
            value=st.session_state.n_itu_beds,
            key="input_n_itu"
        )
        st.session_state.n_itu_beds = n_itu

    with col2:
        st.markdown("**Length of Stay**")

        itu_los = st.number_input(
            "Mean LoS (hours)",
            min_value=12.0,
            max_value=336.0,
            value=float(st.session_state.itu_los_hours_mean),
            step=6.0,
            key="input_itu_los"
        )
        st.session_state.itu_los_hours_mean = itu_los

        itu_cv = st.slider(
            "LoS Variability (CV)",
            min_value=0.3,
            max_value=1.5,
            value=float(st.session_state.itu_los_cv),
            step=0.1,
            key="input_itu_cv",
            help="Higher = more variation (typical ITU has high variability)"
        )
        st.session_state.itu_los_cv = itu_cv

    # Outcomes
    st.markdown("**Outcomes**")
    col1, col2, col3 = st.columns(3)

    with col1:
        stepdown = st.slider(
            "Step-down to Ward (%)",
            min_value=50,
            max_value=95,
            value=int(st.session_state.itu_stepdown_probability * 100),
            key="input_itu_stepdown"
        )
        st.session_state.itu_stepdown_probability = stepdown / 100

    with col2:
        mortality = st.slider(
            "Mortality (%)",
            min_value=0,
            max_value=30,
            value=int(st.session_state.itu_mortality_probability * 100),
            key="input_itu_mortality"
        )
        st.session_state.itu_mortality_probability = mortality / 100

    with col3:
        direct_discharge = 100 - stepdown - mortality
        st.metric("Direct Discharge (%)", direct_discharge)

    st.caption(f"ITU LoS: ~{itu_los:.0f}h ({itu_los/24:.1f} days)")

# ============================================================
# SECTION 6: WARD
# ============================================================
st.header("6. Ward Beds")

with st.container(border=True):
    st.markdown("""
    General ward beds for patients requiring admission but not critical care.
    Ward is often the **system bottleneck** - when full, upstream blocks occur.

    **Critical feedback**: Ward full -> ITU can't step-down -> ED can't admit -> Handover delays

    **Simulation status**: UI ready, full wiring in Phase 8b+
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Capacity**")

        n_ward = st.number_input(
            "Ward Beds",
            min_value=10,
            max_value=200,
            value=st.session_state.n_ward_beds,
            key="input_n_ward"
        )
        st.session_state.n_ward_beds = n_ward

        turnaround = st.number_input(
            "Bed Turnaround (mins)",
            min_value=15.0,
            max_value=90.0,
            value=float(st.session_state.ward_turnaround_mins),
            step=5.0,
            key="input_ward_turnaround",
            help="Cleaning/prep time between patients"
        )
        st.session_state.ward_turnaround_mins = turnaround

    with col2:
        st.markdown("**Length of Stay**")

        ward_los = st.number_input(
            "Mean LoS (hours)",
            min_value=24.0,
            max_value=336.0,
            value=float(st.session_state.ward_los_hours_mean),
            step=12.0,
            key="input_ward_los"
        )
        st.session_state.ward_los_hours_mean = ward_los

        ward_cv = st.slider(
            "LoS Variability (CV)",
            min_value=0.3,
            max_value=1.5,
            value=float(st.session_state.ward_los_cv),
            step=0.1,
            key="input_ward_cv",
            help="Higher = more variation (complex patients stay much longer)"
        )
        st.session_state.ward_los_cv = ward_cv

    st.caption(f"Ward LoS: ~{ward_los:.0f}h ({ward_los/24:.1f} days)")

# ============================================================
# DISTRIBUTION PREVIEW (Optional)
# ============================================================
st.markdown("---")

with st.expander("Distribution Preview - Visualize Length of Stay Variability", expanded=False):
    st.markdown("""
    This section shows how the **Coefficient of Variation (CV)** affects the distribution
    of Length of Stay times for ITU and Ward. Higher CV means more variability - some
    patients have very short stays while others stay much longer than the mean.
    """)

    # Current settings readback
    st.subheader("Current Settings")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ITU Length of Stay**")
        st.write(f"- Mean: **{st.session_state.itu_los_hours_mean:.0f} hours** ({st.session_state.itu_los_hours_mean/24:.1f} days)")
        st.write(f"- CV: **{st.session_state.itu_los_cv:.2f}**")

    with col2:
        st.markdown("**Ward Length of Stay**")
        st.write(f"- Mean: **{st.session_state.ward_los_hours_mean:.0f} hours** ({st.session_state.ward_los_hours_mean/24:.1f} days)")
        st.write(f"- CV: **{st.session_state.ward_los_cv:.2f}**")

    # Generate distribution button for performance
    if st.button("Generate Distribution Preview", key="btn_generate_dist"):
        st.session_state.show_dist_preview = True

    if st.session_state.get('show_dist_preview', False):
        # Helper function to compute lognormal parameters from mean and CV
        def lognormal_params(mean: float, cv: float):
            """Convert mean and CV to lognormal mu (scale) and sigma (shape)."""
            variance = (cv * mean) ** 2
            sigma_sq = np.log(1 + variance / mean**2)
            sigma = np.sqrt(sigma_sq)
            mu = np.log(mean) - sigma_sq / 2
            return mu, sigma

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ITU Length of Stay", "Ward Length of Stay")
        )

        # ITU distribution
        itu_mean = st.session_state.itu_los_hours_mean
        itu_cv = st.session_state.itu_los_cv
        itu_mu, itu_sigma = lognormal_params(itu_mean, itu_cv)

        # Generate x values (0 to 3x mean to show tail)
        itu_x = np.linspace(0.1, itu_mean * 4, 500)
        itu_y = stats.lognorm.pdf(itu_x, s=itu_sigma, scale=np.exp(itu_mu))

        fig.add_trace(
            go.Scatter(x=itu_x, y=itu_y, mode='lines', name='ITU LoS',
                      fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.3)',
                      line=dict(color='rgb(31, 119, 180)', width=2)),
            row=1, col=1
        )

        # Add percentile markers for ITU
        itu_p50 = stats.lognorm.ppf(0.50, s=itu_sigma, scale=np.exp(itu_mu))
        itu_p95 = stats.lognorm.ppf(0.95, s=itu_sigma, scale=np.exp(itu_mu))
        itu_p99 = stats.lognorm.ppf(0.99, s=itu_sigma, scale=np.exp(itu_mu))

        fig.add_vline(x=itu_p50, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_vline(x=itu_p95, line_dash="dash", line_color="orange", row=1, col=1)

        # Ward distribution
        ward_mean = st.session_state.ward_los_hours_mean
        ward_cv = st.session_state.ward_los_cv
        ward_mu, ward_sigma = lognormal_params(ward_mean, ward_cv)

        ward_x = np.linspace(0.1, ward_mean * 4, 500)
        ward_y = stats.lognorm.pdf(ward_x, s=ward_sigma, scale=np.exp(ward_mu))

        fig.add_trace(
            go.Scatter(x=ward_x, y=ward_y, mode='lines', name='Ward LoS',
                      fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)',
                      line=dict(color='rgb(255, 127, 14)', width=2)),
            row=1, col=2
        )

        # Add percentile markers for Ward
        ward_p50 = stats.lognorm.ppf(0.50, s=ward_sigma, scale=np.exp(ward_mu))
        ward_p95 = stats.lognorm.ppf(0.95, s=ward_sigma, scale=np.exp(ward_mu))
        ward_p99 = stats.lognorm.ppf(0.99, s=ward_sigma, scale=np.exp(ward_mu))

        fig.add_vline(x=ward_p50, line_dash="dash", line_color="green", row=1, col=2)
        fig.add_vline(x=ward_p95, line_dash="dash", line_color="orange", row=1, col=2)

        # Update layout
        fig.update_xaxes(title_text="Hours", row=1, col=1)
        fig.update_xaxes(title_text="Hours", row=1, col=2)
        fig.update_yaxes(title_text="Probability Density", row=1, col=1)
        fig.update_yaxes(title_text="Probability Density", row=1, col=2)

        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Percentile interpretation
        st.markdown("---")
        st.subheader("Percentile Interpretation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**ITU**")
            st.write(f"- Median (P50): **{itu_p50:.0f}h** ({itu_p50/24:.1f} days)")
            st.write(f"- 95th percentile: **{itu_p95:.0f}h** ({itu_p95/24:.1f} days)")
            st.write(f"- 99th percentile: **{itu_p99:.0f}h** ({itu_p99/24:.1f} days)")
            st.caption(f"5% of patients stay longer than {itu_p95:.0f} hours")

        with col2:
            st.markdown("**Ward**")
            st.write(f"- Median (P50): **{ward_p50:.0f}h** ({ward_p50/24:.1f} days)")
            st.write(f"- 95th percentile: **{ward_p95:.0f}h** ({ward_p95/24:.1f} days)")
            st.write(f"- 99th percentile: **{ward_p99:.0f}h** ({ward_p99/24:.1f} days)")
            st.caption(f"5% of patients stay longer than {ward_p95:.0f} hours")

        # Legend explanation
        st.markdown("""
        **Chart Legend:**
        - Green dashed line = Median (50th percentile)
        - Orange dashed line = 95th percentile

        **Why this matters:** High CV values create a "long tail" where a small proportion
        of patients have very extended stays. These patients can block beds and cause
        upstream congestion, even when average utilization looks manageable.
        """)

# ============================================================
# SECTION 7: RESOURCE SUMMARY
# ============================================================
st.markdown("---")
st.header("Resource Summary")

with st.container(border=True):
    summary_data = {
        'Resource': [
            'Triage',
            'ED Bays',
            'CT Scanner',
            'X-ray',
            'Bloods',
            'Theatre',
            'ITU',
            'Ward'
        ],
        'Capacity': [
            st.session_state.n_triage,
            st.session_state.n_ed_bays,
            st.session_state.ct_capacity if st.session_state.ct_enabled else '-',
            st.session_state.xray_capacity if st.session_state.xray_enabled else '-',
            st.session_state.bloods_capacity if st.session_state.bloods_enabled else '-',
            st.session_state.n_theatre_tables,
            st.session_state.n_itu_beds,
            st.session_state.n_ward_beds,
        ],
        'Key Timing': [
            f"{st.session_state.triage_time_mean:.0f}m",
            f"{st.session_state.ed_assessment_time_mean:.0f}+{st.session_state.ed_treatment_time_mean:.0f}m",
            f"{st.session_state.ct_scan_time_mean:.0f}+{st.session_state.ct_report_time_mean:.0f}m" if st.session_state.ct_enabled else '-',
            f"{st.session_state.xray_time_mean:.0f}+{st.session_state.xray_report_time_mean:.0f}m" if st.session_state.xray_enabled else '-',
            f"{st.session_state.bloods_draw_time_mean:.0f}+{st.session_state.bloods_lab_time_mean:.0f}m" if st.session_state.bloods_enabled else '-',
            f"{st.session_state.theatre_procedure_time_mean:.0f}m procedure",
            f"{st.session_state.itu_los_hours_mean:.0f}h LoS",
            f"{st.session_state.ward_los_hours_mean:.0f}h LoS",
        ],
        'Sim Status': [
            'Active',
            'Active',
            'Active' if st.session_state.ct_enabled else 'Disabled',
            'Active' if st.session_state.xray_enabled else 'Disabled',
            'Active' if st.session_state.bloods_enabled else 'Disabled',
            'UI Ready',
            'UI Ready',
            'UI Ready',
        ]
    }

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    # Total bed count
    total_beds = st.session_state.n_ed_bays + st.session_state.n_itu_beds + st.session_state.n_ward_beds
    st.metric("Total Bed Capacity", total_beds, help="ED + ITU + Ward")

# ============================================================
# CAPACITY RATIOS & VALIDATION
# ============================================================
st.markdown("---")
st.header("Capacity Ratios & Validation")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bed Ratios**")

        ed = st.session_state.n_ed_bays
        itu = st.session_state.n_itu_beds
        ward = st.session_state.n_ward_beds

        st.write(f"ED : ITU = {ed} : {itu} = {ed/itu:.1f} : 1")
        st.write(f"ED : Ward = {ed} : {ward} = {ed/ward:.2f} : 1")
        st.write(f"ITU : Ward = {itu} : {ward} = 1 : {ward/itu:.1f}")

    with col2:
        st.markdown("**Potential Issues**")

        issues = []

        if ward < ed:
            issues.append("Ward beds < ED bays (likely downstream bottleneck)")

        if itu < 3:
            issues.append("Very low ITU capacity (< 3 beds)")

        if st.session_state.ct_enabled and st.session_state.ct_capacity < 2:
            issues.append("Single CT scanner (no redundancy)")

        if st.session_state.n_triage < 2:
            issues.append("Single triage clinician (no redundancy)")

        if not issues:
            st.success("No obvious capacity issues detected")
        else:
            for issue in issues:
                st.warning(issue)

# ============================================================
# BUILD DIAGNOSTIC CONFIGS HELPER
# ============================================================
def build_diagnostic_configs() -> Dict[DiagnosticType, DiagnosticConfig]:
    """Build diagnostic configs from session state for FullScenario."""
    configs = {}

    if st.session_state.ct_enabled:
        configs[DiagnosticType.CT_SCAN] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.CT_SCAN,
            capacity=st.session_state.ct_capacity,
            process_time_mean=st.session_state.ct_scan_time_mean,
            turnaround_time_mean=st.session_state.ct_report_time_mean,
            process_time_cv=st.session_state.get('ct_time_cv', 0.3),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.70,
                Priority.P2_VERY_URGENT: 0.40,
                Priority.P3_URGENT: 0.15,
                Priority.P4_STANDARD: 0.05,
            }
        )

    if st.session_state.xray_enabled:
        configs[DiagnosticType.XRAY] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.XRAY,
            capacity=st.session_state.xray_capacity,
            process_time_mean=st.session_state.xray_time_mean,
            turnaround_time_mean=st.session_state.xray_report_time_mean,
            process_time_cv=st.session_state.get('xray_time_cv', 0.3),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.30,
                Priority.P2_VERY_URGENT: 0.35,
                Priority.P3_URGENT: 0.40,
                Priority.P4_STANDARD: 0.25,
            }
        )

    if st.session_state.bloods_enabled:
        configs[DiagnosticType.BLOODS] = DiagnosticConfig(
            diagnostic_type=DiagnosticType.BLOODS,
            capacity=st.session_state.bloods_capacity,
            process_time_mean=st.session_state.bloods_draw_time_mean,
            turnaround_time_mean=st.session_state.bloods_lab_time_mean,
            process_time_cv=st.session_state.get('bloods_time_cv', 0.3),
            probability_by_priority={
                Priority.P1_IMMEDIATE: 0.90,
                Priority.P2_VERY_URGENT: 0.80,
                Priority.P3_URGENT: 0.50,
                Priority.P4_STANDARD: 0.20,
            }
        )

    return configs


# Store the helper function in session state for use by Run page
st.session_state.build_diagnostic_configs = build_diagnostic_configs

# ============================================================
# NAVIGATION
# ============================================================
st.markdown("---")
st.info("""
**Next Steps:**
- Go to **3_Schematic** to visualize the full system with your configuration
- Go to **Run** to execute the simulation
- Go to **1_Arrivals** if you need to adjust arrival patterns

All settings are stored in session state and will be combined into a `FullScenario` when you run.
""")
