"""
Aeromedical Evacuation Configuration Page (Phase 10-11).

Configures HEMS (helicopter) and fixed-wing evacuation options for P1 patients.
Aeromed patients hold ward beds during stabilisation and transport wait,
creating upstream blocking effects.

Integration:
- Stores values to st.session_state for FullScenario construction
- Creates AeromedConfig, HEMSConfig, FixedWingConfig, MissedSlotConfig
- Compatible with existing run_full_simulation() flow
"""

import streamlit as st
from typing import List

from faer.core.scenario import (
    AeromedConfig,
    HEMSConfig,
    FixedWingConfig,
    MissedSlotConfig,
)

st.set_page_config(
    page_title="Aeromed",
    page_icon="",
    layout="wide"
)

st.title("Aeromedical Evacuation")

st.info("""
Configure aeromedical evacuation for critical patients requiring transfer to specialist facilities.

**Key behaviours:**
- Only **P1 (Immediate)** patients are eligible for aeromed evacuation
- Patients **hold their ward bed** during stabilisation and transport wait (creates upstream blocking)
- **HEMS** (helicopter): On-demand, constrained by daylight operating hours and slot availability
- **Fixed-wing**: Scheduled slots, requires longer stabilisation but handles longer distances

**Simulation linkage**: Creates `simpy.Resource` for HEMS slots; fixed-wing uses slot scheduling
""")


# ============================================================
# HELPER: Initialize session state with defaults
# ============================================================
def init_aeromed_defaults():
    """Initialize aeromed-related session state with AeromedConfig defaults."""
    defaults = {
        # Master enable
        'aeromed_enabled': False,
        'p1_aeromed_probability': 0.05,
        'fixedwing_proportion': 0.30,

        # HEMS
        'hems_enabled': True,
        'hems_slots_per_day': 6,
        'hems_operating_start_hour': 7,
        'hems_operating_end_hour': 21,
        'hems_stabilisation_min': 30.0,
        'hems_stabilisation_max': 120.0,
        'hems_transfer_to_helipad_min': 15.0,
        'hems_transfer_to_helipad_max': 45.0,
        'hems_flight_duration_min': 15.0,
        'hems_flight_duration_max': 60.0,

        # Fixed-wing
        'fixedwing_enabled': False,
        'fixedwing_slots_am': 1,
        'fixedwing_slots_pm': 0,
        'fixedwing_departure_hour_in_segment': 2,
        'fixedwing_cutoff_hours_before': 4,
        'fixedwing_stabilisation_min': 120.0,
        'fixedwing_stabilisation_max': 240.0,
        'fixedwing_transport_to_airfield_min': 30.0,
        'fixedwing_transport_to_airfield_max': 90.0,
        'fixedwing_flight_duration_min': 60.0,
        'fixedwing_flight_duration_max': 180.0,

        # Missed slot
        'missed_slot_requires_restab': False,
        'missed_slot_restab_factor': 0.30,
        'missed_slot_max_wait_hours': 24.0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize defaults
init_aeromed_defaults()


# ============================================================
# SECTION 1: MASTER ENABLE
# ============================================================
st.header("1. Aeromed Settings")

with st.container(border=True):
    col1, col2 = st.columns([1, 2])

    with col1:
        aeromed_enabled = st.toggle(
            "Enable Aeromedical Evacuation",
            value=st.session_state.aeromed_enabled,
            key="input_aeromed_enabled",
            help="Master switch for all aeromed functionality"
        )
        st.session_state.aeromed_enabled = aeromed_enabled

    with col2:
        if aeromed_enabled:
            st.success("Aeromed evacuation is **enabled**")
            st.markdown("""
            - P1 patients may be selected for aeromed based on probability below
            - Patients hold ward beds during stabilisation and transport wait
            - This creates **upstream blocking** effects when slots are limited
            """)
        else:
            st.warning("Aeromed evacuation is **disabled** - all patients follow standard pathways")

    if aeromed_enabled:
        st.markdown("---")
        st.markdown("**Patient Selection**")

        col1, col2 = st.columns(2)

        with col1:
            p1_prob = st.slider(
                "P1 Aeromed Probability (%)",
                min_value=0,
                max_value=50,
                value=int(st.session_state.p1_aeromed_probability * 100),
                step=1,
                key="input_p1_aeromed_prob",
                help="Percentage of P1 patients requiring aeromed evacuation",
                disabled=not aeromed_enabled
            )
            st.session_state.p1_aeromed_probability = p1_prob / 100

        with col2:
            fw_prop = st.slider(
                "Fixed-Wing Proportion (%)",
                min_value=0,
                max_value=80,
                value=int(st.session_state.fixedwing_proportion * 100),
                step=5,
                key="input_fw_proportion",
                help="Of aeromed patients, what % use fixed-wing vs HEMS",
                disabled=not aeromed_enabled
            )
            st.session_state.fixedwing_proportion = fw_prop / 100

        # Visual breakdown
        hems_pct = 100 - fw_prop
        st.caption(f"Aeromed split: **{hems_pct}% HEMS** | **{fw_prop}% Fixed-Wing**")


# ============================================================
# SECTION 2: HEMS (Helicopter Emergency Medical Services)
# ============================================================
st.header("2. HEMS (Helicopter)")

with st.container(border=True):
    if not aeromed_enabled:
        st.warning("Enable aeromed above to configure HEMS")
    else:
        st.markdown("""
        Helicopter evacuation for time-critical transfers.
        Operates **daylight hours only** with limited concurrent slots.

        **Simulation**: `simpy.Resource(hems_slots)` with operating hours constraint
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            hems_enabled = st.checkbox(
                "HEMS Enabled",
                value=st.session_state.hems_enabled,
                key="input_hems_enabled",
                disabled=not aeromed_enabled
            )
            st.session_state.hems_enabled = hems_enabled

        with col2:
            if hems_enabled:
                st.success("HEMS is available for aeromed patients")
            else:
                st.info("HEMS disabled - aeromed patients will use fixed-wing only")

        if hems_enabled and aeromed_enabled:
            st.markdown("---")

            # Operating hours and slots
            st.markdown("**Operations**")
            col1, col2, col3 = st.columns(3)

            with col1:
                hems_slots = st.number_input(
                    "Slots per Day",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.hems_slots_per_day,
                    key="input_hems_slots",
                    help="Concurrent HEMS missions available"
                )
                st.session_state.hems_slots_per_day = hems_slots

            with col2:
                hems_start = st.number_input(
                    "Operating Start (hour)",
                    min_value=5,
                    max_value=10,
                    value=st.session_state.hems_operating_start_hour,
                    key="input_hems_start",
                    help="First hour of HEMS operations"
                )
                st.session_state.hems_operating_start_hour = hems_start

            with col3:
                hems_end = st.number_input(
                    "Operating End (hour)",
                    min_value=18,
                    max_value=23,
                    value=st.session_state.hems_operating_end_hour,
                    key="input_hems_end",
                    help="Last hour of HEMS operations"
                )
                st.session_state.hems_operating_end_hour = hems_end

            operating_hours = hems_end - hems_start
            st.caption(f"Operating window: **{hems_start:02d}:00 - {hems_end:02d}:00** ({operating_hours} hours/day)")

            # Timing parameters
            st.markdown("---")
            st.markdown("**Timing (min-max range in minutes)**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("*Stabilisation*")
                stab_min = st.number_input(
                    "Min (mins)",
                    min_value=10.0,
                    max_value=120.0,
                    value=float(st.session_state.hems_stabilisation_min),
                    step=5.0,
                    key="input_hems_stab_min"
                )
                st.session_state.hems_stabilisation_min = stab_min

                stab_max = st.number_input(
                    "Max (mins)",
                    min_value=30.0,
                    max_value=240.0,
                    value=float(st.session_state.hems_stabilisation_max),
                    step=10.0,
                    key="input_hems_stab_max"
                )
                st.session_state.hems_stabilisation_max = stab_max

            with col2:
                st.markdown("*Transfer to Helipad*")
                transfer_min = st.number_input(
                    "Min (mins)",
                    min_value=5.0,
                    max_value=60.0,
                    value=float(st.session_state.hems_transfer_to_helipad_min),
                    step=5.0,
                    key="input_hems_transfer_min"
                )
                st.session_state.hems_transfer_to_helipad_min = transfer_min

                transfer_max = st.number_input(
                    "Max (mins)",
                    min_value=15.0,
                    max_value=120.0,
                    value=float(st.session_state.hems_transfer_to_helipad_max),
                    step=5.0,
                    key="input_hems_transfer_max"
                )
                st.session_state.hems_transfer_to_helipad_max = transfer_max

            with col3:
                st.markdown("*Flight Duration*")
                flight_min = st.number_input(
                    "Min (mins)",
                    min_value=10.0,
                    max_value=60.0,
                    value=float(st.session_state.hems_flight_duration_min),
                    step=5.0,
                    key="input_hems_flight_min"
                )
                st.session_state.hems_flight_duration_min = flight_min

                flight_max = st.number_input(
                    "Max (mins)",
                    min_value=30.0,
                    max_value=180.0,
                    value=float(st.session_state.hems_flight_duration_max),
                    step=10.0,
                    key="input_hems_flight_max"
                )
                st.session_state.hems_flight_duration_max = flight_max

            # Typical total time
            avg_total = (stab_min + stab_max) / 2 + (transfer_min + transfer_max) / 2 + (flight_min + flight_max) / 2
            st.caption(f"Typical total HEMS time: **{avg_total:.0f} mins** (stabilisation + transfer + flight)")


# ============================================================
# SECTION 3: FIXED-WING
# ============================================================
st.header("3. Fixed-Wing Aircraft")

with st.container(border=True):
    if not aeromed_enabled:
        st.warning("Enable aeromed above to configure fixed-wing")
    else:
        st.markdown("""
        Fixed-wing evacuation for longer distance transfers.
        Operates on **scheduled slots** (AM/PM) with cutoff times.

        **Simulation**: Slot scheduling with 12-hour segments, pattern repeats
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            fw_enabled = st.checkbox(
                "Fixed-Wing Enabled",
                value=st.session_state.fixedwing_enabled,
                key="input_fw_enabled",
                disabled=not aeromed_enabled
            )
            st.session_state.fixedwing_enabled = fw_enabled

        with col2:
            if fw_enabled:
                st.success("Fixed-wing is available for aeromed patients")
            else:
                st.info("Fixed-wing disabled - aeromed patients will use HEMS only")

        if fw_enabled and aeromed_enabled:
            st.markdown("---")

            # Slot schedule
            st.markdown("**Slot Schedule**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                slots_am = st.number_input(
                    "AM Slots",
                    min_value=0,
                    max_value=5,
                    value=st.session_state.fixedwing_slots_am,
                    key="input_fw_slots_am",
                    help="Flights in morning segment (00:00-12:00)"
                )
                st.session_state.fixedwing_slots_am = slots_am

            with col2:
                slots_pm = st.number_input(
                    "PM Slots",
                    min_value=0,
                    max_value=5,
                    value=st.session_state.fixedwing_slots_pm,
                    key="input_fw_slots_pm",
                    help="Flights in afternoon segment (12:00-24:00)"
                )
                st.session_state.fixedwing_slots_pm = slots_pm

            with col3:
                departure_hour = st.number_input(
                    "Departure Hour (in segment)",
                    min_value=0,
                    max_value=11,
                    value=st.session_state.fixedwing_departure_hour_in_segment,
                    key="input_fw_departure_hour",
                    help="Hour within segment when flight departs (0=start of segment)"
                )
                st.session_state.fixedwing_departure_hour_in_segment = departure_hour

            with col4:
                cutoff_hours = st.number_input(
                    "Cutoff Hours Before",
                    min_value=1,
                    max_value=12,
                    value=st.session_state.fixedwing_cutoff_hours_before,
                    key="input_fw_cutoff",
                    help="Hours before departure that patient must be ready"
                )
                st.session_state.fixedwing_cutoff_hours_before = cutoff_hours

            daily_slots = slots_am + slots_pm
            st.caption(f"Daily capacity: **{daily_slots} flights** ({slots_am} AM + {slots_pm} PM)")

            # Show departure times
            if daily_slots > 0:
                am_time = f"{departure_hour:02d}:00" if slots_am > 0 else "-"
                pm_time = f"{12 + departure_hour:02d}:00" if slots_pm > 0 else "-"
                st.caption(f"Departure times: AM @ **{am_time}**, PM @ **{pm_time}**")

            # Timing parameters
            st.markdown("---")
            st.markdown("**Timing (min-max range in minutes)**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("*Stabilisation*")
                fw_stab_min = st.number_input(
                    "Min (mins)",
                    min_value=60.0,
                    max_value=240.0,
                    value=float(st.session_state.fixedwing_stabilisation_min),
                    step=15.0,
                    key="input_fw_stab_min"
                )
                st.session_state.fixedwing_stabilisation_min = fw_stab_min

                fw_stab_max = st.number_input(
                    "Max (mins)",
                    min_value=120.0,
                    max_value=480.0,
                    value=float(st.session_state.fixedwing_stabilisation_max),
                    step=30.0,
                    key="input_fw_stab_max"
                )
                st.session_state.fixedwing_stabilisation_max = fw_stab_max

            with col2:
                st.markdown("*Transport to Airfield*")
                fw_transport_min = st.number_input(
                    "Min (mins)",
                    min_value=15.0,
                    max_value=90.0,
                    value=float(st.session_state.fixedwing_transport_to_airfield_min),
                    step=5.0,
                    key="input_fw_transport_min"
                )
                st.session_state.fixedwing_transport_to_airfield_min = fw_transport_min

                fw_transport_max = st.number_input(
                    "Max (mins)",
                    min_value=30.0,
                    max_value=180.0,
                    value=float(st.session_state.fixedwing_transport_to_airfield_max),
                    step=10.0,
                    key="input_fw_transport_max"
                )
                st.session_state.fixedwing_transport_to_airfield_max = fw_transport_max

            with col3:
                st.markdown("*Flight Duration*")
                fw_flight_min = st.number_input(
                    "Min (mins)",
                    min_value=30.0,
                    max_value=120.0,
                    value=float(st.session_state.fixedwing_flight_duration_min),
                    step=10.0,
                    key="input_fw_flight_min"
                )
                st.session_state.fixedwing_flight_duration_min = fw_flight_min

                fw_flight_max = st.number_input(
                    "Max (mins)",
                    min_value=60.0,
                    max_value=360.0,
                    value=float(st.session_state.fixedwing_flight_duration_max),
                    step=30.0,
                    key="input_fw_flight_max"
                )
                st.session_state.fixedwing_flight_duration_max = fw_flight_max

            # Typical total time
            fw_avg_total = (fw_stab_min + fw_stab_max) / 2 + (fw_transport_min + fw_transport_max) / 2 + (fw_flight_min + fw_flight_max) / 2
            st.caption(f"Typical total fixed-wing time: **{fw_avg_total:.0f} mins** (stabilisation + transport + flight)")


# ============================================================
# SECTION 4: MISSED SLOT BEHAVIOUR
# ============================================================
st.header("4. Missed Slot Handling")

with st.container(border=True):
    if not aeromed_enabled:
        st.warning("Enable aeromed above to configure missed slot behaviour")
    else:
        st.markdown("""
        When a patient misses their scheduled evacuation slot (e.g., cutoff passed),
        they must wait for the next slot. This creates **blocking effects**.

        **Key impact**: Patient continues to hold ward bed while waiting.
        """)

        col1, col2 = st.columns(2)

        with col1:
            requires_restab = st.checkbox(
                "Requires Re-stabilisation",
                value=st.session_state.missed_slot_requires_restab,
                key="input_missed_restab",
                disabled=not aeromed_enabled,
                help="If checked, patients need partial re-stabilisation after missing slot"
            )
            st.session_state.missed_slot_requires_restab = requires_restab

            if requires_restab:
                restab_factor = st.slider(
                    "Re-stabilisation Factor",
                    min_value=0.1,
                    max_value=0.8,
                    value=float(st.session_state.missed_slot_restab_factor),
                    step=0.05,
                    key="input_restab_factor",
                    help="Fraction of original stabilisation time needed again"
                )
                st.session_state.missed_slot_restab_factor = restab_factor
                st.caption(f"Re-stabilisation = {restab_factor * 100:.0f}% of original stabilisation time")

        with col2:
            max_wait = st.number_input(
                "Max Wait Before Re-stab (hours)",
                min_value=6.0,
                max_value=72.0,
                value=float(st.session_state.missed_slot_max_wait_hours),
                step=6.0,
                key="input_max_wait_hours",
                disabled=not aeromed_enabled,
                help="If waiting longer than this, re-stabilisation is triggered"
            )
            st.session_state.missed_slot_max_wait_hours = max_wait

        st.info("""
        **Blocking cascade**: Missed slot wait time = extended ward bed blocking =
        reduced ward capacity = ITU can't step-down = ED can't admit = ambulance handover delays
        """)


# ============================================================
# SECTION 5: SUMMARY
# ============================================================
st.markdown("---")
st.header("Configuration Summary")

with st.container(border=True):
    if not st.session_state.aeromed_enabled:
        st.info("Aeromedical evacuation is **disabled**. All patients follow standard discharge pathways.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Patient Selection**")
            st.write(f"- P1 aeromed probability: {st.session_state.p1_aeromed_probability * 100:.0f}%")
            st.write(f"- HEMS proportion: {(1 - st.session_state.fixedwing_proportion) * 100:.0f}%")
            st.write(f"- Fixed-wing proportion: {st.session_state.fixedwing_proportion * 100:.0f}%")

        with col2:
            st.markdown("**HEMS**")
            if st.session_state.hems_enabled:
                st.write(f"- Slots/day: {st.session_state.hems_slots_per_day}")
                st.write(f"- Hours: {st.session_state.hems_operating_start_hour:02d}:00-{st.session_state.hems_operating_end_hour:02d}:00")
                avg_stab = (st.session_state.hems_stabilisation_min + st.session_state.hems_stabilisation_max) / 2
                st.write(f"- Avg stabilisation: {avg_stab:.0f} mins")
            else:
                st.write("- Disabled")

        with col3:
            st.markdown("**Fixed-Wing**")
            if st.session_state.fixedwing_enabled:
                st.write(f"- Slots/day: {st.session_state.fixedwing_slots_am + st.session_state.fixedwing_slots_pm}")
                st.write(f"- Cutoff: {st.session_state.fixedwing_cutoff_hours_before}h before")
                avg_fw_stab = (st.session_state.fixedwing_stabilisation_min + st.session_state.fixedwing_stabilisation_max) / 2
                st.write(f"- Avg stabilisation: {avg_fw_stab:.0f} mins")
            else:
                st.write("- Disabled")


# ============================================================
# BUILD AEROMED CONFIG HELPER
# ============================================================
def build_aeromed_config() -> AeromedConfig:
    """Build AeromedConfig from session state for FullScenario."""
    hems_config = HEMSConfig(
        enabled=st.session_state.hems_enabled,
        slots_per_day=st.session_state.hems_slots_per_day,
        operating_start_hour=st.session_state.hems_operating_start_hour,
        operating_end_hour=st.session_state.hems_operating_end_hour,
        stabilisation_mins=(
            st.session_state.hems_stabilisation_min,
            st.session_state.hems_stabilisation_max
        ),
        transfer_to_helipad_mins=(
            st.session_state.hems_transfer_to_helipad_min,
            st.session_state.hems_transfer_to_helipad_max
        ),
        flight_duration_mins=(
            st.session_state.hems_flight_duration_min,
            st.session_state.hems_flight_duration_max
        ),
    )

    fixedwing_config = FixedWingConfig(
        enabled=st.session_state.fixedwing_enabled,
        slots_per_segment=[
            st.session_state.fixedwing_slots_am,
            st.session_state.fixedwing_slots_pm
        ],
        departure_hour_in_segment=st.session_state.fixedwing_departure_hour_in_segment,
        cutoff_hours_before=st.session_state.fixedwing_cutoff_hours_before,
        stabilisation_mins=(
            st.session_state.fixedwing_stabilisation_min,
            st.session_state.fixedwing_stabilisation_max
        ),
        transport_to_airfield_mins=(
            st.session_state.fixedwing_transport_to_airfield_min,
            st.session_state.fixedwing_transport_to_airfield_max
        ),
        flight_duration_mins=(
            st.session_state.fixedwing_flight_duration_min,
            st.session_state.fixedwing_flight_duration_max
        ),
    )

    missed_slot_config = MissedSlotConfig(
        requires_restabilisation=st.session_state.missed_slot_requires_restab,
        restabilisation_factor=st.session_state.missed_slot_restab_factor,
        max_wait_before_restab_hours=st.session_state.missed_slot_max_wait_hours,
    )

    return AeromedConfig(
        enabled=st.session_state.aeromed_enabled,
        p1_aeromed_probability=st.session_state.p1_aeromed_probability,
        fixedwing_proportion=st.session_state.fixedwing_proportion,
        hems=hems_config,
        fixedwing=fixedwing_config,
        missed_slot=missed_slot_config,
    )


# Store the helper function in session state for use by Run page
st.session_state.build_aeromed_config = build_aeromed_config


# ============================================================
# NAVIGATION
# ============================================================
st.markdown("---")
st.info("""
**Next Steps:**
- Go to **Run** to execute the simulation with aeromed enabled
- Go to **Results** to view aeromed-specific metrics after running
- Go to **Resources** to configure ward/ITU capacity (affects blocking)

All settings are stored in session state and will be combined into a `FullScenario` when you run.
""")
