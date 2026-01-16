"""
Capacity Scaling Configuration Page.

Configure dynamic capacity scaling rules for surge protocols,
discharge acceleration, and OPEL-based escalation.
"""

import streamlit as st
from dataclasses import asdict

from faer.core.scaling import (
    ScalingTriggerType,
    ScalingActionType,
    ScalingTrigger,
    ScalingAction,
    ScalingRule,
    CapacityScalingConfig,
    OPELConfig,
    OPELLevel,
    create_opel_rules,
)

st.set_page_config(
    page_title="Capacity Scaling",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.title("Capacity Scaling Configuration")

st.markdown("""
Configure **dynamic capacity scaling** to model surge protocols, discharge acceleration,
and adaptive resource management in response to patient flow peaks.
""")

# Initialize session state
if "scaling_config" not in st.session_state:
    st.session_state.scaling_config = CapacityScalingConfig()

if "scaling_rules" not in st.session_state:
    st.session_state.scaling_rules = []

if "opel_config" not in st.session_state:
    st.session_state.opel_config = OPELConfig()

if "show_rule_editor" not in st.session_state:
    st.session_state.show_rule_editor = False

if "editing_rule_idx" not in st.session_state:
    st.session_state.editing_rule_idx = None


# ============== Global Settings ==============

st.header("Global Settings")

col1, col2, col3 = st.columns(3)

with col1:
    scaling_enabled = st.toggle(
        "Enable Capacity Scaling",
        value=st.session_state.scaling_config.enabled,
        help="Master switch for all capacity scaling features"
    )

with col2:
    eval_interval = st.number_input(
        "Evaluation Interval (min)",
        min_value=1.0,
        max_value=30.0,
        value=st.session_state.scaling_config.evaluation_interval_mins,
        step=1.0,
        help="How often to check trigger conditions"
    )

with col3:
    max_actions = st.number_input(
        "Max Concurrent Actions",
        min_value=1,
        max_value=10,
        value=st.session_state.scaling_config.max_simultaneous_actions,
        help="Limit on simultaneous scaling actions"
    )

st.divider()

# ============== NHS OPEL Framework ==============

st.header("NHS OPEL Framework")

with st.expander("What is OPEL?", expanded=False):
    st.markdown("""
    **OPEL = Operational Pressures Escalation Levels**

    NHS England's standard framework for communicating and responding to
    capacity pressure across acute trusts. Triggers coordinated actions
    at regional/system level.

    | Level | Status | Typical Triggers | Standard Actions |
    |-------|--------|------------------|------------------|
    | **OPEL 1** | Normal | ED <85%, Beds <85% | Business as usual |
    | **OPEL 2** | Moderate | ED 85-90%, Beds 85-90% | Focus on flow & discharge |
    | **OPEL 3** | Severe | ED 90-95%, Beds 90-95% | Surge capacity, discharge push |
    | **OPEL 4** | Critical | ED >95%, Beds >95% | Mutual aid, diverts |

    **Why these thresholds?**
    - **85%** = "functionally full" - no flex for surges
    - **90%** = corridor care likely, 4hr target at risk
    - **95%** = unsafe staffing ratios, care compromised
    """)

opel_enabled = st.toggle(
    "Enable OPEL-based Escalation",
    value=st.session_state.opel_config.enabled,
    help="Automatically generate scaling rules based on OPEL thresholds"
)

if opel_enabled:
    st.subheader("OPEL Thresholds")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**OPEL 2 (Moderate)**")
        opel_2_ed_pct = st.slider(
            "ED Trigger",
            min_value=70,
            max_value=95,
            value=int(st.session_state.opel_config.opel_2_ed_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_2_ed"
        )
        opel_2_ed = opel_2_ed_pct / 100.0
        opel_2_ward_pct = st.slider(
            "Ward Trigger",
            min_value=70,
            max_value=95,
            value=int(st.session_state.opel_config.opel_2_ward_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_2_ward"
        )
        opel_2_ward = opel_2_ward_pct / 100.0

    with col2:
        st.markdown("**OPEL 3 (Severe)**")
        opel_3_ed_pct = st.slider(
            "ED Trigger",
            min_value=80,
            max_value=98,
            value=int(st.session_state.opel_config.opel_3_ed_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_3_ed"
        )
        opel_3_ed = opel_3_ed_pct / 100.0
        opel_3_ward_pct = st.slider(
            "Ward Trigger",
            min_value=80,
            max_value=98,
            value=int(st.session_state.opel_config.opel_3_ward_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_3_ward"
        )
        opel_3_ward = opel_3_ward_pct / 100.0

    with col3:
        st.markdown("**OPEL 4 (Critical)**")
        opel_4_ed_pct = st.slider(
            "ED Trigger",
            min_value=85,
            max_value=100,
            value=int(st.session_state.opel_config.opel_4_ed_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_4_ed"
        )
        opel_4_ed = opel_4_ed_pct / 100.0
        opel_4_ward_pct = st.slider(
            "Ward Trigger",
            min_value=85,
            max_value=100,
            value=int(st.session_state.opel_config.opel_4_ward_threshold * 100),
            step=5,
            format="%d%%",
            key="opel_4_ward"
        )
        opel_4_ward = opel_4_ward_pct / 100.0

    st.subheader("OPEL Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OPEL 3 Actions**")
        opel_3_surge = st.number_input(
            "Surge Beds",
            min_value=0,
            max_value=20,
            value=st.session_state.opel_config.opel_3_surge_beds,
            help="Additional ED bays to open at OPEL 3"
        )
        opel_3_los = st.slider(
            "Discharge Acceleration (% LoS reduction)",
            min_value=0,
            max_value=30,
            value=int(st.session_state.opel_config.opel_3_los_reduction_pct),
            help="Reduce remaining length of stay by this percentage"
        )
        opel_3_lounge = st.checkbox(
            "Enable Discharge Lounge",
            value=st.session_state.opel_config.opel_3_enable_lounge
        )

    with col2:
        st.markdown("**OPEL 4 Actions**")
        opel_4_surge = st.number_input(
            "Full Surge Beds",
            min_value=0,
            max_value=30,
            value=st.session_state.opel_config.opel_4_surge_beds,
            help="Additional ED bays for full surge at OPEL 4"
        )
        opel_4_los = st.slider(
            "Aggressive Discharge (% LoS reduction)",
            min_value=0,
            max_value=40,
            value=int(st.session_state.opel_config.opel_4_los_reduction_pct),
            help="More aggressive LoS reduction at OPEL 4"
        )
        opel_4_divert = st.checkbox(
            "Enable Ambulance Diversion",
            value=st.session_state.opel_config.opel_4_enable_divert
        )

    # Update OPEL config
    st.session_state.opel_config = OPELConfig(
        enabled=opel_enabled,
        opel_2_ed_threshold=opel_2_ed,
        opel_2_ward_threshold=opel_2_ward,
        opel_3_ed_threshold=opel_3_ed,
        opel_3_ward_threshold=opel_3_ward,
        opel_4_ed_threshold=opel_4_ed,
        opel_4_ward_threshold=opel_4_ward,
        opel_3_surge_beds=opel_3_surge,
        opel_3_los_reduction_pct=float(opel_3_los),
        opel_3_enable_lounge=opel_3_lounge,
        opel_4_surge_beds=opel_4_surge,
        opel_4_los_reduction_pct=float(opel_4_los),
        opel_4_enable_divert=opel_4_divert,
    )

    # Show generated rules
    if opel_enabled:
        generated_rules = create_opel_rules(st.session_state.opel_config)
        st.info(f"OPEL configuration will generate **{len(generated_rules)} scaling rules**")

        with st.expander("View Generated OPEL Rules"):
            for rule in generated_rules:
                st.markdown(f"**{rule.name}**")
                st.markdown(f"- Trigger: {rule.trigger.resource} {rule.trigger.trigger_type.value} {rule.trigger.threshold:.0%}")
                st.markdown(f"- Action: {rule.action.action_type.value}")
                if rule.action.magnitude > 0:
                    st.markdown(f"- Magnitude: +{rule.action.magnitude}")
                if rule.action.los_reduction_pct > 0:
                    st.markdown(f"- LoS Reduction: {rule.action.los_reduction_pct:.0f}%")
                st.markdown("---")

st.divider()

# ============== Custom Scaling Rules ==============

st.header("Custom Scaling Rules")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("Add custom rules beyond OPEL presets for specific scenarios.")

with col2:
    if st.button("Add New Rule", type="primary"):
        st.session_state.show_rule_editor = True
        st.session_state.editing_rule_idx = None

# Display existing custom rules
if st.session_state.scaling_rules:
    for idx, rule in enumerate(st.session_state.scaling_rules):
        with st.expander(f"Rule {idx+1}: {rule.name}", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown("**Trigger**")
                st.markdown(f"- Type: {rule.trigger.trigger_type.value}")
                st.markdown(f"- Resource: {rule.trigger.resource}")
                st.markdown(f"- Threshold: {rule.trigger.threshold:.0%}")
                st.markdown(f"- Sustain: {rule.trigger.sustain_mins:.0f} min")

            with col2:
                st.markdown("**Action**")
                st.markdown(f"- Type: {rule.action.action_type.value}")
                st.markdown(f"- Resource: {rule.action.resource}")
                if rule.action.magnitude > 0:
                    st.markdown(f"- Magnitude: {rule.action.magnitude}")
                if rule.action.los_reduction_pct > 0:
                    st.markdown(f"- LoS Reduction: {rule.action.los_reduction_pct:.0f}%")

            with col3:
                if st.button("Edit", key=f"edit_{idx}"):
                    st.session_state.show_rule_editor = True
                    st.session_state.editing_rule_idx = idx
                    st.rerun()
                if st.button("Delete", key=f"del_{idx}"):
                    st.session_state.scaling_rules.pop(idx)
                    st.rerun()

else:
    st.info("No custom rules defined. Use OPEL presets or add custom rules.")

# ============== Rule Editor ==============

if st.session_state.show_rule_editor:
    st.divider()
    st.subheader("Rule Editor")

    # Pre-populate if editing
    if st.session_state.editing_rule_idx is not None:
        edit_rule = st.session_state.scaling_rules[st.session_state.editing_rule_idx]
        default_name = edit_rule.name
        default_trigger_type = edit_rule.trigger.trigger_type.value
        default_trigger_resource = edit_rule.trigger.resource
        default_threshold = edit_rule.trigger.threshold
        default_sustain = edit_rule.trigger.sustain_mins
        default_cooldown = edit_rule.trigger.cooldown_mins
        default_action_type = edit_rule.action.action_type.value
        default_action_resource = edit_rule.action.resource
        default_magnitude = edit_rule.action.magnitude
        default_los_pct = edit_rule.action.los_reduction_pct
        default_deescalate = edit_rule.auto_deescalate
        default_deesc_threshold = edit_rule.deescalation_threshold or 0.7
        default_deesc_delay = edit_rule.deescalation_delay_mins
    else:
        default_name = "New Rule"
        default_trigger_type = "utilization_above"
        default_trigger_resource = "ed_bays"
        default_threshold = 0.85
        default_sustain = 15.0
        default_cooldown = 60.0
        default_action_type = "add_capacity"
        default_action_resource = "ed_bays"
        default_magnitude = 5
        default_los_pct = 0.0
        default_deescalate = True
        default_deesc_threshold = 0.70
        default_deesc_delay = 30.0

    rule_name = st.text_input("Rule Name", value=default_name)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trigger**")

        trigger_type = st.selectbox(
            "Trigger Type",
            options=[t.value for t in ScalingTriggerType],
            index=[t.value for t in ScalingTriggerType].index(default_trigger_type)
        )

        trigger_resource = st.selectbox(
            "Resource to Monitor",
            options=["ed_bays", "ward_beds", "itu_beds", "triage"],
            index=["ed_bays", "ward_beds", "itu_beds", "triage"].index(default_trigger_resource) if default_trigger_resource in ["ed_bays", "ward_beds", "itu_beds", "triage"] else 0
        )

        if trigger_type in ["utilization_above", "utilization_below"]:
            threshold_pct = st.slider(
                "Threshold",
                min_value=0,
                max_value=100,
                value=int(default_threshold * 100),
                step=5,
                format="%d%%"
            )
            threshold = threshold_pct / 100.0
        elif trigger_type == "queue_length_above":
            threshold = float(st.number_input(
                "Queue Length Threshold",
                min_value=1,
                max_value=50,
                value=int(default_threshold)
            ))
        else:
            threshold = st.number_input(
                "Time (minutes)",
                min_value=0.0,
                value=default_threshold
            )

        sustain_mins = st.number_input(
            "Sustain (min)",
            min_value=0.0,
            max_value=60.0,
            value=default_sustain,
            help="Condition must hold for this long before triggering"
        )

        cooldown_mins = st.number_input(
            "Cooldown (min)",
            min_value=0.0,
            max_value=240.0,
            value=default_cooldown,
            help="Minimum time between activations"
        )

    with col2:
        st.markdown("**Action**")

        action_type = st.selectbox(
            "Action Type",
            options=[a.value for a in ScalingActionType],
            index=[a.value for a in ScalingActionType].index(default_action_type)
        )

        action_resource = st.selectbox(
            "Target Resource",
            options=["ed_bays", "ward_beds", "itu_beds", "triage", "arrivals", "discharge_lounge"],
            index=0
        )

        magnitude = 0
        los_reduction = 0.0

        if action_type in ["add_capacity", "remove_capacity"]:
            magnitude = st.number_input(
                "Amount",
                min_value=1,
                max_value=30,
                value=default_magnitude
            )
        elif action_type == "accelerate_discharge":
            los_reduction = st.slider(
                "LoS Reduction %",
                min_value=0,
                max_value=40,
                value=int(default_los_pct)
            )

        st.markdown("**De-escalation**")

        auto_deescalate = st.checkbox(
            "Auto de-escalate",
            value=default_deescalate,
            help="Automatically reverse action when pressure eases"
        )

        if auto_deescalate:
            deesc_threshold_pct = st.slider(
                "De-escalate below",
                min_value=0,
                max_value=100,
                value=int(default_deesc_threshold * 100),
                step=5,
                format="%d%%"
            )
            deesc_threshold = deesc_threshold_pct / 100.0
            deesc_delay = st.number_input(
                "Delay (min)",
                min_value=0.0,
                max_value=120.0,
                value=default_deesc_delay
            )
        else:
            deesc_threshold = None
            deesc_delay = 30.0

    # Save/Cancel buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Save Rule", type="primary"):
            new_rule = ScalingRule(
                name=rule_name,
                trigger=ScalingTrigger(
                    trigger_type=ScalingTriggerType(trigger_type),
                    resource=trigger_resource,
                    threshold=threshold,
                    sustain_mins=sustain_mins,
                    cooldown_mins=cooldown_mins
                ),
                action=ScalingAction(
                    action_type=ScalingActionType(action_type),
                    resource=action_resource,
                    magnitude=magnitude,
                    los_reduction_pct=float(los_reduction)
                ),
                auto_deescalate=auto_deescalate,
                deescalation_threshold=deesc_threshold if auto_deescalate else None,
                deescalation_delay_mins=deesc_delay
            )

            if st.session_state.editing_rule_idx is not None:
                st.session_state.scaling_rules[st.session_state.editing_rule_idx] = new_rule
            else:
                st.session_state.scaling_rules.append(new_rule)

            st.session_state.show_rule_editor = False
            st.session_state.editing_rule_idx = None
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.session_state.show_rule_editor = False
            st.session_state.editing_rule_idx = None
            st.rerun()

st.divider()

# ============== Discharge Lounge ==============

st.header("Discharge Lounge")

st.markdown("""
The discharge lounge allows patients who are medically fit but awaiting
transport/paperwork to free their bed immediately.
""")

col1, col2, col3 = st.columns(3)

with col1:
    lounge_capacity = st.number_input(
        "Lounge Capacity",
        min_value=0,
        max_value=30,
        value=st.session_state.scaling_config.discharge_lounge_capacity,
        help="Number of spaces in discharge lounge (0 = disabled)"
    )

with col2:
    lounge_max_wait = st.number_input(
        "Max Wait (min)",
        min_value=30,
        max_value=480,
        value=int(st.session_state.scaling_config.discharge_lounge_max_wait_mins),
        help="Maximum time a patient can wait in the lounge"
    )

with col3:
    if lounge_capacity > 0:
        st.success(f"Lounge enabled with {lounge_capacity} spaces")
    else:
        st.info("Discharge lounge disabled")

st.divider()

# ============== Quick Presets ==============

st.header("Quick Presets")

st.markdown("Load pre-configured scaling scenarios:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Standard Surge"):
        st.session_state.opel_config = OPELConfig(enabled=True)
        st.session_state.scaling_rules = []
        st.success("Loaded standard OPEL-based surge protocol")
        st.rerun()

with col2:
    if st.button("Discharge Push"):
        st.session_state.scaling_rules = [
            ScalingRule(
                name="Discharge Push",
                trigger=ScalingTrigger(
                    trigger_type=ScalingTriggerType.UTILIZATION_ABOVE,
                    resource="ward_beds",
                    threshold=0.90,
                    sustain_mins=15
                ),
                action=ScalingAction(
                    action_type=ScalingActionType.ACCELERATE_DISCHARGE,
                    resource="ward_beds",
                    los_reduction_pct=15.0
                ),
                deescalation_threshold=0.75
            )
        ]
        st.success("Loaded discharge push protocol")
        st.rerun()

with col3:
    if st.button("Winter Pressure"):
        st.session_state.opel_config = OPELConfig(
            enabled=True,
            opel_3_surge_beds=8,
            opel_4_surge_beds=15,
            opel_3_los_reduction_pct=15.0,
            opel_4_los_reduction_pct=25.0,
        )
        st.success("Loaded winter pressure configuration")
        st.rerun()

with col4:
    if st.button("Clear All"):
        st.session_state.opel_config = OPELConfig(enabled=False)
        st.session_state.scaling_rules = []
        st.info("Cleared all scaling configuration")
        st.rerun()

st.divider()

# ============== Build Final Config ==============

# Build the complete configuration
st.session_state.scaling_config = CapacityScalingConfig(
    enabled=scaling_enabled,
    rules=st.session_state.scaling_rules,
    opel_config=st.session_state.opel_config,
    evaluation_interval_mins=eval_interval,
    max_simultaneous_actions=max_actions,
    discharge_lounge_capacity=lounge_capacity,
    discharge_lounge_max_wait_mins=float(lounge_max_wait),
)

# ============== Summary ==============

st.header("Configuration Summary")

if scaling_enabled:
    total_rules = len(st.session_state.scaling_rules)
    if st.session_state.opel_config.enabled:
        total_rules += len(create_opel_rules(st.session_state.opel_config))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rules", total_rules)

    with col2:
        st.metric("OPEL Enabled", "Yes" if st.session_state.opel_config.enabled else "No")

    with col3:
        st.metric("Lounge Capacity", lounge_capacity if lounge_capacity > 0 else "Disabled")

    st.success(f"Capacity scaling **enabled** with {total_rules} rules")
else:
    st.warning("Capacity scaling is **disabled**. Enable it above to use these features.")

# Navigation hint
st.info("After configuring scaling rules, go to **Run** to execute the simulation with capacity scaling enabled.")
