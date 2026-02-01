"""
FAER-M: MEDEVAC Chain Simulation Demo

Standalone demonstration of military medical evacuation chain modeling.
Uses Pydantic for validated configuration with Streamlit visibility.

This page is intentionally self-contained for portability.
Models can be extracted to src/faer/medevac/ for production use.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generator, Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import numpy as np
import simpy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════════════════════════

class TriageCategory(IntEnum):
    """Battlefield triage categories."""
    T1_IMMEDIATE = 1   # Life-threatening, salvageable
    T2_DELAYED = 2     # Serious, can wait
    T3_MINIMAL = 3     # Walking wounded
    T4_EXPECTANT = 4   # Non-survivable given resources


class Role(IntEnum):
    """NATO medical echelons."""
    POI = 0       # Point of Injury
    ROLE_1 = 1    # Battalion Aid Station
    ROLE_2 = 2    # Forward Surgical Team
    ROLE_3 = 3    # Combat Support Hospital
    ROLE_4 = 4    # Strategic Evacuation


class Outcome(IntEnum):
    """Patient outcomes."""
    IN_SYSTEM = 0
    RTD = 1        # Return to Duty
    EVACUATED = 2  # Strategic evacuation complete
    DIED = 3       # KIA / DOW


TRIAGE_LABELS = {
    TriageCategory.T1_IMMEDIATE: "T1 Immediate",
    TriageCategory.T2_DELAYED: "T2 Delayed",
    TriageCategory.T3_MINIMAL: "T3 Minimal",
    TriageCategory.T4_EXPECTANT: "T4 Expectant",
}

ROLE_LABELS = {
    Role.ROLE_1: "Role 1 (Battalion Aid)",
    Role.ROLE_2: "Role 2 (Forward Surg)",
    Role.ROLE_3: "Role 3 (Combat Hosp)",
    Role.ROLE_4: "Role 4 (Strategic Evac)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC-STYLE CONFIG (using dataclasses for zero dependencies)
# In production, replace with actual Pydantic BaseModel for validation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RoleConfig:
    """Configuration for a single medical echelon."""
    capacity: int
    service_mean_mins: float
    service_cv: float = 0.4

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError("Capacity must be >= 1")
        if self.service_mean_mins <= 0:
            raise ValueError("Service time must be > 0")


@dataclass(frozen=True)
class MEDEVACConfig:
    """Transport asset configuration."""
    ground_ambulances: int = 4
    rotary_wing: int = 2
    ground_transit_mean: float = 45.0
    rotary_transit_mean: float = 25.0
    transit_cv: float = 0.3


@dataclass(frozen=True)
class ChainScenario:
    """Complete MEDEVAC chain scenario configuration."""
    run_length_hours: int = 24
    casualty_rate_per_hour: float = 5.0

    role_1: RoleConfig = field(default_factory=lambda: RoleConfig(4, 15.0))
    role_2: RoleConfig = field(default_factory=lambda: RoleConfig(6, 90.0))
    role_3: RoleConfig = field(default_factory=lambda: RoleConfig(20, 180.0))

    medevac: MEDEVACConfig = field(default_factory=MEDEVACConfig)

    t1_proportion: float = 0.15
    t2_proportion: float = 0.30
    t3_proportion: float = 0.50
    t4_proportion: float = 0.05

    seed: int = 42

    def __post_init__(self):
        total = self.t1_proportion + self.t2_proportion + self.t3_proportion + self.t4_proportion
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Triage proportions must sum to 1.0, got {total}")

    @property
    def triage_probs(self) -> list[float]:
        return [self.t1_proportion, self.t2_proportion, self.t3_proportion, self.t4_proportion]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "Routine Operations": ChainScenario(),

    "Mass Casualty Incident": ChainScenario(
        casualty_rate_per_hour=15.0,
        t1_proportion=0.25,
        t2_proportion=0.35,
        t3_proportion=0.35,
        t4_proportion=0.05,
    ),

    "Contested Environment": ChainScenario(
        medevac=MEDEVACConfig(ground_ambulances=2, rotary_wing=0),
    ),

    "Prolonged Field Care": ChainScenario(
        role_1=RoleConfig(capacity=2, service_mean_mins=120.0),
        medevac=MEDEVACConfig(ground_ambulances=1, rotary_wing=1),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CASUALTY ENTITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Casualty:
    """Individual casualty moving through the chain."""
    id: int
    triage: TriageCategory
    arrival_time: float
    outcome: Outcome = Outcome.IN_SYSTEM

    # Timestamps
    role_1_start: Optional[float] = None
    role_1_end: Optional[float] = None
    role_2_start: Optional[float] = None
    role_2_end: Optional[float] = None
    role_3_start: Optional[float] = None
    role_3_end: Optional[float] = None
    departure_time: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainMetrics:
    """Collected simulation metrics."""
    casualties: list[Casualty] = field(default_factory=list)
    queue_snapshots: list[dict] = field(default_factory=list)

    def record_casualty(self, c: Casualty):
        self.casualties.append(c)

    def snapshot_queues(self, time: float, queues: dict[Role, int]):
        self.queue_snapshots.append({'time': time, **{r.name: q for r, q in queues.items()}})

    @property
    def throughput(self) -> dict[TriageCategory, int]:
        counts = {t: 0 for t in TriageCategory}
        for c in self.casualties:
            if c.outcome in (Outcome.RTD, Outcome.EVACUATED):
                counts[c.triage] += 1
        return counts

    @property
    def outcomes(self) -> dict[Outcome, int]:
        counts = {o: 0 for o in Outcome}
        for c in self.casualties:
            counts[c.outcome] += 1
        return counts

    @property
    def queue_times_by_role(self) -> dict[Role, list[float]]:
        result = {Role.ROLE_1: [], Role.ROLE_2: [], Role.ROLE_3: []}
        for c in self.casualties:
            if c.role_1_start is not None:
                result[Role.ROLE_1].append(c.role_1_start - c.arrival_time)
            if c.role_2_start is not None and c.role_1_end is not None:
                result[Role.ROLE_2].append(c.role_2_start - c.role_1_end)
            if c.role_3_start is not None and c.role_2_end is not None:
                result[Role.ROLE_3].append(c.role_3_start - c.role_2_end)
        return result

    @property
    def t1_survival_rate(self) -> float:
        t1_total = sum(1 for c in self.casualties if c.triage == TriageCategory.T1_IMMEDIATE)
        t1_survived = sum(1 for c in self.casualties
                         if c.triage == TriageCategory.T1_IMMEDIATE
                         and c.outcome in (Outcome.RTD, Outcome.EVACUATED))
        return t1_survived / t1_total if t1_total > 0 else 0.0

    @property
    def mean_time_to_surgery(self) -> float:
        times = []
        for c in self.casualties:
            if c.triage in (TriageCategory.T1_IMMEDIATE, TriageCategory.T2_DELAYED):
                if c.role_2_start is not None:
                    times.append(c.role_2_start - c.arrival_time)
        return np.mean(times) if times else 0.0

    def bottleneck_role(self) -> Role:
        queue_times = self.queue_times_by_role
        means = {r: np.mean(times) if times else 0.0 for r, times in queue_times.items()}
        return max(means, key=means.get)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def sample_lognormal(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Sample from lognormal distribution."""
    if cv <= 0 or mean <= 0:
        return mean
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean) - sigma**2 / 2
    return float(rng.lognormal(mu, sigma))


@dataclass
class ChainResources:
    """SimPy resources for the medical chain."""
    role_1: simpy.PriorityResource
    role_2: simpy.PriorityResource
    role_3: simpy.Resource
    medevac_ground: simpy.Resource
    medevac_rotary: simpy.Resource


def casualty_process(
    env: simpy.Environment,
    casualty: Casualty,
    resources: ChainResources,
    scenario: ChainScenario,
    rng: np.random.Generator,
    metrics: ChainMetrics,
) -> Generator:
    """Casualty journey through medical chain."""

    # ─────────────────────────────────────────────────────────────────────────
    # ROLE 1: Battalion Aid Station
    # ─────────────────────────────────────────────────────────────────────────
    with resources.role_1.request(priority=casualty.triage.value) as req:
        yield req
        casualty.role_1_start = env.now

        duration = sample_lognormal(rng, scenario.role_1.service_mean_mins, scenario.role_1.service_cv)
        yield env.timeout(duration)
        casualty.role_1_end = env.now

    # T3: Return to duty
    if casualty.triage == TriageCategory.T3_MINIMAL:
        casualty.outcome = Outcome.RTD
        casualty.departure_time = env.now
        metrics.record_casualty(casualty)
        return

    # T4: Expectant (comfort care only)
    if casualty.triage == TriageCategory.T4_EXPECTANT:
        casualty.outcome = Outcome.DIED
        casualty.departure_time = env.now
        metrics.record_casualty(casualty)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # MEDEVAC to Role 2
    # ─────────────────────────────────────────────────────────────────────────
    # Try rotary first for T1, ground otherwise
    if casualty.triage == TriageCategory.T1_IMMEDIATE and scenario.medevac.rotary_wing > 0:
        transport = resources.medevac_rotary
        transit_time = sample_lognormal(rng, scenario.medevac.rotary_transit_mean, scenario.medevac.transit_cv)
    else:
        transport = resources.medevac_ground
        transit_time = sample_lognormal(rng, scenario.medevac.ground_transit_mean, scenario.medevac.transit_cv)

    with transport.request() as req:
        yield req
        yield env.timeout(transit_time)

    # ─────────────────────────────────────────────────────────────────────────
    # ROLE 2: Forward Surgical Team
    # ─────────────────────────────────────────────────────────────────────────
    with resources.role_2.request(priority=casualty.triage.value) as req:
        yield req
        casualty.role_2_start = env.now

        duration = sample_lognormal(rng, scenario.role_2.service_mean_mins, scenario.role_2.service_cv)
        yield env.timeout(duration)
        casualty.role_2_end = env.now

    # ─────────────────────────────────────────────────────────────────────────
    # MEDEVAC to Role 3
    # ─────────────────────────────────────────────────────────────────────────
    with resources.medevac_ground.request() as req:
        yield req
        transit_time = sample_lognormal(rng, scenario.medevac.ground_transit_mean, scenario.medevac.transit_cv)
        yield env.timeout(transit_time)

    # ─────────────────────────────────────────────────────────────────────────
    # ROLE 3: Combat Support Hospital
    # ─────────────────────────────────────────────────────────────────────────
    with resources.role_3.request() as req:
        yield req
        casualty.role_3_start = env.now

        duration = sample_lognormal(rng, scenario.role_3.service_mean_mins, scenario.role_3.service_cv)
        yield env.timeout(duration)
        casualty.role_3_end = env.now

    # Strategic evacuation complete
    casualty.outcome = Outcome.EVACUATED
    casualty.departure_time = env.now
    metrics.record_casualty(casualty)


def arrival_generator(
    env: simpy.Environment,
    resources: ChainResources,
    scenario: ChainScenario,
    rng: np.random.Generator,
    metrics: ChainMetrics,
) -> Generator:
    """Generate casualties at POI."""
    casualty_id = 0

    while True:
        # Exponential inter-arrival time
        iat = rng.exponential(60.0 / scenario.casualty_rate_per_hour)
        yield env.timeout(iat)

        # Sample triage category
        triage = rng.choice(list(TriageCategory), p=scenario.triage_probs)

        casualty = Casualty(
            id=casualty_id,
            triage=triage,
            arrival_time=env.now,
        )
        casualty_id += 1

        env.process(casualty_process(env, casualty, resources, scenario, rng, metrics))


def queue_monitor(
    env: simpy.Environment,
    resources: ChainResources,
    metrics: ChainMetrics,
    interval: float = 10.0,
) -> Generator:
    """Periodically snapshot queue lengths."""
    while True:
        yield env.timeout(interval)
        metrics.snapshot_queues(env.now, {
            Role.ROLE_1: len(resources.role_1.queue),
            Role.ROLE_2: len(resources.role_2.queue),
            Role.ROLE_3: len(resources.role_3.queue),
        })


def run_simulation(scenario: ChainScenario) -> ChainMetrics:
    """Execute simulation and return metrics."""
    env = simpy.Environment()
    rng = np.random.default_rng(scenario.seed)

    resources = ChainResources(
        role_1=simpy.PriorityResource(env, capacity=scenario.role_1.capacity),
        role_2=simpy.PriorityResource(env, capacity=scenario.role_2.capacity),
        role_3=simpy.Resource(env, capacity=scenario.role_3.capacity),
        medevac_ground=simpy.Resource(env, capacity=max(1, scenario.medevac.ground_ambulances)),
        medevac_rotary=simpy.Resource(env, capacity=max(1, scenario.medevac.rotary_wing)),
    )

    metrics = ChainMetrics()

    env.process(arrival_generator(env, resources, scenario, rng, metrics))
    env.process(queue_monitor(env, resources, metrics))

    run_length_mins = scenario.run_length_hours * 60
    env.run(until=run_length_mins)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="MEDEVAC Demo", page_icon="", layout="wide")

st.title("FAER-M: MEDEVAC Chain Simulation")
st.caption("Military medical evacuation chain modeling | POI → Role 1 → Role 2 → Role 3 → Role 4")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Configuration")

    # Preset selector
    preset_name = st.selectbox(
        "Load Preset",
        options=["Custom"] + list(PRESETS.keys()),
        index=0,
    )

    if preset_name != "Custom":
        preset = PRESETS[preset_name]
        st.info(f"Loaded: {preset_name}")
    else:
        preset = None

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation Settings
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Simulation")
    run_hours = st.select_slider(
        "Duration (hours)",
        options=[12, 24, 36, 48, 72],
        value=preset.run_length_hours if preset else 24,
    )
    casualty_rate = st.slider(
        "Casualty Rate (per hour)",
        min_value=1.0,
        max_value=30.0,
        value=preset.casualty_rate_per_hour if preset else 5.0,
        step=0.5,
    )
    seed = st.number_input("Random Seed", value=42, min_value=1)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # Triage Mix
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Triage Mix")
    t1_pct = st.slider("T1 Immediate %", 5, 40, int((preset.t1_proportion if preset else 0.15) * 100))
    t2_pct = st.slider("T2 Delayed %", 10, 50, int((preset.t2_proportion if preset else 0.30) * 100))
    t3_pct = st.slider("T3 Minimal %", 20, 70, int((preset.t3_proportion if preset else 0.50) * 100))

    remaining = 100 - t1_pct - t2_pct - t3_pct
    if remaining < 0:
        st.error(f"Triage mix exceeds 100% by {-remaining}%")
        t4_pct = 0
    else:
        t4_pct = remaining
        st.caption(f"T4 Expectant: {t4_pct}%")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # Role Capacities
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Role Capacities")

    col1, col2 = st.columns(2)
    with col1:
        r1_cap = st.number_input(
            "Role 1",
            min_value=1,
            max_value=20,
            value=preset.role_1.capacity if preset else 4,
        )
        r2_cap = st.number_input(
            "Role 2",
            min_value=1,
            max_value=20,
            value=preset.role_2.capacity if preset else 6,
        )
    with col2:
        r3_cap = st.number_input(
            "Role 3",
            min_value=5,
            max_value=100,
            value=preset.role_3.capacity if preset else 20,
        )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # MEDEVAC Assets
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("MEDEVAC Assets")
    ground_amb = st.slider(
        "Ground Ambulances",
        min_value=0,
        max_value=10,
        value=preset.medevac.ground_ambulances if preset else 4,
    )
    rotary = st.slider(
        "Rotary Wing (Helo)",
        min_value=0,
        max_value=6,
        value=preset.medevac.rotary_wing if preset else 2,
    )

    if ground_amb == 0 and rotary == 0:
        st.error("No MEDEVAC assets - casualties will be stranded!")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Build scenario and run
# ─────────────────────────────────────────────────────────────────────────────
valid_config = (t1_pct + t2_pct + t3_pct + t4_pct == 100) and (ground_amb > 0 or rotary > 0)

# Build scenario from UI inputs
try:
    scenario = ChainScenario(
        run_length_hours=run_hours,
        casualty_rate_per_hour=casualty_rate,
        role_1=RoleConfig(capacity=r1_cap, service_mean_mins=15.0),
        role_2=RoleConfig(capacity=r2_cap, service_mean_mins=90.0),
        role_3=RoleConfig(capacity=r3_cap, service_mean_mins=180.0),
        medevac=MEDEVACConfig(ground_ambulances=max(1, ground_amb), rotary_wing=max(0, rotary)),
        t1_proportion=t1_pct / 100,
        t2_proportion=t2_pct / 100,
        t3_proportion=t3_pct / 100,
        t4_proportion=t4_pct / 100,
        seed=seed,
    )
    config_valid = True
except ValueError as e:
    st.error(f"Configuration error: {e}")
    config_valid = False

# ─────────────────────────────────────────────────────────────────────────────
# Chain Schematic
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Medical Chain Schematic")

schematic_cols = st.columns(5)
schematic_data = [
    ("POI", f"{casualty_rate:.1f}/hr", "Point of Injury"),
    ("Role 1", f"{r1_cap} spaces", "Battalion Aid"),
    ("Role 2", f"{r2_cap} tables", "Forward Surg"),
    ("Role 3", f"{r3_cap} beds", "Combat Hosp"),
    ("Role 4", "Exit", "Strategic Evac"),
]

for col, (name, capacity, desc) in zip(schematic_cols, schematic_data):
    with col:
        st.metric(name, capacity)
        st.caption(desc)

# MEDEVAC assets display
st.caption(f"MEDEVAC: {ground_amb} ground ambulances, {rotary} rotary wing")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Run Button
# ─────────────────────────────────────────────────────────────────────────────
run_col, info_col = st.columns([1, 3])

with run_col:
    run_button = st.button(
        "Run Simulation",
        type="primary",
        disabled=not config_valid,
        use_container_width=True,
    )

with info_col:
    expected = int(casualty_rate * run_hours)
    st.caption(f"Expected casualties: ~{expected} | Duration: {run_hours}h")

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if run_button and config_valid:
    with st.spinner(f"Simulating {run_hours}h operation..."):
        start_time = time.time()
        metrics = run_simulation(scenario)
        elapsed = time.time() - start_time

    st.success(f"Simulation complete in {elapsed:.2f}s")
    st.session_state['medevac_metrics'] = metrics
    st.session_state['medevac_scenario'] = scenario

if 'medevac_metrics' in st.session_state:
    metrics = st.session_state['medevac_metrics']
    scenario = st.session_state['medevac_scenario']

    # ─────────────────────────────────────────────────────────────────────────
    # KPI Cards
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Key Metrics")

    kpi_cols = st.columns(6)

    throughput = metrics.throughput
    outcomes = metrics.outcomes

    with kpi_cols[0]:
        st.metric("T1 Evacuated", throughput[TriageCategory.T1_IMMEDIATE])
    with kpi_cols[1]:
        st.metric("T2 Evacuated", throughput[TriageCategory.T2_DELAYED])
    with kpi_cols[2]:
        st.metric("T3 RTD", throughput[TriageCategory.T3_MINIMAL])
    with kpi_cols[3]:
        st.metric("T4 Expectant", throughput[TriageCategory.T4_EXPECTANT])
    with kpi_cols[4]:
        survival = metrics.t1_survival_rate * 100
        st.metric("T1 Survival", f"{survival:.0f}%")
    with kpi_cols[5]:
        time_to_surg = metrics.mean_time_to_surgery
        st.metric("Mean Time to Surgery", f"{time_to_surg:.0f} min")

    # Bottleneck indicator
    bottleneck = metrics.bottleneck_role()
    st.info(f"Bottleneck: **{ROLE_LABELS[bottleneck]}** (longest queue times)")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("Queue Times by Role")
        queue_data = []
        for role, times in metrics.queue_times_by_role.items():
            for t in times:
                queue_data.append({'Role': ROLE_LABELS[role], 'Queue Time (min)': t})

        if queue_data:
            df_queue = pd.DataFrame(queue_data)
            fig_queue = px.box(
                df_queue,
                x='Role',
                y='Queue Time (min)',
                color='Role',
                title="Wait Time Distribution by Echelon",
            )
            fig_queue.update_layout(showlegend=False)
            st.plotly_chart(fig_queue, use_container_width=True)
        else:
            st.warning("No queue data recorded")

    with viz_col2:
        st.subheader("Outcomes Distribution")
        outcome_data = {
            'Outcome': ['RTD', 'Evacuated', 'Died', 'In System'],
            'Count': [
                outcomes[Outcome.RTD],
                outcomes[Outcome.EVACUATED],
                outcomes[Outcome.DIED],
                outcomes[Outcome.IN_SYSTEM],
            ]
        }
        df_outcomes = pd.DataFrame(outcome_data)
        fig_outcomes = px.pie(
            df_outcomes,
            names='Outcome',
            values='Count',
            title="Patient Outcomes",
            color='Outcome',
            color_discrete_map={
                'RTD': '#2ecc71',
                'Evacuated': '#3498db',
                'Died': '#e74c3c',
                'In System': '#95a5a6',
            }
        )
        st.plotly_chart(fig_outcomes, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Queue Over Time
    # ─────────────────────────────────────────────────────────────────────────
    if metrics.queue_snapshots:
        st.subheader("Queue Length Over Time")
        df_queues = pd.DataFrame(metrics.queue_snapshots)
        df_queues['time_hours'] = df_queues['time'] / 60

        fig_timeline = go.Figure()
        for role in [Role.ROLE_1, Role.ROLE_2, Role.ROLE_3]:
            fig_timeline.add_trace(go.Scatter(
                x=df_queues['time_hours'],
                y=df_queues[role.name],
                mode='lines',
                name=ROLE_LABELS[role],
            ))

        fig_timeline.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Queue Length",
            title="Queue Dynamics",
            hovermode='x unified',
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Raw Data Expander
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("View Raw Casualty Data"):
        casualty_records = []
        for c in metrics.casualties[:100]:  # Limit to first 100
            casualty_records.append({
                'ID': c.id,
                'Triage': TRIAGE_LABELS[c.triage],
                'Arrival': f"{c.arrival_time:.1f}",
                'Outcome': c.outcome.name,
                'Time in System': f"{(c.departure_time - c.arrival_time):.1f}" if c.departure_time else "N/A",
            })
        st.dataframe(pd.DataFrame(casualty_records), use_container_width=True)
