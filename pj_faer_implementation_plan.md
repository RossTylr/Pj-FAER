# Pj FAER: Phased Implementation Plan
## From Working Mechanics to Whole-System Hospital Flow Simulation

**Version:** 1.0  
**Date:** January 2026  
**Lead:** Ross (NHS South West Health Data Science)  
**Architecture Philosophy:** Get it working → Get it right → Get it complete

---

## Executive Summary

This plan consolidates:
1. **Tom Monks' repository analysis** (sim-tools, STARS examples, Streamlit patterns)
2. **The FAER PRD** (whole-system hospital flow vision)
3. **The WYRD extraction plan** (practical implementation patterns)

**Core principle:** Build a working simulation engine first (Phases 0–2), then layer on complexity (Phases 3–5). Each phase produces a shippable increment.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pj FAER Stack                               │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 4: Streamlit UI                                              │
│  ├── Scenario Builder                                               │
│  ├── Experiment Runner                                              │
│  ├── Results Dashboard                                              │
│  └── Comparison & Insights                                          │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3: Experimentation Engine                                    │
│  ├── Replication Controller                                         │
│  ├── Precision Guidance (CI-based stopping)                         │
│  ├── Scenario Comparison                                            │
│  └── Sensitivity Sweeps                                             │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2: Results & Metrics                                         │
│  ├── Event Logger                                                   │
│  ├── Time-Weighted Statistics                                       │
│  ├── Queue Metrics                                                  │
│  └── Bottleneck Attribution                                         │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1: SimPy Model Core                                          │
│  ├── Patient Process                                                │
│  ├── Resource Pool (beds, staff, equipment)                         │
│  ├── Routing Logic                                                  │
│  └── Holding States                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 0: Foundation                                                │
│  ├── Scenario Configuration                                         │
│  ├── Arrival Generators (NSPP thinning)                             │
│  ├── Distribution Wrappers                                          │
│  └── Seed Management                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Scaffolding & Foundation (Week 1)
### Objective: Project skeleton with working dependencies

**Deliverable:** Empty but runnable project structure

### 0.1 Project Structure

```
pj_faer/
├── pyproject.toml              # Dependencies, build config
├── README.md
├── LICENSE                     # MIT
├── ATTRIBUTION.md              # Credits to Monks et al.
│
├── src/
│   └── faer/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── scenario.py     # Scenario dataclass
│       │   ├── arrivals.py     # NSPP thinning
│       │   └── distributions.py # Service time wrappers
│       ├── model/
│       │   ├── __init__.py
│       │   ├── patient.py      # Patient entity
│       │   ├── resources.py    # Resource definitions
│       │   └── processes.py    # SimPy process logic
│       ├── results/
│       │   ├── __init__.py
│       │   ├── collector.py    # Event logging
│       │   └── metrics.py      # KPI computation
│       └── experiment/
│           ├── __init__.py
│           ├── runner.py       # Single/batch runs
│           └── analysis.py     # CI, precision guidance
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Scenario.py
│       ├── 2_Run.py
│       ├── 3_Results.py
│       └── 4_Compare.py
│
├── data/
│   └── arrival_profiles/
│       └── default_24h.csv
│
├── tests/
│   ├── conftest.py             # Fixtures
│   ├── test_arrivals.py
│   ├── test_scenario.py
│   ├── test_model.py
│   └── test_experiment.py
│
└── docs/
    ├── architecture.md
    └── model_specification.md
```

### 0.2 Dependencies (pyproject.toml)

```toml
[project]
name = "pj-faer"
version = "0.1.0"
dependencies = [
    "simpy>=4.1.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "sim-tools>=0.4.0",      # Monks' library - use directly!
    "streamlit>=1.30.0",
    "plotly>=5.18.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

### 0.3 Tasks

| Task | Description | Test |
|------|-------------|------|
| Create folder structure | As above | `tree` shows correct layout |
| Initialize pyproject.toml | With all deps | `pip install -e .` succeeds |
| Create empty modules | All `__init__.py` files | `import faer` works |
| Verify sim-tools | Import and test | `from sim_tools.distributions import Exponential` works |
| Setup pytest | conftest.py with fixtures | `pytest` runs (0 tests) |

---

## Phase 1: Stationary Mechanics (Week 2)
### Objective: Simplest possible working simulation

**Deliverable:** One queue, one server, constant arrivals, reproducible results

### 1.1 Minimal Scenario

```python
# src/faer/core/scenario.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Scenario:
    """Minimal scenario for Phase 1."""
    
    # Horizon
    run_length: float = 480.0  # 8 hours in minutes
    warm_up: float = 0.0       # No warm-up for short horizon
    
    # Arrivals (constant rate for Phase 1)
    arrival_rate: float = 4.0  # patients per hour
    
    # Resources
    n_resus_bays: int = 2
    
    # Service (minutes)
    resus_mean: float = 45.0
    resus_cv: float = 0.5      # Coefficient of variation
    
    # Seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Initialize RNG streams."""
        self.rng_arrivals = np.random.default_rng(self.random_seed)
        self.rng_service = np.random.default_rng(self.random_seed + 1)
```

### 1.2 Minimal Model

```python
# src/faer/model/processes.py
import simpy
from faer.core.scenario import Scenario

def patient_process(env: simpy.Environment, 
                    patient_id: int,
                    resus_bays: simpy.Resource,
                    scenario: Scenario,
                    results: dict):
    """Single patient journey: arrive → queue → service → depart."""
    
    arrival_time = env.now
    
    # Request Resus bay
    with resus_bays.request() as req:
        yield req
        
        # Record queue time
        queue_time = env.now - arrival_time
        results['queue_times'].append(queue_time)
        
        # Service (lognormal for realistic LoS)
        mean = scenario.resus_mean
        cv = scenario.resus_cv
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean) - sigma**2 / 2
        service_time = scenario.rng_service.lognormal(mu, sigma)
        
        yield env.timeout(service_time)
    
    # Departed
    results['departures'] += 1
    results['system_times'].append(env.now - arrival_time)


def arrival_generator(env: simpy.Environment,
                      resus_bays: simpy.Resource,
                      scenario: Scenario,
                      results: dict):
    """Generate arrivals at constant rate (Phase 1)."""
    
    patient_id = 0
    mean_iat = 60.0 / scenario.arrival_rate  # Convert rate to IAT
    
    while True:
        # Exponential inter-arrival time
        iat = scenario.rng_arrivals.exponential(mean_iat)
        yield env.timeout(iat)
        
        patient_id += 1
        results['arrivals'] += 1
        env.process(patient_process(env, patient_id, resus_bays, scenario, results))


def run_simulation(scenario: Scenario) -> dict:
    """Execute single simulation run."""
    
    # Initialize
    env = simpy.Environment()
    resus_bays = simpy.Resource(env, capacity=scenario.n_resus_bays)
    
    results = {
        'arrivals': 0,
        'departures': 0,
        'queue_times': [],
        'system_times': [],
    }
    
    # Start arrival process
    env.process(arrival_generator(env, resus_bays, scenario, results))
    
    # Run
    env.run(until=scenario.run_length)
    
    return results
```

### 1.3 Minimal Test

```python
# tests/test_model.py
from faer.core.scenario import Scenario
from faer.model.processes import run_simulation

def test_deterministic_scenario():
    """Same seed → same results."""
    scenario = Scenario(random_seed=42, run_length=120)
    
    results1 = run_simulation(scenario)
    
    # Reset RNGs by recreating scenario
    scenario2 = Scenario(random_seed=42, run_length=120)
    results2 = run_simulation(scenario2)
    
    assert results1['arrivals'] == results2['arrivals']
    assert results1['queue_times'] == results2['queue_times']


def test_basic_flow():
    """Patients arrive, queue, get served, depart."""
    scenario = Scenario(
        arrival_rate=2.0,  # Low rate
        n_resus_bays=5,    # High capacity → no queue
        run_length=60,
    )
    
    results = run_simulation(scenario)
    
    assert results['arrivals'] > 0
    assert results['departures'] > 0
    # With excess capacity, most queue times should be ~0
    assert sum(results['queue_times']) / len(results['queue_times']) < 1.0
```

### 1.4 Phase 1 Success Criteria

- [ ] `run_simulation(scenario)` executes without error
- [ ] Same seed produces identical results
- [ ] Different seeds produce different results
- [ ] Queue times increase when arrival_rate > service_rate × n_bays
- [ ] All tests pass

---

## Phase 2: Non-Stationary Arrivals + Metrics (Weeks 3–4)
### Objective: Time-varying arrivals, proper KPIs, replications

**Deliverable:** Realistic A&E arrivals, P(delay), utilisation, CI-based replications

### 2.1 NSPP Thinning Arrivals

```python
# src/faer/core/arrivals.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class ArrivalProfile:
    """Piecewise constant arrival rate schedule."""
    
    # List of (end_time, rate) tuples
    # e.g., [(60, 3.0), (120, 5.0), ...] means rate=3 from 0-60, rate=5 from 60-120
    schedule: List[Tuple[float, float]]
    
    def __post_init__(self):
        self.max_rate = max(rate for _, rate in self.schedule)
        self._validate()
    
    def _validate(self):
        """Ensure schedule is sorted and non-overlapping."""
        times = [t for t, _ in self.schedule]
        assert times == sorted(times), "Schedule must be sorted by time"
    
    def get_rate(self, t: float) -> float:
        """Get arrival rate at time t."""
        for end_time, rate in self.schedule:
            if t < end_time:
                return rate
        # Beyond schedule: use last rate
        return self.schedule[-1][1]


class NSPPThinning:
    """Non-stationary Poisson Process via thinning."""
    
    def __init__(self, profile: ArrivalProfile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng
        self.max_rate = profile.max_rate
    
    def sample_iat(self, current_time: float) -> float:
        """Sample inter-arrival time from current_time."""
        
        t = current_time
        
        while True:
            # Candidate from homogeneous Poisson at max rate
            candidate_iat = self.rng.exponential(60.0 / self.max_rate)
            t += candidate_iat
            
            # Accept with probability λ(t) / λ_max
            current_rate = self.profile.get_rate(t)
            if self.rng.random() <= current_rate / self.max_rate:
                return t - current_time
            
            # Rejected: continue from t


def load_default_profile() -> ArrivalProfile:
    """Default 24h A&E arrival profile (patients/hour)."""
    # Typical ED pattern: low overnight, peak mid-morning and evening
    return ArrivalProfile([
        (60, 2.0),    # 00:00-01:00
        (120, 1.5),   # 01:00-02:00
        (180, 1.0),   # 02:00-03:00
        (240, 1.0),   # 03:00-04:00
        (300, 1.5),   # 04:00-05:00
        (360, 2.0),   # 05:00-06:00
        (420, 3.0),   # 06:00-07:00
        (480, 4.5),   # 07:00-08:00
        (540, 6.0),   # 08:00-09:00
        (600, 7.0),   # 09:00-10:00
        (660, 7.5),   # 10:00-11:00
        (720, 7.0),   # 11:00-12:00
        (780, 6.5),   # 12:00-13:00
        (840, 6.0),   # 13:00-14:00
        (900, 5.5),   # 14:00-15:00
        (960, 5.5),   # 15:00-16:00
        (1020, 6.0),  # 16:00-17:00
        (1080, 6.5),  # 17:00-18:00
        (1140, 7.0),  # 18:00-19:00
        (1200, 6.5),  # 19:00-20:00
        (1260, 5.5),  # 20:00-21:00
        (1320, 4.5),  # 21:00-22:00
        (1380, 3.5),  # 22:00-23:00
        (1440, 2.5),  # 23:00-24:00
    ])
```

### 2.2 Results Collector with Time-Weighted Metrics

```python
# src/faer/results/collector.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class ResultsCollector:
    """Collect and compute simulation metrics."""
    
    # Event logs
    arrivals: int = 0
    departures: int = 0
    queue_times: List[float] = field(default_factory=list)
    system_times: List[float] = field(default_factory=list)
    
    # Resource state log: [(time, n_busy), ...]
    resource_log: List[Tuple[float, int]] = field(default_factory=list)
    
    def record_arrival(self):
        self.arrivals += 1
    
    def record_queue_time(self, wait: float):
        self.queue_times.append(wait)
    
    def record_system_time(self, time: float):
        self.system_times.append(time)
        self.departures += 1
    
    def record_resource_state(self, time: float, n_busy: int):
        self.resource_log.append((time, n_busy))
    
    def compute_metrics(self, run_length: float, capacity: int) -> Dict:
        """Compute all KPIs."""
        
        queue_times = np.array(self.queue_times) if self.queue_times else np.array([0])
        system_times = np.array(self.system_times) if self.system_times else np.array([0])
        
        # P(delay) = P(queue_time > 0)
        p_delay = np.mean(queue_times > 0) if len(queue_times) > 0 else 0
        
        # Queue time quantiles
        q50 = np.percentile(queue_times, 50) if len(queue_times) > 0 else 0
        q95 = np.percentile(queue_times, 95) if len(queue_times) > 0 else 0
        
        # Time-weighted utilisation
        utilisation = self._compute_utilisation(run_length, capacity)
        
        return {
            'arrivals': self.arrivals,
            'departures': self.departures,
            'p_delay': p_delay,
            'mean_queue_time': np.mean(queue_times),
            'median_queue_time': q50,
            'p95_queue_time': q95,
            'mean_system_time': np.mean(system_times),
            'utilisation': utilisation,
            'throughput_per_hour': self.departures / (run_length / 60),
        }
    
    def _compute_utilisation(self, run_length: float, capacity: int) -> float:
        """Time-weighted resource utilisation."""
        
        if not self.resource_log:
            return 0.0
        
        # Sort by time
        log = sorted(self.resource_log)
        
        total_busy_time = 0.0
        for i in range(len(log) - 1):
            t_start, n_busy = log[i]
            t_end = log[i + 1][0]
            total_busy_time += n_busy * (t_end - t_start)
        
        # Final segment to run_length
        if log:
            t_last, n_last = log[-1]
            total_busy_time += n_last * (run_length - t_last)
        
        return total_busy_time / (capacity * run_length)
```

### 2.3 Replication Runner with CI Guidance

```python
# src/faer/experiment/runner.py
from typing import List, Dict, Optional, Callable
import numpy as np
from scipy import stats
from faer.core.scenario import Scenario
from faer.model.processes import run_simulation

def multiple_replications(
    scenario: Scenario,
    n_reps: int = 30,
    metric_names: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """Run multiple replications, collect specified metrics."""
    
    if metric_names is None:
        metric_names = ['p_delay', 'mean_queue_time', 'utilisation']
    
    results = {name: [] for name in metric_names}
    
    for rep in range(n_reps):
        # Create scenario with different seed per rep
        rep_scenario = Scenario(
            **{k: v for k, v in scenario.__dict__.items() 
               if not k.startswith('rng_')},
        )
        rep_scenario.random_seed = scenario.random_seed + rep
        rep_scenario.__post_init__()  # Re-init RNGs
        
        run_results = run_simulation(rep_scenario)
        
        for name in metric_names:
            results[name].append(run_results[name])
    
    return results


def compute_ci(values: List[float], confidence: float = 0.95) -> Dict:
    """Compute confidence interval for a metric."""
    
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    half_width = t_crit * se
    
    return {
        'mean': mean,
        'std': np.std(values, ddof=1),
        'se': se,
        'ci_lower': mean - half_width,
        'ci_upper': mean + half_width,
        'ci_half_width': half_width,
        'n': n,
    }


def run_until_precision(
    scenario: Scenario,
    target_metric: str = 'p_delay',
    target_half_width: float = 0.02,
    max_reps: int = 200,
    min_reps: int = 10,
    batch_size: int = 5,
    confidence: float = 0.95,
) -> Dict:
    """Run replications until target precision achieved."""
    
    values = []
    
    for rep in range(max_reps):
        rep_scenario = Scenario(
            **{k: v for k, v in scenario.__dict__.items() 
               if not k.startswith('rng_')},
        )
        rep_scenario.random_seed = scenario.random_seed + rep
        rep_scenario.__post_init__()
        
        run_results = run_simulation(rep_scenario)
        values.append(run_results[target_metric])
        
        # Check precision after min_reps, then every batch_size
        if rep >= min_reps and (rep + 1) % batch_size == 0:
            ci = compute_ci(values, confidence)
            
            if ci['ci_half_width'] <= target_half_width:
                return {
                    'converged': True,
                    'n_reps': rep + 1,
                    'values': values,
                    'ci': ci,
                }
    
    # Did not converge
    return {
        'converged': False,
        'n_reps': max_reps,
        'values': values,
        'ci': compute_ci(values, confidence),
    }
```

### 2.4 Phase 2 Success Criteria

- [ ] NSPP arrivals produce correct hourly counts (±10% of expected)
- [ ] Thinning efficiency > 50% for typical ED profiles
- [ ] P(delay) and utilisation computed correctly
- [ ] CI half-width decreases with √n
- [ ] `run_until_precision` stops when precision achieved
- [ ] All metrics match hand calculations on deterministic scenario

---

## Phase 3: Streamlit MVP (Week 5)
### Objective: Interactive UI for running scenarios

**Deliverable:** Working web app with inputs, run button, results display

### 3.1 Home Page

```python
# app/Home.py
import streamlit as st

st.set_page_config(
    page_title="Pj FAER",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Pj FAER: Hospital Flow Simulation")

st.markdown("""
## What is FAER?

FAER is a **discrete-event simulation** platform for understanding how patients 
flow through hospital systems over **1–36 hour horizons**.

### What FAER reveals:
- Where congestion **actually** originates
- Which capacity investments have **system-wide** effects
- What fails first under surge conditions

### Current capabilities (MVP):
- A&E → Resus queue/service → disposition
- Non-stationary arrival patterns (time-of-day effects)
- Time-weighted utilisation metrics
- Replication-based confidence intervals

---

👈 **Use the sidebar** to navigate to Scenario configuration and simulation runs.
""")

st.info("""
**Attribution:** This simulation uses patterns from Tom Monks' 
[sim-tools](https://github.com/TomMonks/sim-tools) and the STARS project.
""")
```

### 3.2 Scenario Configuration Page

```python
# app/pages/1_Scenario.py
import streamlit as st
import pandas as pd
from faer.core.scenario import Scenario
from faer.core.arrivals import load_default_profile

st.title("📊 Scenario Configuration")

# Initialize session state
if 'scenario' not in st.session_state:
    st.session_state.scenario = Scenario()

col1, col2 = st.columns(2)

with col1:
    st.header("Simulation Horizon")
    run_length = st.slider(
        "Run length (hours)", 
        min_value=1, 
        max_value=36, 
        value=12,
        help="Simulation duration"
    )
    
    st.header("Resources")
    n_resus = st.slider(
        "Resus bays",
        min_value=1,
        max_value=10,
        value=2,
    )

with col2:
    st.header("Service Times")
    resus_mean = st.number_input(
        "Mean Resus LoS (minutes)",
        min_value=10,
        max_value=300,
        value=45,
    )
    resus_cv = st.slider(
        "Resus LoS variability (CV)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        help="Coefficient of variation"
    )

st.header("Arrival Profile")
use_custom = st.checkbox("Use custom arrival profile")

if use_custom:
    st.warning("Custom profile upload not yet implemented. Using default.")

# Show default profile
profile = load_default_profile()
profile_df = pd.DataFrame([
    {'Hour': i, 'Rate (patients/hr)': rate}
    for i, (_, rate) in enumerate(profile.schedule)
])
st.line_chart(profile_df.set_index('Hour'))

st.header("Experiment Settings")
col3, col4 = st.columns(2)
with col3:
    n_reps = st.slider("Number of replications", 5, 100, 30)
with col4:
    random_seed = st.number_input("Random seed", 0, 999999, 42)

# Save scenario
if st.button("💾 Save Scenario"):
    st.session_state.scenario = Scenario(
        run_length=run_length * 60,  # Convert to minutes
        n_resus_bays=n_resus,
        resus_mean=resus_mean,
        resus_cv=resus_cv,
        random_seed=random_seed,
    )
    st.session_state.n_reps = n_reps
    st.success("Scenario saved!")
```

### 3.3 Run Page

```python
# app/pages/2_Run.py
import streamlit as st
import time
from faer.experiment.runner import multiple_replications, compute_ci

st.title("▶️ Run Simulation")

if 'scenario' not in st.session_state:
    st.warning("Please configure a scenario first.")
    st.stop()

scenario = st.session_state.scenario
n_reps = st.session_state.get('n_reps', 30)

st.write(f"**Scenario:** {scenario.n_resus_bays} Resus bays, "
         f"{scenario.run_length/60:.0f}h horizon, "
         f"{n_reps} replications")

if st.button("🚀 Run Experiment", type="primary"):
    progress_bar = st.progress(0)
    status = st.empty()
    
    start_time = time.time()
    
    # Run replications with progress
    results = {'p_delay': [], 'mean_queue_time': [], 'utilisation': []}
    
    for rep in range(n_reps):
        # Would call run_simulation here
        # For now, simulate with placeholder
        status.text(f"Running replication {rep + 1}/{n_reps}...")
        progress_bar.progress((rep + 1) / n_reps)
    
    elapsed = time.time() - start_time
    
    st.success(f"✅ Completed {n_reps} replications in {elapsed:.1f}s")
    
    # Store results
    st.session_state.results = results
    st.session_state.run_complete = True
```

### 3.4 Results Page

```python
# app/pages/3_Results.py
import streamlit as st
import pandas as pd
import plotly.express as px
from faer.experiment.runner import compute_ci

st.title("📈 Results")

if not st.session_state.get('run_complete'):
    st.warning("Please run a simulation first.")
    st.stop()

results = st.session_state.results

# KPI cards
st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    ci = compute_ci(results['p_delay'])
    st.metric(
        "P(Delay)",
        f"{ci['mean']:.1%}",
        help="Probability a patient waits for Resus"
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

with col2:
    ci = compute_ci(results['mean_queue_time'])
    st.metric(
        "Mean Queue Time",
        f"{ci['mean']:.1f} min",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

with col3:
    ci = compute_ci(results['utilisation'])
    st.metric(
        "Resus Utilisation",
        f"{ci['mean']:.1%}",
    )
    st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")

# Distribution plots
st.header("Metric Distributions")

fig = px.histogram(
    results['p_delay'],
    nbins=20,
    labels={'value': 'P(Delay)', 'count': 'Frequency'},
    title="Distribution of P(Delay) across replications"
)
st.plotly_chart(fig, use_container_width=True)

# Export
st.header("Export")
if st.button("📥 Download Results CSV"):
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download",
        csv,
        "faer_results.csv",
        "text/csv"
    )
```

### 3.5 Phase 3 Success Criteria

- [ ] App runs locally with `streamlit run app/Home.py`
- [ ] Scenario configuration persists across pages
- [ ] Run button triggers simulation with progress feedback
- [ ] Results display with CIs
- [ ] CSV export works
- [ ] No crashes on edge-case inputs

---

## Phase 4: System Realism (Weeks 6–8)
### Objective: Extend to full A&E pathway with holding states

**Additions:**
- Multiple patient acuities (Resus/Majors/Minors)
- Triage process
- Disposition routing (discharge/admit/transfer)
- Holding states (boarding, awaiting bed)
- Priority queuing

### 4.1 Extended Patient Entity

```python
# src/faer/model/patient.py
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class Acuity(Enum):
    RESUS = auto()      # Life-threatening
    MAJORS = auto()     # Serious
    MINORS = auto()     # Ambulatory

class Disposition(Enum):
    DISCHARGE = auto()
    ADMIT_WARD = auto()
    ADMIT_ICU = auto()
    TRANSFER = auto()

@dataclass
class Patient:
    id: int
    arrival_time: float
    acuity: Acuity
    
    # Timestamps (filled during simulation)
    triage_start: Optional[float] = None
    triage_end: Optional[float] = None
    treatment_start: Optional[float] = None
    treatment_end: Optional[float] = None
    disposition_time: Optional[float] = None
    
    # Outcome
    disposition: Optional[Disposition] = None
    
    @property
    def queue_time(self) -> float:
        if self.triage_end and self.treatment_start:
            return self.treatment_start - self.triage_end
        return 0.0
    
    @property
    def system_time(self) -> float:
        if self.disposition_time:
            return self.disposition_time - self.arrival_time
        return 0.0
```

### 4.2 Extended Scenario

```python
# Extended scenario for Phase 4
@dataclass
class ScenarioV2(Scenario):
    # Acuity mix
    p_resus: float = 0.05
    p_majors: float = 0.55
    p_minors: float = 0.40
    
    # Resources by type
    n_triage_staff: int = 2
    n_majors_bays: int = 10
    n_minors_bays: int = 6
    
    # Service times by acuity (mean, cv)
    triage_mean: float = 5.0
    majors_mean: float = 90.0
    minors_mean: float = 30.0
    
    # Disposition probabilities (by acuity)
    resus_p_admit: float = 0.85
    majors_p_admit: float = 0.35
    minors_p_admit: float = 0.05
    
    # Holding state durations
    boarding_mean: float = 120.0  # Minutes awaiting bed
```

### 4.3 Phase 4 Test Cases

| Test | Description |
|------|-------------|
| Acuity routing | Resus patients go to Resus bays |
| Priority queue | Resus patients pre-empt Majors in shared resources |
| Disposition split | Admit rate matches configured probability |
| Boarding delay | Patients awaiting admission experience holding time |
| Throughput balance | Arrivals ≈ Dispositions over long run |

---

## Phase 5: Scenario Science (Weeks 9–10)
### Objective: Sensitivity analysis, comparison, bottleneck attribution

**Additions:**
- Side-by-side scenario comparison
- Sensitivity sweeps (vary one parameter)
- Dominant bottleneck identification
- "What breaks first?" analysis

### 5.1 Sensitivity Engine

```python
# src/faer/experiment/sensitivity.py
from typing import List, Dict, Tuple
import numpy as np
from faer.core.scenario import Scenario
from faer.experiment.runner import multiple_replications, compute_ci

def sensitivity_sweep(
    base_scenario: Scenario,
    param_name: str,
    param_values: List[float],
    target_metric: str = 'p_delay',
    n_reps: int = 20,
) -> List[Dict]:
    """Sweep a parameter and collect metric response."""
    
    results = []
    
    for value in param_values:
        # Clone scenario with modified parameter
        scenario_dict = {k: v for k, v in base_scenario.__dict__.items()
                        if not k.startswith('rng_')}
        scenario_dict[param_name] = value
        scenario = Scenario(**scenario_dict)
        
        # Run replications
        rep_results = multiple_replications(scenario, n_reps, [target_metric])
        ci = compute_ci(rep_results[target_metric])
        
        results.append({
            'param_value': value,
            'mean': ci['mean'],
            'ci_lower': ci['ci_lower'],
            'ci_upper': ci['ci_upper'],
        })
    
    return results


def find_breaking_point(
    base_scenario: Scenario,
    param_name: str,
    threshold_metric: str = 'p_delay',
    threshold_value: float = 0.5,
    search_range: Tuple[float, float] = (1.0, 3.0),
    n_reps: int = 20,
) -> float:
    """Binary search for parameter value where metric exceeds threshold."""
    
    low, high = search_range
    
    while high - low > 0.1:
        mid = (low + high) / 2
        
        scenario_dict = {k: v for k, v in base_scenario.__dict__.items()
                        if not k.startswith('rng_')}
        scenario_dict[param_name] = mid
        scenario = Scenario(**scenario_dict)
        
        rep_results = multiple_replications(scenario, n_reps, [threshold_metric])
        mean_value = np.mean(rep_results[threshold_metric])
        
        if mean_value < threshold_value:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2
```

### 5.2 Comparison UI

```python
# app/pages/4_Compare.py (sketch)
st.title("🔄 Scenario Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Scenario A (Baseline)")
    # Load from saved scenarios
    
with col2:
    st.subheader("Scenario B (Alternative)")
    # Configure alternative

if st.button("Compare"):
    # Run both, show side-by-side metrics
    # Highlight statistically significant differences
    pass
```

---

## Phase 6: Full FAER Platform (Weeks 11–14)
### Objective: Complete the PRD vision

**Additions:**
- Diagnostics loops (CT, path)
- Inpatient wards
- ICU/HDU
- Theatre scheduling
- Discharge pathways
- Non-clinical holding states

This phase follows the FAER PRD Section 3 scope. Each subsystem is a module that plugs into the routing logic.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SimPy learning curve | Medium | Medium | Use Monks' tutorials, start simple |
| Thinning inefficiency | Low | Low | Profile λ(t) range first |
| Streamlit performance | Medium | Medium | Cache heavy computations |
| Scope creep | High | High | Phase gates, MVP-first |
| Seed collisions | Low | High | Separate RNG streams pattern |
| Results memory | Medium | Medium | Aggregate on-the-fly |

---

## Success Metrics by Phase

| Phase | Metric | Target |
|-------|--------|--------|
| 0 | Project compiles | `pip install -e .` works |
| 1 | Basic flow | Patients arrive → queue → depart |
| 2 | Realism | P(delay) matches intuition |
| 3 | Usability | Stakeholder can run scenario |
| 4 | Fidelity | Model matches A&E structure |
| 5 | Insight | "What if?" questions answerable |
| 6 | Platform | Extendable to full hospital |

---

## Attribution

This implementation uses patterns and concepts from:

- **sim-tools** by Monks, T., Heather, A., Harper, A. (2025) — MIT License
- **STARS Project** by pythonhealthdatascience — MIT License
- **simpy-streamlit-tutorial** by health-data-science-OR — MIT License

All code is original implementation inspired by these open-source resources.

---

## Next Steps

1. **Today:** Create folder structure (Phase 0.1)
2. **This week:** Complete Phase 0, start Phase 1
3. **Next week:** NSPP arrivals working (Phase 2)
4. **Week 3:** Streamlit MVP live (Phase 3)

**The goal:** Working mechanics first. Complexity later.
