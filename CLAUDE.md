# CLAUDE.md - Pj FAER Project Instructions

## Project Overview

**Pj FAER** (Flow Analysis for Emergency Response) is a discrete-event simulation platform for hospital patient flow, built with SimPy and Streamlit. This project simulates A&E → Resus → disposition pathways over 1-36 hour horizons.

## Tech Stack

- **Python 3.11+**
- **SimPy 4.1+** - Discrete-event simulation engine
- **sim-tools** - Tom Monks' DES utilities (thinning, distributions, replications)
- **Streamlit 1.30+** - Web UI
- **NumPy, Pandas, SciPy** - Numerical/statistical computing
- **Plotly** - Visualisation
- **Pytest** - Testing

## Project Structure

```
pj_faer/
├── pyproject.toml
├── README.md
├── CLAUDE.md                   # This file
├── src/
│   └── faer/
│       ├── __init__.py
│       ├── core/               # Foundation layer
│       │   ├── scenario.py     # Scenario dataclass with parameters + seeds
│       │   ├── arrivals.py     # NSPP thinning arrival generator
│       │   └── distributions.py
│       ├── model/              # SimPy model layer
│       │   ├── patient.py      # Patient entity
│       │   ├── resources.py    # Resource definitions
│       │   └── processes.py    # SimPy process logic
│       ├── results/            # Metrics layer
│       │   ├── collector.py    # Event logging
│       │   └── metrics.py      # KPI computation
│       └── experiment/         # Experimentation layer
│           ├── runner.py       # Single/batch runs
│           └── analysis.py     # CI, precision guidance
├── app/                        # Streamlit UI
│   ├── Home.py
│   └── pages/
├── tests/
└── data/
    └── arrival_profiles/
```

## Current Phase: 0 (Scaffolding)

We are building incrementally. Current objective: create project skeleton with working dependencies.

### Phase Progression

- **Phase 0**: Scaffolding - Empty but runnable project
- **Phase 1**: Stationary Mechanics - One queue, one server, constant arrivals
- **Phase 2**: NSPP + Metrics - Time-varying arrivals, P(delay), utilisation, CIs
- **Phase 3**: Streamlit MVP - Interactive web app
- **Phase 4**: System Realism - Full A&E pathway, holding states
- **Phase 5**: Scenario Science - Sensitivity analysis, comparison
- **Phase 6**: Full FAER - Complete hospital flow platform

## Code Style & Conventions

### General

- Use **type hints** everywhere
- Use **dataclasses** for configuration and entities
- Prefer **composition over inheritance**
- Keep functions **small and focused** (< 30 lines ideal)
- Write **docstrings** for all public functions/classes

### Naming

- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `SCREAMING_SNAKE` for constants
- Prefix private methods with `_`

### SimPy Specifics

- Processes are **generator functions** that `yield` events
- Always pass `env: simpy.Environment` as first parameter
- Use `env.now` for current simulation time
- Use `env.timeout(duration)` for delays
- Use `resource.request()` as context manager

### Random Number Generation

- **Never use `np.random` module directly** - use `np.random.Generator`
- Each stochastic element gets its **own RNG stream**
- Derive child seeds from master seed: `seed + offset`
- Store RNG in Scenario, pass to processes

```python
# CORRECT
rng = np.random.default_rng(seed)
value = rng.exponential(mean)

# WRONG - not reproducible
value = np.random.exponential(mean)
```

### Results Collection

- Log events with timestamps during simulation
- Compute metrics **after** run completes
- Use time-weighted calculations for utilisation
- Store raw data, compute summaries separately

## Key Patterns to Follow

### 1. Scenario Configuration

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Scenario:
    run_length: float = 480.0      # minutes
    n_resus_bays: int = 2
    arrival_rate: float = 4.0      # per hour
    resus_mean: float = 45.0       # minutes
    random_seed: int = 42
    
    def __post_init__(self):
        # Create separate RNG streams
        self.rng_arrivals = np.random.default_rng(self.random_seed)
        self.rng_service = np.random.default_rng(self.random_seed + 1)
```

### 2. SimPy Process

```python
import simpy

def patient_process(env: simpy.Environment, 
                    patient_id: int,
                    resource: simpy.Resource,
                    scenario: Scenario,
                    results: ResultsCollector):
    """Patient journey through system."""
    arrival_time = env.now
    
    with resource.request() as req:
        yield req
        queue_time = env.now - arrival_time
        results.record_queue_time(queue_time)
        
        service_time = scenario.rng_service.exponential(scenario.resus_mean)
        yield env.timeout(service_time)
    
    results.record_departure(env.now - arrival_time)
```

### 3. NSPP Thinning (use sim-tools)

```python
from sim_tools.time_dependent import NSPPThinning

# Create thinning sampler with arrival schedule
thinning = NSPPThinning(
    data=arrival_schedule_df,  # DataFrame with 'period_end', 'arrival_rate'
    random_seed=seed
)

# In arrival generator
iat = thinning.sample()
yield env.timeout(iat)
```

### 4. Replication Runner

```python
def multiple_replications(scenario: Scenario, n_reps: int) -> pd.DataFrame:
    results = []
    for rep in range(n_reps):
        rep_scenario = clone_scenario_with_seed(scenario, scenario.random_seed + rep)
        run_result = run_simulation(rep_scenario)
        results.append(run_result)
    return pd.DataFrame(results)
```

## Testing Requirements

- **Every module needs tests**
- Use **deterministic scenarios** for verification (fixed seeds, known outcomes)
- Test **reproducibility**: same seed = same results
- Test **edge cases**: zero arrivals, full capacity, empty system

```python
def test_reproducibility():
    scenario1 = Scenario(random_seed=42)
    scenario2 = Scenario(random_seed=42)
    
    results1 = run_simulation(scenario1)
    results2 = run_simulation(scenario2)
    
    assert results1 == results2
```

## Dependencies (pyproject.toml)

```toml
[project]
name = "pj-faer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "simpy>=4.1.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "sim-tools>=0.4.0",
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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/faer"]
```

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=faer --cov-report=html

# Run Streamlit app
streamlit run app/Home.py

# Format code
black src/ tests/
ruff check src/ tests/ --fix
```

## Phase-Specific Instructions

### Phase 0: Scaffolding (Current)

**Goal**: Create folder structure and verify dependencies work.

**Tasks**:
1. Create all directories as shown in project structure
2. Create empty `__init__.py` files
3. Create `pyproject.toml` with dependencies
4. Verify `pip install -e .` works
5. Verify `from sim_tools.distributions import Exponential` works
6. Create `conftest.py` with basic fixtures
7. Verify `pytest` runs (0 tests is fine)

**Acceptance**: `import faer` works without error.

### Phase 1: Stationary Mechanics

**Goal**: Simplest working simulation - constant arrivals, one resource.

**Tasks**:
1. Implement `Scenario` dataclass in `core/scenario.py`
2. Implement basic `run_simulation()` in `model/processes.py`
3. Implement arrival generator with exponential IAT
4. Implement patient process (queue → service → depart)
5. Implement basic results dict collection
6. Write tests for reproducibility and basic flow

**Acceptance**: Same seed produces identical `queue_times` list.

### Phase 2: NSPP + Metrics

**Goal**: Realistic arrivals, proper KPIs, replication framework.

**Tasks**:
1. Implement `ArrivalProfile` and `NSPPThinning` in `core/arrivals.py`
2. Implement `ResultsCollector` with time-weighted utilisation
3. Implement `multiple_replications()` in `experiment/runner.py`
4. Implement `compute_ci()` for confidence intervals
5. Implement `run_until_precision()` for CI-based stopping
6. Test arrival counts match expected hourly rates

**Acceptance**: P(delay) CI narrows with more replications.

### Phase 3: Streamlit MVP

**Goal**: Interactive web app for running scenarios.

**Tasks**:
1. Create `app/Home.py` with project overview
2. Create `pages/1_Scenario.py` with input widgets
3. Create `pages/2_Run.py` with run button and progress
4. Create `pages/3_Results.py` with KPI display
5. Use `st.session_state` for persistence
6. Use `@st.cache_data` for expensive computations

**Acceptance**: User can configure, run, and view results in browser.

## Attribution

This project uses patterns from:
- **sim-tools** by Monks, T., Heather, A., Harper, A. (MIT License)
- **STARS Project** by pythonhealthdatascience (MIT License)

Include attribution in README and app About page.

## Common Pitfalls

1. **Don't use `np.random` directly** - breaks reproducibility
2. **Don't forget `__post_init__`** - RNGs won't be created
3. **Don't compute metrics during simulation** - do it after
4. **Don't share RNG between processes** - use separate streams
5. **Don't hardcode parameters** - put in Scenario
6. **Don't skip tests** - catch issues early

## Getting Help

- **SimPy docs**: https://simpy.readthedocs.io/
- **sim-tools docs**: https://tommonks.github.io/sim-tools/
- **Streamlit docs**: https://docs.streamlit.io/
- **Monks tutorials**: https://health-data-science-or.github.io/simpy-streamlit-tutorial/
