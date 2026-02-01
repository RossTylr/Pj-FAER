# FAER-M: Military MEDEVAC Chain Implementation

## Overview

FAER-M extends the FAER simulation framework to model military medical evacuation chains following NATO Role 1-4 echelons. This implementation uses **Pydantic CLI with Streamlit visibility** - validated config models that work both from command line and through a visual dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FAER-M Stack                             │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI (app/pages/11_MEDEVAC_Demo.py)                   │
│     ↓ builds                                                    │
│  Pydantic Config (src/faer/medevac/config.py)                  │
│     ↓ validates & passes to                                    │
│  SimPy Model (src/faer/medevac/simulation.py)                  │
│     ↓ produces                                                  │
│  Metrics (src/faer/medevac/metrics.py)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Patient Flow Model

```
POI (Point of Injury)
 │
 ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ROLE 1    │     │   ROLE 2    │     │   ROLE 3    │     │   ROLE 4    │
│             │     │             │     │             │     │             │
│  Battalion  │────▶│  Forward    │────▶│   Combat    │────▶│  Strategic  │
│  Aid Post   │     │  Surgical   │     │   Hospital  │     │  Evacuation │
│             │     │             │     │             │     │             │
│ • Triage    │     │ • Resus     │     │ • Definitive│     │ • Home base │
│ • First aid │     │ • DCS/DCR   │     │   surgery   │     │ • Tertiary  │
│ • Stabilise │     │ • Holding   │     │ • ICU/Ward  │     │   care      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   ▲                   ▲                   ▲
      │                   │                   │                   │
      └───────────────────┴───────────────────┴───────────────────┘
                    MEDEVAC (Ground/Rotary/Fixed-wing)
```

## Triage Categories

| Category | Name | Description | Pathway |
|----------|------|-------------|---------|
| T1 | Immediate | Life-threatening, salvageable | Role 1 → 2 → 3 → 4 |
| T2 | Delayed | Serious, can wait hours | Role 1 → 2 → 3 (→ 4) |
| T3 | Minimal | Walking wounded | Role 1 → RTD |
| T4 | Expectant | Non-survivable | Role 1 (comfort care) |

## Design Principles

### 1. Pydantic-First Configuration

All config validated at instantiation, not at runtime:

```python
from pydantic import BaseModel, Field, field_validator
from enum import IntEnum

class TriageCategory(IntEnum):
    T1_IMMEDIATE = 1
    T2_DELAYED = 2
    T3_MINIMAL = 3
    T4_EXPECTANT = 4

class RoleConfig(BaseModel):
    """Single medical echelon configuration."""
    capacity: int = Field(ge=1, description="Number of treatment spaces")
    service_mean_mins: float = Field(gt=0, description="Mean treatment time")
    service_cv: float = Field(ge=0, le=2, default=0.4)

class MEDEVACConfig(BaseModel):
    """Transport assets between roles."""
    ground_ambulances: int = Field(ge=0, default=4)
    rotary_wing: int = Field(ge=0, default=2)
    fixed_wing_slots_per_day: int = Field(ge=0, default=1)
    ground_transit_mins: tuple[float, float] = (30, 60)
    rotary_transit_mins: tuple[float, float] = (15, 45)

class ChainScenario(BaseModel):
    """Complete MEDEVAC chain configuration."""
    run_length_hours: int = Field(ge=1, le=168, default=24)
    casualty_rate_per_hour: float = Field(gt=0, default=5.0)

    role_1: RoleConfig = Field(default_factory=lambda: RoleConfig(
        capacity=4, service_mean_mins=15
    ))
    role_2: RoleConfig = Field(default_factory=lambda: RoleConfig(
        capacity=6, service_mean_mins=90
    ))
    role_3: RoleConfig = Field(default_factory=lambda: RoleConfig(
        capacity=20, service_mean_mins=180
    ))

    medevac: MEDEVACConfig = Field(default_factory=MEDEVACConfig)

    triage_mix: dict[TriageCategory, float] = Field(
        default={
            TriageCategory.T1_IMMEDIATE: 0.15,
            TriageCategory.T2_DELAYED: 0.30,
            TriageCategory.T3_MINIMAL: 0.50,
            TriageCategory.T4_EXPECTANT: 0.05,
        }
    )

    seed: int = 42

    @field_validator('triage_mix')
    @classmethod
    def validate_mix(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f'Triage mix must sum to 1.0, got {total}')
        return v
```

### 2. Immutable Scenario Transforms

Fluent API for scenario variants:

```python
class ChainScenario(BaseModel):
    # ... fields ...

    def with_surge(self, multiplier: float = 2.0) -> 'ChainScenario':
        """Mass casualty scenario."""
        return self.model_copy(update={
            'casualty_rate_per_hour': self.casualty_rate_per_hour * multiplier,
            'triage_mix': {
                TriageCategory.T1_IMMEDIATE: min(0.30, self.triage_mix[TriageCategory.T1_IMMEDIATE] * 1.5),
                TriageCategory.T2_DELAYED: 0.35,
                TriageCategory.T3_MINIMAL: 0.30,
                TriageCategory.T4_EXPECTANT: 0.05,
            }
        })

    def with_degraded_medevac(self, ground: int, rotary: int) -> 'ChainScenario':
        """Contested logistics scenario."""
        return self.model_copy(update={
            'medevac': self.medevac.model_copy(update={
                'ground_ambulances': ground,
                'rotary_wing': rotary,
            })
        })
```

### 3. Factory Presets

Pre-validated clinical scenarios:

```python
PRESETS = {
    'routine_ops': ChainScenario(),

    'mass_casualty': ChainScenario(
        casualty_rate_per_hour=15.0,
        triage_mix={
            TriageCategory.T1_IMMEDIATE: 0.25,
            TriageCategory.T2_DELAYED: 0.35,
            TriageCategory.T3_MINIMAL: 0.35,
            TriageCategory.T4_EXPECTANT: 0.05,
        }
    ),

    'contested_environment': ChainScenario(
        medevac=MEDEVACConfig(
            ground_ambulances=2,
            rotary_wing=0,  # No air assets
            fixed_wing_slots_per_day=0,
        )
    ),

    'prolonged_field_care': ChainScenario(
        role_1=RoleConfig(capacity=2, service_mean_mins=120),  # Extended hold
        medevac=MEDEVACConfig(ground_ambulances=1, rotary_wing=1),
    ),
}
```

### 4. SimPy Model Pattern

Follows FAER conventions - generator processes, priority queuing:

```python
def casualty_process(
    env: simpy.Environment,
    casualty: Casualty,
    resources: ChainResources,
    scenario: ChainScenario,
    results: ResultsCollector,
) -> Generator:
    """Casualty journey through medical chain."""

    # Role 1: Triage & first aid
    with resources.role_1.request(priority=casualty.triage.value) as req:
        yield req
        queue_time = env.now - casualty.arrival_time
        results.record_queue(Role.ROLE_1, casualty.triage, queue_time)

        duration = sample_lognormal(
            scenario.rng,
            scenario.role_1.service_mean_mins,
            scenario.role_1.service_cv
        )
        yield env.timeout(duration)

    # T3: Return to duty
    if casualty.triage == TriageCategory.T3_MINIMAL:
        results.record_outcome(casualty, Outcome.RTD)
        return

    # T4: Expectant care
    if casualty.triage == TriageCategory.T4_EXPECTANT:
        results.record_outcome(casualty, Outcome.DIED)
        return

    # MEDEVAC to Role 2
    yield from medevac_process(env, casualty, resources, Role.ROLE_2, scenario, results)

    # Role 2: Forward surgery
    with resources.role_2.request(priority=casualty.triage.value) as req:
        yield req
        # ... treatment ...

    # Continue through chain...
```

### 5. Metrics Collection

```python
@dataclass
class ChainMetrics:
    """Simulation output metrics."""

    # Throughput by triage category
    throughput: dict[TriageCategory, int]

    # Outcomes
    outcomes: dict[Outcome, int]  # RTD, EVACUATED, DIED

    # Queue times by role and triage
    queue_times: dict[Role, dict[TriageCategory, list[float]]]

    # Bottleneck identification
    bottleneck_role: Role
    max_queue_seen: dict[Role, int]

    # Key clinical metrics
    t1_survival_rate: float
    mean_time_to_surgery: float  # T1/T2 only
    medevac_utilisation: float

    def summary(self) -> dict:
        """Return summary dict for display."""
        return {
            'T1 Evacuated': self.throughput[TriageCategory.T1_IMMEDIATE],
            'T2 Evacuated': self.throughput[TriageCategory.T2_DELAYED],
            'T3 RTD': self.throughput[TriageCategory.T3_MINIMAL],
            'T1 Survival %': f"{self.t1_survival_rate:.1%}",
            'Bottleneck': self.bottleneck_role.name,
            'MEDEVAC Util %': f"{self.medevac_utilisation:.1%}",
        }
```

## File Structure

```
src/faer/medevac/
├── __init__.py
├── config.py          # Pydantic models (ChainScenario, RoleConfig, etc.)
├── entities.py        # Enums (TriageCategory, Role, Outcome)
├── simulation.py      # SimPy model (run_chain, casualty_process)
├── metrics.py         # ChainMetrics dataclass
├── presets.py         # PRESETS dict with named scenarios
└── cli.py             # argparse CLI runner

app/pages/
└── 11_MEDEVAC_Demo.py # Streamlit UI using same Pydantic models
```

## Usage

### CLI Mode

```bash
# Run with defaults
python -m faer.medevac.cli

# Run with config file
python -m faer.medevac.cli --config scenarios/mass_casualty.yaml

# Run preset
python -m faer.medevac.cli --preset contested_environment

# Override params
python -m faer.medevac.cli --preset routine_ops --rate 10.0 --hours 48
```

### Programmatic

```python
from faer.medevac import ChainScenario, PRESETS, run_chain

# Use preset
results = run_chain(PRESETS['mass_casualty'])

# Custom scenario
scenario = ChainScenario(
    casualty_rate_per_hour=8.0,
    role_2=RoleConfig(capacity=4, service_mean_mins=120),
)
results = run_chain(scenario)

# Transform existing
contested = PRESETS['routine_ops'].with_degraded_medevac(ground=2, rotary=0)
results = run_chain(contested)
```

### Streamlit Demo

Navigate to **MEDEVAC Demo** page in the FAER app. The UI:
1. Loads Pydantic models
2. Renders sliders/inputs bound to model fields
3. Validates on change (red highlight if invalid)
4. Runs simulation on button click
5. Displays metrics and visualizations

## Key Differences from FAER Hospital Model

| Aspect | FAER (Hospital) | FAER-M (Military) |
|--------|-----------------|-------------------|
| Arrival | NSPP time-varying | Poisson / MCI bursts |
| Triage | P1-P4 (clinical) | T1-T4 (battlefield) |
| Routing | Probabilistic | Triage-deterministic |
| Transport | Ambulance handover | MEDEVAC (contested) |
| Downstream | Ward/ITU/Theatre | Role 2/3/4 echelons |
| Exit states | Discharge/Admit | RTD/Evacuated/KIA |
| Capacity | Fixed | Degradable (attrition) |

## Validation Checklist

- [ ] T1 queue times < T2 < T3 (priority working)
- [ ] T3 patients exit at Role 1 (RTD path)
- [ ] T4 patients don't consume Role 2+ resources
- [ ] MEDEVAC bottleneck visible when assets reduced
- [ ] Same seed = identical results (reproducibility)
- [ ] Throughput scales linearly with run length
- [ ] Surge multiplier increases T1 proportion correctly

## Next Steps

1. **Phase 1**: Implement core simulation (this doc)
2. **Phase 2**: Add MEDEVAC transport modeling (ground vs rotary)
3. **Phase 3**: Fixed-wing slot scheduling (strategic evacuation)
4. **Phase 4**: MCI burst arrivals (non-Poisson)
5. **Phase 5**: Resource attrition (degraded capacity over time)
