# CLAUDE.md - Major Incident Module

## Context

You are working on the Major Incident module for Pj-FAER, a healthcare discrete-event simulation. This module adds mass casualty surge events that overwhelm the configured hospital resources.

**Key Principle**: Major incident is an OVERLAY on normal operations. The hospital runs at baseline, then incident triggers and injects additional casualties that stress the system beyond its configured capacity.

---

## Feature Scope

### What This Module Does

- Injects a surge of casualties at a specified time
- Casualties arrive in configurable patterns (bolus, waves, sustained)
- Overload percentage determines how much surge exceeds current capacity
- Tracks how the system degrades under overload
- Supports both Defence (CBRN, combat) and NHS (transport, terrorism) profiles

### What This Module Does NOT Do

- Change baseline hospital configuration (that's the main scenario)
- Model staff fatigue or surge capacity unlocking (future phase)
- Connect to live incident feeds (planner sets parameters manually)

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Trigger | Scheduled time + toggle | Planner sets when incident occurs |
| Overload | Percentage slider | "50% overload" = 1.5x current throughput capacity |
| Profiles | Configurable for Defence + NHS | Single system, scenario profiles select domain |
| Arrival pattern | Bolus / Waves / Sustained | Different incident types have different shapes |
| Integration | Post-Phase 11 | Requires downstream resources to be meaningful |

---

## Data Structures

### MajorIncidentConfig

```python
@dataclass
class MajorIncidentConfig:
    """Major incident configuration.
    
    Triggers a mass casualty event that overwhelms hospital resources
    by the specified overload percentage.
    """
    enabled: bool = False
    
    # Trigger
    trigger_time_hours: float = 6.0  # Hours into simulation
    
    # Overload calculation
    # If ED processes 20/hr normally, 50% overload = 30/hr arrival rate during incident
    overload_percentage: float = 50.0  # 0-200% typical range
    
    # Duration and pattern
    duration_hours: float = 2.0
    arrival_pattern: str = "bolus"  # "bolus" | "waves" | "sustained"
    
    # Wave pattern settings (if pattern == "waves")
    wave_count: int = 3
    wave_interval_mins: float = 30.0
    
    # Casualty profile
    profile: str = "generic"  # "generic" | "blast" | "rta" | "cbrn" | "burns"
    priority_distribution: Dict[str, float] = field(default_factory=lambda: {
        "P1": 0.15,
        "P2": 0.35,
        "P3": 0.40,
        "P4": 0.10,
    })
    
    # Optional: specific injury patterns for detailed modelling
    injury_mix: Optional[Dict[str, float]] = None
```

### CasualtyProfile Presets

```python
CASUALTY_PROFILES = {
    "generic": {
        "name": "Generic Major Incident",
        "priority_distribution": {"P1": 0.15, "P2": 0.35, "P3": 0.40, "P4": 0.10},
        "injury_mix": None,
        "description": "Standard mass casualty distribution",
    },
    "blast": {
        "name": "Blast / Explosion",
        "priority_distribution": {"P1": 0.20, "P2": 0.40, "P3": 0.30, "P4": 0.10},
        "injury_mix": {"blast_lung": 0.15, "penetrating": 0.25, "burns": 0.20, "blunt": 0.40},
        "description": "IED, industrial explosion, terrorist bomb",
    },
    "rta": {
        "name": "Road Traffic Accident (Multi-vehicle)",
        "priority_distribution": {"P1": 0.10, "P2": 0.30, "P3": 0.45, "P4": 0.15},
        "injury_mix": {"head": 0.20, "chest": 0.25, "abdo": 0.15, "limb": 0.40},
        "description": "Motorway pile-up, coach crash",
    },
    "cbrn": {
        "name": "CBRN Incident",
        "priority_distribution": {"P1": 0.25, "P2": 0.35, "P3": 0.30, "P4": 0.10},
        "injury_mix": {"chemical": 0.40, "respiratory": 0.35, "dermal": 0.25},
        "description": "Chemical, biological, radiological, nuclear",
        "requires_decon": True,
        "decon_delay_mins": (15, 45),
    },
    "burns": {
        "name": "Burns Incident",
        "priority_distribution": {"P1": 0.20, "P2": 0.45, "P3": 0.30, "P4": 0.05},
        "injury_mix": {"burns_major": 0.30, "burns_minor": 0.45, "inhalation": 0.25},
        "description": "Industrial fire, nightclub fire",
    },
    "combat": {
        "name": "Combat Casualties (Defence)",
        "priority_distribution": {"P1": 0.25, "P2": 0.40, "P3": 0.25, "P4": 0.10},
        "injury_mix": {"penetrating": 0.35, "blast": 0.30, "burns": 0.15, "blunt": 0.20},
        "description": "Military combat scenario",
    },
}
```

### IncidentArrivalPattern

```python
@dataclass
class IncidentArrivalPattern:
    """Defines how casualties arrive during incident."""
    
    pattern_type: str  # "bolus" | "waves" | "sustained"
    
    def generate_arrival_times(
        self,
        start_time: float,
        duration_hours: float,
        total_casualties: int,
        rng: np.random.Generator,
        wave_count: int = 3,
        wave_interval_mins: float = 30.0,
    ) -> List[float]:
        """Generate arrival times for all casualties."""
        
        if self.pattern_type == "bolus":
            # All arrive within first 30 mins, peak at start
            # Exponential decay from start
            times = start_time + rng.exponential(15, total_casualties)
            return sorted(times[times < start_time + 30])  # Cap at 30 mins
        
        elif self.pattern_type == "waves":
            # Multiple distinct waves
            times = []
            per_wave = total_casualties // wave_count
            for i in range(wave_count):
                wave_start = start_time + (i * wave_interval_mins)
                wave_times = wave_start + rng.exponential(10, per_wave)
                times.extend(wave_times)
            return sorted(times)
        
        elif self.pattern_type == "sustained":
            # Steady elevated rate throughout duration
            duration_mins = duration_hours * 60
            times = start_time + rng.uniform(0, duration_mins, total_casualties)
            return sorted(times)
        
        else:
            raise ValueError(f"Unknown pattern: {self.pattern_type}")
```

---

## Overload Calculation Logic

The overload percentage determines how many additional casualties arrive, relative to the hospital's normal processing capacity.

```python
def calculate_incident_casualties(
    scenario: FullScenario,
    incident_config: MajorIncidentConfig,
) -> int:
    """
    Calculate total incident casualties based on overload percentage.
    
    Overload % is relative to what the hospital can process during
    the incident duration at normal arrival rates.
    
    Example:
    - Normal arrival rate: 20/hr
    - Incident duration: 2 hours
    - Normal throughput in window: 40 patients
    - Overload 50%: Additional 20 casualties (total 60 in 2hr window)
    - Overload 100%: Additional 40 casualties (total 80 in 2hr window)
    """
    # Calculate baseline arrivals during incident window
    normal_rate_per_hour = (
        scenario.ambulance_arrivals_per_hour +
        scenario.walkin_arrivals_per_hour +
        scenario.hems_arrivals_per_hour
    )
    
    baseline_in_window = normal_rate_per_hour * incident_config.duration_hours
    
    # Overload adds this percentage ON TOP of baseline
    additional_casualties = int(baseline_in_window * (incident_config.overload_percentage / 100))
    
    return additional_casualties


def calculate_incident_arrival_rate(
    scenario: FullScenario,
    incident_config: MajorIncidentConfig,
) -> float:
    """
    Calculate arrival rate during incident.
    
    This is the COMBINED rate (normal + incident casualties).
    """
    normal_rate = (
        scenario.ambulance_arrivals_per_hour +
        scenario.walkin_arrivals_per_hour +
        scenario.hems_arrivals_per_hour
    )
    
    multiplier = 1.0 + (incident_config.overload_percentage / 100)
    
    return normal_rate * multiplier
```

---

## Integration Points

### Where Incident Connects to Existing Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     INTEGRATION POINTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ARRIVAL GENERATOR                                           │
│     └── At trigger_time, switch to incident arrival rate        │
│     └── Generate casualties with incident priority distribution │
│     └── After duration, return to normal rate                   │
│                                                                 │
│  2. PATIENT CREATION                                            │
│     └── Tag incident patients: patient.is_incident_casualty    │
│     └── Apply injury profile if detailed modelling enabled     │
│     └── CBRN patients get decon_required flag                  │
│                                                                 │
│  3. RESULTS COLLECTION                                          │
│     └── Separate metrics for incident vs normal patients        │
│     └── Track system state at incident start vs peak vs end    │
│     └── Record "time to return to normal operations"           │
│                                                                 │
│  4. SCHEMATIC (Phase 15)                                        │
│     └── Visual indicator when incident active                   │
│     └── Show surge arrivals as different colour                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Modified Arrival Process

```python
def arrival_generator(
    env: simpy.Environment,
    scenario: FullScenario,
    resources: AEResources,
    results: ResultsCollector,
) -> Generator:
    """
    Generate patient arrivals, including major incident surge.
    """
    incident_config = scenario.major_incident_config
    incident_active = False
    incident_end_time = None
    incident_casualties_remaining = []
    
    # Pre-calculate incident if enabled
    if incident_config and incident_config.enabled:
        incident_start = incident_config.trigger_time_hours * 60  # Convert to mins
        incident_end_time = incident_start + (incident_config.duration_hours * 60)
        
        total_casualties = calculate_incident_casualties(scenario, incident_config)
        
        pattern = IncidentArrivalPattern(incident_config.arrival_pattern)
        incident_casualties_remaining = pattern.generate_arrival_times(
            incident_start,
            incident_config.duration_hours,
            total_casualties,
            scenario.rng_incident,
            incident_config.wave_count,
            incident_config.wave_interval_mins,
        )
        
        results.record_incident_planned(
            start_time=incident_start,
            end_time=incident_end_time,
            total_casualties=total_casualties,
            profile=incident_config.profile,
        )
    
    while True:
        # Check if incident should activate
        if incident_config and incident_config.enabled:
            if not incident_active and env.now >= incident_start:
                incident_active = True
                results.record_incident_started(env.now)
            
            if incident_active and env.now >= incident_end_time:
                incident_active = False
                results.record_incident_ended(env.now)
        
        # Generate normal arrival
        normal_interarrival = scenario.rng_arrivals.exponential(
            60 / scenario.total_arrival_rate
        )
        
        # Check if incident casualty arrives before next normal arrival
        if incident_casualties_remaining:
            next_incident = incident_casualties_remaining[0]
            
            if next_incident < env.now + normal_interarrival:
                # Incident casualty arrives first
                yield env.timeout(next_incident - env.now)
                incident_casualties_remaining.pop(0)
                
                patient = create_incident_patient(scenario, incident_config)
                env.process(patient_journey(env, patient, resources, scenario, results))
                continue
        
        # Normal arrival
        yield env.timeout(normal_interarrival)
        patient = create_normal_patient(scenario)
        env.process(patient_journey(env, patient, resources, scenario, results))


def create_incident_patient(
    scenario: FullScenario,
    incident_config: MajorIncidentConfig,
) -> Patient:
    """Create a patient from major incident."""
    
    # Sample priority from incident distribution
    priority = sample_priority(
        scenario.rng_incident,
        incident_config.priority_distribution
    )
    
    patient = Patient(
        id=generate_patient_id(),
        priority=priority,
        arrival_mode="INCIDENT",  # Special arrival mode
        is_incident_casualty=True,
        incident_profile=incident_config.profile,
    )
    
    # CBRN-specific handling
    if incident_config.profile == "cbrn":
        patient.requires_decon = True
        profile = CASUALTY_PROFILES["cbrn"]
        patient.decon_delay_mins = scenario.rng_incident.uniform(
            *profile["decon_delay_mins"]
        )
    
    return patient
```

---

## UI Panel Design

```
┌─────────────────────────────────────────────────────────────────┐
│  🚨 Major Incident Configuration                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [✓] Enable Major Incident                                      │
│                                                                 │
│  ┌─ Trigger ──────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Incident Start Time                                        │ │
│  │  [━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━] 6.0 hours           │ │
│  │                                                             │ │
│  │  Incident Duration                                          │ │
│  │  [━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 2.0 hours           │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Overload ─────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Overload Percentage                                        │ │
│  │  [━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━] 50%                 │ │
│  │                                                             │ │
│  │  ℹ️ At 50% overload with your current arrival rate of       │ │
│  │     20/hr, this creates ~20 additional casualties over      │ │
│  │     2 hours (total ~60 patients in incident window)         │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Arrival Pattern ──────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  (●) Bolus - All casualties arrive in first 30 mins        │ │
│  │  ( ) Waves - Multiple waves at intervals                   │ │
│  │  ( ) Sustained - Steady elevated rate throughout           │ │
│  │                                                             │ │
│  │  [If Waves selected:]                                       │ │
│  │  Wave Count: [ 3 ]    Interval: [ 30 ] mins                │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Casualty Profile ─────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Profile: [▼ Blast / Explosion                        ]    │ │
│  │                                                             │ │
│  │  ℹ️ IED, industrial explosion, terrorist bomb               │ │
│  │                                                             │ │
│  │  Priority Distribution:                                     │ │
│  │  P1 (Resus):  [━━━━●━━━━━━━━━━━━━━━━━━━] 20%               │ │
│  │  P2 (Urgent): [━━━━━━━━●━━━━━━━━━━━━━━━] 40%               │ │
│  │  P3 (Standard):[━━━━━━●━━━━━━━━━━━━━━━━] 30%               │ │
│  │  P4 (Minor):  [━━●━━━━━━━━━━━━━━━━━━━━━] 10%               │ │
│  │                                                             │ │
│  │  [Profile presets adjust these automatically]               │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Summary ──────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  📊 Incident Impact Estimate                                │ │
│  │                                                             │ │
│  │  Normal arrivals in window:     40 patients                │ │
│  │  Additional incident casualties: 20 patients                │ │
│  │  Total in incident window:       60 patients                │ │
│  │                                                             │ │
│  │  Expected P1 casualties:         12 (20%)                   │ │
│  │  Expected P2 casualties:         24 (40%)                   │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Metrics

### Incident-Specific Metrics

```python
# Pre-incident baseline (snapshot at trigger time)
metrics["pre_incident_ed_occupancy"] = ...
metrics["pre_incident_ed_wait_mean"] = ...

# During incident
metrics["incident_casualties_total"] = ...
metrics["incident_casualties_p1"] = ...
metrics["incident_casualties_p2"] = ...
metrics["incident_peak_ed_occupancy"] = ...
metrics["incident_peak_ed_wait"] = ...
metrics["incident_peak_queue_length"] = ...

# Post-incident recovery
metrics["time_to_normal_operations_mins"] = ...  # When ED returns to <90% occupancy
metrics["incident_mortality_count"] = ...
metrics["incident_patients_transferred_out"] = ...

# Comparison
metrics["incident_vs_normal_wait_ratio"] = ...  # How much worse were waits?
```

---

## Test Scenarios

### Unit Tests

```python
def test_overload_calculation():
    """50% overload doubles arrivals in window."""
    scenario = create_scenario(arrival_rate=20)  # 20/hr
    incident = MajorIncidentConfig(
        enabled=True,
        duration_hours=2.0,
        overload_percentage=50.0,
    )
    
    casualties = calculate_incident_casualties(scenario, incident)
    
    # 20/hr * 2hr = 40 baseline, 50% of 40 = 20 additional
    assert casualties == 20


def test_bolus_pattern_front_loaded():
    """Bolus pattern delivers most casualties in first 30 mins."""
    pattern = IncidentArrivalPattern("bolus")
    times = pattern.generate_arrival_times(
        start_time=0,
        duration_hours=2.0,
        total_casualties=100,
        rng=np.random.default_rng(42),
    )
    
    first_30_mins = len([t for t in times if t < 30])
    assert first_30_mins > 80  # >80% in first 30 mins


def test_incident_patients_tagged():
    """Incident patients have is_incident_casualty flag."""
    patient = create_incident_patient(scenario, incident_config)
    assert patient.is_incident_casualty is True
    assert patient.arrival_mode == "INCIDENT"
```

### Integration Tests

```python
def test_incident_overwhelms_ed():
    """Major incident causes ED to exceed capacity."""
    scenario = create_scenario(
        ed_bays=10,
        arrival_rate=5,  # Normally manageable
        major_incident=MajorIncidentConfig(
            enabled=True,
            trigger_time_hours=2.0,
            duration_hours=1.0,
            overload_percentage=200.0,  # Triple arrivals
        )
    )
    
    results = run_simulation(scenario, duration_hours=6)
    
    # ED should have been overwhelmed
    assert results.metrics["incident_peak_ed_occupancy"] > 10
    assert results.metrics["incident_peak_queue_length"] > 0
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `faer/core/incident.py` | CREATE | MajorIncidentConfig, CasualtyProfiles, patterns |
| `faer/model/full_model.py` | MODIFY | Integrate incident into arrival generator |
| `faer/model/patient.py` | MODIFY | Add incident flags to Patient |
| `faer/experiment/results.py` | MODIFY | Add incident metrics |
| `faer/ui/panels/incident_panel.py` | CREATE | Streamlit UI panel |
| `tests/unit/test_incident.py` | CREATE | Unit tests |
| `tests/integration/test_incident_scenarios.py` | CREATE | Integration tests |

---

## Implementation Order

1. **Data structures** - MajorIncidentConfig, CasualtyProfiles
2. **Overload calculation** - Pure functions, easy to test
3. **Arrival pattern generation** - Bolus, waves, sustained
4. **Patient tagging** - Add flags to Patient dataclass
5. **Arrival generator modification** - Integrate incident logic
6. **Metrics collection** - Incident-specific metrics
7. **UI panel** - Streamlit interface
8. **Tests** - Unit and integration

---

## Dependencies

- Phase 9 (Downstream) - Should be complete for meaningful overload
- Phase 10 (Aeromed) - Incident may generate aeromed transfers
- Phase 11 (Aeromed UI) - Not strictly required

---

## Notes for Implementation

- Keep incident logic SEPARATE from normal arrival logic where possible
- Use feature flag pattern - `if incident_config and incident_config.enabled`
- Pre-calculate incident casualties at simulation start for reproducibility
- Tag patients clearly for post-hoc analysis
- Consider seed management for incident RNG (separate from main RNG)
