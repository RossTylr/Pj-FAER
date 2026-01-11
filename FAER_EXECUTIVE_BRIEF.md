# FAER: Framework for Acute and Emergency Resources
## Executive Technical Brief & Slide Deck Instructions

---

# Part 1: Executive Summary

## What is FAER?

**FAER** is a discrete-event simulation (DES) platform that models patient flow through hospital emergency departments with unprecedented fidelity. Unlike static capacity planning tools, FAER captures the dynamic feedback loops and cascading bottlenecks that cause real-world ED crises.

### The One-Liner
> FAER answers the question every NHS Trust asks but cannot solve with spreadsheets: *"If Ward beds fill up, how long until ambulances queue outside?"*

---

# Part 2: Problem Statement

## The Refined Problem

### For Stakeholders (Plain Language)

**Hospital emergency departments don't fail in isolation - they fail in cascades.**

When ward beds fill up, patients "board" in ED bays waiting for admission. Those blocked ED bays can't accept new patients from triage. Ambulance crews wait longer to hand over. Those ambulances become unavailable for 999 calls. A bed shortage 3 floors up creates a life-threatening delay for someone having a heart attack in the community.

**Current tools cannot model this.** Spreadsheet-based capacity planning treats each department as independent. FAER models the entire system as interconnected, revealing:
- Where congestion *actually originates* (not where symptoms appear)
- Which capacity investments create *system-wide* benefits
- What *fails first* under surge conditions

### For Technical Audiences

FAER implements a **multi-stage queueing network with finite capacity constraints and priority scheduling**. Key innovations include:

1. **Blocking cascade mechanics**: When downstream resources (Ward, ITU) reach capacity, upstream resources (ED bays) cannot release patients, creating backpressure through the system
2. **Diagnostic loop holding**: Patients retain their ED bay while undergoing CT/X-ray/Bloods, accurately modeling real resource contention
3. **Multi-stream arrivals with acuity mix**: Ambulance, helicopter, and walk-in streams each have independent arrival rates and different priority distributions
4. **Time-varying demand**: Non-stationary Poisson process (NSPP) arrival patterns capture hour-of-day and day-of-week effects

---

# Part 3: FAER vs. Alternatives

## Competitive Landscape

| Approach | What It Does | Limitation | FAER Advantage |
|----------|--------------|------------|----------------|
| **Spreadsheet models** | Static bed counts, average LOS | No dynamics, no blocking | Full event-driven simulation |
| **Generic DES tools** (Arena, Simul8) | General-purpose simulation | Requires custom build; no healthcare semantics | Purpose-built for NHS pathways |
| **Capacity planning consultants** | Point-in-time analysis | Expensive, snapshot only | Continuous scenario testing |
| **Queueing theory** | Mathematical elegance | Assumes independence; no blocking | Captures dependencies explicitly |
| **Erlang calculators** | Simple queue sizing | Single queue only | Full multi-stage network |

### Why Not Just Use Simul8/Arena?

Generic DES tools *can* build similar models, but:
1. **Time to value**: FAER is purpose-built; a comparable Arena model takes 6-12 months to develop
2. **Validated patterns**: FAER implements sim-tools patterns validated in published healthcare OR research
3. **Accessibility**: Streamlit UI allows clinicians and managers to run scenarios without simulation expertise
4. **Reproducibility**: Built-in RNG management ensures identical scenarios produce identical results

---

# Part 4: How FAER Works

## System Architecture (Technical Overview)

### Core Simulation Engine

```
Arrival Generator (NSPP)
    |
    v
+---[Handover Bays]---+
    |                 |
    | Ambulance/HEMS  | Walk-in
    |                 |
    v                 v
+-----[Triage]--------+
    |
    v
+------[ED Bays]------+  <--[Diagnostics Loop]
    |    |    |    |          (CT, X-ray, Bloods)
    |    |    |    |
    v    v    v    v
Discharge | Theatre | ITU | Ward
          |    |      |
          +--> ITU -->+
               |
               v
             Ward --> Discharge
```

### Key Components

| Layer | Module | Function |
|-------|--------|----------|
| **Core** | `entities.py` | Priority (P1-P4), NodeType, ArrivalMode enums |
| **Core** | `scenario.py` | 40+ configurable parameters (capacities, times, probabilities) |
| **Core** | `arrivals.py` | NSPP thinning algorithm for time-varying demand |
| **Model** | `full_model.py` | SimPy process orchestration (2,200+ lines) |
| **Model** | `patient.py` | Patient entity tracking 50+ journey timestamps |
| **Results** | `collector.py` | Real-time event logging during simulation |
| **Results** | `costs.py` | Post-hoc financial analysis (NHS Reference Costs) |
| **Experiment** | `runner.py` | Multi-replication with confidence intervals |

### Simulation Mechanics

**Priority Queuing**: All patient-facing resources use `PriorityResource`. P1 (resuscitation) patients bypass queues; P4 (minor) patients wait.

**Blocking Cascade**: When `downstream_enabled=True`:
```python
# Patient cannot release ED bay until downstream bed secured
with ward_beds.request() as ward_req:
    yield ward_req  # Wait here if Ward full
    # Only NOW release ED bay
    ed_bay_release()
```

**Diagnostic Loop**: Patients leave ED bay for scans but *keep* the bay reserved:
```python
# Patient goes to CT but bay remains OCCUPIED
yield from ct_scan_process(patient)  # Returns to same bay
```

---

# Part 5: Use Cases

## Primary Applications

### 1. Winter Planning
**Question**: *"How many extra ward beds do we need to prevent ED crowding during flu season?"*
- Model baseline capacity
- Increase arrival rates by 15-30%
- Identify downstream capacity that prevents blocking cascade

### 2. Capital Investment Appraisal
**Question**: *"Should we build 4 more ED bays or 10 more ward beds?"*
- Run scenarios with each option
- Compare system-wide impact on P(delay), mean wait, admission rate
- Quantify cost per patient improvement

### 3. Major Incident Planning
**Question**: *"Can we handle a mass casualty event with current capacity?"*
- Configure surge arrivals (50 P1 patients in 2 hours)
- Identify when system saturates
- Test resilience of aeromed evacuation pathways

### 4. Operational Improvement
**Question**: *"What happens if we reduce diagnostic turnaround by 10 minutes?"*
- Adjust CT/X-ray/Bloods service times
- Measure impact on ED bay holding time
- Quantify throughput improvement

### 5. Staff Rostering
**Question**: *"Do we need a second triage nurse at 2am?"*
- Run scenarios with 1 vs 2 triage staff
- Compare triage wait times
- Model cost-benefit of additional shift

---

# Part 6: Metrics & Outputs

## Key Performance Indicators

| Metric | Definition | NHS Context |
|--------|------------|-------------|
| **P(Delay)** | Probability a patient waits for treatment | 4-hour target proxy |
| **Mean Treatment Wait** | Average time from triage to ED bay | Clinical priority metric |
| **Mean System Time** | Arrival to departure (entire journey) | Patient experience |
| **95th Percentile System Time** | Worst-case experience | Outlier impact |
| **Admission Rate** | Proportion admitted vs discharged | Casemix indicator |
| **Resource Utilisation** | Time-weighted occupancy per resource | Capacity headroom |

## Outputs Available

- **Confidence Intervals**: All KPIs reported with 95% CIs from multi-replication
- **Utilisation Heatmaps**: Visual identification of bottlenecks
- **Distribution Plots**: Full distributions across replications (not just means)
- **Cost Breakdown**: By location, by priority, by transport mode
- **CSV Export**: Raw data for further analysis

---

# Part 7: Next Steps / Roadmap

## Current State (Phase 10)
- Full ED pathway simulation
- Priority queuing (P1-P4)
- Multi-stream arrivals (Ambulance, HEMS, Walk-in)
- Diagnostics loop (CT, X-ray, Bloods)
- Downstream blocking (Theatre, ITU, Ward)
- Aeromedical evacuation (HEMS, fixed-wing)
- Post-hoc cost modeling

## Planned Enhancements

| Phase | Feature | Impact |
|-------|---------|--------|
| **11** | Advanced transfers (CCU, Burns, Paediatric) | Specialist network modeling |
| **12** | Staffing constraints | Not just beds - staff availability |
| **13** | Infection control | Isolation room mechanics |
| **14** | Discharge delays | Social care dependency |
| **15** | Multi-site modeling | Hospital networks |

## Integration Opportunities

- **Real-time data feeds**: Connect to ED dashboards for live parameter calibration
- **Automated scenario running**: Nightly batch runs for capacity alerts
- **API endpoint**: Integration with Trust planning systems

---

# Part 8: LLM Instructions for Slide Deck

## Objective

Create an 8-10 slide executive presentation introducing FAER to a mixed audience of NHS executives, clinical leads, and technical stakeholders.

## Slide Structure

### Slide 1: Title
- "FAER: Framework for Acute and Emergency Resources"
- Subtitle: "Discrete-event simulation for hospital patient flow"
- Include Streamlit app badge if available

### Slide 2: The Problem
- Lead with the cascade insight: "Beds don't fail alone - they fail together"
- Visual: Simple diagram showing Ward full -> ED blocked -> Ambulances waiting
- Stats: Reference NHS winter pressure stories (e.g., "30-hour ED waits", "ambulance handover delays")

### Slide 3: Why Current Tools Fail
- Spreadsheets: Static, no dynamics
- Generic DES: Requires build, expensive
- Queueing theory: Assumes independence
- Consultants: Snapshot, not continuous

### Slide 4: FAER's Innovation
- Key insight: Models the *feedback loops*, not just the queues
- Three innovations:
  1. Blocking cascade mechanics
  2. Diagnostic loop holding
  3. Multi-stream priority queuing
- Visual: System schematic from FAER (include DOT diagram)

### Slide 5: How It Works (High Level)
- Patient journey visual: Arrival -> Triage -> ED -> Downstream -> Exit
- Mention: SimPy engine, Streamlit UI, reproducible results
- Keep technical details minimal for executive audience

### Slide 6: Use Cases
- 4 quadrant layout:
  1. Winter planning
  2. Capital investment
  3. Major incident
  4. Operational improvement
- One sentence each with quantifiable example

### Slide 7: Sample Output
- Screenshot or mockup of Results dashboard
- Highlight KPIs: P(Delay), Mean Wait, Utilisation
- Show confidence intervals (conveys rigor)

### Slide 8: Competitive Positioning
- Table comparing FAER vs alternatives
- Emphasize: Purpose-built, validated, accessible

### Slide 9: Roadmap & Next Steps
- Current state (Phase 10)
- Near-term additions (staffing, infection control)
- Integration potential (real-time feeds, API)

### Slide 10: Call to Action
- "Run your scenarios today"
- Link to Streamlit app
- Contact details for pilot engagement

## Design Guidelines

- **Color scheme**: NHS blue (#005EB8) and white, with red for warnings
- **Font**: Sans-serif (Arial, Helvetica)
- **Visuals over text**: One diagram per slide where possible
- **Data-driven**: Include at least one chart/metric per technical slide
- **Accessibility**: High contrast, minimum 24pt body text

## Tone Guidelines

| Audience | Emphasis |
|----------|----------|
| Executives | Business impact, cost savings, risk reduction |
| Clinicians | Patient safety, wait times, operational improvement |
| Technical | Simulation fidelity, reproducibility, validation |

## Key Phrases to Include

- "Blocking cascade" (the core insight)
- "System-wide impact" (not siloed thinking)
- "Reproducible results" (scientific rigor)
- "Purpose-built for NHS" (not generic)
- "Run scenarios in minutes" (accessibility)

---

# ANNEX A: Technical Deep Dive

## A.1 Simulation Engine Architecture

### SimPy Foundation

FAER uses **SimPy 4.1+**, a process-based discrete-event simulation library. Key concepts:

- **Environment**: The simulation clock and event scheduler
- **Processes**: Generator functions that `yield` events (timeouts, resource requests)
- **Resources**: Capacity-constrained entities (beds, staff, equipment)
- **PriorityResource**: Resources where requests are served in priority order

### Patient Journey Implementation

```python
def patient_process(env, patient, resources, scenario, collector):
    """Core patient journey through ED system."""

    # 1. HANDOVER (if ambulance/helicopter)
    if patient.mode != ArrivalMode.SELF_PRESENTATION:
        yield from handover_process(env, patient, resources)

    # 2. TRIAGE (P1 bypass, P2-P4 queue with priority)
    if patient.priority != Priority.P1_IMMEDIATE:
        yield from triage_process(env, patient, resources)

    # 3. ED TREATMENT (priority queuing)
    with resources.ed_bays.request(priority=patient.priority.value) as req:
        yield req
        patient.treatment_start = env.now

        # 4. DIAGNOSTICS (patient keeps bay)
        if patient.requires_diagnostics:
            yield from diagnostics_loop(env, patient, resources)

        # 5. TREATMENT TIME
        service_time = sample_lognormal(scenario.ed_mean, scenario.ed_cv)
        yield env.timeout(service_time)

        # 6. DOWNSTREAM (blocking mechanics)
        if scenario.downstream_enabled and patient.is_admitted:
            yield from downstream_process(env, patient, resources)

        patient.treatment_end = env.now
```

### Blocking Mechanics (Critical)

The key to FAER's realism is that patients cannot release upstream resources until downstream resources are secured:

```python
def downstream_process(env, patient, resources, scenario):
    """Process patient through downstream pathway with blocking."""

    if patient.disposition == Disposition.ADMIT_WARD:
        # Patient CANNOT release ED bay until Ward bed secured
        patient.boarding_start = env.now
        with resources.ward_beds.request() as req:
            yield req  # BLOCKING WAIT - ED bay still occupied
        patient.boarding_end = env.now

        # Only now release ED bay (handled by outer context manager)
```

This creates the cascade: Ward full -> ED boarding -> Triage backed up -> Handover delayed -> Ambulances unavailable.

### Random Number Generation

FAER uses a disciplined RNG strategy for reproducibility:

```python
@dataclass
class FullScenario:
    random_seed: int = 42

    def __post_init__(self):
        # Each stochastic element gets its own RNG stream
        self.rng_arrivals = np.random.default_rng(self.random_seed)
        self.rng_service = np.random.default_rng(self.random_seed + 1)
        self.rng_routing = np.random.default_rng(self.random_seed + 2)
        self.rng_diagnostics = np.random.default_rng(self.random_seed + 3)
        # ... etc
```

**Why this matters**:
- Same seed = identical results (reproducibility)
- Independent streams = changing one parameter doesn't affect others (variance reduction)
- Easy parallelization (each replication uses seed + replication_number)

## A.2 Arrival Modeling

### Non-Stationary Poisson Process (NSPP)

Hospital arrivals vary by hour and day. FAER implements NSPP via the **thinning algorithm**:

```python
def nspp_thinning_sample(hourly_rates, max_rate, rng):
    """Sample next inter-arrival time using thinning."""
    while True:
        # Sample from homogeneous Poisson at max rate
        candidate_iat = rng.exponential(1 / max_rate)
        candidate_time = current_time + candidate_iat

        # Get rate at candidate time
        hour = int(candidate_time / 60) % 24
        rate_at_time = hourly_rates[hour]

        # Accept with probability rate/max_rate
        if rng.random() < rate_at_time / max_rate:
            return candidate_iat
```

### Arrival Configuration Tiers

FAER supports three levels of demand specification:

| Model | Use Case | Parameters |
|-------|----------|------------|
| **Simple** | Quick exploration | Single demand multiplier (0.5x - 2.0x) |
| **Profile 24H** | Realistic patterns | Hourly rates + day type (Monday, Weekend, etc.) |
| **Detailed** | Major incident | Per-hour, per-mode explicit counts |

### Acuity Mix

Each arrival stream (Ambulance, Helicopter, Walk-in) has its own priority distribution:

```python
# Example: Ambulance arrivals
ambulance_acuity_mix = {
    Priority.P1_IMMEDIATE: 0.15,    # 15% immediate
    Priority.P2_VERY_URGENT: 0.40,  # 40% very urgent
    Priority.P3_URGENT: 0.35,       # 35% urgent
    Priority.P4_STANDARD: 0.10,     # 10% standard
}
```

Walk-ins have lower acuity (more P3/P4); helicopter arrivals have higher (more P1/P2).

## A.3 Resource Modeling

### Resource Types

| Resource | SimPy Type | Priority? | FAER Usage |
|----------|------------|-----------|------------|
| Handover Bays | `Resource` | No | FIFO queuing |
| Triage | `PriorityResource` | Yes | P1 bypass, P2-P4 priority |
| ED Bays | `PriorityResource` | Yes | Main treatment area |
| CT/X-ray/Bloods | `PriorityResource` | Yes | Diagnostic services |
| Theatre | `PriorityResource` | Yes | Surgical cases |
| ITU Beds | `Resource` | No | Critical care |
| Ward Beds | `Resource` | No | General admission |
| Ambulance Fleet | `Resource` | No | Vehicle availability |
| HEMS Fleet | `Resource` | No | Helicopter availability |

### Diagnostic Loop Detail

The diagnostic loop is FAER's most sophisticated resource interaction:

```python
def diagnostics_loop(env, patient, resources, scenario):
    """Patient undergoes diagnostics while holding ED bay."""

    for diagnostic_type in patient.required_diagnostics:
        config = scenario.diagnostic_configs[diagnostic_type]
        resource = resources.diagnostics[diagnostic_type]

        # Patient leaves bay but it stays OCCUPIED
        patient.diagnostic_queue_start[diagnostic_type] = env.now

        with resource.request(priority=patient.priority.value) as req:
            yield req
            patient.diagnostic_start[diagnostic_type] = env.now

            # Procedure time
            yield env.timeout(sample_lognormal(config.mean, config.cv))

            # Results turnaround (radiologist reporting, lab analysis)
            yield env.timeout(config.turnaround_time)

            patient.diagnostic_end[diagnostic_type] = env.now

        patient.diagnostics_completed.append(diagnostic_type)

    # Patient returns to same ED bay (never released it)
```

## A.4 Metrics Calculation

### Time-Weighted Utilisation

Simple occupancy percentage is misleading for time-varying systems. FAER computes **time-weighted utilisation**:

```python
def compute_utilisation(resource_log, capacity, warm_up, run_length):
    """Compute time-weighted utilisation from event log."""

    total_occupied_time = 0
    last_event_time = warm_up
    current_occupancy = 0

    for event in resource_log:
        if event.time < warm_up:
            continue
        if event.time > warm_up + run_length:
            break

        # Add time at previous occupancy level
        duration = event.time - last_event_time
        total_occupied_time += current_occupancy * duration

        # Update state
        if event.type == 'request':
            current_occupancy += 1
        elif event.type == 'release':
            current_occupancy -= 1

        last_event_time = event.time

    # Final segment to end of run
    total_occupied_time += current_occupancy * (warm_up + run_length - last_event_time)

    return total_occupied_time / (run_length * capacity)
```

### Confidence Intervals

FAER uses the standard formula for 95% CI:

```python
def compute_ci(values):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    ci_half_width = 1.96 * se

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - ci_half_width,
        'ci_upper': mean + ci_half_width,
        'ci_half_width': ci_half_width,
    }
```

## A.5 Cost Modeling

### Post-Hoc Calculation

Cost modeling is **separate from simulation**. This design choice means:
- Simulation results are reusable across different cost assumptions
- Focus remains on flow mechanics, not financial distortion
- Easy sensitivity analysis on cost parameters

### Cost Configuration

```python
@dataclass
class CostConfig:
    # Bed-day rates (per 24 hours)
    ed_bay_per_day: float = 500.0      # High staff ratio
    itu_bed_per_day: float = 2000.0    # 1:1 nursing, ventilators
    ward_bed_per_day: float = 400.0    # Lower acuity

    # Theatre (per hour)
    theatre_per_hour: float = 2000.0   # Team, consumables

    # Transport (per journey)
    ambulance_per_journey: float = 275.0
    hems_per_flight: float = 3500.0
    fixedwing_per_flight: float = 15000.0

    # Priority multipliers
    priority_multipliers: Dict[str, float] = {
        "P1": 2.0,   # Resus: 2x resource intensity
        "P2": 1.4,   # Urgent: 1.4x
        "P3": 1.0,   # Standard: baseline
        "P4": 0.7,   # Minor: 0.7x
    }
```

---

# ANNEX B: Streamlit UI Guide

## B.1 Page Structure

| Page | Purpose | Key Controls |
|------|---------|--------------|
| **Home** | Overview, quick stats | Navigation links |
| **1_Arrivals** | Fleet config, arrival model selection | Ambulances, helicopters, demand slider |
| **2_Resources** | Capacity settings | ED bays, Triage, Diagnostics, Downstream |
| **3_Schematic** | Live flow diagram | Auto-updates from config |
| **4_Aeromed** | Aeromedical settings | HEMS/fixed-wing, operating hours |
| **5_Run** | Execute simulation | Run button, progress bar |
| **6_Results** | KPI dashboard | Charts, export |
| **7_Compare** | Scenario comparison | Side-by-side A/B |
| **8_Sensitivity** | One-factor analysis | Tornado plots |

## B.2 Session State Management

Streamlit session state persists scenario configuration across pages:

```python
# Initialize defaults
if 'n_ed_bays' not in st.session_state:
    st.session_state.n_ed_bays = 20

# Widget updates state
st.number_input("ED Bays", key='n_ed_bays', min_value=1, max_value=100)

# Access anywhere
scenario = FullScenario(n_ed_bays=st.session_state.n_ed_bays, ...)
```

## B.3 Results Visualization

The Results page uses Plotly for interactive charts:

- **KPI Cards**: Metric widgets with CI captions
- **Utilisation Heatmap**: Bar chart colored by category
- **Distribution Histograms**: Show variability across replications
- **Pie Charts**: Acuity mix, disposition breakdown

---

# ANNEX C: Testing & Validation

## C.1 Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Reproducibility** | Same seed = same results | `test_reproducibility()` |
| **Flow** | Patients traverse correct path | `test_patient_journey()` |
| **Priority** | P1 gets served before P4 | `test_priority_queuing()` |
| **Blocking** | Downstream full blocks upstream | `test_blocking_cascade()` |
| **Arrivals** | Rates match configured values | `test_arrival_rates()` |
| **Diagnostics** | Loop mechanics work correctly | `test_diagnostic_loop()` |
| **Costs** | Calculations are accurate | `test_cost_calculation()` |

## C.2 Validation Patterns

### Deterministic Scenarios

For testing, use configurations with known outcomes:

```python
def test_no_queue_when_overprovisioned():
    """With excess capacity, no patient should wait."""
    scenario = FullScenario(
        n_ed_bays=100,  # Way more than needed
        arrival_rate=1.0,  # Low demand
        random_seed=42,
    )
    results = run_simulation(scenario)

    assert all(p.treatment_wait == 0 for p in results.patients)
```

### Statistical Validation

For stochastic behavior, check distributions:

```python
def test_arrival_count_distribution():
    """Arrivals should follow Poisson distribution."""
    scenario = FullScenario(arrival_rate=10.0, random_seed=42)

    arrival_counts = []
    for rep in range(100):
        scenario_rep = scenario.clone_with_seed(scenario.random_seed + rep)
        results = run_simulation(scenario_rep)
        arrival_counts.append(results['arrivals'])

    # Mean should be close to rate * time
    expected_mean = 10.0 * (scenario.run_length / 60)
    assert abs(np.mean(arrival_counts) - expected_mean) < 0.1 * expected_mean
```

---

# ANNEX D: Glossary

| Term | Definition |
|------|------------|
| **Acuity** | Patient severity classification (Resus, Majors, Minors) |
| **Boarding** | Patient occupying ED bay while waiting for ward bed |
| **Blocking cascade** | Chain of capacity constraints from downstream to upstream |
| **DES** | Discrete-event simulation |
| **IAT** | Inter-arrival time |
| **LOS** | Length of stay |
| **NSPP** | Non-stationary Poisson process |
| **P(Delay)** | Probability of experiencing wait for treatment |
| **Priority** | Triage urgency (P1=immediate, P4=standard) |
| **Thinning** | Algorithm for generating NSPP arrivals |
| **Utilisation** | Time-weighted fraction of capacity in use |
| **Warm-up** | Initial simulation period discarded from statistics |

---

# ANNEX E: References & Attribution

## Simulation Framework

- **SimPy**: Discrete-event simulation library. https://simpy.readthedocs.io/
- **sim-tools**: Healthcare DES utilities by Monks, T., Heather, A., Harper, A. https://github.com/TomMonks/sim-tools (MIT License)
- **STARS Project**: pythonhealthdatascience (MIT License)

## Healthcare Operations Research

- Monks T, et al. (2016). "A modelling tool for capacity planning in acute and community stroke services." BMC Health Services Research.
- Harper A, et al. (2023). "Developing a framework for the evaluation and comparison of health service simulation software."

## NHS Context

- NHS England. "Urgent and Emergency Care Review."
- Getting It Right First Time (GIRFT). Emergency Medicine.
- NHS Reference Costs 2022/23.

---

*Document version: 1.0*
*Last updated: January 2026*
*FAER Phase: 10 (Aeromedical & Cost Modeling)*
