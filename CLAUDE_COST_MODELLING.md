# CLAUDE.md - Cost Modelling Module

## Context

You are working on the Cost Modelling module for Pj-FAER, a healthcare discrete-event simulation. This module calculates financial costs from simulation outputs using simple bed-day rates.

**Key Principle**: Cost modelling is POST-HOC calculation. It does not change simulation behaviour - it interprets results through a financial lens.

---

## Feature Scope

### What This Module Does

- Calculates costs from simulation outputs after run completes
- Uses simple bed-day and per-episode rates
- Supports GBP, USD, EUR currencies
- Provides cost breakdown by location, priority, transport mode
- Enables cost comparison between scenarios

### What This Module Does NOT Do

- Activity-based costing (too granular for this phase)
- Staff cost modelling (no staff resources yet)
- Real-time cost accumulation during simulation
- Opportunity cost calculation (complex, subjective)
- Budget forecasting or financial planning

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Simple bed-day rates | Sufficient for planning, avoids complexity |
| Timing | Post-hoc calculation | Doesn't slow simulation, easy to adjust rates |
| Granularity | Location + Priority + Transport | Actionable breakdown without over-engineering |
| Currency | Configurable (GBP default) | Defence may need USD, NHS uses GBP |

---

## Data Structures

### CostConfig

```python
@dataclass
class CostConfig:
    """Cost modelling configuration.
    
    All rates are per-unit costs. Bed-day rates are per 24-hour period.
    Theatre rate is per hour (procedures vary in length).
    """
    enabled: bool = True
    currency: str = "GBP"  # "GBP" | "USD" | "EUR"
    
    # === BED-DAY RATES ===
    # Cost per 24 hours of bed occupancy
    
    ed_bay_per_day: float = 500.0
    # ED bays are expensive: high staff ratio, equipment, turnover
    
    itu_bed_per_day: float = 2000.0
    # ITU most expensive: 1:1 nursing, ventilators, monitoring
    
    ward_bed_per_day: float = 400.0
    # General ward: lower acuity, shared nursing
    
    # === HOURLY RATES ===
    # Theatre charged per hour (procedures vary)
    
    theatre_per_hour: float = 2000.0
    # Includes surgeon, anaesthetist, scrub team, consumables
    
    # === PER-EPISODE COSTS ===
    # Fixed costs incurred once per patient
    
    triage_cost: float = 20.0
    # Brief assessment, documentation
    
    diagnostics_base_cost: float = 75.0
    # Average across bloods, X-ray, CT (simplified)
    
    discharge_cost: float = 40.0
    # Medications, paperwork, transport booking
    
    # === TRANSPORT COSTS ===
    # Per-journey costs for patient transport
    
    ambulance_per_journey: float = 275.0
    # Emergency ambulance (arrival)
    
    hems_per_flight: float = 3500.0
    # Helicopter: fuel, crew, maintenance amortised
    
    fixedwing_per_flight: float = 15000.0
    # Fixed-wing aeromed: longer range, higher cost
    
    road_transfer_per_journey: float = 225.0
    # Non-emergency patient transport
    
    # === OPTIONAL MULTIPLIERS ===
    # Adjust costs by patient priority (P1 uses more resources)
    
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "P1": 2.0,   # Resus: 2x resource intensity
        "P2": 1.4,   # Urgent: 1.4x
        "P3": 1.0,   # Standard: baseline
        "P4": 0.7,   # Minor: 0.7x
    })
    
    # Apply priority multipliers to these cost categories
    apply_priority_to: List[str] = field(default_factory=lambda: [
        "ed_bay",
        "diagnostics",
    ])
```

### CostBreakdown

```python
@dataclass
class CostBreakdown:
    """Detailed cost breakdown from simulation results."""
    
    currency: str
    
    # By location
    ed_bay_costs: float = 0.0
    theatre_costs: float = 0.0
    itu_bed_costs: float = 0.0
    ward_bed_costs: float = 0.0
    
    # Per-episode
    triage_costs: float = 0.0
    diagnostics_costs: float = 0.0
    discharge_costs: float = 0.0
    
    # Transport
    ambulance_costs: float = 0.0
    hems_costs: float = 0.0
    fixedwing_costs: float = 0.0
    road_transfer_costs: float = 0.0
    
    # Aggregates
    @property
    def total_bed_costs(self) -> float:
        return self.ed_bay_costs + self.itu_bed_costs + self.ward_bed_costs
    
    @property
    def total_transport_costs(self) -> float:
        return (self.ambulance_costs + self.hems_costs + 
                self.fixedwing_costs + self.road_transfer_costs)
    
    @property
    def total_episode_costs(self) -> float:
        return self.triage_costs + self.diagnostics_costs + self.discharge_costs
    
    @property
    def grand_total(self) -> float:
        return (self.total_bed_costs + self.theatre_costs + 
                self.total_transport_costs + self.total_episode_costs)
    
    # Per-patient metrics (set during calculation)
    total_patients: int = 0
    
    @property
    def cost_per_patient(self) -> float:
        if self.total_patients == 0:
            return 0.0
        return self.grand_total / self.total_patients


@dataclass
class CostByPriority:
    """Cost breakdown by patient priority."""
    p1_total: float = 0.0
    p1_count: int = 0
    p2_total: float = 0.0
    p2_count: int = 0
    p3_total: float = 0.0
    p3_count: int = 0
    p4_total: float = 0.0
    p4_count: int = 0
    
    @property
    def p1_per_patient(self) -> float:
        return self.p1_total / self.p1_count if self.p1_count > 0 else 0.0
    
    @property
    def p2_per_patient(self) -> float:
        return self.p2_total / self.p2_count if self.p2_count > 0 else 0.0
    
    @property
    def p3_per_patient(self) -> float:
        return self.p3_total / self.p3_count if self.p3_count > 0 else 0.0
    
    @property
    def p4_per_patient(self) -> float:
        return self.p4_total / self.p4_count if self.p4_count > 0 else 0.0
```

---

## Cost Calculation Logic

### Main Calculator

```python
def calculate_costs(
    results: SimulationResults,
    config: CostConfig,
) -> CostBreakdown:
    """
    Calculate costs from simulation results.
    
    This is a POST-HOC calculation - it processes completed results,
    not live simulation state.
    """
    breakdown = CostBreakdown(currency=config.currency)
    breakdown.total_patients = len(results.patients)
    
    # === BED-DAY COSTS ===
    
    # ED bay time (convert minutes to days)
    total_ed_mins = sum(
        (p.ed_end - p.ed_start) 
        for p in results.patients 
        if p.ed_start and p.ed_end
    )
    breakdown.ed_bay_costs = (total_ed_mins / 1440) * config.ed_bay_per_day
    
    # ITU bed time
    total_itu_mins = sum(
        (p.itu_end - p.itu_start)
        for p in results.patients
        if p.itu_start and p.itu_end
    )
    breakdown.itu_bed_costs = (total_itu_mins / 1440) * config.itu_bed_per_day
    
    # Ward bed time
    total_ward_mins = sum(
        (p.ward_end - p.ward_start)
        for p in results.patients
        if p.ward_start and p.ward_end
    )
    breakdown.ward_bed_costs = (total_ward_mins / 1440) * config.ward_bed_per_day
    
    # === THEATRE COSTS (HOURLY) ===
    
    total_theatre_mins = sum(
        (p.surgery_end - p.surgery_start)
        for p in results.patients
        if p.surgery_start and p.surgery_end
    )
    breakdown.theatre_costs = (total_theatre_mins / 60) * config.theatre_per_hour
    
    # === PER-EPISODE COSTS ===
    
    # Every patient gets triaged
    breakdown.triage_costs = len(results.patients) * config.triage_cost
    
    # Diagnostics (with optional priority multiplier)
    for patient in results.patients:
        multiplier = get_priority_multiplier(patient, config, "diagnostics")
        breakdown.diagnostics_costs += config.diagnostics_base_cost * multiplier
    
    # Discharge costs for patients who completed journey
    discharged = [p for p in results.patients if p.outcome and "DISCHARGE" in p.outcome]
    breakdown.discharge_costs = len(discharged) * config.discharge_cost
    
    # === TRANSPORT COSTS ===
    
    # Ambulance arrivals
    ambulance_arrivals = [p for p in results.patients if p.arrival_mode == "AMBULANCE"]
    breakdown.ambulance_costs = len(ambulance_arrivals) * config.ambulance_per_journey
    
    # HEMS arrivals + HEMS evacuations
    hems_arrivals = [p for p in results.patients if p.arrival_mode == "HEMS"]
    hems_evacuations = [p for p in results.patients if p.aeromed_type == "HEMS"]
    breakdown.hems_costs = (len(hems_arrivals) + len(hems_evacuations)) * config.hems_per_flight
    
    # Fixed-wing evacuations
    fw_evacuations = [p for p in results.patients if p.aeromed_type == "FIXED_WING"]
    breakdown.fixedwing_costs = len(fw_evacuations) * config.fixedwing_per_flight
    
    # Road transfers
    road_transfers = [p for p in results.patients if p.outcome == "TRANSFER_ROAD"]
    breakdown.road_transfer_costs = len(road_transfers) * config.road_transfer_per_journey
    
    return breakdown


def get_priority_multiplier(
    patient: Patient,
    config: CostConfig,
    cost_category: str,
) -> float:
    """Get priority-based cost multiplier for a category."""
    if cost_category not in config.apply_priority_to:
        return 1.0
    
    priority_key = f"P{patient.priority}"
    return config.priority_multipliers.get(priority_key, 1.0)
```

### Cost by Priority

```python
def calculate_costs_by_priority(
    results: SimulationResults,
    config: CostConfig,
) -> CostByPriority:
    """Calculate cost breakdown by patient priority."""
    
    by_priority = CostByPriority()
    
    for patient in results.patients:
        patient_cost = calculate_patient_cost(patient, config)
        
        if patient.priority == 1:
            by_priority.p1_total += patient_cost
            by_priority.p1_count += 1
        elif patient.priority == 2:
            by_priority.p2_total += patient_cost
            by_priority.p2_count += 1
        elif patient.priority == 3:
            by_priority.p3_total += patient_cost
            by_priority.p3_count += 1
        elif patient.priority == 4:
            by_priority.p4_total += patient_cost
            by_priority.p4_count += 1
    
    return by_priority


def calculate_patient_cost(patient: Patient, config: CostConfig) -> float:
    """Calculate total cost for a single patient."""
    cost = 0.0
    
    # Triage (everyone)
    cost += config.triage_cost
    
    # ED time
    if patient.ed_start and patient.ed_end:
        ed_days = (patient.ed_end - patient.ed_start) / 1440
        multiplier = get_priority_multiplier(patient, config, "ed_bay")
        cost += ed_days * config.ed_bay_per_day * multiplier
    
    # Theatre time
    if patient.surgery_start and patient.surgery_end:
        theatre_hours = (patient.surgery_end - patient.surgery_start) / 60
        cost += theatre_hours * config.theatre_per_hour
    
    # ITU time
    if patient.itu_start and patient.itu_end:
        itu_days = (patient.itu_end - patient.itu_start) / 1440
        cost += itu_days * config.itu_bed_per_day
    
    # Ward time
    if patient.ward_start and patient.ward_end:
        ward_days = (patient.ward_end - patient.ward_start) / 1440
        cost += ward_days * config.ward_bed_per_day
    
    # Diagnostics
    multiplier = get_priority_multiplier(patient, config, "diagnostics")
    cost += config.diagnostics_base_cost * multiplier
    
    # Transport (arrival)
    if patient.arrival_mode == "AMBULANCE":
        cost += config.ambulance_per_journey
    elif patient.arrival_mode == "HEMS":
        cost += config.hems_per_flight
    
    # Transport (evacuation)
    if patient.aeromed_type == "HEMS":
        cost += config.hems_per_flight
    elif patient.aeromed_type == "FIXED_WING":
        cost += config.fixedwing_per_flight
    
    # Discharge
    if patient.outcome and "DISCHARGE" in patient.outcome:
        cost += config.discharge_cost
    
    return cost
```

---

## Scenario Comparison

```python
def compare_scenario_costs(
    results_a: SimulationResults,
    results_b: SimulationResults,
    config: CostConfig,
    labels: Tuple[str, str] = ("Scenario A", "Scenario B"),
) -> Dict[str, Any]:
    """Compare costs between two scenarios."""
    
    costs_a = calculate_costs(results_a, config)
    costs_b = calculate_costs(results_b, config)
    
    return {
        "labels": labels,
        "currency": config.currency,
        
        # Totals
        "total_a": costs_a.grand_total,
        "total_b": costs_b.grand_total,
        "difference": costs_b.grand_total - costs_a.grand_total,
        "difference_pct": ((costs_b.grand_total - costs_a.grand_total) / costs_a.grand_total * 100)
                          if costs_a.grand_total > 0 else 0,
        
        # Per patient
        "per_patient_a": costs_a.cost_per_patient,
        "per_patient_b": costs_b.cost_per_patient,
        
        # Category breakdown
        "bed_costs_a": costs_a.total_bed_costs,
        "bed_costs_b": costs_b.total_bed_costs,
        "transport_costs_a": costs_a.total_transport_costs,
        "transport_costs_b": costs_b.total_transport_costs,
        
        # Full breakdowns
        "breakdown_a": costs_a,
        "breakdown_b": costs_b,
    }
```

---

## UI Panel Design

```
┌─────────────────────────────────────────────────────────────────┐
│  💷 Cost Modelling Configuration                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [✓] Enable Cost Modelling                                      │
│                                                                 │
│  Currency: [▼ GBP (£)                                    ]     │
│                                                                 │
│  ┌─ Bed-Day Rates ────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  ED Bay (per day)      ITU Bed (per day)   Ward (per day)  │ │
│  │  £ [ 500  ]            £ [ 2000 ]          £ [ 400  ]      │ │
│  │                                                             │ │
│  │  Theatre (per hour)                                         │ │
│  │  £ [ 2000 ]                                                │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Per-Episode Costs ────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Triage         Diagnostics (avg)    Discharge             │ │
│  │  £ [ 20   ]     £ [ 75    ]          £ [ 40   ]            │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Transport Costs ──────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Ambulance (journey)    HEMS (flight)    Road Transfer     │ │
│  │  £ [ 275  ]             £ [ 3500 ]       £ [ 225  ]        │ │
│  │                                                             │ │
│  │  Fixed-Wing Aeromed (flight)                                │ │
│  │  £ [ 15000 ]                                               │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ Priority Multipliers (optional) ──────────────────────────┐ │
│  │                                                             │ │
│  │  [✓] Apply priority-based cost adjustments                 │ │
│  │                                                             │ │
│  │  P1 (Resus): [ 2.0 ]x    P2 (Urgent): [ 1.4 ]x            │ │
│  │  P3 (Std):   [ 1.0 ]x    P4 (Minor):  [ 0.7 ]x            │ │
│  │                                                             │ │
│  │  Applied to: [✓] ED Bay  [✓] Diagnostics  [ ] Ward        │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Results Display

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Cost Summary                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total Cost: £1,847,230                                        │
│  Patients: 1,298                                               │
│  Cost per Patient: £1,423 (mean)                               │
│                                                                 │
│  ┌─ By Location ──────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  ED Bays     ████████████░░░░░░░░░░░░░░   £342,000 (18.5%) │ │
│  │  Theatre     ██████░░░░░░░░░░░░░░░░░░░░   £287,000 (15.5%) │ │
│  │  ITU         ████████████████░░░░░░░░░░   £456,000 (24.7%) │ │
│  │  Ward        ██████████████████████░░░░   £612,000 (33.1%) │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ By Transport ─────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Ambulance   ████████████████████████░░   £89,000  (59.3%) │ │
│  │  HEMS        ████████████░░░░░░░░░░░░░░   £42,000  (28.0%) │ │
│  │  Fixed-Wing  ████░░░░░░░░░░░░░░░░░░░░░░   £19,230  (12.8%) │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─ By Priority ──────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Priority │ Patients │ Total Cost  │ Per Patient           │ │
│  │  ─────────┼──────────┼─────────────┼────────────           │ │
│  │  P1       │    87    │  £368,010   │  £4,230               │ │
│  │  P2       │   298    │  £563,220   │  £1,890               │ │
│  │  P3       │   612    │  £599,760   │  £980                 │ │
│  │  P4       │   301    │  £126,420   │  £420                 │ │
│  │                                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  [Export CSV]  [Export PDF]                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### Where Cost Module Connects

```
┌─────────────────────────────────────────────────────────────────┐
│                     INTEGRATION POINTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. RESULTS PROCESSING (Post-simulation)                        │
│     └── After run_simulation() returns                         │
│     └── Calculate costs from SimulationResults                 │
│     └── Add to results object or return separately             │
│                                                                 │
│  2. UI - RESULTS TAB                                            │
│     └── Show cost breakdown alongside operational metrics       │
│     └── Cost config in sidebar or dedicated panel              │
│                                                                 │
│  3. SCENARIO COMPARISON                                         │
│     └── Compare costs between two scenarios                    │
│     └── "Scenario B costs £X more/less than A"                 │
│                                                                 │
│  4. EXPORT                                                      │
│     └── Include cost breakdown in CSV/PDF exports              │
│     └── Financial summary for planning documents               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Usage Pattern

```python
# In experiment runner or UI
results = run_simulation(scenario)

if scenario.cost_config.enabled:
    costs = calculate_costs(results, scenario.cost_config)
    costs_by_priority = calculate_costs_by_priority(results, scenario.cost_config)
    
    # Add to results or display
    display_cost_summary(costs, costs_by_priority)
```

---

## Default Rate Sources

For reference when users ask about default values:

| Rate | Default | Source/Basis |
|------|---------|--------------|
| ED Bay/day | £500 | NHS Reference Costs 2022/23, ED attendance |
| ITU Bed/day | £2,000 | NHS Critical Care costs |
| Ward Bed/day | £400 | NHS General ward costs |
| Theatre/hour | £2,000 | NHS Operating theatre costs |
| Ambulance | £275 | NHS See & Treat + Convey average |
| HEMS | £3,500 | Air ambulance charity data |
| Fixed-wing | £15,000 | Military aeromed estimates |

Users should adjust these to their local context.

---

## Test Cases

```python
def test_cost_calculation_basic():
    """Basic cost calculation produces expected output."""
    patient = Patient(
        id="test",
        priority=2,
        ed_start=0,
        ed_end=120,  # 2 hours = 0.083 days
        arrival_mode="AMBULANCE",
        outcome="DISCHARGE_ED",
    )
    results = SimulationResults(patients=[patient])
    config = CostConfig()  # Default rates
    
    costs = calculate_costs(results, config)
    
    # ED: 0.083 days * £500 = £41.67
    # Triage: £20
    # Diagnostics: £75 * 1.4 (P2) = £105
    # Ambulance: £275
    # Discharge: £40
    # Total: ~£481.67
    
    assert 480 < costs.grand_total < 485


def test_priority_multipliers():
    """Priority multipliers adjust costs correctly."""
    p1_patient = Patient(id="p1", priority=1, ed_start=0, ed_end=60)
    p4_patient = Patient(id="p4", priority=4, ed_start=0, ed_end=60)
    
    config = CostConfig()
    
    p1_cost = calculate_patient_cost(p1_patient, config)
    p4_cost = calculate_patient_cost(p4_patient, config)
    
    # P1 should cost more than P4 due to multipliers
    assert p1_cost > p4_cost


def test_aeromed_costs():
    """Aeromed evacuations add transport costs."""
    patient = Patient(
        id="test",
        priority=1,
        ed_start=0, ed_end=60,
        ward_start=60, ward_end=180,
        aeromed_type="FIXED_WING",
    )
    
    results = SimulationResults(patients=[patient])
    config = CostConfig()
    
    costs = calculate_costs(results, config)
    
    # Should include £15,000 fixed-wing cost
    assert costs.fixedwing_costs == 15000
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `faer/analysis/costs.py` | CREATE | CostConfig, calculation functions |
| `faer/ui/panels/cost_panel.py` | CREATE | Cost configuration UI |
| `faer/ui/components/cost_display.py` | CREATE | Cost results display |
| `faer/experiment/runner.py` | MODIFY | Integrate cost calculation |
| `tests/unit/test_costs.py` | CREATE | Unit tests |

---

## Implementation Order

1. **CostConfig dataclass** - All rate fields with sensible defaults
2. **calculate_costs()** - Main calculation from results
3. **calculate_patient_cost()** - Per-patient calculation
4. **calculate_costs_by_priority()** - Priority breakdown
5. **Cost configuration UI panel**
6. **Cost results display component**
7. **Integration with runner/UI**
8. **Scenario comparison helper**
9. **Tests**

---

## Dependencies

- Phase 9 (Downstream) - Needs ward/ITU/theatre timing for full costing
- Phase 10 (Aeromed) - Needs aeromed fields for transport costs
- No hard blockers - can implement with partial data

---

## Notes for Implementation

- Keep cost calculation PURE - no side effects, easy to test
- All times in simulation are minutes - convert to days/hours for rates
- Currency symbol lookup: `{"GBP": "£", "USD": "$", "EUR": "€"}`
- Consider rounding for display (2 decimal places for totals)
- Export should preserve full precision
