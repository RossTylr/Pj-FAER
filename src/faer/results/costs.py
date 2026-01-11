"""Cost modelling module for post-hoc financial analysis.

This module calculates financial costs from simulation outputs using
simple bed-day rates. Cost modelling is POST-HOC calculation - it does
not change simulation behaviour, only interprets results financially.

Key features:
- Configurable bed-day and per-episode rates
- Multi-currency support (GBP, USD, EUR)
- Cost breakdown by location, priority, transport mode
- Priority-based cost multipliers
- Scenario cost comparison
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from faer.model.patient import Patient


# Currency symbols for display
CURRENCY_SYMBOLS = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
}


@dataclass
class CostConfig:
    """Cost modelling configuration.

    All rates are per-unit costs. Bed-day rates are per 24-hour period.
    Theatre rate is per hour (procedures vary in length).

    Default rates are based on NHS Reference Costs 2022/23 and similar
    sources. Users should adjust to their local context.
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

    def get_currency_symbol(self) -> str:
        """Get the currency symbol for display."""
        return CURRENCY_SYMBOLS.get(self.currency, self.currency)


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

    # Totals computed from aggregates
    total_patients: int = 0

    @property
    def total_bed_costs(self) -> float:
        """Total bed-related costs (ED + ITU + Ward)."""
        return self.ed_bay_costs + self.itu_bed_costs + self.ward_bed_costs

    @property
    def total_transport_costs(self) -> float:
        """Total transport costs."""
        return (
            self.ambulance_costs
            + self.hems_costs
            + self.fixedwing_costs
            + self.road_transfer_costs
        )

    @property
    def total_episode_costs(self) -> float:
        """Total per-episode costs."""
        return self.triage_costs + self.diagnostics_costs + self.discharge_costs

    @property
    def grand_total(self) -> float:
        """Grand total of all costs."""
        return (
            self.total_bed_costs
            + self.theatre_costs
            + self.total_transport_costs
            + self.total_episode_costs
        )

    @property
    def cost_per_patient(self) -> float:
        """Mean cost per patient."""
        if self.total_patients == 0:
            return 0.0
        return self.grand_total / self.total_patients

    def get_currency_symbol(self) -> str:
        """Get the currency symbol for display."""
        return CURRENCY_SYMBOLS.get(self.currency, self.currency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "currency": self.currency,
            "ed_bay_costs": self.ed_bay_costs,
            "theatre_costs": self.theatre_costs,
            "itu_bed_costs": self.itu_bed_costs,
            "ward_bed_costs": self.ward_bed_costs,
            "triage_costs": self.triage_costs,
            "diagnostics_costs": self.diagnostics_costs,
            "discharge_costs": self.discharge_costs,
            "ambulance_costs": self.ambulance_costs,
            "hems_costs": self.hems_costs,
            "fixedwing_costs": self.fixedwing_costs,
            "road_transfer_costs": self.road_transfer_costs,
            "total_patients": self.total_patients,
            "total_bed_costs": self.total_bed_costs,
            "total_transport_costs": self.total_transport_costs,
            "total_episode_costs": self.total_episode_costs,
            "grand_total": self.grand_total,
            "cost_per_patient": self.cost_per_patient,
        }


@dataclass
class CostByPriority:
    """Cost breakdown by patient priority."""

    currency: str = "GBP"

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
        """Mean cost per P1 patient."""
        return self.p1_total / self.p1_count if self.p1_count > 0 else 0.0

    @property
    def p2_per_patient(self) -> float:
        """Mean cost per P2 patient."""
        return self.p2_total / self.p2_count if self.p2_count > 0 else 0.0

    @property
    def p3_per_patient(self) -> float:
        """Mean cost per P3 patient."""
        return self.p3_total / self.p3_count if self.p3_count > 0 else 0.0

    @property
    def p4_per_patient(self) -> float:
        """Mean cost per P4 patient."""
        return self.p4_total / self.p4_count if self.p4_count > 0 else 0.0

    def get_currency_symbol(self) -> str:
        """Get the currency symbol for display."""
        return CURRENCY_SYMBOLS.get(self.currency, self.currency)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "currency": self.currency,
            "p1_total": self.p1_total,
            "p1_count": self.p1_count,
            "p1_per_patient": self.p1_per_patient,
            "p2_total": self.p2_total,
            "p2_count": self.p2_count,
            "p2_per_patient": self.p2_per_patient,
            "p3_total": self.p3_total,
            "p3_count": self.p3_count,
            "p3_per_patient": self.p3_per_patient,
            "p4_total": self.p4_total,
            "p4_count": self.p4_count,
            "p4_per_patient": self.p4_per_patient,
        }


def get_priority_multiplier(
    patient: "Patient",
    config: CostConfig,
    cost_category: str,
) -> float:
    """Get priority-based cost multiplier for a category.

    Args:
        patient: Patient with priority attribute.
        config: Cost configuration with multipliers.
        cost_category: The cost category (e.g., "ed_bay", "diagnostics").

    Returns:
        Multiplier to apply to the cost (1.0 if not applicable).
    """
    if cost_category not in config.apply_priority_to:
        return 1.0

    # Get priority key (P1, P2, P3, P4)
    priority_key = f"P{patient.priority.value}"
    return config.priority_multipliers.get(priority_key, 1.0)


def calculate_patient_cost(patient: "Patient", config: CostConfig) -> float:
    """Calculate total cost for a single patient.

    Args:
        patient: Completed patient with timing data.
        config: Cost configuration.

    Returns:
        Total cost for this patient in configured currency.
    """
    from faer.model.patient import Disposition
    from faer.core.entities import ArrivalMode

    cost = 0.0

    # Triage (everyone gets triaged)
    cost += config.triage_cost

    # ED time - use treatment timestamps as proxy for ED bay time
    # treatment_start to treatment_end represents core ED time
    if patient.treatment_start is not None and patient.treatment_end is not None:
        ed_mins = patient.treatment_end - patient.treatment_start
        ed_days = ed_mins / 1440  # Convert minutes to days
        multiplier = get_priority_multiplier(patient, config, "ed_bay")
        cost += ed_days * config.ed_bay_per_day * multiplier

    # Include boarding time as ED bay cost (patient still occupying bay)
    if patient.boarding_start is not None and patient.boarding_end is not None:
        boarding_mins = patient.boarding_end - patient.boarding_start
        boarding_days = boarding_mins / 1440
        multiplier = get_priority_multiplier(patient, config, "ed_bay")
        cost += boarding_days * config.ed_bay_per_day * multiplier

    # Theatre time (hourly rate)
    if patient.surgery_start is not None and patient.surgery_end is not None:
        theatre_mins = patient.surgery_end - patient.surgery_start
        theatre_hours = theatre_mins / 60
        cost += theatre_hours * config.theatre_per_hour

    # ITU time
    if patient.itu_start is not None and patient.itu_end is not None:
        itu_mins = patient.itu_end - patient.itu_start
        itu_days = itu_mins / 1440
        cost += itu_days * config.itu_bed_per_day

    # Ward time
    if patient.ward_start is not None and patient.ward_end is not None:
        ward_mins = patient.ward_end - patient.ward_start
        ward_days = ward_mins / 1440
        cost += ward_days * config.ward_bed_per_day

    # Diagnostics - base cost with priority multiplier
    # Check if patient had any diagnostics
    if patient.diagnostics_completed:
        multiplier = get_priority_multiplier(patient, config, "diagnostics")
        # Cost per diagnostic type completed
        cost += len(patient.diagnostics_completed) * config.diagnostics_base_cost * multiplier
    else:
        # Default: assume everyone gets some basic diagnostics
        multiplier = get_priority_multiplier(patient, config, "diagnostics")
        cost += config.diagnostics_base_cost * multiplier

    # Transport (arrival)
    if patient.mode == ArrivalMode.AMBULANCE:
        cost += config.ambulance_per_journey
    elif patient.mode == ArrivalMode.HELICOPTER:
        cost += config.hems_per_flight

    # Transport (aeromed evacuation)
    if patient.requires_aeromed and patient.aeromed_type:
        if patient.aeromed_type == "HEMS":
            cost += config.hems_per_flight
        elif patient.aeromed_type == "FIXED_WING":
            cost += config.fixedwing_per_flight

    # Road transfer (inter-facility)
    if patient.requires_transfer and patient.transfer_departed_time is not None:
        # Patient was transferred
        cost += config.road_transfer_per_journey

    # Discharge costs for patients who completed journey
    if patient.disposition is not None:
        if patient.disposition == Disposition.DISCHARGE:
            cost += config.discharge_cost
        elif patient.disposition in (Disposition.ADMIT_WARD, Disposition.ADMIT_ICU):
            # Admitted patients also incur discharge costs when they leave
            cost += config.discharge_cost

    return cost


def calculate_costs(
    patients: List["Patient"],
    config: CostConfig,
) -> CostBreakdown:
    """Calculate costs from simulation results.

    This is a POST-HOC calculation - it processes completed results,
    not live simulation state.

    Args:
        patients: List of completed Patient objects from simulation.
        config: Cost configuration with rates.

    Returns:
        CostBreakdown with detailed cost breakdown.
    """
    from faer.model.patient import Disposition
    from faer.core.entities import ArrivalMode

    breakdown = CostBreakdown(currency=config.currency)
    breakdown.total_patients = len(patients)

    if not patients:
        return breakdown

    # === BED-DAY COSTS ===

    # ED bay time (treatment + boarding)
    total_ed_mins = 0.0
    for p in patients:
        if p.treatment_start is not None and p.treatment_end is not None:
            multiplier = get_priority_multiplier(p, config, "ed_bay")
            ed_mins = p.treatment_end - p.treatment_start
            total_ed_mins += ed_mins * multiplier

        if p.boarding_start is not None and p.boarding_end is not None:
            multiplier = get_priority_multiplier(p, config, "ed_bay")
            boarding_mins = p.boarding_end - p.boarding_start
            total_ed_mins += boarding_mins * multiplier

    breakdown.ed_bay_costs = (total_ed_mins / 1440) * config.ed_bay_per_day

    # ITU bed time
    total_itu_mins = sum(
        (p.itu_end - p.itu_start)
        for p in patients
        if p.itu_start is not None and p.itu_end is not None
    )
    breakdown.itu_bed_costs = (total_itu_mins / 1440) * config.itu_bed_per_day

    # Ward bed time
    total_ward_mins = sum(
        (p.ward_end - p.ward_start)
        for p in patients
        if p.ward_start is not None and p.ward_end is not None
    )
    breakdown.ward_bed_costs = (total_ward_mins / 1440) * config.ward_bed_per_day

    # === THEATRE COSTS (HOURLY) ===

    total_theatre_mins = sum(
        (p.surgery_end - p.surgery_start)
        for p in patients
        if p.surgery_start is not None and p.surgery_end is not None
    )
    breakdown.theatre_costs = (total_theatre_mins / 60) * config.theatre_per_hour

    # === PER-EPISODE COSTS ===

    # Every patient gets triaged
    breakdown.triage_costs = len(patients) * config.triage_cost

    # Diagnostics (with optional priority multiplier)
    for patient in patients:
        multiplier = get_priority_multiplier(patient, config, "diagnostics")
        if patient.diagnostics_completed:
            breakdown.diagnostics_costs += (
                len(patient.diagnostics_completed) * config.diagnostics_base_cost * multiplier
            )
        else:
            # Default baseline diagnostics
            breakdown.diagnostics_costs += config.diagnostics_base_cost * multiplier

    # Discharge costs for patients who completed journey
    discharged = [
        p for p in patients
        if p.disposition is not None and p.disposition in (
            Disposition.DISCHARGE, Disposition.ADMIT_WARD, Disposition.ADMIT_ICU
        )
    ]
    breakdown.discharge_costs = len(discharged) * config.discharge_cost

    # === TRANSPORT COSTS ===

    # Ambulance arrivals
    ambulance_arrivals = [p for p in patients if p.mode == ArrivalMode.AMBULANCE]
    breakdown.ambulance_costs = len(ambulance_arrivals) * config.ambulance_per_journey

    # HEMS arrivals + HEMS evacuations
    hems_arrivals = [p for p in patients if p.mode == ArrivalMode.HELICOPTER]
    hems_evacuations = [
        p for p in patients
        if p.requires_aeromed and p.aeromed_type == "HEMS"
    ]
    breakdown.hems_costs = (
        len(hems_arrivals) + len(hems_evacuations)
    ) * config.hems_per_flight

    # Fixed-wing evacuations
    fw_evacuations = [
        p for p in patients
        if p.requires_aeromed and p.aeromed_type == "FIXED_WING"
    ]
    breakdown.fixedwing_costs = len(fw_evacuations) * config.fixedwing_per_flight

    # Road transfers
    road_transfers = [
        p for p in patients
        if p.requires_transfer and p.transfer_departed_time is not None
    ]
    breakdown.road_transfer_costs = len(road_transfers) * config.road_transfer_per_journey

    return breakdown


def calculate_costs_by_priority(
    patients: List["Patient"],
    config: CostConfig,
) -> CostByPriority:
    """Calculate cost breakdown by patient priority.

    Args:
        patients: List of completed Patient objects.
        config: Cost configuration.

    Returns:
        CostByPriority with breakdown by P1-P4.
    """
    by_priority = CostByPriority(currency=config.currency)

    for patient in patients:
        patient_cost = calculate_patient_cost(patient, config)
        priority_value = patient.priority.value

        if priority_value == 1:
            by_priority.p1_total += patient_cost
            by_priority.p1_count += 1
        elif priority_value == 2:
            by_priority.p2_total += patient_cost
            by_priority.p2_count += 1
        elif priority_value == 3:
            by_priority.p3_total += patient_cost
            by_priority.p3_count += 1
        elif priority_value == 4:
            by_priority.p4_total += patient_cost
            by_priority.p4_count += 1

    return by_priority


def compare_scenario_costs(
    patients_a: List["Patient"],
    patients_b: List["Patient"],
    config: CostConfig,
    labels: Tuple[str, str] = ("Scenario A", "Scenario B"),
) -> Dict[str, Any]:
    """Compare costs between two scenarios.

    Args:
        patients_a: Patients from first scenario.
        patients_b: Patients from second scenario.
        config: Cost configuration (same for both).
        labels: Display labels for the scenarios.

    Returns:
        Dictionary with comparison metrics.
    """
    costs_a = calculate_costs(patients_a, config)
    costs_b = calculate_costs(patients_b, config)

    difference = costs_b.grand_total - costs_a.grand_total
    difference_pct = (
        (difference / costs_a.grand_total * 100)
        if costs_a.grand_total > 0
        else 0.0
    )

    return {
        "labels": labels,
        "currency": config.currency,
        "currency_symbol": config.get_currency_symbol(),
        # Totals
        "total_a": costs_a.grand_total,
        "total_b": costs_b.grand_total,
        "difference": difference,
        "difference_pct": difference_pct,
        # Per patient
        "per_patient_a": costs_a.cost_per_patient,
        "per_patient_b": costs_b.cost_per_patient,
        # Patient counts
        "patients_a": costs_a.total_patients,
        "patients_b": costs_b.total_patients,
        # Category breakdown
        "bed_costs_a": costs_a.total_bed_costs,
        "bed_costs_b": costs_b.total_bed_costs,
        "transport_costs_a": costs_a.total_transport_costs,
        "transport_costs_b": costs_b.total_transport_costs,
        # Full breakdowns
        "breakdown_a": costs_a,
        "breakdown_b": costs_b,
    }


def format_currency(value: float, symbol: str = "£", decimals: int = 0) -> str:
    """Format a value as currency string.

    Args:
        value: The monetary value.
        symbol: Currency symbol (default £).
        decimals: Decimal places (default 0 for whole numbers).

    Returns:
        Formatted currency string (e.g., "£1,234").
    """
    if decimals == 0:
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.{decimals}f}"
