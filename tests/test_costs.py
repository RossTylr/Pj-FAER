"""Unit tests for cost modelling module."""

import pytest
from faer.results.costs import (
    CostConfig,
    CostBreakdown,
    CostByPriority,
    calculate_costs,
    calculate_patient_cost,
    calculate_costs_by_priority,
    compare_scenario_costs,
    get_priority_multiplier,
    format_currency,
    CURRENCY_SYMBOLS,
)
from faer.model.patient import Patient, Acuity, Disposition
from faer.core.entities import Priority, ArrivalMode


def create_test_patient(
    patient_id: int = 1,
    priority: Priority = Priority.P3_URGENT,
    mode: ArrivalMode = ArrivalMode.AMBULANCE,
    treatment_start: float = 10.0,
    treatment_end: float = 70.0,  # 60 mins = 1 hour = 0.042 days
    boarding_start: float = None,
    boarding_end: float = None,
    surgery_start: float = None,
    surgery_end: float = None,
    itu_start: float = None,
    itu_end: float = None,
    ward_start: float = None,
    ward_end: float = None,
    disposition: Disposition = Disposition.DISCHARGE,
    requires_aeromed: bool = False,
    aeromed_type: str = None,
    requires_transfer: bool = False,
    transfer_departed_time: float = None,
) -> Patient:
    """Create a test patient with specified parameters."""
    patient = Patient(
        id=patient_id,
        arrival_time=0.0,
        acuity=Acuity.MAJORS,
        priority=priority,
        mode=mode,
    )
    patient.treatment_start = treatment_start
    patient.treatment_end = treatment_end
    patient.boarding_start = boarding_start
    patient.boarding_end = boarding_end
    patient.surgery_start = surgery_start
    patient.surgery_end = surgery_end
    patient.itu_start = itu_start
    patient.itu_end = itu_end
    patient.ward_start = ward_start
    patient.ward_end = ward_end
    patient.disposition = disposition
    patient.requires_aeromed = requires_aeromed
    patient.aeromed_type = aeromed_type
    patient.requires_transfer = requires_transfer
    patient.transfer_departed_time = transfer_departed_time
    patient.departure_time = 100.0  # Set departure time
    return patient


class TestCostConfig:
    """Tests for CostConfig dataclass."""

    def test_default_values(self):
        """CostConfig has sensible defaults."""
        config = CostConfig()

        assert config.enabled is True
        assert config.currency == "GBP"
        assert config.ed_bay_per_day == 500.0
        assert config.itu_bed_per_day == 2000.0
        assert config.ward_bed_per_day == 400.0
        assert config.theatre_per_hour == 2000.0
        assert config.triage_cost == 20.0
        assert config.ambulance_per_journey == 275.0
        assert config.hems_per_flight == 3500.0

    def test_currency_symbol(self):
        """get_currency_symbol returns correct symbols."""
        config_gbp = CostConfig(currency="GBP")
        config_usd = CostConfig(currency="USD")
        config_eur = CostConfig(currency="EUR")

        assert config_gbp.get_currency_symbol() == "£"
        assert config_usd.get_currency_symbol() == "$"
        assert config_eur.get_currency_symbol() == "€"

    def test_priority_multipliers(self):
        """Default priority multipliers are set correctly."""
        config = CostConfig()

        assert config.priority_multipliers["P1"] == 2.0
        assert config.priority_multipliers["P2"] == 1.4
        assert config.priority_multipliers["P3"] == 1.0
        assert config.priority_multipliers["P4"] == 0.7


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_total_calculations(self):
        """Property calculations work correctly."""
        breakdown = CostBreakdown(
            currency="GBP",
            ed_bay_costs=1000.0,
            itu_bed_costs=2000.0,
            ward_bed_costs=500.0,
            theatre_costs=3000.0,
            triage_costs=100.0,
            diagnostics_costs=200.0,
            discharge_costs=50.0,
            ambulance_costs=500.0,
            hems_costs=3500.0,
            total_patients=10,
        )

        assert breakdown.total_bed_costs == 3500.0  # 1000 + 2000 + 500
        assert breakdown.total_transport_costs == 4000.0  # 500 + 3500
        assert breakdown.total_episode_costs == 350.0  # 100 + 200 + 50
        assert breakdown.grand_total == 10850.0  # 3500 + 3000 + 4000 + 350
        assert breakdown.cost_per_patient == 1085.0  # 10850 / 10

    def test_cost_per_patient_zero_patients(self):
        """cost_per_patient returns 0 when no patients."""
        breakdown = CostBreakdown(currency="GBP", total_patients=0)
        assert breakdown.cost_per_patient == 0.0

    def test_to_dict(self):
        """to_dict serializes correctly."""
        breakdown = CostBreakdown(
            currency="GBP",
            ed_bay_costs=100.0,
            total_patients=5,
        )
        result = breakdown.to_dict()

        assert result["currency"] == "GBP"
        assert result["ed_bay_costs"] == 100.0
        assert result["total_patients"] == 5
        assert "grand_total" in result
        assert "cost_per_patient" in result


class TestGetPriorityMultiplier:
    """Tests for priority multiplier logic."""

    def test_multiplier_applied_to_configured_categories(self):
        """Multiplier applies to categories in apply_priority_to list."""
        config = CostConfig()
        p1_patient = create_test_patient(priority=Priority.P1_IMMEDIATE)
        p4_patient = create_test_patient(priority=Priority.P4_STANDARD)

        # ed_bay is in apply_priority_to by default
        assert get_priority_multiplier(p1_patient, config, "ed_bay") == 2.0
        assert get_priority_multiplier(p4_patient, config, "ed_bay") == 0.7

    def test_multiplier_not_applied_to_unconfigured_categories(self):
        """Categories not in apply_priority_to get multiplier of 1.0."""
        config = CostConfig()
        p1_patient = create_test_patient(priority=Priority.P1_IMMEDIATE)

        # itu is not in apply_priority_to by default
        assert get_priority_multiplier(p1_patient, config, "itu") == 1.0

    def test_empty_apply_priority_to(self):
        """Empty apply_priority_to means no multipliers applied."""
        config = CostConfig(apply_priority_to=[])
        p1_patient = create_test_patient(priority=Priority.P1_IMMEDIATE)

        assert get_priority_multiplier(p1_patient, config, "ed_bay") == 1.0
        assert get_priority_multiplier(p1_patient, config, "diagnostics") == 1.0


class TestCalculatePatientCost:
    """Tests for individual patient cost calculation."""

    def test_basic_patient_cost(self):
        """Basic patient with ED time incurs expected costs."""
        config = CostConfig()
        # 60 mins ED = 0.0417 days * £500/day = £20.83
        # + triage £20 + diagnostics £75 + ambulance £275 + discharge £40
        patient = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,  # 60 mins = 0.0417 days
        )

        cost = calculate_patient_cost(patient, config)

        # Expected: ~£20.83 (ED) + £20 (triage) + £75 (diag) + £275 (amb) + £40 (disch)
        # = ~£430.83
        assert 400 < cost < 460  # Allow for rounding

    def test_p1_patient_costs_more(self):
        """P1 patient costs more than P4 due to multipliers."""
        config = CostConfig()
        p1_patient = create_test_patient(
            priority=Priority.P1_IMMEDIATE,
            treatment_start=0.0,
            treatment_end=60.0,
        )
        p4_patient = create_test_patient(
            priority=Priority.P4_STANDARD,
            treatment_start=0.0,
            treatment_end=60.0,
        )

        p1_cost = calculate_patient_cost(p1_patient, config)
        p4_cost = calculate_patient_cost(p4_patient, config)

        assert p1_cost > p4_cost

    def test_hems_arrival_costs(self):
        """HEMS arrival adds helicopter cost."""
        config = CostConfig()
        hems_patient = create_test_patient(
            mode=ArrivalMode.HELICOPTER,
            treatment_start=0.0,
            treatment_end=60.0,
        )
        ambulance_patient = create_test_patient(
            mode=ArrivalMode.AMBULANCE,
            treatment_start=0.0,
            treatment_end=60.0,
        )

        hems_cost = calculate_patient_cost(hems_patient, config)
        amb_cost = calculate_patient_cost(ambulance_patient, config)

        # HEMS costs £3500 vs ambulance £275
        assert hems_cost > amb_cost
        assert hems_cost - amb_cost == pytest.approx(3500 - 275, abs=1)

    def test_surgery_costs(self):
        """Surgery adds theatre hourly costs."""
        config = CostConfig()
        # 2 hour surgery = £4000
        surgical_patient = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
            surgery_start=60.0,
            surgery_end=180.0,  # 2 hours
        )
        non_surgical = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
        )

        surgical_cost = calculate_patient_cost(surgical_patient, config)
        non_surgical_cost = calculate_patient_cost(non_surgical, config)

        # Difference should be 2 hours * £2000/hr = £4000
        assert surgical_cost - non_surgical_cost == pytest.approx(4000, abs=10)

    def test_itu_costs(self):
        """ITU stay adds bed-day costs."""
        config = CostConfig()
        # 1 day ITU = £2000
        itu_patient = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
            itu_start=60.0,
            itu_end=60.0 + 1440.0,  # 1 day
        )
        non_itu = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
        )

        itu_cost = calculate_patient_cost(itu_patient, config)
        non_itu_cost = calculate_patient_cost(non_itu, config)

        # Difference should be 1 day * £2000/day = £2000
        assert itu_cost - non_itu_cost == pytest.approx(2000, abs=10)

    def test_aeromed_evacuation_costs(self):
        """Aeromed evacuation adds transport costs."""
        config = CostConfig()

        hems_evac = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
            requires_aeromed=True,
            aeromed_type="HEMS",
        )
        fw_evac = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
            requires_aeromed=True,
            aeromed_type="FIXED_WING",
        )
        no_evac = create_test_patient(
            treatment_start=0.0,
            treatment_end=60.0,
        )

        hems_cost = calculate_patient_cost(hems_evac, config)
        fw_cost = calculate_patient_cost(fw_evac, config)
        base_cost = calculate_patient_cost(no_evac, config)

        assert hems_cost - base_cost == pytest.approx(3500, abs=10)
        assert fw_cost - base_cost == pytest.approx(15000, abs=10)


class TestCalculateCosts:
    """Tests for aggregate cost calculation."""

    def test_empty_patient_list(self):
        """Empty patient list returns zero costs."""
        config = CostConfig()
        breakdown = calculate_costs([], config)

        assert breakdown.grand_total == 0.0
        assert breakdown.total_patients == 0

    def test_multiple_patients(self):
        """Costs aggregate correctly for multiple patients."""
        config = CostConfig()
        patients = [
            create_test_patient(patient_id=i, treatment_start=0.0, treatment_end=60.0)
            for i in range(10)
        ]

        breakdown = calculate_costs(patients, config)

        assert breakdown.total_patients == 10
        # 10 patients * £20 triage = £200
        assert breakdown.triage_costs == 200.0
        # 10 patients * £275 ambulance = £2750
        assert breakdown.ambulance_costs == 2750.0

    def test_cost_breakdown_by_location(self):
        """Costs correctly attributed to different locations."""
        config = CostConfig()

        patients = [
            # ED only patient
            create_test_patient(
                patient_id=1,
                treatment_start=0.0,
                treatment_end=1440.0,  # 1 day ED
            ),
            # ITU patient
            create_test_patient(
                patient_id=2,
                treatment_start=0.0,
                treatment_end=60.0,
                itu_start=60.0,
                itu_end=60.0 + 1440.0,  # 1 day ITU
            ),
            # Ward patient
            create_test_patient(
                patient_id=3,
                treatment_start=0.0,
                treatment_end=60.0,
                ward_start=60.0,
                ward_end=60.0 + 1440.0,  # 1 day ward
            ),
        ]

        breakdown = calculate_costs(patients, config)

        # Check ITU and ward costs are present
        assert breakdown.itu_bed_costs > 0
        assert breakdown.ward_bed_costs > 0


class TestCalculateCostsByPriority:
    """Tests for priority-based cost breakdown."""

    def test_priority_breakdown(self):
        """Costs correctly attributed by priority."""
        config = CostConfig()

        patients = [
            create_test_patient(patient_id=1, priority=Priority.P1_IMMEDIATE),
            create_test_patient(patient_id=2, priority=Priority.P1_IMMEDIATE),
            create_test_patient(patient_id=3, priority=Priority.P2_VERY_URGENT),
            create_test_patient(patient_id=4, priority=Priority.P3_URGENT),
            create_test_patient(patient_id=5, priority=Priority.P4_STANDARD),
        ]

        by_priority = calculate_costs_by_priority(patients, config)

        assert by_priority.p1_count == 2
        assert by_priority.p2_count == 1
        assert by_priority.p3_count == 1
        assert by_priority.p4_count == 1

    def test_per_patient_costs_by_priority(self):
        """Per-patient costs calculated correctly."""
        config = CostConfig()

        patients = [
            create_test_patient(patient_id=1, priority=Priority.P1_IMMEDIATE),
            create_test_patient(patient_id=2, priority=Priority.P4_STANDARD),
        ]

        by_priority = calculate_costs_by_priority(patients, config)

        # P1 should cost more per patient than P4
        assert by_priority.p1_per_patient > by_priority.p4_per_patient


class TestCompareScenarioCosts:
    """Tests for scenario comparison."""

    def test_basic_comparison(self):
        """Compare two scenarios correctly."""
        config = CostConfig()

        patients_a = [
            create_test_patient(patient_id=i, treatment_start=0.0, treatment_end=60.0)
            for i in range(5)
        ]
        patients_b = [
            create_test_patient(patient_id=i, treatment_start=0.0, treatment_end=120.0)
            for i in range(10)
        ]

        comparison = compare_scenario_costs(
            patients_a, patients_b, config,
            labels=("Small", "Large")
        )

        assert comparison["labels"] == ("Small", "Large")
        assert comparison["patients_a"] == 5
        assert comparison["patients_b"] == 10
        assert comparison["total_b"] > comparison["total_a"]
        assert "difference" in comparison
        assert "difference_pct" in comparison


class TestFormatCurrency:
    """Tests for currency formatting."""

    def test_gbp_formatting(self):
        """GBP formats with £ symbol."""
        assert format_currency(1234, "£") == "£1,234"
        assert format_currency(1234567, "£") == "£1,234,567"

    def test_decimal_places(self):
        """Decimal places work correctly."""
        assert format_currency(1234.56, "£", decimals=2) == "£1,234.56"
        assert format_currency(1234.567, "£", decimals=2) == "£1,234.57"

    def test_different_currencies(self):
        """Different currency symbols work."""
        assert format_currency(1000, "$") == "$1,000"
        assert format_currency(1000, "€") == "€1,000"


class TestCurrencySymbols:
    """Tests for currency symbol constants."""

    def test_symbols_defined(self):
        """All expected currency symbols are defined."""
        assert "GBP" in CURRENCY_SYMBOLS
        assert "USD" in CURRENCY_SYMBOLS
        assert "EUR" in CURRENCY_SYMBOLS
        assert CURRENCY_SYMBOLS["GBP"] == "£"
        assert CURRENCY_SYMBOLS["USD"] == "$"
        assert CURRENCY_SYMBOLS["EUR"] == "€"
