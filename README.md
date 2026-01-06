<p align="center">
  <img src="assets/banner.jpeg" alt="Pj FAER Banner" width="100%">
</p>

# Pj FAER - Flow Analysis for Emergency Response

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pj-faer.streamlit.app)

A discrete-event simulation platform for hospital patient flow, built with SimPy and Streamlit.

## Overview

FAER simulates A&E → Resus → disposition pathways over 1-36 hour horizons, helping healthcare teams understand:

- Where congestion actually originates
- Which capacity investments have system-wide effects
- What fails first under surge conditions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Pj-FAER

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the Streamlit app
streamlit run app/Home.py

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=faer --cov-report=html
```

## Project Structure

```
pj_faer/
├── src/faer/           # Core simulation library
│   ├── core/           # Scenario config, arrivals, distributions
│   ├── model/          # SimPy patient processes
│   ├── results/        # Event logging and metrics
│   └── experiment/     # Replication runner, CI analysis
├── app/                # Streamlit web UI
├── tests/              # Pytest test suite
└── data/               # Arrival profiles
```

## Current Phase

**Phase 4: System Realism** - Full A&E pathway with realistic hospital flow.

### Phase 4 Features
- **Priority Queuing (P1-P4)**: Triage priority levels with `PriorityResource`
- **Simplified ED**: Single ED bay pool with priority-based service order
- **Multi-Stream Arrivals**: Ambulance, Helicopter, Walk-in with different acuity mixes
- **Handover Gate**: Ambulance/helicopter patients go through handover bays
- **Fleet Resources**: Finite ambulance/helicopter fleet with turnaround times
- **Downstream Nodes**: Surgery, ITU, Ward resources beyond ED

### Recent Changes (January 2026)
- Fixed `multiple_replications()` default metrics to include Phase 5 utilisation keys (`util_ed_bays`, `util_handover`, `util_ambulance_fleet`)
- Aligns runner output with Streamlit UI expectations in `2_Run.py` and `3_Results.py`

See [CLAUDE.md](CLAUDE.md) for development instructions and phase progression.

## Tech Stack

- **SimPy 4.1+** - Discrete-event simulation engine
- **sim-tools** - Tom Monks' DES utilities
- **Streamlit 1.30+** - Web UI
- **NumPy, Pandas, SciPy** - Numerical computing
- **Plotly** - Visualisation

## Attribution

This project uses patterns and concepts from:

- **sim-tools** by Monks, T., Heather, A., Harper, A. (2025) - [GitHub](https://github.com/TomMonks/sim-tools) - MIT License
- **STARS Project** by pythonhealthdatascience - MIT License
- **simpy-streamlit-tutorial** by health-data-science-OR - MIT License

## License

MIT License - see LICENSE file for details.
