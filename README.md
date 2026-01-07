<p align="center">
  <img src="assets/banner.jpeg" alt="Pj FAER Banner" width="100%">
</p>

# Pj FAER - Framework for Acute and Emergency Resources

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

**Phase 7: Diagnostics & Transfers** - Complete hospital flow with diagnostics and inter-facility transfers.

### Phase 7 Features
- **Diagnostic Nodes (CT, X-ray, Bloods)**: Patients queue for scans/tests with configurable capacity and turnaround times
- **Diagnostics Loop**: Patients keep their ED bay while going for diagnostics (realistic blocking behaviour)
- **Inter-Facility Transfers**: Land ambulance and helicopter transfers to specialist centres (Major Trauma, Neurosurgery, Cardiac, Burns, PICU)
- **Arrival Models UI**: Three configuration modes - Simple (demand slider), 24-Hour Profile (day type patterns), and Detailed (hourly breakdown by mode)

### Previous Phases
- **Phase 4-6**: Priority queuing (P1-P4), multi-stream arrivals, handover gate, fleet resources, downstream nodes (Surgery/ITU/Ward), experimentation framework

### Recent Changes (January 2026)
- Added CT scanner, X-ray, and Bloods diagnostic resources with priority-based probability
- Implemented diagnostic loop where patients hold ED bay during scans
- Added inter-facility transfer pathway with land ambulance and helicopter options
- Updated Streamlit UI with arrival model selector (Simple/24-Hour Profile/Detailed modes)
- Added diagnostic and transfer metrics tracking

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
