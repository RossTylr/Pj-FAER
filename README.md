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

## Features

- **Full A&E Pathway**: Triage → ED Bays → Theatre/ITU/Ward → Discharge
- **Priority Queuing**: P1-P4 triage levels with clinical priority handling
- **Multi-Stream Arrivals**: Ambulance, helicopter (HEMS), and walk-in patients
- **Diagnostics**: CT scanner, X-ray, and Bloods with realistic ED bay blocking
- **Aeromedical Evacuation**: HEMS and fixed-wing transfers to specialist centres
- **Interactive Schematic**: Real-time React/SVG visualisation of patient flow
- **Clinical Insights**: Heuristic-based agent that detects risk patterns (NHS thresholds)
- **Scenario Comparison**: Run multiple configurations and compare outcomes

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
Pj-FAER/
├── src/faer/              # Core simulation library
│   ├── core/              # Scenario config, arrivals, distributions
│   ├── model/             # SimPy patient processes
│   ├── results/           # Event logging and metrics
│   ├── experiment/        # Replication runner, CI analysis
│   └── agents/            # Clinical shadow agents for insights
├── app/                   # Streamlit web UI
│   ├── pages/             # App pages (Arrivals, Resources, Run, Results, etc.)
│   └── components/        # React schematic component
├── tests/                 # Pytest test suite
├── docs/                  # Documentation and PRDs
└── config/                # Configuration files
```

## App Pages

| Page | Description |
|------|-------------|
| **Arrivals** | Configure arrival rates (simple, 24-hour profile, or detailed) |
| **Resources** | Set capacity for ED bays, theatre, ITU, ward, diagnostics |
| **Schematic** | Interactive patient flow diagram |
| **Aeromed** | Configure aeromedical evacuation settings |
| **Run** | Execute simulation with progress tracking |
| **Results** | KPIs, utilisation metrics, and flow schematic |
| **Compare** | Side-by-side scenario comparison |
| **Sensitivity** | Parameter sensitivity analysis |
| **Insights** | Clinical shadow agent risk detection |

## Tech Stack

- **SimPy 4.1+** - Discrete-event simulation engine
- **sim-tools** - Tom Monks' DES utilities
- **Streamlit 1.30+** - Web UI
- **React/TypeScript** - Interactive schematic component
- **NumPy, Pandas, SciPy** - Numerical computing
- **Plotly** - Visualisation

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development instructions and coding standards
- [docs/PRD_AGENT_INTEGRATION.md](docs/PRD_AGENT_INTEGRATION.md) - Clinical agent integration specification

## Attribution

This project uses patterns and concepts from:

- **sim-tools** by Monks, T., Heather, A., Harper, A. (2025) - [GitHub](https://github.com/TomMonks/sim-tools) - MIT License
- **STARS Project** by pythonhealthdatascience - MIT License
- **simpy-streamlit-tutorial** by health-data-science-OR - MIT License

## License

MIT License - see LICENSE file for details.
