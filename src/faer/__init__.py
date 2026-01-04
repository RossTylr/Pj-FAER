"""
Pj FAER - Flow Analysis for Emergency Response.

A discrete-event simulation platform for hospital patient flow,
built with SimPy and Streamlit.
"""

__version__ = "0.1.0"

from faer.core.scenario import Scenario
from faer.model.processes import run_simulation

__all__ = ["Scenario", "run_simulation", "__version__"]
