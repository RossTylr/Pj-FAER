"""Pj FAER - Home page."""

import streamlit as st

st.set_page_config(
    page_title="Pj FAER",
    page_icon="",
    layout="wide",
)

st.image("assets/banner.jpeg", use_container_width=True)

st.title("Pj FAER: Hospital Flow Simulation")

st.markdown("""
## What is FAER?

**FAER** (Framework for Acute and Emergency Resources) is a discrete-event simulation
platform for understanding how patients flow through hospital systems over
**1-36 hour horizons**.

### What FAER reveals:
- Where congestion **actually** originates
- Which capacity investments have **system-wide** effects
- What fails first under surge conditions

### Current capabilities:
- A&E to Resus queue/service to disposition
- Non-stationary arrival patterns (time-of-day effects)
- Time-weighted utilisation metrics
- Replication-based confidence intervals
- CI-guided precision stopping

---

**Use the sidebar** to navigate:
1. **Scenario** - Configure simulation parameters
2. **Run** - Execute the simulation
3. **Results** - View KPIs and analysis
""")

st.info("""
**Attribution:** This simulation uses patterns from Tom Monks'
[sim-tools](https://github.com/TomMonks/sim-tools) and the STARS project.
""")

# Show quick stats if results exist
if "results" in st.session_state and st.session_state.get("run_complete"):
    st.success("Simulation complete! View results in the Results page.")
