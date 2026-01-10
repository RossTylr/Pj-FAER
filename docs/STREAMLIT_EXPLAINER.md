# Streamlit Explainer for Pj FAER

This document explains how Streamlit powers the FAER (Flow Analysis for Emergency Response) web interface, enabling users to configure, run, and analyze hospital patient flow simulations without writing code.

## What is Streamlit?

Streamlit is a Python framework that transforms Python scripts into interactive web applications. For FAER, it provides:

- **Interactive widgets** for configuring simulation parameters
- **Real-time visualization** of results with Plotly charts
- **Session persistence** to maintain state across pages
- **Rapid iteration** - changes to Python code instantly update the UI

## App Architecture

FAER uses Streamlit's **multi-page app** structure:

```
app/
├── Home.py                 # Landing page (entry point)
├── components/             # Reusable UI components
│   ├── schematic.py        # Graphviz flow diagrams
│   └── scenario_summary.py # Config display helpers
└── pages/                  # Navigation pages (auto-ordered by prefix)
    ├── 1_Arrivals.py       # Emergency services & arrival patterns
    ├── 2_Resources.py      # Hospital resource configuration
    ├── 3_Schematic.py      # Visual system diagram
    ├── 4_Run.py            # Execute simulation
    ├── 5_Results.py        # KPI analysis & export
    ├── 6_Compare.py        # Scenario comparison
    └── 7_Sensitivity.py    # Parameter sensitivity analysis
```

### How Multi-Page Apps Work

1. `Home.py` is the entry point (must be in `app/` root)
2. Files in `pages/` become sidebar navigation items
3. Numeric prefixes control ordering (`1_`, `2_`, etc.)
4. Each page is a standalone Python script that runs top-to-bottom

## Core Streamlit Concepts in FAER

### 1. Page Configuration

Every page starts with `st.set_page_config()`:

```python
# From app/pages/4_Run.py
st.set_page_config(
    page_title="Run Simulation",  # Browser tab title
    page_icon="▶️",               # Tab favicon
    layout="wide"                 # Full-width layout
)
```

### 2. Session State (Data Persistence)

Streamlit re-runs scripts on every interaction. `st.session_state` persists data between re-runs and across pages.

**Pattern: Initialize with Defaults**
```python
# From app/pages/1_Arrivals.py
def init_session_defaults():
    defaults = {
        'n_ambulances': 10,
        'ambulance_turnaround': 45.0,
        'arrival_model': 'profile_24h',
        # ... more defaults
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_defaults()
```

**Pattern: Two-Way Binding**
```python
# Widget reads from AND writes to session_state
n_ambulances = st.number_input(
    "Available Ambulances",
    min_value=1,
    max_value=50,
    value=st.session_state.n_ambulances,  # Read current value
    key="input_n_ambulances"              # Unique widget key
)
st.session_state.n_ambulances = n_ambulances  # Write back
```

### 3. Layout: Columns, Containers, Tabs

**Columns for Side-by-Side Layout**
```python
# From app/pages/1_Arrivals.py
col_amb, col_heli, col_walk = st.columns(3)

with col_amb:
    st.subheader("Ambulance Service")
    # Ambulance widgets here

with col_heli:
    st.subheader("HEMS / Air Ambulance")
    # Helicopter widgets here
```

**Containers with Borders**
```python
with st.container(border=True):
    st.markdown("**Capacity**")
    n_triage = st.number_input("Triage Clinicians", ...)
```

**Tabs for Organized Content**
```python
# From app/pages/4_Run.py
tab_arrivals, tab_resources, tab_schematic = st.tabs([
    "Arrivals", "Resources", "Schematic"
])

with tab_arrivals:
    # Arrival summary content
with tab_resources:
    # Resource summary content
```

### 4. Input Widgets

FAER uses various input widgets:

| Widget | Use Case | Example |
|--------|----------|---------|
| `st.number_input()` | Numeric parameters | ED bays, turnaround times |
| `st.slider()` | Bounded ranges | CV values, multipliers |
| `st.selectbox()` | Single selection | Day type, arrival model |
| `st.checkbox()` | Toggle options | Enable/disable diagnostics |
| `st.radio()` | Exclusive options | Arrival model selection |
| `st.data_editor()` | Editable tables | Detailed hourly arrivals |

**Example: Number Input with Validation**
```python
n_ed_bays = st.number_input(
    "ED Bays",
    min_value=5,
    max_value=100,
    value=st.session_state.n_ed_bays,
    help="Total treatment bays. Maps to FullScenario.n_ed_bays"
)
```

**Example: Slider with Fine Control**
```python
demand_scale = st.slider(
    "Overall Demand Scale",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Scales all arrivals proportionally"
)
```

### 5. Display Elements

**Metrics for KPIs**
```python
# From app/pages/5_Results.py
st.metric(
    "P(Delay)",
    f"{ci['mean']:.1%}",
    help="Probability a patient waits for treatment"
)
st.caption(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
```

**Progress Indicators**
```python
# Visual acuity distribution
for priority, proportion in amb_acuity.items():
    st.progress(proportion, text=f"{label}: {proportion:.0%}")
```

**Alerts and Messages**
```python
st.info("Configuration tip...")
st.success("All checks passed")
st.warning("Capacity approaching limit")
st.error("Configuration invalid")
```

### 6. Data Visualization

**Plotly Charts**
```python
# From app/pages/5_Results.py
fig = px.bar(
    util_data,
    x="Resource",
    y="Utilisation",
    title="Mean Resource Utilisation",
    color="Category",
)
fig.add_hline(y=0.85, line_dash="dash", annotation_text="Target")
st.plotly_chart(fig, use_container_width=True)
```

**Graphviz Diagrams**
```python
# From app/pages/4_Run.py
schematic = build_capacity_graph_from_params(
    n_ambulances=st.session_state.n_ambulances,
    n_ed_bays=st.session_state.n_ed_bays,
    # ... more params
)
st.graphviz_chart(schematic, use_container_width=True)
```

**DataFrames**
```python
st.dataframe(
    pd.DataFrame(summary_data),
    use_container_width=True,
    hide_index=True
)
```

### 7. Running the Simulation

The Run page demonstrates how to execute long-running processes:

```python
# From app/pages/4_Run.py
if st.button("Run Simulation", type="primary"):

    # Build scenario from session state
    scenario = build_scenario_from_session(run_length, warm_up, seed)

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(current: int, total: int):
        progress_bar.progress(current / total)
        status_text.text(f"Running replication {current}/{total}...")

    # Run simulation
    results = multiple_replications(
        scenario,
        n_reps=n_reps,
        progress_callback=progress_callback
    )

    # Store results in session state
    st.session_state.run_results = results
    st.session_state.run_complete = True

    st.rerun()  # Refresh to show results
```

### 8. File Downloads

```python
# From app/pages/5_Results.py
csv = export_df.to_csv(index=False)
st.download_button(
    label="Download Results (CSV)",
    data=csv,
    file_name="faer_results.csv",
    mime="text/csv"
)
```

## Data Flow Through the App

```
┌─────────────────────────────────────────────────────────────────┐
│                        st.session_state                         │
│  (Persistent storage across pages and re-runs)                  │
└─────────────────────────────────────────────────────────────────┘
        ↑                    ↑                    ↑
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │Arrivals │          │Resources│          │   Run   │
   │  Page   │          │  Page   │          │  Page   │
   └─────────┘          └─────────┘          └────┬────┘
   - Fleet config       - Triage             - Builds FullScenario
   - Arrival model      - ED bays            - Runs simulation
   - Day type           - Diagnostics        - Stores results
   - Multipliers        - Downstream
                                                  │
                                                  ↓
                                          ┌──────────────┐
                                          │   Results    │
                                          │    Page      │
                                          └──────────────┘
                                          - KPI display
                                          - Visualizations
                                          - CSV export
```

## Key Session State Variables

| Variable | Set By | Used By | Purpose |
|----------|--------|---------|---------|
| `n_ambulances` | Arrivals | Run, Schematic | Fleet size |
| `arrival_model` | Arrivals | Run | 'simple', 'profile_24h', 'detailed' |
| `n_ed_bays` | Resources | Run, Schematic | ED capacity |
| `scenario` | Run | Results | Built FullScenario object |
| `run_results` | Run | Results | Dict of metric lists |
| `run_complete` | Run | Results, Home | Flag for valid results |

## Connecting UI to Simulation Engine

The Run page builds a `FullScenario` from session state:

```python
def build_scenario_from_session(run_length, warm_up, seed):
    # Map UI strings to enums
    arrival_model_enum = {
        'simple': ArrivalModel.SIMPLE,
        'profile_24h': ArrivalModel.PROFILE_24H,
        'detailed': ArrivalModel.DETAILED,
    }[st.session_state.arrival_model]

    # Build diagnostic configs from individual settings
    diagnostic_configs = {}
    if st.session_state.ct_enabled:
        diagnostic_configs[DiagnosticType.CT_SCAN] = DiagnosticConfig(
            capacity=st.session_state.ct_capacity,
            process_time_mean=st.session_state.ct_scan_time_mean,
            # ...
        )

    # Create complete scenario
    return FullScenario(
        run_length=run_length,
        n_ed_bays=st.session_state.n_ed_bays,
        arrival_model=arrival_model_enum,
        diagnostic_configs=diagnostic_configs,
        # ... all other parameters
    )
```

## Running the App

```bash
# From project root
streamlit run app/Home.py

# With custom port
streamlit run app/Home.py --server.port 8501
```

## Common Patterns

### Conditional Rendering
```python
if st.session_state.get('run_complete'):
    st.success("Simulation complete!")
    # Show results
else:
    st.info("Run a simulation to see results")
```

### Expanders for Advanced Options
```python
with st.expander("Advanced Settings", expanded=False):
    # Less common options hidden by default
```

### Page Links
```python
st.page_link("pages/4_Run.py", label="Go to Run", icon="▶️")
```

### Rerun After State Change
```python
if st.button("Reset All"):
    st.session_state.clear()
    st.rerun()  # Refresh page with cleared state
```

## Best Practices Used in FAER

1. **Initialize defaults in each page** - Ensures pages work independently
2. **Use help text on widgets** - Documents what each parameter does
3. **Show validation feedback** - Warnings when configs may cause issues
4. **Link session state to FullScenario attributes** - Clear mapping
5. **Group related settings** - Containers and columns for organization
6. **Provide navigation hints** - Tell users what to do next

## Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Session State](https://docs.streamlit.io/library/api-reference/session-state)
- [Multi-Page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)
- [SimPy Streamlit Tutorial](https://health-data-science-or.github.io/simpy-streamlit-tutorial/) (Tom Monks)
