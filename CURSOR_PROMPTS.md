# Cursor Prompts for Pj FAER

Copy-paste these prompts into Cursor's Claude Code chat. Run them in order.

---

## PROMPT 1: Project Scaffolding (Phase 0)

```
Create the Pj FAER project scaffolding as specified in CLAUDE.md.

Do the following in order:

1. Create the full folder structure:
   - src/faer/core/
   - src/faer/model/
   - src/faer/results/
   - src/faer/experiment/
   - app/pages/
   - tests/
   - data/arrival_profiles/

2. Create all __init__.py files (can be empty for now)

3. Create pyproject.toml with these dependencies:
   - simpy>=4.1.1
   - numpy>=1.24.0
   - pandas>=2.0.0
   - scipy>=1.10.0
   - sim-tools>=0.4.0
   - streamlit>=1.30.0
   - plotly>=5.18.0
   - pytest, black, ruff as dev dependencies

4. Create a minimal README.md

5. Create tests/conftest.py with a basic fixture

6. Verify the structure is correct with a simple test that imports faer

Do not implement any simulation logic yet - just the skeleton.
```

---

## PROMPT 2: Scenario Configuration (Phase 1a)

```
Implement the Scenario dataclass in src/faer/core/scenario.py

Requirements:
- Use @dataclass decorator
- Include these parameters with defaults:
  - run_length: float = 480.0 (minutes, 8 hours)
  - warm_up: float = 0.0
  - arrival_rate: float = 4.0 (patients per hour)
  - n_resus_bays: int = 2
  - resus_mean: float = 45.0 (minutes)
  - resus_cv: float = 0.5 (coefficient of variation)
  - random_seed: int = 42

- Implement __post_init__ that creates separate RNG streams:
  - self.rng_arrivals = np.random.default_rng(self.random_seed)
  - self.rng_service = np.random.default_rng(self.random_seed + 1)
  - self.rng_routing = np.random.default_rng(self.random_seed + 2)

- Add type hints everywhere
- Add docstring explaining the class

Then create tests/test_scenario.py with tests for:
- Default scenario creation
- Custom parameter override
- RNG reproducibility (same seed = same first random value)
- Different seeds produce different values
```

---

## PROMPT 3: Basic SimPy Model (Phase 1b)

```
Implement the minimal SimPy model in src/faer/model/processes.py

Create these functions:

1. patient_process(env, patient_id, resus_bays, scenario, results)
   - Generator function for patient journey
   - Record arrival time
   - Request resus bay (use `with resus_bays.request() as req: yield req`)
   - Record queue_time = env.now - arrival_time
   - Sample service time using lognormal (use scenario.rng_service)
   - yield env.timeout(service_time)
   - Record system_time and increment departures

2. arrival_generator(env, resus_bays, scenario, results)
   - Generator function for arrivals
   - Infinite loop with exponential inter-arrival times
   - Use scenario.rng_arrivals
   - Convert arrival_rate (per hour) to mean IAT (minutes)
   - Spawn patient_process for each arrival

3. run_simulation(scenario: Scenario) -> dict
   - Create simpy.Environment
   - Create simpy.Resource for resus_bays
   - Initialize results dict with arrivals, departures, queue_times, system_times
   - Start arrival_generator process
   - Run until scenario.run_length
   - Return results

Use the patterns from CLAUDE.md. Add type hints and docstrings.

Then create tests/test_model.py with tests for:
- Basic flow: arrivals > 0, departures > 0
- Reproducibility: same seed = same queue_times list
- High capacity (no queuing): mean queue time near 0
- Low capacity (queuing occurs): mean queue time > 0
```

---

## PROMPT 4: Results Collector (Phase 2a)

```
Implement ResultsCollector in src/faer/results/collector.py

Create a dataclass with:
- arrivals: int = 0
- departures: int = 0
- queue_times: List[float] = field(default_factory=list)
- system_times: List[float] = field(default_factory=list)
- resource_log: List[Tuple[float, int]] = field(default_factory=list)

Add methods:
- record_arrival()
- record_queue_time(wait: float)
- record_system_time(time: float)
- record_resource_state(time: float, n_busy: int)
- compute_metrics(run_length: float, capacity: int) -> Dict

The compute_metrics method should return:
- arrivals, departures
- p_delay: proportion with queue_time > 0
- mean_queue_time, median_queue_time, p95_queue_time
- mean_system_time
- utilisation: time-weighted from resource_log
- throughput_per_hour

Implement _compute_utilisation as a private method that calculates
time-weighted average of busy resources.

Update processes.py to use ResultsCollector instead of raw dict.

Add tests in tests/test_results.py
```

---

## PROMPT 5: NSPP Thinning Arrivals (Phase 2b)

```
Implement non-stationary arrivals in src/faer/core/arrivals.py

Create:

1. ArrivalProfile dataclass:
   - schedule: List[Tuple[float, float]]  # (end_time, rate) pairs
   - Compute max_rate in __post_init__
   - Method get_rate(t: float) -> float

2. NSPPThinning class:
   - __init__(profile: ArrivalProfile, rng: np.random.Generator)
   - sample_iat(current_time: float) -> float
   - Implement thinning algorithm:
     - Generate candidate IAT from Exp(max_rate)
     - Accept with probability current_rate / max_rate
     - If rejected, continue from new time

3. load_default_profile() -> ArrivalProfile:
   - Return 24-hour ED arrival pattern
   - Low overnight (1-2/hr), peak morning (7/hr), evening peak (6-7/hr)

Update arrival_generator in processes.py to optionally use NSPPThinning
when scenario has an arrival_profile attribute.

Add tests:
- Constant rate profile should match exponential
- Time-varying profile produces correct hourly counts (within 20%)
- Thinning efficiency > 30% for typical ED profile
```

---

## PROMPT 6: Replication Runner (Phase 2c)

```
Implement experimentation in src/faer/experiment/runner.py

Create:

1. multiple_replications(scenario, n_reps, metric_names=None) -> Dict[str, List[float]]
   - Run n_reps simulations with different seeds (base_seed + rep)
   - Clone scenario for each rep, reset RNGs
   - Collect specified metrics from each run
   - Return dict mapping metric name to list of values

2. compute_ci(values: List[float], confidence=0.95) -> Dict
   - Compute mean, std, se
   - Compute t-critical value for confidence level
   - Return dict with mean, ci_lower, ci_upper, ci_half_width, n

3. run_until_precision(scenario, target_metric, target_half_width, 
                       max_reps=200, min_reps=10, batch_size=5) -> Dict
   - Run replications until CI half-width <= target
   - Check precision every batch_size reps after min_reps
   - Return converged flag, n_reps, values, final CI

Add tests:
- CI half-width decreases with sqrt(n)
- run_until_precision stops when target met
- run_until_precision hits max_reps if target too tight
```

---

## PROMPT 7: Streamlit App (Phase 3)

```
Create the Streamlit app structure.

1. app/Home.py:
   - Page config with title "Pj FAER" and wide layout
   - Title and markdown explaining the project
   - Brief description of current capabilities
   - Attribution to Monks/sim-tools

2. app/pages/1_Scenario.py:
   - Two-column layout
   - Left: Horizon (slider 1-36 hours), Resources (n_resus slider)
   - Right: Service times (mean, CV inputs)
   - Show default arrival profile as line chart
   - Experiment settings: n_reps slider, seed input
   - Save button that stores Scenario in st.session_state

3. app/pages/2_Run.py:
   - Check for scenario in session state
   - Show scenario summary
   - Run button with st.spinner and progress bar
   - Call multiple_replications
   - Store results in session state
   - Show success message with elapsed time

4. app/pages/3_Results.py:
   - Check for results in session state
   - Three-column KPI cards: P(delay), Mean queue time, Utilisation
   - Show CI below each metric
   - Histogram of P(delay) across replications using Plotly
   - CSV download button

Use st.cache_data where appropriate. Follow Streamlit best practices.
```

---

## PROMPT 8: Integration Test

```
Create an integration test that verifies the full pipeline works.

In tests/test_integration.py:

1. test_full_pipeline():
   - Create a Scenario with known parameters
   - Run multiple_replications with n_reps=10
   - Verify all expected metrics are present
   - Verify P(delay) is between 0 and 1
   - Verify utilisation is between 0 and 1
   - Verify CI computation works

2. test_nspp_pipeline():
   - Create Scenario with arrival_profile
   - Run simulation
   - Verify arrivals occurred
   - Verify results are reasonable

3. test_precision_convergence():
   - Run run_until_precision with loose target (0.1)
   - Verify it converges in reasonable number of reps
   - Verify final CI meets target

Run all tests and fix any issues.
```

---

## PROMPT 9: Documentation & Cleanup

```
Finalize the project:

1. Update README.md with:
   - Project description
   - Installation instructions
   - Quick start (run Streamlit app)
   - Project structure overview
   - Attribution section

2. Create ATTRIBUTION.md listing:
   - sim-tools by Monks et al.
   - STARS project
   - simpy-streamlit-tutorial
   - Include MIT license notice

3. Add docstrings to any functions missing them

4. Run black and ruff to format/lint all code

5. Run pytest with coverage and report results

6. List any remaining TODOs for Phase 4
```

---

## Usage Tips

1. **Run prompts in order** - each builds on the previous
2. **Test after each prompt** - run `pytest` before moving on
3. **Commit after each phase** - good checkpoints
4. **If something breaks** - ask Claude to debug with the error message
5. **To continue a prompt** - say "continue" or "finish implementing X"

## Quick Debug Prompts

```
# If tests fail
The test test_X is failing with this error: [paste error]. Fix it.

# If import fails
I'm getting ImportError when importing faer. Check the __init__.py files and fix.

# If Streamlit crashes
The Streamlit app crashes with this error: [paste error]. Debug and fix.

# To add a feature
Add [feature] following the patterns in CLAUDE.md. Include tests.
```
