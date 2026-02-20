# Numba Simulator Framework

A lightweight Monte-Carlo simulation framework for simulation and analysis of
probabilistic scenarios. The simulation core is Numba-JIT compiled and runs in
parallel; the analysis layer is pure Python.

---

## Files

| File | Purpose |
|---|---|
| `general_numba_simulator.py` | Core framework — parallel driver factory, stats, plotting, comparison |
| `numba_simulator_template.py` | Blank scenario template — copy and fill in per experiment |

---

## How It Works

### 1. Write a kernel (`numba_simulator_template.py`)

Define one or more `@nb.njit` functions. Each function represents one "trial" of
your experiment and must return a 1-D array (or scalar) of output values.

```python
@nb.njit(cache=True)
def roll_advantage(num_dice, die_size):
    """Roll num_dice d{die_size} and return the highest."""
    best = 0
    for _ in range(num_dice):
        roll = np.random.randint(1, die_size + 1)   # randint is exclusive on right
        if roll > best:
            best = roll
    return best
```

### 2. Specify arguments as a dict

Pass kernel arguments as a plain `dict` — no class definition needed. Dict keys
appear in the printed header when `show_args=True` and in plot legend labels.

```python
args={"num_dice": 3, "die_size": 6}   # keys must match kernel parameter names (order doesn't matter)
```

> **namedtuple alternative:** If you want a reusable type or IDE autocompletion on
> argument names, namedtuples still work identically:
> ```python
> from collections import namedtuple
> Arguments = namedtuple("Arguments", ["num_dice", "die_size"])
> args=Arguments(num_dice=3, die_size=6)
> ```

### 3. Build a scenarios list

`scenarios_list` is a flat list of `SimulatorSpec` entries. Each entry contains
the simulator, one set of argument values, and the output-category labels. Import
`SimulatorSpec` and `compare_results` from the framework:

```python
from general_numba_simulator import gen_simulator, compare_results, SimulatorSpec

scenarios_list = [
    SimulatorSpec(
        sim_func=roll_advantage,
        args={"num_dice": 3, "die_size": 6},
        categories=("result",),
    ),
    SimulatorSpec(
        sim_func=roll_advantage,
        args={"num_dice": 2, "die_size": 6},
        categories=("result",),
    ),
]
```

Add one entry per run. Repeating the simulator with different `args` dicts
compares the same experiment under different parameters.

### 4. Call `gen_simulator`, then `compare_results`

```python
collected = []
for scenario in scenarios_list:
    results, analysis = gen_simulator(
        cap=10**6,
        spec=scenario,
        show_args=True,
        should_plot=True,
    )
    collected.append((scenario, results))

if len(collected) > 1:
    compare_results(collected)
```

`results` is a `(cap, n_outputs)` float32 array of raw trial data.
`analysis` is `(means, stds)` — one value per output category.

---

## Key Parameters

| Parameter | Type | Description |
|---|---|---|
| `cap` | `int` | Number of simulation trials (e.g. `10**6`) |
| `show_args` | `bool` | Print the argument values alongside results |
| `should_plot` | `bool` | Show proportion and cumulative-probability charts per scenario |
| `plot_mode` | `str` | (template only) `"none"` \| `"individual"` \| `"compare"` \| `"both"` — controls which plots are shown; derives `should_plot` and whether `compare_results` is called |
| `seed` | `int \| None` | Integer seed for Numba's RNG; `None` (default) means no seeding |
| `percentiles` | `Sequence[float]` | Percentile values to report; defaults to `(10, 25, 50, 75, 90)` |
| `prob_thresholds` | `Sequence[float] \| None` | If set, prints `P(X >= t)` and `P(X <= t)` for each `t` |
| `do_profiling` | `bool` | (template only) Wrap run with `cProfile` |

---

## Statistics Output

For each output category, the following are printed per scenario:

- **mean** and **standard deviation**
- **median**
- **Percentile row** — one value per entry in `percentiles` (default P10/P25/P50/P75/P90)
- **Threshold queries** — `P(X >= t)` and `P(X <= t)` for each value in `prob_thresholds`

All labels within a category are padded to the same width so the values column aligns:

```
result mean:               13.271600
result standard deviation:  3.108200
result median:             14.000000
result P10/P25/P50/P75/P90: 9.0 / 11.0 / 14.0 / 16.0 / 18.0
result P(X >= 15):          0.382100
result P(X <= 15):          0.617900
```

---

## Plotting

### Per-scenario (`should_plot=True`)

Two matplotlib windows are produced per output category:

1. **Proportion distribution** — bar chart of P(X = x)
2. **Cumulative probability** — post-step plot with scatter markers of F(x), y-axis in [0, 1]

All windows for a scenario open immediately without blocking (`show(block=False)` +
`pause`). Execution blocks until the user closes all windows before moving to the
next scenario.

### Multi-scenario comparison (`compare_results`)

After all scenarios have run, `compare_results(collected)` produces two additional
windows per output category, with one series per scenario drawn in a distinct colour:

1. **Proportion overlay** — transparent bars (alpha=0.5) for each scenario
2. **CDF overlay** — step lines with scatter markers for each scenario

Legend entries are labelled with the kernel name and its argument values
(e.g. `roll_advantage(num_dice=3, die_size=6)`). All comparison windows open
non-blocking and block together at the end.

### Bucketing (`_MAX_EXACT_BUCKETS = 200`)

For integer-valued data with ≤ 200 unique values (the common die-roll case),
counts are computed via `np.bincount` — O(n), no sort.
For data with > 200 unique values, `np.histogram` buckets the range into 200 bins
so the bar chart stays readable. This logic lives in the private `_compute_counts`
helper and is shared by both per-scenario and comparison plots.

---

## Numba Notes

- All kernels must be decorated `@nb.njit(cache=True)`.
- Use `np.random.randint(low, high)` — the upper bound is **exclusive**.
- The parallel driver (`make_mass_simulator`) uses `nb.prange` over trials;
  kernels themselves should **not** be `parallel=True`.
- Numba caches compiled kernels to disk; the second run is near-instant.
- **Reproducibility:** When `seed` is set, results are reproducible only if the
  number of threads is held constant. Numba distributes trials across threads via
  `nb.prange`, so different thread counts produce different per-thread RNG streams
  even with the same seed. Fix the thread count with the `NUMBA_NUM_THREADS`
  environment variable if exact reproducibility is required.
