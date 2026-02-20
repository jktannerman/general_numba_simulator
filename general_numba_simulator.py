"""
general_numba_simulator.py
==========================
Core framework for Monte-Carlo simulation of discrete-probability / die-roll
scenarios using Numba-JIT compiled kernels.

Public API
----------
SimulatorSpec        – Named container for a scenario (kernel, args, labels).
make_mass_simulator  – Factory: wraps a Numba kernel in a parallel driver.
gen_simulator        – Run a scenario and return raw results + summary stats.
results_analyser     – Print stats and optionally plot distributions.
compare_results      – Overlay proportion and CDF plots across multiple scenarios.

See README.md for usage examples and template workflow.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import numba as nb
import time
from typing import Any, Callable, NamedTuple, Sequence


# Maximum number of discrete bars / histogram bins shown in plots.
# Data with more unique values is bucketed via np.histogram to keep the
# bar chart readable and matplotlib rendering fast.
_MAX_EXACT_BUCKETS = 200

# float32 values above this threshold cannot be represented exactly as integers
# (float32 has 24 bits of mantissa, so 2**24 is the largest exactly-representable
# integer). The fast bincount path is restricted to values at or below this limit.
_FLOAT32_EXACT_INT_MAX = 2**24


class SimulatorSpec(NamedTuple):
    """Specification for a single simulation scenario.

    Attributes:
        sim_func: A Numba ``@njit`` kernel that runs one trial and returns a
            scalar or 1-D array of float-compatible values.
        args: A ``namedtuple`` whose fields are the fixed arguments forwarded
            to ``sim_func`` on every trial. Field names appear in printed output
            when ``show_args=True``.
        categories: Labels for each output value returned by ``sim_func``.
            The length must match the number of values the kernel returns.
    """

    sim_func: Callable
    args: Any
    categories: Sequence[str]


def _compute_counts(
    column_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-value counts and proportions for a 1-D array of simulation results.

    Uses an O(n) bincount fast path for non-negative integer data with at most
    ``_MAX_EXACT_BUCKETS`` unique values; falls back to ``np.unique`` or
    ``np.histogram`` otherwise.

    Args:
        column_data: 1-D float32 array of trial results for one output category.

    Returns:
        Tuple of ``(unique_values, counts, proportions)`` where each array has one
        entry per distinct value or histogram bucket.
    """
    col_min = int(column_data.min())
    col_max = int(column_data.max())
    col_range = col_max - col_min

    # Check whether all values are non-negative whole numbers within float32's
    # exact-representation range. np.floor comparison avoids the int32 overflow
    # that a cast-based check would silently produce for values above 2**31.
    is_integer_valued = (
        col_min >= 0
        and col_max <= _FLOAT32_EXACT_INT_MAX
        and bool((np.floor(column_data) == column_data).all())
    )

    if is_integer_valued and col_range <= _MAX_EXACT_BUCKETS:
        # O(n) fast path via bincount — avoids the O(n log n) sort inside np.unique.
        bc = np.bincount(column_data.astype(np.int32) - col_min)
        unique_values = np.arange(col_min, col_max + 1, dtype=np.float32)
        counts = bc
        proportions = bc / bc.sum()
    else:
        # General path: use np.unique, then bucket if too many unique values.
        unique_values, counts = np.unique(column_data, return_counts=True)
        n_unique = len(unique_values)
        if n_unique > _MAX_EXACT_BUCKETS:
            counts, bin_edges = np.histogram(column_data, bins=_MAX_EXACT_BUCKETS)
            unique_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centres
            proportions = counts / counts.sum()
        else:
            proportions = counts / len(column_data)

    return unique_values, counts, proportions


def _scenario_label(spec: SimulatorSpec) -> str:
    """Build a human-readable legend label from a scenario's function name and args."""
    args_str = ", ".join(f"{k}={v}" for k, v in spec.args._asdict().items())
    return f"{spec.sim_func.__name__}({args_str})"


def make_mass_simulator(
    spec: SimulatorSpec,
) -> Callable[[int, int], np.ndarray]:
    """Build a Numba-parallel driver that runs a single-trial kernel in bulk.

    Args:
        spec: A ``SimulatorSpec`` containing the kernel, its fixed arguments,
            and the output-category labels. ``len(spec.categories)`` determines
            the column count of the output array.

    Returns:
        A compiled function ``driver(cap, seed) -> ndarray`` of shape
        ``(cap, len(spec.categories))``, where each row is the result of one
        independent trial. Pass ``seed=-1`` to skip seeding.
    """
    sim_func, args, categories = spec
    fixed_args = tuple(args)
    n_outputs = len(categories)

    # The inner driver is compiled with parallel=True so that nb.prange
    # distributes trials across CPU cores automatically.
    @nb.njit(parallel=True, cache=True)
    def _driver(cap: int, seed: int) -> np.ndarray:
        if seed >= 0:
            np.random.seed(seed)
        out = np.empty((cap, n_outputs), dtype=np.float32)
        for i in nb.prange(cap):
            out[i] = sim_func(*fixed_args)
        return out

    return _driver


def results_analyser(
    simulator_w_args: SimulatorSpec,
    results: np.ndarray,
    show_args: bool,
    should_plot: bool,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    prob_thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Print summary statistics and optionally plot distributions for all output categories.

    Args:
        simulator_w_args: ``SimulatorSpec`` of ``(kernel, args_namedtuple, category_labels)``.
        results: Raw simulation output, shape ``(n_trials, n_categories)``, dtype float32.
        show_args: If True, print the argument values alongside the simulator name.
        should_plot: If True, display a proportion bar chart and cumulative-probability
            step plot for each output category.
        percentiles: Sequence of percentile values (0–100) to report per category.
            Defaults to ``(10, 25, 50, 75, 90)``.
        prob_thresholds: Optional sequence of threshold values. For each threshold ``t``,
            prints ``P(X >= t)`` and ``P(X <= t)`` for each output category.

    Returns:
        Tuple of ``(means, stds)``, each a 1-D array with one value per category.
    """
    simulator, args, categories = simulator_w_args

    # Compute column-wise statistics across all trials.
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    print(
        f"{simulator.__name__} results - '{simulator.__doc__}'"
    )

    if show_args:
        print(
            f"{simulator.__name__} args: "
            f'{", ".join(f"{name}={value}" for name, value in args._asdict().items())}'
        )

    print("")

    for i, category in enumerate(categories):
        # Extract the column once; reused for stats, percentiles, thresholds, and plotting.
        column_data = results[:, i]
        median_val = float(np.median(column_data))

        pct_values = np.percentile(column_data, percentiles)
        pct_labels = "/".join(f"P{int(p)}" for p in percentiles)
        pct_str = " / ".join(str(round(float(v), 4)) for v in pct_values)

        # Build all label-value pairs first so column width can be computed once.
        rows: list[tuple[str, str]] = [
            (f"{category} mean",               str(round(means[i], 6))),
            (f"{category} standard deviation", str(round(stds[i], 6))),
            (f"{category} median",             str(round(median_val, 6))),
            (f"{category} {pct_labels}",       pct_str),
        ]
        if prob_thresholds is not None:
            for t in prob_thresholds:
                p_geq = float((column_data >= t).mean())
                p_leq = float((column_data <= t).mean())
                rows.append((f"{category} P(X >= {t})", str(round(p_geq, 6))))
                rows.append((f"{category} P(X <= {t})", str(round(p_leq, 6))))

        col_width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"{label:<{col_width}}: {value}")

        print("")

        if not should_plot:
            continue

        unique_values, counts, proportions = _compute_counts(column_data)

        # –– Plot 1: proportion distribution as a bar chart, with grid twice as dense

        fig, ax = plt.subplots()
        ax.set_axisbelow(True)  # grid goes behind bars/lines

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.minorticks_on()

        ax.bar(unique_values, proportions, color='blue', alpha=0.7, zorder=2)

        ax.set_xlabel(f"{category} values")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Proportion Distribution of {category} Values")

        ax.grid(which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        # –– Plot 2: cumulative probability (CDF) as a step + red crosses
        # Normalize to [0, 1] so both plots share the same probability scale.
        cdf = np.cumsum(counts) / counts.sum()

        fig, ax = plt.subplots()
        ax.set_axisbelow(True)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.minorticks_on()

        ax.step(unique_values, cdf, where='post', linewidth=2, zorder=2)
        ax.scatter(unique_values, cdf, color='red', marker='x', s=50, zorder=3)

        ax.set_xlabel(f"{category} values")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Probability Distribution of {category} Values")

        ax.grid(which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    print("")

    # Block until the user closes all open plot windows before returning.
    if should_plot:
        plt.show(block=True)

    return means, stds


def gen_simulator(
    cap: int,
    simulator_w_args: SimulatorSpec,
    show_args: bool,
    should_plot: bool,
    seed: int | None = None,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    prob_thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Run a simulation scenario and analyse the results.

    Compiles (or loads from cache) a parallel Numba driver for the given kernel,
    runs ``cap`` trials, then delegates to ``results_analyser`` for stats and plots.

    Args:
        cap: Number of independent trials to simulate (e.g. ``10**6``).
        simulator_w_args: ``SimulatorSpec`` of ``(kernel, args_namedtuple, category_labels)``.
        show_args: If True, print the argument values in the output header.
        should_plot: If True, display proportion and cumulative-probability charts.
        seed: Optional integer seed for Numba's RNG. When provided, the driver
            calls ``np.random.seed(seed)`` before running trials, making results
            reproducible across runs with the same seed and the same number of
            threads (``NUMBA_NUM_THREADS``). Defaults to ``None`` (no seeding).
        percentiles: Sequence of percentile values (0–100) to report per category.
            Defaults to ``(10, 25, 50, 75, 90)``.
        prob_thresholds: Optional sequence of threshold values. For each threshold ``t``,
            prints ``P(X >= t)`` and ``P(X <= t)`` for each output category.

    Returns:
        Tuple of:
            - ``results``: Raw float32 array of shape ``(cap, n_categories)``.
            - ``analysis``: ``(means, stds)`` summary statistics from ``results_analyser``.
    """
    # Build (or retrieve from Numba's on-disk cache) the parallel driver.
    mass_sim = make_mass_simulator(simulator_w_args)
    if len(mass_sim.nopython_signatures) == 0:
        print("Preparing JIT kernel...", end="", flush=True)
        t0 = time.perf_counter()
        mass_sim(0, -1)          # zero-trial warmup: compiles/loads cache, runs no trials
        print(f" done in {time.perf_counter() - t0:.2f}s")

    print(f"Running {cap:,} trials...", end="", flush=True)
    t0 = time.perf_counter()
    results = mass_sim(cap, seed if seed is not None else -1)
    print(f" done in {time.perf_counter() - t0:.2f}s")
    print("")

    analysis = results_analyser(
        simulator_w_args, results, show_args, should_plot, percentiles, prob_thresholds
    )

    return results, analysis


def compare_results(collected: list[tuple[SimulatorSpec, np.ndarray]]) -> None:
    """Plot overlaid proportion and CDF charts for multiple simulation scenarios.

    For each output category, opens two figures: a proportion bar chart and a
    cumulative-probability step chart, with one series per scenario. All windows
    open non-blocking and block together at the end.

    Args:
        collected: List of ``(SimulatorSpec, results_array)`` pairs, one per scenario.
            All specs must share the same number of output categories. Results arrays
            must have shape ``(n_trials, n_categories)``.
    """
    if not collected:
        return

    categories = collected[0][0].categories
    assert all(len(spec.categories) == len(categories) for spec, _ in collected), (
        "All scenarios passed to compare_results must have the same number of output categories."
    )

    prop_cycle_colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    for cat_idx, category in enumerate(categories):
        # Pre-compute counts for every scenario so each figure iterates a simple list.
        scenario_data = []
        for spec, results in collected:
            column_data = results[:, cat_idx]
            unique_values, counts, proportions = _compute_counts(column_data)
            cdf = np.cumsum(counts) / counts.sum()
            scenario_data.append((spec, unique_values, proportions, cdf))

        # –– Figure A: proportion overlay
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.minorticks_on()

        for idx, (spec, unique_values, proportions, cdf) in enumerate(scenario_data):
            color = prop_cycle_colors[idx % len(prop_cycle_colors)]
            ax.bar(
                unique_values, proportions,
                alpha=0.5, color=color, label=_scenario_label(spec), zorder=2,
            )

        ax.set_xlabel(f"{category} values")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Proportion Distribution of {category} — Comparison")
        ax.grid(which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        # –– Figure B: CDF overlay
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.minorticks_on()

        for idx, (spec, unique_values, proportions, cdf) in enumerate(scenario_data):
            color = prop_cycle_colors[idx % len(prop_cycle_colors)]
            ax.step(
                unique_values, cdf,
                where='post', color=color, linewidth=2, label=_scenario_label(spec), zorder=2,
            )
            ax.scatter(unique_values, cdf, color=color, marker='x', s=50, zorder=3)

        ax.set_xlabel(f"{category} values")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Probability of {category} — Comparison")
        ax.grid(which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    plt.show(block=True)
