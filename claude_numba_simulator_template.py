"""
numba_simulator_template.py
===========================
Blank scenario template for the Numba simulation framework.

Usage
-----
1. Copy this file and rename it for your experiment.
2. Fill in each ``@nb.njit`` kernel to implement one trial of your scenario.
   The function must return a scalar or 1-D array of float-compatible values.
3. Build ``scenarios_list`` — a flat list of ``SimulatorSpec`` entries, one per run.
   Each holds the simulator, a dict of argument values (keys must match the kernel's
   parameter names, in the same order), and a tuple of output-category label strings.
   Add one entry per run; repeat the simulator with different
   argument values to compare scenarios.
4. Set ``cap``, ``show_args``, ``should_plot``, and ``do_profiling`` as needed,
   then run the file.

See README.md for a worked example.
"""
import numpy as np
import numba as nb

from claude_general_numba_simulator import gen_simulator, compare_results, SimulatorSpec


# np.random.randint(low, high) — note: high is EXCLUSIVE
# Example: np.random.randint(1, 7) gives a d6 roll (1–6 inclusive).

@nb.njit(cache=True)
def simulator_0():
    """[description]"""
    # Implement one trial here.
    # Return a scalar (single output) or a fixed-length 1-D array (multiple outputs).
    
    
    
    return None


if __name__ == "__main__":
    # --- Scenarios ---
    # Each SimulatorSpec holds: the simulator kernel, a dict of argument values,
    # and a tuple of output-category labels (one label per return value).
    # Dict keys must match the kernel's parameter names in the same order.
    # Add one entry per run; duplicate the simulator with different args to compare.
    # Output category labels must not be empty — they label the printed stats and plot axes.
    scenarios_list = [
        SimulatorSpec(
            sim_func=simulator_0,
            args={}, 
            categories=("",), 
        ),

    ]


    # --- Run settings ---
    cap = 10**6     # number of simulation trials
    show_args = True

    # plot_mode controls which plots are shown:
    #   "none"       — no plots
    #   "individual" — per-scenario proportion + CDF plots
    #   "compare"    — overlay comparison across scenarios (no-op if only one scenario)
    #   "both"       — individual plots, then comparison overlay
    plot_mode = "individual"

    should_plot    = plot_mode in ("individual", "both")
    should_compare = plot_mode in ("compare", "both")

    do_profiling = False   # set True to print a cProfile timing breakdown

    if do_profiling:
        import cProfile, pstats, io
        from pstats import SortKey
        profiler = cProfile.Profile()
        profiler.enable()

    collected = []
    for scenario in scenarios_list:
        results, analysis = gen_simulator(cap, scenario, show_args, should_plot)
        collected.append((scenario, results))

    if should_compare and len(collected) > 1:
        compare_results(collected)

    if do_profiling:
        profiler.disable()
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE).print_stats(25)
        print(s.getvalue())
