"""
numba_simulator_template.py
===========================
Blank scenario template for the Numba simulation framework.

Usage
-----
1. Copy this file and rename it for your experiment.
2. Fill in each ``@nb.njit`` kernel to implement one trial of your scenario.
   The function must return a scalar or 1-D array of float-compatible values.
3. Declare a ``namedtuple`` for the kernel's arguments at module level (one per
   simulator). Field names appear in the printed header when ``show_args=True``.
4. Build ``scenarios_list`` — a flat list of ``SimulatorSpec`` entries, one per run.
   Each holds the simulator, one ``Arguments`` instance, and a tuple of output-category
   label strings. Add one entry per run; repeat the simulator with different
   argument values to compare scenarios.
5. Set ``cap``, ``show_args``, ``should_plot``, and ``do_profiling`` as needed,
   then run the file.

See README.md for a worked example.
"""
import numpy as np
import numba as nb
from collections import namedtuple

from general_numba_simulator import gen_simulator, compare_results, SimulatorSpec


# np.random.randint(low, high) — note: high is EXCLUSIVE
# Example: np.random.randint(1, 7) gives a d6 roll (1–6 inclusive).

# Argument types are defined at module level — one namedtuple per simulator.
# Field names appear in the printed output header when show_args=True.
Arguments_0 = namedtuple("Arguments_0", ["", ])   # replace with your argument names

@nb.njit(cache=True)
def simulator_0():
    """[description]"""
    # Implement one trial here.
    # Return a scalar (single output) or a fixed-length 1-D array (multiple outputs).
    
    
    
    return None


if __name__ == "__main__":
    # --- Scenarios ---
    # Each SimulatorSpec holds: the simulator kernel, one Arguments instance,
    # and a tuple of output-category labels (one label per return value).
    # Add one entry per run; duplicate the simulator with different arguments to compare.
    # Output category labels must not be empty — they label the printed stats and plot axes.
    scenarios_list = [
        SimulatorSpec(
            sim_func=simulator_0,
            args=Arguments_0(),          # fill in argument values
            categories=("",),            # one label per return value of the simulator
        ),
        
    ]


    # --- Run settings ---
    cap = 10**6     # number of simulation trials
    show_args = True
    should_plot = True
    should_compare = True

    do_profiling = False   # set True to print a cProfile timing breakdown

    import cProfile, pstats, io
    from pstats import SortKey

    profiler = cProfile.Profile()
    if do_profiling:
        profiler.enable()

    collected = []
    for scenario in scenarios_list:
        results, analysis = gen_simulator(cap, scenario, show_args, should_plot)
        collected.append((scenario, results))

    if should_compare and (len(collected) > 1 or not should_plot):
        compare_results(collected)

    if do_profiling:
        profiler.disable()
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE).print_stats(25)
        print(s.getvalue())
