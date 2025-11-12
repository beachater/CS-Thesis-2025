
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
import sys

from test4_benchmark import get_function_by_name

# Algorithm runners: (name, runner function)
# --- Import IICO and HybridCSA variants ---
from test4_IICO import iico as IICO_func
from test4_fscsa import FCSA
# from dset import ETFCSA_Lite
from tsd import ETFCSA_TSD
from test4_reformed_hybrid_iico_spark import HybridFCSA_IICO

from test4_reformed_hybrid import HybridRolePartitionedOriginal

# Integrate additional CSA variants
from ADECSA import ADECSA
from DUSCSA import DUSCSA
from MSHCSA import MSHCSA
from CSA import CSA

from progress_logger import ProgressLogger

# Known global optimum values by internal name used in get_function_by_name
KNOWN_OPTIMA = {
    "SchafferN2": 0.0,
    "SchafferN4": 0.0,
    "Ackley": 0.0,
    "Griewank": 0.0,
    "Rastrigin": 0.0,
    "Shubert": -186.7309,     # 2D
    "Eggholder": -959.6407,   # 2D
    "HolderTable": -19.2085,  # 2D
    "Levy": 0.0,
    "Schwefel": 0.0,          # at x_i = 420.9687... for all i
}

alg_color_map = {
    "FCSA": "#2E8B57",
    "TSD": "#007ACC",
    "IICO": "#FF8C00",
    "ADECSA": "#C71585",
    "MSHCSA": "#8A2BE2",
}



# List of benchmark function names and their canonical names in benchmark.py
benchmarks = [
    ("Schaffer 2", "SchafferN2"),
    ("Schaffer 4", "SchafferN4"),
    ("Ackley", "Ackley"),
    ("Griewank", "Griewank"),
    ("Rastrigin", "Rastrigin"),
    ("Shubert", "Shubert"),
    ("Eggholder", "Eggholder"),
    ("Holdertable", "HolderTable"),
    ("Schwefel", "Schwefel"),
    ("Levy", "Levy"),

]

# Global dimension support per benchmark. Move here so interactive flow can consult it.
DIM_SUPPORT = {
    "SchafferN2": [2],
    "SchafferN4": [2],
    "Ackley": [2, 50, 100],
    "Griewank": [2, 50, 100],
    "Rastrigin": [2, 50, 100],
    "Shubert": [2],
    "Eggholder": [2],
    "HolderTable": [2],
    "Levy": [2, 50, 100],
    "Schwefel": [2, 50, 100],
}

# Per-algorithm supported dims. We'll initialize this after `algorithms` is defined below.
ALGO_DIM_SUPPORT = {}


# Experiment defaults (use these to ensure fair comparisons)
DEFAULT_POP = 60
DEFAULT_N_SELECT = 15
DEFAULT_N_CLONES = 5
DEFAULT_A_FRAC = 0.15
DEFAULT_R = 2.0
DEFAULT_MAX_GENS = 1000
DEFAULT_BASE_SEED = 1000



def run_csa(obj, bounds, dim, max_evals, seed=None, bench_func=None):
    """
    Adapter to match your pipeline.
    obj: callable mapping ndarray -> scalar objective to minimize
    bounds: list of (lo, hi) per dimension
    returns: np.array of best fitness per generation
    """
    def scalar_obj(x: np.ndarray) -> float:
        # your pipeline passes obj that returns scalar y
        return float(obj(np.asarray(x, dtype=float)))

    alg = CSA(
        func=scalar_obj,
        bounds=bounds,
        N=60,
        n_select=15,
        n_clones=5,
        r=2.0,
        a_frac=0.15,
        max_gens=1000,
        max_evals=int(max_evals),
        seed=seed,
        # progress=progress,
    )
    _, _, info = alg.optimize()
    # normalize to numpy array like other runners
    return np.asarray(info.get("history", []), dtype=float)


def run_iico(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    """
    Wrapper for IICO with proper progress support and signature parity.
    """
    pop_size = DEFAULT_POP

    # IICO expects fun(x) -> (y,) tuple, not plain float.
    def iico_fun(x):
        x = np.asarray(x, dtype=float)
        y = float(obj(x))
        return (y,)  # return as tuple so IICO can unpack with [0]

    # Call IICO_func using its new signature:
    # iico(fun, bounds, dim, max_FEs, pop_size, bench_func=None, seed=None, progress=None)
    best_fitness_list, gbest, info = IICO_func(
        iico_fun,
        bounds,
        dim,
        int(max_evals),
        int(pop_size),
        bench_func=bench_func,
        seed=seed,
        progress=progress
    )

    # Return per-generation fitness curve
    return np.asarray(info.get("history", best_fitness_list), dtype=float)


def run_fcsa(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    # Ensure FCSA uses the provided eval budget and fixed defaults
    opt = FCSA(
        obj,
        bounds,
        N=DEFAULT_POP,
        n_select=DEFAULT_N_SELECT,
        n_clones=DEFAULT_N_CLONES,
        a_frac=DEFAULT_A_FRAC,
        r=DEFAULT_R,
        seed=seed,
        max_gens=DEFAULT_MAX_GENS,
        max_evals=int(max_evals),
        progress=progress
    )
        # Debug: prove the same instance has the callback
    try:
        print(f"[run_fcsa] attached progress? {opt._progress is not None}")
    except Exception:
        print("[run_fcsa] could not inspect opt._progress")
        
    x_best, f_best, info = opt.optimize(progress=progress)
    return np.array(info.get('history', [f_best]))


def run_ade(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    """Wrapper for ADECSA."""
    try:
        # ADECSA signature returns best, best_f, pop, fitness, info
        best, best_f, pop, fitness, info = ADECSA(
            obj,
            bounds,
            D_dim=dim,
            Ninit=DEFAULT_POP,
            Nc=3,
            H=10,
            FESmax=int(max_evals),
            max_iter=DEFAULT_MAX_GENS,
            seed=seed,
            progress=progress
        )
        return np.array(info.get('history', [float(best_f)]))
    except TypeError:
        # fallback minimal call
        try:
            best, best_f, pop, fitness, info = ADECSA(
                obj,
                bounds,
                D_dim=dim,
                FESmax=int(max_evals),
                seed=seed,
                progress=progress
            )
            return np.array(info.get('history', [float(best_f)]))
        except Exception as e:
            print(f"ADECSA runner error: {e}")
            return np.array([float('nan')])


def run_dus(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    """Wrapper for DUSCSA."""
    try:
        best, best_val, info = DUSCSA(
            obj,
            bounds,
            pop_size=DEFAULT_POP,
            elite_size=DEFAULT_N_SELECT,
            clone_factor=DEFAULT_N_CLONES,
            mutation_rate=0.5,
            crowding_factor=0.1,
            max_iter=DEFAULT_MAX_GENS,
            progress=progress
        )
        return np.array(info.get('history', [float(best_val)]))
    except Exception as e:
        print(f"DUSCSA runner error: {e}")
        return np.array([float('nan')])


def run_msh(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    """Wrapper for MSHCSA."""
    try:
        best, best_val, pop, fitness, info = MSHCSA(
            obj,
            bounds,
            D_dim=dim,
            N=None,
            Nc=1,
            H=10,
            max_evals=int(max_evals),
            max_gens=DEFAULT_MAX_GENS,
            seed=seed,
            verbose=False,
            progress=progress
        )
        return np.array(info.get('history', [float(best_val)]))
    except TypeError:
        try:
            best, best_val, pop, fitness, info = MSHCSA(
                obj,
                bounds,
                D_dim=dim,
                max_evals=int(max_evals),
                seed=seed,
                progress=progress
            )
            return np.array(info.get('history', [float(best_val)]))
        except Exception as e:
            print(f"MSHCSA runner error: {e}")
            return np.array([float('nan')])



def run_tsd(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    # TSD: enforce eval budget; pass common defaults
    opt = ETFCSA_TSD(
        obj,
        bounds,
        N=DEFAULT_POP,
        n_select=DEFAULT_N_SELECT,
        n_clones=DEFAULT_N_CLONES,
        seed=seed,
        progress=progress,
        max_evals=int(max_evals),
    )
    x_best, f_best, info = opt.optimize()
    history = info.get("history", [f_best]) if isinstance(info, dict) else [f_best]
    return np.array(history)


def run_hybrid_reformed_spark(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    # Try to pass max_evals and max_gens if the hybrid supports them
    try:
        opt = HybridFCSA_IICO(
            obj, bounds,
            N=DEFAULT_POP, n_select=DEFAULT_N_SELECT, 
            n_clones=DEFAULT_N_CLONES,
            a_frac=DEFAULT_A_FRAC, 
            r=DEFAULT_R, 
            seed=seed,
            max_gens=DEFAULT_MAX_GENS, 
            max_evals=int(max_evals),
            progress=progress  # <-- add
        )
    except TypeError:
        opt = HybridFCSA_IICO(obj, bounds, seed=seed, progress=progress)  # <-- add

    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))



def run_hybrid_reformed(obj, bounds, dim, max_evals, progress=None, seed=None, bench_func=None):
    # Try to pass max_evals and max_gens if the hybrid supports them
    try:
        opt = HybridRolePartitionedOriginal(
            obj,
            bounds,
            N=DEFAULT_POP,
            n_select=DEFAULT_N_SELECT,
            n_clones=DEFAULT_N_CLONES,
            a_frac=DEFAULT_A_FRAC,
            r=DEFAULT_R,
            seed=seed,
            max_gens=DEFAULT_MAX_GENS,
            max_evals=int(max_evals),
        )
    except TypeError:
        opt = HybridRolePartitionedOriginal(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))

algorithms = [
    # ("HybridReformed", run_hybrid_reformed),
    ("CSA", run_csa),
    ("HybridBase", run_hybrid_reformed_spark),
    ("TSD", run_tsd),
    ("FCSA", run_fcsa), 
    ("IICO", run_iico),
    # Newly integrated CSA variants
    ("ADECSA", run_ade),
    ("DUSCSA", run_dus),
    #("MSHCSA", run_msh),




]

# Initialize per-algorithm support using DIM_SUPPORT as a safe default where possible
for alg_name, _ in algorithms:
    # default: intersection of all benchmark dims where possible, else common set
    all_dims = set()
    for dims in DIM_SUPPORT.values():
        all_dims.update(dims)
    # use the common dims list as default
    ALGO_DIM_SUPPORT[alg_name] = sorted(all_dims) if all_dims else [2, 50, 100]


import argparse



def pad_histories(histories):
    # Pad all histories to the same length with their last value
    max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        if len(h) < max_len:
            pad_val = h[-1]
            h = np.concatenate([h, np.full(max_len - len(h), pad_val)])
        padded.append(h)
    return np.array(padded)


def save_checkpoint(checkpoint_path, state):
    try:
        # Ensure directory exists
        d = os.path.dirname(checkpoint_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        tmp_path = checkpoint_path + '.tmp'
        # write to temporary file first
        with open(tmp_path, 'w') as fh:
            json.dump(state, fh)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass

        # if an existing checkpoint exists, keep a backup
        if os.path.exists(checkpoint_path):
            bak_path = checkpoint_path + '.bak'
            try:
                os.replace(checkpoint_path, bak_path)
            except Exception:
                # best-effort; if cannot replace, ignore
                pass

        # atomically replace
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: failed to write checkpoint {checkpoint_path}: {e}")


def load_checkpoint(checkpoint_path):
    try:
        with open(checkpoint_path, 'r') as fh:
            data = json.load(fh)
            print(f"Loaded checkpoint from {checkpoint_path} (keys={len(data)})")
            return data
    except json.JSONDecodeError as e:
        # file corrupted; try backup
        print(f"Warning: checkpoint {checkpoint_path} corrupted: {e}")
        bak_path = checkpoint_path + '.bak'
        if os.path.exists(bak_path):
            try:
                with open(bak_path, 'r') as fh:
                    data = json.load(fh)
                    print(f"Loaded checkpoint from backup {bak_path} (keys={len(data)})")
                    return data
            except Exception as e2:
                print(f"Warning: backup also failed to load: {e2}")
        # move corrupt file aside for inspection
        try:
            corrupt_path = checkpoint_path + '.corrupt'
            os.replace(checkpoint_path, corrupt_path)
            print(f"Moved corrupt checkpoint to {corrupt_path}")
        except Exception:
            pass
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: failed to load checkpoint {checkpoint_path}: {e}")
        return None


def format_eta(seconds):
    if seconds is None or seconds == float('inf'):
        return 'unknown'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark experiments and save results")
    parser.add_argument("--algorithms", "-a", nargs="*", help="Algorithm names to run (default: all)")
    parser.add_argument("--benchmarks", "-b", nargs="*", help="Benchmark functions to run (names or numbers from the list). If provided, only these benchmarks are run (default: all)")
    parser.add_argument("--dims", "-d", nargs="*", type=int, help="Dimensions to run (e.g. 2 50 100). Default: auto-detect per benchmark")
    parser.add_argument("--priority-dims", "-p", nargs="*", type=int, help="Dimensions to prioritise and run first for each benchmark (e.g. 2). These dims will be moved to the front if supported by the benchmark")
    parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive: accept defaults or provided args")
    parser.add_argument("--base-seed", type=int, default=1000, help="Base seed for runs; per-run seed = base_seed + run_index")
    args = parser.parse_args()

    # Note: by default we run all benchmarks; the user can select specific benchmark functions
    selected_algs = None
    selected_dims = None
    selected_benchmarks = None

    # If user provided benchmark names via CLI, use them
    if args.benchmarks:
        selected_benchmarks = args.benchmarks

    # If not in non-interactive mode and no benchmarks provided, prompt for benchmarks
    if not args.yes and not selected_benchmarks:
        # Interactive guided selection flow:
        # 1) choose algorithms (or 'all')
        print("Available algorithms:")
        for i, (alg_name, _) in enumerate(algorithms):
            print(f"  {i+1}. {alg_name}")
        alg_sel = input("Enter comma-separated algorithm numbers or names to run (or 'all'): ")
        sel_algorithms = None
        if alg_sel.strip().lower() == 'all' or alg_sel.strip() == '':
            sel_algorithms = [name for name, _ in algorithms]
        else:
            choices = [s.strip() for s in alg_sel.split(',') if s.strip()]
            sel_algorithms = []
            for c in choices:
                try:
                    idx = int(c) - 1
                    sel_algorithms.append(algorithms[idx][0])
                except Exception:
                    for name, _ in algorithms:
                        if c.lower() == name.lower():
                            sel_algorithms.append(name)
                            break

        # 2) choose benchmarks (or 'all')
        print("Available benchmark functions:")
        for i, (disp, name) in enumerate(benchmarks):
            print(f"  {i+1}. {disp} ({name})")
        sel = input("Enter comma-separated benchmark numbers or names to run (or 'all'): ")
        if sel.strip().lower() == 'all' or sel.strip() == '':
            selected_benchmarks = [name for _, name in benchmarks]
        else:
            choices = [s.strip() for s in sel.split(',') if s.strip()]
            selected_benchmarks = []
            for c in choices:
                try:
                    idx = int(c) - 1
                    selected_benchmarks.append(benchmarks[idx][1])
                except Exception:
                    for disp, name in benchmarks:
                        if c.lower() == disp.lower() or c.lower() == name.lower():
                            selected_benchmarks.append(name)
                            break

        # 3) choose dims mode: 'all' (run all supported dims), 'select' (pick specific dims),
        #    'priority' (pick dims to prioritise first)
        # Compute candidate dims supported by the selected algorithms (intersection)
        # If the user selected 'all' algorithms earlier, ALGO_DIM_SUPPORT defaults to broad support.
        algs_for_prompt = sel_algorithms if sel_algorithms is not None else [name for name, _ in algorithms]
        # intersection of per-algorithm supported dims
        candidate_dims = None
        for a in algs_for_prompt:
            dims = ALGO_DIM_SUPPORT.get(a, [])
            if candidate_dims is None:
                candidate_dims = set(dims)
            else:
                candidate_dims &= set(dims)
        candidate_dims = sorted(candidate_dims) if candidate_dims else []

        print("Dimension selection modes:")
        print("  1. all    -> run all supported dims for each benchmark")
        print("  2. select -> choose specific dims to run (e.g. 2 50 100)")
        print("  3. priority -> choose dims to prioritise (they'll be run first if supported)")
        mode = input("Choose dims mode (1/2/3) [default=1]: ")
        mode = mode.strip() or '1'

        def prompt_for_dims(prompt_msg, valid_set):
            # re-prompt until valid integer dims are entered or blank to cancel
            while True:
                ds = input(prompt_msg)
                if ds.strip() == '':
                    return None
                parts = [p for p in ds.split() if p.strip()]
                vals = []
                ok = True
                for p in parts:
                    try:
                        v = int(p)
                        vals.append(v)
                    except Exception:
                        print(f"  Invalid dim value: '{p}'. Please enter integers separated by spaces.")
                        ok = False
                        break
                if not ok:
                    continue
                # if valid_set provided, ensure at least one belongs to valid_set
                if valid_set is not None:
                    invalid = [v for v in vals if v not in valid_set]
                    if invalid:
                        print(f"  These dims are not supported by the selected algorithm(s): {invalid}")
                        print(f"  Supported dims for chosen algorithm(s): {sorted(valid_set)}")
                        continue
                return vals

        if mode == '2':
            prompt = f"Enter space-separated dims to run (available dims by selected algorithm(s): {candidate_dims}) or blank to cancel: "
            selected_dims = prompt_for_dims(prompt, set(candidate_dims) if candidate_dims else None)
        elif mode == '3':
            prompt = f"Enter space-separated priority dims (available dims: {candidate_dims}) or blank to cancel: "
            priority_dims = prompt_for_dims(prompt, set(candidate_dims) if candidate_dims else None)
        else:
            # default: run all supported dims (selected_dims stays None)
            selected_dims = None

        # If a single algorithm was chosen, filter benchmark dims if algorithm-specific
        # NOTE: We approximate algorithm-specific dim support via the dim_support mapping later when running.
        # We will apply filtering at runtime per benchmark.

    # Respect CLI over interactive selections. If CLI provided values, they take precedence.
    if args.algorithms:
        selected_algs = args.algorithms
    else:
        # if interactive flow created sel_algorithms, use it
        if 'sel_algorithms' in locals():
            selected_algs = sel_algorithms

    if args.dims:
        selected_dims = args.dims
    # else: keep selected_dims set by interactive flow (may be None for 'all')

    # priority dims: prefer CLI if provided, otherwise keep interactive value if set
    if args.priority_dims is not None:
        priority_dims = args.priority_dims
    else:
        if 'priority_dims' in locals():
            # keep interactive value
            priority_dims = priority_dims
        else:
            priority_dims = None


    def plot_search_history_2d(json_path, fig_path, title=None):
        try:
            with open(json_path, "r") as jf:
                data = json.load(jf)
            pop_xy = data["pop_xy"]
            gens = data["gen"]
            plt.figure(figsize=(6,6))
            # fade older generations
            for i, pts in enumerate(pop_xy):
                if pts is None:
                    continue
                pts = np.asarray(pts)
                alpha = 0.15 + 0.75 * (i + 1) / (len(pop_xy))
                plt.scatter(pts[:,0], pts[:,1], s=6, alpha=alpha)
            plt.xlabel("x1")
            plt.ylabel("x2")
            if title:
                plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
        except Exception as e:
            print(f"Search history plot failed for {json_path}: {e}")


    # Wrap original run_all_dims to accept filters via closure
    def run_all_dims_filtered():
        only_plot_run1 = True  # plot individual search-history PNGs only for run 1

        # selections computed earlier
        sel_algs = selected_algs
        sel_dims = selected_dims
        sel_benchmarks = selected_benchmarks

        import os, sys, csv, glob, time, json
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        dim_support = {
            "SchafferN2": [2],
            "SchafferN4": [2],
            "Ackley": [2, 50, 100],
            "Griewank": [2, 50, 100],
            "Rastrigin": [2, 50, 100],
            "Shubert": [2],
            "Eggholder": [2],
            "HolderTable": [2],
            "Levy": [2, 50, 100],
            "Schwefel": [2, 50, 100],
        }

        checkpoint_file = os.path.join(os.path.dirname(__file__), '../test4_v2_checkpoint_results/experiment_checkpoint.json')
        checkpoint = load_checkpoint(checkpoint_file) or {}

        # consistent color per algorithm
        alg_color_map = {
            "TSD":  "#211fb4",
            "FCSA": "#d62727",
            "IICO": "#67bd75",
        }

        # -------- helper: single-json scatter (kept for per-run PNGs) --------
        def plot_search_history_2d(
            json_path,
            fig_path,
            title=None,
            color="#007ACC",
            dot_size=35,
            frame_stride=10,
            point_stride=5,
            highlight_last=True
        ):
            try:
                with open(json_path, "r") as jf:
                    data = json.load(jf)

                pop_xy_frames = data.get("pop_xy", [])
                if not isinstance(pop_xy_frames, list) or len(pop_xy_frames) == 0:
                    print(f"[search] no frames in {json_path}")
                    return

                plt.figure(figsize=(6, 6))

                last_non_empty = None
                frames = pop_xy_frames[::max(1, int(frame_stride))]
                for pts in frames:
                    if pts is None:
                        continue
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
                        continue
                    arr = arr[::max(1, int(point_stride)), :2]
                    plt.scatter(arr[:, 0], arr[:, 1], s=dot_size, alpha=0.7, color=color, edgecolors="none")
                    last_non_empty = arr

                if highlight_last and last_non_empty is not None and last_non_empty.shape[0] > 0:
                    centroid = last_non_empty.mean(axis=0)
                    plt.scatter([centroid[0]], [centroid[1]], s=90, marker="*", color="black", alpha=0.95)

                plt.xlabel("x1")
                plt.ylabel("x2")
                if title:
                    plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"Search history plot failed for {json_path}: {e}")

        # -------- NEW helper: combined scatter for TSD/IICO/FCSA on one axes --------
        def plot_combined_search_history_2d(
            json_paths_by_alg: dict,
            fig_path: str,
            title: str,
            color_map: dict,
            dot_size=35,
            frame_stride=10,
            point_stride=5,
            highlight_last=True
        ):
            """
            json_paths_by_alg: {"TSD": [json1, json2, ...], "IICO": [...], "FCSA": [...]}
            Renders all available JSONs for the listed algs onto one figure.
            """
            plt.figure(figsize=(6.5, 6.5))
            legend_handles = []
            from matplotlib.lines import Line2D

            for alg in ["TSD", "IICO", "FCSA"]:
                paths = json_paths_by_alg.get(alg, [])
                if not paths:
                    continue

                color = color_map.get(alg, "#444")
                last_centroid = None
                any_points = False

                for jp in paths:
                    try:
                        with open(jp, "r") as jf:
                            data = json.load(jf)
                        pop_xy_frames = data.get("pop_xy", [])
                        if not isinstance(pop_xy_frames, list) or len(pop_xy_frames) == 0:
                            continue

                        # stride frames and points for readability
                        for pts in pop_xy_frames[::max(1, int(frame_stride))]:
                            if pts is None:
                                continue
                            arr = np.asarray(pts, dtype=float)
                            if arr.ndim != 2 or arr.shape[1] < 2 or arr.size == 0:
                                continue
                            arr = arr[::max(1, int(point_stride)), :2]
                            plt.scatter(arr[:, 0], arr[:, 1], s=dot_size, alpha=0.65, color=color, edgecolors="none")
                            any_points = True
                            last_centroid = arr.mean(axis=0)
                    except Exception as e:
                        print(f"[combined] failed to read {jp}: {e}")

                # highlight last centroid per algorithm
                if highlight_last and last_centroid is not None:
                    plt.scatter([last_centroid[0]], [last_centroid[1]], s=95, marker="*", color="black", alpha=0.95)

                if any_points:
                    legend_handles.append(Line2D([0], [0], marker='o', linestyle='',
                                                markersize=max(6, dot_size/6),
                                                markerfacecolor=color, markeredgecolor=color,
                                                label=alg))

            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title(title)
            if legend_handles:
                plt.legend(handles=legend_handles, loc="best")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

        for bench_disp, bench_name in benchmarks:
            if sel_benchmarks is not None and bench_name not in sel_benchmarks:
                continue

            supported_dims_local = dim_support.get(bench_name, [2])

            if sel_dims is not None:
                supported_dims_local = [d for d in supported_dims_local if d in sel_dims]
                if not supported_dims_local:
                    print(f"  Skipping {bench_disp}: no selected dimensions supported")
                    continue

            if 'priority_dims' in globals() and priority_dims:
                pri_supported = [d for d in priority_dims if d in supported_dims_local]
                if pri_supported:
                    remaining = [d for d in supported_dims_local if d not in pri_supported]
                    supported_dims_local = pri_supported + remaining

            func = get_function_by_name(bench_name)

            for dim in supported_dims_local:
                fig_dir = os.path.join(os.path.dirname(__file__), f'../test4_v2_results/dim_{dim}')
                os.makedirs(fig_dir, exist_ok=True)
                log_file = os.path.join(fig_dir, f'Test_{dim}_log.txt')

                with open(log_file, "a") as f:
                    f.write(f"\n=== Benchmark: {bench_disp} | Dimension: {dim} ===\n")
                    print(f"\nBenchmark: {bench_disp} (dim={dim})")

                    algs_to_run = [alg_name for alg_name, _ in algorithms if (sel_algs is None or alg_name in sel_algs)]
                    print(f"  Algorithms to run: {algs_to_run}")
                    print(f"  Ordered dims for this benchmark: {supported_dims_local}")
                    f.write(f"  Algorithms: {algs_to_run}\n")
                    f.write(f"  Ordered dims: {supported_dims_local}\n")

                    # bounds
                    try:
                        dummy_x = np.zeros(dim)
                        _, lb, ub = func(dummy_x)
                    except Exception as e:
                        print(f"  Skipping {bench_disp}: incompatible with dim={dim} ({e})")
                        f.write(f"  Skipped {bench_disp}: incompatible with dim={dim} ({e})\n")
                        continue
                    bounds = [(lb, ub)] * dim

                    # figures for this (benchmark, dim)
                    conv_fig = plt.figure(figsize=(10, 6))  # Convergence
                    avg_fig = plt.figure(figsize=(10, 6))   # Average Fitness

                    n_runs = 100
                    closeness_vals = {}

                    # NEW: store per-alg list of JSONs to combine later
                    combined_jsons = {"TSD": [], "IICO": [], "FCSA": []}

                    for alg_name, runner in algorithms:
                        if alg_name not in algs_to_run:
                            continue

                        print(f"  Algorithm: {alg_name}")
                        histories = []
                        final_vals = []

                        ck_key = f"dim{dim}_{bench_name}_{alg_name}"
                        ck = checkpoint.get(ck_key, {"completed_runs": []})
                        completed = ck.get("completed_runs", [])

                        prev_runs = [ent.get("run") for ent in completed if isinstance(ent.get("run"), int)]
                        start_run = (max(prev_runs) + 1) if prev_runs else 0
                        alg_start_time = time.time()

                        def obj(x):
                            y, _, _ = func(x)
                            return y

                        for run in range(start_run, n_runs):
                            run_start = time.time()
                            eval_budget = 350_000
                            per_run_seed = args.base_seed + run

                            want_positions = (dim == 2)
                            prog = ProgressLogger(want_positions=want_positions)

                            # execute runner with progress if possible
                            try:
                                result = runner(obj, bounds, dim, eval_budget, seed=per_run_seed, progress=prog, bench_func=func)
                            except TypeError:
                                try:
                                    result = runner(obj, bounds, dim, eval_budget, seed=per_run_seed, bench_func=func)
                                except TypeError:
                                    if alg_name == "IICO":
                                        pop_size = DEFAULT_POP
                                        result = runner(obj, bounds, dim, eval_budget, pop_size, func, seed=per_run_seed)
                                    else:
                                        result = runner(obj, bounds, seed=per_run_seed)

                            # normalize
                            if isinstance(result, tuple) and len(result) == 2:
                                history, _ = result
                            else:
                                history = result
                            history = np.asarray(history, dtype=float).reshape(-1)
                            run_time = time.time() - run_start

                            histories.append(history)
                            final_vals.append(float(history[-1]))

                            # per-run dir
                            per_run_dir = os.path.join(fig_dir, f"runs_{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}")
                            os.makedirs(per_run_dir, exist_ok=True)

                            # metrics
                            gens_m = prog.metrics["gen"]
                            if not gens_m:
                                gens_m = list(range(len(history)))
                                best_m = history.tolist()
                                mean_m = [float('nan')] * len(history)
                                evals_m = [-1] * len(history)
                                time_m = [float('nan')] * len(history)
                            else:
                                best_m = prog.metrics["best_fitness"]
                                mean_m = prog.metrics["mean_fitness"]
                                evals_m = prog.metrics["evals"]
                                time_m = prog.metrics["time_sec"]

                            # per-gen CSV
                            per_gen_csv = os.path.join(per_run_dir, f"run_{run+1:02d}_per_gen.csv")
                            with open(per_gen_csv, "w", newline="") as cf:
                                wr = csv.writer(cf)
                                wr.writerow(["gen", "best_fitness", "mean_fitness", "evals", "time_sec"])
                                for j in range(len(gens_m)):
                                    wr.writerow([gens_m[j], best_m[j], mean_m[j], evals_m[j], time_m[j]])

                            # 2D search history JSON + per-run PNG
                            # 2D search history
                            if want_positions:
                                frames = sum(p is not None for p in prog.metrics["pop_xy"])
                                print(f"  [search] dim={dim} alg={alg_name} run={run+1} frames={frames} want_positions={want_positions}")

                                # always write JSON for post-processing (every run)
                                search_json = os.path.join(per_run_dir, f"run_{run+1:02d}_search_history.json")
                                try:
                                    with open(search_json, "w") as jf:
                                        json.dump(
                                            {
                                                "gen": prog.metrics["gen"],
                                                "pop_xy": prog.metrics["pop_xy"],  # list of per-gen arrays (or None)
                                                "bounds": [lb, ub],
                                            },
                                            jf,
                                        )

                                                                        # NEW: feed the combined plot ONLY with run 1 (index 0) JSONs for TSD/IICO/FCSA
                                    if want_positions and frames > 0 and alg_name in ("TSD", "IICO", "FCSA") and run == 0:
                                        combined_jsons[alg_name].append(search_json)
                                except Exception as e:
                                    print(f"  [search] failed to write JSON: {e}")

                                # only render PNG for run 1 (run index 0) if frames exist
                                if frames > 0 and (not only_plot_run1 or run == 0):
                                    try:
                                        plot_search_history_2d(
                                            search_json,
                                            os.path.join(per_run_dir, f"run_{run+1:02d}_search_history.png"),
                                            title=f"{alg_name} • {bench_disp} dim={dim} run={run+1}",
                                            color=alg_color_map.get(alg_name, "#444"),
                                            dot_size=35,
                                            frame_stride=10,
                                            point_stride=5,
                                            highlight_last=True,
                                        )
                                        print(f"  [search] saved search history PNG for run {run+1}")
                                    except Exception as e:
                                        print(f"  [search] plot failed: {e}")
                                else:
                                    if frames == 0:
                                        print("  [search] no non-empty frames; PNG skipped")
                                    elif only_plot_run1 and run != 0:
                                        print("  [search] PNG skipped (only plotting run 1 by design)")

                            # scalability row
                            scal_csv = os.path.join(fig_dir, f"scalability_{bench_disp.replace(' ', '_')}.csv")
                            new_file = not os.path.exists(scal_csv)
                            with open(scal_csv, "a", newline="") as sf:
                                wrs = csv.writer(sf)
                                if new_file:
                                    wrs.writerow(["algorithm", "benchmark", "dim", "run", "final_best", "time_sec"])
                                wrs.writerow([alg_name, bench_name, dim, run+1, float(history[-1]), run_time])

                            # combined best-fitness runs CSV (padded)
                            histories_to_save = pad_histories(histories)
                            combined_csv = os.path.join(fig_dir, f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv")
                            with open(combined_csv, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(["generation"] + [f"run_{i+1}" for i in range(histories_to_save.shape[0])])
                                for gen_idx in range(histories_to_save.shape[1]):
                                    row = [gen_idx] + [float(histories_to_save[r, gen_idx]) for r in range(histories_to_save.shape[0])]
                                    writer.writerow(row)

                            # checkpoint
                            ck.setdefault("completed_runs", []).append({
                                "run": run,
                                "final": float(history[-1]),
                                "history_len": len(history),
                                "time_sec": run_time
                            })
                            checkpoint[ck_key] = ck
                            save_checkpoint(checkpoint_file, checkpoint)

                            runs_done = len(ck.get("completed_runs", []))
                            runs_left = max(0, n_runs - runs_done)
                            elapsed = time.time() - alg_start_time
                            avg_per_run = elapsed / runs_done if runs_done > 0 else None
                            eta = avg_per_run * runs_left if avg_per_run is not None else None
                            best_so_far = np.min(final_vals) if final_vals else float('inf')
                            sys.stdout.write(
                                f"    {alg_name} run {runs_done}/{n_runs} | best={best_so_far:.2e} | ETA={format_eta(eta)}\r"
                            )
                            sys.stdout.flush()
                        # end runs

                        # Convergence + Average Fitness + Closeness
                        if histories:
                            histories = pad_histories(histories)
                            mean_curve = np.mean(histories, axis=0)
                            std_curve  = np.std(histories, axis=0)
                            gens = np.arange(len(mean_curve))

                            plt.figure(conv_fig.number)
                            plt.plot(gens, mean_curve, label=alg_name)
                            plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

                            # Average fitness (from per-run CSVs)
                            mean_fit_runs = []
                            for ridx in range(len(final_vals)):
                                per_run_dir = os.path.join(fig_dir, f"runs_{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}")
                                per_gen_csv = os.path.join(per_run_dir, f"run_{ridx+1:02d}_per_gen.csv")
                                if os.path.exists(per_gen_csv):
                                    try:
                                        df = pd.read_csv(per_gen_csv)
                                        if "mean_fitness" in df.columns and len(df) > 0:
                                            mean_fit_runs.append(df["mean_fitness"].values.astype(float))
                                    except Exception:
                                        pass
                            if mean_fit_runs:
                                maxL = max(len(a) for a in mean_fit_runs)
                                mf_pad = []
                                for a in mean_fit_runs:
                                    if len(a) < maxL:
                                        a = np.concatenate([a, np.full(maxL - len(a), a[-1])])
                                    mf_pad.append(a)
                                mf_pad = np.vstack(mf_pad)
                                mf_mean = np.mean(mf_pad, axis=0)
                                mf_std  = np.std(mf_pad, axis=0)
                                gens_mf = np.arange(len(mf_mean))

                                plt.figure(avg_fig.number)
                                plt.plot(gens_mf, mf_mean, label=alg_name)
                                plt.fill_between(gens_mf, mf_mean - mf_std, mf_mean + mf_std, alpha=0.2)

                            # How close to optimum
                            opt = KNOWN_OPTIMA.get(bench_name, None)
                            if opt is not None and len(final_vals) > 0:
                                closeness_vals[alg_name] = float(np.mean(final_vals)) - float(opt)
                            else:
                                closeness_vals[alg_name] = np.nan
                    # end algorithm loop

                    # finalize and save Convergence
                    plt.figure(conv_fig.number)
                    plt.title(f"Convergence Curve • {bench_disp} (dim={dim}, {n_runs} runs)")
                    plt.xlabel("Iteration / Generation")
                    plt.ylabel("Best Fitness")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    conv_path = os.path.join(fig_dir, f"convergence_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                    plt.savefig(conv_path)
                    plt.close()
                    print(f"  Saved: {conv_path}")

                    # finalize and save Average Fitness
                    plt.figure(avg_fig.number)
                    plt.title(f"Average Fitness • {bench_disp} (dim={dim}, {n_runs} runs)")
                    plt.xlabel("Iteration / Generation")
                    plt.ylabel("Average Fitness")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    avg_path = os.path.join(fig_dir, f"avgfitness_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                    plt.savefig(avg_path)
                    plt.close()
                    print(f"  Saved: {avg_path}")

                    # How close to optimum (bar)
                    valid_close = {k: v for k, v in closeness_vals.items() if not (v is None or (isinstance(v, float) and np.isnan(v)))}
                    if valid_close:
                        algs = list(valid_close.keys())
                        vals = [valid_close[a] for a in algs]
                        plt.figure(figsize=(8, 5))
                        plt.bar(algs, vals)
                        plt.axhline(0.0, linestyle="--", linewidth=1)
                        plt.ylabel("Mean(final best) - Optimum")
                        plt.xlabel("Algorithm")
                        plt.title(f"How Close to Optimum • {bench_disp} (dim={dim})")
                        plt.grid(axis="y")
                        plt.tight_layout()
                        close_path = os.path.join(fig_dir, f"closeness_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                        plt.savefig(close_path)
                        plt.close()
                        close_csv = os.path.join(fig_dir, f"closeness_{bench_disp.replace(' ', '_')}_dim{dim}.csv")
                        with open(close_csv, "w", newline="") as cf:
                            wr = csv.writer(cf)
                            wr.writerow(["algorithm", "mean_final_minus_optimum"])
                            for a, v in zip(algs, vals):
                                wr.writerow([a, v])
                        print(f"  Saved: {close_path} and CSV")

                    # --- NEW: after running all algs, write a combined TSD+IICO+FCSA search-history plot (dim==2 only) ---
                    if dim == 2:
                        have_any = sum(len(v) for v in combined_jsons.values()) > 0
                        if have_any:
                            combined_png = os.path.join(
                                fig_dir,
                                f"search_history_combined_TSD_IICO_FCSA_{bench_disp.replace(' ', '_')}_dim{dim}.png"
                            )
                            plot_combined_search_history_2d(
                                json_paths_by_alg=combined_jsons,
                                fig_path=combined_png,
                                title=f"Search History (Combined) • {bench_disp} dim={dim}",
                                color_map=alg_color_map,
                                dot_size=35,
                                frame_stride=10,
                                point_stride=5,
                                highlight_last=True
                            )
                            print(f"  Saved: {combined_png}")

                    # Scalability (only for scalable functions: dims include 2,50,100)
                    dims_for_bench = dim_support.get(bench_name, [2])
                    is_scalable = set([2, 50, 100]).issubset(set(dims_for_bench))
                    if is_scalable:
                        rows = []
                        root_results = os.path.dirname(fig_dir)
                        for ddir in glob.glob(os.path.join(root_results, "dim_*")):
                            candidate = os.path.join(ddir, f"scalability_{bench_disp.replace(' ', '_')}.csv")
                            if os.path.exists(candidate):
                                with open(candidate, "r") as rf:
                                    rdr = csv.DictReader(rf)
                                    for row in rdr:
                                        rows.append(row)
                        if rows:
                            df_scal = pd.DataFrame(rows)
                            df_scal["dim"] = pd.to_numeric(df_scal["dim"], errors="coerce")
                            df_scal["final_best"] = pd.to_numeric(df_scal["final_best"], errors="coerce")
                            df_scal["time_sec"] = pd.to_numeric(df_scal["time_sec"], errors="coerce")

                            plt.figure(figsize=(8, 5))
                            for alg in sorted(df_scal["algorithm"].dropna().unique()):
                                df_a = df_scal[df_scal["algorithm"] == alg]
                                g = df_a.groupby("dim", as_index=True).agg({"final_best": "mean"}).sort_index()
                                if not g.empty:
                                    plt.plot(g.index.values, g["final_best"].values, marker="o", label=alg)
                            plt.xlabel("Dimension")
                            plt.ylabel("Average Best Fitness")
                            plt.title(f"Scalability • {bench_disp} (y: avg best fitness)")
                            plt.grid(True)
                            plt.legend()
                            out_path = os.path.join(root_results, f"scalability_avgBest_{bench_disp.replace(' ', '_')}.png")
                            plt.tight_layout()
                            plt.savefig(out_path)
                            plt.close()
                            print(f"  Saved: {out_path}")

                    f.write("\n")



        # end for all benchmarks/dims


    # update todo: mark first as completed
    run_all_dims_filtered()
