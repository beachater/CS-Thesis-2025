
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


# Experiment defaults (use these to ensure fair comparisons)
DEFAULT_POP = 60
DEFAULT_N_SELECT = 15
DEFAULT_N_CLONES = 5
DEFAULT_A_FRAC = 0.15
DEFAULT_R = 2.0
DEFAULT_MAX_GENS = 1000
DEFAULT_BASE_SEED = 1000




def run_iico(obj, bounds, dim, max_evals, seed=None, progress=None, bench_func=None):
    # IICO signature: iico(fun, max_FEs, n, dim, ...)
    pop_size = DEFAULT_POP
    def iico_obj(x):
        return obj(np.array(x))
    def iico_fun(x):
        xa = np.array(x)
        y, lb, ub = bench_func(xa)
        return y, lb, ub
    best_fitness_list, gbest, info = IICO_func(iico_fun, int(max_evals), pop_size, dim, seed=seed, progress=progress)
    return np.array(info.get('history', best_fitness_list))

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
        progress=progress,
        max_gens=DEFAULT_MAX_GENS,
        max_evals=int(max_evals),
    )
    x_best, f_best, info = opt.optimize()
    return np.array(info.get('history', [f_best]))



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
        opt = HybridFCSA_IICO(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))

algorithms = [

    ("HybridReformed", run_hybrid_reformed_spark),
    ("TSD", run_tsd),
    ("FCSA", run_fcsa), 
    ("IICO", run_iico),




]


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


def run_all_dims():
    import matplotlib.pyplot as plt
    import os
    import csv

    # Define benchmark compatibility by dimension
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

    # checkpoint file to resume interrupted experiments
    checkpoint_file = os.path.join(os.path.dirname(__file__), '../test4_checkpoint_results/experiment_checkpoint.json')
    checkpoint = load_checkpoint(checkpoint_file) or {}

    for bench_disp, bench_name in benchmarks:
        supported_dims = dim_support.get(bench_name, [2])  # default: only 2D
        for dim in supported_dims:
            fig_dir = os.path.join(os.path.dirname(__file__), f'../test4_fig_results/dim_{dim}')
            os.makedirs(fig_dir, exist_ok=True)
            log_file = os.path.join(fig_dir, f'Test_{dim}_log.txt')

            with open(log_file, "a") as f:
                f.write(f"\n=== Benchmark: {bench_disp} | Dimension: {dim} ===\n")
                print(f"\nBenchmark: {bench_disp} (dim={dim})")

                func = get_function_by_name(bench_name)

                try:
                    dummy_x = np.zeros(dim)
                    _, lb, ub = func(dummy_x)
                except Exception as e:
                    print(f"  Skipping {bench_disp}: incompatible with dim={dim} ({e})")
                    f.write(f"  Skipped {bench_disp}: incompatible with dim={dim} ({e})\n")
                    continue

                bounds = [(lb, ub)] * dim

                def obj(x):
                    y, _, _ = func(x)
                    return y

                plt.figure(figsize=(10, 6))
                n_runs = 2  # can be adjusted globally if needed
                for alg_name, runner in algorithms:
                    print(f"  Algorithm: {alg_name}")
                    histories = []
                    final_vals = []

                    ck_key = f"dim{dim}_{bench_name}_{alg_name}"
                    ck = checkpoint.get(ck_key, {"completed_runs": []})
                    # preload completed runs (final values and histories if CSV exists)
                    completed = ck.get("completed_runs", [])
                    # If previous runs exist, try to load combined CSV of past runs to reconstruct histories
                    combined_csv = os.path.join(fig_dir, f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv")
                    if os.path.exists(combined_csv):
                        try:
                            with open(combined_csv, 'r', newline='') as csvfile_prev:
                                reader_prev = csv.reader(csvfile_prev)
                                header = next(reader_prev, [])
                                # header: ['generation', 'run_1', 'run_2', ...]
                                cols = []
                                for _ in range(max(0, len(header) - 1)):
                                    cols.append([])
                                for row in reader_prev:
                                    # row[0] is generation
                                    for ci, val in enumerate(row[1:]):
                                        try:
                                            cols[ci].append(float(val))
                                        except Exception:
                                            cols[ci].append(float('nan'))
                                for col in cols:
                                    histories.append(np.array(col))
                                    if len(col) > 0:
                                        final_vals.append(float(col[-1]))
                        except Exception:
                            # fallback: if CSV can't be read, leave histories empty and rely on checkpoint metadata
                            histories = []
                    else:
                        # no combined CSV present; rely on checkpoint entries for final values
                        for ent in completed:
                            final_val = float(ent.get("final", float('nan')))
                            final_vals.append(final_val)

                    # compute next run index robustly
                    prev_runs = [ent.get("run") for ent in completed if isinstance(ent.get("run"), int)]
                    start_run = (max(prev_runs) + 1) if prev_runs else 0
                    alg_start_time = time.time()

                    for run in range(start_run, n_runs):
                        run_start = time.time()
                        eval_budget = 350_000

                        # call runner with standard signature: (obj, bounds, dim, max_evals, seed=..., progress=..., bench_func=...)
                        try:
                            # derive per-run seed from default base to keep this function safe when called independently
                            per_run_seed = DEFAULT_BASE_SEED + run
                            result = runner(obj, bounds, dim, eval_budget, seed=per_run_seed, progress=None, bench_func=func)
                        except TypeError:
                            # fallback: some runners (IICO) expect pop_size argument; handle explicitly
                            if alg_name == "IICO":
                                pop_size = 60
                                result = runner(obj, bounds, dim, eval_budget, pop_size, func, seed=run, progress=None)
                            else:
                                # last-resort: try old signature
                                try:
                                    result = runner(obj, bounds, seed=run, progress=None)
                                except TypeError:
                                    result = runner(obj, bounds, seed=run)

                        # handle returned history
                        if isinstance(result, tuple) and len(result) == 2:
                            history, _ = result
                        else:
                            history = result

                        history = np.array(history)
                        histories.append(history)
                        final_vals.append(float(history[-1]))


                        # --- Save combined CSV of all runs (overwrite single file) ---
                        histories_to_save = pad_histories(histories)
                        combined_csv = os.path.join(fig_dir, f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv")
                        with open(combined_csv, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            header = ["generation"] + [f"run_{i+1}" for i in range(histories_to_save.shape[0])]
                            writer.writerow(header)
                            for gen_idx in range(histories_to_save.shape[1]):
                                row = [gen_idx] + [float(histories_to_save[r, gen_idx]) for r in range(histories_to_save.shape[0])]
                                writer.writerow(row)

                        # checkpoint update
                        ck.setdefault("completed_runs", []).append({
                            "run": run,
                            "final": float(history[-1]),
                            "history_len": len(history)
                        })
                        checkpoint[ck_key] = ck
                        save_checkpoint(checkpoint_file, checkpoint)

                        # progress
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

                    print("")
                    if not histories:
                        print(f"    No runs completed for {alg_name}")
                        f.write(f"  {alg_name}: No runs completed\n")
                        continue

                    histories = pad_histories(histories)
                    mean_curve = np.mean(histories, axis=0)
                    std_curve = np.std(histories, axis=0)
                    gens = np.arange(len(mean_curve))
                    plt.plot(gens, mean_curve, label=alg_name)
                    plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

                    final_vals_arr = np.array(final_vals)
                    avg_final = float(np.mean(final_vals_arr))
                    max_final = float(np.max(final_vals_arr))
                    min_final = float(np.min(final_vals_arr))

                    f.write(f"  {alg_name}: mean={avg_final:.2e}, max={max_final:.2e}, min={min_final:.2e}\n")
                    print(f"    mean_final: {avg_final:.2e}, max: {max_final:.2e}, min: {min_final:.2e}")

                    summary = {
                        "algorithm": alg_name,
                        "benchmark": bench_name,
                        "benchmark_display": bench_disp,
                        "dim": dim,
                        "n_runs": len(final_vals_arr),
                        "mean_final": avg_final,
                        "max_final": max_final,
                        "min_final": min_final,
                        "completed_runs": ck.get("completed_runs", [])
                    }
                    summary_path = os.path.join(fig_dir, f"summary_{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}.json")
                    with open(summary_path, 'w') as sfh:
                        json.dump(summary, sfh, indent=2)

                    f.flush()

                # --- Plot and save convergence ---
                plt.title(f"Convergence Curves: {bench_disp} (dim={dim}, {n_runs} runs)")
                plt.xlabel("Generation")
                plt.ylabel("Best Fitness")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                fig_path = os.path.join(fig_dir, f"convergence_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                plt.savefig(fig_path)
                plt.close()
                print(f"  Saved plot: {fig_path}")
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark experiments and save results")
    parser.add_argument("--algorithms", "-a", nargs="*", help="Algorithm names to run (default: all)")
    parser.add_argument("--benchmarks", "-b", nargs="*", help="Benchmark functions to run (names or numbers from the list). If provided, only these benchmarks are run (default: all)")
    parser.add_argument("--dims", "-d", nargs="*", type=int, help="Dimensions to run (e.g. 2 50 100). Default: auto-detect per benchmark")
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
                    # try matching by display or canonical name
                    for disp, name in benchmarks:
                        if c.lower() == disp.lower() or c.lower() == name.lower():
                            selected_benchmarks.append(name)
                            break

    if args.algorithms:
        selected_algs = args.algorithms
    if args.dims:
        selected_dims = args.dims

    # Wrap original run_all_dims to accept filters via closure
    def run_all_dims_filtered():
        # make selected lists available inside
        sel_algs = selected_algs
        sel_dims = selected_dims
        sel_benchmarks = selected_benchmarks

        # Modified loop to respect selections
        import matplotlib.pyplot as plt
        import os
        import csv

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
        checkpoint_file = os.path.join(os.path.dirname(__file__), '../test4_checkpoint_results/experiment_checkpoint.json')
        checkpoint = load_checkpoint(checkpoint_file) or {}

        for bench_disp, bench_name in benchmarks:
            # if benchmarks were selected, skip those not selected
            if sel_benchmarks is not None and bench_name not in sel_benchmarks:
                continue
            supported_dims_local = dim_support.get(bench_name, [2])
            # apply selected dims filter if provided
            if sel_dims is not None:
                supported_dims_local = [d for d in supported_dims_local if d in sel_dims]
                if not supported_dims_local:
                    print(f"  Skipping {bench_disp}: no selected dimensions supported")
                    continue

            for dim in supported_dims_local:
                fig_dir = os.path.join(os.path.dirname(__file__), f'../test4_fig_results/dim_{dim}')
                os.makedirs(fig_dir, exist_ok=True)
                log_file = os.path.join(fig_dir, f'Test_{dim}_log.txt')

                with open(log_file, "a") as f:
                    f.write(f"\n=== Benchmark: {bench_disp} | Dimension: {dim} ===\n")
                    print(f"\nBenchmark: {bench_disp} (dim={dim})")

                    func = get_function_by_name(bench_name)

                    try:
                        dummy_x = np.zeros(dim)
                        _, lb, ub = func(dummy_x)
                    except Exception as e:
                        print(f"  Skipping {bench_disp}: incompatible with dim={dim} ({e})")
                        f.write(f"  Skipped {bench_disp}: incompatible with dim={dim} ({e})\n")
                        continue

                    bounds = [(lb, ub)] * dim

                    def obj(x):
                        y, _, _ = func(x)
                        return y

                    plt.figure(figsize=(10, 6))
                    n_runs = 100
                    # iterate only selected algorithms if provided
                    for alg_name, runner in algorithms:
                        if sel_algs is not None and alg_name not in sel_algs:
                            continue
                        print(f"  Algorithm: {alg_name}")
                        histories = []
                        final_vals = []

                        ck_key = f"dim{dim}_{bench_name}_{alg_name}"
                        ck = checkpoint.get(ck_key, {"completed_runs": []})
                        completed = ck.get("completed_runs", [])
                        combined_csv = os.path.join(fig_dir, f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv")
                        if os.path.exists(combined_csv):
                            try:
                                with open(combined_csv, 'r', newline='') as csvfile_prev:
                                    reader_prev = csv.reader(csvfile_prev)
                                    header = next(reader_prev, [])
                                    cols = []
                                    for _ in range(max(0, len(header) - 1)):
                                        cols.append([])
                                    for row in reader_prev:
                                        for ci, val in enumerate(row[1:]):
                                            try:
                                                cols[ci].append(float(val))
                                            except Exception:
                                                cols[ci].append(float('nan'))
                                    for col in cols:
                                        histories.append(np.array(col))
                                        if len(col) > 0:
                                            final_vals.append(float(col[-1]))
                            except Exception:
                                histories = []
                        else:
                            for ent in completed:
                                final_val = float(ent.get("final", float('nan')))
                                final_vals.append(final_val)

                        prev_runs = [ent.get("run") for ent in completed if isinstance(ent.get("run"), int)]
                        start_run = (max(prev_runs) + 1) if prev_runs else 0
                        alg_start_time = time.time()

                        for run in range(start_run, n_runs):
                            run_start = time.time()
                            eval_budget = 350_000

                            # derive per-run seed from CLI-provided base
                            per_run_seed = args.base_seed + run
                            try:
                                result = runner(obj, bounds, dim, eval_budget, seed=per_run_seed, progress=None, bench_func=func)
                            except TypeError:
                                if alg_name == "IICO":
                                    pop_size = DEFAULT_POP
                                    result = runner(obj, bounds, dim, eval_budget, pop_size, func, seed=per_run_seed, progress=None)
                                else:
                                    try:
                                        result = runner(obj, bounds, seed=per_run_seed, progress=None)
                                    except TypeError:
                                        result = runner(obj, bounds, seed=per_run_seed)

                            if isinstance(result, tuple) and len(result) == 2:
                                history, _ = result
                            else:
                                history = result

                            history = np.array(history)
                            histories.append(history)
                            final_vals.append(float(history[-1]))

                            histories_to_save = pad_histories(histories)
                            combined_csv = os.path.join(fig_dir, f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv")
                            with open(combined_csv, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                header = ["generation"] + [f"run_{i+1}" for i in range(histories_to_save.shape[0])]
                                writer.writerow(header)
                                for gen_idx in range(histories_to_save.shape[1]):
                                    row = [gen_idx] + [float(histories_to_save[r, gen_idx]) for r in range(histories_to_save.shape[0])]
                                    writer.writerow(row)

                            ck.setdefault("completed_runs", []).append({
                                "run": run,
                                "final": float(history[-1]),
                                "history_len": len(history)
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

                        print("")
                        if not histories:
                            print(f"    No runs completed for {alg_name}")
                            f.write(f"  {alg_name}: No runs completed\n")
                            continue

                        histories = pad_histories(histories)
                        mean_curve = np.mean(histories, axis=0)
                        std_curve = np.std(histories, axis=0)
                        gens = np.arange(len(mean_curve))
                        plt.plot(gens, mean_curve, label=alg_name)
                        plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

                        final_vals_arr = np.array(final_vals)
                        avg_final = float(np.mean(final_vals_arr))
                        max_final = float(np.max(final_vals_arr))
                        min_final = float(np.min(final_vals_arr))

                        f.write(f"  {alg_name}: mean={avg_final:.2e}, max={max_final:.2e}, min={min_final:.2e}\n")
                        print(f"    mean_final: {avg_final:.2e}, max: {max_final:.2e}, min: {min_final:.2e}")

                        summary = {
                            "algorithm": alg_name,
                            "benchmark": bench_name,
                            "benchmark_display": bench_disp,
                            "dim": dim,
                            "n_runs": len(final_vals_arr),
                            "mean_final": avg_final,
                            "max_final": max_final,
                            "min_final": min_final,
                            "completed_runs": ck.get("completed_runs", [])
                        }
                        summary_path = os.path.join(fig_dir, f"summary_{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}.json")
                        with open(summary_path, 'w') as sfh:
                            json.dump(summary, sfh, indent=2)

                        f.flush()

                    plt.title(f"Convergence Curves: {bench_disp} (dim={dim}, {n_runs} runs)")
                    plt.xlabel("Generation")
                    plt.ylabel("Best Fitness")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    fig_path = os.path.join(fig_dir, f"convergence_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                    plt.savefig(fig_path)
                    plt.close()
                    print(f"  Saved plot: {fig_path}")
                    f.write("\n")

    # update todo: mark first as completed
    run_all_dims_filtered()
