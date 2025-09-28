
import numpy as np
import matplotlib.pyplot as plt
from benchmark import get_function_by_name

# Algorithm runners: (name, runner function)
# --- Import IICO and HybridCSA variants ---
from IICO import iico as IICO_func
from reformed_hybrid import HybridRolePartitioned
from hybrid_top import HybridCSAOriginal_sbm
from fscsa import FCSA







# List of benchmark function names and their canonical names in benchmark.py
benchmarks = [
    ("Ackley", "Ackley"),
    ("Griewank", "Griewank"),
    ("Rastrigin", "Rastrigin"),
    ("Shubert", "Shubert_06"),
    ("Eggholder", "Eggcrate"),  # Eggholder is not present, using Eggcrate as a placeholder
    ("Holdertable", "HolderTable"),
    ("Schaffer 01", "ModifiedSchaffer_01"),
    ("Schaffer 02", "ModifiedSchaffer_02"),
]



def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None, seed=None):
    def iico_obj(x):
        return obj(np.array(x))
    def iico_fun(x):
        y, lb, ub = bench_func(x)
        return y, lb, ub
    best_fitness_list, gbest, info = IICO_func(iico_fun, max_evals, pop_size, dim, seed=seed)
    return np.array(info['history'])

def run_fcsa(obj, bounds, seed=None):
    opt = FCSA(obj, bounds, seed=seed)
    x_best, f_best, info = opt.optimize()
    # If info['history'] is available, return it for plotting, else just f_best
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_reformed(obj, bounds, seed=None):
    opt = HybridRolePartitioned(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))


def run_hybrid_sbm(obj, bounds, seed=None):
    opt = HybridCSAOriginal_sbm(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    history = np.array(info.get("history", [f_best]))
    diag = info.get("diagnostics", None)
    return history, diag  # <-- return both


algorithms = [
    ("FCSA", run_fcsa), 
    ("IICO", run_iico),
    ("Hybrid Reformed", run_hybrid_reformed),
    ("Hybrid sbm", run_hybrid_sbm),

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


def run_all_dims():
    import matplotlib.pyplot as plt
    import os
    import csv

    # dims = [2, 50, 100]
    dims = [2]
    n_runs = 1

    for dim in dims:
        # Folder and log for this dimension
        fig_dir = os.path.join(os.path.dirname(__file__), f'../test_3_fig/dim_{dim}')
        os.makedirs(fig_dir, exist_ok=True)
        log_file = os.path.join(fig_dir, f'Test_{dim}_log.txt')
        with open(log_file, "w") as f:
            f.write(f"Test {dim} - FCSA Benchmark Results ({n_runs} runs per algorithm)\n\n")
            f.write("NOTE: IICO and HybridCSA require matplotlib, openpyxl, scipy, numpy.\n")
            for bench_disp, bench_name in benchmarks:
                f.write(f"Benchmark: {bench_disp}\n")
                print(f"\nBenchmark: {bench_disp} (dim={dim})")
                func = get_function_by_name(bench_name)
                dummy_x = np.zeros(2)
                _, lb, ub = func(dummy_x)
                bench_dim = dim
                bounds = [(lb, ub)] * bench_dim

                def obj(x):
                    y, _, _ = func(x)
                    return y

                plt.figure(figsize=(10, 6))
                for alg_name, runner in algorithms:
                    print(f"  Algorithm: {alg_name}")
                    histories = []
                    diags_for_alg_bench = []  # store per-run diagnostics if available

                    for run in range(n_runs):
                        if alg_name == "IICO":
                            max_evals = 350_000
                            pop_size = 60
                            result = runner(obj, bounds, bench_dim, max_evals, pop_size, func, seed=run)
                        else:
                            result = runner(obj, bounds, seed=run)

                        # result may be history (np.array) or (history, diag)
                        if isinstance(result, tuple) and len(result) == 2:
                            history, diag = result
                            diags_for_alg_bench.append(diag)
                        else:
                            history = result

                        histories.append(np.array(history))

                    histories = pad_histories(histories)
                    mean_curve = np.mean(histories, axis=0)
                    std_curve = np.std(histories, axis=0)
                    gens = np.arange(len(mean_curve))
                    plt.plot(gens, mean_curve, label=alg_name)
                    plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

                    # Log only the final generation's mean/std
                    f.write(f"  {alg_name}: mean={mean_curve[-1]:.2e}, std={std_curve[-1]:.2e}\n")
                    print(f"    mean: {mean_curve[-1]:.2e}, std: {std_curve[-1]:.2e}")

                    # --- Save CSV of convergence histories ---
                    csv_filename = f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv"
                    csv_path = os.path.join(fig_dir, csv_filename)
                    with open(csv_path, "w", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        header = ["generation"] + [f"run_{i}" for i in range(n_runs)]
                        writer.writerow(header)
                        for gen in range(len(mean_curve)):
                            row = [gen] + [histories[run][gen] for run in range(n_runs)]
                            writer.writerow(row)
                    # print(f"    Saved CSV: {csv_path}")

                    
                plt.title(f"Convergence Curves: {bench_disp} ({n_runs} runs, dim={dim})")
                plt.xlabel("Generation")
                plt.ylabel("Best Fitness")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                fig_path = os.path.join(fig_dir, f"convergence_{bench_disp.replace(' ', '_')}_dim{dim}.png")
                plt.savefig(fig_path)
                plt.close()
                # print(f"  Saved plot: {fig_path}")
                f.write("\n")

if __name__ == "__main__":
    run_all_dims()
