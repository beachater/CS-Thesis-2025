
import numpy as np
import matplotlib.pyplot as plt
from benchmark import get_function_by_name

# Algorithm runners: (name, runner function)
# --- Import IICO and HybridCSA variants ---
from IICO import iico as IICO_func
from reformed_hybrid import HybridRolePartitioned
from hybrid_top import HybridCSAOriginal_sbm
from fscsa import FCSA
from nova import NOVAPlus
from FCSA_IICO_Hybrid_original import HybridCSAOriginal
from novanew import NOVA_Enhanced
from tqdm import tqdm
# from dset import ETFCSA_Lite
from dest2 import ETFCSA_Lite
from tsd import ETFCSA_TSD










# List of benchmark function names and their canonical names in benchmark.py
benchmarks = [
    ("Schaffer 01", "ModifiedSchaffer_01"),
    ("Schaffer 02", "ModifiedSchaffer_02"),
    ("Ackley", "Ackley"),
    ("Griewank", "Griewank"),
    ("Rastrigin", "Rastrigin"),
    ("Shubert", "Shubert_06"),
    ("Eggholder", "Eggcrate"),  # Eggholder is not present, using Eggcrate as a placeholder
    ("Holdertable", "HolderTable"),
    
]



def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None, seed=None, progress=None):
    def iico_obj(x):
        return obj(np.array(x))
    def iico_fun(x):
        y, lb, ub = bench_func(x)
        return y, lb, ub
    best_fitness_list, gbest, info = IICO_func(iico_fun, max_evals, pop_size, dim, seed=seed, progress=progress)
    return np.array(info['history'])

def run_fcsa(obj, bounds, seed=None, progress=None):
    opt = FCSA(obj, bounds, seed=seed, progress=progress)
    x_best, f_best, info = opt.optimize()
    # If info['history'] is available, return it for plotting, else just f_best
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_original(obj, bounds, seed=None):
    opt = HybridCSAOriginal(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
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


def run_novaplus(obj, bounds, seed=None):
    opt = NOVAPlus(obj, bounds, seed=seed)
    x_best, f_best, history = opt.minimize()
    return np.array(history)

def run_novanew(obj, bounds, seed=None, progress=None):
    opt = NOVA_Enhanced(obj, bounds, seed=seed, progress=progress)
    x_best, f_best, history = opt.optimize()
    return np.array(history)
    
def run_tsd(obj, bounds, seed=None, progress=None):
    opt = ETFCSA_TSD(obj, bounds, seed=seed, progress=progress)
    x_best, f_best, info = opt.optimize()
    # ETFCSA_TSD.optimize() returns (x_best, f_best, info_dict)
    history = info.get("history", [f_best]) if isinstance(info, dict) else [f_best]
    return np.array(history)


def run_dest(obj, bounds, seed=None, progress=None):
    opt = ETFCSA_Lite(obj, bounds, seed=seed, progress=progress)
    x_best, f_best, info = opt.optimize()
    # opt.optimize() returns (x_best, f_best, info_dict) where info_dict['history'] is the list
    # of best fitness values per tick. Ensure we return a proper sized numpy array so
    # pad_histories can call len() on each history.
    history = info.get("history", [f_best]) if isinstance(info, dict) else [f_best]
    return np.array(history)

algorithms = [
    # ("Hybrid Reformed", run_hybrid_reformed),
    # ("Hybrid sbm", run_hybrid_sbm),
    # ("NOVAPlus", run_novaplus),
    # ("NOVANew", run_novanew),
    # ("DEST", run_dest),
    ("TSD", run_tsd),
     ("FCSA", run_fcsa), 
    ("IICO", run_iico),
   
    # ("Hybrid Original", run_hybrid_original),
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

    dims = [100]
    # dims = [100]
    n_runs = 1

    for dim in dims:
        # Folder and log for this dimension
        fig_dir = os.path.join(os.path.dirname(__file__), f'../new_test/dim_{dim}')
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
                        # choose evaluation budget per algorithm (fallback to 350k)
                        eval_budget = 350_000
                        if alg_name == "IICO":
                            max_evals = eval_budget
                            pop_size = 60
                            with tqdm(total=max_evals, desc=f"{alg_name} run {run+1}/{n_runs}", unit='eval', leave=False) as pbar:
                                result = runner(obj, bounds, bench_dim, max_evals, pop_size, func, seed=run, progress=pbar.update)
                        else:
                            with tqdm(total=eval_budget, desc=f"{alg_name} run {run+1}/{n_runs}", unit='eval', leave=False) as pbar:
                                # many algorithms accept a progress callback named 'progress'
                                try:
                                    result = runner(obj, bounds, seed=run, progress=pbar.update)
                                except TypeError:
                                    # fallback if runner doesn't accept progress
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
                    print(f"    Saved CSV: {csv_path}")

                    
                plt.title(f"Convergence Curves: {bench_disp} ({n_runs} runs, dim={dim})")
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
    run_all_dims()
