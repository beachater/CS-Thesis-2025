
import numpy as np
import matplotlib.pyplot as plt
from benchmark import get_function_by_name
from fscsa import FCSA
from fscsa_sp import FCSASP

from fscsa_sp_qmc import FCSASPQMC
from fcsa_levy import fcsa_hybrid_levy
from hybrid_csa_chaos_sobol_δx_memory_adaptive_pruning import HybridCSAPlus
# Algorithm runners: (name, runner function)
def run_fcsa_levy(obj, bounds, seed=None):
    x_best, f_best, info = fcsa_hybrid_levy(obj, bounds, seed=seed)
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_memory_adaptive(obj, bounds, seed=None):
    opt = HybridCSAPlus(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    # info['history'] may be a list of (fitness, x), so extract fitness if needed
    hist = info.get('history', [f_best])
    if len(hist) > 0 and isinstance(hist[0], (tuple, list, np.ndarray)):
        # If tuple/list, assume (fitness, x)
        fitness_hist = [h[0] if isinstance(h, (tuple, list)) else h for h in hist]
        return np.array(fitness_hist)
    else:
        return np.array(hist)

# --- Import IICO and HybridCSA variants ---
from IICO import iico as IICO_func
from hybrid_pruning import HybridCSA as HybridCSA_pruning
from hybrid_w_sobol_synaptic_pruning import HybridCSA as HybridCSA_sobol
from hybrid_pruning_adaptive import HybridCSA_adaptive
from hybrid_pruning_de_version import HybridCSA_pruning_de_version
from hybrid_pruning_de_version_adaptive import HybridCSA_pruning_de_version_adaptive
from FCSA_IICO_Hybrid_original import HybridCSAOriginal
from hybrid_original_de import HybridCSAOriginal_de
from hybrid_original_composed import HybridCSAOperators

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

# Algorithm runners: (name, runner function)
def run_fcsa(obj, bounds, seed=None):
    opt = FCSA(obj, bounds, seed=seed)
    x_best, f_best, info = opt.optimize()
    # If info['history'] is available, return it for plotting, else just f_best
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_fcsa_sp(obj, bounds, seed=None):
    opt = FCSASP(obj, bounds, seed=seed)
    x_best, f_best, info = opt.optimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_fcsa_sp_qmc(obj, bounds, seed=None):
    opt = FCSASPQMC(obj, bounds, seed=seed)
    x_best, f_best, info = opt.optimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_adaptive(obj, bounds, seed=None):
    opt = HybridCSA_adaptive(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_de_version(obj, bounds, seed=None):
    opt = HybridCSA_pruning_de_version(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_de_version_adaptive(obj, bounds, seed=None):
    opt = HybridCSA_pruning_de_version_adaptive(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_original_de(obj, bounds, seed=None):
    opt = HybridCSAOriginal_de(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None, seed=None):
    def iico_obj(x):
        return obj(np.array(x))
    def iico_fun(x):
        y, lb, ub = bench_func(x)
        return y, lb, ub
    best_fitness_list, gbest, info = IICO_func(iico_fun, max_evals, pop_size, dim, seed=seed)
    return np.array(info['history'])

def run_hybrid_pruning(obj, bounds, seed=None):
    opt = HybridCSA_pruning(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_sobol(obj, bounds, seed=None):
    opt = HybridCSA_sobol(obj, bounds, seed=seed)
    x_best, f_best, info = opt.minimize()
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
    
def run_hybrid_tfo(obj, bounds, seed=None):
    opt = HybridCSAOperators(obj, bounds, operator="tfo", seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))

def run_hybrid_cdo(obj, bounds, seed=None):
    opt = HybridCSAOperators(obj, bounds, operator="cdo", seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))

def run_hybrid_rso(obj, bounds, seed=None):
    opt = HybridCSAOperators(obj, bounds, operator="rso", seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))

def run_hybrid_ibp(obj, bounds, seed=None):
    opt = HybridCSAOperators(obj, bounds, operator="ibp", seed=seed)
    x_best, f_best, info = opt.minimize()
    return np.array(info.get("history", [f_best]))


algorithms = [
    # ("FCSA", run_fcsa),
    # ("FCSA-Levy", run_fcsa_levy),
    # ("Hybrid Original", run_hybrid_original),
    # ("Hybrid Original + DE", run_hybrid_original_de),
    # ("Adaptive Pruning", run_hybrid_pruning_adaptive),
    # ("Hybrid Pruning+DE", run_hybrid_pruning_de_version),
    # ("Adaptive Pruning+DE", run_hybrid_pruning_de_version_adaptive),
    # ("FCSA+SP", run_fcsa_sp),
    # ("FCSA+SP+QMC", run_fcsa_sp_qmc),
    ("IICO", run_iico),
    # ("HybridCSA-Pruning", run_hybrid_pruning),
    # ("HybridCSA-Sobol", run_hybrid_pruning_sobol),
    # ("HybridCSA++ (Memory Adaptive)", run_hybrid_memory_adaptive),
    ("Hybrid TFO", run_hybrid_tfo),
    ("Hybrid CDO", run_hybrid_cdo),
    ("Hybrid RSO", run_hybrid_rso),
    ("Hybrid IBP", run_hybrid_ibp),
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
    dims = [100]
    n_runs = 2
    for dim in dims:
        # Folder and log for this dimension
        fig_dir = os.path.join(os.path.dirname(__file__), f'../feb_khan_minitest_3_fig/dim_{dim}')
        os.makedirs(fig_dir, exist_ok=True)
        log_file = os.path.join(fig_dir, f'Test_{dim}_log.txt')
        with open(log_file, "w") as f:
            f.write(f"Test {dim} - FCSA Benchmark Results (100 runs per algorithm)\n\n")
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
                    for run in range(n_runs):
                        if alg_name == "IICO":
                            max_evals = 350_000
                            pop_size = 60
                            history = runner(obj, bounds, bench_dim, max_evals, pop_size, func, seed=run)
                        else:
                            history = runner(obj, bounds, seed=run)
                        histories.append(np.array(history))
                    histories = pad_histories(histories)
                    mean_curve = np.mean(histories, axis=0)
                    std_curve = np.std(histories, axis=0)
                    gens = np.arange(len(mean_curve))
                    plt.plot(gens, mean_curve, label=alg_name)
                    plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
                    # Log only the final generation's mean/std
                    f.write(f"  {alg_name}: mean={mean_curve[-1]:.6g}, std={std_curve[-1]:.6g}\n")
                    print(f"    mean: {mean_curve[-1]:.6g}, std: {std_curve[-1]:.6g}")

                    # --- Save CSV for this algorithm/benchmark/dim ---
                    csv_filename = f"{alg_name}_{bench_disp.replace(' ', '_')}_dim{dim}_runs.csv"
                    csv_path = os.path.join(fig_dir, csv_filename)
                    with open(csv_path, "w", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # Header: generation, run_0, run_1, ...
                        header = ["generation"] + [f"run_{i}" for i in range(n_runs)]
                        writer.writerow(header)
                        for gen in range(len(mean_curve)):
                            row = [gen] + [histories[run][gen] for run in range(n_runs)]
                            writer.writerow(row)
                    print(f"    Saved CSV: {csv_path}")

                plt.title(f"Convergence Curves: {bench_disp} (100 runs, dim={dim})")
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
