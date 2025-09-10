
import numpy as np
import matplotlib.pyplot as plt
from benchmark import get_function_by_name
from fscsa import FCSA
from fscsa_sp import FCSASP
from fscsa_sp_qmc import FCSASPQMC

# --- Import IICO and HybridCSA variants ---
from IICO import iico as IICO_func
from hybrid_pruning import HybridCSA as HybridCSA_pruning
from hybrid_w_sobol_synaptic_pruning import HybridCSA as HybridCSA_sobol
from hybrid_pruning_adaptive import HybridCSA_adaptive
from hybrid_pruning_de_version import HybridCSA_pruning_de_version
from hybrid_pruning_de_version_adaptive import HybridCSA_pruning_de_version_adaptive
from FCSA_IICO_Hybrid_original import HybridCSAOriginal
from hybrid_original_de import HybridCSAOriginal_de

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
def run_fcsa(obj, bounds):
    opt = FCSA(obj, bounds)
    x_best, f_best, info = opt.optimize()
    # If info['history'] is available, return it for plotting, else just f_best
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_fcsa_sp(obj, bounds):
    opt = FCSASP(obj, bounds)
    x_best, f_best, info = opt.optimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_fcsa_sp_qmc(obj, bounds):
    opt = FCSASPQMC(obj, bounds)
    x_best, f_best, info = opt.optimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_adaptive(obj, bounds):
    opt = HybridCSA_adaptive(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_de_version(obj, bounds):
    opt = HybridCSA_pruning_de_version(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_de_version_adaptive(obj, bounds):
    opt = HybridCSA_pruning_de_version_adaptive(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_original_de(obj, bounds):
    opt = HybridCSAOriginal_de(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None):
    def iico_obj(x):
        return obj(np.array(x))
    def iico_fun(x):
        y, lb, ub = bench_func(x)
        return y, lb, ub
    best_fitness_list, gbest, info = IICO_func(iico_fun, max_evals, pop_size, dim)
    return np.array(info['history'])

def run_hybrid_pruning(obj, bounds):
    opt = HybridCSA_pruning(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_pruning_sobol(obj, bounds):
    opt = HybridCSA_sobol(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

def run_hybrid_original(obj, bounds):
    opt = HybridCSAOriginal(obj, bounds)
    x_best, f_best, info = opt.minimize()
    if 'history' in info:
        return np.array(info['history'])
    else:
        return np.array([f_best])

algorithms = [
    ("FCSA", run_fcsa),
    ("Hybrid Original", run_hybrid_original),
    ("Hybrid Original + DE", run_hybrid_original_de),
    ("Adaptive Pruning", run_hybrid_pruning_adaptive),
    ("Hybrid Pruning+DE", run_hybrid_pruning_de_version),
    ("Adaptive Pruning+DE", run_hybrid_pruning_de_version_adaptive),
    ("FCSA+SP", run_fcsa_sp),
    ("FCSA+SP+QMC", run_fcsa_sp_qmc),
    ("IICO", run_iico),
    ("HybridCSA-Pruning", run_hybrid_pruning),
    ("HybridCSA-Sobol", run_hybrid_pruning_sobol),
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

def run_all(n_runs=100, log_file="fcsa_benchmark_log.txt", dim=None):
    import matplotlib.pyplot as plt
    import os
    results = []
    fig_dir = os.path.join(os.path.dirname(__file__), '../fig')
    os.makedirs(fig_dir, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"FCSA Benchmark Results (runs per algorithm: {n_runs})\n\n")
        f.write("NOTE: IICO and HybridCSA require matplotlib, openpyxl, scipy, numpy.\n")
        for bench_disp, bench_name in benchmarks:
            f.write(f"Benchmark: {bench_disp}\n")
            print(f"\nBenchmark: {bench_disp}")
            func = get_function_by_name(bench_name)
            dummy_x = np.zeros(2)
            _, lb, ub = func(dummy_x)
            if dim is not None:
                bench_dim = dim
            else:
                bench_dim = 30 if bench_disp not in ["Holdertable", "Eggholder", "Eggcrate", "Schaffer 01", "Schaffer 02"] else 2
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
                        history = runner(obj, bounds, bench_dim, max_evals, pop_size, func)
                    else:
                        history = runner(obj, bounds)
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
            plt.title(f"Convergence Curves: {bench_disp}")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fig_path = os.path.join(fig_dir, f"convergence_{bench_disp.replace(' ', '_')}.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"  Saved plot: {fig_path}")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per algorithm")
    parser.add_argument("--log", type=str, default="fcsa_benchmark_log.txt", help="Log file name")
    parser.add_argument("--dim", type=int, default=None, help="Dimension of the benchmark functions (overrides default)")
    args = parser.parse_args()
    run_all(n_runs=args.runs, log_file=args.log, dim=args.dim)
