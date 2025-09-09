
import numpy as np
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
    return f_best

def run_fcsa_sp(obj, bounds):
    opt = FCSASP(obj, bounds)
    x_best, f_best, info = opt.optimize()
    return f_best

def run_fcsa_sp_qmc(obj, bounds):
    opt = FCSASPQMC(obj, bounds)
    x_best, f_best, info = opt.optimize()
    return f_best

def run_hybrid_pruning_adaptive(obj, bounds):
    opt = HybridCSA_adaptive(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_hybrid_pruning_de_version(obj, bounds):
    opt = HybridCSA_pruning_de_version(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_hybrid_pruning_de_version_adaptive(obj, bounds):
    opt = HybridCSA_pruning_de_version_adaptive(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_hybrid_original_de(obj, bounds):
    opt = HybridCSAOriginal_de(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None):
    # IICO expects fun (returns value, lb, ub), max_FEs, n, dim
    # Use bench_func for bounds, obj for fitness
    def iico_obj(x):
        return obj(np.array(x))
    # Pass bench_func for bounds, iico_obj for fitness
    def iico_fun(x):
        # For initialization, return (value, lb, ub) as in benchmark.py
        y, lb, ub = bench_func(x)
        return y, lb, ub
    # For fitness, IICO will call fun(x)[0], which is fine
    best_fitness_list, gbest = IICO_func(iico_fun, max_evals, pop_size, dim)
    return best_fitness_list[-1]

def run_hybrid_pruning(obj, bounds):
    opt = HybridCSA_pruning(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_hybrid_pruning_sobol(obj, bounds):
    opt = HybridCSA_sobol(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

def run_hybrid_original(obj, bounds):
    opt = HybridCSAOriginal(obj, bounds)
    x_best, f_best, info = opt.minimize()
    return f_best

algorithms = [
    ("FCSA", run_fcsa),
    ("Hybrid Original", run_hybrid_original),
    ("Hybrid Original + DE", run_hybrid_original_de),
    ("Adaptive Pruning", run_hybrid_pruning_adaptive),
    ("Hybrid Pruning+DE", run_hybrid_pruning_de_version),
    ("Adaptive Pruning+DE", run_hybrid_pruning_de_version_adaptive),
    # ("FCSA+SP", run_fcsa_sp),
    # ("FCSA+SP+QMC", run_fcsa_sp_qmc),
    # ("IICO", run_iico),
    # ("HybridCSA-Pruning", run_hybrid_pruning),
    # ("HybridCSA-Sobol", run_hybrid_pruning_sobol),
]


import argparse


def run_all(n_runs=100, log_file="fcsa_benchmark_log.txt", dim=None):
    # Note: IICO and HybridCSA require extra packages: matplotlib, openpyxl, scipy (for sobol), numpy
    # Install with: pip install matplotlib openpyxl scipy numpy
    results = []
    with open(log_file, "w") as f:
        f.write(f"FCSA Benchmark Results (runs per algorithm: {n_runs})\n\n")
        f.write("NOTE: IICO and HybridCSA require matplotlib, openpyxl, scipy, numpy.\n")
        for bench_disp, bench_name in benchmarks:
            f.write(f"Benchmark: {bench_disp}\n")
            print(f"\nBenchmark: {bench_disp}")
            func = get_function_by_name(bench_name)
            # Get default bounds from the function
            dummy_x = np.zeros(2)
            _, lb, ub = func(dummy_x)
            # If dim is provided, use it, else use default logic
            if dim is not None:
                bench_dim = dim
            else:
                bench_dim = 30 if bench_disp not in ["Holdertable", "Eggholder", "Eggcrate", "Schaffer 01", "Schaffer 02"] else 2
            bounds = [(lb, ub)] * bench_dim
            def obj(x):
                y, _, _ = func(x)
                return y
            for alg_name, runner in algorithms:
                print(f"  Algorithm: {alg_name}")
                best_vals = []
                for run in range(n_runs):
                    if alg_name == "IICO":
                        # IICO expects: obj, bounds, dim, max_evals, pop_size, bench_func
                        max_evals = 350_000
                        pop_size = 60
                        val = runner(obj, bounds, bench_dim, max_evals, pop_size, func)
                    else:
                        val = runner(obj, bounds)
                    best_vals.append(val)
                mean = np.mean(best_vals)
                std = np.std(best_vals)
                f.write(f"  {alg_name}: mean={mean:.6g}, std={std:.6g}\n")
                print(f"    mean: {mean:.6g}, std: {std:.6g}")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per algorithm")
    parser.add_argument("--log", type=str, default="fcsa_benchmark_log.txt", help="Log file name")
    parser.add_argument("--dim", type=int, default=None, help="Dimension of the benchmark functions (overrides default)")
    args = parser.parse_args()
    run_all(n_runs=args.runs, log_file=args.log, dim=args.dim)
