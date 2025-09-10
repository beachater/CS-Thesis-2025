import numpy as np
from benchmark import get_function_by_name
from fcsa_levy import fcsa_hybrid_levy   # using the function wrapper for simplicity

# IICO
from IICO import iico as IICO_func

benchmarks = [
    ("Ackley", "Ackley"),
    ("Griewank", "Griewank"),
    ("Rastrigin", "Rastrigin"),
    ("Shubert", "Shubert_06"),
    ("Eggholder", "Eggcrate"),   # placeholder
    ("Holdertable", "HolderTable"),
    ("Schaffer 01", "ModifiedSchaffer_01"),
    ("Schaffer 02", "ModifiedSchaffer_02"),
]

# runners
def run_fcsa_levy(obj, bounds):
    x_best, f_best, info = fcsa_hybrid_levy(obj, bounds)
    return f_best

def run_iico(obj, bounds, dim, max_evals, pop_size, bench_func=None):
    # IICO expects fun(x) -> (value, lb, ub)
    # bench_func is the function from benchmark.py that returns (y, lb, ub) when called
    def iico_fun(x):
        y, lb, ub = bench_func(x)
        return y, lb, ub
    best_fitness_list, gbest = IICO_func(iico_fun, max_evals, pop_size, dim)
    return best_fitness_list[-1]

algorithms = [
    ("LEVY", run_fcsa_levy),
    ("IICO", run_iico),
]

import argparse

def run_all(n_runs=100, log_file="fcsa_benchmark_log.txt", dim=None):
    with open(log_file, "w") as f:
        f.write(f"FCSA Benchmark Results runs per algorithm: {n_runs}\n\n")
        f.write("NOTE IICO may require matplotlib openpyxl numpy\n")
        for bench_disp, bench_name in benchmarks:
            f.write(f"Benchmark {bench_disp}\n")
            print(f"\nBenchmark {bench_disp}")
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

            for alg_name, runner in algorithms:
                print(f"  Algorithm {alg_name}")
                best_vals = []
                for _ in range(n_runs):
                    if alg_name == "IICO":
                        max_evals = 350_000
                        pop_size = 60
                        val = runner(obj, bounds, bench_dim, max_evals, pop_size, func)
                    else:
                        val = runner(obj, bounds)
                    best_vals.append(val)
                mean = np.mean(best_vals)
                std = np.std(best_vals)
                f.write(f"  {alg_name} mean={mean:.6g} std={std:.6g}\n")
                print(f"    mean {mean:.6g}  std {std:.6g}")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per algorithm")
    parser.add_argument("--log", type=str, default="fcsa_benchmark_log.txt", help="Log file name")
    parser.add_argument("--dim", type=int, default=None, help="Dimension override")
    args = parser.parse_args()
    run_all(n_runs=args.runs, log_file=args.log, dim=args.dim)
