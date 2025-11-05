import numpy as np
from typing import List, Dict
from datetime import datetime
from daea import DAEAState, daea_epoch
from cectest import DMMOProblem

# Constants
EPS_LEVELS = [1e-3, 1e-4, 1e-5]

PROBLEMS = [
    ("P1",  "F1", "C1", 5),
    ("P2",  "F2", "C1", 5),
    ("P3",  "F3", "C1", 5),
    ("P4",  "F4", "C1", 5),
    ("P5",  "F5", "C1", 5),
    ("P6",  "F6", "C1", 5),
    ("P7",  "F7", "C1", 5),
    ("P8",  "F8", "C1", 5),

    ("P9",  "F8", "C1", 5),
    ("P10", "F8", "C2", 5),
    ("P11", "F8", "C3", 5),
    ("P12", "F8", "C4", 5),
    ("P13", "F8", "C5", 5),
    ("P14", "F8", "C6", 5),
    ("P15", "F8", "C7", 5),
    ("P16", "F8", "C8", 5),

    ("P17", "F1", "C1", 10),
    ("P18", "F2", "C1", 10),
    ("P19", "F3", "C1", 10),
    ("P20", "F4", "C1", 10),
    ("P21", "F5", "C1", 10),
    ("P22", "F6", "C1", 10),
    ("P23", "F7", "C1", 10),
    ("P24", "F8", "C1", 10),
]


def run_experiment_daea_detailed(
    problem_id: str,
    change_mode: str,
    dim: int,
    runs: int = 30,
    envs: int = 60,
    fe_per_env_factor: int = 5000,
    pop_size: int = 100,
    F: float = 0.6,
    CR: float = 0.9,
    seed_start: int = 1,
    eps_f_levels: List[float] = EPS_LEVELS,
) -> Dict:

    fe_per_env = fe_per_env_factor * dim
    run_num_peaks_found = [{eps: 0 for eps in eps_f_levels} for _ in range(runs)]
    run_total_peaks = [0 for _ in range(runs)]
    total_fe_overall = 0
    pr_runs = {eps: [] for eps in eps_f_levels}

    for run_i in range(runs):
        seed = seed_start + run_i
        rng = np.random.default_rng(seed)
        prob = DMMOProblem(problem_id, change_mode=change_mode, dim=dim, seed=seed)
        daea_state = DAEAState(max_archive=200)

        pop = daea_state.inject_population(pop_size, dim, rng, (-5, 5), reuse_rate=0.0)
        fit = prob.evaluate(pop)
        fe_count_this_run = pop_size

        for e in range(envs):
            prob.update_environment()
            pop = daea_state.inject_population(pop_size, dim, rng, (-5, 5), reuse_rate=0.5)
            fit = prob.evaluate(pop)
            fe_count_this_run += pop_size

            fe_in_env = 0
            while fe_in_env < fe_per_env:
                pop, fit = daea_epoch(pop, fit, prob, rng, daea_state, F=F, CR=CR)
                fe_in_env += pop_size
                fe_count_this_run += pop_size

            peaks_now = len(prob.get_global_centers())
            run_total_peaks[run_i] += peaks_now

            for eps in eps_f_levels:
                found = prob.count_found_peaks(pop, eps_d=0.05, eps_f=eps)
                run_num_peaks_found[run_i][eps] += found

            daea_state.update_archive(pop, fit, niche_radius=0.1)

        total_fe_overall += fe_count_this_run

    # Compute PR stats
    for run_i in range(runs):
        denom = run_total_peaks[run_i] if run_total_peaks[run_i] > 0 else 1.0
        for eps in eps_f_levels:
            pr_val = run_num_peaks_found[run_i][eps] / denom
            pr_runs[eps].append(pr_val)

    summary = {}
    for eps in eps_f_levels:
        vals = np.array(pr_runs[eps])
        summary[eps] = {
            "PR_mean":  float(np.mean(vals)),
            "PR_best":  float(np.max(vals)),
            "PR_worst": float(np.min(vals)),
        }

    return {
        "per_eps": summary,
        "pr_runs": pr_runs,
        "total_FE": total_fe_overall,
        "runs": runs,
        "envs": envs,
        "dim": dim,
        "problem_id": problem_id,
        "change_mode": change_mode,
    }


def run_full_suite_daea(output_file: str = "daea_results.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Density-Assisted Evolutionary Dynamic Multimodal Optimization Results\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("Problem, ε_f, PR_mean, PR_best, PR_worst\n")

        for pid, f_id, c_mode, dim in PROBLEMS:
            print(f"Running {pid}: {f_id} / {c_mode} / D={dim}")
            res = run_experiment_daea_detailed(
                problem_id=f_id,
                change_mode=c_mode,
                dim=dim,
                runs=30,
                envs=60,
                fe_per_env_factor=5000,
                pop_size=100,
                F=0.6,
                CR=0.9,
                seed_start=1,
            )

            for eps in EPS_LEVELS:
                row = res["per_eps"][eps]
                line = f"{pid},{eps:.0e},{row['PR_mean']:.6f},{row['PR_best']:.6f},{row['PR_worst']:.6f}\n"
                f.write(line)
                print(line.strip())

        f.write("\nEnd of Results\n")
        print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    run_full_suite_daea()
