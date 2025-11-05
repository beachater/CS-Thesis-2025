import numpy as np
from cectest import DMMOProblem
from amlp import AMLP_RS_CMSA_ESII, compute_peak_ratio


def run_experiment(dim=10, seeds=(0,1,2), eps_levels=(1e-3,1e-4,1e-5)):
    """
    Loop across all 8 test functions and 8 change modes
    For each seed compute mean best worst PR for each eps level

    This mirrors dmmo_runner style output
    """

    results = []

    for func_id in range(1,9):
        for change_id in range(1,9):
            all_seed_stats = {eps: [] for eps in eps_levels}

            for seed in seeds:
                # init problem
                problem = DMMOProblem(func_id, change_id, dim, seed=seed)
                solver = AMLP_RS_CMSA_ESII(problem, rng_seed=seed)

                # run full experiment
                solver.run_full()

                # final archive
                arch_pts, arch_vals = solver.archive.export_env_archive()

                # compute peak ratio per eps setting
                for eps in eps_levels:
                    pr = compute_peak_ratio(problem, arch_pts, eps_d=eps, eps_f=eps)
                    all_seed_stats[eps].append(pr)

            # summarize for this func/change pair
            for eps in eps_levels:
                arr = np.array(all_seed_stats[eps])
                mean_pr = float(np.mean(arr))
                best_pr = float(np.max(arr))
                worst_pr = float(np.min(arr))
                results.append({
                    "func": func_id,
                    "change": change_id,
                    "eps": eps,
                    "mean_pr": mean_pr,
                    "best_pr": best_pr,
                    "worst_pr": worst_pr
                })

    return results


if __name__ == "__main__":
    stats = run_experiment(dim=10, seeds=(0,1,2))
    import csv
    with open("results_AMLP_RS_CMSA_ESII.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["func","change","eps","mean_pr","best_pr","worst_pr"]
        )
        writer.writeheader()
        for row in stats:
            writer.writerow(row)
    print("Done writing results_AMLP_RS_CMSA_ESII.csv")
