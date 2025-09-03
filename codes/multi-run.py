

import numpy as np
import matplotlib.pyplot as plt
from fscsa import FSCSA
from fscsa_sp import FSCSASP
import benchmark as bm

# --------------------------
# Experiment Configuration
# --------------------------
func = bm.get_function_by_name('Ackley')
dim = 30
runs = 30
max_iters = 1000
seed_base = 42

# --------------------------
# Storage for Results
# --------------------------
# results_fscsa = []
results_fscsa_sp = []

# # --------------------------
# # Run FSCSA (baseline)
# # --------------------------
# print(f"Running FSCSA baseline for {runs} runs...")
# for i in range(runs):
#     print(f"FSCSA Run {i+1}/{runs}")
#     algo = FSCSA(func=func, dim=dim, max_iters=max_iters, seed=seed_base + i)
#     _, best_f, _ = algo.optimize()
#     results_fscsa.append(best_f)

# --------------------------
# Run FSCSA + SP (with pruning)
# --------------------------
print(f"\nRunning FSCSA + SP for {runs} runs...")
for i in range(runs):
    print(f"FSCSA + SP Run {i+1}/{runs}")
    algo = FSCSASP(func=func, dim=dim, max_iters=max_iters, seed=seed_base + i)
    _, best_f, _ = algo.optimize()
    results_fscsa_sp.append(best_f)

# --------------------------
# Report Results
# --------------------------
def summarize(name, data):
    mean = np.mean(data)
    std = np.std(data)
    print(f"{name}: Mean = {mean:.6f}, Std = {std:.6f}")

print("\nSummary of Results:")
# summarize("FSCSA", results_fscsa)
summarize("FSCSA + SP", results_fscsa_sp)

# --------------------------
# Optional: Boxplot Comparison
# --------------------------
plt.figure()
# plt.boxplot([results_fscsa, results_fscsa_sp], labels=["FSCSA", "FSCSA + SP"])
plt.title(f"Fitness Distribution over {runs} Runs (Ackley, dim={dim})")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig/fscsa_vs_sp_boxplot.png")
print("Saved boxplot to fig/fscsa_vs_sp_boxplot.png")
