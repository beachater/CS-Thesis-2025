
import numpy as np
from fscsa import FSCSA
import benchmark as bm
import matplotlib.pyplot as plt

import os

os.makedirs("fig", exist_ok=True)


func = bm.get_function_by_name('Ackley')
dim = 2

algo = FSCSA(func=func, dim=dim, N=50, n_select=10, n_clones=5, m=2.0, c=3.0, max_iters=1000, seed=42)
best_x, best_f, history = algo.optimize()

print('Best fitness:', best_f)
print('Best x (first 5 dims):', best_x[:5])

plt.figure()
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Best fitness')
plt.title('FSCSA on Ackley')
plt.tight_layout()
plt.savefig("fig/fscsa_convergence.png")
print('Saved convergence plot to /mnt/data/fscsa_convergence.png')
