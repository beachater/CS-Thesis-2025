import numpy as np
from fscsa import FCSA
from fscsa_sp import FCSASP
from fscsa_sp_qmc import FCSASPQMC
import benchmark as bm
import matplotlib.pyplot as plt
import os

os.makedirs("fig", exist_ok=True)

def as_scalar(f):
    def g(x):
        y = f(x)
        # supports f returning either float or (value, lb, ub)
        return float(y[0] if isinstance(y, tuple) else y)
    return g




# Ackley function in 2D
raw_func = bm.get_function_by_name('Ackley')
func = as_scalar(raw_func)
dim = 100
bounds = [(-32.768, 32.768)] * dim  # define bounds explicitly

# Initialize FCSA
algo = FCSASPQMC(
    func=func,
    bounds=bounds,
    N=50,
    n_select=10,
    n_clones=5,
    r=2.0,
    c_threshold=3.0,
    max_gens=1000,
    seed=42
)

best_x, best_f, result = algo.optimize()

print('Best fitness:', best_f)
print('Best x (first 5 dims):', best_x[:5])

# Extract fitness history
history = [h[0] for h in result['history']]

plt.figure()
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Best fitness')
plt.title('FCSA on Ackley')
plt.tight_layout()
plt.savefig("fig/fscsa_convergence.png")
print('Saved convergence plot to fig/fscsa_convergence.png')
