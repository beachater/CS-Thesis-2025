from tqdm import tqdm
from benchmark import get_function_by_name
from novanew import NOVA_Enhanced
import numpy as np

# small smoke test for NOVA_Enhanced (2D Ackley)
func = get_function_by_name('Ackley')
# get bounds from the benchmark function
_, lb, ub = func(np.zeros(2))
bounds = [(lb, ub)] * 2

# progress bar for evaluations
pbar = tqdm(total=1000, desc='NOVA evals', unit='eval')
opt = NOVA_Enhanced(lambda x: func(x)[0], bounds, N=12, max_evals=1000, seed=0, progress=pbar.update)
gbest, fbest, history = opt.optimize()
pbar.close()
print('Done. best f =', fbest)
print('History length:', len(history))
