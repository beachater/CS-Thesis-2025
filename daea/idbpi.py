import numpy as np
import math


def init_population_idbpi(lb, ub, init_popsize, dim, run_seed, rng, save_path=None):
    """
    Density-based initialization (IDBPI.m).
    lb, ub: arrays of shape [dim]
    returns pop_I: (init_popsize, dim)
    """

    # local RNG seeded from run_seed for reproducibility
    local_rng = np.random.default_rng(run_seed)

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    pop = np.zeros((init_popsize, dim), dtype=float)

    # constants from MATLAB
    # Cd = volume of dim-ball radius 1
    Cd = (math.pi ** (dim / 2.0)) / math.gamma(dim / 2.0 + 1.0)
    R = (((1.0 / init_popsize) * np.prod(ub - lb)) / Cd) ** (1.0 / dim)

    h = R
    k = 10

    # threshold
    rho_min = math.exp(-(h ** 2) / (2 * (h ** 2)))

    # first point random
    pop[0, :] = lb + (ub - lb) * local_rng.random(dim)

    # plan sample pool: k * (init_popsize - 1) candidates total
    ppl = lb + (ub - lb) * local_rng.random((k * (init_popsize - 1), dim))

    # construct population iteratively
    for i in range(1, init_popsize):
        # candidate chunk
        cand_block = ppl[k * (i - 1): k * i, :]  # shape (k, dim)

        # distance from candidates to already chosen pop[0:i]
        dist = np.sqrt(((cand_block[:, None, :] - pop[None, :i, :]) ** 2).sum(axis=2))
        # kernel density
        rho = np.exp(-(dist ** 2) / (2 * h ** 2))
        rho[rho < rho_min] = 0.0
        rho = (1.0 / i) * np.sum(rho, axis=1)  # mean density vs existing points
        # pick lowest rho (least crowded)
        select_idx = np.argmin(rho)
        pop[i, :] = cand_block[select_idx, :]

    # optionally save
    if save_path is not None:
        np.savez(save_path, pop_I=pop)

    return pop
