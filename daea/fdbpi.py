import numpy as np
import math


def fdbpi_generate(lb, ub, pop, fit_vals, init_popsize, dim, rng):
    """
    FDBPI.m
    Generate extra candidate points biased by density around good areas.
    Returns new_pop shape (init_popsize, dim)
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    pop = np.asarray(pop, dtype=float)
    fit_vals = np.asarray(fit_vals, dtype=float)

    # same R formula as IDBPI
    Cd = (math.pi ** (dim / 2.0)) / math.gamma(dim / 2.0 + 1.0)
    R = (((1.0 / init_popsize) * np.prod(ub - lb)) / Cd) ** (1.0 / dim)
    h = R
    k = 10

    # shift fitness >=0
    if fit_vals.min() < 0:
        fit_shift = fit_vals - fit_vals.min()
    else:
        fit_shift = fit_vals.copy()
    sum_val = np.sum(fit_shift) if np.sum(fit_shift) != 0 else 1.0

    rho_list = []
    ppl_list = []

    # MATLAB does for i = 1 : k*10
    total_batches = k * 10
    block_size = init_popsize // 10 if init_popsize >= 10 else max(1, init_popsize)

    for _ in range(total_batches):
        # generate batch of candidate points
        cand = lb + (ub - lb) * rng.random((block_size, dim))
        ppl_list.append(cand)

        # distance from cand to pop
        dist = np.sqrt(((cand[:, None, :] - pop[None, :, :]) ** 2).sum(axis=2))
        # compute density-weighted "score" approx
        temp = (1.0 / sum_val) * np.sum(
            fit_shift[None, :] * np.exp(-(dist ** 2) / (2 * (h ** 2))), axis=1
        )
        rho_list.append(temp)

    ppl_all = np.vstack(ppl_list)  # shape (?, dim)
    rho_all = np.concatenate(rho_list)  # shape (?)
    # pick top init_popsize by rho desc
    sort_idx = np.argsort(-rho_all)
    new_pop = ppl_all[sort_idx[:init_popsize], :]
    return new_pop
