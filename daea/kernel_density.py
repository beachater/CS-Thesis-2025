import numpy as np
import math


def kernel_density(temp_x, temp_pop, h=1.1):
    """
    kernel.m
    temp_x: (n_new, dim)
    temp_pop: (n_ref, dim)
    Returns rho_mean per row in temp_x.
    """

    temp_x = np.asarray(temp_x, dtype=float)
    temp_pop = np.asarray(temp_pop, dtype=float)

    if temp_pop.size == 0:
        return np.zeros((temp_x.shape[0],), dtype=float)

    # pairwise distances
    dist = np.sqrt(((temp_x[:, None, :] - temp_pop[None, :, :]) ** 2).sum(axis=2))

    rho = np.exp(-(dist ** 2) / (2 * (h ** 2)))
    rho_min = math.exp(-(h ** 2) / (2 * (h ** 2)))
    rho[rho < rho_min] = 0.0

    # mean across ref set
    rho_mean = np.mean(rho, axis=1)
    return rho_mean
