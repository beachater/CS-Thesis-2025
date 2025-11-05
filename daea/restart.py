import numpy as np
from kernel_density import kernel_density


def restart_groups(
    pro,
    rng,
    bestmem_set,
    bestval_set,
    tracker,
    temp_best_pop,
    min_popsize,
):
    """
    Python port of restart.m
    This is called when groups are empty or environment nearly out of budget.
    Strategy:
      - sample many random points
      - pick least dense (using kernel_density vs track history)
      - spawn a new group around it
      - also update archive if close to best global val
    """

    rest = pro.freq - (pro.evaluated % pro.freq)
    groups = []

    if rest == 0 or rest == pro.freq:
        # environment already about to roll / rolled
        return groups, bestmem_set, bestval_set, temp_best_pop, tracker

    lam = 7 + int(np.floor(3 * np.log(pro.D)))
    D = pro.D

    rand_pool = rng.random((10 * rest, D)) * (pro.upper - pro.lower) + pro.lower
    # measure density vs tracker history of means
    track_points = tracker.to_array()  # all past xmean snapshots
    rho = kernel_density(rand_pool, track_points)
    dens_order = np.argsort(rho)  # ascending density

    if rest >= lam * 2:
        # choose single best position
        chosen = rand_pool[dens_order[0], :]
        g_pop = chosen[None, :] + 0.5 * rng.normal(size=(min_popsize, D))
        g_fit = pro.get_fits(g_pop)
        cc = np.std(g_fit)
        best_idx = np.argmax(g_fit)
        bestval = float(g_fit[best_idx])
        bestmem = g_pop[best_idx, :].copy()

        groups.append(
            {
                "idx": 0,
                "OPTS": {
                    "first": 1,
                    "pop": g_pop.copy(),
                    "val": g_fit.copy(),
                    "sigma": 0.5,
                    "count": 0,
                },
                "xmean": chosen.copy(),
                "bestmem": bestmem,
                "bestval": bestval,
                "delta": 0.0,
                "iters": 0,
                "cc": float(cc),
                "mean_distance": 10.0,
            }
        )
    else:
        # we pick multiple points up to 'rest' and add any that match global best ~1e-3 to archive
        chosen_many = rand_pool[dens_order[:rest], :]
        fits_many = pro.get_fits(chosen_many)
        if len(bestval_set) > 0:
            global_best = np.max(bestval_set)
        else:
            global_best = np.max(fits_many)

        for i in range(chosen_many.shape[0]):
            if abs(fits_many[i] - global_best) <= 1e-3:
                bestmem_set.append(chosen_many[i].copy())
                bestval_set.append(float(fits_many[i]))

    # temp_best_pop isn't reset here in MATLAB, but they do accumulate in main
    # We'll just return unchanged temp_best_pop.
    return groups, bestmem_set, bestval_set, temp_best_pop, tracker
