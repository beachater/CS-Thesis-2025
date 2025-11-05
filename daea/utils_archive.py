import numpy as np


def try_archive_group_if_converged(
    groups,
    global_bestval,
    bestmem_set,
    bestval_set,
    temp_best_pop,
    tol_bestval=1e-5,
    tol_sigma=1e-6,
):
    """
    Check each group:
    If converged (OPTS.sigma < tol_sigma) and
    |group.bestval - global_bestval| <= tol_bestval
    then:
      - add group.bestmem to archive (bestmem_set, bestval_set)
      - add group's OPTS.pop to temp_best_pop
      - remove group from active list
    """

    keep_groups = []
    for g in groups:
        sigma_now = g["OPTS"]["sigma"]
        if sigma_now <= tol_sigma and abs(g["bestval"] - global_bestval) <= tol_bestval:
            # archive
            bestmem_set.append(g["bestmem"].copy())
            bestval_set.append(float(g["bestval"]))

            # update temp_best_pop
            if temp_best_pop is None or temp_best_pop.size == 0:
                temp_best_pop = g["OPTS"]["pop"].copy()
            else:
                temp_best_pop = np.vstack([temp_best_pop, g["OPTS"]["pop"]])
            # discard g
        else:
            keep_groups.append(g)

    return keep_groups, bestmem_set, bestval_set, temp_best_pop


def merge_archived_into_groups(
    pro,
    rng,
    bestmem_set,
    bestval_set,
    min_popsize,
    tracker,
):
    """
    If we have no groups left but we do have archive (bestmem_set),
    rebuild new groups from them (like init_groups archive branch).
    """

    groups = []
    if len(bestmem_set) == 0:
        return groups, tracker

    D = pro.D
    for i, center in enumerate(bestmem_set):
        xmean = center.copy()
        pop = xmean[None, :] + rng.normal(size=(min_popsize, D))
        fit = pro.get_fits(pop)
        best_idx = np.argmax(fit)
        groups.append(
            {
                "idx": i,
                "OPTS": {
                    "first": 1,
                    "pop": pop.copy(),
                    "val": fit.copy(),
                    "sigma": 0.5,
                    "count": 0,
                },
                "xmean": xmean.copy(),
                "bestmem": pop[best_idx, :].copy(),
                "bestval": float(fit[best_idx]),
                "delta": 0.0,
                "iters": 0,
                "cc": float(np.std(fit)),
                "mean_distance": 10.0,
            }
        )

    tracker.add_snapshot(groups)
    return groups, tracker


def prune_groups_if_stalled(groups, pro, verbose=True):
    """
    Drop groups that are clearly useless:
    - OPTS.sigma < 0.02 but group.bestval NOT equal to global best (ceil logic in MATLAB)
    We'll approximate: kill if sigma <0.02 and bestval < max(bestval)-1e-9.
    """

    if len(groups) == 0:
        return groups
    global_best = max(g["bestval"] for g in groups)
    kept = []
    for g in groups:
        if g["OPTS"]["sigma"] < 0.02 and (g["bestval"] + 1e-9) < global_best:
            # drop it
            if verbose:
                pass
        else:
            kept.append(g)
    return kept
