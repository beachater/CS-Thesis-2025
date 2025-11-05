import numpy as np


def init_groups_from_species(
    pro,
    lambda_size,
    init_pop,
    fitness,
    species_list,
    species_order,
    species_sizes,
    species_best_vals,
    rng,
    bestmem_set_arr=None,
):
    """
    Build initial CMA-ES "groups" just like init_groups.m

    Each group is a dict:
      {
        "idx": int,                # species index
        "OPTS": { ... CMA-ES state ... },
        "xmean": (D,),
        "bestmem": (D,),
        "bestval": float,
        "delta": float,
        "iters": int,
        "cc": float (std of vals),
        "mean_distance": float
      }
    """

    groups = []
    D = pro.D
    lb = pro.lower
    ub = pro.upper

    min_popsize = lambda_size
    max_groups_from_species = min(len(species_list), 9)  # MATLAB uses "if i>9 break"

    for i in range(max_groups_from_species):
        sp = species_list[i]
        idxs = sp["idx"]
        sp_len = sp["len"]

        group = {}
        group["idx"] = i
        group["iters"] = 0
        group["delta"] = 0.0

        if sp_len >= min_popsize:
            # enough individuals already
            sel = idxs[:min_popsize]
            g_pop = init_pop[sel, :]
            g_fit = fitness[sel]

            xmean = np.mean(g_pop, axis=0)
            diff = g_pop - xmean[None, :]
            sigma0 = np.sqrt((1.0 / (min_popsize * D)) * np.sum(diff ** 2))
            cc = np.std(g_fit)
            bestmem = init_pop[sp["seed"], :].copy()
            bestval = float(fitness[sp["seed"]])

            group["OPTS"] = {
                "first": 1,
                "pop": g_pop.copy(),
                "val": g_fit.copy(),
                "sigma": sigma0,
                "count": 0,
            }
            group["xmean"] = xmean.copy()
            group["bestmem"] = bestmem.copy()
            group["bestval"] = bestval
            group["cc"] = float(cc)

        else:
            # need to pad with Gaussian samples around cluster
            sel = idxs[:sp_len]
            g_pop = init_pop[sel, :]
            g_fit = fitness[sel]

            if sp_len == 1:
                xmean = g_pop[0, :].copy()
                sigma0 = 0.5
            else:
                xmean = np.mean(g_pop, axis=0)
                diff = g_pop - xmean[None, :]
                sigma0 = np.sqrt((1.0 / (sp_len * D)) * np.sum(diff ** 2))

            # fill missing
            add_size = min_popsize - sp_len
            add_pop = xmean[None, :] + sigma0 * rng.normal(size=(add_size, D))
            add_fit = pro.get_fits(add_pop)

            g_pop_full = np.vstack([g_pop, add_pop])
            g_fit_full = np.concatenate([g_fit, add_fit])
            cc = np.std(g_fit_full)

            bestmem = init_pop[sp["seed"], :].copy()
            bestval = float(fitness[sp["seed"]])

            group["OPTS"] = {
                "first": 1,
                "pop": g_pop_full.copy(),
                "val": g_fit_full.copy(),
                "sigma": sigma0 if sp_len > 1 else 0.5,
                "count": 0,
            }
            group["xmean"] = xmean.copy()
            group["bestmem"] = bestmem.copy()
            group["bestval"] = bestval
            group["cc"] = float(cc)

        groups.append(group)

    # also add groups from archive bestmem_set_arr if available
    if bestmem_set_arr is not None and bestmem_set_arr.size > 0:
        for k in range(bestmem_set_arr.shape[0]):
            g = {}
            g["idx"] = len(groups)
            g["iters"] = 0
            g["delta"] = 0.0

            xmean = bestmem_set_arr[k, :].copy()
            # random gaussian around xmean
            g_pop = xmean[None, :] + rng.normal(size=(lambda_size, D))
            g_fit = pro.get_fits(g_pop)
            cc = 0.01  # copied from MATLAB
            bestval = float(g_fit.max())
            bestmem = g_pop[np.argmax(g_fit), :].copy()

            g["OPTS"] = {
                "first": 1,
                "pop": g_pop,
                "val": g_fit,
                "sigma": 0.5,
                "count": 0,
            }
            g["xmean"] = xmean.copy()
            g["bestmem"] = bestmem.copy()
            g["bestval"] = bestval
            g["cc"] = float(cc)

            groups.append(g)

    # compute mean_distance for each group (average centroid distance to others)
    if len(groups) > 1:
        centroids = np.vstack([g["xmean"] for g in groups])
        dist_mat = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        for gi, g in enumerate(groups):
            # avoid div by zero
            tmp = np.delete(dist_mat[gi, :], gi)
            g["mean_distance"] = float(np.mean(tmp)) if tmp.size else 0.0
    else:
        for g in groups:
            g["mean_distance"] = 0.0

    return groups
