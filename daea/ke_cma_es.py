import numpy as np
from kernel_density import kernel_density
from nd_sort import fast_nd_sort


def ke_cma_es(
    pro,
    group,
    lb,
    ub,
    itermax,
    rng,
    temp_pop,
    group_index,
    temp_best_pop,
    tracker,
):
    """
    Python port of KE_CMA_ES.m (trimmed but faithful).
    Evolve one group for up to itermax CMA-ES iterations or until stall.
    Returns updated group dict or None if group is discarded.
    """

    if group is None:
        return None, tracker

    D = pro.D
    xmean = group["xmean"].copy()
    bestmem = group["bestmem"].copy()
    bestval = float(group["bestval"])
    OPTS = group["OPTS"]

    old_bestval = bestval
    min_popsize = 7 + int(np.floor(3 * np.log(D)))  # lambda formula in MATLAB
    lam = min_popsize

    # concat temp_pop like MATLAB: all groups' OPTS.pop plus temp_best_pop
    if temp_pop is None:
        temp_pop_concat = group["OPTS"]["pop"]
    else:
        # temp_pop was already built in caller as concat of all groups + temp_best_pop
        temp_pop_concat = temp_pop

    # init CMA-ES strategy parameters
    if OPTS.get("first", 1) == 1:
        OPTS["first"] = 0

        mu = lam // 2
        # recombination weights (log)
        raw_w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(np.arange(1, mu + 1))
        # but MATLAB uses log(mu+1/2)-log(1:mu). We'll replicate that:
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = raw_w / np.sum(raw_w)

        mueff = (np.sum(w) ** 2) / np.sum(w ** 2)

        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / (((D + 1.3) ** 2) + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / (((D + 2) ** 2) + 2 * mueff / 2)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs

        pc = np.zeros(D)
        ps = np.zeros(D)
        B = np.eye(D)
        DiagD = np.ones(D)  # diag of D matrix
        C = B @ np.diag(DiagD) @ (B @ np.diag(DiagD)).T
        chiN = (D ** 0.5) * (1 - 1 / (4 * D) + 1 / (21 * (D ** 2)))
        countval = 0
        iters = 0

        sigma = OPTS["sigma"]

    else:
        # resume CMA-ES parameters from OPTS
        lam = OPTS["lambda"]
        w = OPTS["weights"]
        mu = OPTS["mu"]
        mueff = OPTS["mueff"]
        cc = OPTS["cc"]
        cs = OPTS["cs"]
        c1 = OPTS["c1"]
        cmu = OPTS["cmu"]
        damps = OPTS["damps"]
        pc = OPTS["pc"]
        ps = OPTS["ps"]
        B = OPTS["B"]
        DiagD = OPTS["D"]
        C = OPTS["C"]
        chiN = OPTS["chiN"]
        countval = OPTS["countval"]
        iters = group["iters"]
        sigma = OPTS["sigma"]

    # generation loop
    stopiters = iters + itermax

    while iters < stopiters:
        # sample lam offspring from multivariate normal
        arz = rng.normal(size=(D, lam))
        arx = (xmean[:, None] + sigma * (B @ np.diag(DiagD) @ arz)).T  # (lam,D)

        # clamp to [lb, ub], fix arz accordingly
        for j in range(lam):
            too_hi = arx[j, :] > ub
            too_lo = arx[j, :] < lb
            if np.any(too_hi) or np.any(too_lo):
                arx[j, too_hi] = ub[too_hi]
                arx[j, too_lo] = lb[too_lo]
                # update arz row j accordingly
                arz[:, j] = np.linalg.pinv(np.diag(DiagD)) @ np.linalg.pinv(B) @ (
                    (arx[j, :] - xmean) / sigma
                )

        # Evaluate objective fitness and kernel density rho
        fit_vals = pro.get_fits(arx)
        rho_vals = kernel_density(arx, temp_pop_concat)

        # Build multi-objective matrix and fast non-dominated sort
        # We want best "front" first, then best fitness.
        sorted_idx = fast_nd_sort(fit_vals, rho_vals)
        arx = arx[sorted_idx, :]
        fit_vals = fit_vals[sorted_idx]
        arz = arz[:, sorted_idx]

        # update group count of "OPTS.count" like MATLAB
        # "group.OPTS.count = group.OPTS.count + length(rho(rho>0))"
        group["OPTS"]["count"] += np.count_nonzero(rho_vals > 0)

        # quick stall kill check:
        # if group.iters>0 && group.OPTS.count > 100/x (x = group.mean_distance)
        # if so this group is useless -> discard
        if group["iters"] > 0:
            x = group.get("mean_distance", 1.0)
            if x <= 0:
                x = 1e-12
            if group["OPTS"]["count"] > 100.0 / x:
                if True:
                    # discard group
                    return None, tracker

        iters += 1
        countval += lam

        # recombination
        mu_eff_indices = np.arange(mu)
        xmean = np.sum(arx[mu_eff_indices, :] * w[:, None], axis=0)
        zmean = np.sum(arz[:, mu_eff_indices] * w[None, :], axis=1)

        # evolution paths ps, pc
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ zmean)
        norm_ps = np.linalg.norm(ps)
        hsig = (
            norm_ps
            / np.sqrt(1 - (1 - cs) ** (2 * countval / lam))
            / chiN
            < (1.4 + 2 / (D + 1))
        )

        pc = (1 - cc) * pc + (
            hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (DiagD * zmean))
        )

        # covariance update
        # rank-one
        rank_one = np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C
        # rank-mu
        BDz = (B @ np.diag(DiagD) @ arz[:, mu_eff_indices])
        rank_mu = BDz @ np.diag(w) @ BDz.T
        C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

        # step-size sigma
        sigma = sigma * np.exp((cs / damps) * (norm_ps / chiN - 1))

        # eigen-decomposition to update B, D
        C = np.triu(C) + np.triu(C, 1).T
        evals, evecs = np.linalg.eigh(C)
        # numerical guard
        evals = np.maximum(evals, 1e-12)
        B = evecs
        DiagD = np.sqrt(evals)

        # bestval / bestmem tracking
        idx_best = np.argmax(fit_vals)
        best_fit_now = fit_vals[idx_best]
        if best_fit_now > bestval:
            bestval = float(best_fit_now)
            bestmem = arx[idx_best, :].copy()

        # if population is extremely converged in fitness, exit early
        if np.std(fit_vals) < 1e-6:
            break

    # end CMA loop

    # update group dict with new CMA-ES state
    OPTS.update(
        {
            "pc": pc,
            "ps": ps,
            "B": B,
            "D": DiagD,
            "C": C,
            "sigma": sigma,
            "lambda": lam,
            "weights": w,
            "mu": mu,
            "mueff": mueff,
            "cc": cc,
            "cs": cs,
            "c1": c1,
            "cmu": cmu,
            "damps": damps,
            "chiN": chiN,
            "countval": countval,
            "pop": arx.copy(),
            "val": fit_vals.copy(),
        }
    )

    group["xmean"] = xmean.copy()
    group["bestmem"] = arx[0, :].copy()  # sorted by fast_nd_sort
    group["bestval"] = float(fit_vals[0])
    group["OPTS"] = OPTS
    group["delta"] = bestval - old_bestval
    group["cc"] = float(np.std(fit_vals))
    group["iters"] = iters

    # track for restart heuristic
    tracker.add_snapshot([group])

    return group, tracker
