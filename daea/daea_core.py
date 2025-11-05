import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ============================================================
# kernel.m
# ============================================================

def kernel_density(candidates: np.ndarray,
                   reference: np.ndarray,
                   h: float = 1.1) -> np.ndarray:
    """
    Port of kernel.m
    candidates: (L, D)
    reference: (N, D)  (track of xmeans, temp_pop, etc.)
    returns rho_mean: (L,)
    """
    if reference is None or reference.size == 0:
        return np.zeros(candidates.shape[0], dtype=float)

    # pairwise squared euclidean distance
    diff = candidates[:, None, :] - reference[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)

    rho = np.exp(-dist_sq / (2.0 * (h ** 2)))

    # MATLAB enforces a rho_min cutoff
    rho_min = np.exp(-(h ** 2) / (2.0 * (h ** 2)))  # = exp(-1/2)
    rho[rho < rho_min] = 0.0

    rho_mean = np.mean(rho, axis=1)
    return rho_mean


# ============================================================
# Fast_ND_SORT.m (uses NDSort.m internally)
# We only need the behavior they actually use:
#   obj(1,:) = -fit (so higher fit is better)
#   obj(2,:) = rho  (so higher rho is better)
# They then:
#   index = Fast_ND_SORT(obj)
#   ...and take arx[:, index]
#
# We'll:
#   1. turn that into a 2D objective matrix for MINIMIZATION
#      so we negate both "maximize" terms
#   2. do nondominated sorting (Pareto fronts)
#   3. produce final ordering = by (frontNo asc, fitness desc)
# ============================================================

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    # strict Pareto dominance for minimization:
    # a dominates b if a is no worse in all dims AND strictly better in at least one
    return np.all(a <= b) and np.any(a < b)

def nondominated_fronts_min(objs: np.ndarray) -> List[List[int]]:
    """
    Classic fast nondominated sort for small N.
    objs: (N, M) minimization
    returns: list of fronts, each a list of indices
    """
    N = objs.shape[0]
    dominated_count = np.zeros(N, dtype=int)
    dominates_list = [[] for _ in range(N)]

    # build dominance graph
    for i in range(N):
        for j in range(i + 1, N):
            if _dominates(objs[i], objs[j]):
                dominates_list[i].append(j)
                dominated_count[j] += 1
            elif _dominates(objs[j], objs[i]):
                dominates_list[j].append(i)
                dominated_count[i] += 1

    # layer peeling
    fronts: List[List[int]] = []
    current = np.where(dominated_count == 0)[0].tolist()
    remaining = dominated_count.copy()

    while current:
        fronts.append(current)
        nxt: List[int] = []
        for p in current:
            for q in dominates_list[p]:
                remaining[q] -= 1
                if remaining[q] == 0:
                    nxt.append(q)
        current = nxt

    return fronts

def fast_nd_sort_for_daea(fitness: np.ndarray,
                          rho: np.ndarray) -> np.ndarray:
    """
    Recreates Fast_ND_SORT(obj) output ordering for KE_CMA_ES.

    fitness: shape (L,)        higher is better
    rho:     shape (L,)        higher is better

    Returns sorted indices "index" that MATLAB uses directly.
    """
    # build objective matrix for MINIMIZATION
    # obj(1,:) = -fitness, obj(2,:) = -rho
    objs = np.stack([-fitness, -rho], axis=1)

    fronts = nondominated_fronts_min(objs)

    # Now mimic what Fast_ND_SORT.m does:
    # It builds obj_new = [F; obj(1,:)] and then sortrows
    # That effectively means:
    # primary key = front number asc
    # secondary key = obj(1,:) asc = (-fitness) asc = fitness desc
    ordering = []
    for front_no, front in enumerate(fronts, start=1):
        for idx in front:
            ordering.append((front_no, -fitness[idx], idx))

    # sort by (front_no asc, -fitness asc) -> second key sorts by fitness desc
    ordering.sort(key=lambda t: (t[0], t[1]))
    sorted_idx = np.array([t[2] for t in ordering], dtype=int)
    return sorted_idx


# ============================================================
# NBC.m
# ============================================================

@dataclass
class Species:
    seed: int          # index of the "seed" solution (cluster head)
    idx: np.ndarray    # indices of all members in this species
    length: int        # len(idx)

def nbc_clustering(pop: np.ndarray) -> Tuple[List[Species], float]:
    """
    Port of NBC.m (Nearest-Better Clustering style in the paper code)
    pop: (N, D)

    Returns
     species_list: list of Species structs
     meandis: the distance threshold they used
    """
    N, dim = pop.shape
    min_num_edge = min(10, N)
    factor = 4 - np.log(dim)

    # nbc matrix: [node_i, parent_idx, dist_to_parent]
    nbc = np.zeros((N, 3), dtype=float)
    nbc[:, 0] = np.arange(N)
    nbc[0, 1] = -1   # first point is always a seed
    nbc[0, 2] = 0.0

    if dim == 5:
        # MATLAB "time ↓ memory ↑" branch
        dist_mat = np.sqrt(((pop[:, None, :] - pop[None, :, :]) ** 2).sum(axis=2))

        # replicate:
        # arrdis = tril(dist,1);
        # arrdis = arrdis + triu(Inf(n));
        # then for each i pick min along row i
        arrdis = np.tril(dist_mat, k=1) + np.triu(np.inf * np.ones_like(dist_mat))
        u = np.min(arrdis, axis=1)
        v = np.argmin(arrdis, axis=1)

        nbc[1:, 1] = v[1:]
        nbc[1:, 2] = u[1:]
    else:
        # MATLAB "time ↑ memory ↓" branch
        for i in range(1, N):
            dists = np.sqrt(((pop[i, :] - pop[:i, :]) ** 2).sum(axis=1))
            v = np.argmin(dists)
            u = dists[v]
            nbc[i, 1] = v
            nbc[i, 2] = u

    meandis = factor * np.mean(nbc[1:, 2])

    # they either chop all edges longer than meandis OR, if that would leave
    # too few, chop the top "min_num_edge" longest edges instead
    long_edges_mask = nbc[:, 2] > meandis
    if np.count_nonzero(long_edges_mask) >= min_num_edge:
        nbc[long_edges_mask, 1] = -1
        nbc[long_edges_mask, 2] = 0.0
    else:
        # fallback
        sort_idx = np.argsort(-nbc[:, 2])  # descending by distance
        if len(sort_idx) >= min_num_edge:
            cutoff = nbc[sort_idx[min_num_edge - 1], 2]
            mask_big = nbc[:, 2] >= cutoff
            nbc[mask_big, 1] = -1
            nbc[mask_big, 2] = 0.0

    # get "seeds"
    seeds = np.where(nbc[:, 1] == -1)[0]

    # assign each point to its seed by walking parent pointers
    parent_seed = np.zeros(N, dtype=int)
    for i in range(N):
        j = int(nbc[i, 1])
        k = j
        while j != -1:
            k = j
            j = int(nbc[j, 1])
        if k == -1:
            parent_seed[i] = i
        else:
            parent_seed[i] = k

    species_list: List[Species] = []
    for s in seeds:
        members = np.where(parent_seed == s)[0]
        species_list.append(
            Species(seed=int(s),
                    idx=members,
                    length=members.size)
        )

    return species_list, meandis


# ============================================================
# init_groups.m
# ============================================================

@dataclass
class CMAESOPTS:
    first: int = 1
    pop: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    val: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    sigma: float = 0.5
    count: int = 0

    # these get filled after first iteration
    lambda_: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    mu: int = 0
    mueff: float = 0.0
    cc: float = 0.0
    cs: float = 0.0
    c1: float = 0.0
    cmu: float = 0.0
    damps: float = 0.0
    pc: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    ps: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    D: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    C: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    chiN: float = 0.0
    countval: int = 0

@dataclass
class Group:
    idx: int
    OPTS: CMAESOPTS
    xmean: np.ndarray     # (D,)
    bestmem: np.ndarray   # (D,)
    bestval: float
    delta: float
    cc: float             # std(OPTS.val) after init / after CMA-ES step
    iters: int
    mean_distance: float  # mean distance of this group's xmean to others

def init_groups(problem,
                lambda_base: int,
                init_pop: np.ndarray,
                species_list: List[Species],
                fits: np.ndarray,
                sort_index: np.ndarray,
                rng: np.random.Generator,
                bestmem_set: Optional[np.ndarray]):
    """
    Port of init_groups.m
    problem: benchmark wrapper (DMMOP equivalent in Python)
    lambda_base: min_popsize = 7 + floor(3*log(D))
    init_pop: combined initial population (IDBPI + FDBPI result). shape (P,D)
    species_list: from nbc_clustering()
    fits: fitness of init_pop. shape (P,)
    sort_index: species sorted by size desc (exact like MATLAB)
    rng: seeded numpy RNG
    bestmem_set: archive from previous envs (array [K,D]) or None

    returns: List[Group] with mean_distance filled
    """
    groups: List[Group] = []
    D = problem.dim
    min_popsize = lambda_base

    # Build up to 9 groups from biggest species first
    for rank_i, sp_id in enumerate(sort_index):
        if rank_i >= 9:
            break

        sp = species_list[sp_id]

        # if species has enough individuals
        if sp.length >= min_popsize:
            base_pop = init_pop[sp.idx[:min_popsize], :]
            base_fit = fits[sp.idx[:min_popsize]]
        else:
            # take what exists, then we fill the rest by sampling ~N(xmean, sigma)
            base_pop = init_pop[sp.idx, :]
            base_fit = fits[sp.idx]

        xmean = np.mean(base_pop, axis=0)
        if sp.length == 1:
            sigma0 = 0.5
        else:
            # sigma = sqrt((1/(n*D))*sum((x - mean)^2))
            diff = base_pop - xmean
            sigma0 = np.sqrt(np.sum(diff ** 2) / ((base_pop.shape[0]) * D))

        # add Gaussian fill if not enough size
        if base_pop.shape[0] < min_popsize:
            add_n = min_popsize - base_pop.shape[0]
            add_pop = xmean + sigma0 * rng.normal(size=(add_n, D))
            add_pop = np.clip(add_pop, problem.lower_bound, problem.upper_bound)
            add_fit = problem.evaluate(add_pop)

            base_pop = np.vstack([base_pop, add_pop])
            base_fit = np.concatenate([base_fit, add_fit])

        cc_val = float(np.std(base_fit))

        # bestmem/bestval comes from the seed position of that species
        seed_idx = sp.seed
        bestmem = init_pop[seed_idx, :].astype(float)
        bestval = float(fits[seed_idx])

        g = Group(
            idx=len(groups),
            OPTS=CMAESOPTS(
                first=1,
                pop=base_pop.copy(),
                val=base_fit.copy(),
                sigma=float(sigma0),
                count=0
            ),
            xmean=xmean.astype(float),
            bestmem=bestmem,
            bestval=bestval,
            delta=0.0,
            cc=cc_val,
            iters=0,
            mean_distance=0.0  # we'll fill after we finish all groups
        )
        groups.append(g)

    # Archive-driven extra groups
    if bestmem_set is not None and bestmem_set.size > 0:
        for bm in bestmem_set:
            xmean = bm.astype(float)
            pop_gauss = xmean + rng.normal(size=(lambda_base, D))
            pop_gauss = np.clip(pop_gauss,
                                problem.lower_bound,
                                problem.upper_bound)
            val_gauss = problem.evaluate(pop_gauss)

            best_idx = int(np.argmax(val_gauss))
            bestval = float(val_gauss[best_idx])
            bestmem = pop_gauss[best_idx, :].copy()

            g = Group(
                idx=len(groups),
                OPTS=CMAESOPTS(
                    first=1,
                    pop=pop_gauss.copy(),
                    val=val_gauss.copy(),
                    sigma=0.5,
                    count=0
                ),
                xmean=xmean,
                bestmem=bestmem,
                bestval=bestval,
                delta=0.0,
                cc=float(np.std(val_gauss)),
                iters=0,
                mean_distance=10.0  # they hardcode 10 in MATLAB for archive groups
            )
            groups.append(g)

    # After all groups exist, compute mean_distance for each group
    if groups:
        means = np.stack([g.xmean for g in groups], axis=0)  # (G,D)
        dist_mat = np.sqrt(((means[:, None, :] - means[None, :, :]) ** 2).sum(axis=2))
        for gi, g in enumerate(groups):
            if len(groups) > 1:
                g.mean_distance = float(np.sum(dist_mat[gi, :]) / (len(groups) - 1))
            else:
                g.mean_distance = 0.0

    return groups


# ============================================================
# track_record.m
# ============================================================

def track_record(groups: List[Group]) -> np.ndarray:
    """Return float32 [xmean] stack like track_record.m."""
    if not groups:
        return np.zeros((0,), dtype=np.float32)
    return np.stack([g.xmean for g in groups], axis=0).astype(np.float32)


# ============================================================
# KE_CMA_ES.m
# ============================================================

def ke_cma_es(problem,
              group: Group,
              lb: np.ndarray,
              ub: np.ndarray,
              itermax: int,
              rng: np.random.Generator,
              temp_pop_all: np.ndarray,
              group_index: int,
              temp_best_pop: np.ndarray,
              track: np.ndarray):
    """
    Port of KE_CMA_ES.m with 1-to-1 mechanics:
     - sigma / CMA state carries across calls in group.OPTS
     - offspring is ranked using Fast_ND_SORT (fitness + density rho)
     - group can be discarded ("useless") if it stops helping
     - updates track with new xmean(s)

    Returns
     new_group (or None if removed),
     updated_track
    """

    D = problem.dim
    xmean = group.xmean.copy()
    bestmem = group.bestmem.copy()
    bestval = group.bestval
    OPTS = group.OPTS
    old_bestval = bestval

    # lambda = 7 + floor(3*log(D)), same as MATLAB
    lam = int(7 + np.floor(3 * np.log(D)))
    min_popsize = lam

    # temp_pop_all in MATLAB:
    #   temp_pop((ind-1)*lambda+1:ind*lambda,:) = group.OPTS.pop
    # then they truncate beyond that before feeding density.
    # For simplicity, caller should already pass the sliced temp_pop_all.
    temp_ref = temp_pop_all
    if temp_best_pop is not None and temp_best_pop.size > 0:
        temp_ref = np.vstack([temp_ref, temp_best_pop])

    # init CMA-ES strategy parameters
    sigma = OPTS.sigma
    if OPTS.first == 1:
        mu = lam // 2
        # recombination weights
        weights = np.log(np.arange(1, mu + 1) + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

        cc = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        c1 = 2 / (((D + 1.3) ** 2) + mu_eff)
        cmu = 2 * (mu_eff - 2 + 1 / mu_eff) / (((D + 2) ** 2) + 2 * mu_eff / 2.0)
        damps = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (D + 1)) - 1) + cs

        pc = np.zeros(D)
        ps = np.zeros(D)
        B = np.eye(D)
        Dmat = np.eye(D)
        C = B @ Dmat @ (B @ Dmat).T
        chiN = (D ** 0.5) * (1 - 1 / (4 * D) + 1 / (21 * (D ** 2)))
        countval = 0
        iters = group.iters
    else:
        mu = OPTS.mu
        weights = OPTS.weights.copy()
        mu_eff = OPTS.mueff
        cc = OPTS.cc
        cs = OPTS.cs
        c1 = OPTS.c1
        cmu = OPTS.cmu
        damps = OPTS.damps
        pc = OPTS.pc.copy()
        ps = OPTS.ps.copy()
        B = OPTS.B.copy()
        Dmat = OPTS.D.copy()
        C = OPTS.C.copy()
        chiN = OPTS.chiN
        countval = OPTS.countval
        iters = group.iters

    # run CMA-ES "itermax" generations or until convergence / killed
    stopiters = iters + itermax

    while iters < stopiters:
        # 1 Generate offspring (lambda)
        arz = rng.normal(size=(D, lam))
        arx = xmean[:, None] + sigma * (B @ Dmat @ arz)

        # reflect/clip to bounds, then recompute arz for clipped coords
        arx = np.clip(arx, lb[0], ub[0])

        # 2 Evaluate objective and density
        fit = problem.evaluate(arx.T)  # shape (lam,)
        rho_vals = kernel_density(arx.T, temp_ref)

        # 3 Fast_ND_SORT style ranking
        sorted_idx = fast_nd_sort_for_daea(fit, rho_vals)
        arx = arx[:, sorted_idx]
        arz = arz[:, sorted_idx]
        fit = fit[sorted_idx]
        rho_vals = rho_vals[sorted_idx]

        # 4 "useless group" check from MATLAB:
        # group.OPTS.count += length(rho>0)
        # if group.iters>0 && group.OPTS.count > 100/x (x=mean_distance)
        #   kill group
        group.OPTS.count += int(np.sum(rho_vals > 0))
        if group.iters > 0:
            x = group.mean_distance if group.mean_distance != 0 else 1e-9
            if group.OPTS.count > 100.0 / x:
                # kill / drop this group
                return None, track

        # 5 Recombination
        parent_idx = np.arange(mu)
        xmean = (arx[:, parent_idx] * weights[None, :]).sum(axis=1)
        zmean = (arz[:, parent_idx] * weights[None, :]).sum(axis=1)

        # 6 Evolution paths (ps, pc) and step-size / covariance updates
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ zmean)

        # hsig
        if (1 - cs) ** (2 * countval / lam) < 1e-30:
            denom_term = 1e-30  # avoid blowup
        else:
            denom_term = np.sqrt(1 - (1 - cs) ** (2 * countval / lam))
        hsig_cond = (
            np.linalg.norm(ps) / denom_term / chiN
            < (1.4 + 2 / (D + 1))
        )
        hsig = 1.0 if hsig_cond else 0.0

        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (B @ Dmat @ zmean)

        # covariance update
        BDz = (B @ Dmat @ arz[:, parent_idx])  # shape (D, mu)
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * BDz @ np.diag(weights) @ BDz.T)

        # sigma update
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))

        # 7 Eigen-decompose C to refresh B, Dmat
        # enforce symmetry
        C = np.triu(C) + np.triu(C, 1).T
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.abs(eigvals)
        Dmat = np.diag(np.sqrt(eigvals))
        B = eigvecs
        C = B @ Dmat @ (B @ Dmat).T  # keep it consistent/clean

        # 8 Best offspring update
        best_off_idx = int(np.argmax(fit))
        best_off_val = float(fit[best_off_idx])
        best_off_mem = arx[:, best_off_idx].copy()

        if best_off_val > bestval:
            bestval = best_off_val
            bestmem = best_off_mem

        iters += 1
        countval += 1

        # 9 stop if converged on that niche
        if np.std(fit) < 1e-6:
            break

        # track_record equivalent (append current state)
        track = np.vstack([track, xmean.astype(np.float32)])

    # write updated CMA-ES state back into the group
    OPTS.first = 0
    OPTS.pc = pc
    OPTS.ps = ps
    OPTS.B = B
    OPTS.D = Dmat
    OPTS.C = C
    OPTS.sigma = sigma
    OPTS.lambda_ = lam
    OPTS.weights = weights
    OPTS.mu = mu
    OPTS.mueff = mu_eff
    OPTS.cc = cc
    OPTS.cs = cs
    OPTS.c1 = c1
    OPTS.cmu = cmu
    OPTS.damps = damps
    OPTS.chiN = chiN
    OPTS.countval = countval
    OPTS.pop = arx.T.copy()
    OPTS.val = fit.copy()

    group.xmean = xmean
    group.bestmem = bestmem
    group.bestval = bestval
    group.OPTS = OPTS
    group.delta = bestval - old_bestval
    group.cc = float(np.std(fit))
    group.iters = iters

    return group, track


# ============================================================
# restart.m
# ============================================================

def restart_logic(track_arr: np.ndarray,
                  problem,
                  rng: np.random.Generator,
                  bestmem_set: np.ndarray,
                  bestval_set: np.ndarray):
    """
    Port of restart.m logic.
    Called when groups list becomes empty mid-env and we still have eval
    budget left in that environment.

    Returns:
     groups_new      (list[Group])
     updated_bestmem_set
     updated_bestval_set
    """

    rest = problem.freq - (problem.fe_counter % problem.freq)

    # "if rest == 0 || rest == pro.freq -> groups = [];"
    if rest == 0 or rest == problem.freq:
        return [], bestmem_set, bestval_set

    lam = int(7 + np.floor(3 * np.log(problem.dim)))

    # generate a big random pool in the decision space
    # (10 * rest , D)
    rand_pool = rng.uniform(problem.lower_bound,
                            problem.upper_bound,
                            size=(10 * rest, problem.dim)).astype(np.float32)

    rho_vals = kernel_density(rand_pool, track_arr.astype(np.float32))
    # sort ascending by rho (so pick low-density area first)
    order = np.argsort(rho_vals)

    if rest >= lam * 2:
        # we have enough budget to spawn a fresh CMA-ES group
        center = rand_pool[order[0], :]
        min_popsize = lam
        pop0 = center[None, :] + 0.5 * rng.normal(size=(min_popsize, problem.dim))
        pop0 = np.clip(pop0, problem.lower_bound, problem.upper_bound)
        fit0 = problem.evaluate(pop0)

        best_idx = int(np.argmax(fit0))
        new_group = Group(
            idx=0,
            OPTS=CMAESOPTS(
                first=1,
                pop=pop0.copy(),
                val=fit0.copy(),
                sigma=0.5,
                count=0,
            ),
            xmean=np.mean(pop0, axis=0),
            bestmem=pop0[best_idx, :].copy(),
            bestval=float(fit0[best_idx]),
            delta=0.0,
            cc=float(np.std(fit0)),
            iters=0,
            mean_distance=10.0
        )
        return [new_group], bestmem_set, bestval_set

    else:
        # not enough evals to properly restart
        # MATLAB branch tries to just harvest promising sols near global max
        subset = rand_pool[order[:rest], :]
        subset_fit = problem.evaluate(subset)

        if bestval_set.size > 0:
            thresh = np.max(bestval_set) - 1e-3
        else:
            thresh = np.max(subset_fit) - 1e-3

        promising_mask = subset_fit >= thresh
        if np.any(promising_mask):
            promising_pop = subset[promising_mask, :]
            promising_fit = subset_fit[promising_mask]

            if bestmem_set.size > 0:
                bestmem_set = np.vstack([bestmem_set, promising_pop])
                bestval_set = np.concatenate([bestval_set, promising_fit])
            else:
                bestmem_set = promising_pop.copy()
                bestval_set = promising_fit.copy()

        return [], bestmem_set, bestval_set


