import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ============================================================
# === STUB HELPERS (IDBPI / FDBPI / predictor)
# You should replace these with your real implementations
# from earlier when you plug in.
# ============================================================

def init_population_idbpi(lb, ub, init_popsize, dim, run_seed, rng, save_path=None):
    """
    MATLAB: IDBPI creates an initial distributed/balanced population.
    Here we just sample uniformly inside [lb, ub].
    """
    lb = np.array(lb, dtype=float).reshape(1, -1)
    ub = np.array(ub, dtype=float).reshape(1, -1)
    pop = rng.uniform(lb, ub, size=(init_popsize, dim))
    return pop

def fdbpi_generate(lb, ub, pop, fit_vals, init_popsize, dim, rng):
    """
    MATLAB: FDBPI = further density-based population injection.
    Real logic tries to generate more points near good stuff but
    still diverse.
    We'll approximate by adding Gaussian noise around the best few.
    """
    n = pop.shape[0]
    if n == 0:
        return np.zeros((0, dim))
    # pick top 10% best
    k = max(1, n // 10)
    best_idx = np.argsort(fit_vals)[::-1][:k]
    centers = pop[best_idx]
    out = []
    while len(out) < init_popsize:
        base = centers[rng.integers(0, centers.shape[0])]
        jitter = base + 0.1 * rng.normal(size=(1, dim))
        out.append(jitter[0])
    out = np.array(out, dtype=float)
    lb = np.array(lb).reshape(1, -1)
    ub = np.array(ub).reshape(1, -1)
    out = np.clip(out, lb, ub)
    return out

def evaluate_with_predictor(pro,
                            pop_I,
                            predictor_state,
                            Fn,
                            Run,
                            step,
                            rng,
                            verbose):
    """
    MATLAB used LSTM to 'predict forward' environments and adjust fitness.
    We do the simple thing: evaluate now.
    """
    fits = pro.evaluate(pop_I)
    return fits, predictor_state


# ============================================================
# kernel.m
# ============================================================

def kernel_density(candidates: np.ndarray,
                   reference: np.ndarray,
                   h: float = 1.1) -> np.ndarray:
    """
    Port of kernel.m
    candidates: (L, D)
    reference: (N, D)
    returns rho_mean: (L,)
    """
    if reference is None or reference.size == 0:
        return np.zeros(candidates.shape[0], dtype=float)

    diff = candidates[:, None, :] - reference[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    rho = np.exp(-dist_sq / (2.0 * (h ** 2)))

    # MATLAB cutoff
    rho_min = np.exp(-(h ** 2) / (2.0 * (h ** 2)))  # = exp(-1/2)
    rho[rho < rho_min] = 0.0

    rho_mean = np.mean(rho, axis=1)
    return rho_mean


# ============================================================
# Fast_ND_SORT.m (via nondominated sorting helper)
# ============================================================

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    # Pareto dominance for minimization
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

    # dominance graph
    for i in range(N):
        for j in range(i + 1, N):
            if _dominates(objs[i], objs[j]):
                dominates_list[i].append(j)
                dominated_count[j] += 1
            elif _dominates(objs[j], objs[i]):
                dominates_list[j].append(i)
                dominated_count[i] += 1

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
    Recreates Fast_ND_SORT(obj) ordering for KE_CMA_ES.
    fitness: shape (L,) higher is better
    rho:     shape (L,) higher is better

    Returns sorted indices "index".
    """
    objs = np.stack([-fitness, -rho], axis=1)  # we minimize -fitness,-rho
    fronts = nondominated_fronts_min(objs)

    ordering = []
    # MATLAB builds obj_new = [F; obj(1,:)] then sortrows
    # => first by front number asc, then by (-fitness) asc == fitness desc
    for front_no, front in enumerate(fronts, start=1):
        for idx in front:
            ordering.append((front_no, -fitness[idx], idx))

    ordering.sort(key=lambda t: (t[0], t[1]))
    sorted_idx = np.array([t[2] for t in ordering], dtype=int)
    return sorted_idx


# ============================================================
# NBC.m
# ============================================================

@dataclass
class Species:
    seed: int
    idx: np.ndarray
    length: int

def nbc_clustering(pop: np.ndarray) -> Tuple[List[Species], float]:
    """
    Port of NBC.m
    pop: (N, D)
    """
    N, dim = pop.shape
    min_num_edge = min(10, N)
    factor = 4 - np.log(dim)

    nbc = np.zeros((N, 3), dtype=float)
    nbc[:, 0] = np.arange(N)
    nbc[0, 1] = -1
    nbc[0, 2] = 0.0

    if dim == 5:
        dist_mat = np.sqrt(((pop[:, None, :] - pop[None, :, :]) ** 2).sum(axis=2))
        arrdis = np.tril(dist_mat, k=1) + np.triu(np.inf * np.ones_like(dist_mat))
        u = np.min(arrdis, axis=1)
        v = np.argmin(arrdis, axis=1)
        nbc[1:, 1] = v[1:]
        nbc[1:, 2] = u[1:]
    else:
        for i in range(1, N):
            dists = np.sqrt(((pop[i, :] - pop[:i, :]) ** 2).sum(axis=1))
            v = np.argmin(dists)
            u = dists[v]
            nbc[i, 1] = v
            nbc[i, 2] = u

    meandis = factor * np.mean(nbc[1:, 2])

    long_mask = nbc[:, 2] > meandis
    if np.count_nonzero(long_mask) >= min_num_edge:
        nbc[long_mask, 1] = -1
        nbc[long_mask, 2] = 0.0
    else:
        sort_idx = np.argsort(-nbc[:, 2])  # desc by distance
        if len(sort_idx) >= min_num_edge:
            cutoff = nbc[sort_idx[min_num_edge - 1], 2]
            mask_big = nbc[:, 2] >= cutoff
            nbc[mask_big, 1] = -1
            nbc[mask_big, 2] = 0.0

    # assign to seeds
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

    seeds = np.where(nbc[:, 1] == -1)[0]
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
    xmean: np.ndarray
    bestmem: np.ndarray
    bestval: float
    delta: float
    cc: float
    iters: int
    mean_distance: float

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
    """
    groups: List[Group] = []
    D = problem.dim
    min_popsize = lambda_base

    # biggest species first, up to 9
    for rank_i, sp_id in enumerate(sort_index):
        if rank_i >= 9:
            break
        sp = species_list[sp_id]

        if sp.length >= min_popsize:
            base_pop = init_pop[sp.idx[:min_popsize], :]
            base_fit = fits[sp.idx[:min_popsize]]
        else:
            base_pop = init_pop[sp.idx, :]
            base_fit = fits[sp.idx]

        xmean = np.mean(base_pop, axis=0)
        if sp.length == 1:
            sigma0 = 0.5
        else:
            diff = base_pop - xmean
            sigma0 = np.sqrt(np.sum(diff ** 2) / (base_pop.shape[0] * D))

        if base_pop.shape[0] < min_popsize:
            add_n = min_popsize - base_pop.shape[0]
            add_pop = xmean + sigma0 * rng.normal(size=(add_n, D))
            add_pop = np.clip(add_pop, problem.lower_bound, problem.upper_bound)
            add_fit = problem.evaluate(add_pop)

            base_pop = np.vstack([base_pop, add_pop])
            base_fit = np.concatenate([base_fit, add_fit])

        cc_val = float(np.std(base_fit))

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
            mean_distance=0.0
        )
        groups.append(g)

    # archive-driven groups
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
                mean_distance=10.0
            )
            groups.append(g)

    # fill mean_distance for each group
    if groups:
        means = np.stack([g.xmean for g in groups], axis=0)
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
              track_arr: np.ndarray):
    """
    Port of KE_CMA_ES.m in a near one-to-one manner.
    Returns (new_group_or_None, updated_track_arr).
    """
    D = problem.dim
    xmean = group.xmean.copy()
    bestmem = group.bestmem.copy()
    bestval = group.bestval
    OPTS = group.OPTS
    old_bestval = bestval

    lam = int(7 + np.floor(3 * np.log(D)))

    # ref set for kernel density
    temp_ref = temp_pop_all
    if temp_best_pop is not None and temp_best_pop.size > 0:
        temp_ref = np.vstack([temp_ref, temp_best_pop])

    sigma = OPTS.sigma
    if OPTS.first == 1:
        mu = lam // 2
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

    stopiters = iters + itermax

    while iters < stopiters:
        # offspring
        arz = rng.normal(size=(D, lam))
        arx = xmean[:, None] + sigma * (B @ Dmat @ arz)
        arx = np.clip(arx, lb[0], ub[0])

        # evaluate
        fit = problem.evaluate(arx.T)
        rho_vals = kernel_density(arx.T, temp_ref)

        # non-dom sort
        sorted_idx = fast_nd_sort_for_daea(fit, rho_vals)
        arx = arx[:, sorted_idx]
        arz = arz[:, sorted_idx]
        fit = fit[sorted_idx]
        rho_vals = rho_vals[sorted_idx]

        # useless group logic
        group.OPTS.count += int(np.sum(rho_vals > 0))
        if group.iters > 0:
            md = group.mean_distance if group.mean_distance != 0 else 1e-9
            if group.OPTS.count > 100.0 / md:
                # kill group
                return None, track_arr

        # recombination
        parent_idx = np.arange(mu)
        xmean = (arx[:, parent_idx] * weights[None, :]).sum(axis=1)
        zmean = (arz[:, parent_idx] * weights[None, :]).sum(axis=1)

        # evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ zmean)

        # hsig
        denom_term = np.sqrt(
            max(1e-30, 1 - (1 - cs) ** (2 * countval / lam))
        )
        hsig_cond = (np.linalg.norm(ps) / denom_term / chiN) < (1.4 + 2 / (D + 1))
        hsig = 1.0 if hsig_cond else 0.0

        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (B @ Dmat @ zmean)

        # covariance update
        BDz = (B @ Dmat @ arz[:, parent_idx])
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * BDz @ np.diag(weights) @ BDz.T)

        # sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))

        # eigen-decomp
        C = np.triu(C) + np.triu(C, 1).T
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.abs(eigvals)
        Dmat = np.diag(np.sqrt(eigvals))
        B = eigvecs
        C = B @ Dmat @ (B @ Dmat).T

        # update best
        best_off_idx = int(np.argmax(fit))
        best_off_val = float(fit[best_off_idx])
        best_off_mem = arx[:, best_off_idx].copy()

        if best_off_val > bestval:
            bestval = best_off_val
            bestmem = best_off_mem

        iters += 1
        countval += 1

        # convergence stop
        if np.std(fit) < 1e-6:
            break

        # track record
        track_arr = np.vstack([track_arr, xmean.astype(np.float32)])

    # writeback
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

    return group, track_arr


# ============================================================
# restart.m
# ============================================================

def restart_logic(track_arr: np.ndarray,
                  problem,
                  rng: np.random.Generator,
                  bestmem_set: np.ndarray,
                  bestval_set: np.ndarray):
    """
    Port of restart.m
    Returns:
      new_groups, bestmem_set, bestval_set
    """
    rest = problem.freq - (problem.fe_counter % problem.freq)
    if rest == 0 or rest == problem.freq:
        return [], bestmem_set, bestval_set

    lam = int(7 + np.floor(3 * np.log(problem.dim)))

    rand_pool = rng.uniform(
        problem.lower_bound,
        problem.upper_bound,
        size=(10 * rest, problem.dim)
    ).astype(np.float32)

    rho_vals = kernel_density(rand_pool, track_arr.astype(np.float32))
    order = np.argsort(rho_vals)  # pick sparsest area first

    if rest >= lam * 2:
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


# ============================================================
# Extra helpers from Main.m loop
# ============================================================

def prune_groups_if_stalled(groups: List[Group],
                            problem,
                            verbose: bool = False) -> List[Group]:
    """
    Remove groups that are clearly stalled.
    Heuristic: kill if cc ~ 0 AND sigma ~ tiny.
    """
    pruned = []
    for g in groups:
        sigma_now = g.OPTS.sigma
        cc_now = g.cc
        # if both diversity and step size collapsed extremely
        if (sigma_now < 1e-8) and (cc_now < 1e-8):
            if verbose:
                print("[prune] Dropping stalled group.")
            continue
        pruned.append(g)
    return pruned

def try_archive_group_if_converged(groups: List[Group],
                                   global_bestval: float,
                                   bestmem_set: List[np.ndarray],
                                   bestval_set: List[float],
                                   temp_best_pop: np.ndarray,
                                   tol_bestval: float = 1e-5,
                                   tol_sigma: float = 1e-6):
    """
    If a group's sigma is tiny and its bestval is near global_bestval,
    move its bestmem to the archive and remove the group.
    Also store its population into temp_best_pop (for density calc later).
    """
    keep_groups = []
    for g in groups:
        if (g.OPTS.sigma < tol_sigma) and (abs(g.bestval - global_bestval) <= tol_bestval):
            # archive this niche
            bestmem_set.append(g.bestmem.copy())
            bestval_set.append(g.bestval)
            # temp_best_pop keeps track of "already good" solutions
            if temp_best_pop.size == 0:
                temp_best_pop = g.OPTS.pop.copy()
            else:
                temp_best_pop = np.vstack([temp_best_pop, g.OPTS.pop.copy()])
        else:
            keep_groups.append(g)

    return keep_groups, bestmem_set, bestval_set, temp_best_pop

def merge_archived_into_groups(problem,
                               rng: np.random.Generator,
                               bestmem_set: List[np.ndarray],
                               bestval_set: List[float],
                               min_popsize: int,
                               tracker_arr: np.ndarray):
    """
    If we lost all groups but we *do* have an archive,
    respawn groups from archive bestmem_set.
    """
    groups: List[Group] = []
    if len(bestmem_set) == 0:
        # nothing to merge
        return groups, tracker_arr

    D = problem.dim
    # rebuild a CMAES group for each archived bestmem
    for bm in bestmem_set:
        xmean = bm.astype(float)
        pop_gauss = xmean + rng.normal(size=(min_popsize, D))
        pop_gauss = np.clip(pop_gauss,
                            problem.lower_bound,
                            problem.upper_bound)
        fit_gauss = problem.evaluate(pop_gauss)

        best_idx = int(np.argmax(fit_gauss))
        bestval = float(fit_gauss[best_idx])
        bestmem = pop_gauss[best_idx, :].copy()

        g = Group(
            idx=len(groups),
            OPTS=CMAESOPTS(
                first=1,
                pop=pop_gauss.copy(),
                val=fit_gauss.copy(),
                sigma=0.5,
                count=0
            ),
            xmean=xmean.copy(),
            bestmem=bestmem,
            bestval=bestval,
            delta=0.0,
            cc=float(np.std(fit_gauss)),
            iters=0,
            mean_distance=10.0
        )
        groups.append(g)

    # refresh tracker_arr with these xmeans
    tracker_arr = track_record(groups)
    return groups, tracker_arr

def select_two_groups_for_next_iter(groups: List[Group],
                                    bestmem_set: List[np.ndarray]) -> Tuple[int, Optional[int]]:
    """
    In MATLAB they pick:
      best group = highest bestval
      second group = next best distinct group if exists
    If there's only one group, second is None.
    """
    if len(groups) == 0:
        return 0, None
    # groups are usually kept sorted by bestval desc in our usage.
    idx_best = 0
    if len(groups) > 1:
        idx_second = 1
    else:
        idx_second = None
    return idx_best, idx_second

def compute_env_peaks_summary(problem,
                              pop_union: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    After finishing one environment:
      - count_found_peaks for eps_f in [1e-3, 1e-4, 1e-5]
      - get number of actual global peaks this env
    Returns (found_vec[3], total_peaks_scalar)
    """
    eps_levels = [1e-3, 1e-4, 1e-5]
    found_arr = []
    for eps in eps_levels:
        found_arr.append(
            problem.count_found_peaks(pop_union,
                                      eps_d=0.05,
                                      eps_f=eps)
        )
    found_arr = np.array(found_arr, dtype=float)

    true_peaks_now = len(problem.get_global_centers())
    total_peaks_scalar = float(true_peaks_now)

    return found_arr, total_peaks_scalar


# ============================================================
# DAEA global state across environments
# ============================================================

@dataclass
class DAEAState:
    groups: List[Group]
    bestmem_set: List[np.ndarray]
    bestval_set: List[float]
    temp_best_pop: np.ndarray
    tracker_arr: np.ndarray
    predictor_state: object
    pop_I: np.ndarray  # "seed" pop for the NEXT environment init


def _concat_temp_pop(groups: List[Group],
                     temp_best_pop: np.ndarray,
                     D: int) -> np.ndarray:
    all_pops = []
    for g in groups:
        all_pops.append(g.OPTS.pop)
    if temp_best_pop is not None and temp_best_pop.size > 0:
        all_pops.append(temp_best_pop)
    if len(all_pops) == 0:
        return np.zeros((0, D), dtype=float)
    return np.vstack(all_pops)


def run_one_env_cycle(pro,
                      state: DAEAState,
                      rng: np.random.Generator,
                      verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Recreates the inner while loop of Main.m:
    - We assume we're *at the start* of an environment already and state.groups
      has been initialized for this env.
    - We evolve until pro.check_change(...) says the env changed.
    - We return (peak_vec_for_this_env, total_peaks_this_env)
      so main can do PR accounting.
    """

    D = pro.dim
    min_popsize = 7 + int(np.floor(3 * np.log(D)))

    groups = state.groups
    bestmem_set = state.bestmem_set
    bestval_set = state.bestval_set
    temp_best_pop = state.temp_best_pop
    tracker_arr = state.tracker_arr

    # before exploitation, ensure groups are sorted by bestval desc
    groups = sorted(groups, key=lambda g: g.bestval, reverse=True)

    # exploitation loop:
    while not pro.check_change(bestmem_set, bestval_set):
        # prune collapsed groups
        groups = prune_groups_if_stalled(groups, pro, verbose=verbose)
        if len(groups) == 0:
            # restart
            new_groups, bestmem_set_arr, bestval_set_arr = restart_logic(
                track_arr=tracker_arr,
                problem=pro,
                rng=rng,
                bestmem_set=np.array(bestmem_set) if len(bestmem_set) else np.zeros((0, D)),
                bestval_set=np.array(bestval_set) if len(bestval_set) else np.zeros((0,))
            )
            # convert back to python lists
            bestmem_set = list(bestmem_set_arr) if bestmem_set_arr.ndim == 2 else []
            bestval_set = bestval_set_arr.tolist() if bestval_set_arr.ndim == 1 else []
            groups = new_groups
            if len(groups) == 0:
                # nothing to do until environment changes
                break

        # choose best / second
        idx_best, idx_second = select_two_groups_for_next_iter(groups, bestmem_set)

        # evolve best group
        temp_pop_all = _concat_temp_pop(groups, temp_best_pop, D)
        evolved_best, tracker_arr = ke_cma_es(
            problem=pro,
            group=groups[idx_best],
            lb=pro.lower_bound,
            ub=pro.upper_bound,
            itermax=20,
            rng=rng,
            temp_pop_all=temp_pop_all,
            group_index=idx_best,
            temp_best_pop=temp_best_pop,
            track_arr=tracker_arr,
        )
        if evolved_best is None:
            # group got killed
            groups.pop(idx_best)
        else:
            groups[idx_best] = evolved_best

        groups = [g for g in groups if g is not None]
        if len(groups) == 0:
            continue
        groups = sorted(groups, key=lambda g: g.bestval, reverse=True)

        # evolve second-best if still around and env not flagged to change
        if (idx_second is not None) and (not pro.change) and len(groups) > 1:
            idx_second = min(idx_second, len(groups) - 1)
            temp_pop_all = _concat_temp_pop(groups, temp_best_pop, D)
            evolved_second, tracker_arr = ke_cma_es(
                problem=pro,
                group=groups[idx_second],
                lb=pro.lower_bound,
                ub=pro.upper_bound,
                itermax=20,
                rng=rng,
                temp_pop_all=temp_pop_all,
                group_index=idx_second,
                temp_best_pop=temp_best_pop,
                track_arr=tracker_arr,
            )
            if evolved_second is None:
                groups.pop(idx_second)
            else:
                groups[idx_second] = evolved_second

            groups = [g for g in groups if g is not None]
            if len(groups) > 0:
                groups = sorted(groups, key=lambda g: g.bestval, reverse=True)

        if len(groups) == 0:
            # if everybody died, try to rebuild from archive
            groups, tracker_arr = merge_archived_into_groups(
                problem=pro,
                rng=rng,
                bestmem_set=bestmem_set,
                bestval_set=bestval_set,
                min_popsize=min_popsize,
                tracker_arr=tracker_arr,
            )
            if len(groups) == 0:
                # still none => we rely on restart on next loop
                continue

        # archive converged groups
        if len(groups) > 0:
            global_bestval = max(g.bestval for g in groups)
            (groups,
             bestmem_set,
             bestval_set,
             temp_best_pop) = try_archive_group_if_converged(
                groups=groups,
                global_bestval=global_bestval,
                bestmem_set=bestmem_set,
                bestval_set=bestval_set,
                temp_best_pop=temp_best_pop,
                tol_bestval=1e-5,
                tol_sigma=1e-6,
            )

        # if we removed everything by archiving
        if len(groups) == 0:
            groups, tracker_arr = merge_archived_into_groups(
                problem=pro,
                rng=rng,
                bestmem_set=bestmem_set,
                bestval_set=bestval_set,
                min_popsize=min_popsize,
                tracker_arr=tracker_arr,
            )
            if len(groups) == 0:
                # nothing else to do this environment
                continue

        # resort groups after updates
        groups = sorted(groups, key=lambda g: g.bestval, reverse=True)

    # environment ended (pro.check_change triggered a change internally)

    # gather union pop from all groups plus archive to evaluate "peak found"
    union_pops = []
    for g in groups:
        union_pops.append(g.OPTS.pop)
    if len(bestmem_set) > 0:
        union_pops.append(np.vstack(bestmem_set))
    if len(union_pops) > 0:
        all_union = np.vstack(union_pops)
    else:
        all_union = np.zeros((0, D))

    peak_vec_env, total_peaks_env = compute_env_peaks_summary(
        problem=pro,
        pop_union=all_union
    )

    # prepare for NEXT environment:
    # 1. reset temp_best_pop like MATLAB
    temp_best_pop = np.zeros((0, D), dtype=float)
    # 2. new pop_I (IDBPI) for next env
    lambda_size = min_popsize
    new_pop_I = init_population_idbpi(
        lb=pro.lower_bound,
        ub=pro.upper_bound,
        init_popsize=lambda_size,
        dim=D,
        run_seed=int(rng.integers(10**9)),
        rng=rng,
        save_path=None,
    )

    # 3. we have not yet re-run FDBPI/NBC/init_groups for the new env here.
    #    That happens in init_state() for env1, but for env>1
    #    the main loop outside can rebuild state.groups based on new_pop_I
    #    OR we can rebuild here immediately. We'll rebuild here so the caller
    #    of run_one_env_cycle() can just loop.

    # build new groups for the NEXT environment we just entered:
    fits_I, state.predictor_state = evaluate_with_predictor(
        pro=pro,
        pop_I=new_pop_I,
        predictor_state=state.predictor_state,
        Fn=0,
        Run=0,
        step=10,
        rng=rng,
        verbose=verbose,
    )

    pop_F = fdbpi_generate(
        lb=pro.lower_bound,
        ub=pro.upper_bound,
        pop=new_pop_I,
        fit_vals=fits_I,
        init_popsize=int(0.1 * pro.freq),
        dim=D,
        rng=rng,
    )
    fits_F = pro.evaluate(pop_F)

    init_pop_next = np.vstack([new_pop_I, pop_F])
    init_fit_next = np.concatenate([fits_I, fits_F])

    species_list, _ = nbc_clustering(init_pop_next)
    # sort species by length desc
    species_lengths = np.array([sp.length for sp in species_list])
    sort_index = np.argsort(-species_lengths)

    groups_next = init_groups(
        problem=pro,
        lambda_base=lambda_size,
        init_pop=init_pop_next,
        species_list=species_list,
        fits=init_fit_next,
        sort_index=sort_index,
        rng=rng,
        bestmem_set=np.array(bestmem_set) if len(bestmem_set) else None,
    )

    tracker_arr_next = track_record(groups_next)

    # update state in-place for caller (so outer loop can reuse)
    state.groups = groups_next
    state.bestmem_set = bestmem_set
    state.bestval_set = bestval_set
    state.temp_best_pop = temp_best_pop
    state.tracker_arr = tracker_arr_next
    state.pop_I = new_pop_I

    return peak_vec_env, total_peaks_env


def init_state(pro,
               rng: np.random.Generator,
               Fn: int,
               Run: int,
               step: int,
               verbose: bool = False) -> DAEAState:
    """
    Equivalent to the "Initialization ... IDBPI/FDBPI/NBC/init_groups"
    part of Main.m at env = 1.
    Builds first state for env1.
    """

    D = pro.dim
    min_popsize = 7 + int(np.floor(3 * np.log(D)))

    # IDBPI
    pop_I = init_population_idbpi(
        lb=pro.lower_bound,
        ub=pro.upper_bound,
        init_popsize=min_popsize,
        dim=D,
        run_seed=Run,
        rng=rng,
        save_path=None,
    )

    # predictor_state stub
    predictor_state = None

    # evaluate_with_predictor
    fits_I, predictor_state = evaluate_with_predictor(
        pro=pro,
        pop_I=pop_I,
        predictor_state=predictor_state,
        Fn=Fn,
        Run=Run,
        step=step,
        rng=rng,
        verbose=verbose,
    )

    # FDBPI
    pop_F = fdbpi_generate(
        lb=pro.lower_bound,
        ub=pro.upper_bound,
        pop=pop_I,
        fit_vals=fits_I,
        init_popsize=int(0.1 * pro.freq),
        dim=D,
        rng=rng,
    )
    fits_F = pro.evaluate(pop_F)

    init_pop = np.vstack([pop_I, pop_F])
    init_fit = np.concatenate([fits_I, fits_F])

    # NBC clustering
    species_list, _ = nbc_clustering(init_pop)
    species_lengths = np.array([sp.length for sp in species_list])
    sort_index = np.argsort(-species_lengths)

    # no archive yet at env1
    bestmem_set: List[np.ndarray] = []
    bestval_set: List[float] = []
    temp_best_pop = np.zeros((0, D), dtype=float)

    # init groups
    groups = init_groups(
        problem=pro,
        lambda_base=min_popsize,
        init_pop=init_pop,
        species_list=species_list,
        fits=init_fit,
        sort_index=sort_index,
        rng=rng,
        bestmem_set=None,
    )

    tracker_arr = track_record(groups)

    return DAEAState(
        groups=groups,
        bestmem_set=bestmem_set,
        bestval_set=bestval_set,
        temp_best_pop=temp_best_pop,
        tracker_arr=tracker_arr,
        predictor_state=predictor_state,
        pop_I=pop_I,
    )


# ============================================================
# === Official CEC2022 DAEA Benchmark Runner (P1–P24)
# ============================================================
if __name__ == "__main__":
    from cectest import DMMOProblem

    configs = [
        # --- G1: F1–F8, C1, D=5 ---
        *[(f"F{i}", "C1", 5) for i in range(1, 9)],
        # --- G2: F8, C1–C8, D=5 ---
        *[("F8", f"C{i}", 5) for i in range(1, 9)],
        # --- G3: F1–F8, C1, D=10 ---
        *[(f"F{i}", "C1", 10) for i in range(1, 9)],
    ]

    run_id = 26
    step = 10
    num_envs = 60
    all_results = []

    print("=== Starting Official DAEA Reproduction (P1–P24) ===")

    for idx, (f_id, c_mode, dim) in enumerate(configs, start=1):
        label = f"P{idx}_{f_id}_{c_mode}_D{dim}"
        print(f"\n=== Running {label} ===")

        rng = np.random.default_rng(run_id)
        pro = DMMOProblem(func_id=f_id, change_mode=c_mode, dim=dim, seed=run_id)

        state = init_state(pro=pro, rng=rng, Fn=int(f_id[1:]), Run=run_id, step=step, verbose=False)

        pr_accum = np.zeros(3, dtype=float)
        total_peaks_accum = 0.0

        for env_idx in range(1, num_envs + 1):
            print(f"[{label}] Env {env_idx}/{num_envs}")
            peak_vec, total_peaks = run_one_env_cycle(pro=pro, state=state, rng=rng, verbose=False)
            pr_accum += peak_vec
            total_peaks_accum += total_peaks

        pr_result = pr_accum / total_peaks_accum
        print(f"[{label}] Final PR: {pr_result}")

        fname = f"DAEA_{label}_Run{run_id}.txt"
        np.savetxt(fname, pr_result.reshape(1, -1), fmt="%.6f")
        all_results.append((label, pr_result))
        print(f"Saved to {fname}")

    print("\n=== ALL RESULTS SUMMARY ===")
    for label, pr in all_results:
        print(f"{label}: PR(1e-3,1e-4,1e-5) = {pr}")

    np.savetxt(
        "DAEA_AllResults_P1_P24.txt",
        np.array([r[1] for r in all_results]),
        fmt="%.6f",
        header="PR(1e-3, 1e-4, 1e-5) for P1–P24 per Table 1"
    )
    print("\nSaved summary file: DAEA_AllResults_P1_P24.txt")
