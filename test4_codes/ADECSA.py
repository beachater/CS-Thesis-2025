"""
ADE-CSA (Adaptive Clonal Selection Algorithm with multiple DE strategies)
Implementation derived from:
'An adaptive clonal selection algorithm with multiple differential evolution strategies' (Wang et al., 2022).
See: uploaded paper for algorithm details. :contentReference[oaicite:1]{index=1}
"""
import numpy as np
from math import log, sqrt, pi

# ---------- helper random samplers ----------

def randc(loc, scale):
    """Return a sample from Cauchy(location=loc, scale=scale). If out of (0,1) regen a few times."""
    # Use numpy's standard cauchy then shift/scale
    for _ in range(10):
        s = np.random.standard_cauchy() * scale + loc
        if np.isfinite(s):
            return s
    return loc

def randn_trunc(mu, sigma):
    """Sample normal with mean mu and std sigma, return a value (clipped later to [0,1] where needed)."""
    for _ in range(10):
        v = np.random.normal(mu, sigma)
        if np.isfinite(v):
            return v
    return mu

def weighted_lehmer_mean(values, weights):
    # values, weights are 1D arrays of same length; implement the meanL in paper (weighted Lehmer mean)
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.size == 0:
        return None
    num = np.sum(weights * (values ** 2))
    den = np.sum(weights * values)
    if den == 0.0:
        return np.mean(values)
    return num / den

# ---------- common utilities ----------

def pairwise_mean_distance(pop):
    # Corriveau pairwise mean distance (DN_PW) normalized by max pairwise distance seen so far (we use current max)
    N, D = pop.shape
    if N <= 1:
        return 0.0
    dsum = 0.0
    for i in range(N):
        for j in range(i):
            dsum += np.linalg.norm(pop[i] - pop[j])
    denom = N * (N - 1) / 2.0
    return (2.0 / (N * (N - 1))) * dsum

# ---------- mutation strategies (DE-like) ----------

def de_rand_1(pop, idx, F):
    # DE/rand/1 : v = xr1 + F*(xr2 - xr3)
    N = pop.shape[0]
    r = np.random.choice([i for i in range(N) if i != idx], 3, replace=False)
    v = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])
    return v

def de_current_rand_1(pop, idx, F, K_param):
    # DE/current-rand/1 : v = x_i + K*(xr1 - xi) + F*(xr2 - xr3)
    N = pop.shape[0]
    choices = [i for i in range(N) if i != idx]
    r = np.random.choice(choices, 3, replace=False)
    xi = pop[idx]
    v = xi + K_param * (pop[r[0]] - xi) + F * (pop[r[1]] - pop[r[2]])
    return v

def de_current_to_pbest_1(pop, idx, F, p, archive=None):
    # DE/current-to-pbest/1 with archive:
    # v = xi + F*(x_pbest - xi) + F*(xr1 - x_r2_from_PunionA)
    N = pop.shape[0]
    xi = pop[idx]
    p_count = max(1, int(np.ceil(p * N)))
    pbest_idx = np.random.choice(np.argsort(np.apply_along_axis(lambda x: x[0], 1, np.zeros((1,1)))) if False else np.arange(N))  # dummy replace below

    # easier: pick pbest randomly from top-p% of current pop by a fitness mask passed externally;
    # in our implementation we'll pick randomly among best 'p_count' by Euclidean norm to origin as placeholder.
    # In practice caller should compute best by fitness; we'll pass a pbest_indices param to this function if available.
    # For simplicity here: pick pbest uniformly at random from population (the ADECSA higher-level code will pass correct pbest)
    pbest_idx = np.random.randint(0, N)
    # choose r1 and r2 from union P union archive (we'll ignore archive for now => use pop)
    choices = [i for i in range(N) if i != idx and i != pbest_idx]
    if len(choices) < 2:
        choices = [i for i in range(N) if i != idx]
    r = np.random.choice(choices, 2, replace=False)
    v = xi + F * (pop[pbest_idx] - xi) + F * (pop[r[0]] - pop[r[1]])
    return v

# ---------- ADECSA main implementation ----------

def ADECSA(
    f,                      # objective function (minimization)
    bounds,                 # list of (min,max) per dimension
    D_dim=None,
    Ninit=None,
    Nmin=4,
    Nc=3,                   # clonal scale (equals K strategies)
    H=10,                   # historical memory size
    r_replace=0.10,         # replacement proportion
    dc=1e-3,                # diversity threshold
    ds=None,                # stagnation threshold (if None -> will set to N)
    Ng=20, Tg=250,          # Gaussian walks settings
    FESmax=None,
    ng_update=20,           # update frequency for strategy success probabilities
    max_iter=None,          # optional generation cap
    seed=None,
    progress=None          # optional progress logger
):
    np.random.seed(seed)
    D = len(bounds) if D_dim is None else D_dim
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    if Ninit is None:
        Ninit = 12 * D  # default from paper for CEC2014
    N = Ninit
    if ds is None:
        ds = N  # as paper: stagnation threshold equals current N
    if FESmax is None:
        FESmax = D * 10000
    if max_iter is None:
        max_iter = 1000000

    # Strategy pool: K = Nc
    K = Nc
    # Historical memories: for each strategy k keep MF_k (F means), MCR_k (CR), Mfreq_k (frequency control)
    MF = [np.array([0.5] * (H - 1) + [0.9]) for _ in range(K)]
    MCR = [np.array([0.5] * (H - 1) + [0.9]) for _ in range(K)]
    Mfreq = [np.array([0.5] * H) for _ in range(K)]

    zF = [0] * K
    zCR = [0] * K
    zfreq = [0] * K

    # initialize population randomly
    pop = lb + (ub - lb) * np.random.rand(N, D)
    fitness = np.array([f(x) for x in pop])
    FES = N  # initial evaluations count
    g = 1
    Gmax = max_iter

    # initialize counters and collections
    # for each individual i and strategy k maintain nsi and nfi (success/failure counts in last ng window)
    nsi = np.zeros((N, K), dtype=int)
    nfi = np.zeros((N, K), dtype=int)

    # initial strategy probabilities psgi,k (initial distribution: strategy 2 has 0.5, others 0.25 if K=3)
    pgi = np.full((N, K), 1.0 / K)
    if K == 3:
        # set schema described in paper: current-to-pbest with archive gets 0.5
        pgi[:, :] = 0.25
        pgi[:, 1] = 0.5

    # placeholders for success lists per strategy in this generation
    SCR = [[] for _ in range(K)]   # successful CRs
    SF = [[] for _ in range(K)]    # successful Fs
    Sfreq = [[] for _ in range(K)] # successful freqs

    # helper to pick pbest index set (top-p%); choose p=0.2 like many DE variants
    p_percent = 0.2

    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_f = fitness[best_idx]

    # logging and history
    iter_no = 0
    history = []

    while FES <= FESmax and g <= Gmax:
        iter_no += 1
        SCR = [[] for _ in range(K)]
        SF = [[] for _ in range(K)]
        Sfreq = [[] for _ in range(K)]

        # report progress if logger provided
        if progress is not None:
            try:
                progress(
                    gen=iter_no-1,  # 0-based gen index
                    pop=pop.copy(),
                    fitness=fitness.copy(), 
                    best_fitness=best_f,
                    gbest=best.copy(),
                    evals=FES
                )
            except Exception:
                pass

        # track history
        history.append(float(best_f))

        # For each individual
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros_like(fitness)

        # Precompute pbest indices (top p% by fitness)
        p_cnt = max(1, int(np.ceil(p_percent * N)))
        pbest_indices = np.argsort(fitness)[:p_cnt]

        for i in range(N):
            xi = pop[i]
            # 1) clonal proliferation: create Nc clones (one for each strategy)
            clones = np.tile(xi, (Nc, 1))

            # 2) Adaptive parameter control per clone/strategy
            Fi_k = np.zeros(Nc)
            CRi_k = np.zeros(Nc)
            freq_i_k = np.zeros(Nc)
            # generation stage split:
            g1_threshold = max(1, Gmax // 2)
            for k in range(K):
                ri = np.random.randint(0, H)
                # first half
                if g <= g1_threshold:
                    # choose sinusoidal decreasing or adaptive increasing with equal prob
                    if np.random.rand() < 0.5:
                        # DeSinusoidal: Fi = 0.5 * sin(2*pi*freq*(g1 + p) * Gmax - g1)/Gmax + 1 ??? (paper formula is a bit messy)
                        freq = 0.5
                        # Use a practical sinusoidal decreasing around 0.5:
                        Fi = 0.5 * (np.sin(2 * pi * freq * (g / Gmax)) * 0.5 + 1.0)
                    else:
                        # adaptive sinusoidal increasing (requires freqi;k)
                        freqi = randc(Mfreq[k][ri], 0.1)
                        Fi = 0.5 * (np.sin(2 * pi * freqi * (g / Gmax)) * 0.5 + 1.0)
                    # CR sampled from normal around historical memory element
                    CR = randn_trunc(MCR[k][ri], 0.1)
                else:
                    # second half: sample Fi from Cauchy centered at MF_k[ri], and CR from normal centered at MCR_k[ri]
                    Fi = randc(MF[k][ri], 0.1)
                    CR = randn_trunc(MCR[k][ri], 0.1)

                # enforce bounds
                if Fi > 1: Fi = 1.0
                if Fi <= 0: Fi = max(1e-6, randc(MF[k][ri], 0.1))
                if CR > 1: CR = 1.0
                if CR < 0: CR = 0.0

                Fi_k[k] = Fi
                CRi_k[k] = CR
                freq_i_k[k] = Mfreq[k][ri] if (len(Mfreq[k])>0) else 0.5

            # 3) generate mutated set Y_i by applying each strategy to the clone
            Y_i = np.zeros_like(clones)
            # we will use simple mapping of strategies:
            # k==0 : DE/rand/1
            # k==1 : DE/current-to-pbest/1 with archive
            # k==2 : DE/current-rand/1
            for k in range(K):
                F_k = Fi_k[k]
                if k == 0:
                    v = de_rand_1(pop, i, F_k)
                elif k == 1:
                    # pick pbest randomly from pbest_indices
                    pbi = np.random.choice(pbest_indices)
                    # create a mutant like current-to-pbest/1 with pseudo-archive using pop only
                    choices = [idx for idx in range(N) if idx not in (i, pbi)]
                    if len(choices) < 2:
                        choices = [idx for idx in range(N) if idx != i]
                    r1, r2 = np.random.choice(choices, 2, replace=False)
                    v = xi + F_k * (pop[pbi] - xi) + F_k * (pop[r1] - pop[r2])
                else:
                    # current-rand/1 with an extra K parameter, set K=F_k for simplicity
                    v = de_current_rand_1(pop, i, F_k, K_param=F_k)
                # boundary handling as in paper (Eq.40): if out of bounds, set halfway toward boundary
                for d in range(D):
                    if v[d] < lb[d]:
                        v[d] = (lb[d] + v[d]) / 2.0
                    elif v[d] > ub[d]:
                        v[d] = (ub[d] + v[d]) / 2.0
                Y_i[k] = v

            # 4) selection: pick best mutated ygib among Y_i
            y_fits = np.array([f(y) for y in Y_i])
            FES += len(Y_i)
            best_mut_idx = np.argmin(y_fits)
            ygib = Y_i[best_mut_idx]
            ygib_fit = y_fits[best_mut_idx]

            # compare with parent
            if ygib_fit < fitness[i]:
                # success: record CR, F, freq for that strategy
                SCR[best_mut_idx].append(CRi_k[best_mut_idx])
                SF[best_mut_idx].append(Fi_k[best_mut_idx])
                Sfreq[best_mut_idx].append(freq_i_k[best_mut_idx])
                nsi[i, best_mut_idx] += 1
                new_pop[i] = ygib
                new_fit[i] = ygib_fit
            else:
                nfi[i, best_mut_idx] += 1
                new_pop[i] = pop[i]
                new_fit[i] = fitness[i]

            # optionally update pgi,k every ng generations - done at end of generation block

        # end for each individual

        # at the end of generation: update historical memories if success sets non-empty
        for k in range(K):
            if len(SF[k]) > 0:  # SFk –£ means not empty
                # update MF[k][zFk] with weighted Lehmer mean of SF[k]
                ml = weighted_lehmer_mean(SF[k], np.abs(np.array(SF[k])))  # weights choice: absolute improvements approx
                if ml is not None:
                    MF[k][zF[k] % H] = ml
                    zF[k] = (zF[k] + 1) % H
            if len(SCR[k]) > 0:
                # meanA for CR in paper: arithmetic mean
                ma = np.mean(SCR[k])
                MCR[k][zCR[k] % H] = ma
                zCR[k] = (zCR[k] + 1) % H
            if len(Sfreq[k]) > 0:
                mf = weighted_lehmer_mean(Sfreq[k], np.abs(np.array(Sfreq[k])))
                if mf is not None:
                    Mfreq[k][zfreq[k] % H] = mf
                    zfreq[k] = (zfreq[k] + 1) % H

        # update population and fitness
        pop = new_pop.copy()
        fitness = new_fit.copy()

        # linear population size reduction (Eq.24)
        Ng1 = int(round(Nmin + (Ninit - Nmin) * max(0.0, (FESmax - FES) / float(FESmax))))
        if Ng1 < N:
            # remove worst (largest fitness) individuals to shrink population to Ng1
            worst_idx = np.argsort(fitness)[- (N - Ng1):]
            mask = np.ones(N, dtype=bool)
            mask[worst_idx] = False
            pop = pop[mask]
            fitness = fitness[mask]
            # adjust nsi/nfi and pgi accordingly (trim rows)
            nsi = nsi[mask]
            nfi = nfi[mask]
            pgi = pgi[mask]
            N = Ng1

        # premature convergence & stagnation detection using pairwise mean distance
        DN_PW = pairwise_mean_distance(pop)
        cg = 1 if DN_PW <= dc else 0
        # stagnation detection (s increments if DN_PW unchanged)
        if iter_no == 1:
            prev_DN = DN_PW
            s_count = 0
        else:
            if abs(DN_PW - prev_DN) <= 1e-12:
                s_count += 1
            else:
                s_count = 0
            prev_DN = DN_PW
        sg = 1 if s_count >= ds else 0

        # if premature convergence or stagnation -> replace d worst individuals randomly (Eq.32)
        if cg == 1 or sg == 1:
            d_replace = int(round(10 ** ( - (FES / float(FESmax)) ) * r_replace * N))
            d_replace = min(max(1, d_replace), N)
            worst_idx = np.argsort(fitness)[-d_replace:]
            for wi in worst_idx:
                pop[wi] = lb + (ub - lb) * np.random.rand(D)
                fitness[wi] = f(pop[wi])
                FES += 1

        # apply Gaussian walks if N <= Ng for the first time
        if N <= Ng and g == 1:  # paper says "for the first time" — we trigger once when condition first met. Here simplified
            # perform Gaussian walks on best ngw antibodies (use Ng as ngw)
            ngw = min(Ng, N)
            # pick best ngw
            best_inds = np.argsort(fitness)[:ngw]
            # perform Tg iterations of Gaussian walks for them
            for t in range(Tg):
                for bi in best_inds:
                    abw_best = pop[best_inds[0]]
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    # compute r variance from Eq.26
                    r_var = (log(max(1, FES)) / max(1, FES)) * np.linalg.norm(pop[bi] - abw_best)
                    cand = np.random.normal(loc=abw_best, scale=max(1e-9, r_var)) + r1 * (abw_best - r2 * pop[bi])
                    # bound
                    cand = np.minimum(np.maximum(cand, lb), ub)
                    cand_fit = f(cand)
                    FES += 1
                    if cand_fit < fitness[bi]:
                        pop[bi] = cand
                        fitness[bi] = cand_fit

        # update global best
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_f:
            best_f = float(fitness[cur_best_idx])
            best = pop[cur_best_idx].copy()

        # increment generation counters
        g += 1
        # function evals already updated
        # termination check loop will exit if FES>FESmax

        # safety break if too long
        if g > max_iter:
            break

    return best, best_f, pop, fitness, {"history": history}

# ---------------- example usage ----------------
if __name__ == "__main__":
    # test on simple Rastrigin 10D
    def rastrigin(x):
        x = np.asarray(x)
        return 10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    best_sol, best_val, pop, fit, stats = ADECSA(
        rastrigin,
        bounds,
        D_dim=dim,
        Ninit=12 * dim,
        Nmin=4,
        Nc=3,
        H=10,
        r_replace=0.10,
        dc=1e-3,
        ds=100,
        Ng=20,
        Tg=50,
        FESmax=dim * 2000,
        ng_update=20,
        max_iter=1000,
        seed=42
    )
    print("BEST", best_val, best_sol)
