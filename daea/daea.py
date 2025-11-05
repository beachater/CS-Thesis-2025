import numpy as np
from typing import Optional, Tuple

class DAEAState:
    """
    Holds cross-environment memory for Density-Assisted Evolutionary Dynamic
    Multimodal Optimization (DAEA) as described by Zhu et al. 2025. :contentReference[oaicite:7]{index=7}

    - archive_X: elite historical solutions from previous environments
    - archive_F: their fitness
    """
    def __init__(self, max_archive:int=200):
        self.max_archive = max_archive
        self.archive_X = None  # shape (A,D)
        self.archive_F = None  # shape (A,)
        # we can also keep a generation counter if needed later

    def update_archive(self, pop:np.ndarray, fit:np.ndarray, niche_radius:float=0.1):
        """
        After finishing an environment, store sparse elites into archive.
        We keep best representative per niche (niche defined by Euclidean radius).
        This matches the spirit of keeping one elite per basin/peak. :contentReference[oaicite:8]{index=8}
        """
        # sort by fitness descending (maximization)
        idx_sort = np.argsort(-fit)
        kept = []
        for idx in idx_sort:
            cand = pop[idx]
            ok = True
            for k in kept:
                if np.linalg.norm(cand - k) < niche_radius:
                    ok = False
                    break
            if ok:
                kept.append(cand.copy())
            if len(kept) >= self.max_archive:
                break

        arch_X_new = np.array(kept) if kept else pop[np.argmax(fit)][None,:]
        arch_F_new = np.array([np.max(fit[:len(kept)])]) if kept else np.array([np.max(fit)])

        if self.archive_X is None:
            self.archive_X = arch_X_new
            self.archive_F = arch_F_new
        else:
            # merge with existing archive then sparsify again
            comb_X = np.vstack([self.archive_X, arch_X_new])
            comb_F = np.concatenate([self.archive_F, arch_F_new])
            # keep best sparse set again
            idx_sort2 = np.argsort(-comb_F)
            kept2 = []
            kept2_f = []
            for idx in idx_sort2:
                cand = comb_X[idx]
                ok = True
                for k in kept2:
                    if np.linalg.norm(cand - k) < niche_radius:
                        ok = False
                        break
                if ok:
                    kept2.append(cand.copy())
                    kept2_f.append(comb_F[idx])
                if len(kept2) >= self.max_archive:
                    break
            self.archive_X = np.array(kept2)
            self.archive_F = np.array(kept2_f)

    def inject_population(self,
                          pop_size:int,
                          dim:int,
                          rng:np.random.Generator,
                          bounds:Tuple[float,float]=(-5.0,5.0),
                          reuse_rate:float=0.5) -> np.ndarray:
        """
        When a new environment starts, create a new starting population
        that mixes archived elites (to help track moving peaks)
        and random exploration (to adapt to new peaks). :contentReference[oaicite:9]{index=9}
        """
        low, high = bounds
        if (self.archive_X is None) or (self.archive_X.shape[0] == 0):
            # no memory yet -> full random init
            return rng.uniform(low, high, size=(pop_size, dim))

        k_reuse = int(pop_size * reuse_rate)
        k_rand = pop_size - k_reuse

        # pick top k_reuse archive members (or all if fewer)
        arch_sel_idx = np.arange(min(k_reuse, self.archive_X.shape[0]))
        reused = self.archive_X[arch_sel_idx].copy()

        # add small gaussian jitter so they can re-track shifted peaks
        jitter = rng.normal(scale=0.05, size=reused.shape)
        reused = np.clip(reused + jitter, low, high)

        fresh = rng.uniform(low, high, size=(k_rand, dim))

        return np.vstack([reused, fresh])


def _estimate_density(pop:np.ndarray,
                      archive:Optional[np.ndarray],
                      k:int=5) -> np.ndarray:
    """
    Estimate density for each individual.
    We'll define density ~ 1 / (distance to k-th nearest neighbor),
    using combined set [pop ∪ archive] as reference.
    Lower density value means isolated => good.
    We'll actually return crowding = 1 / (d_k + eps), so higher crowding = more crowded.
    """
    X = pop if archive is None else np.vstack([pop, archive])
    N = pop.shape[0]
    crowding = np.zeros(N)

    for i in range(N):
        diff = X - pop[i]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dist.sort()
        # dist[0] is 0 (self). take k-th neighbor safely
        kk = min(k, len(dist)-1)
        dk = dist[kk]
        crowding[i] = 1.0 / (dk + 1e-12)

    return crowding  # higher = more crowded, lower = more isolated


def daea_epoch(pop: np.ndarray,
               fitness: np.ndarray,
               problem,
               rng: Optional[np.random.Generator],
               daea_state: DAEAState,
               F: float = 0.5,
               CR: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    One evolutionary epoch of DAEA-like update:
    - DE/rand/1/bin to propose trial vectors
    - density-assisted survivor selection
    This is aligned with the density-assisted selection idea in DAEA, where
    sparsity can override small fitness disadvantage to preserve niches. :contentReference[oaicite:10]{index=10}
    """
    if rng is None:
        rng = np.random.default_rng()

    NP, D = pop.shape
    new_pop = pop.copy()
    new_fit = fitness.copy()

    # Precompute crowding of parents using archive
    parent_crowd = _estimate_density(pop, daea_state.archive_X, k=5)

    for i in range(NP):
        # classic DE/rand/1/bin mutation
        idxs = [idx for idx in range(NP) if idx != i]
        if len(idxs) < 3:
            continue
        a,b,c = rng.choice(idxs, size=3, replace=False)
        mutant = pop[a] + F * (pop[b] - pop[c])

        # binomial crossover
        trial = pop[i].copy()
        jrand = rng.integers(0, D)
        for j in range(D):
            if rng.random() < CR or j == jrand:
                trial[j] = mutant[j]

        # clip to search bounds
        trial = np.clip(trial, -5, 5)

        # evaluate offspring
        trial_fit = float(problem.evaluate(trial.reshape(1,-1))[0])

        # density (crowding) for parent and trial
        # cheaply approximate trial crowding by comparing to pop+archive
        trial_crowd = _estimate_density(trial.reshape(1,-1), daea_state.archive_X, k=5)[0]

        # survivor selection rule:
        # if offspring better fitness -> take it
        # else if offspring slightly worse (<delta) BUT less crowded -> take it anyway
        # delta acts like tolerance for 'almost as good':
        delta = 1e-6
        if (trial_fit > fitness[i]) or (
            (fitness[i] - trial_fit) < delta and trial_crowd < parent_crowd[i]
        ):
            new_pop[i] = trial
            new_fit[i] = trial_fit
            parent_crowd[i] = trial_crowd  # update crowding info

    return new_pop, new_fit
