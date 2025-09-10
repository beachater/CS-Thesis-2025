from __future__ import annotations

"""
HybridCSA++
-----------
A CSA-based metaheuristic that integrates:
  • Hybrid initialization (Sobol QMC + Chaotic Logistic map)
  • Antibody-level Δx memory (momentum-like)
  • Adaptive mutation schedule (annealing-style)
  • Probabilistic QOBL / QRBL exploration moves
  • Rac1-like forgetting rule (T/S activity)
  • Adaptive synaptic pruning (diversity-triggered) with chaotic reseeding
  • Optional local search intensification (Nelder–Mead) on elite(s)

Dependencies: numpy (required), scipy (optional: qmc + local search)
This file is self-contained; fallbacks are provided when SciPy is absent.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np

# ----- Optional SciPy features -----
_HAS_SCIPY_QMC = False
_HAS_SCIPY_OPT = False
try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except Exception:
    pass

try:
    from scipy.optimize import minimize as scipy_minimize
    _HAS_SCIPY_OPT = True
except Exception:
    pass


# --------------------- utilities ---------------------
def logistic_map_sequence(n: int, x0: float = None, mu: float = 4.0) -> np.ndarray:
    """Generate n values in (0,1) using the chaotic logistic map.
    If x0 is None, seed from random.random()."""
    if x0 is None:
        x = random.random()
    else:
        x = float(x0)
    seq = np.empty(n, dtype=float)
    for i in range(n):
        x = mu * x * (1.0 - x)
        # Keep in (0,1): reflect if numerical spill
        if x <= 0.0 or x >= 1.0:
            x = abs(x) % 1.0
            if x == 0.0:
                x = np.nextafter(0.0, 1.0)
        seq[i] = x
    return seq


def chaotic_matrix(rows: int, cols: int, mu: float = 4.0) -> np.ndarray:
    """Create a rows×cols matrix in (0,1) using independent logistic sequences."""
    U = np.empty((rows, cols), dtype=float)
    for d in range(cols):
        U[:, d] = logistic_map_sequence(rows, x0=random.random(), mu=mu)
    return U


def pairwise_avg_distance(X: np.ndarray) -> float:
    """Average pairwise Euclidean distance among rows of X.
    O(N^2) but fine for typical N<=200."""
    if len(X) < 2:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    # upper triangle without diagonal
    iu = np.triu_indices_from(dists, k=1)
    return float(np.mean(dists[iu]))


# --------------------- core data structure ---------------------
@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0  # time active
    S: int = 0  # times selected/cloned
    dx: np.ndarray | None = None  # last move (Δx memory)


# --------------------- algorithm ---------------------
class HybridCSAPlus:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        n_select: int = 15,
        n_clones: int = 5,
        max_gens: int = 1000,
        seed: Optional[int] = 42,
        # mutation / schedule
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        a_frac: float = 0.15,  # uniform step radius as fraction of domain width
        # opposition probabilities
        p_qobl: float = 0.35,
        p_qrbl: float = 0.35,
        # forgetting
        c_threshold: float = 3.0,
        # pruning
        prune_base_scale: float = 0.01,  # novelty radius fraction of domain norm
        prune_every: int = 5,
        diversity_trigger: float = 0.05,  # trigger pruning if normalized diversity < this
        max_prunes_per_gen: Optional[int] = None,
        # init
        use_hybrid_init: bool = True,
        hybrid_split: float = 0.5,  # fraction Sobol (if available) vs Chaos
        chaos_mu: float = 4.0,
        # local search
        local_search_every: int = 50,
        n_local_elites: int = 1,
        local_budget: int = 200,
        verbose: bool = True,
    ):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]

        self.N = int(N)
        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.max_gens = int(max_gens)
        self.seed = seed
        self.verbose = verbose

        # schedules
        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.a_frac = float(a_frac)
        self.p_qobl = float(p_qobl)
        self.p_qrbl = float(p_qrbl)

        self.c_threshold = float(c_threshold)

        self.prune_base_scale = float(prune_base_scale)
        self.prune_every = int(prune_every)
        self.diversity_trigger = float(diversity_trigger)
        self.max_prunes_per_gen = (
            None if max_prunes_per_gen is None else int(max_prunes_per_gen)
        )

        self.use_hybrid_init = bool(use_hybrid_init)
        self.hybrid_split = float(hybrid_split)
        self.chaos_mu = float(chaos_mu)

        self.local_search_every = int(local_search_every)
        self.n_local_elites = int(n_local_elites)
        self.local_budget = int(local_budget)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.domain_norm = float(np.linalg.norm(self.ub - self.lb))
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.best_hist: List[Tuple[float, np.ndarray]] = []

    # --------- helpers ---------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform(self, m: int) -> np.ndarray:
        return self.lb + np.random.rand(m, self.dim) * self.widths

    def _sample_sobol(self, m: int) -> np.ndarray:
        if not _HAS_SCIPY_QMC:
            return self._sample_uniform(m)
        # Use power-of-two length for better Sobol quality
        mm = 1 << int(math.ceil(math.log2(max(2, m))))
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        U = sampler.random_base2(m=int(math.log2(mm)))
        if mm > m:
            U = U[:m]
        return qmc.scale(U, self.lb, self.ub)

    def _sample_chaos(self, m: int) -> np.ndarray:
        U = chaotic_matrix(m, self.dim, mu=self.chaos_mu)
        return self.lb + U * self.widths

    def _qobl(self, x: np.ndarray) -> np.ndarray:
        opp = self.lb + self.ub - x
        lows = np.minimum(self.mid, opp)
        highs = np.maximum(self.mid, opp)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _qrbl(self, x: np.ndarray) -> np.ndarray:
        lows = np.minimum(self.mid, x)
        highs = np.maximum(self.mid, x)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _forget(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x_new = self._sample_uniform(1)[0]
                pop[i] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0, dx=None)

    def _adaptive_sigma(self, gen: int, k: int) -> float:
        return ((k - gen) / max(1.0, (k - 1))) ** self.exponent * (
            self.sigma_initial - self.sigma_final
        ) + self.sigma_final

    def _alpha_iter(self, gen: int, k: int, beta: float) -> float:
        Z = math.exp(-beta * gen / k)
        if Z <= self.gamma:
            beta = -math.log(10.0 * self.gamma) * k / max(1, gen)
            Z = math.exp(-beta * gen / k)
        a = 10.0 * math.log(max(1e-9, self.M)) * Z
        if not np.isfinite(a):
            a = self.sigma_final
        return a

    def _diversity_normalized(self, X: np.ndarray) -> float:
        if self.domain_norm <= 0:
            return 0.0
        return pairwise_avg_distance(X) / (self.domain_norm + 1e-12)

    def _prune(self, pop: List[Antibody], scale: float, max_prunes: Optional[int]) -> None:
        if scale <= 0.0 or len(pop) < 3:
            return
        X = np.stack([ab.x for ab in pop], axis=0)
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.nan)
        avg_d = np.nanmean(dists, axis=1)
        thr = float(scale) * self.domain_norm
        to_prune = np.where(avg_d < thr)[0].tolist()
        if max_prunes is not None and len(to_prune) > max_prunes:
            order = np.argsort(avg_d[to_prune])
            to_prune = [to_prune[i] for i in order[:max_prunes]]
        # Chaotic reseeding for pruned individuals
        m = len(to_prune)
        if m == 0:
            return
        Xc = self._sample_chaos(m)
        for idx, xi in zip(to_prune, Xc):
            pop[idx] = Antibody(x=xi, affinity=self._affinity(xi), T=0, S=0, dx=None)

    def _local_search(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        if not _HAS_SCIPY_OPT:
            return x0, self._objective(x0)
        # Bound-constrained Nelder–Mead via penalty + projection (simple & robust)
        lb, ub = self.lb.copy(), self.ub.copy()

        def proj(z):
            return np.minimum(ub, np.maximum(lb, z))

        def penalized(z):
            zc = proj(z)
            return self._objective(zc)

        res = scipy_minimize(
            penalized,
            x0=x0.copy(),
            method="Nelder-Mead",
            options={"maxfev": self.local_budget, "maxiter": self.local_budget, "xatol": 1e-9, "fatol": 1e-9},
        )
        x_best = proj(res.x)
        return x_best, self._objective(x_best)

    # --------- initialization ---------
    def _init_population(self) -> List[Antibody]:
        if not self.use_hybrid_init:
            X0 = self._sample_uniform(self.N)
        else:
            n_sob = int(round(self.N * self.hybrid_split))
            n_cha = self.N - n_sob
            XS = self._sample_sobol(n_sob) if n_sob > 0 else np.empty((0, self.dim))
            XC = self._sample_chaos(n_cha) if n_cha > 0 else np.empty((0, self.dim))
            X0 = np.vstack([XS, XC]) if (len(XS) or len(XC)) else self._sample_uniform(self.N)
        pop: List[Antibody] = []
        for i in range(self.N):
            x = X0[i]
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0, dx=None))
        return pop

    # --------- selection / cloning ---------
    def _select_top(self, pop: List[Antibody]) -> List[Antibody]:
        return sorted(pop, key=lambda ab: ab.affinity, reverse=True)[: self.n_select]

    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        if np.ptp(affs) == 0.0:
            norm = np.ones_like(affs)
        else:
            norm = (affs - affs.min()) / (np.ptp(affs) + 1e-12)
        clones: List[Antibody] = []
        for ab, a in zip(selected, norm):
            ab.S += 1
            k = max(1, int(round(1 + a * (self.n_clones - 1))))
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S, dx=(ab.dx.copy() if ab.dx is not None else None)))
        return clones

    # --------- main minimize ---------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_population()
        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        k = max(2, self.max_gens)
        beta = self.beta
        stagnation = 0

        for gen in range(1, self.max_gens + 1):
            # time passes
            for ab in pop:
                ab.T += 1

            # selection & elite center (for directional move)
            selected = self._select_top(pop)
            elites_for_center = max(1, int(round(self.n_select * (0.5 + 0.5 * max(0, 3 - stagnation) / 3.0))))
            elites = sorted(pop, key=lambda a: a.affinity, reverse=True)[:elites_for_center]
            W = np.array([a.affinity for a in elites], dtype=float)
            if np.ptp(W) == 0.0:
                weights = np.ones_like(W) / max(1, len(W))
            else:
                weights = (W - W.min()) / (np.ptp(W) + 1e-12)
                weights = weights / (weights.sum() + 1e-12)
            X_el = np.stack([a.x for a in elites], axis=0)
            F1 = (X_el * weights[:, None]).sum(axis=0)

            # schedules
            sigma_iter = self._adaptive_sigma(gen, k)
            alpha_iter = self._alpha_iter(gen, k, beta)

            # cloning + variation
            clones = self._clone(selected)
            new_clones: List[Antibody] = []
            if clones:
                max_aff = max(ab.affinity for ab in clones)
                max_aff = max(max_aff, 1e-12)

            for ab in clones:
                # gating between uniform-like step vs directed + noise
                arg = -2.0 * (ab.affinity / max_aff)
                if arg > 700:
                    p = 1.0
                elif arg < -700:
                    p = 0.0
                else:
                    p = math.exp(arg)
                p = float(min(1.0, max(0.0, p)))
                use_uniform = (np.random.rand() < p)

                if use_uniform:
                    step = np.random.uniform(-self.a_vec, self.a_vec)
                    cand = ab.x + step
                else:
                    diff = F1 - ab.x
                    norm = np.linalg.norm(diff) + 1e-12
                    A_vec = 20.0 * alpha_iter * (diff / norm)
                    noise = np.random.randn(self.dim) * alpha_iter
                    # Δx memory contribution
                    if ab.dx is not None:
                        noise = noise + 0.5 * ab.dx
                    cand = ab.x + A_vec + noise

                # probabilistic opposition moves
                if np.random.rand() < self.p_qobl:
                    cand = self._qobl(cand)
                if np.random.rand() < self.p_qrbl:
                    cand = self._qrbl(cand)

                self._clip(cand)
                f_new = self._affinity(cand)
                dx = cand - ab.x
                new_clones.append(Antibody(x=cand, affinity=f_new, T=0, S=1, dx=dx))

            # replacement: keep best N from pool
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]

            # forgetting
            self._forget(pop)

            # adaptive pruning: diversity check + periodic
            Xcur = np.stack([ab.x for ab in pop], axis=0)
            div_norm = self._diversity_normalized(Xcur)
            apply_prune = (gen % self.prune_every == 0) or (div_norm < self.diversity_trigger)
            if apply_prune:
                # scale pruning stronger when diversity is very low
                scale = self.prune_base_scale * (1.0 + max(0.0, (self.diversity_trigger - div_norm)) / max(1e-9, self.diversity_trigger))
                self._prune(pop, scale=scale, max_prunes=self.max_prunes_per_gen)

            # optional local search on top elites
            if self.local_search_every > 0 and gen % self.local_search_every == 0 and self.n_local_elites > 0:
                pop.sort(key=lambda a: a.affinity, reverse=True)
                topk = min(self.n_local_elites, len(pop))
                for i in range(topk):
                    xi, fi = pop[i].x.copy(), -pop[i].affinity
                    xl, fl = self._local_search(xi)
                    afl = -fl
                    if afl > pop[i].affinity:
                        dx = xl - pop[i].x
                        pop[i] = Antibody(x=xl, affinity=afl, T=0, S=pop[i].S, dx=dx)

            # update global best & bookkeeping
            cur_best = max(pop, key=lambda a: a.affinity)
            cur_val = -cur_best.affinity
            self.best_hist.append((cur_val, cur_best.x.copy()))
            if cur_val + 1e-12 < best_val:
                best_val = cur_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S, dx=cur_best.dx.copy() if cur_best.dx is not None else None)
                stagnation = 0
            else:
                stagnation += 1

        return best.x.copy(), best_val, {"generations": self.max_gens, "history": self.best_hist}


# --------------------- Example benchmark functions ---------------------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2.0 * math.pi * x)))


def ackley(x: np.ndarray) -> float:
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    d = x.size
    s1 = np.sum(x * x)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    return float(term1 + term2 + a + math.e)


# --------------------- Experiment runner ---------------------
def run_experiments(func, name: str, runs: int = 10, dim: int = 30) -> Dict[str, Any]:
    bounds = [(-5.12, 5.12)] * dim
    results = []
    for r in range(runs):
        algo = HybridCSAPlus(
            func=func,
            bounds=bounds,
            N=60,
            n_select=15,
            n_clones=5,
            max_gens=1000,
            seed=r,
            sigma_initial=0.5,
            sigma_final=0.1,
            exponent=2.0,
            beta=100.0,
            gamma=1e-19,
            a_frac=0.15,
            p_qobl=0.35,
            p_qrbl=0.35,
            c_threshold=3.0,
            prune_base_scale=0.01,
            prune_every=5,
            diversity_trigger=0.05,
            max_prunes_per_gen=None,
            use_hybrid_init=True,
            hybrid_split=0.5,
            chaos_mu=4.0,
            local_search_every=50,
            n_local_elites=1,
            local_budget=200,
            verbose=(r == 0),
        )
        _, fbest, _ = algo.minimize()
        results.append(fbest)
    arr = np.asarray(results, dtype=float)
    summary = {
        "name": name,
        "runs": runs,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print(f"\n{name}: mean={summary['mean']:.6g}, std={summary['std']:.6g}, min={summary['min']:.6g}, max={summary['max']:.6g}")
    return summary


if __name__ == "__main__":
    # Small smoke test (reduce runs for speed)
    run_experiments(sphere, "Sphere", runs=5, dim=30)
    run_experiments(rastrigin, "Rastrigin", runs=5, dim=30)
    run_experiments(ackley, "Ackley", runs=5, dim=30)
