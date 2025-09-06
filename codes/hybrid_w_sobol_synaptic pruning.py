from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np

# Try to import Sobol QMC from SciPy
_HAS_SCIPY_QMC = False
try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except Exception:
    pass


@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0
    S: int = 0


class HybridCSA:
    """
    Hybrid Clonal Selection with:
      1) Rac1-style forgetting (T/S threshold)
      2) Quasi-Opposition Based Learning (QOBL)
      3) Adaptive parameter schedule (IICO-style)
      4) Quasi-Reflection Based Learning (QRBL)
      5) Sobol QMC initialization (scrambled, fallback to uniform if SciPy missing)
      6) Synaptic Pruning (low-novelty filter) to re-seed crowded individuals
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        n_select: int = 15,
        n_clones: int = 5,
        a_frac: float = 0.15,
        c_threshold: float = 3.0,
        max_gens: int = 500,
        seed: Optional[int] = 42,
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        maxStag: int = 3,
        use_qmc: bool = True,
        qmc_engine: str = "sobol",
        prune_scale: float = 0.01,        # <- NEW: novelty radius as fraction of domain norm
        prune_every: int = 5,             # <- NEW: apply pruning every N generations
        max_prunes_per_gen: Optional[int] = None,  # <- NEW: cap prunes per generation
        verbose: bool = True,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)

        self.N = int(N)
        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.a_frac = float(a_frac)
        self.c_threshold = float(c_threshold)
        self.max_gens = int(max_gens)

        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.maxStag = int(maxStag)

        self.use_qmc = bool(use_qmc)
        self.qmc_engine = qmc_engine.lower().strip()
        self.seed = seed
        self.verbose = verbose

        # Synaptic pruning params
        self.prune_scale = float(prune_scale)
        self.prune_every = int(prune_every)
        self.max_prunes_per_gen = max_prunes_per_gen if (max_prunes_per_gen is None) else int(max_prunes_per_gen)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Domain info
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []

    # ---------- utilities ----------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform_point(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ---------- Sobol QMC initializer ----------
    def _qmc_population_init(self) -> np.ndarray:
        if not _HAS_SCIPY_QMC:
            if self.verbose:
                print(
                    "Sobol QMC unavailable. Install SciPy: pip install --upgrade scipy\n"
                    "Proceeding with uniform random initialization so the code runs."
                )
            X = np.stack([self._sample_uniform_point() for _ in range(self.N)], axis=0)
            return X

        if self.qmc_engine != "sobol":
            raise ValueError(f"Unsupported qmc_engine '{self.qmc_engine}'. Use 'sobol'.")

        m = int(math.ceil(math.log2(max(2, self.N))))
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        U = sampler.random_base2(m=m)       # (2**m, dim) in [0,1)
        U = U[: self.N, :]
        X = qmc.scale(U, self.lb, self.ub)
        return X

    def _population_init(self) -> List[Antibody]:
        pop: List[Antibody] = []
        if self.use_qmc:
            X0 = self._qmc_population_init()
            for i in range(self.N):
                x = X0[i]
                pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        else:
            for _ in range(self.N):
                x = self._sample_uniform_point()
                pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        return pop

    def _select_top(self, pop: List[Antibody]) -> List[Antibody]:
        return sorted(pop, key=lambda ab: ab.affinity, reverse=True)[: self.n_select]

    def _elite_center(self, pop: List[Antibody], k: int) -> np.ndarray:
        k = max(1, min(k, len(pop)))
        elites = sorted(pop, key=lambda ab: ab.affinity, reverse=True)[:k]
        affs = np.array([ab.affinity for ab in elites], dtype=float)

        if np.ptp(affs) == 0.0:  # NumPy 2.0 compatible
            weights = np.ones_like(affs)
        else:
            weights = (affs - affs.min()) / (np.ptp(affs) + 1e-12)

        weights = weights / (np.sum(weights) + 1e-12)
        X = np.stack([ab.x for ab in elites], axis=0)
        return np.sum(X * weights[:, None], axis=0)

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
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # ---------- learning operators ----------
    def _qobl(self, x: np.ndarray) -> np.ndarray:
        opposite = self.lb + self.ub - x
        lows = np.minimum(self.mid, opposite)
        highs = np.maximum(self.mid, opposite)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _qrbl(self, x: np.ndarray) -> np.ndarray:
        lows = np.minimum(self.mid, x)
        highs = np.maximum(self.mid, x)
        return lows + np.random.rand(self.dim) * (highs - lows)

    # ---------- forgetting (Rac1 surrogate) ----------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform_point()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ---------- Synaptic Pruning: Low-Novelty Filter ----------
    def _prune_low_novelty(self, pop: List[Antibody], scale: Optional[float] = None,
                           max_prunes: Optional[int] = None) -> None:
        """
        Re-seed individuals that are too similar to the rest (low average distance),
        which increases diversity and helps escape crowding.
        """
        if scale is None:
            scale = self.prune_scale
        if scale <= 0.0 or len(pop) < 3:
            return

        threshold = float(scale) * np.linalg.norm(self.ub - self.lb)
        X = np.stack([ab.x for ab in pop], axis=0)  # (N, dim)

        # Pairwise distances via broadcasting
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (N, N)
        np.fill_diagonal(dists, np.nan)        # ignore self-distance
        avg_dists = np.nanmean(dists, axis=1)  # (N,)

        # Indices that fall below novelty threshold
        to_prune = np.where(avg_dists < threshold)[0].tolist()
        if max_prunes is not None and len(to_prune) > max_prunes:
            # prune the most crowded first (smallest average distance)
            order = np.argsort(avg_dists[to_prune])
            to_prune = [to_prune[i] for i in order[:max_prunes]]

        for i in to_prune:
            x_new = self._sample_uniform_point()
            pop[i] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0)

    # ---------- main loop ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()

        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagnation = 0

        k = max(2, self.max_gens)
        beta = self.beta

        for gen in range(1, self.max_gens + 1):
            for ab in pop:
                ab.T += 1

            selected = self._select_top(pop)

            elites_for_center = max(
                1,
                int(round(self.n_select * (0.5 + 0.5 * max(0, self.maxStag - stagnation) / max(1, self.maxStag))))
            )
            F1 = self._elite_center(pop, elites_for_center)

            Z = math.exp(-beta * gen / k)
            if Z <= self.gamma:
                beta = -math.log(10.0 * self.gamma) * k / max(1, gen)
                Z = math.exp(-beta * gen / k)

            sigma_iter = ((k - gen) / max(1.0, (k - 1))) ** self.exponent * (self.sigma_initial - self.sigma_final) + self.sigma_final
            alpha_iter = 10.0 * math.log(max(1e-9, self.M)) * Z
            if not np.isfinite(alpha_iter):
                alpha_iter = self.sigma_final

            clones = self._clone(selected)

            new_clones: List[Antibody] = []
            if clones:
                max_aff = max(ab.affinity for ab in clones)
                max_aff = max(max_aff, 1e-12)

            for ab in clones:
                # numerically safe gate probability
                arg = -2.0 * (ab.affinity / max_aff)
                if arg > 700.0:
                    p = 1.0
                elif arg < -700.0:
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
                    cand = ab.x + A_vec + noise

                self._clip(cand)
                cand_fit = self._affinity(cand)

                if stagnation <= self.maxStag:
                    qop = self._qobl(cand)
                    self._clip(qop)
                    qop_fit = self._affinity(qop)
                    if qop_fit > cand_fit:
                        cand, cand_fit = qop, qop_fit
                else:
                    qrf = self._qrbl(cand)
                    self._clip(qrf)
                    qrf_fit = self._affinity(qrf)
                    if qrf_fit > cand_fit:
                        cand, cand_fit = qrf, qrf_fit

                new_clones.append(Antibody(x=cand, affinity=cand_fit, T=0, S=1))

            # Replacement + forgetting
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]

            self._forget_in_place(pop)

            # ---- Synaptic pruning every prune_every generations ----
            if self.prune_scale > 0.0 and self.prune_every > 0 and (gen % self.prune_every == 0):
                self._prune_low_novelty(pop, scale=self.prune_scale, max_prunes=self.max_prunes_per_gen)

            # Book-keeping
            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            self.history_best.append((cur_best_val, cur_best.x.copy()))

            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagnation = 0
            else:
                stagnation += 1

        return best.x.copy(), best_val, {
            "generations": self.max_gens,
            "history": self.history_best,
        }


# ---------- Test functions ----------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2.0 * math.pi * x)))


# ---------- Experiment runner (100 runs) ----------
def run_experiments(func, func_name, runs=100, dim=30):
    bounds = [(-5.12, 5.12)] * dim
    results = []

    for run in range(runs):
        algo = HybridCSA(
            func=func,
            bounds=bounds,
            N=60,
            n_select=15,
            n_clones=5,
            a_frac=0.15,
            c_threshold=3.0,
            max_gens=500,
            seed=run,
            sigma_initial=0.5,
            sigma_final=0.1,
            exponent=2.0,
            beta=100.0,
            gamma=1e-19,
            maxStag=3,
            use_qmc=True,
            qmc_engine="sobol",
            prune_scale=0.01,          # adjust if you want more/less aggressive pruning
            prune_every=5,
            max_prunes_per_gen=None,   # e.g., set to 5 to cap re-seeds each pruning step
            verbose=(run == 0),
        )
        _, f_best, _ = algo.minimize()
        results.append(f_best)

    results = np.asarray(results, dtype=float)
    print(f"\nResults for {func_name}")
    print(f"Average Optimal Solution: {results.mean()}")
    print(f"Maximum Optimal Solution: {results.max()}")
    print(f"Minimum Optimal Solution: {results.min()}")


if __name__ == "__main__":
    run_experiments(sphere, "Sphere Function", runs=100, dim=30)
    run_experiments(rastrigin, "Rastrigin Function", runs=100, dim=30)
