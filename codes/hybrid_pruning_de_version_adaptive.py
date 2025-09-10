from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0
    S: int = 0


class HybridCSA_pruning_de_version_adaptive:
    """
    Hybrid Clonal Selection with:
      1) Rac1-style forgetting (T/S threshold)
      2) Quasi-Opposition Based Learning (QOBL)
      3) Adaptive parameter schedule (IICO-style)
      4) Quasi-Reflection Based Learning (QRBL)
      5) Synaptic Pruning (low-novelty filter) to re-seed crowded individuals
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
        max_gens: int = 1000,
        seed: Optional[int] = 42,
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        maxStag: int = 3,
        prune_scale: float = 0.01,
        prune_every: int = 5,
        max_prunes_per_gen: Optional[int] = None,
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

        self.seed = seed
        self.verbose = verbose

        # Synaptic pruning params
        self.prune_scale = float(prune_scale)
        self.prune_every = int(prune_every)
        self.max_prunes_per_gen = (
            max_prunes_per_gen if (max_prunes_per_gen is None) else int(max_prunes_per_gen)
        )

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

        # Adaptive Mutation Strategy Pool
        self.strategies = ["rand1", "current_to_pbest", "current_rand"]
        self.strategy_success = {s: 1 for s in self.strategies}  # success counts

    # ---------- utilities ----------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform_point(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

        # ---------- Differential Evolution Strategies ----------
    def _de_rand1(self, pop: List[Antibody]) -> np.ndarray:
        idxs = np.random.choice(len(pop), 3, replace=False)
        x1, x2, x3 = [pop[i].x for i in idxs]
        F = np.random.uniform(0.4, 0.9)
        return x1 + F * (x2 - x3)

    def _de_current_to_pbest(self, ab: Antibody, pop: List[Antibody], p=0.2) -> np.ndarray:
        p_num = max(2, int(p * len(pop)))
        pbest = sorted(pop, key=lambda ab: ab.affinity, reverse=True)[:p_num]
        xpbest = random.choice(pbest).x
        x1, x2 = [pop[i].x for i in np.random.choice(len(pop), 2, replace=False)]
        F = np.random.uniform(0.4, 0.9)
        return ab.x + F * (xpbest - ab.x) + F * (x1 - x2)

    def _de_current_rand(self, ab: Antibody, pop: List[Antibody]) -> np.ndarray:
        x1, x2 = [pop[i].x for i in np.random.choice(len(pop), 2, replace=False)]
        F = np.random.uniform(0.4, 0.9)
        return ab.x + F * (x1 - x2)
    
    def _choose_strategy(self) -> str:
        total = sum(self.strategy_success.values())
        probs = [self.strategy_success[s] / total for s in self.strategies]
        return np.random.choice(self.strategies, p=probs)


    # ---------- population init ----------
    def _population_init(self) -> List[Antibody]:
        pop: List[Antibody] = []
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

        if np.ptp(affs) == 0.0:
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

    # ---------- Synaptic Pruning ----------
    def _adaptive_prune(
        self,
        pop: List[Antibody],
        gen: int,
        stagnation: int,
    ) -> None:
        if len(pop) < 3:
            return

        # Base distance threshold from bounds
        base_threshold = self.prune_scale * np.linalg.norm(self.ub - self.lb)

        # Diversity measure (avg pairwise distance)
        X = np.stack([ab.x for ab in pop], axis=0)
        diffs = X[:, None, :] - X[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.nan)
        avg_dists = np.nanmean(dists, axis=1)
        pop_diversity = np.nanmean(avg_dists)

        # Adaptive threshold scaling:
        # - Increase pruning if diversity is too low or stagnation is high
        # - Decrease pruning if diversity is healthy
        if pop_diversity < 0.05 * np.linalg.norm(self.ub - self.lb):
            threshold = base_threshold * (1.5 + stagnation * 0.5)
        else:
            threshold = base_threshold * max(0.5, 1.0 - 0.1 * stagnation)

        # Identify individuals to prune
        to_prune = np.where(avg_dists < threshold)[0].tolist()
        if self.max_prunes_per_gen is not None and len(to_prune) > self.max_prunes_per_gen:
            order = np.argsort(avg_dists[to_prune])  # prune the most crowded
            to_prune = [to_prune[i] for i in order[: self.max_prunes_per_gen]]

        # Replace selected individuals
        for i in to_prune:
            x_new = self._sample_uniform_point()
            pop[i] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0)

    # ---------- main loop ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history = []  # best fitness per generation


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
                int(
                    round(
                        self.n_select
                        * (0.5 + 0.5 * max(0, self.maxStag - stagnation) / max(1, self.maxStag))
                    )
                ),
            )
            F1 = self._elite_center(pop, elites_for_center)

            Z = math.exp(-beta * gen / k)
            if Z <= self.gamma:
                beta = -math.log(10.0 * self.gamma) * k / max(1, gen)
                Z = math.exp(-beta * gen / k)

            sigma_iter = ((k - gen) / max(1.0, (k - 1))) ** self.exponent * (
                self.sigma_initial - self.sigma_final
            ) + self.sigma_final
            alpha_iter = 10.0 * math.log(max(1e-9, self.M)) * Z
            if not np.isfinite(alpha_iter):
                alpha_iter = self.sigma_final

            clones = self._clone(selected)

            new_clones: List[Antibody] = []
            if clones:
                max_aff = max(ab.affinity for ab in clones)
                max_aff = max(max_aff, 1e-12)

            for ab in clones:
                # ---------- Adaptive Mutation Strategy ----------
                strategy = self._choose_strategy()
                if strategy == "rand1":
                    cand = self._de_rand1(pop)
                elif strategy == "current_to_pbest":
                    cand = self._de_current_to_pbest(ab, pop)
                else:
                    cand = self._de_current_rand(ab, pop)

                self._clip(cand)
                cand_fit = self._affinity(cand)

                # Update strategy success if better
                if cand_fit > ab.affinity:
                    self.strategy_success[strategy] += 1

                # ---------- QOBL / QRBL ----------
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

            # ---------- Pool update ----------
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]

            self._forget_in_place(pop)

            # ---------- Synaptic Pruning ----------
            if self.prune_scale > 0.0 and self.prune_every > 0 and (gen % self.prune_every == 0):
                self._adaptive_prune(pop, gen, stagnation)

            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            self.history_best.append((cur_best_val, cur_best.x.copy()))
            history.append(cur_best_val)
            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagnation = 0
            else:
                stagnation += 1

        return best.x.copy(), best_val, {
            "generations": self.max_gens,
            # "history": self.history_best,
            "history": history,  # just fitness per generation

        }



# ---------- Test functions ----------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2.0 * math.pi * x)))


# ---------- Experiment runner ----------
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
            max_gens=1000,
            seed=run,
            sigma_initial=0.5,
            sigma_final=0.1,
            exponent=2.0,
            beta=100.0,
            gamma=1e-19,
            maxStag=3,
            prune_scale=0.01,
            prune_every=5,
            max_prunes_per_gen=None,
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
