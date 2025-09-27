from dataclasses import dataclass

@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0
    S: int = 0
# hybrid_original_tfo.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np

class HybridCSAOperatorsTFO:
    def __init__(self,
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
        max_stagnation: int = 3,
        verbose: bool = True,
        tfo_fold_prob: float = 0.25,
        tfo_fold_strength: float = 0.6,
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
        self.max_stagnation = int(max_stagnation)
        self.seed = seed
        self.verbose = verbose
        self.operator = 'tfo'
        self.tfo_fold_prob = float(tfo_fold_prob)
        self.tfo_fold_strength = float(tfo_fold_strength)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)
        self.history_best: List[Tuple[float, np.ndarray]] = []
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))
    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)
    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)
    def _sample_uniform_point(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths
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
    def _qobl(self, x: np.ndarray) -> np.ndarray:
        opposite = self.lb + self.ub - x
        lows = np.minimum(self.mid, opposite)
        highs = np.maximum(self.mid, opposite)
        return lows + np.random.rand(self.dim) * (highs - lows)
    def _qrbl(self, x: np.ndarray) -> np.ndarray:
        lows = np.minimum(self.mid, x)
        highs = np.maximum(self.mid, x)
        return lows + np.random.rand(self.dim) * (highs - lows)
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform_point()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)
    def _tfo_mutation(self, base: np.ndarray, pop: List[Antibody]) -> np.ndarray:
        x = base.copy()
        D = self.dim
        for d in range(D):
            if np.random.rand() < self.tfo_fold_prob:
                j = np.random.randint(0, D)
                if j == d:
                    continue
                md = self.mid[d]
                mj = self.mid[j]
                rd = x[d] - md
                rj = x[j] - mj
                lam = self.tfo_fold_strength
                new_d = md + lam * rd + (1 - lam) * rj
                new_j = mj + lam * rj + (1 - lam) * rd
                x[d] = new_d
                x[j] = new_j
        x += np.random.normal(scale=0.01 * np.mean(self.widths), size=D)
        return x
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history = []
        self.history_best = []
        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagnation = 0
        k = max(2, self.max_gens)
        beta = self.beta
        for gen in range(1, self.max_gens + 1):
            for ab in pop:
                ab.T += 1
            selected = self._select_top(pop)
            elites_for_center = max(1, int(round(self.n_select * (0.5 + 0.5 * max(0, self.max_stagnation - stagnation) / max(1, self.max_stagnation)))) )
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
                cand = self._tfo_mutation(ab.x, pop)
                self._clip(cand)
                cand_fit = self._affinity(cand)
                if stagnation <= self.max_stagnation:
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
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]
            self._forget_in_place(pop)
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
            "history": history,
            "history_best": self.history_best
        }
