from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    affinity: float          # higher is better (we maximize affinity)
    T: int = 0               # survival time
    S: int = 0               # appropriate memory strength


class FCSA:
    """
    FCSA: Improved Clonal Selection with biological forgetting.

    Arguments
      func            objective function to minimize
      bounds          list of (lo, hi) per dimension
      N               population size
      n_select        number selected for cloning each generation
      n_clones        max clones per selected parent
      r               mutation exponent rate from Eq. 3
      a_frac          variation range as a fraction of domain width
      c_threshold     Rac1 activity threshold
      max_gens        fallback termination by generations
      max_evals       primary termination by function evaluation budget
      seed            RNG seed for reproducibility
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        n_select: int = 15,
        n_clones: int = 5,
        r: float = 2.0,
        a_frac: float = 0.15,
        c_threshold: float = 3.0,
        max_gens: int = 500,
        max_evals: int = 350_000,
        seed: int | None = 42,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)

        self.N = int(N)
        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.r = float(r)
        self.a_frac = float(a_frac)
        self.c_threshold = float(c_threshold)
        self.max_gens = int(max_gens)
        self.max_evals = int(max_evals)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths  # per dimension mutation span
        self.history_best: list[tuple[float, np.ndarray]] = []

        self.eval_count = 0

    # ------------- evaluation helpers -------------
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        # maximize opposite of the test function value
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.bounds[:, 0], self.bounds[:, 1], out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.bounds[:, 0] + np.random.rand(self.dim) * self.widths

    # ------------- population init -------------
    def _init_population(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        return pop

    # ------------- select, memory, clone -------------
    def _select_top(self, pop: List[Antibody]) -> List[Antibody]:
        pop_sorted = sorted(pop, key=lambda ab: ab.affinity, reverse=True)
        return pop_sorted[: self.n_select]

    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        # normalize affinities to [0,1] among selected for clone scaling
        sel_affs = np.array([ab.affinity for ab in selected], dtype=float)
        a_min = float(sel_affs.min())
        a_max = float(sel_affs.max())
        denom = max(a_max - a_min, 1e-12)
        clones: List[Antibody] = []
        for ab in selected:
            ab.S += 1
            a_norm = (ab.affinity - a_min) / denom     # 0..1
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # ------------- variation step -------------
    def _mutate_variation(self, clones: List[Antibody]) -> None:
        if not clones:
            return
        # compute normalized goodness in [0,1] across clones
        affs = np.array([ab.affinity for ab in clones], dtype=float)
        a_min = float(affs.min())
        a_max = float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for ab in clones:
            a_norm = (ab.affinity - a_min) / denom     # 0 = worst, 1 = best
            # mutation gate probability from Eq. 3 using bounded input
            p = math.exp(-self.r * a_norm)             # in (e^-r, 1]
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.affinity = self._affinity(ab.x)
                ab.T = 0
                ab.S = 1

    # ------------- elitist replacement -------------
    def _downselect(self, parents: List[Antibody], clones: List[Antibody]) -> List[Antibody]:
        pool = parents + clones
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return pool[: self.N]

    # ------------- forgetting mechanism -------------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                # optional softness: treat never-selected individuals with small T as low activity
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ------------- main loop -------------
    def optimize(self):
        pop = self._init_population()

        for gen in range(self.max_gens):
            if self.eval_count >= self.max_evals:
                break

            for ab in pop:
                ab.T += 1

            selected = self._select_top(pop)
            clones = self._clone(selected)
            self._mutate_variation(clones)
            pop = self._downselect(pop, clones)
            self._forget_in_place(pop)

            best = max(pop, key=lambda ab: ab.affinity)
            # store objective value and position
            self.history_best.append((-best.affinity, best.x.copy()))

            if self.eval_count >= self.max_evals:
                break

        best = max(pop, key=lambda ab: ab.affinity)
        return best.x.copy(), -best.affinity, {
            "generations_run": len(self.history_best),
            "evals_used": self.eval_count,
            "history": self.history_best,
        }
