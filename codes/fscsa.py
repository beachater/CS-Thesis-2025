from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Antibody:
    x: np.ndarray                # position
    affinity: float              # higher is better (we use -objective)
    T: int = 0                   # survival time
    S: int = 0                   # appropriate memory strength


# -----------------------------
# FCSA implementation
# -----------------------------
class FCSA:
    """
    FCSA: Improved Clonal Selection with biological forgetting.

    Parameters closely follow the paper:
      N .............. population size
      n_select ....... number selected for cloning each generation
      n_clones ....... maximum clones per selected parent
      r .............. mutation rate used inside exp in Eq.(3)
      a_frac ......... variation range as a fraction of domain width
      c_threshold .... Rac1 activity threshold in Eq.(4)
      max_gens ....... termination condition (generations)
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
        seed: int | None = 42,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)

        self.N = N
        self.n_select = max(1, min(n_select, N))
        self.n_clones = max(1, n_clones)
        self.r = float(r)
        self.a_frac = float(a_frac)
        self.c_threshold = float(c_threshold)
        self.max_gens = max_gens

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths  # variation range per dim
        self.history_best = []

    # --------- helpers ---------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        # Maximize opposite of the test function
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.bounds[:, 0], self.bounds[:, 1], out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.bounds[:, 0] + np.random.rand(self.dim) * self.widths

    # --------- Algorithm 1: steps (2), (4) ---------
    def _init_population(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        return pop

    # --------- Steps (5)-(8): select, update memory, clone ---------
    def _select_top(self, pop: List[Antibody]) -> List[Antibody]:
        pop_sorted = sorted(pop, key=lambda ab: ab.affinity, reverse=True)
        return pop_sorted[: self.n_select]

    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []

        # normalize affinities among selected to [0,1]
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        if affs.ptp() == 0:
            norm = np.ones_like(affs)
        else:
            norm = (affs - affs.min()) / affs.ptp()

        clones: List[Antibody] = []
        for ab, a in zip(selected, norm):
            ab.S += 1                       # step (7): memory += 1
            k = max(1, int(round(1 + a * (self.n_clones - 1))))  # proportional to affinity
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # --------- Steps (9)-(14): variation (Eq.3) and reset flags ---------
    def _mutate_variation(self, clones: List[Antibody]) -> None:
        if not clones:
            return
        max_aff = max(ab.affinity for ab in clones)
        max_aff = max(max_aff, 1e-12)

        for ab in clones:
            # per-dimension Bernoulli gate with probability exp(-r * aff / max_aff)
            p = math.exp(-self.r * (ab.affinity / max_aff))
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                # uniform step in [-a, a] on the masked dimensions
                step = np.random.uniform(-self.a_vec, self.a_vec)
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.affinity = self._affinity(ab.x)
                # step (11)-(13): if variant, reset T and set S to 1
                ab.T = 0
                ab.S = 1

    # --------- Step (15): elitist replacement from parents+clones ---------
    def _downselect(self, parents: List[Antibody], clones: List[Antibody]) -> List[Antibody]:
        pool = parents + clones
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return [pool[i] for i in range(self.N)]

    # --------- Steps (16)-(19): forgetting (Eq.4) ---------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            # Rac1 activity proxy: T / S
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)

            if activity > self.c_threshold:
                # forget this antibody → reinitialize uniformly in domain
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # --------- Main loop ---------
    def minimize(self) -> Tuple[np.ndarray, float, dict]:
        # Step (2)
        pop = self._init_population()

        # Loop Step (3)
        for gen in range(self.max_gens):
            # Step (4): record survival time
            for ab in pop:
                ab.T += 1

            # Step (5): select top n
            selected = self._select_top(pop)

            # Steps (6)-(8): memory update and cloning
            clones = self._clone(selected)

            # Steps (9)-(14): variation and resets
            self._mutate_variation(clones)

            # Step (15): choose best N from parents + clones
            pop = self._downselect(pop, clones)

            # Steps (16)-(19): forgetting
            self._forget_in_place(pop)

            # Track best for convergence curve
            best = max(pop, key=lambda ab: ab.affinity)
            self.history_best.append((-best.affinity, best.x.copy()))

        best = max(pop, key=lambda ab: ab.affinity)
        return best.x.copy(), -best.affinity, {
            "generations": self.max_gens,
            "history": self.history_best,
        }
