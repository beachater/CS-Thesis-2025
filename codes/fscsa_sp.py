from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


# Keep the same Antibody as FCSA
@dataclass
class Antibody:
    x: np.ndarray
    affinity: float          # higher is better (we maximize affinity = -f)
    T: int = 0               # survival time
    S: int = 0               # memory strength (times selected)


class FCSASP:
    """
    FCSASP: FCSA + Synaptic Pruning (low-novelty replacement).

    API mirrors the latest FCSA. Only differences:
      - extra pruning knobs: prune_every, prune_scale, protect_k, prune_knn
    If prune_every=None, behavior is identical to FCSA.
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
        # --- pruning extras (optional) ---
        prune_every: int | None = 50,   # set to None to disable
        prune_scale: float = 0.01,      # novelty threshold as frac of domain norm
        protect_k: int = 3,             # keep top-k from pruning
        prune_knn: int = 5,             # k for k-NN novelty
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

        # pruning
        self.prune_every = prune_every
        self.prune_scale = float(prune_scale)
        self.protect_k = int(protect_k)
        self.prune_knn = int(prune_knn)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths  # per-dimension mutation span
        self.history_best: list[tuple[float, np.ndarray]] = []
        self.eval_count = 0

    # ------------- evaluation helpers (same as FCSA) -------------
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity_of(self, x: np.ndarray) -> float:
        # maximize opposite of the objective
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.bounds[:, 0], self.bounds[:, 1], out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.bounds[:, 0] + np.random.rand(self.dim) * self.widths

    # ------------- population init (same) -------------
    def _init_population(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity_of(x), T=0, S=0))
        return pop

    # ------------- select, memory, clone (same) -------------
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

    # ------------- variation (same) -------------
    def _mutate_variation(self, clones: List[Antibody]) -> None:
        if not clones:
            return
        # normalized goodness across clones
        affs = np.array([ab.affinity for ab in clones], dtype=float)
        a_min = float(affs.min())
        a_max = float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for ab in clones:
            a_norm = (ab.affinity - a_min) / denom     # 0 (worst) .. 1 (best)
            # Eq.(3)-style probability with bounded input
            p = math.exp(-self.r * a_norm)             # in (e^-r, 1]
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.affinity = self._affinity_of(ab.x)
                # paper behavior on successful variation
                ab.T = 0
                ab.S = 1

    # ------------- elitist replacement (same) -------------
    def _downselect(self, parents: List[Antibody], clones: List[Antibody]) -> List[Antibody]:
        pool = parents + clones
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return pool[: self.N]

    # ------------- forgetting (same softness) -------------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                # gentle on brand-new ones
                activity = float("inf") if ab.T > 2 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity_of(x), T=0, S=0)

    # ------------- synaptic pruning (only extra step) -------------
    def _prune_low_novelty(self, pop: List[Antibody]) -> None:
        # Protect top-k by affinity
        pop.sort(key=lambda ab: ab.affinity, reverse=True)
        elites = pop[: self.protect_k]
        rest = pop[self.protect_k:]
        if not rest:
            return

        positions = np.array([ab.x for ab in rest])
        dom_norm = np.linalg.norm(self.widths)
        threshold = self.prune_scale * dom_norm

        def knn_avg(i: int, k: int) -> float:
            d = np.linalg.norm(positions - positions[i], axis=1)
            d.sort()
            k = min(k + 1, len(d))  # include self at d[0]=0
            return d[1:k].mean() if k > 1 else 0.0

        for i, ab in enumerate(rest):
            if knn_avg(i, self.prune_knn) < threshold:
                x = self._sample_uniform()
                rest[i] = Antibody(x=x, affinity=self._affinity_of(x), T=0, S=0)

        # reassemble
        pop[: self.protect_k] = elites
        pop[self.protect_k:] = rest

    # ------------- main (same + optional pruning) -------------
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

            # Only extra vs FCSA:
            if self.prune_every and (gen + 1) % self.prune_every == 0:
                self._prune_low_novelty(pop)

            best = max(pop, key=lambda ab: ab.affinity)
            self.history_best.append((-best.affinity, best.x.copy()))

            if self.eval_count >= self.max_evals:
                break

        best = max(pop, key=lambda ab: ab.affinity)
        return best.x.copy(), -best.affinity, {
            "generations_run": len(self.history_best),
            "evals_used": self.eval_count,
            "history": self.history_best,
        }
