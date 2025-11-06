# =========================
# Canonical CSA
# =========================
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np


@dataclass
class _CSAAb:
    x: np.ndarray
    f: float           # objective value to minimize
    T: int = 0         # age counter (not essential but useful for debugging)


class CSA:
    """
    Canonical Clonal Selection Algorithm for minimization with progress callback parity.

    Objective is minimized. Bounds are respected by clipping.
    Progress callback will receive, once per generation:
        progress(gen, pop, fitness, best_fitness, gbest, evals)
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,             # population size
        n_select: int = 15,      # number of parents selected for cloning
        n_clones: int = 5,       # max clones per selected parent
        r: float = 2.0,          # mutation exponent rate
        a_frac: float = 0.15,    # mutation span as fraction of domain width
        max_gens: int = 1000,
        max_evals: int = 350_000,
        seed: Optional[int] = None,
        # progress: Optional[Callable[..., None]] = None,
    ):
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]

        self.N = int(N)
        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.r = float(r)
        self.a_frac = float(a_frac)
        self.max_gens = int(max_gens)
        self.max_evals = int(max_evals)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths
        self.eval_count = 0
        self.history_best: List[Tuple[float, np.ndarray]] = []
        # self._progress = progress

    # ------------- helpers -------------
    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.bounds[:, 0], self.bounds[:, 1], out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.bounds[:, 0] + np.random.rand(self.dim) * self.widths

    def _eval(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    # ------------- init -------------
    def _init_pop(self) -> List[_CSAAb]:
        pop: List[_CSAAb] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            f = self._eval(x)
            pop.append(_CSAAb(x=x, f=f, T=0))
        return pop

    # ------------- select, clone, mutate -------------
    def _select_top(self, pop: List[_CSAAb]) -> List[_CSAAb]:
        pop_sorted = sorted(pop, key=lambda ab: ab.f)  # lower is better
        return pop_sorted[: self.n_select]

    def _clone(self, selected: List[_CSAAb]) -> List[_CSAAb]:
        if not selected:
            return []
        # lower f is better, convert to affinity scale for clone count
        fs = np.array([ab.f for ab in selected], dtype=float)
        f_max = float(fs.max())
        f_min = float(fs.min())
        span = max(f_max - f_min, 1e-12)
        clones: List[_CSAAb] = []
        for ab in selected:
            # normalize so best gets most clones
            a_norm = (f_max - ab.f) / span  # 0..1 where 1 is best
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                clones.append(_CSAAb(x=ab.x.copy(), f=ab.f, T=ab.T))
        return clones

    def _mutate(self, clones: List[_CSAAb]) -> None:
        if not clones:
            return
        # determine mutation probability from normalized quality among clones
        fs = np.array([ab.f for ab in clones], dtype=float)
        f_max = float(fs.max())
        f_min = float(fs.min())
        span = max(f_max - f_min, 1e-12)
        for ab in clones:
            a_norm = (f_max - ab.f) / span  # 0..1 higher is better
            p = math.exp(-self.r * a_norm)  # worse solutions mutate more
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)  # per dim
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.f = self._eval(ab.x)
                ab.T = 0

    # ------------- replacement -------------
    def _downselect(self, parents: List[_CSAAb], clones: List[_CSAAb]) -> List[_CSAAb]:
        pool = parents + clones
        pool.sort(key=lambda ab: ab.f)
        return pool[: self.N]

    # ------------- main optimize -------------
    def optimize(self, progress: Optional[Callable[..., None]] = None):
        if progress is not None:
            self._progress = progress

        pop = self._init_pop()
        history: List[float] = []

        # gen 0 snapshot
        # if self._progress is not None:
        #     try:
        #         pop_arr = np.stack([ab.x for ab in pop]).astype(float) if pop else np.empty((0, self.dim))
        #         fit_arr = np.array([ab.f for ab in pop], dtype=float)
        #         best0 = min(pop, key=lambda ab: ab.f)
        #         self._progress(
        #             gen=0,
        #             pop=pop_arr,
        #             fitness=fit_arr,
        #             best_fitness=best0.f,
        #             gbest=best0.x.copy(),
        #             evals=self.eval_count,
        #         )
        #     except Exception:
        #         pass

        for gen in range(1, self.max_gens + 1):
            if self.eval_count >= self.max_evals:
                break

            # age
            for ab in pop:
                ab.T += 1

            selected = self._select_top(pop)
            clones = self._clone(selected)
            self._mutate(clones)
            pop = self._downselect(pop, clones)

            best = min(pop, key=lambda ab: ab.f)
            self.history_best.append((best.f, best.x.copy()))
            history.append(best.f)

            # if self._progress is not None:
            #     try:
            #         pop_arr = np.stack([ab.x for ab in pop]) if len(pop) > 0 else np.empty((0, self.dim))
            #         fit_arr = np.array([ab.f for ab in pop], dtype=float)
            #         self._progress(
            #             gen=gen,
            #             pop=pop_arr,
            #             fitness=fit_arr,
            #             best_fitness=best.f,
            #             gbest=best.x.copy(),
            #             evals=self.eval_count,
            #         )
            #     except Exception:
            #         pass

            if self.eval_count >= self.max_evals:
                break

        best = min(pop, key=lambda ab: ab.f)
        return best.x.copy(), best.f, {
            "generations_run": len(history),
            "evals_used": self.eval_count,
            "history": history,               # best fitness per generation
            "history_best": self.history_best # (best_f, best_x) tuples
        }