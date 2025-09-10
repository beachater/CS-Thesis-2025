from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    affinity: float          # higher is better because we maximize affinity = -objective
    T: int = 0               # survival time
    S: int = 0               # memory strength


class FCSA:
    """
    FCSA with biological forgetting and Levy Flight Guided Replacement.

    Arguments
      func              objective function to minimize, signature f(x: np.ndarray) -> float
      bounds            list of (lo, hi) per dimension
      N                 population size
      n_select          number selected for cloning each generation
      n_clones          max clones per selected parent
      r                 mutation exponent rate
      a_frac            variation range as a fraction of domain width
      c_threshold       forgetting threshold based on activity T / S
      max_gens          generation cap
      max_evals         evaluation budget cap
      seed              RNG seed

    Levy parameters
      p_apply_lfgr      probability to attempt a Levy guided move per antibody
      alpha_levy        Levy step size scale in domain units
      beta_levy         Levy index commonly 1.5
      guide_mode        "self_to_best" or "best_around"
      accept_rule       "improve" or "metropolis"
      metropolis_T      temperature for Metropolis rule
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
        max_gens: int = 1000,
        max_evals: int = 350_000,
        seed: int | None = 42,
        p_apply_lfgr: float = 0.4,
        alpha_levy: float = 0.03,
        beta_levy: float = 1.5,
        guide_mode: str = "self_to_best",
        accept_rule: str = "improve",
        metropolis_T: float = 1e-3,
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

        self.p_apply_lfgr = float(p_apply_lfgr)
        self.alpha_levy = float(alpha_levy)
        self.beta_levy = float(beta_levy)
        self.guide_mode = str(guide_mode)
        self.accept_rule = str(accept_rule)
        self.metropolis_T = float(metropolis_T)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths
        self.history_best: list[tuple[float, np.ndarray]] = []
        self.eval_count = 0

    # evaluation helpers
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.bounds[:, 0], self.bounds[:, 1], out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.bounds[:, 0] + np.random.rand(self.dim) * self.widths

    # Levy utilities
    @staticmethod
    def _levy_flight(beta: float, size: int) -> np.ndarray:
        # Mantegna algorithm
        sigma_u = (
            (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1.0 / beta)
        u = np.random.normal(0.0, sigma_u, size)
        v = np.random.normal(0.0, 1.0, size)
        return u / (np.abs(v) ** (1.0 / beta))

    # population init
    def _init_population(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        return pop

    # select and clone
    def _select_top(self, pop: List[Antibody]) -> List[Antibody]:
        pop_sorted = sorted(pop, key=lambda ab: ab.affinity, reverse=True)
        return pop_sorted[: self.n_select]

    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        sel_affs = np.array([ab.affinity for ab in selected], dtype=float)
        a_min = float(sel_affs.min())
        a_max = float(sel_affs.max())
        denom = max(a_max - a_min, 1e-12)
        clones: List[Antibody] = []
        for ab in selected:
            ab.S += 1
            a_norm = (ab.affinity - a_min) / denom
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # variation
    def _mutate_variation(self, clones: List[Antibody]) -> None:
        if not clones:
            return
        affs = np.array([ab.affinity for ab in clones], dtype=float)
        a_min = float(affs.min())
        a_max = float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for ab in clones:
            a_norm = (ab.affinity - a_min) / denom
            p = math.exp(-self.r * a_norm)
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.affinity = self._affinity(ab.x)
                ab.T = 0
                ab.S = 1

    # elitist replacement
    def _downselect(self, parents: List[Antibody], clones: List[Antibody]) -> List[Antibody]:
        pool = parents + clones
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return pool[: self.N]

    # forgetting
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # levy flight guided replacement
    def _levy_guided_replacement(self, pop: List[Antibody]) -> None:
        if self.p_apply_lfgr <= 0.0 or self.alpha_levy <= 0.0:
            return

        best_ab = max(pop, key=lambda ab: ab.affinity)
        x_best = best_ab.x

        for i, ab in enumerate(pop):
            if np.random.rand() >= self.p_apply_lfgr:
                continue

            step = self._levy_flight(self.beta_levy, self.dim)

            if self.guide_mode == "self_to_best":
                proposal = ab.x + self.alpha_levy * step * (ab.x - x_best)
            elif self.guide_mode == "best_around":
                proposal = x_best + self.alpha_levy * step * self.widths + 0.25 * (ab.x - x_best)
            else:
                proposal = ab.x + self.alpha_levy * step * (ab.x - x_best)

            self._clip(proposal)
            new_aff = self._affinity(proposal)

            accept = False
            if self.accept_rule == "improve":
                accept = new_aff > ab.affinity
            elif self.accept_rule == "metropolis":
                if new_aff > ab.affinity:
                    accept = True
                else:
                    delta = new_aff - ab.affinity
                    accept = np.random.rand() < math.exp(delta / max(self.metropolis_T, 1e-12))
            else:
                accept = new_aff > ab.affinity

            if accept:
                pop[i] = Antibody(x=proposal, affinity=new_aff, T=0, S=max(1, ab.S))

    # main loop
    def optimize(self):
        pop = self._init_population()
        history = []  # best fitness per generation
        for _ in range(self.max_gens):
            if self.eval_count >= self.max_evals:
                break

            for ab in pop:
                ab.T += 1

            selected = self._select_top(pop)
            clones = self._clone(selected)
            self._mutate_variation(clones)
            pop = self._downselect(pop, clones)
            self._forget_in_place(pop)
            self._levy_guided_replacement(pop)

            best = max(pop, key=lambda ab: ab.affinity)
            self.history_best.append((-best.affinity, best.x.copy()))
            history.append(-best.affinity)
            if self.eval_count >= self.max_evals:
                break

        best = max(pop, key=lambda ab: ab.affinity)
        return best.x.copy(), -best.affinity, {
            "generations_run": len(self.history_best),
            "evals_used": self.eval_count,
            # "history": self.history_best,
            "history": history,  # just fitness per generation
        }

    # keep a method named for your earlier calls if you prefer
    def fcsa_hybrid_levy(self):
        return self.optimize()


def fcsa_hybrid_levy(func: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]], **kwargs):
    """
    Thin function wrapper so callers can do:
      from fcsa_levy import fcsa_hybrid_levy
      x_best, f_best, info = fcsa_hybrid_levy(obj, bounds)
    """
    return FCSA(func, bounds, **kwargs).fcsa_hybrid_levy()
