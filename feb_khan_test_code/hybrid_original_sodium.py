# hybrid_original_with_pump.py
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


class HybridCSAOriginal_Sodium:
    """
    Hybrid Clonal Selection with:
      1) Rac1-style forgetting (T/S threshold)
      2) Quasi-Opposition Based Learning (QOBL)
      3) Adaptive parameter schedule (IICO-style)
      4) Quasi-Reflection Based Learning (QRBL)

    Enhanced with a Sodium–Potassium pump inspired operator:
      - pump_enabled: turn the pump on/off
      - pump_out_frac: fraction of worst individuals to evict each generation (Na+ out)
      - pump_in_frac: fraction of population to (re)introduce from memory/elite (K+ in)
      - pump ATP budget (energy) controls how many directed expensive inflow moves can be used
      - memory: small elite archive used as source for inflow
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
        max_stagnation: int = 3,
        verbose: bool = True,
        # Pump params
        pump_enabled: bool = True,
        pump_out_frac: float = 0.20,  # fraction of population to evict each gen
        pump_in_frac: float = 0.10,   # fraction of population to inject back (should be <= pump_out_frac)
        pump_atp_init: float = 5.0,   # initial ATP budget (energy units)
        pump_atp_max: float = 20.0,   # maximum ATP store
        pump_cost_direct: float = 1.0,  # ATP cost for one directed expensive inflow (using elite-guided perturbation)
        pump_recharge_on_improve: float = 1.0,  # ATP gained when global best improves
        memory_max_size: int = 10,    # memory (immune) archive size
        memory_perturb_scale: float = 0.01,  # perturbation scale when injecting from memory (relative to domain)
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)

        # core CSA params
        self.N = int(N)
        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.a_frac = float(a_frac)
        self.c_threshold = float(c_threshold)
        self.max_gens = int(max_gens)

        # schedule params
        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_stagnation = int(max_stagnation)

        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # domain info
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []

        # Pump parameters & state
        self.pump_enabled = bool(pump_enabled)
        self.pump_out_frac = float(np.clip(pump_out_frac, 0.0, 1.0))
        self.pump_in_frac = float(np.clip(min(pump_in_frac, pump_out_frac), 0.0, 1.0))
        self.pump_atp = float(pump_atp_init)
        self.pump_atp_max = float(pump_atp_max)
        self.pump_cost_direct = float(pump_cost_direct)
        self.pump_recharge_on_improve = float(pump_recharge_on_improve)
        # immune-like memory to source inflow
        self.memory_max_size = int(max(1, memory_max_size))
        self.memory_perturb_scale = float(memory_perturb_scale)
        self._memory: List[Dict[str, Any]] = []  # entries: {"x":..., "fitness":..., "age":0}

    # ---------- utilities ----------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform_point(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ---------- population init ----------
    def _population_init(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform_point()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        # initialize memory with a few best from initial pop
        self._memory = []
        for ab in sorted(pop, key=lambda a: a.affinity, reverse=True)[: max(1, self.memory_max_size // 4)]:
            self._memory.append({"x": ab.x.copy(), "fitness": -ab.affinity, "age": 0})
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

    # ---------- pump helpers ----------
    def _memory_add(self, x: np.ndarray, fitness: float) -> None:
        """Add candidate to memory (keep best entries)."""
        entry = {"x": x.copy(), "fitness": float(fitness), "age": 0}
        self._memory.append(entry)
        # prune by fitness (lowest objective = best)
        self._memory.sort(key=lambda e: e["fitness"])
        if len(self._memory) > self.memory_max_size:
            self._memory = self._memory[: self.memory_max_size]

    def _memory_age_and_prune(self) -> None:
        for e in self._memory:
            e["age"] += 1
        # optionally prune very old if memory grows (keep best)
        self._memory.sort(key=lambda e: e["fitness"])
        if len(self._memory) > self.memory_max_size:
            self._memory = self._memory[: self.memory_max_size]

    def _pump_inject(self, pop: List[Antibody], replace_idxs: List[int], elite_center: np.ndarray, alpha_iter: float) -> List[Antibody]:
        """
        Replace individuals at indices replace_idxs in pop with inflow candidates.
        Uses memory items (preferred) or elite-guided directed perturbations (costly).
        Returns modified pop.
        """
        if not replace_idxs:
            return pop

        # prefer to use memory items when available
        mem_count = len(self._memory)
        for i, ridx in enumerate(replace_idxs):
            use_memory = (mem_count > 0) and (i < mem_count)
            if use_memory:
                mem = self._memory[min(i, mem_count - 1)]
                # perturb memory candidate slightly; perturb scale relative to domain width
                perturb = np.random.normal(scale=self.memory_perturb_scale * np.mean(self.widths), size=self.dim)
                x_new = mem["x"].copy() + perturb
                self._clip(x_new)
                pop[ridx] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0)
                continue

            # otherwise create inflow candidate. Try expensive directed inflow if ATP allows.
            if (self.pump_enabled and self.pump_atp >= self.pump_cost_direct) and (random.random() < 0.9):
                # directed inflow: push a new candidate toward elite center with scaled amplitude
                diff = elite_center - pop[ridx].x
                norm = np.linalg.norm(diff) + 1e-12
                A_vec = 10.0 * alpha_iter * (diff / norm) if norm > 0 else np.random.randn(self.dim) * alpha_iter
                cand = pop[ridx].x + A_vec + np.random.randn(self.dim) * alpha_iter * 0.5
                self._clip(cand)
                cand_fit = self._affinity(cand)
                pop[ridx] = Antibody(x=cand, affinity=cand_fit, T=0, S=0)
                # consume ATP
                self.pump_atp = max(0.0, self.pump_atp - self.pump_cost_direct)
            else:
                # cheap inflow: uniform random replacement
                x_new = self._sample_uniform_point()
                pop[ridx] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0)
        return pop

    # ---------- main loop ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history = []  # best fitness per generation

        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagnation = 0

        k = max(2, self.max_gens)
        beta = self.beta

        # pump diagnostics
        pump_injected_total = 0
        pump_evicted_total = 0

        for gen in range(1, self.max_gens + 1):
            for ab in pop:
                ab.T += 1

            selected = self._select_top(pop)
            elites_for_center = max(
                1,
                int(
                    round(
                        self.n_select
                        * (0.5 + 0.5 * max(0, self.max_stagnation - stagnation) / max(1, self.max_stagnation))
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

            # pool update
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]

            # forgetting
            self._forget_in_place(pop)

            # --- Sodium-Potassium pump operator (population flow) ---
            if self.pump_enabled:
                # determine how many to evict/inject
                out_count = int(round(self.pump_out_frac * self.N))
                in_count = int(round(self.pump_in_frac * self.N))
                out_count = min(out_count, self.N - 1)  # keep at least one
                in_count = min(in_count, out_count)

                if out_count > 0:
                    # evict worst out_count individuals (worst affinity)
                    worst_indices = np.argsort([ab.affinity for ab in pop])[:out_count]
                    # mark evicted (we will replace some subset)
                    evicted_idxs = list(worst_indices)
                    pump_evicted_total += len(evicted_idxs)

                    # choose which evicted slots will be used for inflow (prefer first in list)
                    inject_slots = evicted_idxs[:in_count]
                    # replace via pump injection (memory preferred, else directed or uniform)
                    pop = self._pump_inject(pop, inject_slots, F1, alpha_iter)
                    pump_injected_total += len(inject_slots)

                    # remaining evicted slots (if any) replace with uniform samples
                    remaining_slots = [idx for idx in evicted_idxs[in_count:]]
                    for ridx in remaining_slots:
                        x_new = self._sample_uniform_point()
                        pop[ridx] = Antibody(x=x_new, affinity=self._affinity(x_new), T=0, S=0)

            # update memory with current elites (immune memory accumulation)
            cur_best_obj = min(self._objective(ab.x) for ab in pop)
            cur_best_ab = min(pop, key=lambda ab: self._objective(ab.x))  # best = smallest objective
            # Add best to memory when it's an improvement or memory not full
            if cur_best_obj <= best_val or len(self._memory) < self.memory_max_size:
                self._memory_add(cur_best_ab.x, cur_best_obj)

            # memory maintenance
            self._memory_age_and_prune()

            # ATP recharge when global improvement occurs
            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            if cur_best_val + 1e-12 < best_val:
                # improvement found — recharge ATP
                self.pump_atp = min(self.pump_atp + self.pump_recharge_on_improve, self.pump_atp_max)
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagnation = 0
            else:
                stagnation += 1

            # track best
            self.history_best.append((cur_best_val, cur_best.x.copy()))
            history.append(cur_best_val)

        diagnostics = {
            "generations": self.max_gens,
            "history": history,
            "pump_injected_total": pump_injected_total,
            "pump_evicted_total": pump_evicted_total,
            "pump_atp_final": float(self.pump_atp),
            "memory_size": len(self._memory),
        }
        return best.x.copy(), best_val, diagnostics
