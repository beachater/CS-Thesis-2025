from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    aff: float          # affinity = -f (maximize)
    I: float = 0.0      # improvement EWMA
    J: float = 0.0      # novelty score
    tag: int = 0        # short-lived tag counter
    T: int = 0          # age
    S: int = 0          # memory/selection count


class ETFCSA_Lite:
    """
    Event-Triggered FCSA (Lite) with explicit Rac1-style forgetting:
    - FCSA anchor: affinity, Eq.3 mutation gate, cloning, T/S bookkeeping
    - Event-triggered firing of individuals (no global generations)
    - Tiny chaotic spark for nucleation (IICO-inspired, single step)
    - Rac1 hooks:
        1) Scheduler penalty by activity A = T/(S+eps)
        2) Mutation floor grows with A when A > c_threshold
        3) Cheap reseed for inactive high-activity antibodies each tick
    - Lightweight clearance and short final polish
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        n_select: int = 12,
        n_clones: int = 4,
        r: float = 2.0,           # FCSA Eq.3 exponent
        a_frac: float = 0.12,     # mutation span fraction of box
        seed: Optional[int] = 42,
        max_evals: int = 350_000,
        # event scheduler
        fire_target: float = 0.2,
        alpha_I: float = 1.0,
        beta_J: float = 0.5,
        threshold_eta: float = 0.05,
        # spark and clearance
        spark_prob: float = 0.04,
        tag_half_life: int = 250,
        clearance_period: float = 0.06,  # fraction of budget
        budget_per_tick: int = 200,
        # Rac1 parameters
        c_threshold: float = 3.0,
        gamma_rac1: float = 0.75,        # scheduler penalty strength
        progress: Optional[Callable[[int], None]] = None,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.N = int(N)
        self.n_select = int(n_select)
        self.n_clones = int(n_clones)
        self.r = float(r)
        self.a_frac = float(a_frac)
        self.max_evals = int(max_evals)

        self.fire_target = float(fire_target)
        self.alpha_I = float(alpha_I)
        self.beta_J = float(beta_J)
        self.threshold_eta = float(threshold_eta)

        self.spark_prob = float(spark_prob)
        self.tag_half_life = int(tag_half_life)
        self.clear_every = max(1, int(clearance_period * self.max_evals))
        self.budget_per_tick = int(budget_per_tick)

        self.c_threshold = float(c_threshold)
        self.gamma_rac1 = float(gamma_rac1)

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.widths = self.bounds[:, 1] - self.bounds[:, 0]
        self.a_vec = self.a_frac * self.widths

        self._progress = progress
        self.evals = 0

        self.pop: List[Antibody] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float("inf")
        self.history: List[float] = []
        self.theta = 0.0  # scheduler threshold
        self._grid_idx: List[int] = []  # small subset for novelty J

    # ---------- evaluation ----------
    def _objective(self, x: np.ndarray) -> float:
        self.evals += 1
        if self._progress:
            try: self._progress(1)
            except Exception: pass
        return float(self.func(x))

    # ---------- init ----------
    def _init(self):
        self.pop.clear()
        for _ in range(self.N):
            x = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
            f = self._objective(x)
            ab = Antibody(x=x, aff=-f)
            self.pop.append(ab)
            if f < self.best_f:
                self.best_f, self.best_x = f, x.copy()
        # novelty reference set
        self._grid_idx = list(range(min(self.N, max(8, self.dim))))

    # ---------- signals ----------
    def _update_signals(self, idx: int, old_aff: float):
        ab = self.pop[idx]
        gain = max(0.0, ab.aff - old_aff)
        ab.I = 0.85 * ab.I + 0.15 * gain
        if self._grid_idx:
            xs = np.stack([self.pop[j].x for j in self._grid_idx], axis=0)
            d = np.linalg.norm(xs - ab.x[None, :], axis=1)
            if d.size:
                ab.J = float(np.min(d) / (np.mean(d) + 1e-12))
        if ab.tag > 0:
            ab.tag -= 1

    # ---------- FCSA mutate with Rac1 floor ----------
    def _mutate_fcsa(self, x: np.ndarray, a_norm: float, A: float) -> np.ndarray:
        p_base = math.exp(-self.r * a_norm)  # Eq.3
        # Rac1 floor if high activity
        over = max(0.0, A - self.c_threshold)
        p_floor = min(0.9, 0.2 + 0.15 * over)
        p = max(p_base, p_floor)
        mask = self.rng.random(self.dim) < p
        if np.any(mask):
            step = self.rng.uniform(-self.a_vec, self.a_vec)
            y = x + mask.astype(float) * step
            np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
            return y
        return x

    # ---------- spark (IICO-ish) ----------
    def _spark(self, x: np.ndarray) -> np.ndarray:
        z = self.rng.random(self.dim)
        z = 3.99 * z * (1 - z)
        kick = (z - 0.5) * 0.5 * self.widths
        y = x + kick
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        opp = self.bounds[:, 0] + self.bounds[:, 1] - y
        mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        lo = np.minimum(mid, opp)   # ensure valid per-dimension
        hi = np.maximum(mid, opp)
        y = self.rng.uniform(lo, hi)
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        return y

    # ---------- one individual fire ----------
    def _fire_one(self, i: int) -> int:
        ab = self.pop[i]
        old_aff = ab.aff
        A = ab.T / (ab.S + 1e-12)  # Rac1 activity

        if self.rng.random() < self.spark_prob:
            cand = self._spark(ab.x)
        else:
            a_norm = 0.5
            cand = self._mutate_fcsa(ab.x, a_norm, A)

        f = self._objective(cand)
        if f < -ab.aff:
            ab.x = cand
            ab.aff = -f
            ab.S = max(1, ab.S + 1)
            ab.T = 0
            ab.tag = max(ab.tag, self.tag_half_life)
            if f < self.best_f:
                self.best_f, self.best_x = f, cand.copy()

        self._update_signals(i, old_aff)
        return 1

    # ---------- small local cloning when hot ----------
    def _micro_clone(self, hot_indices: List[int], budget: int) -> int:
        if budget <= 0 or not hot_indices:
            return 0
        used = 0
        hot = sorted(hot_indices, key=lambda j: self.pop[j].I, reverse=True)[:min(len(hot_indices), self.n_select)]
        affs = np.array([self.pop[j].aff for j in hot])
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for j in hot:
            if used >= budget:
                break
            ab = self.pop[j]
            A = ab.T / (ab.S + 1e-12)
            a_norm = (ab.aff - a_min) / denom if denom > 0 else 0.5
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                if used >= budget: break
                y = self._mutate_fcsa(ab.x, a_norm, A)
                f = self._objective(y); used += 1
                if f < -ab.aff:
                    ab.x = y
                    ab.aff = -f
                    ab.S = max(1, ab.S + 1)
                    ab.T = 0
                    ab.tag = max(ab.tag, self.tag_half_life)
                    if f < self.best_f:
                        self.best_f, self.best_x = f, y.copy()
        return used

    # ---------- Rac1 reseed of zombies ----------
    def _rac1_reseed(self):
        # reseed only if high activity and not improving and untagged
        new_pop: List[Antibody] = []
        for ab in self.pop:
            improving = ab.I > 1e-12
            A = ab.T / (ab.S + 1e-12)
            if (A > self.c_threshold) and (not improving) and (ab.tag == 0):
                # opposition-biased reseed near best
                if self.best_x is None:
                    y = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
                else:
                    opp = self.bounds[:, 0] + self.bounds[:, 1] - self.best_x
                    mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
                    lo = np.minimum(mid, opp)
                    hi = np.maximum(mid, opp)
                    y = self.rng.uniform(lo, hi)
                np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
                f = self._objective(y)
                ab = Antibody(x=y, aff=-f, tag=self.tag_half_life, T=0, S=0)
                if f < self.best_f:
                    self.best_f, self.best_x = f, y.copy()
            new_pop.append(ab)
        self.pop = new_pop

    # ---------- clearance (fast and safe) ----------
    def _clearance(self):
        survivors: List[Antibody] = []
        for ab in self.pop:
            if ab.tag > 0 or ab.I > 1e-12:
                survivors.append(ab)
        need = self.N - len(survivors)
        for _ in range(need):
            if self.best_x is None:
                y = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
            else:
                opp = self.bounds[:, 0] + self.bounds[:, 1] - self.best_x
                mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
                lo = np.minimum(mid, opp)
                hi = np.maximum(mid, opp)
                y = self.rng.uniform(lo, hi)
            np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
            f = self._objective(y)
            survivors.append(Antibody(x=y, aff=-f))
            if f < self.best_f:
                self.best_f, self.best_x = f, y.copy()
        self.pop = survivors

    # ---------- scheduler pick with Rac1 penalty ----------
    def _pick_indices(self) -> List[int]:
        scores = []
        for i, ab in enumerate(self.pop):
            ab.T += 1
            A = ab.T / (ab.S + 1e-12)
            base = self.alpha_I * ab.I + self.beta_J * ab.J
            # Rac1 scheduler penalty
            over = max(0.0, A - self.c_threshold)
            penal = math.exp(-self.gamma_rac1 * over)
            scores.append(base * penal)
        order = np.argsort(scores)[::-1]
        hot = [int(i) for i in order if scores[i] > self.theta]
        if len(hot) < max(2, self.N // 10):
            extra = self.rng.choice(self.N, size=min(self.N // 10, self.N), replace=False)
            hot = list(dict.fromkeys(list(hot) + list(map(int, extra))))
        return hot

    # ---------- final polish ----------
    def _polish(self, steps: int = 120):
        if self.best_x is None:
            return
        x = self.best_x.copy()
        f = self.best_f
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        step = 0.04 * self.widths
        for t in range(steps):
            d = t % self.dim
            for sgn in (+1.0, -1.0):
                cand = x.copy()
                cand[d] = np.clip(cand[d] + sgn * step[d], lo[d], hi[d])
                fc = self._objective(cand)
                if fc < f:
                    x, f = cand, fc
            step *= 0.985
            if np.max(step) < 1e-12:
                break
        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()

    # ---------- main ----------
    def optimize(self):
        self._init()
        last_clear = 0
        while self.evals < self.max_evals:
            budget = min(self.budget_per_tick, self.max_evals - self.evals)

            hot = self._pick_indices()
            if not hot:
                self.theta *= 0.9
                continue

            b1 = max(1, budget // 2)
            b2 = budget - b1

            used = 0
            for i in hot:
                if used >= b1: break
                used += self._fire_one(i)

            k = max(1, len(hot) // 3)
            used += self._micro_clone(hot[:k], b2)

            fired_fraction = len(hot) / max(1, self.N)
            self.theta += self.threshold_eta * (fired_fraction - self.fire_target)

            # cheap Rac1 reseed each tick
            self._rac1_reseed()

            if self.evals - last_clear >= self.clear_every:
                self._clearance()
                last_clear = self.evals

            self.history.append(self.best_f)
            if self.evals >= self.max_evals:
                break

        self._polish()
        return self.best_x.copy(), self.best_f, {
            "evals_used": self.evals,
            "history": self.history,
        }
