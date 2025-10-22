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
    tag: int = 0        # short-lived tag (half-life counter)
    T: int = 0          # age
    S: int = 0          # memory count


class ETFCSA_Lite:
    """
    Event-Triggered FCSA (Lite):
    - FCSA anchored cloning/mutation
    - Event-triggered updates (no full generations)
    - Tiny chaotic spark (IICO-like) for nucleation
    - Lightweight clearance + final short polish
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
        # tiny spark and clearance
        spark_prob: float = 0.04,
        tag_half_life: int = 250,
        clearance_period: float = 0.06,  # fraction of budget
        budget_per_tick: int = 200,
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
        # novelty reference set (tiny)
        self._grid_idx = list(range(min(self.N, max(8, self.dim))))

    # ---------- signals ----------
    def _update_signals(self, idx: int, old_aff: float):
        ab = self.pop[idx]
        # improvement EWMA of positive deltas
        gain = max(0.0, ab.aff - old_aff)
        ab.I = 0.85 * ab.I + 0.15 * gain
        # novelty = normalized nearest neighbor distance wrt small reference set
        xs = np.stack([self.pop[j].x for j in self._grid_idx], axis=0)
        d = np.linalg.norm(xs - ab.x[None, :], axis=1)
        if d.size:
            ab.J = float(np.min(d) / (np.mean(d) + 1e-12))
        # tag decay
        if ab.tag > 0:
            ab.tag -= 1

    # ---------- FCSA mutate ----------
    def _mutate_fcsa(self, x: np.ndarray, a_norm: float) -> np.ndarray:
        p = math.exp(-self.r * a_norm)  # Eq.3 gate
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
        z = 3.99 * z * (1 - z)  # two logistic steps are overkill; one is enough
        kick = (z - 0.5) * 0.5 * self.widths
        y = x + kick
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        # quasi-opposite pull
        opp = self.bounds[:, 0] + self.bounds[:, 1] - y
        mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        lo = np.minimum(mid, opp)      # FIX: ensure low < high per-dimension
        hi = np.maximum(mid, opp)
        y = self.rng.uniform(lo, hi)
        np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
        return y

    # ---------- one individual fire ----------
    def _fire_one(self, i: int) -> int:
        ab = self.pop[i]
        old_aff = ab.aff

        # choose operator
        if self.rng.random() < self.spark_prob:
            cand = self._spark(ab.x)
        else:
            # light FCSA: scale a_norm by rank among neighbors (cheap proxy)
            a_norm = 0.5
            cand = self._mutate_fcsa(ab.x, a_norm)

        f = self._objective(cand)
        if f < -ab.aff:
            ab.x = cand
            ab.aff = -f
            ab.S = max(1, ab.S + 1)
            ab.T = 0
            ab.tag = max(ab.tag, self.tag_half_life)
            if f < self.best_f:
                self.best_f, self.best_x = f, cand.copy()

        # update signals after potential move
        self._update_signals(i, old_aff)
        return 1  # one FE consumed above

    # ---------- small local cloning when hot ----------
    def _micro_clone(self, hot_indices: List[int], budget: int) -> int:
        if budget <= 0 or not hot_indices:
            return 0
        used = 0
        # pick top few by I
        hot = sorted(hot_indices, key=lambda j: self.pop[j].I, reverse=True)[:min(len(hot_indices), self.n_select)]
        affs = np.array([self.pop[j].aff for j in hot])
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)

        for j in hot:
            if used >= budget:
                break
            a_norm = (self.pop[j].aff - a_min) / denom if denom > 0 else 0.5
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                if used >= budget:
                    break
                y = self._mutate_fcsa(self.pop[j].x, a_norm)
                f = self._objective(y); used += 1
                if f < -self.pop[j].aff:
                    self.pop[j].x = y
                    self.pop[j].aff = -f
                    self.pop[j].S = max(1, self.pop[j].S + 1)
                    self.pop[j].T = 0
                    self.pop[j].tag = max(self.pop[j].tag, self.tag_half_life)
                    if f < self.best_f:
                        self.best_f, self.best_x = f, y.copy()
        return used

    # ---------- clearance (fast and safe) ----------
    def _clearance(self):
        survivors: List[Antibody] = []
        for ab in self.pop:
            if ab.tag > 0 or ab.I > 1e-12:
                survivors.append(ab)
        need = self.N - len(survivors)
        for _ in range(need):
            # reseed near opposite of current best to keep novelty high
            if self.best_x is None:
                y = self.bounds[:, 0] + self.rng.random(self.dim) * self.widths
            else:
                opp = self.bounds[:, 0] + self.bounds[:, 1] - self.best_x
                mid = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
                lo = np.minimum(mid, opp)   # FIX: enforce valid ranges
                hi = np.maximum(mid, opp)
                y = self.rng.uniform(lo, hi)
            np.clip(y, self.bounds[:, 0], self.bounds[:, 1], out=y)
            f = self._objective(y)
            survivors.append(Antibody(x=y, aff=-f))
            if f < self.best_f:
                self.best_f, self.best_x = f, y.copy()
        self.pop = survivors

    # ---------- scheduler pick ----------
    def _pick_indices(self) -> List[int]:
        scores = []
        for i, ab in enumerate(self.pop):
            ab.T += 1
            scores.append(self.alpha_I * ab.I + self.beta_J * ab.J)
        order = np.argsort(scores)[::-1]
        # fire those above threshold + a few random for insurance
        hot = [int(i) for i in order if scores[i] > self.theta]
        if len(hot) < max(2, self.N // 10):  # ensure some activity
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

            # pick who fires
            hot = self._pick_indices()
            if not hot:
                self.theta *= 0.9
                continue

            # spend half budget on single fires, half on micro-clone of the hottest
            b1 = max(1, budget // 2)
            b2 = budget - b1

            # single fires
            used = 0
            for i in hot:
                if used >= b1: break
                used += self._fire_one(i)

            # micro-clone on top fraction of HOT set
            k = max(1, len(hot) // 3)
            used += self._micro_clone(hot[:k], b2)

            # adapt threshold: aim at target firing density
            fired_fraction = len(hot) / max(1, self.N)
            self.theta += self.threshold_eta * (fired_fraction - self.fire_target)

            # periodic clearance
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
