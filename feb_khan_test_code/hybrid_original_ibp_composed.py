# hybrid_csa_with_operators.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np

@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0
    S: int = 0

class HybridCSA_IBP_Composed:
    """
    Hybrid CSA with IBP operator and three variants:
    - ibp_hp   : IBP + Hebbian Plasticity
    - ibp_ip   : IBP + Immune Pruning
    - ibp_hpip : IBP + Hebbian Plasticity + Immune Pruning
    """
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        operator: str = "ibp",   # "ibp_hp", "ibp_ip", "ibp_hpip"
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
        ibp_k_frac: float = 0.05,
        ibp_recon_noise: float = 0.01,
        ibp_update_window: int = 10,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)
        self.operator = operator

        # parameters
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

        # IBP params
        self.ibp_k_frac = float(ibp_k_frac)
        self.ibp_recon_noise = float(ibp_recon_noise)
        self.ibp_update_window = int(ibp_update_window)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # domain
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []

        # IBP cache
        self._ibp_P = None
        self._ibp_mean = None
        self._ibp_history = []

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
        return [Antibody(x=self._sample_uniform_point(),
                         affinity=self._affinity(self._sample_uniform_point()))
                for _ in range(self.N)]

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
        weights /= (np.sum(weights) + 1e-12)
        X = np.stack([ab.x for ab in elites], axis=0)
        return np.sum(X * weights[:, None], axis=0)

    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        norm = (affs - affs.min()) / (np.ptp(affs) + 1e-12) if np.ptp(affs) > 0 else np.ones_like(affs)
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

    # ---------- new mechanisms ----------
    def _hebbian_update(self, candidate: np.ndarray, best: np.ndarray) -> np.ndarray:
        """Hebbian plasticity: reinforce direction of improvement."""
        lr = 0.1
        return candidate + lr * (candidate - best)

    def _immune_prune(self, pop: List[Antibody]) -> List[Antibody]:
        """Immune pruning: remove weakest and repopulate."""
        fits = [ab.affinity for ab in pop]
        threshold = np.percentile(fits, 70)  # keep best 30%
        survivors = [ab for ab in pop if ab.affinity >= threshold]
        while len(survivors) < self.N:
            x = self._sample_uniform_point()
            survivors.append(Antibody(x=x, affinity=self._affinity(x)))
        return survivors

    # ---------- IBP operator ----------
    def _ibp_update_projection(self, pop_array: np.ndarray) -> None:
        D = self.dim
        k = max(2, int(max(2, math.ceil(self.ibp_k_frac * D))))
        mean = pop_array.mean(axis=0)
        Xc = pop_array - mean
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Vt[:k].T
        except Exception:
            P = np.random.randn(D, k)
            Q, _ = np.linalg.qr(P)
            P = Q[:, :k]
        self._ibp_P, self._ibp_mean = P, mean

    def _ibp_mutation(self, base: np.ndarray, pop_array: np.ndarray) -> np.ndarray:
        if self._ibp_P is None:
            self._ibp_update_projection(pop_array)
        P, mean = self._ibp_P, self._ibp_mean
        z = P.T.dot(base - mean)
        zp = z + np.random.normal(scale=self.ibp_recon_noise * np.linalg.norm(z) + 1e-8, size=z.shape)
        recon = mean + P.dot(zp)
        recon += np.random.normal(scale=0.002 * np.mean(self.widths), size=self.dim)
        return recon

    # ---------- main loop ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history, best = [], max(pop, key=lambda ab: ab.affinity)
        best_val, stagnation = -best.affinity, 0
        ibp_window = []

        for gen in range(1, self.max_gens + 1):
            for ab in pop:
                ab.T += 1
            selected = self._select_top(pop)
            clones = self._clone(selected)

            pop_array = np.stack([ab.x for ab in pop], axis=0)
            ibp_window.append(pop_array)
            if len(ibp_window) > self.ibp_update_window:
                ibp_window.pop(0)
            big = np.vstack(ibp_window)
            self._ibp_update_projection(big)

            new_clones: List[Antibody] = []
            for ab in clones:
                cand = self._ibp_mutation(ab.x, pop_array)

                # operator-specific adjustments
                if self.operator in ("ibp_hp", "ibp_hpip"):
                    best_x = max(pop, key=lambda ab: ab.affinity).x
                    cand = self._hebbian_update(cand, best_x)

                self._clip(cand)
                cand_fit = self._affinity(cand)
                new_clones.append(Antibody(x=cand, affinity=cand_fit, T=0, S=1))

            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]

            # immune pruning if needed
            if self.operator in ("ibp_ip", "ibp_hpip"):
                pop = self._immune_prune(pop)

            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            self.history_best.append((cur_best_val, cur_best.x.copy()))
            history.append(cur_best_val)

            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity)
                stagnation = 0
            else:
                stagnation += 1

        diagnostics = {"generations": self.max_gens, "history": history, "history_best": self.history_best}
        return best.x.copy(), best_val, diagnostics
