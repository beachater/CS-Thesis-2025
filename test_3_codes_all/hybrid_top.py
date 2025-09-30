# hybrid_top.py
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


class HybridCSAOriginal_sbm:
    """
    FCSA core with:
      - Rac1-style forgetting (T/S threshold)
      - QOBL/QRBL opportunistic proposals
      - Adaptive parameter schedule (IICO-like Z, sigma)
      - SBM (Spectrum-bridged manifold step):
          * Acceptance-aware transport scale t
          * Gaussian bridge from elite covariance
          * Self-adaptive F per-strategy
          * Short backtracking line search on non-improving proposals

    This file is self-contained and compatible with your runner.
    """

    # ----------------------------- init -----------------------------
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

        # IICO-style schedule (used for anneals / logging)
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        max_stagnation: int = 3,

        # --- SBM base params (t, F, noise) ---
        sbm_p_top: float = 0.25,
        sbm_batch: int = 6,
        t_min: float = 0.02,
        t_max: float = 0.60,
        F_min: float = 0.15,
        F_max: float = 0.60,
        sbm_eps: float = 1e-6,
        noise_scale_max: float = 0.01,

        # --- precision & diversity boosters ---
        entropy_bins: int = 16,
        entropy_floor: float = 0.25,
        entropy_boost_t: float = 0.20,
        entropy_boost_noise: float = 1.5,
        stagnation_boost_t: float = 0.10,
        curvature_alpha_max: float = 0.30,

        # logging
        verbose: bool = True,
        log_every: int = 50,
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

        # schedule
        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_stagnation = int(max_stagnation)

        # SBM params
        self.sbm_p_top = float(np.clip(sbm_p_top, 0.05, 1.0))
        self.sbm_batch = int(max(1, sbm_batch))
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.F_min = float(F_min)
        self.F_max = float(F_max)
        self.sbm_eps = float(sbm_eps)
        self.noise_scale_max = float(noise_scale_max)

        # boosters
        self.entropy_bins = int(max(8, entropy_bins))
        self.entropy_floor = float(np.clip(entropy_floor, 0.0, 1.0))
        self.entropy_boost_t = float(entropy_boost_t)
        self.entropy_boost_noise = float(max(1.0, entropy_boost_noise))
        self.stagnation_boost_t = float(stagnation_boost_t)
        self.curvature_alpha_max = float(np.clip(curvature_alpha_max, 0.0, 1.0))

        self.seed = seed
        self.verbose = verbose
        self.log_every = int(max(1, log_every))

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Domain
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []

        # strategy pool & successes
        self.strategies = ["sbm_rand1", "sbm_current_to_pcenter", "sbm_current_rand"]
        self.strategy_success = {s: 1 for s in self.strategies}

        # acceptance memory & per-strategy F-intervals
        self._last_accept = 1.0
        self._F_stats = {
            "sbm_rand1": [self.F_min, self.F_max, 1, 1],                 # [F_lo, F_hi, succ, trials]
            "sbm_current_to_pcenter": [self.F_min, self.F_max, 1, 1],
            "sbm_current_rand": [self.F_min, self.F_max, 1, 1],
        }

        # transport state (set each gen)
        self._mu_t = np.zeros(self.dim, dtype=float)
        self._G_t = np.eye(self.dim, dtype=float)

        # diagnostics (lists append per generation)
        self.diag: Dict[str, Any] = {
            "gen": [],
            "best": [],
            "mean": [],
            "entropy": [],
            "t": [],
            "noise": [],
            "alpha": [],
            "stagn": [],
            "accept_rate": [],
            "succ_rand1": [],
            "succ_current_pc": [],
            "succ_current_rand": [],
        }

    # ----------------------------- utils -----------------------------
    def _objective(self, x: np.ndarray) -> float:
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform_point(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # population init
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

    # QOBL / QRBL
    def _qobl(self, x: np.ndarray) -> np.ndarray:
        opposite = self.lb + self.ub - x
        lows = np.minimum(self.mid, opposite)
        highs = np.maximum(self.mid, opposite)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _qrbl(self, x: np.ndarray) -> np.ndarray:
        lows = np.minimum(self.mid, x)
        highs = np.maximum(self.mid, x)
        return lows + np.random.rand(self.dim) * (highs - lows)

    # forgetting (Rac1 surrogate)
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform_point()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # population entropy (normalized)
    def _population_entropy(self, pop: List[Antibody]) -> float:
        X = np.stack([ab.x for ab in pop], axis=0)
        D = X.shape[1]
        ent = 0.0
        for d in range(D):
            x_d = X[:, d]
            data_range = np.max(x_d) - np.min(x_d)
            if data_range == 0.0:
                ent_d = 0.0  # No diversity in this dimension
            else:
                hist, _ = np.histogram(x_d, bins=self.entropy_bins, density=True)
                p = hist + 1e-12
                p = p / p.sum()
                ent_d = -np.sum(p * np.log(p + 1e-12))
            ent += ent_d
        ent /= float(D)
        ent = ent / math.log(self.entropy_bins + 1e-12)
        return float(ent)

    # --------------------- SBM transport components ---------------------
    def _noise(self, scale: float) -> np.ndarray:
        # isotropic domain-scaled noise
        return scale * self.widths * np.random.randn(self.dim)

    def _cov_reg(self, X):
        if X.shape[0] <= 1:
            return np.eye(self.dim) * 1e-6
        C = np.cov(X.T)
        return C + 1e-9 * np.eye(C.shape[0])

    def _compute_gaussian_bridge(
        self, pop: List[Antibody], gen: int, stagnation: int, best_val: float, last_accept: float
    ) -> Tuple[float, float, float, float]:
        """
        Set transport center mu_t and bridge matrix G_t from elite covariance.
        Returns (t_used, noise_scale, alpha_used, entropy).
        """
        prog = gen / max(1, self.max_gens)

        # acceptance-aware base t
        t_base = self.t_max * (1.0 - prog) + self.t_min * prog
        if last_accept < 0.05:
            t_base *= 0.5
        elif last_accept < 0.10:
            t_base *= 0.7
        elif last_accept > 0.25:
            t_base = min(self.t_max, t_base * 1.05)

        # entropy & noise
        entropy = self._population_entropy(pop)
        noise_scale = (1.0 - prog) * self.noise_scale_max

        t_bump = 0.0
        if stagnation > self.max_stagnation:
            t_bump += self.stagnation_boost_t
        if entropy < self.entropy_floor:
            t_bump += self.entropy_boost_t
            noise_scale *= self.entropy_boost_noise

        t_used = float(np.clip(t_base + t_bump, self.t_min, self.t_max))

        # elite set for covariance
        k_elite = max(2, int(self.sbm_p_top * len(pop)))
        elites = sorted(pop, key=lambda ab: ab.affinity, reverse=True)[:k_elite]
        X = np.stack([ab.x for ab in elites], axis=0)
        mu = np.mean(X, axis=0)
        C = self._cov_reg(X - mu)

        # spectral bridge: G = t * U diag(sqrt(lam) / (trace norm)) U^T
        lam, U = np.linalg.eigh(C)
        lam = np.maximum(lam, 1e-12)
        sqrt_lam = np.sqrt(lam)
        # normalize step magnitude by Frobenius norm to keep proposals tempered
        norm_factor = np.linalg.norm(sqrt_lam) + 1e-12
        S = U @ np.diag(sqrt_lam / norm_factor) @ U.T
        self._G_t = t_used * S
        self._mu_t = mu

        # curvature alpha (lightweight logging / future hook)
        alpha_used = self.curvature_alpha_max * (1.0 - prog)

        return t_used, noise_scale, alpha_used, entropy

    # --------------------- F self-adaptation ---------------------
    def _F_adapt(self, strat: str) -> float:
        F_lo, F_hi, succ, trials = self._F_stats[strat]
        rate = succ / max(1, trials)
        # adjust interval by success rate
        if rate < 0.05:
            F_hi = max(F_lo + 0.05, 0.7 * F_hi)
        elif rate > 0.25:
            F_hi = min(self.F_max, F_hi * 1.05)
            F_lo = max(self.F_min, 0.95 * F_lo)
        self._F_stats[strat] = [F_lo, F_hi, succ, trials]
        return float(np.random.uniform(F_lo, F_hi))

    def _F_update_stats(self, strat: str, improved: bool):
        F_lo, F_hi, succ, trials = self._F_stats[strat]
        self._F_stats[strat] = [F_lo, F_hi, succ + (1 if improved else 0), trials + 1]

    # --------------------- SBM strategies ---------------------
    def _choose_strategy(self) -> str:
        total = sum(self.strategy_success.values())
        probs = [self.strategy_success[s] / total for s in self.strategies]
        return str(np.random.choice(self.strategies, p=probs))

    def _sbm_rand1(self, pop: List[Antibody], noise_scale: float, strategy_name="sbm_rand1") -> np.ndarray:
        idxs = np.random.choice(len(pop), 3, replace=False)
        x1, x2, x3 = [pop[i].x for i in idxs]
        F = self._F_adapt(strategy_name)
        d = x2 - x3
        return x1 + F * (self._G_t @ d) + self._noise(noise_scale)

    def _sbm_current_to_pcenter(
        self, ab: Antibody, pop: List[Antibody], noise_scale: float, strategy_name="sbm_current_to_pcenter"
    ) -> np.ndarray:
        x1, x2 = [pop[i].x for i in np.random.choice(len(pop), 2, replace=False)]
        F = self._F_adapt(strategy_name)
        drift = self._G_t @ (self._mu_t - ab.x)
        kick = self._G_t @ (x1 - x2)
        return ab.x + F * drift + F * kick + self._noise(noise_scale)

    def _sbm_current_rand(
        self, ab: Antibody, pop: List[Antibody], noise_scale: float, strategy_name="sbm_current_rand"
    ) -> np.ndarray:
        x1, x2 = [pop[i].x for i in np.random.choice(len(pop), 2, replace=False)]
        F = self._F_adapt(strategy_name)
        return ab.x + F * (self._G_t @ (x1 - x2)) + self._noise(noise_scale)

    # ----------------------------- main -----------------------------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history = []

        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagnation = 0

        k = max(2, self.max_gens)
        beta = self.beta

        for gen in range(1, self.max_gens + 1):
            for ab in pop:
                ab.T += 1

            # schedule bits (also used for numerical stability like original)
            Z = math.exp(-beta * gen / k)
            if Z <= self.gamma:
                beta = -math.log(10.0 * self.gamma) * k / max(1, gen)
                Z = math.exp(-beta * gen / k)
            sigma_iter = ((k - gen) / max(1.0, (k - 1))) ** self.exponent * (
                self.sigma_initial - self.sigma_final
            ) + self.sigma_final
            # alpha_iter is informative; bridge uses curvature_alpha_max anyway
            alpha_iter = 10.0 * math.log(max(1e-9, self.M)) * Z
            if not np.isfinite(alpha_iter):
                alpha_iter = self.sigma_final

            # compute bridge transport
            t_used, noise_scale, alpha_used, entropy = self._compute_gaussian_bridge(
                pop, gen, stagnation, best_val, self._last_accept
            )

            # selection & cloning
            selected = self._select_top(pop)
            clones = self._clone(selected)

            # proposals with short backtracking line search
            new_clones: List[Antibody] = []
            improved_raw = 0
            total_props = 0

            for ab in clones:
                strategy = self._choose_strategy()
                total_props += 1

                # 1) propose according to strategy
                if strategy == "sbm_rand1":
                    cand = self._sbm_rand1(pop, noise_scale, "sbm_rand1")
                elif strategy == "sbm_current_to_pcenter":
                    cand = self._sbm_current_to_pcenter(ab, pop, noise_scale, "sbm_current_to_pcenter")
                else:
                    cand = self._sbm_current_rand(ab, pop, noise_scale, "sbm_current_rand")

                self._clip(cand)
                cand_fit = self._affinity(cand)
                base_improved = (cand_fit > ab.affinity)

                # 2) short backtracking line search if not improved
                if not base_improved:
                    direction = cand - ab.x
                    tried = 0
                    best_local_x = cand
                    best_local_fit = cand_fit
                    while tried < 3:
                        tried += 1
                        cand_bt = ab.x + 0.5 * direction
                        self._clip(cand_bt)
                        fit_bt = self._affinity(cand_bt)
                        if fit_bt > best_local_fit:
                            best_local_x, best_local_fit = cand_bt, fit_bt
                            direction = cand_bt - ab.x
                        else:
                            # try pure drift once (towards mu_t)
                            if tried == 1:
                                drift = self._G_t @ (self._mu_t - ab.x)
                                cand_drift = ab.x + 0.5 * drift
                                self._clip(cand_drift)
                                fit_drift = self._affinity(cand_drift)
                                if fit_drift > best_local_fit:
                                    best_local_x, best_local_fit = cand_drift, fit_drift
                                    direction = cand_drift - ab.x
                    cand, cand_fit = best_local_x, best_local_fit

                improved = (cand_fit > ab.affinity)
                if improved:
                    improved_raw += 1
                    self.strategy_success[strategy] += 1
                self._F_update_stats(strategy, improved)

                # 3) QOBL / QRBL polish
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

            # pool update + forgetting
            pool = pop + new_clones
            pool.sort(key=lambda a: a.affinity, reverse=True)
            pop = pool[: self.N]
            self._forget_in_place(pop)

            # track best
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

            # acceptance stats (EMA for next gen)
            acc = (improved_raw / max(1, total_props))
            self._last_accept = 0.5 * self._last_accept + 0.5 * acc

            # diagnostics
            mean_fit = np.mean([-ab.affinity for ab in pop])
            self.diag["gen"].append(gen)
            self.diag["best"].append(best_val)
            self.diag["mean"].append(mean_fit)
            self.diag["entropy"].append(float(entropy))
            self.diag["t"].append(float(t_used))
            self.diag["noise"].append(float(noise_scale))
            self.diag["alpha"].append(float(alpha_used))
            self.diag["stagn"].append(int(stagnation))
            self.diag["accept_rate"].append(float(acc))
            self.diag["succ_rand1"].append(self._F_stats["sbm_rand1"][2] / max(1, self._F_stats["sbm_rand1"][3]))
            self.diag["succ_current_pc"].append(
                self._F_stats["sbm_current_to_pcenter"][2] / max(1, self._F_stats["sbm_current_to_pcenter"][3])
            )
            self.diag["succ_current_rand"].append(
                self._F_stats["sbm_current_rand"][2] / max(1, self._F_stats["sbm_current_rand"][3])
            )

            # if self.verbose and (gen % self.log_every == 0 or gen == 1 or gen == self.max_gens):
            #     print(
            #         f"[gen {gen}] best={best_val:.3e}  t={t_used:.3f}  "
            #         f"noise={noise_scale:.4g}  stagn={stagnation}  acc={acc:.2f}"
            #     )

        info = {
            "generations": self.max_gens,
            "history": history,      # best fitness per generation
            "diag": self.diag,       # per-generation diagnostics (monitored by your runner)
        }
        return best.x.copy(), best_val, info
