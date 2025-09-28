# hybrid_fCSA_bdt_opf.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class Antibody:
    x: np.ndarray
    affinity: float  # maximize (=-objective)
    T: int = 0
    S: int = 0


class HybridFCSA_BDT_OPF:
    """
    FCSA-centric role-partitioned optimizer with:
      - Explorers: FCSA + BDT (Barycentric Directional Transport) + OPF (Orthogonal Perturbation Frames)
      - Exploiters: IICO-style center-direction with momentum (NOT GA/DE/PSO/etc.)
      - Role-aware Rac1 forgetting (explorers only)
      - Compactness trigger via elite covariance log-det
      - Annealed schedules for stability and fast convergence
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        p_exploit: float = 0.25,
        n_select: int = 15,
        n_clones: int = 5,
        a_frac: float = 0.15,
        r: float = 2.0,
        c_threshold: float = 3.0,
        max_gens: int = 1000,
        max_evals: int = 350_000,
        seed: Optional[int] = 42,
        # exchange params
        exchange_interval: int = 5,
        exchange_k: int = 2,
        # IICO-like schedules
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        # compactness trigger (elite covariance)
        spread_window: int = 5,
        spread_drop_threshold: float = 0.15,
        # BDT params
        bdt_tau_initial: float = 6.0,
        bdt_tau_final: float = 1.2,
        bdt_eta_initial: float = 0.40,
        bdt_eta_final: float = 0.05,
        # OPF params
        opf_cols: Optional[int] = None,
        opf_scale_initial: float = 0.20,
        opf_scale_final: float = 0.02,
        verbose: bool = False,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)

        self.N = int(N)
        self.p_exploit = float(np.clip(p_exploit, 0.0, 0.9))
        self.n_exploit = max(1, int(round(self.p_exploit * self.N)))
        self.n_explore = max(1, self.N - self.n_exploit)

        self.n_select = max(1, min(int(n_select), self.N))
        self.n_clones = max(1, int(n_clones))
        self.a_frac = float(a_frac)
        self.r = float(r)
        self.c_threshold = float(c_threshold)

        self.max_gens = int(max_gens)
        self.max_evals = int(max_evals)

        # NEW: initialize exchange params (fix AttributeError)
        self.exchange_interval = int(max(1, exchange_interval))
        self.exchange_k = int(max(1, exchange_k))

        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.spread_window = int(max(1, spread_window))
        self.spread_drop_threshold = float(np.clip(spread_drop_threshold, 0.0, 1.0))

        self.bdt_tau_initial = float(bdt_tau_initial)
        self.bdt_tau_final = float(bdt_tau_final)
        self.bdt_eta_initial = float(bdt_eta_initial)
        self.bdt_eta_final = float(bdt_eta_final)

        self.opf_cols = opf_cols if opf_cols is not None else min(self.dim, 12)
        self.opf_scale_initial = float(opf_scale_initial)
        self.opf_scale_final = float(opf_scale_final)

        self.verbose = verbose
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # domain helpers
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        # book-keeping
        self.eval_count = 0
        self.history_best: List[Tuple[float, np.ndarray]] = []
        self._spread_hist: List[float] = []

        # momentum for exploiters
        self._delta_X_exploit: List[np.ndarray] = []

    # ---------- utils ----------
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ---------- init & partition ----------
    def _init_pop(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
        self._exploit_idx = list(idx_sorted[: self.n_exploit])
        self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]
        self._delta_X_exploit = [np.zeros(self.dim) for _ in range(self.N)]
        return pop

    # ---------- elite helpers ----------
    def _topk(self, pop: List[Antibody], k: int) -> List[Antibody]:
        return sorted(pop, key=lambda ab: ab.affinity, reverse=True)[: max(1, min(k, len(pop)))]

    def _elite_weights(self, elites: List[Antibody]) -> np.ndarray:
        a = np.array([e.affinity for e in elites], dtype=float)
        if np.ptp(a) == 0:
            return np.ones_like(a) / len(a)
        z = (a - a.min()) / (np.ptp(a) + 1e-12)
        w = z / (z.sum() + 1e-12)
        return w

    def _elite_center(self, pop: List[Antibody], k: int) -> np.ndarray:
        elites = self._topk(pop, k)
        W = self._elite_weights(elites)
        X = np.stack([e.x for e in elites], axis=0)
        return (W[:, None] * X).sum(axis=0)

    def _elite_cov_logdet(self, pop: List[Antibody], k: int) -> float:
        elites = self._topk(pop, k)
        X = np.stack([e.x for e in elites], axis=0)
        C = np.cov(X.T) if X.shape[0] > 1 else np.eye(self.dim) * 1e-6
        sign, logdet = np.linalg.slogdet(C + 1e-9 * np.eye(self.dim))
        return float(logdet if sign > 0 else -1e6)

    # ---------- FCSA explorers ----------
    def _select_top(self, pop: List[Antibody], idxs: List[int], k: int) -> List[Antibody]:
        sub = [pop[i] for i in idxs]
        sub_sorted = sorted(sub, key=lambda ab: ab.affinity, reverse=True)
        return sub_sorted[: max(1, min(k, len(sub_sorted)))]

    def _clone_fsca(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        clones: List[Antibody] = []
        for ab in selected:
            ab.S += 1
            a_norm = (ab.affinity - a_min) / denom
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    def _mutate_fsca(self, clones: List[Antibody], gen: int) -> None:
        if not clones:
            return
        scale = (self.sigma_initial - self.sigma_final) * (1 - gen / max(1.0, self.max_gens)) ** self.exponent + self.sigma_final
        affs = np.array([ab.affinity for ab in clones], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        for ab in clones:
            a_norm = (ab.affinity - a_min) / denom
            p = math.exp(-self.r * a_norm)
            mask = (np.random.rand(self.dim) < p).astype(float)
            if mask.any():
                step = np.random.uniform(-scale * self.a_vec, scale * self.a_vec)
                ab.x = ab.x + mask * step
                self._clip(ab.x)
                ab.affinity = self._affinity(ab.x)
                ab.T = 0
                ab.S = 1

    # ---------- BDT (explorers) ----------
    def _bdt_step(self, x: np.ndarray, elites: List[Antibody], tau: float, eta: float) -> np.ndarray:
        if not elites:
            return x.copy()
        y = (x - self.lb) / (self.widths + 1e-12)
        y = np.clip(y, 1e-12, 1 - 1e-12)

        A = np.array([e.affinity for e in elites], dtype=float)
        s = (A - A.max()) / max(1e-12, tau)
        w = np.exp(s)
        w /= (w.sum() + 1e-12)

        E = np.stack([e.x for e in elites], axis=0)
        y_elites = (E - self.lb) / (self.widths + 1e-12)
        y_elites = np.clip(y_elites, 1e-12, 1 - 1e-12)

        y_bar = (w[:, None] * y_elites).sum(axis=0)
        delta = y_bar - y
        y_new = y * np.exp(eta * delta)
        y_new = np.clip(y_new, 1e-12, 1 - 1e-12)

        x_new = self.lb + y_new * self.widths
        self._clip(x_new)
        return x_new

    # ---------- OPF (explorers) ----------
    def _opf_perturb(self, x: np.ndarray, elites: List[Antibody], amp: float) -> np.ndarray:
        if len(elites) < 2 or self.opf_cols <= 0:
            return x.copy()
        X = np.stack([e.x for e in elites], axis=0)
        mu = X.mean(axis=0)
        M = (X - mu)[0: max(2, min(X.shape[0], self.dim)), :]
        M = M + 1e-9 * np.random.randn(*M.shape)
        Q, _ = np.linalg.qr(M.T)
        k = min(self.opf_cols, Q.shape[1])
        Qk = Q[:, :k]
        coeff = np.random.randn(k) * amp
        step = Qk @ coeff
        x_new = x + step
        self._clip(x_new)
        return x_new

    # ---------- exploiters (IICO-like with momentum) ----------
    def _iico_offspring(self, pop: List[Antibody], idxs: List[int], it: int, k_param: float, stagn: int) -> List[Antibody]:
        if not idxs:
            return []
        n = len(idxs)
        fit = np.array([-pop[i].affinity for i in idxs], dtype=float)
        f_best, f_worst = float(np.min(fit)), float(np.max(fit))
        NF = np.ones(n, dtype=float) if f_best == f_worst else (fit - f_worst) / (f_best - f_worst)

        y = round(n * (98 * (1 - it / max(1.0, k_param)) + 2) / 100)
        n_iter = max(1, int(y))
        sorted_ind = np.argsort(NF)
        ES_inds = sorted_ind[:n_iter]

        F1 = np.zeros(self.dim, dtype=float)
        denom = max(1.0, float(n_iter))
        for d in range(self.dim):
            F1[d] = sum(NF[ei] * pop[idxs[ei]].x[d] for ei in ES_inds) / denom

        sigma_iter = (self.sigma_initial - self.sigma_final) * (1 - it / max(1.0, k_param)) ** self.exponent + self.sigma_final
        alpha_iter = sigma_iter

        q_prob = min(0.1 + 0.4 * (stagn / 3.0), 0.5)
        rho = 0.7

        offspring: List[Antibody] = []
        for i_local, idx in enumerate(idxs):
            r = random.random()
            TT_i = F1 * r
            R = np.linalg.norm(pop[idx].x - TT_i)
            E = (TT_i - pop[idx].x) / (R + np.finfo(float).eps)

            self._delta_X_exploit[idx] = rho * self._delta_X_exploit[idx] + (20.0 * alpha_iter * E)

            S = max(1, int(math.floor(2.0 * NF[i_local])))
            for _ in range(S):
                if random.random() < sigma_iter:
                    X_temp = pop[idx].x + alpha_iter * np.random.randn(self.dim)
                    self._clip(X_temp)
                    f = self._objective(X_temp)
                    cand = Antibody(x=X_temp.copy(), affinity=-f, T=0, S=1)
                else:
                    X_temp = pop[idx].x + self._delta_X_exploit[idx]
                    self._clip(X_temp)
                    f = self._objective(X_temp)
                    cand = Antibody(x=X_temp.copy(), affinity=-f, T=0, S=1)

                    if np.random.rand() < q_prob:
                        if it / max(1.0, k_param) > 0.5:
                            lows = np.minimum(self.mid, cand.x)
                            highs = np.maximum(self.mid, cand.x)
                            qrf = lows + np.random.rand(self.dim) * (highs - lows)
                            qrf = np.clip(qrf, self.lb, self.ub)
                            f2 = self._objective(qrf)
                            if f2 < -cand.affinity:
                                cand.x, cand.affinity = qrf, -f2
                        else:
                            opposite = self.lb + self.ub - cand.x
                            lows = np.minimum(self.mid, opposite)
                            highs = np.maximum(self.mid, opposite)
                            qop = lows + np.random.rand(self.dim) * (highs - lows)
                            qop = np.clip(qop, self.lb, self.ub)
                            f2 = self._objective(qop)
                            if f2 < -cand.affinity:
                                cand.x, cand.affinity = qop, -f2

                offspring.append(cand)

        return offspring

    # ---------- forgetting (explorers only) ----------
    def _forget_indices(self, pop: List[Antibody], idxs: List[int]) -> None:
        for i in idxs:
            ab = pop[i]
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ---------- downselect ----------
    def _downselect(self, pop: List[Antibody]) -> List[Antibody]:
        pop.sort(key=lambda ab: ab.affinity, reverse=True)
        return pop[: self.N]

    # ---------- main ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_pop()
        history: List[float] = []

        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagn = 0

        k_param = 0.25 * self.max_evals * (1 + 2) / (2 * max(1, self.N))

        for gen in range(1, self.max_gens + 1):
            if self.eval_count >= self.max_evals:
                break

            for ab in pop:
                ab.T += 1

            if len(self._exploit_idx) != self.n_exploit:
                idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
                self._exploit_idx = list(idx_sorted[: self.n_exploit])
                self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            tau = self.bdt_tau_initial + (self.bdt_tau_final - self.bdt_tau_initial) * (gen / max(1.0, self.max_gens))
            eta = self.bdt_eta_initial + (self.bdt_eta_final - self.bdt_eta_initial) * (gen / max(1.0, self.max_gens))
            opf_amp = self.opf_scale_initial + (self.opf_scale_final - self.opf_scale_initial) * (gen / max(1.0, self.max_gens))

            elites_global = self._topk(pop, self.n_select)
            elite_center = self._elite_center(pop, self.n_select)

            # explorers
            explorers_selected = self._select_top(pop, self._explore_idx, max(1, int(self.n_select * (len(self._explore_idx) / max(1, self.N)))))
            explorer_clones = self._clone_fsca(explorers_selected)
            self._mutate_fsca(explorer_clones, gen)

            for ab in explorer_clones:
                x_bdt = self._bdt_step(ab.x, elites_global, tau=tau, eta=eta)
                f_bdt = self._affinity(x_bdt)
                if f_bdt > ab.affinity:
                    ab.x, ab.affinity, ab.T, ab.S = x_bdt, f_bdt, 0, 1

            for ab in explorer_clones:
                x_opf = self._opf_perturb(ab.x, elites_global, amp=opf_amp * np.mean(self.widths))
                f_opf = self._affinity(x_opf)
                if f_opf > ab.affinity:
                    ab.x, ab.affinity, ab.T, ab.S = x_opf, f_opf, 0, 1

            # exploiters
            exploit_offspring = self._iico_offspring(pop, self._exploit_idx, it=gen, k_param=k_param, stagn=stagn)

            # merge + downselect
            pool = pop + explorer_clones + exploit_offspring
            pop = self._downselect(pool)

            # recompute partitions
            idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
            self._exploit_idx = list(idx_sorted[: self.n_exploit])
            self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # compactness & forgetting
            spread = self._elite_cov_logdet(pop, k=self.n_select)
            self._spread_hist.append(spread)
            spread_ma = np.mean(self._spread_hist[-self.spread_window:]) if self._spread_hist else spread
            drop = (spread_ma - spread) / (abs(spread_ma) + 1e-9)
            if (stagn >= 3) or (drop > self.spread_drop_threshold):
                self._forget_indices(pop, self._explore_idx)

            # periodic exchange
            if (gen % self.exchange_interval) == 0:
                explorers_sorted = sorted([(i, pop[i]) for i in self._explore_idx], key=lambda t: t[1].affinity, reverse=True)
                top_from_explore = [i for i, _ in explorers_sorted[: self.exchange_k]]
                if top_from_explore:
                    exploiters_sorted = sorted([(i, pop[i]) for i in self._exploit_idx], key=lambda t: t[1].affinity)
                    demote = [i for i, _ in exploiters_sorted[: min(len(exploiters_sorted), len(top_from_explore))]]
                    for i_in, i_out in zip(top_from_explore, demote):
                        if i_out in self._exploit_idx:
                            self._exploit_idx.remove(i_out)
                        if i_in in self._explore_idx:
                            self._explore_idx.remove(i_in)
                        self._exploit_idx.append(i_in)
                        self._explore_idx.append(i_out)
                    self._exploit_idx = self._exploit_idx[: self.n_exploit]
                    self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # update best/stagn
            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            self.history_best.append((cur_best_val, cur_best.x.copy()))
            history.append(cur_best_val)

            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagn = 0
            else:
                stagn += 1

            if self.verbose and (gen % 50 == 0 or gen == 1):
                print(f"[gen {gen}] best={best_val:.6e} evals={self.eval_count} spread={spread:.3f} stagn={stagn}")

            if self.eval_count >= self.max_evals:
                break

        diagnostics = {
            "generations_run": len(history),
            "evals_used": self.eval_count,
            "history": history,
            "history_best": self.history_best,
            "final_exploiters": list(self._exploit_idx),
            "elite_spread_logdet": float(self._elite_cov_logdet(pop, k=self.n_select)),
        }
        return best.x.copy(), best_val, diagnostics


# quick smoke test
if __name__ == "__main__":
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    dim = 30
    bounds = [(-100.0, 100.0)] * dim
    algo = HybridFCSA_BDT_OPF(
        func=sphere,
        bounds=bounds,
        N=60,
        p_exploit=0.25,
        n_select=15,
        n_clones=5,
        max_gens=1000,
        max_evals=200_000,
        verbose=True,
        seed=42,
    )
    x_best, f_best, info = algo.minimize()
    print("best f:", f_best)
    print("evals:", info["evals_used"])
