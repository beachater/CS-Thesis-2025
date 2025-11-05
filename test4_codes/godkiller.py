# hybrid_role_partitioned_tsd_adaptive.py
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

class HybridRolePartitionedTSDAdaptive:
    """
    Hybrid role-partitioned (FCSA explorers + IICO exploiters) with:
      - global Temporal Substrate Drift (TSD)
      - adaptive global eta (substrate learning rate)
      - directional substrate partial reset when over-drifted
      - diagnostics: eta_history, drift_norm_history, reset counts/dims
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
        # IICO params (kept as before)
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        # TSD defaults
        eta: float = 0.25,
        lambda_s: float = 0.5,
        rho: float = 0.985,
        drift_interval: int = 1500,
        # adaptive eta controls
        eta_min: float = 0.05,
        eta_max: float = 0.5,
        eta_increase_factor: float = 1.05,
        eta_decrease_factor: float = 0.95,
        stagn_for_eta: int = 5,
        # substrate reset controls
        reset_norm_frac: float = 0.20,   # trigger when ||s|| > reset_norm_frac * mean(width)
        reset_dim_frac: float = 0.20,    # fraction of coordinates to zero on reset
        # diagnostics sampling
        diag_sample_interval: int = 50,
        verbose: bool = False,
    ):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = self.bounds.shape[0]

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

        self.seed = seed
        self.verbose = verbose

        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)

        # TSD
        self.eta = float(eta)
        self.lambda_s = float(lambda_s)
        self.rho = float(rho)
        self.drift_interval = int(drift_interval)
        self._last_decay_eval = 0
        self.s = np.zeros(self.dim, dtype=float)
        self.substrate_updates = 0
        self.substrate_decays = 0

        # adaptive eta
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.eta_inc = float(eta_increase_factor)
        self.eta_dec = float(eta_decrease_factor)
        self.stagn_for_eta = int(stagn_for_eta)

        # reset controls
        self.reset_norm_frac = float(reset_norm_frac)
        self.reset_dim_frac = float(reset_dim_frac)
        self.substrate_reset_count = 0
        self.substrate_reset_dims = 0

        self.diag_sample_interval = int(diag_sample_interval)
        self.eta_history: List[float] = []
        self.drift_norm_history: List[float] = []

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
        self.reset_norm_thresh = float(self.reset_norm_frac * np.mean(self.widths))

        self.history_best: List[Tuple[float, np.ndarray]] = []
        self.eval_count = 0

    # ---------------- basic wrappers ----------------
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        if (self.eval_count - self._last_decay_eval) >= self.drift_interval:
            self.s *= self.rho
            self._last_decay_eval = self.eval_count
            self.substrate_decays += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ---------------- population ----------------
    def _init_pop(self) -> List[Antibody]:
        pop = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
        self._exploit_idx = list(idx_sorted[: self.n_exploit])
        self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]
        self._last_decay_eval = self.eval_count
        return pop

    # ---------------- TSD helpers ----------------
    def _to_drifted(self, x: np.ndarray) -> np.ndarray:
        return x - self.lambda_s * self.s

    def _from_drifted(self, x_adj: np.ndarray) -> np.ndarray:
        y = x_adj + self.lambda_s * self.s
        np.clip(y, self.lb, self.ub, out=y)
        return y

    def _update_substrate(self, dx: np.ndarray) -> None:
        self.s += self.eta * dx
        self.substrate_updates += 1

    def _maybe_reset_substrate(self) -> None:
        norm = float(np.linalg.norm(self.s))
        if norm > self.reset_norm_thresh:
            k = max(1, int(math.ceil(self.reset_dim_frac * self.dim)))
            coords = list(range(self.dim))
            random.shuffle(coords)
            zeroed = coords[:k]
            for j in zeroed:
                if abs(self.s[j]) > 0:
                    self.s[j] = 0.0
                    self.substrate_reset_dims += 1
            self.substrate_reset_count += 1
            if self.verbose:
                print(f"[reset] ||s||={norm:.4g} > {self.reset_norm_thresh:.4g}; zeroed {k} dims")

    # ---------------- FCSA explorers (with TSD) ----------------
    def _select_top(self, pop: List[Antibody], idxs: List[int], k: int) -> List[Antibody]:
        if not idxs:
            return []
        sub = [pop[i] for i in idxs]
        return sorted(sub, key=lambda ab: ab.affinity, reverse=True)[: max(1, min(k, len(sub)))]

    def _clone_fsca(self, selected: List[Antibody]) -> List[Tuple[Antibody, Antibody]]:
        if not selected:
            return []
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        pairs: List[Tuple[Antibody, Antibody]] = []
        for ab in selected:
            a_norm = (ab.affinity - a_min) / denom
            k = max(1, int(round(1 + a_norm * (self.n_clones - 1))))
            for _ in range(k):
                pairs.append((ab, Antibody(x=ab.x.copy(), affinity=ab.affinity, T=ab.T, S=ab.S)))
        return pairs

    def _mutate_variation_explorer(self, clone_pairs: List[Tuple[Antibody, Antibody]]) -> List[Antibody]:
        result: List[Antibody] = []
        if not clone_pairs:
            return result
        affs = np.array([parent.affinity for parent, _ in clone_pairs], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        for parent, clone in clone_pairs:
            a_norm = (parent.affinity - a_min) / denom if denom > 0 else 0.5
            p = math.exp(-self.r * a_norm)
            mask = np.random.rand(self.dim) < p
            x_adj = self._to_drifted(clone.x)
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)
                x_adj = x_adj + mask.astype(float) * step
            x_new = self._from_drifted(x_adj)
            self._clip(x_new)
            f_new = self._objective(x_new)
            if f_new < -parent.affinity - 1e-15:
                dx = x_new - parent.x
                parent.x = x_new
                parent.affinity = -f_new
                parent.T = 0
                parent.S = max(1, parent.S + 1)
                self._update_substrate(dx)
            else:
                clone.x = x_new
                clone.affinity = -f_new
                clone.T = 0
                clone.S = 1
                result.append(clone)
        return result

    def _idp_operator(self, x: np.ndarray, elite: np.ndarray) -> np.ndarray:
        x_adj = self._to_drifted(x)
        elite_adj = self._to_drifted(elite)
        rand_adj = self._to_drifted(self._sample_uniform())
        w = np.random.rand()
        direction = w * (elite_adj - x_adj) + (1 - w) * (rand_adj - x_adj)
        step = 0.1 * self.widths * np.random.randn(self.dim)
        cand_adj = x_adj + direction * np.random.rand() + step
        cand = self._from_drifted(cand_adj)
        self._clip(cand)
        return cand

    # ---------------- IICO exploiters (unchanged flow, local acceptance updates s) ----------------
    def _iico_exploit_generate(self, pop: List[Antibody], idxs: List[int], it: int, k_param: float) -> List[Antibody]:
        if not idxs:
            return []
        n = len(idxs)
        fit = np.array([ -pop[i].affinity for i in idxs ], dtype=float)
        f_best = float(np.min(fit))
        f_worst = float(np.max(fit))
        if f_best == f_worst:
            NF = np.ones(n, dtype=float)
        else:
            NF = (fit - f_worst) / (f_best - f_worst)
        y = round(n * (98 * (1 - it / k_param) + 2) / 100)
        n_iter = max(1, int(y))
        sorted_ind = np.argsort(NF)
        ES_inds = sorted_ind[: n_iter]
        F1 = np.zeros(self.dim, dtype=float)
        for d in range(self.dim):
            sec = 0.0
            for e_local in ES_inds:
                sec += NF[e_local] * pop[idxs[e_local]].x[d]
            F1[d] = sec / max(1, n_iter)
        E_list = []
        for i_local, idx in enumerate(idxs):
            r = random.random()
            TT_i = F1 * r
            R = np.linalg.norm(pop[idx].x - TT_i)
            denom = (R + np.finfo(float).eps)
            E = (TT_i - pop[idx].x) / denom
            E_list.append(E)
        Z = math.exp(-self.beta * it / k_param)
        if Z <= self.gamma:
            self.beta = -math.log(10 * self.gamma) * k_param / max(1, it)
            Z = math.exp(-self.beta * it / k_param)
        sigma_iter = ((k_param - it) / max(1.0, (k_param - 1))) ** self.exponent * (self.sigma_initial - self.sigma_final) + self.sigma_final
        alpha_iter = 10.0 * math.log(max(1e-9, self.M)) * Z
        A_list = [ (20 * alpha_iter * E) for E in E_list ]
        offspring: List[Antibody] = []
        for i_local, idx in enumerate(idxs):
            S = max(0, int(math.floor(0 + (2 - 0) * NF[i_local])))
            S = max(1, S)
            parent = pop[idx]
            for _ in range(S):
                if random.random() < sigma_iter:
                    X_temp = parent.x + alpha_iter * np.random.randn(self.dim)
                    self._clip(X_temp)
                    cand_fit = self._objective(X_temp)
                    if cand_fit < -parent.affinity - 1e-15:
                        dx = X_temp - parent.x
                        parent.x = X_temp
                        parent.affinity = -cand_fit
                        parent.T = 0
                        parent.S = max(1, parent.S + 1)
                        self._update_substrate(dx)
                    else:
                        offspring.append(Antibody(x=np.array(X_temp), affinity=-cand_fit, T=0, S=1))
                else:
                    delta_temp = A_list[i_local]
                    X_temp = parent.x + delta_temp
                    self._clip(X_temp)
                    cand_fit = self._objective(X_temp)
                    if random.random() < 0.1:
                        opposite = self.lb + self.ub - X_temp
                        qop = np.minimum(self.mid, opposite) + np.random.rand(self.dim) * (np.maximum(self.mid, opposite) - np.minimum(self.mid, opposite))
                        qop = np.clip(qop, self.lb, self.ub)
                        qop_fit = self._objective(qop)
                        if qop_fit < cand_fit:
                            X_temp = np.array(qop)
                            cand_fit = qop_fit
                    if cand_fit < -parent.affinity - 1e-15:
                        dx = X_temp - parent.x
                        parent.x = X_temp
                        parent.affinity = -cand_fit
                        parent.T = 0
                        parent.S = max(1, parent.S + 1)
                        self._update_substrate(dx)
                    else:
                        offspring.append(Antibody(x=np.array(X_temp), affinity=-cand_fit, T=0, S=1))
        return offspring

    # ---------------- Rac1 forgetting unchanged ----------------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ---------------- entropy and downselect ----------------
    def _population_entropy(self, pop: List[Antibody], bins: int = 16) -> float:
        X = np.stack([ab.x for ab in pop], axis=0)
        D = X.shape[1]
        ent = 0.0
        for d in range(D):
            x_d = X[:, d]
            data_range = np.max(x_d) - np.min(x_d)
            if data_range < 1e-8:
                ent_d = 0.0
            else:
                hist, _ = np.histogram(x_d, bins=bins, density=True)
                p = hist + 1e-12
                p = p / p.sum()
                ent_d = -np.sum(p * np.log(p + 1e-12))
            ent += ent_d
        ent /= float(D)
        ent = ent / math.log(bins + 1e-12)
        return float(ent)

    def _downselect_global(self, pool: List[Antibody]) -> List[Antibody]:
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return pool[: self.N]

    # ---------------- main loop ----------------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_pop()
        history: List[float] = []
        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagn = 0

        k_param = max(2, 0.25 * self.max_evals * (1 + 2) / (2 * max(1, self.n_exploit)))

        for gen in range(1, self.max_gens + 1):
            if self.eval_count >= self.max_evals:
                break

            for ab in pop:
                ab.T += 1

            if len(self._exploit_idx) != self.n_exploit:
                idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
                self._exploit_idx = list(idx_sorted[: self.n_exploit])
                self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # explorers
            explorers_selected = self._select_top(pop, self._explore_idx, max(1, int(self.n_select * (len(self._explore_idx)/self.N))))
            clone_pairs = self._clone_fsca(explorers_selected)
            explorer_clones = self._mutate_variation_explorer(clone_pairs)

            # IDP in explorers
            if explorers_selected:
                elite = explorers_selected[0].x.copy()
                idp_candidates: List[Antibody] = []
                for ab in explorers_selected[: max(1, len(explorers_selected)//3)]:
                    cand = self._idp_operator(ab.x.copy(), elite)
                    f_cand = self._objective(cand)
                    if f_cand < -ab.affinity - 1e-15:
                        dx = cand - ab.x
                        ab.x = cand
                        ab.affinity = -f_cand
                        ab.T = 0
                        ab.S = max(1, ab.S + 1)
                        self._update_substrate(dx)
                    else:
                        idp_candidates.append(Antibody(x=cand, affinity=-f_cand, T=0, S=1))
                explorer_clones.extend(idp_candidates)

            # exploiters
            exploit_offspring = self._iico_exploit_generate(pop, self._exploit_idx, it=gen, k_param=k_param)

            # merge and downselect
            new_candidates = explorer_clones + exploit_offspring
            pool = pop + new_candidates
            pool = self._downselect_global(pool)

            pop = pool[: self.N]
            idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
            new_exploit = []
            for idx in idx_sorted:
                if len(new_exploit) >= self.n_exploit:
                    break
                new_exploit.append(int(idx))
            self._exploit_idx = list(new_exploit)
            self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # stagnation-gated forgetting unchanged
            entropy = self._population_entropy(pop)
            if (stagn >= 3) or (entropy < 0.20):
                self._forget_in_place(pop)

            # periodic exchange
            if (gen % 5) == 0:
                explorers_sorted = sorted([ (i, pop[i]) for i in self._explore_idx ], key=lambda t: t[1].affinity, reverse=True)
                top_from_explore = [i for i, _ in explorers_sorted[: 2]]
                if top_from_explore:
                    exploiters_sorted = sorted([ (i, pop[i]) for i in self._exploit_idx ], key=lambda t: t[1].affinity)
                    demote_idxs = [i for i, _ in exploiters_sorted[: min(len(exploiters_sorted), len(top_from_explore)) ]]
                    for idx_in, idx_out in zip(top_from_explore, demote_idxs):
                        if idx_out in self._exploit_idx:
                            self._exploit_idx.remove(idx_out)
                        if idx_in in self._explore_idx:
                            self._explore_idx.remove(idx_in)
                        self._exploit_idx.append(idx_in)
                        self._explore_idx.append(idx_out)
                    self._exploit_idx = self._exploit_idx[: self.n_exploit]
                    self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # update best / stagnation
            cur_best = max(pop, key=lambda ab: ab.affinity)
            cur_best_val = -cur_best.affinity
            self.history_best.append((cur_best_val, cur_best.x.copy()))
            history.append(cur_best_val)

            improved = False
            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagn = 0
                improved = True
            else:
                stagn += 1

            # adaptive eta (global)
            if improved:
                self.eta = max(self.eta_min, self.eta * self.eta_dec)
            elif stagn >= self.stagn_for_eta:
                self.eta = min(self.eta_max, self.eta * self.eta_inc)

            # substrate reset if over-drifted
            self._maybe_reset_substrate()

            # diagnostics sampling
            if (gen % self.diag_sample_interval) == 0 or gen == 1:
                self.eta_history.append(self.eta)
                self.drift_norm_history.append(float(np.linalg.norm(self.s)))

        diagnostics = {
            "generations_run": len(history),
            "evals_used": self.eval_count,
            "history": history,
            "history_best": self.history_best,
            "final_exploiters": list(self._exploit_idx),
            "final_entropy": self._population_entropy(pop),
            "substrate_norm": float(np.linalg.norm(self.s)),
            "substrate_updates": int(self.substrate_updates),
            "substrate_decays": int(self.substrate_decays),
            "substrate_reset_count": int(self.substrate_reset_count),
            "substrate_reset_dims": int(self.substrate_reset_dims),
            "eta": float(self.eta),
            "eta_history": list(self.eta_history),
            "drift_norm_history": list(self.drift_norm_history),
        }
        return best.x.copy(), best_val, diagnostics

# demo block
if __name__ == "__main__":
    def ackley(x: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * math.pi
        d = x.size
        sum_sq = np.sum(x * x)
        sum_cos = np.sum(np.cos(c * x))
        return -a * math.exp(-b * math.sqrt(sum_sq / d)) - math.exp(sum_cos / d) + a + math.e

    dim = 2
    bounds = [(-5.0, 5.0)] * dim

    opt = HybridRolePartitionedTSDAdaptive(
        func=ackley,
        bounds=bounds,
        N=60,
        max_gens=1000,
        max_evals=350_000,
        seed=123,
        eta=0.25,
        lambda_s=0.5,
        rho=0.985,
        drift_interval=1500,
        verbose=False
    )
    xbest, fbest, info = opt.minimize()
    print("best f", fbest, "evals", info["evals_used"],
          "||s||", info["substrate_norm"],
          "updates", info["substrate_updates"],
          "decays", info["substrate_decays"],
          "resets", info["substrate_reset_count"],
          "reset_dims", info["substrate_reset_dims"],
          "eta", info["eta"])
