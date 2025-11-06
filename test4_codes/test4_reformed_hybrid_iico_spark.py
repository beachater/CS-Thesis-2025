# hybrid_fcsa_iico.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np

@dataclass
class Antibody:
    x: np.ndarray
    affinity: float          # higher is better (we maximize affinity = -objective)
    T: int = 0               # survival time / age
    S: int = 0               # selection count / memory

class HybridFCSA_IICO:
    """
    Hybrid anchored on FCSA (explorers) with IICO-style exploiters.
    Includes a small IICO-style 'spark' perturbation. No TSD.
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
        progress: Optional[Callable] = None,
        # exchange / gating
        exchange_interval: int = 5,
        exchange_k: int = 2,
        stagn_thresh: int = 3,
        entropy_frac_threshold: float = 0.20,
        # IICO schedule
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
        # spark
        spark_prob: float = 0.04,
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

        self.seed = seed
        self.verbose = verbose

        self.exchange_interval = int(max(1, exchange_interval))
        self.exchange_k = int(max(1, exchange_k))
        self.stagn_thresh = int(max(1, stagn_thresh))
        self.entropy_frac_threshold = float(np.clip(entropy_frac_threshold, 0.0, 1.0))

        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.spark_prob = float(spark_prob)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.eval_count = 0
        self.history_best: List[Tuple[float, np.ndarray]] = []
        self.progress = progress

    # ---- utilities ----
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return - self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ---- init ----
    def _init_pop(self) -> List[Antibody]:
        pop: List[Antibody] = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        sorted_idx = np.argsort([ab.affinity for ab in pop])[::-1]
        self._exploit_idx = list(sorted_idx[: self.n_exploit])
        self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]
        return pop

    # ---- FCSA explorers ----
    def _select_top(self, pop: List[Antibody], idxs: List[int], k: int) -> List[Antibody]:
        if not idxs:
            return []
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

    def _mutate_variation(self, clones: List[Antibody]) -> None:
        if not clones:
            return
        affs = np.array([ab.affinity for ab in clones], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        for ab in clones:
            a_norm = (ab.affinity - a_min) / denom if denom > 0 else 0.5
            p = math.exp(-self.r * a_norm)
            mask = np.random.rand(self.dim) < p
            if np.any(mask):
                step = np.random.uniform(-self.a_vec, self.a_vec)
                ab.x = ab.x + mask.astype(float) * step
                self._clip(ab.x)
                ab.affinity = self._affinity(ab.x)
                ab.T = 0
                ab.S = 1

    # optional directional operator (IDP)
    def _idp_operator(self, x: np.ndarray, elite: np.ndarray) -> np.ndarray:
        rand = self._sample_uniform()
        w = np.random.rand()
        direction = w * (elite - x) + (1 - w) * (rand - x)
        step = 0.1 * self.widths * np.random.randn(self.dim)
        cand = x + direction * np.random.rand() + step
        self._clip(cand)
        return cand

    # ---- IICO exploiters ----
    def _spark(self, x: np.ndarray) -> np.ndarray:
        z = np.random.rand(self.dim)
        z = 3.99 * z * (1 - z)
        kick = (z - 0.5) * 0.5 * self.widths
        y = x + kick
        self._clip(y)
        return y

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
        A_list = [ (20 * alpha_iter * E) for E in E_list]

        offspring: List[Antibody] = []
        for i_local, idx in enumerate(idxs):
            S = max(1, int(math.floor(0 + (2 - 0) * NF[i_local])))
            S = max(1, S)
            for _ in range(S):
                if random.random() < sigma_iter:
                    X_temp = pop[idx].x + alpha_iter * np.random.randn(self.dim)
                    self._clip(X_temp)
                else:
                    X_temp = pop[idx].x + A_list[i_local]
                    self._clip(X_temp)
                # apply spark occasionally
                if random.random() < self.spark_prob:
                    X_temp = self._spark(X_temp)
                cand_fit = self._objective(X_temp)
                offspring.append(Antibody(x=np.array(X_temp), affinity=-cand_fit, T=0, S=1))
        return offspring

    # ---- forgetting ----
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ---- entropy ----
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

    # ---- downselect ----
    def _downselect_global(self, pool: List[Antibody]) -> List[Antibody]:
        pool.sort(key=lambda ab: ab.affinity, reverse=True)
        return pool[: self.N]

    # ---- main optimize loop ----
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_pop()
        history: List[float] = []
        best_ab = max(pop, key=lambda ab: ab.affinity)
        best_val = -best_ab.affinity
        best_tracker = {"best_f": best_val, "best_x": best_ab.x.copy()}

        stagn = 0
        k_param = max(2, 0.25 * self.max_evals * (1 + 2) / (2 * max(1, self.n_exploit)))

        for gen in range(1, self.max_gens + 1):
            if self.eval_count >= self.max_evals:
                break

            # age
            for ab in pop:
                ab.T += 1

            if len(self._exploit_idx) != self.n_exploit:
                idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
                self._exploit_idx = list(idx_sorted[: self.n_exploit])
                self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # explorers (FCSA)
            explorers_selected = self._select_top(pop, self._explore_idx, max(1, int(self.n_select * (len(self._explore_idx) / self.N))))
            explorer_clones = self._clone_fsca(explorers_selected)
            self._mutate_variation(explorer_clones)

            # exploiters (IICO-like)
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

            # stagnation-gated forgetting
            entropy = self._population_entropy(pop)
            if (stagn >= self.stagn_thresh) or (entropy < self.entropy_frac_threshold):
                self._forget_in_place(pop)

            # periodic exchange
            if (gen % self.exchange_interval) == 0:
                explorers_sorted = sorted([(i, pop[i]) for i in self._explore_idx], key=lambda t: t[1].affinity, reverse=True)
                top_from_explore = [i for i, _ in explorers_sorted[: self.exchange_k]]
                if top_from_explore:
                    exploiters_sorted = sorted([(i, pop[i]) for i in self._exploit_idx], key=lambda t: t[1].affinity)
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

            # Report progress if logger provided
            if self.progress is not None:
                try:
                    # Convert affinities to fitness (minimize)
                    fitness = np.array([-ab.affinity for ab in pop])
                    self.progress(
                        gen=gen-1,  # 0-based gen index
                        pop=np.array([ab.x for ab in pop]),
                        fitness=fitness,
                        best_fitness=cur_best_val,
                        gbest=cur_best.x,
                        evals=self.eval_count
                    )
                except Exception:
                    pass

            if cur_best_val + 1e-12 < best_val:
                best_val = cur_best_val
                best_ab = Antibody(x=cur_best.x.copy(), affinity=cur_best.affinity, T=cur_best.T, S=cur_best.S)
                stagn = 0
            else:
                stagn += 1

        diagnostics = {
            "generations_run": len(history),
            "evals_used": self.eval_count,
            "history": history,
            "history_best": self.history_best,
            "final_exploiters": list(self._exploit_idx),
            "final_entropy": self._population_entropy(pop),
        }
        return best_ab.x.copy(), best_val, diagnostics

# ---- demo ----
if __name__ == "__main__":
    def ackley(x: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * math.pi
        d = x.size
        sum_sq = np.sum(x * x)
        sum_cos = np.sum(np.cos(c * x))
        return -a * math.exp(-b * math.sqrt(sum_sq / d)) - math.exp(sum_cos / d) + a + math.e

    dim = 2
    bounds = [(-5.0, 5.0)] * dim

    opt = HybridFCSA_IICO(
        func=ackley,
        bounds=bounds,
        N=60,
        p_exploit=0.25,
        n_select=15,
        n_clones=5,
        max_gens=400,
        max_evals=60_000,
        seed=123,
        spark_prob=0.06
    )
    xbest, fbest, info = opt.minimize()
    print("best f", fbest, "evals", info["evals_used"])
