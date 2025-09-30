# hybrid_role_partitioned.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np

@dataclass
class Antibody:
    x: np.ndarray
    affinity: float          # higher is better (we maximize affinity)
    T: int = 0               # survival time
    S: int = 0               # selection count / strength

class HybridRolePartitioned:
    """
    Role-partitioned hybrid of FCSA (explorers) + IICO-style exploitation (exploiters)
    with stagnation-gated Rac1 forgetting.

    - explorers: run FCSA pipeline (broad exploration, forgetting anchored)
    - exploiters: run simplified IICO-style pipeline (adaptive cloning/exploitation)
    - periodic exchange: move best from explorers -> exploiters, demote worst exploiters
    - forgetting (Rac1) only runs when stagnation >= stagn_thresh OR population entropy < entropy_thresh
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        N: int = 60,
        p_exploit: float = 0.25,          # fraction of population assigned to exploiters
        n_select: int = 15,
        n_clones: int = 5,
        a_frac: float = 0.15,
        r: float = 2.0,                   # FCSA mutation exponent
        c_threshold: float = 3.0,         # Rac1 forgetting threshold
        max_gens: int = 1000,
        max_evals: int = 350_000,
        seed: Optional[int] = 42,
        # exchange params
        exchange_interval: int = 5,
        exchange_k: int = 2,
        # stagnation-gated forgetting params
        stagn_thresh: int = 3,
        entropy_frac_threshold: float = 0.20,
        # IICO-ish params for exploiters (simplified)
        sigma_initial: float = 0.5,
        sigma_final: float = 0.1,
        exponent: float = 2.0,
        beta: float = 100.0,
        gamma: float = 1e-19,
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

        # exchange and gating
        self.exchange_interval = int(max(1, exchange_interval))
        self.exchange_k = int(max(1, exchange_k))
        self.stagn_thresh = int(max(1, stagn_thresh))
        self.entropy_frac_threshold = float(np.clip(entropy_frac_threshold, 0.0, 1.0))

        # IICO-ish schedule params (used in exploiters)
        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # domain helpers
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        # self.M = float(np.mean(self.widths) / 2.0)
        # for IICO-style M
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []
        self.eval_count = 0

    # ----------------- utilities -----------------
    def _objective(self, x: np.ndarray) -> float:
        self.eval_count += 1
        return float(self.func(x))

    def _affinity(self, x: np.ndarray) -> float:
        return -self._objective(x)

    def _clip(self, x: np.ndarray) -> None:
        np.clip(x, self.lb, self.ub, out=x)

    def _sample_uniform(self) -> np.ndarray:
        return self.lb + np.random.rand(self.dim) * self.widths

    # ----------------- population init -----------------
    def _init_pop(self) -> List[Antibody]:
        pop = []
        for _ in range(self.N):
            x = self._sample_uniform()
            pop.append(Antibody(x=x, affinity=self._affinity(x), T=0, S=0))
        # partition indices: pick exploiters as top by affinity initially to seed exploitation
        pop_sorted_indices = np.argsort([ab.affinity for ab in pop])[::-1]  # best->worst
        exploit_idx = list(pop_sorted_indices[: self.n_exploit])
        explore_idx = [i for i in range(self.N) if i not in exploit_idx]
        self._exploit_idx = exploit_idx
        self._explore_idx = explore_idx
        return pop

    # ----------------- FCSA helpers for explorers -----------------
    def _select_top(self, pop: List[Antibody], idxs: List[int], k: int) -> List[Antibody]:
        if not idxs:
            return []
        sub = [pop[i] for i in idxs]
        sub_sorted = sorted(sub, key=lambda ab: ab.affinity, reverse=True)
        return sub_sorted[: max(1, min(k, len(sub_sorted)))]

    def _clone_fsca(self, selected: List[Antibody]) -> List[Antibody]:
        # identical to FCSA._clone but local
        if not selected:
            return []
        affs = np.array([ab.affinity for ab in selected], dtype=float)
        a_min, a_max = float(affs.min()), float(affs.max())
        denom = max(a_max - a_min, 1e-12)
        clones = []
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

     # Immune Directional Perturbation (IDP)
    def _idp_operator(self, x: np.ndarray, elite: np.ndarray) -> np.ndarray:
        # Pick a random point in the domain
        rand = self.lb + np.random.rand(self.dim) * self.widths
        
        # Direction = weighted combo of elite and random
        w = np.random.rand()
        direction = w * (elite - x) + (1 - w) * (rand - x)
        
        # Step size proportional to domain scale
        step = 0.1 * self.widths * np.random.randn(self.dim)
        cand = x + direction * np.random.rand() + step

        # Clip to bounds
        self._clip(cand)
        return cand

    # ----------------- IICO-like helpers for exploiters -----------------
    def _iico_exploit_generate(self, pop: List[Antibody], idxs: List[int], it: int, k_param: float) -> List[Antibody]:
        """
        Simplified IICO offspring generation for exploiters subpopulation.
        - idxs : indices of exploiters within pop
        - it   : iteration counter (used in schedule)
        - k_param : derived constant similar to iico's k (controls schedule)
        Returns list of offspring Antibody objects (not integrated into pop).
        """
        if not idxs:
            return []
        n = len(idxs)
        # extract fitness list (objective), lower is better
        fit = np.array([ -pop[i].affinity for i in idxs ], dtype=float)
        f_best = float(np.min(fit))
        f_worst = float(np.max(fit))
        # normalised fitness NF (following IICO style: lower objective -> larger NF)
        if f_best == f_worst:
            NF = np.ones(n, dtype=float)
        else:
            NF = (fit - f_worst) / (f_best - f_worst)  # NF in ... note same transform as iico
        # selection count n_iter
        y = round(n * (98 * (1 - it / k_param) + 2) / 100)
        n_iter = max(1, int(y))
        sorted_ind = np.argsort(NF)  # ascending NF (IICO used this)
        ES_inds = sorted_ind[: n_iter]   # indices within exploiters list
        # compute elite center F1 (in original iico they used NF weights)
        F1 = np.zeros(self.dim, dtype=float)
        for d in range(self.dim):
            sec = 0.0
            for e_local in ES_inds:
                sec += NF[e_local] * pop[idxs[e_local]].x[d]
            F1[d] = sec / max(1, n_iter)

        # compute direction vectors E and amplitude A
        E_list = []
        for i_local, idx in enumerate(idxs):
            r = random.random()
            TT_i = F1 * r
            R = np.linalg.norm(pop[idx].x - TT_i)
            # avoid divide by zero:
            denom = (R + np.finfo(float).eps)
            E = (TT_i - pop[idx].x) / denom
            E_list.append(E)
        # amplitude A akin to IICO:
        # compute current schedule alpha_iter
        # derive k param passed to function; use same formula as IICO caller can compute k
        Z = math.exp(-self.beta * it / k_param)
        if Z <= self.gamma:
            self.beta = -math.log(10 * self.gamma) * k_param / max(1, it)
            Z = math.exp(-self.beta * it / k_param)
        sigma_iter = ((k_param - it) / max(1.0, (k_param - 1))) ** self.exponent * (self.sigma_initial - self.sigma_final) + self.sigma_final
        alpha_iter = 10.0 * math.log(max(1e-9, self.M)) * Z
        A_list = [ (20 * alpha_iter * E) for E in E_list ]

        # now generate offsprings: for each exploiter we may create several offsprings proportional to NF
        offspring = []
        for i_local, idx in enumerate(idxs):
            S = max(0, int(math.floor(0 + (2 - 0) * NF[i_local])))  # simplify s_min/s_max mapping; s_max=2
            S = max(1, S)  # produce at least one candidate per exploiter
            for _ in range(S):
                if random.random() < sigma_iter:
                    X_temp = pop[idx].x + alpha_iter * np.random.randn(self.dim)
                    self._clip(X_temp)
                    cand_fit = self._objective(X_temp)
                    offspring.append(Antibody(x=np.array(X_temp), affinity=-cand_fit, T=0, S=1))
                else:
                    delta_temp = (np.random.rand(self.dim) * 0.0) + A_list[i_local]
                    X_temp = pop[idx].x + delta_temp
                    self._clip(X_temp)
                    cand_fit = self._objective(X_temp)
                    # quasi-opposite/reflection check as a small local step (approx)
                    # implement a simplified quasi-opposite: mix with mid/opposite randomly
                    if random.random() < 0.1:
                        opposite = self.lb + self.ub - X_temp
                        qop = np.minimum(self.mid, opposite) + np.random.rand(self.dim) * (np.maximum(self.mid, opposite) - np.minimum(self.mid, opposite))
                        qop = np.clip(qop, self.lb, self.ub)
                        qop_fit = self._objective(qop)
                        if qop_fit < cand_fit:
                            X_temp = np.array(qop)
                            cand_fit = qop_fit
                    offspring.append(Antibody(x=np.array(X_temp), affinity=-cand_fit, T=0, S=1))
        return offspring

    # ----------------- forgetting -----------------
    def _forget_in_place(self, pop: List[Antibody]) -> None:
        # Rac1-style forgetting applied to full population
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ----------------- entropy -----------------
    def _population_entropy(self, pop: List[Antibody], bins: int = 16) -> float:
        # compute per-dim entropy and return normalized mean entropy
        X = np.stack([ab.x for ab in pop], axis=0)
        D = X.shape[1]
        ent = 0.0
        for d in range(D):
            x_d = X[:, d]
            data_range = np.max(x_d) - np.min(x_d)
            if data_range == 0.0:
                ent_d = 0.0  # No diversity in this dimension
            else:
                hist, _ = np.histogram(x_d, bins=bins, density=True)
                p = hist + 1e-12
                p = p / p.sum()
                ent_d = -np.sum(p * np.log(p + 1e-12))
            ent += ent_d
        ent /= float(D)
        # normalize by log(bins)
        ent = ent / math.log(bins + 1e-12)
        return float(ent)

    # ----------------- downselect -----------------
    def _downselect_global(self, pop: List[Antibody]) -> List[Antibody]:
        pop.sort(key=lambda ab: ab.affinity, reverse=True)
        return pop[: self.N]

    # ----------------- main optimize loop -----------------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._init_pop()
        history = []
        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagn = 0

        # compute IICO-like k parameter (rough analog)
        k_param = max(2, 0.25 * self.max_evals * (1 + 2) / (2 * max(1, self.n_exploit)))  # heuristic

        for gen in range(1, self.max_gens + 1):
            if self.eval_count >= self.max_evals:
                break

            # age
            for ab in pop:
                ab.T += 1

            # update partition sizes (keep same indices but adjust if counts changed)
            # ensure index lists consistent
            if len(self._exploit_idx) != self.n_exploit:
                # recompute exploit/explore assignment by best affinity
                idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
                self._exploit_idx = list(idx_sorted[: self.n_exploit])
                self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # ---- explorers pipeline (FCSA) ----
            explorers_selected = self._select_top(pop, self._explore_idx, max(1, int(self.n_select * (len(self._explore_idx)/self.N))))
            explorer_clones = self._clone_fsca(explorers_selected)
            self._mutate_variation(explorer_clones)

            # ---- exploiters pipeline (IICO-like) ----
            exploit_offspring = self._iico_exploit_generate(pop, self._exploit_idx, it=gen, k_param=k_param)

            # ---- merge offspring into a candidate pool ----
            new_candidates = explorer_clones + exploit_offspring
            pool = pop + new_candidates
            pool = self._downselect_global(pool)

            # assign new pop and recompute partitions indices (keep exploiters as indices of best)
            pop = pool[: self.N]
            idx_sorted = np.argsort([ab.affinity for ab in pop])[::-1]
            # prefer preserving previous exploiters if still good; otherwise choose top-n_exploit
            new_exploit = []
            for idx in idx_sorted:
                if len(new_exploit) >= self.n_exploit:
                    break
                new_exploit.append(int(idx))
            self._exploit_idx = list(new_exploit)
            self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # ---- stagnation-gated forgetting ----
            entropy = self._population_entropy(pop)
            # decide threshold dynamic: compare to historical average entropy if available
            # here we use fraction of max entropy (which is ~1.0 after normalization)
            if (stagn >= self.stagn_thresh) or (entropy < self.entropy_frac_threshold):
                self._forget_in_place(pop)
            # else skip forgetting to allow IICO exploitation to finish fine-tuning

            # ---- periodic exchange (small seeds) ----
            if (gen % self.exchange_interval) == 0:
                # move top from explorers -> exploiters (seed)
                explorers_sorted = sorted([ (i, pop[i]) for i in self._explore_idx ], key=lambda t: t[1].affinity, reverse=True)
                top_from_explore = [i for i, _ in explorers_sorted[: self.exchange_k ]]
                if top_from_explore:
                    # demote worst exploiters
                    exploiters_sorted = sorted([ (i, pop[i]) for i in self._exploit_idx ], key=lambda t: t[1].affinity)
                    demote_idxs = [i for i, _ in exploiters_sorted[: min(len(exploiters_sorted), len(top_from_explore)) ]]
                    # swap roles (preserve population entries but exchange indices)
                    for idx_in, idx_out in zip(top_from_explore, demote_idxs):
                        # swap index membership
                        if idx_out in self._exploit_idx:
                            self._exploit_idx.remove(idx_out)
                        if idx_in in self._explore_idx:
                            self._explore_idx.remove(idx_in)
                        self._exploit_idx.append(idx_in)
                        self._explore_idx.append(idx_out)
                    # trim if oversize
                    self._exploit_idx = self._exploit_idx[: self.n_exploit]
                    self._explore_idx = [i for i in range(self.N) if i not in self._exploit_idx]

            # ---- update best / stagnation ----
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

        diagnostics = {
            "generations_run": len(history),
            "evals_used": self.eval_count,
            "history": history,
            "history_best": self.history_best,
            "final_exploiters": list(self._exploit_idx),
            "final_entropy": self._population_entropy(pop),
        }
        return best.x.copy(), best_val, diagnostics
