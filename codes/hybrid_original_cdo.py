from dataclasses import dataclass

@dataclass
class Antibody:
    x: np.ndarray
    affinity: float
    T: int = 0
    S: int = 0
# hybrid_csa_with_operators.py
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


class HybridCSAOperators:
    """
    Hybrid Clonal Selection with 4 alternative operators that replace DE:
      operator: 'tfo'  -> Topological Folding Operator
                'cdo'  -> Cognitive Dissonance Operator
                'rso'  -> Resonance Synchronization Operator
                'ibp'  -> Information Bottleneck Perturbation
                'base' -> original (existing diff-like scheme preserved)

    The rest of the CSA pipeline (cloning, QOBL/QRBL, forgetting, selection) is preserved.
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
        # Operator selection and hyperparams
        operator: str = "tfo",  # 'tfo'|'cdo'|'rso'|'ibp'|'base'
        # TFO params
        tfo_fold_prob: float = 0.25,
        tfo_fold_strength: float = 0.6,
        # CDO params
        cdo_init_delta: float = 0.5,  # initial dissonance factor
        cdo_eta: float = 0.05,  # adaptation step for delta
        cdo_lr_opt: float = 0.9,
        cdo_lr_pess: float = 0.2,
        # RSO params
        rso_amp: Optional[float] = None,  # amplitude (if None compute from widths)
        rso_freq_min: float = 0.1,
        rso_freq_max: float = 3.0,
        rso_sync_threshold: float = 0.6,
        rso_desynch: float = 0.01,
        # IBP params
        ibp_k_frac: float = 0.05,
        ibp_recon_noise: float = 0.01,
        ibp_update_window: int = 10,
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

        self.sigma_initial = float(sigma_initial)
        self.sigma_final = float(sigma_final)
        self.exponent = float(exponent)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.max_stagnation = int(max_stagnation)

        self.seed = seed
        self.verbose = verbose

        # operator
        # assert operator in ("tfo", "cdo", "rso", "ibp", "base")
        self.operator = "cdo"

        # TFO
        self.tfo_fold_prob = float(tfo_fold_prob)
        self.tfo_fold_strength = float(tfo_fold_strength)

        # CDO
        self.cdo_init_delta = float(cdo_init_delta)
        self.cdo_eta = float(cdo_eta)
        self.cdo_lr_opt = float(cdo_lr_opt)
        self.cdo_lr_pess = float(cdo_lr_pess)

        # RSO
        self.rso_amp = rso_amp  # if None compute
        self.rso_freq_min = float(rso_freq_min)
        self.rso_freq_max = float(rso_freq_max)
        self.rso_sync_threshold = float(rso_sync_threshold)
        self.rso_desynch = float(rso_desynch)

        # IBP
        self.ibp_k_frac = float(ibp_k_frac)
        self.ibp_recon_noise = float(ibp_recon_noise)
        self.ibp_update_window = int(ibp_update_window)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Domain info
        self.lb = self.bounds[:, 0].copy()
        self.ub = self.bounds[:, 1].copy()
        self.widths = self.ub - self.lb
        self.mid = (self.lb + self.ub) / 2.0
        self.a_vec = self.a_frac * self.widths
        self.M = float(np.mean(self.widths) / 2.0)

        self.history_best: List[Tuple[float, np.ndarray]] = []

        # operator bookkeeping
        # CDO per-individual hypotheses: store optimistic/pessimistic per population index
        self._cdo_opt = None
        self._cdo_pess = None
        self._cdo_delta = None

        # RSO oscillator states
        self._rso_phase = None
        self._rso_freq = None
        if self.rso_amp is None:
            self._rso_amp = 0.05 * np.mean(self.widths)
        else:
            self._rso_amp = float(self.rso_amp)

        # IBP PCA projection cache
        self._ibp_P = None  # projection matrix (D x k)
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

    def _qobl(self, x: np.ndarray) -> np.ndarray:
        opposite = self.lb + self.ub - x
        lows = np.minimum(self.mid, opposite)
        highs = np.maximum(self.mid, opposite)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _qrbl(self, x: np.ndarray) -> np.ndarray:
        lows = np.minimum(self.mid, x)
        highs = np.maximum(self.mid, x)
        return lows + np.random.rand(self.dim) * (highs - lows)

    def _forget_in_place(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S <= 0:
                activity = float("inf") if ab.T > 0 else 0.0
            else:
                activity = ab.T / float(ab.S)
            if activity > self.c_threshold:
                x = self._sample_uniform_point()
                pop[i] = Antibody(x=x, affinity=self._affinity(x), T=0, S=0)

    # ---------- ESD helpers: entropy/subspace if desired ----------
    def _compute_entropy_per_dim(self, pop_array: np.ndarray, bins: int = 20) -> np.ndarray:
        N, D = pop_array.shape
        ent = np.zeros(D)
        for d in range(D):
            hist, _ = np.histogram(pop_array[:, d], bins=bins, density=True)
            p = hist + 1e-12
            p = p / p.sum()
            ent[d] = -np.sum(p * np.log(p))
        return ent

    def _select_active_dims_by_entropy(self, ent: np.ndarray, frac: float = 0.6) -> np.ndarray:
        D = len(ent)
        k = max(1, int(np.ceil(frac * D)))
        idx = np.argsort(-ent)[:k]
        mask = np.zeros(D, dtype=bool)
        mask[idx] = True
        return mask

    # ---------- Operator implementations ----------

    # 1) Topological Folding Operator (TFO)
    def _tfo_mutation(self, base: np.ndarray, pop: List[Antibody]) -> np.ndarray:
        """
        For a base vector (ab.x), apply random folding operations between dimension pairs.
        Folding maps two dims (i,j) -> combine values across a fold plane (midpoint).
        Conservative default: only fold a fraction of dims per mutation.
        """
        x = base.copy()
        D = self.dim
        # number of fold attempts
        for d in range(D):
            if np.random.rand() < self.tfo_fold_prob:
                j = np.random.randint(0, D)
                if j == d:
                    continue
                # fold pair (d,j)
                md = self.mid[d]
                mj = self.mid[j]
                # relative positions from mid
                rd = x[d] - md
                rj = x[j] - mj
                lam = self.tfo_fold_strength
                # folded values: linear blend of mirrored coordinates
                new_d = md + lam * rd + (1 - lam) * rj
                new_j = mj + lam * rj + (1 - lam) * rd
                x[d] = new_d
                x[j] = new_j
        # small jitter to keep exploration
        x += np.random.normal(scale=0.01 * np.mean(self.widths), size=D)
        return x

    # 2) Cognitive Dissonance Operator (CDO)
    def _cdo_init(self, pop: List[Antibody]) -> None:
        # initialize optimistic/pessimistic hypotheses and delta per-pop
        X = np.stack([ab.x for ab in pop], axis=0)
        self._cdo_opt = X.copy()
        self._cdo_pess = X.copy()
        self._cdo_delta = np.ones(self.N) * self.cdo_init_delta

    def _cdo_mutation(self, idx_pop: int, base: np.ndarray, pop: List[Antibody], stagnation: int) -> np.ndarray:
        """
        Each population member maintains two hypotheses:
          - optimistic: aggressive step toward elite center
          - pessimistic: conservative/noisy estimate
        We interpolate with dissonance factor delta and adapt delta by stagnation/improvement signals.
        idx_pop: index in population corresponding to the source of clone (best-effort mapping)
        """
        # map idx_pop into [0,N-1]
        i = idx_pop % self.N
        if self._cdo_opt is None:
            # lazy init if not yet
            X = np.stack([ab.x for ab in pop], axis=0)
            self._cdo_opt = X.copy()
            self._cdo_pess = X.copy()
            self._cdo_delta = np.ones(self.N) * self.cdo_init_delta

        elite = self._elite_center(pop, max(1, int(0.2 * len(pop))))
        # optimistic update: large step toward elite + noise
        opt_lr = self.cdo_lr_opt
        opt_next = self._cdo_opt[i] + opt_lr * (elite - self._cdo_opt[i]) + np.random.randn(self.dim) * 0.01 * np.mean(self.widths)
        # pessimistic update: small local exploration (conservative)
        pess_lr = self.cdo_lr_pess
        pess_next = self._cdo_pess[i] + pess_lr * (np.random.randn(self.dim) * 0.02 * np.mean(self.widths))
        # adapt delta: if stagnation high, increase dissonance (encourage oscillation); if improving, reduce
        delta = self._cdo_delta[i]
        if stagnation > max(1, self.max_stagnation // 2):
            delta = min(1.0, delta + self.cdo_eta)
        else:
            delta = max(0.0, delta - self.cdo_eta)
        # store back
        self._cdo_opt[i] = opt_next
        self._cdo_pess[i] = pess_next
        self._cdo_delta[i] = delta
        # candidate: interpolation between opt and pess but add small random shift
        cand = (1 - delta) * opt_next + delta * pess_next + np.random.normal(scale=0.005 * np.mean(self.widths), size=self.dim)
        return cand

    # 3) Resonance Synchronization Operator (RSO)
    def _rso_init(self, pop: List[Antibody]) -> None:
        # initialize phases and frequencies per-pop
        self._rso_phase = np.random.rand(self.N) * 2 * math.pi
        self._rso_freq = np.random.uniform(self.rso_freq_min, self.rso_freq_max, size=self.N)

    def _rso_mutation(self, idx_pop: int, base: np.ndarray, pop: List[Antibody], generation: int) -> np.ndarray:
        """
        Each individual is an oscillator; candidate = base + A*sin(omega * t + phi) (applied on selected dims).
        When local fitness correlation indicates beneficial alignment, partially synchronize frequencies/phase toward elites.
        """
        i = idx_pop % self.N
        if self._rso_phase is None or self._rso_freq is None:
            self._rso_init(pop)
        # oscillatory perturbation
        t = float(generation)
        omega = self._rso_freq[i]
        phi = self._rso_phase[i]
        osc = self._rso_amp * np.sin(omega * t + phi)
        # apply oscillation to a subset of dims (higher-entropy dims)
        pop_array = np.stack([ab.x for ab in pop], axis=0)
        ent = self._compute_entropy_per_dim(pop_array)
        active = self._select_active_dims_by_entropy(ent, frac=0.4)
        cand = base.copy()
        cand[active] += osc  # scalar osc added to all active dims
        # measure simple local correlation: if many neighbors improved when aligned, move freq closer to elite
        elite = self._elite_center(pop, max(1, int(0.1 * len(pop))))
        # compute simple alignment score (cosine similarity) between base->elite and oscillation direction
        v = elite - base
        if np.linalg.norm(v) > 1e-12:
            cos = np.dot(v[active], np.ones(active.sum())) / (np.linalg.norm(v[active]) * math.sqrt(active.sum()) + 1e-12)
        else:
            cos = 0.0
        # if cos high, nudge frequency/phase toward a small adjustment (synchronization)
        if cos > self.rso_sync_threshold:
            # synchronize a bit (reduce freq difference to median)
            target_freq = np.median(self._rso_freq)
            self._rso_freq[i] += 0.01 * (target_freq - self._rso_freq[i])
            self._rso_phase[i] += 0.05 * (0.0 - self._rso_phase[i])  # nudge toward 0-phase (arbitrary)
        else:
            # desynchronize slightly
            self._rso_freq[i] += self.rso_desynch * (np.random.rand() - 0.5)
            self._rso_phase[i] += self.rso_desynch * (np.random.rand() - 0.5)
        # small jitter
        cand += np.random.normal(scale=0.002 * np.mean(self.widths), size=self.dim)
        return cand

    # 4) Information Bottleneck Perturbation (IBP)
    def _ibp_update_projection(self, pop_array: np.ndarray) -> None:
        """
        Compute a linear bottleneck projection using top-k PCA on recent population.
        We store projection P (D x k) and mean.
        """
        D = self.dim
        k = max(2, int(max(2, math.ceil(self.ibp_k_frac * D))))
        # center
        mean = pop_array.mean(axis=0)
        Xc = pop_array - mean
        # compute top-k PCA using SVD (economy)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Vt[:k].T  # D x k
        except Exception:
            # fallback random projection
            P = np.random.randn(D, k)
            # orthonormalize
            Q, _ = np.linalg.qr(P)
            P = Q[:, :k]
        self._ibp_P = P
        self._ibp_mean = mean

    def _ibp_mutation(self, base: np.ndarray, pop_array: np.ndarray) -> np.ndarray:
        """
        Project base to bottleneck, perturb in bottleneck, reconstruct.
        Add small reconstruction noise for exploration.
        """
        if self._ibp_P is None:
            self._ibp_update_projection(pop_array)
        P = self._ibp_P
        mean = self._ibp_mean
        z = P.T.dot(base - mean)  # k vector
        # perturb in compressed space (Gaussian)
        zp = z + np.random.normal(scale=self.ibp_recon_noise * np.linalg.norm(z) + 1e-8, size=z.shape)
        recon = mean + P.dot(zp)
        # add small full-space jitter too
        recon += np.random.normal(scale=0.002 * np.mean(self.widths), size=self.dim)
        return recon

    # ---------- main loop ----------
    def minimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        pop = self._population_init()
        history = []  # best fitness per generation

        # Initialize operator bookkeeping
        if self.operator == "cdo":
            # initialize per-pop hypotheses
            self._cdo_init(pop)
        if self.operator == "rso":
            self._rso_init(pop)

        best = max(pop, key=lambda ab: ab.affinity)
        best_val = -best.affinity
        stagnation = 0

        k = max(2, self.max_gens)
        beta = self.beta

        # IBP rolling window storage
        ibp_window = []

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

            # precompute population array for operators that need it
            pop_array = np.stack([ab.x for ab in pop], axis=0)

            # optionally update IBP projection every ibp_update_window gens
            if self.operator == "ibp":
                ibp_window.append(pop_array)
                if len(ibp_window) > self.ibp_update_window:
                    ibp_window.pop(0)
                big = np.vstack(ibp_window)
                self._ibp_update_projection(big)

            for clone_idx, ab in enumerate(clones):
                # decide candidate using the selected operator (completely replaces DE-like steps)
                if self.operator == "base":
                    # original behavior preserved (random uniform or directed)
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
                elif self.operator == "tfo":
                    cand = self._tfo_mutation(ab.x, pop)
                elif self.operator == "cdo":
                    # give clone_idx as approximate population index mapping
                    cand = self._cdo_mutation(clone_idx, ab.x, pop, stagnation)
                elif self.operator == "rso":
                    cand = self._rso_mutation(clone_idx, ab.x, pop, gen)
                elif self.operator == "ibp":
                    cand = self._ibp_mutation(ab.x, pop_array)
                else:
                    raise ValueError("Unknown operator")

                self._clip(cand)
                cand_fit = self._affinity(cand)

                # QOBL / QRBL as in original workflow
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

            self._forget_in_place(pop)

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

        diagnostics: Dict[str, Any] = {"generations": self.max_gens, "history": history}
        # operator-specific diagnostics
        if self.operator == "cdo":
            diagnostics["cdo_delta"] = (self._cdo_delta.copy() if self._cdo_delta is not None else None)
        if self.operator == "rso":
            diagnostics["rso_phase"] = (self._rso_phase.copy() if self._rso_phase is not None else None)
            diagnostics["rso_freq"] = (self._rso_freq.copy() if self._rso_freq is not None else None)
        if self.operator == "ibp":
            diagnostics["ibp_P_shape"] = (None if self._ibp_P is None else self._ibp_P.shape)
            diagnostics["ibp_mean_norm"] = (None if self._ibp_mean is None else float(np.linalg.norm(self._ibp_mean)))
        return best.x.copy(), best_val, diagnostics