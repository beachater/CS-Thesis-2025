import numpy as np
from copy import deepcopy
from typing import List, Tuple
from problem_cec2022 import DMMOProblem


class CoreSearchOptions:
    """
    Matches MATLAB main.m defaults
    """
    def __init__(self):
        self.iniSubpopSizeCoeff = 9.0
        self.finSubpopSizeCoeff = 9.0
        self.muToPopSizeRatio = 0.5
        self.iniSigCoeff = 1.0
        self.maxIniSigma = 0.1
        self.eltRatio = 0.0  # percent elites reinserted
        self.tauSigmaCoeff = 0.5
        self.tauCovCoeff = 1.0
        self.sigmaUpdateBiasImp = 1.0
        # numeric safety
        self.sigMinLim = 1e-6
        self.sigMaxLim = 0.1
        # stop floor for sigma seen in CMSA style
        self.sigmaStopFloor = 1e-6


class StopCriteriaOptions:
    def __init__(self):
        # idle improvement tolerance
        self.tolHistFun = 1e-5
        self.idleIterLimit = 30
        self.maxCondC = 1e12
        self.maxIter = 10_000  # fallback


class DynamicOptions:
    """
    Dynamic manager controls AMLP reseeding between restarts

    Based on MATLAB DynamicOption
    """
    def __init__(self):
        self.maxPL = 10  # trajectory depth in past envs
        self.recStepSizeCoeff = 0.3
        self.chCheckFr = 30
        self.confAlphaInit = 0.5
        self.confAlphaMin = 0.1
        self.confAlphaMax = 1.0


class OptimProcess:
    """
    This corresponds to MATLAB OptimProcess
    Tracks recombination weights etc
    """
    def __init__(self, pop_size, core_opts: CoreSearchOptions):
        self.pop_size = pop_size
        self.mu = max(2, int(np.floor(core_opts.muToPopSizeRatio * pop_size)))
        ranks = np.arange(1, self.mu + 1)
        w = np.log(1.0 + self.mu) - np.log(ranks)
        self.recWeights = w / np.sum(w)
        # mu_eff for step adaptation if needed
        self.muEff = 1.0 / np.sum(self.recWeights**2)


class ArchiveEntry:
    def __init__(self, x, f, eval_count):
        self.x = np.array(x, dtype=float)
        self.f = float(f)
        self.foundEval = eval_count
        # each niche has adaptive taboo radius
        self.radius = None
        self.hitCount = 1


class ArchiveStore:
    """
    This is MATLAB Archive logic simplified but with adaptive taboo radius

    We keep a list of niches each niche has center x and best f
    radius grows smaller when repeatedly rediscovered
    """
    def __init__(self, tolFunArch=1e-3):
        self.entries: List[ArchiveEntry] = []
        self.tolFunArch = tolFunArch

    def _euclid(self, a, b):
        return np.linalg.norm(a - b)

    def add_solution(self, x, f, eval_count):
        if not self.entries:
            e = ArchiveEntry(x, f, eval_count)
            e.radius = 0.5  # start fairly wide
            self.entries.append(e)
            return

        # see if near an existing niche
        best_i = None
        best_d = np.inf
        for i, ent in enumerate(self.entries):
            d = self._euclid(x, ent.x)
            if d < best_d:
                best_d = d
                best_i = i

        ent = self.entries[best_i]
        if best_d < ent.radius:
            # same niche
            ent.hitCount += 1
            if f > ent.f + self.tolFunArch:
                ent.x = np.array(x, dtype=float)
                ent.f = float(f)
                ent.foundEval = eval_count
            # shrink taboo radius a bit because we keep rediscovering
            ent.radius *= 0.95
            ent.radius = max(ent.radius, 0.05)
        else:
            # new niche
            e = ArchiveEntry(x, f, eval_count)
            # initial radius is distance to the closest niche
            e.radius = best_d * 0.5
            e.radius = max(min(e.radius, 0.5), 0.05)
            self.entries.append(e)

    def get_taboo_regions(self):
        """
        Return list of (center, radius)
        """
        return [(ent.x.copy(), ent.radius) for ent in self.entries]

    def export_env_archive(self):
        """
        Save snapshot of this archive for AMLP after current environment
        Returns arrays of shape [K, D] and [K]
        """
        if not self.entries:
            return np.zeros((0,)), np.zeros((0,))
        pts = np.stack([ent.x for ent in self.entries], axis=0)
        vals = np.array([ent.f for ent in self.entries])
        return pts, vals


class Subpopulation:
    """
    This approximates MATLAB SubpopulationCMSA

    State includes:
    mean       center
    smean      global step size (sigma)
    C          covariance matrix
    elite_X    elite carryover
    elite_F
    idleIter   how many iterations no improvement
    """

    def __init__(self, center, init_sigma, core_opts: CoreSearchOptions,
                 process: OptimProcess, rng: np.random.Generator):
        self.rng = rng
        self.core_opts = core_opts
        self.process = process

        self.D = center.shape[0]
        self.lambda_ = process.pop_size
        self.mu = process.mu

        self.mean = center.astype(float)
        self.smean = float(init_sigma)
        self.C = np.eye(self.D)
        self.C_sqrt = np.eye(self.D)

        self.elite_X = np.empty((0, self.D))
        self.elite_F = np.empty((0,))
        self.best_f = -np.inf
        self.best_x = self.mean.copy()

        self.idleIter = 0
        self.iter_count = 0
        self.converged = False

    def _update_cholesky(self):
        # numeric safety
        # ensure symmetric
        self.C = 0.5 * (self.C + self.C.T)
        # eigen decomposition
        vals, vecs = np.linalg.eigh(self.C)
        vals = np.clip(vals, 1e-14, None)
        self.C = (vecs * vals) @ vecs.T
        self.C_sqrt = vecs * np.sqrt(vals) @ vecs.T

    def _sample_candidate(self):
        z = self.rng.normal(size=self.D)
        step = self.C_sqrt @ z
        return self.mean + self.smean * step

    def sample_population(self, taboo_regions: List[Tuple[np.ndarray, float]], lb, ub):
        """
        Sample lambda_ offspring respecting taboo regions
        plus we will later merge elites before selection
        """
        X = []
        for _ in range(self.lambda_):
            attempt = 0
            while True:
                x = self._sample_candidate()
                # project to bounds
                x = np.minimum(np.maximum(x, lb), ub)

                taboo_ok = True
                for (c, r) in taboo_regions:
                    if np.linalg.norm(x - c) < r:
                        taboo_ok = False
                        break
                if taboo_ok:
                    X.append(x)
                    break
                attempt += 1
                if attempt > 50:
                    # give up taboo this time
                    X.append(x)
                    break
        return np.array(X)

    def select_and_recombine(self, X, F):
        """
        Selection with elitism and recombination as in RS CMSA ESII
        """
        # maintain elites
        elite_quota = int(np.floor(self.core_opts.eltRatio * self.lambda_))
        if elite_quota > 0 and self.elite_X.shape[0] > 0:
            # take top elite_quota elites
            elite_idx = np.argsort(self.elite_F)[::-1][:elite_quota]
            eliteX = self.elite_X[elite_idx]
            eliteF = self.elite_F[elite_idx]
            X = np.vstack([X, eliteX])
            F = np.concatenate([F, eliteF])

        # sort offspring by fitness desc
        order = np.argsort(F)[::-1]
        X = X[order]
        F = F[order]

        # track best
        if F[0] > self.best_f + 1e-12:
            self.best_f = float(F[0])
            self.best_x = X[0].copy()
            self.idleIter = 0
        else:
            self.idleIter += 1

        # update elite pool
        self.elite_X = deepcopy(X[:self.mu])
        self.elite_F = deepcopy(F[:self.mu])

        # recombination weights
        w = self.process.recWeights  # length mu
        # ensure shapes
        parents = X[:self.mu]
        old_mean = self.mean.copy()
        new_mean = np.sum(parents * w[:, None], axis=0)

        # update covariance following CMSA style
        Y = (parents - old_mean) / max(self.smean, 1e-16)
        # weighted covariance update
        cov_incr = np.zeros_like(self.C)
        for i in range(self.mu):
            yi = Y[i][:, None]
            cov_incr += w[i] * (yi @ yi.T)

        tau_cov = self.core_opts.tauCovCoeff
        self.C = (1.0 - tau_cov) * self.C + tau_cov * cov_incr

        # update sigma global step size
        # CMA-ES like path length adaptation approximate
        # use muEff and tauSigmaCoeff
        muEff = self.process.muEff
        tau_sigma = np.sqrt(self.core_opts.tauSigmaCoeff / (2.0 * self.D))
        # success estimate is distance moved vs expected
        step_vec = new_mean - old_mean
        step_norm = np.linalg.norm(step_vec) / (self.smean + 1e-16)
        expected_norm = np.sqrt(self.D)  # rough expectation for N(0,I)

        bias_imp = self.core_opts.sigmaUpdateBiasImp
        sigma_factor = np.exp(
            (tau_sigma / bias_imp) * ((step_norm / (expected_norm + 1e-16)) - 1.0)
        )
        new_sigma = self.smean * sigma_factor
        new_sigma = np.clip(
            new_sigma,
            self.core_opts.sigMinLim,
            self.core_opts.sigMaxLim
        )

        self.mean = new_mean
        self.smean = float(new_sigma)
        self._update_cholesky()

        self.iter_count += 1

        # convergence test similar spirit to MATLAB stop criteria
        condC = np.linalg.cond(self.C)
        sigma_small = self.smean < self.core_opts.sigmaStopFloor
        stuck = self.idleIter >  self.core_opts.iniSubpopSizeCoeff  # heuristic tie to 9
        bad_cond = condC > 1e12
        if sigma_small or stuck or bad_cond:
            self.converged = True

    def get_center_sigma(self):
        return self.mean.copy(), float(self.smean)


class DynamicManager:
    """
    This reproduces MATLAB DynamicManager at high level

    It keeps a history of per environment archives and produces predicted
    centers and step sizes (recCenter, recStepSize) for the next restart
    using AMLP like multi level linear fitting
    """

    def __init__(self, dyna_opts: DynamicOptions, dim: int, lb, ub):
        self.opts = dyna_opts
        self.dim = dim
        self.lb = lb
        self.ub = ub

        # list of env snapshots
        # each snapshot is dict with keys 'pts' [K,D] and 'vals' [K]
        self.history = []

        # confidence alpha used to scale step
        self.confAlpha = self.opts.confAlphaInit

    def record_environment_archive(self, pts, vals):
        """
        Called after finishing one environment
        pts [K,D], vals [K]
        We also normalize pts to [0,1] per dim for AMLP fit
        """
        if pts.size == 0:
            norm_pts = np.zeros((0, self.dim))
        else:
            span = (self.ub - self.lb)
            span_safe = np.where(span == 0.0, 1.0, span)
            norm_pts = (pts - self.lb) / span_safe
        self.history.append({
            "pts": pts.copy(),
            "vals": vals.copy(),
            "norm_pts": norm_pts.copy()
        })
        # limit length
        if len(self.history) > self.opts.maxPL:
            self.history = self.history[-self.opts.maxPL:]

    def _fit_linear_extrapolation(self, traj):
        """
        traj shape [T, D] in normalized space
        We try multi level linear fit and choose best level by back error

        MATLAB AMLP does multilevel prediction
        Here we try fits for horizon L in [2..min(T, maxPL)]
        Then pick L that best predicts last point from previous points
        """
        T = traj.shape[0]
        if T < 2:
            # not enough data
            pred = traj[-1]
            err_est = 0.3  # fallback
            return pred, err_est

        best_err = np.inf
        best_pred = traj[-1]
        for L in range(2, T+1):
            sub = traj[-L:]
            # time index 0..L-1
            ts = np.arange(L)[:, None]  # [L,1]
            # fit linear for each dim: dim-wise least squares
            coeffs = np.linalg.lstsq(
                np.hstack([ts, np.ones_like(ts)]),
                sub,
                rcond=None
            )[0]  # shape [2, D]
            a = coeffs[0]
            b = coeffs[1]
            # predict next time = L
            pred_norm = a * L + b
            # estimate back error = mse of fit on sub
            recon = (a[None, :] * ts + b[None, :])
            err = np.mean((recon - sub)**2)
            if err < best_err:
                best_err = err
                best_pred = pred_norm
        err_est = np.sqrt(best_err)
        return best_pred, err_est

    def generate_reseed_info(self):
        """
        returns list of tuples (center, stepSize)
        recCenter in original coordinates
        recStepSize scalar sigma

        We build trajectories for each niche by nearest neighbor matching
        across envs just like MATLAB does to "track past history"
        """
        if len(self.history) < 2:
            return []

        # Step 1 collect niches from last environment
        last = self.history[-1]
        last_pts = last["pts"]
        last_norm_pts = last["norm_pts"]
        if last_pts.shape[0] == 0:
            return []

        # Step 2 build trajectory per niche
        # we greedily match each last niche backward in time using NN in norm space
        reseeds = []
        for i in range(last_pts.shape[0]):
            traj_norm_list = [last_norm_pts[i]]
            prev_center = last_norm_pts[i]

            # walk backwards through history
            for hist_idx in range(len(self.history)-2, -1, -1):
                hist_norm = self.history[hist_idx]["norm_pts"]
                if hist_norm.shape[0] == 0:
                    break
                dists = np.linalg.norm(hist_norm - prev_center[None, :], axis=1)
                j = np.argmin(dists)
                prev_center = hist_norm[j]
                traj_norm_list.append(prev_center)

                if len(traj_norm_list) >= self.opts.maxPL:
                    break

            traj_norm = np.array(traj_norm_list[::-1])  # oldest first
            # Step 3 AMLP style extrapolation
            pred_norm, err_est = self._fit_linear_extrapolation(traj_norm)

            # map back to real space
            span = (self.ub - self.lb)
            span_safe = np.where(span == 0.0, 1.0, span)
            pred_real = self.lb + pred_norm * span_safe

            # estimate step size
            # MATLAB uses recStepSizeCoeff * prediction_error but also clamps by maxIniSigma
            step_sigma = min(
                self.opts.recStepSizeCoeff * (err_est + 1e-12),
                0.1  # maxIniSigma in main.m
            )
            step_sigma = max(step_sigma, 1e-4)

            reseeds.append((pred_real, step_sigma))

        # merge close reseeds to avoid duplicates
        merged = []
        for center, sig in reseeds:
            keep = True
            for (c2, s2) in merged:
                if np.linalg.norm(center - c2) < 0.1:
                    keep = False
                    break
            if keep:
                merged.append((center, sig))

        return merged


class AMLP_RS_CMSA_ESII:
    """
    Full solver
    Matches MATLAB driver.m logic at high level

    Steps for each environment
    while env not finished:
        start a restart
        init subpop using either AMLP predicted center+sigma or random init
        loop evolve until subpop converges or budget break
        archive best
        next restart

    After environment ends
    push environment archive snapshot to DynamicManager
    call problem.update_environment()
    """

    def __init__(self, problem: DMMOProblem, rng_seed: int = 0):
        self.problem = problem
        self.rng = np.random.default_rng(rng_seed)

        self.core_opts = CoreSearchOptions()
        self.stop_opts = StopCriteriaOptions()
        self.dyna_opts = DynamicOptions()

        # global archive across restarts and across environments
        self.archive = ArchiveStore(tolFunArch=1e-3)

        # dynamic manager that stores per environment archive
        self.dmanager = DynamicManager(
            self.dyna_opts,
            dim=self.problem.D,
            lb=self.problem.lb,
            ub=self.problem.ub
        )

    def _init_random_center_sigma(self):
        center = self.rng.uniform(self.problem.lb, self.problem.ub)
        # iniSigCoeff * range / something
        # in MATLAB iniSigCoeff = 1 and then clamped by maxIniSigma = 0.1
        span = self.problem.ub - self.problem.lb
        guessed_sigma = np.mean(span) / (2.0 * self.core_opts.iniSubpopSizeCoeff)
        guessed_sigma = min(guessed_sigma, self.core_opts.maxIniSigma)
        guessed_sigma = max(guessed_sigma, 1e-3)
        return center, guessed_sigma

    def _spawn_subpop(self, reseed_pool, used_reseeds):
        """
        pick reseed if available else random
        reseed_pool is list of (center, sigma)
        used_reseeds is index set
        """
        pick_idx = None
        for idx in range(len(reseed_pool)):
            if idx not in used_reseeds:
                pick_idx = idx
                break
        if pick_idx is None:
            center, sigma = self._init_random_center_sigma()
        else:
            center, sigma = reseed_pool[pick_idx]
            used_reseeds.add(pick_idx)

        # create Subpopulation
        # popsize = iniSubpopSizeCoeff * D rounded
        pop_size = int(np.round(self.core_opts.iniSubpopSizeCoeff * self.problem.D))
        process = OptimProcess(pop_size, self.core_opts)
        sp = Subpopulation(center, sigma, self.core_opts, process, self.rng)
        return sp

    def _evolve_subpopulation(self, sp: Subpopulation):
        """
        run CMSA loop until stop
        return best_x, best_f
        """
        # track history to compute fitness deltas for stop criteria
        hist_best = []

        while True:
            # budget check
            if self.problem.environment_finished():
                break
            if sp.iter_count >= self.stop_opts.maxIter:
                break
            if sp.converged:
                break

            # sample with taboo
            taboo_regions = self.archive.get_taboo_regions()
            X = sp.sample_population(taboo_regions, self.problem.lb, self.problem.ub)

            # evaluate
            F = self.problem.evaluate(X)

            # select recombine adapt
            sp.select_and_recombine(X, F)

            hist_best.append(sp.best_f)
            # idleIter stop check duplicated inside select_and_recombine

            # if too idle by StopCriteriaOptions
            if sp.idleIter > self.stop_opts.idleIterLimit:
                sp.converged = True

        return sp.best_x.copy(), float(sp.best_f), sp

    def optimize_one_environment(self):
        """
        Run restarts until we hit env budget
        Produce environment archive snapshot
        """
        used_reseeds = set()
        reseed_pool = self.dmanager.generate_reseed_info()

        # subpop restart loop
        while not self.problem.environment_finished():
            subpop = self._spawn_subpop(reseed_pool, used_reseeds)
            best_x, best_f, subpop = self._evolve_subpopulation(subpop)

            # update archive with best
            self.archive.add_solution(best_x, best_f, self.problem.total_fe_used)

            # if we are almost out of budget break
            if self.problem.environment_finished():
                break

        # after env finishes
        env_pts, env_vals = self.archive.export_env_archive()
        self.dmanager.record_environment_archive(env_pts, env_vals)

    def run_full(self):
        """
        Run for problem.maxEnv environments or until total_fe_used saturates
        Return full archive log so we can compute Peak Ratio
        """
        for _ in range(self.problem.maxEnv):
            self.optimize_one_environment()
            # advance environment
            self.problem.update_environment()

        # final export
        all_pts, all_vals = self.archive.export_env_archive()
        return all_pts, all_vals


def compute_peak_ratio(problem: DMMOProblem, archive_pts, eps_d=1e-3, eps_f=1e-3):
    """
    Compute Peak Ratio for a given environment snapshot and problem
    NPF / NG
    Where NPF is number of distinct global optima detected
    """
    true_pos, true_fit = problem.get_true_global_optima()
    if archive_pts.size == 0 or true_pos.size == 0:
        return 0.0

    found_mask = np.zeros(true_pos.shape[0], dtype=bool)
    for i in range(true_pos.shape[0]):
        tp = true_pos[i]
        tf = true_fit[i]
        # check if any archive point matches
        dists = np.linalg.norm(archive_pts - tp[None, :], axis=1)
        if np.min(dists) < eps_d:
            found_mask[i] = True
    NPF = np.sum(found_mask)
    NG = true_pos.shape[0]
    if NG == 0:
        return 1.0
    return NPF / NG
