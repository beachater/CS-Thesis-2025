import numpy as np

class NOVA_Enhanced:
    """
    NOVA_Enhanced: FCSA-anchored optimizer with:
      - Adaptive Subspace Clonal Sampling (learn variable importance, mutate in small subspaces)
      - Chaotic quasi-opposition seeding/mutation (lightweight chaotic map)
      - Sparse local linear surrogate polish (in subspace, very cheap)
      - Budgeted cloning and aging to limit computations

    Usage:
      opt = NOVA_Enhanced(func, bounds, N=60, max_evals=350000, seed=0)
      best_x, best_f, history = opt.optimize()
    """
    def __init__(self, func, bounds, N=60, max_evals=350000, seed=None,
                 subspace_k=None, clone_factor=3, surrogate_points=8,
                 chaos_mu=4.0, quasi_opp_prob=0.15, progress: callable = None,
                 importance_samples: int = 20, chaos_chunk_size: int = 200, max_offspring_factor: int = 2):
        self.func = func
        self.bounds = np.array(bounds, dtype=float)
        self.lb = self.bounds[:,0]
        self.ub = self.bounds[:,1]
        self.dim = len(bounds)
        self.N = N
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
        self.clone_factor = clone_factor   # max clones per selected individual
        self.surrogate_points = surrogate_points  # points to fit local linear model
        self.chaos_mu = chaos_mu  # logistic map multiplier for chaos
        self.quasi_opp_prob = quasi_opp_prob
        # performance tuning options
        # number of samples used to estimate variable importance (archive preferred)
        self.importance_samples = max(1, int(importance_samples))
        # chaos chunk size used for clonal generation (populated in optimize)
        self.chaos_chunk_size = max(10, int(chaos_chunk_size))
        # cap factor for offspring budget (max offspring ~ max_offspring_factor * N)
        self.max_offspring_factor = max(1, int(max_offspring_factor))
        # subspace size (must not exceed problem dimension)
        if subspace_k is None:
            # default subspace size: sqrt(dim), but ensure at least 1 and at most dim
            self.k = min(self.dim, max(1, int(np.sqrt(self.dim))))
        else:
            self.k = min(max(1, int(subspace_k)), self.dim)
        # prefer small subspaces but allow up to sqrt(dim) or user-specified
        self.k = max(2, min(self.k, max(2, int(np.sqrt(max(2, self.dim))))))

        # small archive to fit surrogates
        self.archive_X = []
        self.archive_F = []
        # optional progress callback
        self._progress_cb = progress

    def _safe_eval(self, x):
        """Evaluate self.func and return scalar fitness (handles tuple returns)."""
        val = self.func(x)
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(np.asarray(val).ravel()[0])
        return float(val)

    def _init_pop(self):
        return self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.dim))

    def _eval_pop(self, pop):
        res = np.empty(len(pop), dtype=float)
        for i, ind in enumerate(pop):
            res[i] = self._safe_eval(ind)
            # update archive
            self.archive_X.append(ind.copy())
            self.archive_F.append(res[i])
        return res

    def _logistic_chaos(self, x_scalar):
        # simple logistic map chaotic generator; input x in (0,1), returns new in (0,1)
        return self.chaos_mu * x_scalar * (1 - x_scalar)

    def _quasi_opposite(self, x):
        # quasi-opposite near the real opposite, but biased toward midpoint
        mid = (self.lb + self.ub) / 2.0
        opp = self.lb + self.ub - x
        r = self.rng.random(self.dim)
        # quasi-opposite: random point between midpoint and full opposite
        qo = mid + r * (opp - mid)
        return np.clip(qo, self.lb, self.ub)

    def _estimate_variable_importance(self, pop, fit, samples=20, eps=1e-4):
        """
        Cheap finite-difference style importance: pick a subset of individuals and
        perturb each dimension by eps (relative) and measure average |df|.
        Returns normalized importance array of length dim.
        """
        dim = self.dim
        imp = np.zeros(dim, dtype=float)
        # If we have a reasonable archive of past evaluations, compute importance
        # from the archive (cheap, no extra FEs). Otherwise fall back to a small
        # finite-difference style perturbation (limited samples).
        try:
            X_arr = np.asarray(self.archive_X)
            F_arr = np.asarray(self.archive_F)
        except Exception:
            X_arr = np.empty((0, dim))
            F_arr = np.empty((0,))

        if X_arr.shape[0] >= max(6, self.importance_samples):
            # Use absolute Pearson correlation between each dimension and fitness as importance
            # This is cheap and leverages past evaluations stored in the archive.
            for d in range(dim):
                xd = X_arr[:, d]
                xd_mean = xd.mean()
                fd_mean = F_arr.mean()
                num = ((xd - xd_mean) * (F_arr - fd_mean)).sum()
                den = np.sqrt(((xd - xd_mean) ** 2).sum() * ((F_arr - fd_mean) ** 2).sum()) + 1e-12
                imp[d] = abs(num) / den
            # if all zeros (unlikely), fallback to uniform
            if imp.sum() == 0:
                return np.ones(dim) / dim
            imp = imp + 1e-12
            imp = imp / imp.sum()
            return imp

        # fallback: small-sample finite-difference perturbation (expensive path)
        samples_to_use = min(len(pop), samples, self.importance_samples)
        # limit samples used to a small number to avoid heavy extra evaluations
        samples_to_use = min(samples_to_use, 6)
        m = max(1, int(samples_to_use))
        replace = False if m <= len(pop) else True
        idxs = self.rng.choice(len(pop), m, replace=replace)
        base = pop[idxs]
        base_f = fit[idxs]
        # perturb per-dimension using small relative epsilon
        for d in range(dim):
            pert = base.copy()
            delta = eps * (self.ub[d] - self.lb[d] + 1e-12)
            pert[:, d] = np.clip(pert[:, d] + delta, self.lb[d], self.ub[d])
            f_pert = np.array([self._safe_eval(ind) for ind in pert])
            imp[d] = np.mean(np.abs(f_pert - base_f))
        # normalize
        if imp.sum() == 0:
            return np.ones(dim) / dim
        imp = imp + 1e-12
        imp = imp / imp.sum()
        return imp

    def _select_subspace_dims(self, importance):
        """
        Select k dimensions for a subspace using importance-proportional sampling
        combined with small random jitter to keep exploration.
        """
        probs = importance / (importance.sum() + 1e-12)
        # choose without replacement; if k >= dim fallback to range
        if self.k >= self.dim:
            dims = np.arange(self.dim)
        else:
            dims = self.rng.choice(self.dim, size=self.k, replace=False, p=probs)
        return np.sort(dims)

    def _mutate_in_subspace(self, base, dims, scale=0.05):
        """
        Mutate only in selected dims: gaussian noise scaled by bounds per-dim.
        scale is relative fraction of (ub-lb).
        """
        child = base.copy()
        if len(dims) == 0:
            return child
        deltas = self.rng.normal(0, 1, size=len(dims))
        ranges = (self.ub[dims] - self.lb[dims])
        child[dims] = np.clip(child[dims] + deltas * scale * ranges, self.lb[dims], self.ub[dims])
        return child

    def _budgeted_clonal_generation(self, pop, fit, importance, chaos_seq, gen=1, max_gen=1000):
        """
        Apply clonal selection: choose top individuals by fitness, clone them (budgeted),
        mutate clones inside adaptive subspaces, produce a pool of offspring.
        Use chaotic quasi-opposition occasionally for stronger diversification.
        """
        offspring = []
        # sorted indices (ascending fitness)
        idx_sorted = np.argsort(fit)
        # pick top individuals to clone (adaptive count)
        top_m = max(2, min(self.N//4, 8))
        elites_idx = idx_sorted[:top_m]
        # approximate offspring budget to avoid explosion
        target_offspring = max(8, min(len(pop) * self.max_offspring_factor, self.N * self.max_offspring_factor * 4))
        for rank, i in enumerate(elites_idx):
            parent = pop[i]
            # affinity ~ relative fitness rank; better -> more clones
            affinity = (top_m - rank) / top_m  # in (0,1]
            clones = max(1, int(np.ceil(self.clone_factor * affinity)))
            # but budget clones to not explode
            clones = min(clones, 8)
            for c in range(clones):
                # select subspace dims based on learned importance
                dims = self._select_subspace_dims(importance)
                # chaotic seed: use logistic map seeded from chaos_seq (one scalar per clone)
                chaos_val = chaos_seq.pop() if chaos_seq else self.rng.random()
                chaos_next = self._logistic_chaos(chaos_val)
                # mutate scale adapted by chaos
                # anneal mutation scale from larger exploration early to smaller later
                anneal = max(0.1, 1.0 - (gen / float(max_gen)))
                base_scale = 0.05
                scale = base_scale * (0.5 + 0.5 * anneal) * (0.8 + 0.4 * chaos_next)
                child = self._mutate_in_subspace(parent, dims, scale=scale)
                # occasionally generate quasi-opposite in subspace
                if self.rng.random() < self.quasi_opp_prob:
                    qo = self._quasi_opposite(parent)
                    # mix with child but only in dims
                    child[dims] = 0.5 * (child[dims] + qo[dims])
                offspring.append(child)
                if len(offspring) >= target_offspring:
                    break
            if len(offspring) >= target_offspring:
                break
        # also add a few random exploratory individuals (to keep exploration)
        extra = max(1, self.N // 10)
        for _ in range(extra):
            if len(offspring) >= target_offspring:
                break
            rnd = self.lb + (self.ub - self.lb) * self.rng.random(self.dim)
            offspring.append(rnd)
        return np.array(offspring)

    def _sparse_local_surrogate_propose(self, center, dims, max_step_frac=0.1):
        """
        Fit a linear surrogate in selected dims using a few archived points near center.
        Return a proposed new point (one or None).
        Very cheap: uses at most surrogate_points points and simple lstsq.
        """
        if len(self.archive_X) < 4:
            return None
        # collect archived points and their deltas in dims
        X = np.array(self.archive_X)
        F = np.array(self.archive_F)
        # compute distances in the chosen dims to center
        if X.shape[0] < 4:
            return None
        dists = np.linalg.norm((X[:, dims] - center[dims]), axis=1)
        # pick nearest surrogate_points points
        order = np.argsort(dists)
        sel = order[:self.surrogate_points]
        Xs = X[sel][:, dims]
        Fs = F[sel]
        # build linear model: Fs = a + b dot Xs  => fit [1, Xs] -> Fs
        A = np.hstack([np.ones((len(sel),1)), Xs - np.mean(Xs, axis=0)])
        try:
            coef, *_ = np.linalg.lstsq(A, Fs, rcond=None)
        except Exception:
            return None
        # gradient estimate is coef[1:]
        grad = coef[1:]
        # step direction: move opposite to gradient in subspace, scaled by range
        ranges = (self.ub[dims] - self.lb[dims])
        # normalize grad
        if np.linalg.norm(grad) == 0:
            return None
        step = -grad / (np.linalg.norm(grad) + 1e-12)
        step_size = max_step_frac * ranges  # vector
        proposal = center.copy()
        proposal[dims] = np.clip(proposal[dims] + step * step_size, self.lb[dims], self.ub[dims])
        return proposal
    def optimize(self):
        pop = self._init_pop()
        fit = self._eval_pop(pop)
        evals = len(pop)
        if self._progress_cb:
            try:
                self._progress_cb(len(pop))
            except Exception:
                pass

        best_idx = np.argmin(fit)
        gbest = pop[best_idx].copy()
        fbest = fit[best_idx]
        history = [fbest]

        # main loop with generation counter for annealing
        gen = 1
        est_gen = max(10, int(self.max_evals / max(1, self.N)))
        max_gen = est_gen

        while evals < self.max_evals:
            # estimate variable importance cheaply (archive preferred)
            importance = self._estimate_variable_importance(pop, fit, samples=min(20, self.N))

            # prepare chaos sequence chunk (use configured size)
            chaos_chunk = list(self.rng.random(self.chaos_chunk_size))[::-1]

            # clonal generation in adaptive subspaces (pass generation for annealing)
            offspring = self._budgeted_clonal_generation(pop, fit, importance, chaos_chunk, gen=gen, max_gen=max_gen)

            # evaluate offspring but budget: if offspring too many, sample subset
            if len(offspring) == 0:
                break
            max_off = min(len(offspring), max(4, self.N * 2))
            if len(offspring) > max_off:
                idxs = self.rng.choice(len(offspring), max_off, replace=False)
                offspring_eval = offspring[idxs]
            else:
                offspring_eval = offspring

            off_fit = self._eval_pop(offspring_eval)
            evals += len(offspring_eval)
            if self._progress_cb:
                try:
                    self._progress_cb(len(offspring_eval))
                except Exception:
                    pass

            # replacement: combine and keep top N
            combined_X = np.vstack([pop, offspring_eval])
            combined_F = np.hstack([fit, off_fit])
            idx_keep = np.argsort(combined_F)[:self.N]
            pop = combined_X[idx_keep]
            fit = combined_F[idx_keep]

            # lightweight elite hillclimb: try a few micro-steps in important dims
            elites = np.argsort(fit)[:max(1, min(3, self.N // 10))]
            for ei in elites:
                center = pop[ei].copy()
                imp_dims = self._select_subspace_dims(importance)
                for d in imp_dims[:max(1, len(imp_dims) // 3)]:
                    step = 1e-3 * (self.ub[d] - self.lb[d])
                    for sign in (-1, 1):
                        probe = center.copy()
                        probe[d] = np.clip(probe[d] + sign * step, self.lb[d], self.ub[d])
                        val = self._safe_eval(probe)
                        evals += 1
                        if self._progress_cb:
                            try:
                                self._progress_cb(1)
                            except Exception:
                                pass
                        if val < fit.max():
                            worst_idx = np.argmax(fit)
                            pop[worst_idx] = probe
                            fit[worst_idx] = val

            # local surrogate polishing for a small number of elites
            elites = np.argsort(fit)[:max(1, min(3, self.N // 10))]
            for ei in elites:
                center = pop[ei]
                dims = self._select_subspace_dims(importance)
                prop = self._sparse_local_surrogate_propose(center, dims)
                if prop is not None:
                    val = self._safe_eval(prop)
                    evals += 1
                    if self._progress_cb:
                        try:
                            self._progress_cb(1)
                        except Exception:
                            pass
                    # add to archive handled in _safe_eval indirectly
                    if val < fit.max():
                        worst_idx = np.argmax(fit)
                        pop[worst_idx] = prop
                        fit[worst_idx] = val

            # occasional quasi-opposition injection of best solution
            if self.rng.random() < 0.05:
                qo = self._quasi_opposite(gbest)
                val = self._safe_eval(qo)
                evals += 1
                if self._progress_cb:
                    try:
                        self._progress_cb(1)
                    except Exception:
                        pass
                if val < fbest:
                    gbest = qo.copy()
                    fbest = val

            # update global best and record history
            cur_best_idx = np.argmin(fit)
            if fit[cur_best_idx] < fbest:
                gbest = pop[cur_best_idx].copy()
                fbest = fit[cur_best_idx]

            history.append(fbest)
            gen += 1

        return gbest, fbest, history
