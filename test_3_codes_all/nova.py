import numpy as np
import csv

# ================================================================
# NOVAPlus Algorithm
# ================================================================
class NOVAPlus:
    def __init__(self, func, bounds, N=60, max_evals=350000, seed=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(bounds)
        self.N = N
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
        self.F_mean, self.CR_mean = 0.5, 0.9

    def _init_pop(self):
        return self.lb + (self.ub - self.lb) * self.rng.random((self.N, self.dim))

    def _eval(self, pop):
        return np.array([self.func(ind) for ind in pop])

    def _obl(self, x, noise=0.01):
        return self.lb + self.ub - x + self.rng.normal(0, noise, size=x.shape)

    def _local_polish(self, x, step=1e-3):
        best = x.copy()
        fbest = self.func(best)
        for d in range(self.dim):
            for delta in [-step, step]:
                cand = best.copy()
                cand[d] = np.clip(cand[d] + delta, self.lb[d], self.ub[d])
                fval = self.func(cand)
                if fval < fbest:
                    best, fbest = cand, fval
        return best, fbest

    def minimize(self):
        pop = self._init_pop()
        fit = self._eval(pop)
        evals = self.N
        best_idx = np.argmin(fit)
        gbest, fbest = pop[best_idx].copy(), fit[best_idx]

        history = []
        switch_FE = int(0.7 * self.max_evals)

        while evals < self.max_evals:
            offspring = []

            if evals < switch_FE:
                # Exploration phase
                for i in range(self.N):
                    step = (0.1 / np.sqrt(self.dim)) * (self.ub - self.lb) * self.rng.normal(size=self.dim)
                    cand = np.clip(pop[i] + step, self.lb, self.ub)
                    offspring.append(cand)

                for i in range(self.N):
                    idxs = self.rng.choice(self.N, 3, replace=False)
                    x, a, b, c = pop[i], pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
                    F = np.clip(self.rng.normal(self.F_mean, 0.1), 0, 1)
                    CR = np.clip(self.rng.normal(self.CR_mean, 0.1), 0, 1)
                    mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                    cross = self.rng.random(self.dim) < CR
                    trial = np.where(cross, mutant, x)
                    offspring.append(trial)

                elite_k = max(2, self.N // 5)
                elites = pop[np.argsort(fit)[:elite_k]]
                mu = np.mean(elites, axis=0)
                cov = np.cov(elites.T) + 1e-6 * np.eye(self.dim)
                cov_samples = self.rng.multivariate_normal(mu, cov, size=self.N//5)
                cov_samples = np.clip(cov_samples, self.lb, self.ub)
                offspring.extend(cov_samples)

                for i in range(self.N//5):
                    offspring.append(self._obl(pop[i], noise=0.01))

            else:
                # Polishing phase
                new_pop = []
                for i in range(self.N):
                    cand, fval = self._local_polish(pop[i])
                    new_pop.append(cand)
                offspring = np.array(new_pop)
                off_fit = self._eval(offspring)
                evals += len(offspring)

                best_idx = np.argmin(off_fit)
                if off_fit[best_idx] < fbest:
                    gbest, fbest = offspring[best_idx].copy(), off_fit[best_idx]
                history.append(fbest)
                continue

            offspring = np.array(offspring)
            off_fit = self._eval(offspring)
            evals += len(offspring)

            combined = np.vstack([pop, offspring])
            combined_fit = np.hstack([fit, off_fit])
            best_idx = np.argsort(combined_fit)[:self.N]
            pop, fit = combined[best_idx], combined_fit[best_idx]

            if fit[0] < fbest:
                gbest, fbest = pop[0].copy(), fit[0]

            history.append(fbest)

        return gbest, fbest, history


