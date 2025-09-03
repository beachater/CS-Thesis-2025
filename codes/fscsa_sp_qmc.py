#-----------------
# Forgetting Based Clonal Selection Algorithm with Sobol Initialization
#-----------------------------
import numpy as np
import random
from dataclasses import dataclass
from typing import Callable, Tuple, List
from scipy.stats import qmc   # <-- NEW: for Sobol sequences

# --------------------------
# Antibody representation
# --------------------------
@dataclass
class Antibody:
    x: np.ndarray        # position (candidate solution)
    fitness: float       # objective value
    affinity: float      # affinity = inverse of fitness
    T: int = 0           # survival time (how long antibody has existed)
    S: int = 0           # memory intensity (how many times selected for cloning)


# --------------------------
# FSCSA main class
# --------------------------
class FSCSASPQMC:
    """Forgetful Clonal Selection Algorithm (Sobol-based version).
    """

    def __init__(self, func: Callable[[np.ndarray], Tuple[float,float,float]],
                 dim: int, N: int=50, n_select: int=10, n_clones: int=5,
                 m: float=2.0, c: float=3.0, max_iters: int=1000, seed: int=None):
        self.func = func
        self.dim = dim
        self.N = N
        self.n_select = n_select
        self.n_clones = n_clones
        self.m = m
        self.c = c
        self.max_iters = max_iters

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        f0, lb, ub = self.func(np.zeros(dim))
        self.lb = lb
        self.ub = ub

        self.history = []

        # Sobol sampler (scrambled for variety)
        self.sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        self.sobol_used = 0  # keep track of samples consumed

    # --------------------------
    # Fitness evaluation
    # --------------------------
    def _fitness(self, x: np.ndarray) -> float:
        y, _, _ = self.func(x)
        return float(y)

    # --------------------------
    # Affinity measure
    # --------------------------
    def _affinity(self, fitness: float) -> float:
        return 1.0 / (1.0 + max(fitness, 0.0))

    # --------------------------
    # Get next Sobol sample
    # --------------------------
    def _sobol_sample(self, n: int = 1) -> np.ndarray:
        pts = self.sampler.random(n)
        self.sobol_used += n
        return qmc.scale(pts, self.lb, self.ub)

    # --------------------------
    # Boundary handling
    # --------------------------
    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    # --------------------------
    # Initialize antibody population (Sobol instead of uniform random)
    # --------------------------
    def _init_population(self) -> List[Antibody]:
        pop = []
        sobol_points = self._sobol_sample(self.N)
        for x in sobol_points:
            fit = self._fitness(x)
            aff = self._affinity(fit)
            pop.append(Antibody(x=x, fitness=fit, affinity=aff, T=0, S=0))
        return pop

    # --------------------------
    # Cloning process
    # --------------------------
    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        if not selected:
            return []
        max_aff = max(ab.affinity for ab in selected)
        clones = []
        for ab in selected:
            ab.S += 1
            ratio = ab.affinity / (max_aff + 1e-12)
            k = max(1, int(round(self.n_clones * ratio)))
            for _ in range(k):
                clones.append(Antibody(x=ab.x.copy(), fitness=ab.fitness,
                                       affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # --------------------------
    # Mutation
    # --------------------------
    def _mutate(self, clones: List[Antibody]) -> None:
        scale_base = (self.ub - self.lb) * 0.1
        for ab in clones:
            step = scale_base * ((1.0 - ab.affinity) ** self.m)
            if step < 1e-12:
                step = 1e-12
            noise = np.random.normal(loc=0.0, scale=step, size=self.dim)
            x_new = self._clip(ab.x + noise)
            fit = self._fitness(x_new)
            aff = self._affinity(fit)
            ab.x, ab.fitness, ab.affinity = x_new, fit, aff

    # --------------------------
    # Replacement
    # --------------------------
    def _replace(self, pop: List[Antibody], offspring: List[Antibody]) -> List[Antibody]:
        for ab in pop:
            ab.T += 1
        union = pop + offspring
        union.sort(key=lambda a: a.fitness)
        return [Antibody(x=a.x, fitness=a.fitness, affinity=a.affinity, T=a.T, S=a.S)
                for a in union[:self.N]]

    # --------------------------
    # Forgetting (Sobol reinit instead of random)
    # --------------------------
    def _forgetting(self, pop: List[Antibody]) -> None:
        for i, ab in enumerate(pop):
            if ab.S == 0:
                activity = float('inf') if ab.T > 0 else 0.0
            else:
                activity = ab.T / ab.S
            if activity > self.c:
                x = self._sobol_sample(1)[0]
                fit = self._fitness(x)
                aff = self._affinity(fit)
                pop[i] = Antibody(x=x, fitness=fit, affinity=aff, T=0, S=0)

    # --------------------------
    # Synaptic Pruning
    # --------------------------
    def _prune_low_novelty(self, pop: List[Antibody], scale: float = 0.01) -> None:
        threshold = scale * np.linalg.norm(self.ub - self.lb)
        positions = np.array([ab.x for ab in pop])
        for i, ab in enumerate(pop):
            diffs = positions - ab.x
            dists = np.linalg.norm(diffs, axis=1)
            avg_dist = (np.sum(dists) - dists[i]) / (len(pop) - 1)
            if avg_dist < threshold:
                x_new = self._sobol_sample(1)[0]
                fit = self._fitness(x_new)
                aff = self._affinity(fit)
                pop[i] = Antibody(x=x_new, fitness=fit, affinity=aff, T=0, S=0)

    # --------------------------
    # Main optimization loop
    # --------------------------
    def optimize(self):
        pop = self._init_population()
        best = min(pop, key=lambda a: a.fitness)
        self.history = [best.fitness]

        for it in range(1, self.max_iters + 1):
            pop.sort(key=lambda a: a.fitness)
            selected = pop[:self.n_select]

            clones = self._clone(selected)
            self._mutate(clones)
            pop = self._replace(pop, clones)
            self._forgetting(pop)
            self._prune_low_novelty(pop)

            current_best = min(pop, key=lambda a: a.fitness)
            if current_best.fitness < best.fitness:
                best = Antibody(x=current_best.x.copy(), fitness=current_best.fitness,
                                affinity=current_best.affinity, T=current_best.T, S=current_best.S)

            self.history.append(best.fitness)

        return best.x, best.fitness, self.history
