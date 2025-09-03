#-----------------
# Forgetting Based Clonal Selection Algorithm
#-----------------------------
import numpy as np
import random
from dataclasses import dataclass
from typing import Callable, Tuple, List

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
class FSCSASP:
    """Forgetful Clonal Selection Algorithm (practical Python version).
    Implements Yang et al. (2020) flow:
    selection -> cloning -> variation -> replacement -> forgetting (Rac1 activity).
    """

    def __init__(self, func: Callable[[np.ndarray], Tuple[float,float,float]],
                dim: int, N: int=50, n_select: int=10, n_clones: int=5,
                m: float=2.0, c: float=3.0, max_iters: int=1000, seed: int=None):
        # Benchmark function to minimize
        self.func = func
        # Problem dimensionality
        self.dim = dim
        # Population size (number of antibodies)
        self.N = N
        # Number of best antibodies selected each generation
        self.n_select = n_select
        # Maximum clone count per selected antibody
        self.n_clones = n_clones
        # Degree of variation (controls mutation strength)
        self.m = m
        # Forgetting threshold (Rac1 activity level)
        self.c = c
        # Maximum iterations (termination)
        self.max_iters = max_iters

        # Optional: fix random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Get function bounds (lb, ub) from benchmark function definition
        f0, lb, ub = self.func(np.zeros(dim))
        self.lb = lb
        self.ub = ub

        # Store convergence curve
        self.history = []

    # --------------------------
    # Fitness evaluation
    # --------------------------
    def _fitness(self, x: np.ndarray) -> float:
        y, _, _ = self.func(x)   # benchmark returns (fitness, lb, ub)
        return float(y)

    # --------------------------
    # Affinity measure
    # --------------------------
    def _affinity(self, fitness: float) -> float:
        # Higher affinity = better antibody (lower fitness)
        return 1.0 / (1.0 + max(fitness, 0.0))

    # --------------------------
    # Boundary handling
    # --------------------------
    def _clip(self, x):
        # Force values back into search space [lb, ub]
        return np.clip(x, self.lb, self.ub)

    # --------------------------
    # Initialize antibody population
    # --------------------------
    def _init_population(self) -> List[Antibody]:
        pop = []
        for _ in range(self.N):
            # Random position within bounds
            x = np.random.uniform(self.lb, self.ub, self.dim)
            fit = self._fitness(x)
            aff = self._affinity(fit)
            # New antibody starts with T=0 (just born), S=0 (never cloned yet)
            pop.append(Antibody(x=x, fitness=fit, affinity=aff, T=0, S=0))
        return pop

    # --------------------------
    # Cloning process
    # --------------------------
    def _clone(self, selected: List[Antibody]) -> List[Antibody]:
        # Clone count proportional to normalized affinity
        if not selected:
            return []
        max_aff = max(ab.affinity for ab in selected)
        clones = []
        for ab in selected:
            ab.S += 1  # memory intensity increases when selected
            ratio = ab.affinity / (max_aff + 1e-12)
            k = max(1, int(round(self.n_clones * ratio)))  # at least one clone
            for _ in range(k):
                # Copy antibody (clones start identical to parent)
                clones.append(Antibody(x=ab.x.copy(), fitness=ab.fitness,
                                    affinity=ab.affinity, T=ab.T, S=ab.S))
        return clones

    # --------------------------
    # Mutation (variation step)
    # --------------------------
    def _mutate(self, clones: List[Antibody]) -> None:
        # Gaussian noise scaled by affinity and variation degree
        scale_base = (self.ub - self.lb) * 0.1  # mutation base scale = 10% of range
        for ab in clones:
            # Better antibodies mutate less (since affinity ~ 1/fitness)
            step = scale_base * ((1.0 - ab.affinity) ** self.m)
            if step < 1e-12:
                step = 1e-12  # safeguard
            noise = np.random.normal(loc=0.0, scale=step, size=self.dim)
            x_new = self._clip(ab.x + noise)
            fit = self._fitness(x_new)
            aff = self._affinity(fit)
            # Replace clone with mutated values
            ab.x, ab.fitness, ab.affinity = x_new, fit, aff

    # --------------------------
    # Replacement (survivor selection)
    # --------------------------
    def _replace(self, pop: List[Antibody], offspring: List[Antibody]) -> List[Antibody]:
        # Increase survival time of old antibodies
        for ab in pop:
            ab.T += 1
        # Merge parents and offspring, keep best N
        union = pop + offspring
        union.sort(key=lambda a: a.fitness)
        new_pop = [Antibody(x=a.x, fitness=a.fitness, affinity=a.affinity, T=a.T, S=a.S)
                for a in union[:self.N]]
        return new_pop

    # --------------------------
    # Forgetting mechanism (Rac1 activity)
    # --------------------------
    def _forgetting(self, pop: List[Antibody]) -> None:
        # Rac1 activity = survival time / memory intensity
        for i, ab in enumerate(pop):
            if ab.S == 0:  # never selected before
                activity = float('inf') if ab.T > 0 else 0.0
            else:
                activity = ab.T / ab.S
            if activity > self.c:
                # Forget this antibody: reinitialize randomly
                x = np.random.uniform(self.lb, self.ub, self.dim)
                fit = self._fitness(x)
                aff = self._affinity(fit)
                pop[i] = Antibody(x=x, fitness=fit, affinity=aff, T=0, S=0)

        # --------------------------
    # Synaptic Pruning: Low-Novelty Filter
    # --------------------------
    def _prune_low_novelty(self, pop: List[Antibody], scale: float = 0.01) -> None:
        threshold = scale * np.linalg.norm(self.ub - self.lb)
        positions = np.array([ab.x for ab in pop])
        for i, ab in enumerate(pop):
            diffs = positions - ab.x
            dists = np.linalg.norm(diffs, axis=1)
            avg_dist = (np.sum(dists) - dists[i]) / (len(pop) - 1)
            if avg_dist < threshold:
                x_new = np.random.uniform(self.lb, self.ub, self.dim)
                fit = self._fitness(x_new)
                aff = self._affinity(fit)
                pop[i] = Antibody(x=x_new, fitness=fit, affinity=aff, T=0, S=0)



    # --------------------------
    # Main optimization loop
    # --------------------------
    def optimize(self):
        # 1. Initialize population
        pop = self._init_population()
        best = min(pop, key=lambda a: a.fitness)
        self.history = [best.fitness]

        # 2. Repeat for max_iters
        for it in range(1, self.max_iters + 1):
            # Selection: choose best antibodies
            pop.sort(key=lambda a: a.fitness)
            selected = pop[:self.n_select]

            # Cloning
            clones = self._clone(selected)

            # Variation (mutation of clones)
            self._mutate(clones)

            # Replacement (elitist: keep best N from parents+clones)
            pop = self._replace(pop, clones)

            # Forgetting (biological forgetting mechanism)
            self._forgetting(pop)
                        
                
            # Synaptic pruning: remove low-novelty antibodies every 50 iterations
            if it % 50 == 0:
                before = np.array([ab.x for ab in pop])
                self._prune_low_novelty(pop)
                after = np.array([ab.x for ab in pop])
                delta = np.mean(np.linalg.norm(before - after, axis=1))
                print(f"Iteration {it}: Avg position change after pruning = {delta:.4f}")


            # Track global best solution
            current_best = min(pop, key=lambda a: a.fitness)
            if current_best.fitness < best.fitness:
                best = Antibody(x=current_best.x.copy(), fitness=current_best.fitness,
                                affinity=current_best.affinity, T=current_best.T, S=current_best.S)

            self.history.append(best.fitness)

        # Return: best solution vector, fitness, and convergence curve
        return best.x, best.fitness, self.history
