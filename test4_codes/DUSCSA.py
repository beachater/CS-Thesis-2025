import numpy as np

# ================================================
# Improved Clonal Selection Algorithm (DUSCSA)
# Based on Yang et al., 2023
# ================================================

def DUSCSA(
    f,                      # objective function
    bounds,                 # [(min, max), ...] for each dimension
    pop_size=50,            # D
    elite_size=10,          # N
    clone_factor=5,         # c
    mutation_rate=0.5,      # p
    crowding_factor=0.1,    # cdf
    max_iter=1000
):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Step 1: Initialize population
    population = lb + (ub - lb) * np.random.rand(pop_size, dim)
    affinity = np.array([f(x) for x in population])
    
    best_idx = np.argmin(affinity)
    best = population[best_idx].copy()
    best_aff = affinity[best_idx]
    
    # Distance threshold σ (Eq.7)
    sigma = crowding_factor * np.sqrt(np.sum((ub - lb) ** 2))
    
    for g in range(max_iter):
        # Step 2: Sort by affinity
        idx = np.argsort(affinity)
        population = population[idx]
        affinity = affinity[idx]
        
        # Top N elite antibodies
        elites = population[:elite_size]
        elite_aff = affinity[:elite_size]
        
        # Step 3: Cloning (Eq.3)
        clones = np.repeat(elites, clone_factor, axis=0)
        
        # Step 4: Mutation (Eq.6)
        for i in range(len(clones)):
            if np.random.rand() < mutation_rate:
                clones[i] += np.random.normal(0, 0.1 * (ub - lb), dim)
                clones[i] = np.clip(clones[i], lb, ub)
        
        # Step 5: Evaluate clones and select best per parent
        clone_aff = np.array([f(x) for x in clones])
        new_elites = []
        for i in range(elite_size):
            subset = clone_aff[i * clone_factor:(i + 1) * clone_factor]
            best_idx = np.argmin(subset)
            new_elites.append(clones[i * clone_factor + best_idx])
        new_elites = np.array(new_elites)
        new_elite_aff = np.array([f(x) for x in new_elites])
        
        # Step 6: Directed Update (Eq.7–9)
        d = pop_size - elite_size
        new_population = new_elites.copy()
        while len(new_population) < pop_size:
            x_rand = lb + (ub - lb) * np.random.rand(dim)
            div = np.linalg.norm(x_rand - best)
            
            if div > sigma:
                new_population = np.vstack([new_population, x_rand])
            else:
                aff_rand = f(x_rand)
                if aff_rand < best_aff:
                    new_population = np.vstack([new_population, x_rand])
                # else, regenerate a new random one next loop
        
        # Step 7: Update population and affinity
        population = new_population
        affinity = np.array([f(x) for x in population])
        
        # Step 8: Track best
        cur_best_idx = np.argmin(affinity)
        cur_best = population[cur_best_idx]
        cur_best_aff = affinity[cur_best_idx]
        if cur_best_aff < best_aff:
            best_aff = cur_best_aff
            best = cur_best.copy()
        
        # Optional progress print
        if (g + 1) % 50 == 0:
            print(f"Iter {g+1}/{max_iter}: Best = {best_aff:.6f}")
    
    return best, best_aff


# ========================
# Example usage
# ========================

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    best_sol, best_val = DUSCSA(
        rastrigin,
        bounds=[(-5.12, 5.12)] * 10,
        pop_size=50,
        elite_size=10,
        clone_factor=5,
        mutation_rate=0.5,
        crowding_factor=0.1,
        max_iter=500
    )
    print("\nBest Solution:", best_sol)
    print("Best Value:", best_val)
