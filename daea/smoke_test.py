"""
Quick Smoke Test for DAEA + CEC2022 DMMO Benchmark
This runs a tiny version (2 envs, 1 run, small pop) just to verify
everything works end-to-end: initialization, updates, fitness eval,
archive logic, and PR computation.
"""

import numpy as np
from daea import DAEAState, daea_epoch
from cectest import DMMOProblem

def smoke_test_daea():
    print("=== Smoke Test: DAEA + CEC2022 DMMO ===")
    
    # parameters for quick test
    problem_id = "F1"
    change_mode = "C1"
    dim = 5
    runs = 1
    envs = 2
    pop_size = 10
    F, CR = 0.6, 0.9
    eps_f_levels = [1e-3, 1e-4]
    
    rng = np.random.default_rng(42)
    prob = DMMOProblem(problem_id, change_mode=change_mode, dim=dim, seed=42)
    daea_state = DAEAState(max_archive=50)
    
    # initial population
    pop = daea_state.inject_population(pop_size, dim, rng, (-5,5), reuse_rate=0.0)
    fit = prob.evaluate(pop)
    print(f"Initial best fitness: {fit.max():.4f}")
    
    for e in range(envs):
        prob.update_environment()
        pop = daea_state.inject_population(pop_size, dim, rng, (-5,5), reuse_rate=0.5)
        fit = prob.evaluate(pop)
        print(f"\nEnvironment {e+1}:")
        print(f"  Peaks: {len(prob.get_global_centers())}")
        
        # 1 quick generation
        pop, fit = daea_epoch(pop, fit, prob, rng, daea_state, F=F, CR=CR)
        print(f"  Best fitness after epoch: {fit.max():.4f}")
        
        # check found peaks
        for eps in eps_f_levels:
            found = prob.count_found_peaks(pop, eps_d=0.05, eps_f=eps)
            print(f"  Found peaks (ε_f={eps}): {found}")
        
        daea_state.update_archive(pop, fit, niche_radius=0.1)
        print(f"  Archive size after update: {len(daea_state.archive_X) if daea_state.archive_X is not None else 0}")
    
    print("\n✅ Smoke test completed successfully.\n")

if __name__ == "__main__":
    smoke_test_daea()
