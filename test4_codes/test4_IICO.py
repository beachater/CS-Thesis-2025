"""
The implementation of the intelligent chaotic clonal optimizer.

Date: 2021.11.30
Author: Jiahao Zhang
"""


import math
import random

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook




def iico(fun,
         bounds,            # List[Tuple[lo, hi]]
         dim,
         max_FEs,
         pop_size,
         bench_func=None,   # kept for signature parity; not used here
         seed=None,
         progress=None):
    """
    IICO optimizer with progress callback parity to FCSA/TSD.

    Enforces BOTH:
      - max iterations (num_it = 1000)
      - max function evaluations (max_FEs, e.g., 350_000)

    Calls: progress(gen, pop, fitness, best_fitness, gbest, evals) once per gen.
    """
    import math, random
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Normalize bounds -> arrays (lb, ub) length dim
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim == 2 and bounds.shape[0] == dim:
        lb = bounds[:, 0]
        ub = bounds[:, 1]
    else:
        lb = np.full(dim, float(bounds[0]), dtype=float)
        ub = np.full(dim, float(bounds[1]), dtype=float)

    def space_bound(vec):
        v = np.asarray(vec, dtype=float)
        np.clip(v, lb, ub, out=v)
        return v.tolist()

    n = int(pop_size)

    # ----- parameters -----
    s_max = 2
    mu = 4
    exponent = 2
    gamma = 1e-19
    s_min = 0
    sigma_initial = 0.5
    sigma_final = 0.1
    beta = 100
    minimum = 0.0
    num_it = 1000               # <- hard cap on iterations
    max_stagnation = 3
    stagnation_num = 0

    best_fitness_list = []
    FEs_counts = []
    history = []

    # ----- init population -----
    X = [random.random() for _ in range(dim)]
    pop = [[0.0] * dim for _ in range(n)]
    fitness_list = [float("inf")] * n

    for i in range(n):
        for d in range(dim):
            X[d] = mu * X[d] * (1 - X[d])
            pop[i][d] = lb[d] + (ub[d] - lb[d]) * X[d]
        fitness_list[i] = fun(pop[i])[0]

    FEs = n
    FEs_counts.append(FEs)

    # k as in original paper/code (depends on FE budget & pop size)
    k = 0.25 * max_FEs * (1 + s_max) / (s_max * n)
    # mean variation interval (use first dim just like original)
    M = float((ub[0] - lb[0]) / 2.0)

    idx = int(np.argmin(fitness_list))
    gbest_fitness_value = float(fitness_list[idx])
    gbest = pop[idx].copy()
    best_fitness_list.append(gbest_fitness_value)

    delta_X = [[0.0] * dim for _ in range(n)]
    it = 1
    iter_current = 1

    # ---- gen 0 snapshot ----
    if progress is not None:
        try:
            pop_arr = np.array(pop, dtype=float)
            fit_arr = np.array(fitness_list, dtype=float)
            progress(gen=0,
                     pop=pop_arr,
                     fitness=fit_arr,
                     best_fitness=gbest_fitness_value,
                     gbest=np.array(gbest, dtype=float),
                     evals=FEs)
        except Exception:
            pass

    # ================== main loop with BOTH caps ==================
    while (FEs < max_FEs) and (iter_current <= num_it):
        Z = math.exp(-beta * it / k)
        if Z <= gamma:
            beta = -math.log(10 * gamma) * k / max(it, 1)
            Z = math.exp(-beta * it / k)

        sigma_iter = ((k - it) / max(k - 1, 1)) ** exponent * (sigma_initial - sigma_final) + sigma_final
        alpha_iter = 10 * math.log(max(M, 1e-12)) * Z

        f_best = min(fitness_list)
        f_worst = max(fitness_list)
        if f_best == f_worst:
            NF = [1.0] * n
        else:
            NF = [(fitness_list[i] - f_worst) / (f_best - f_worst) for i in range(n)]

        y = round(n * (98 * (1 - it / k) + 2) / 100)
        if stagnation_num > max_stagnation and y > 1:
            y -= 1
            it = round(k * (1 - (100 * y / n - 2) / 98))
            stagnation_num = 0
        n_iter = max(y, 1)

        # elite centroid F1
        sorted_ind = np.argsort(NF)
        ES = sorted_ind[:n_iter]
        F1 = []
        for d in range(dim):
            acc = 0.0
            for ES_i in ES:
                acc += NF[ES_i] * pop[ES_i][d]
            F1.append(acc / n_iter)

        # E and A
        E = []
        for i in range(n):
            r = random.random()
            TT_i = [F1[d] * r for d in range(dim)]
            R = np.linalg.norm([pop[i][d] - TT_i[d] for d in range(dim)])
            E.append([(TT_i[d] - pop[i][d]) / (R + np.spacing(1)) for d in range(dim)])
        A = [[20 * alpha_iter * E[i][d] for d in range(dim)] for i in range(n)]

        XL, FL, XL_delta_X = [], [], []
        XB, FB, XB_delta_X = [], [], []

        # ----- clone & variation -----
        for i in range(n):
            S = math.floor(s_min + (s_max - s_min) * NF[i])

            # FE guard: don't overshoot FE budget
            if FEs >= max_FEs:
                break
            take = min(S, max(0, max_FEs - FEs))
            FEs += take

            for _ in range(take):
                if random.random() < sigma_iter:
                    X_temp = [pop[i][d] + alpha_iter * random.gauss(0, 1) for d in range(dim)]
                    X_temp = space_bound(X_temp)
                    X_temp_fit = fun(X_temp)[0]
                    delta_temp = [random.random() * delta_X[i][d] + A[i][d] for d in range(dim)]
                    XL.append(X_temp); FL.append(X_temp_fit); XL_delta_X.append(delta_temp)
                else:
                    delta_temp = [random.random() * delta_X[i][d] + A[i][d] for d in range(dim)]
                    X_temp = [pop[i][d] + delta_temp[d] for d in range(dim)]
                    X_temp = space_bound(X_temp)
                    X_temp_fit = fun(X_temp)[0]

                    if n_iter == 1:
                        mid = (lb + ub) / 2.0
                        quasi_reflected = [random.uniform(mid[d], X_temp[d]) for d in range(dim)]
                        quasi_reflected_fit = fun(quasi_reflected)[0]
                        if quasi_reflected_fit < X_temp_fit:
                            X_temp = quasi_reflected
                            X_temp_fit = quasi_reflected_fit
                    else:
                        mid = (lb + ub) / 2.0
                        opp = (lb + ub - np.array(X_temp))
                        quasi_opposite = [random.uniform(mid[d], opp[d]) for d in range(dim)]
                        quasi_opposite_fit = fun(quasi_opposite)[0]
                        if quasi_opposite_fit < X_temp_fit:
                            X_temp = quasi_opposite
                            X_temp_fit = quasi_opposite_fit

                    # extra FE guard on this branch
                    if FEs >= max_FEs:
                        break
                    FEs += 1
                    XB.append(X_temp); FB.append(X_temp_fit); XB_delta_X.append(delta_temp)

            if FEs >= max_FEs:
                break

        # selection helpers
        def omit_extra(costs, XX, dX):
            arr = list(zip(costs, XX, dX))
            arr.sort(key=lambda t: t[0])
            return [a[0] for a in arr], [a[1] for a in arr], [a[2] for a in arr]

        costs, Xs, dXtemp = omit_extra(fitness_list, pop, delta_X)
        FL, XL, XL_delta_X = omit_extra(FL, XL, XL_delta_X)
        FB, XB, XB_delta_X = omit_extra(FB, XB, XB_delta_X)

        NL = min(math.ceil(sigma_iter * n), len(XL))
        u = n - NL
        NB = min(math.ceil(u * 0.9), len(XB))
        NE = min(u - NB, len(Xs))
        if NE == 0:
            if NB > 0:
                NB -= 1
            else:
                NL -= 1
            NE = 1

        # random injections (respect FE cap)
        EX, E_costs, E_delta = [], [], []
        need = n - (NB + NE + NL)
        for _ in range(need):
            if FEs >= max_FEs:
                break
            vec = [random.uniform(lb[d], ub[d]) for d in range(dim)]
            EX.append(vec)
            E_costs.append(fun(vec)[0])
            FEs += 1
            E_delta.append([0.0] * dim)

        # rebuild population
        pop[:NE] = Xs[:NE]
        pop[NE:NE+NB] = XB[:NB]
        pop[NE+NB:NE+NB+NL] = XL[:NL]
        pop[NE+NB+NL:NE+NB+NL+len(EX)] = EX

        fitness_list[:NE] = costs[:NE]
        fitness_list[NE:NE+NB] = FB[:NB]
        fitness_list[NE+NB:NE+NB+NL] = FL[:NL]
        fitness_list[NE+NB+NL:NE+NB+NL+len(EX)] = E_costs

        delta_X[:NE] = dXtemp[:NE]
        delta_X[NE:NE+NB] = XB_delta_X[:NB]
        delta_X[NE+NB:NE+NB+NL] = XL_delta_X[:NL]
        delta_X[NE+NB+NL:NE+NB+NL+len(EX)] = E_delta

        # update best
        idx = int(np.argmin(fitness_list))
        if fitness_list[idx] < gbest_fitness_value:
            gbest = pop[idx].copy()
            gbest_fitness_value = float(fitness_list[idx])
            stagnation_num = 0
        else:
            stagnation_num += 1

        best_fitness_list.append(gbest_fitness_value)
        FEs_counts.append(FEs)
        history.append(gbest_fitness_value)

        # progress per generation
        if progress is not None:
            try:
                pop_arr = np.array(pop, dtype=float)
                fit_arr = np.array(fitness_list, dtype=float)
                progress(gen=iter_current,
                         pop=pop_arr,
                         fitness=fit_arr,
                         best_fitness=gbest_fitness_value,
                         gbest=np.array(gbest, dtype=float),
                         evals=FEs)
            except Exception:
                pass

        it += 1
        iter_current += 1

        # stop if either cap is reached
        if FEs >= max_FEs or iter_current > num_it or gbest_fitness_value == minimum:
            break
    # ================== end loop ==================

    # Keep legacy outputs trimmed/padded to exactly num_it points
    if iter_current <= num_it:
        best_fitness_list.extend([gbest_fitness_value] * (num_it - len(best_fitness_list)))
        FEs_counts.extend([FEs] * (num_it - len(FEs_counts)))
    else:
        L = len(best_fitness_list)
        for i in range(num_it):
            ind = round(i * L / num_it)
            best_fitness_list[i] = best_fitness_list[ind]
            FEs_counts[i] = FEs_counts[ind]

    best_fitness_list = best_fitness_list[:num_it]
    FEs_counts = FEs_counts[:num_it]
    return best_fitness_list, gbest, {"history": best_fitness_list}


def space_bound(X, dim, lb, ub):
    """
    Solutions that go out of the search space are reinitialized randomly.
    """
    if isinstance(lb, list):
        for d in range(dim):
            if X[d] < lb[d] or X[d] > ub[d]:
                X[d] = random.random() * (ub[d] - lb[d]) + lb[d]
    else:
        for d in range(dim):
            if X[d] < lb or X[d] > ub:
                X[d] = random.random() * (ub - lb) + lb


def omit_extra(costs, X, delta_X):
    """
    Delete duplicates of the parameters and sort them.
    """
    _, unique_ind = np.unique(costs, return_index=True)
    costs = [costs[ind] for ind in unique_ind]
    X = [X[ind] for ind in unique_ind]
    delta_X = [delta_X[ind] for ind in unique_ind]
    return costs, X, delta_X


def iico_test():
    fun = get_function_by_name('Sphere')

    dim = 2
    n = 30
    max_FEs = 350000

    best_fitness_list = iico(fun, max_FEs, n, dim)[0]
    print("best fitness value: ", best_fitness_list[-1])
    plt.semilogy(np.linspace(0, maxIter, maxIter), best_fitness_list)
    plt.legend(["IICO"], fontsize=9)
    plt.title(fun.__name__)
    plt.show()


if __name__ == '__main__':
    iico_test()
