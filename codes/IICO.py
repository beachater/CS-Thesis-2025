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


from benchmark import get_function_by_name


def iico(fun, max_FEs, n, dim, max_stagnation=3, seed=None):
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    _, lb, ub = fun([random.random() for _ in range(dim)])

    pop = [[0. for _ in range(dim)] for _ in range(n)]
    fitness_list = [float("inf") for _ in range(n)]

    # Parameters
    s_max = 2  # unimodal: 2, multimodal: 40
    mu = 4
    exponent = 2
    gamma = 1e-19
    s_min = 0
    sigma_initial = 0.5  # The initial value of standard deviation
    sigma_final = 0.1  # The final value of standard deviation
    beta = 100
    minimum = 0.
    # num_it = round(max_FEs / n)
    num_it = 1000
    stagnation_num = 0

    best_fitness_list = []
    FEs_counts = []
    search_history = []
    trajectory_first_dimension = []
    average_fitness = []
    history = []  # best fitness per generation


    # Initialization
    X = [random.random() for _ in range(dim)]
    for i in range(n):
        if isinstance(lb, list):
            for d in range(dim):
                X[d] = mu * X[d] * (1 - X[d])
                pop[i][d] = lb[d] + (ub[d] - lb[d]) * X[d]
        else:
            for d in range(dim):
                X[d] = mu * X[d] * (1 - X[d])
                pop[i][d] = lb + (ub - lb) * X[d]
        fitness_list[i] = fun(pop[i])[0]

    FEs = n
    FEs_counts.append(FEs)

    k = 0.25 * max_FEs * (1 + s_max) / (s_max * n)
    if isinstance(lb, list):
        M = (ub[0] - lb[0]) / 2
    else:
        M = (ub - lb) / 2  # The mean variation interval of the variable

    index = np.argmin(fitness_list)
    gbest_fitness_value = fitness_list[index]
    gbest = pop[index].copy()
    best_fitness_list.append(gbest_fitness_value)

    delta_X = [[0. for _ in range(dim)] for _ in range(n)]
    it = 1  # It's only a control parameter
    iter_current = 1  # The current iteration number

    while FEs <= max_FEs:
        Z = math.exp(-beta * it / k)
        if Z <= gamma:
            beta = -math.log(10 * gamma) * k / it
            Z = math.exp(-beta * it / k)

        # The standard deviation in the current iteration
        sigma_iter = ((k - it) / (k - 1)) ** exponent * (sigma_initial - sigma_final) + sigma_final
        alpha_iter = 10 * math.log(M) * Z

        f_best = min(fitness_list)
        f_worst = max(fitness_list)

        if f_best == f_worst:
            NF = [1. for _ in range(n)]
        else:
            # The normalized fitness of each solution
            NF = []
            for i in range(n):
                NF.append((fitness_list[i] - f_worst) / (f_best - f_worst))

        y = round(n * (98 * (1 - it / k) + 2) / 100)
        if stagnation_num > max_stagnation and y > 1:
            y -= 1
            it = round(k * (1 - (100 * y / n - 2) / 98))
            stagnation_num = 0
        n_iter = max(y, 1)

        sorted_ind = np.argsort(NF)
        # The set of selected elite solutions
        ES = sorted_ind[:n_iter]
        F1 = []
        for d in range(dim):
            sec = 0.
            for ES_i in ES:
                sec = sec + NF[ES_i] * pop[ES_i][d]
            F1.append(sec / n_iter)

        E = []
        for i in range(n):
            r = random.random()
            TT_i = [F1[d] * r for d in range(dim)]
            R = np.linalg.norm([pop[i][d] - TT_i[d] for d in range(dim)])
            E.append([(TT_i[d] - pop[i][d]) / (R + np.spacing(1)) for d in range(dim)])

        A = []
        for i in range(n):
            A.append([20 * alpha_iter * E[i][d] for d in range(dim)])

        XL = []
        FL = []
        XL_delta_X = []
        XB = []
        FB = []
        XB_delta_X = []

        # Clone
        for i in range(n):
            # The number of offsprings generated from parent i
            S = math.floor(s_min + (s_max - s_min) * NF[i])

            # Update the number of fitness evaluations
            FEs = FEs + S

            for j in range(S):
                if random.random() < sigma_iter:
                    X_temp = [pop[i][d] + alpha_iter * random.gauss(0, 1) for d in range(dim)]
                    space_bound(X_temp, dim, lb, ub)
                    X_temp_fit = fun(X_temp)[0]
                    delta_temp = [random.random() * delta_X[i][d] + A[i][d] for d in range(dim)]
                    XL.append(X_temp)
                    FL.append(X_temp_fit)
                    XL_delta_X.append(delta_temp)
                else:
                    delta_temp = [random.random() * delta_X[i][d] + A[i][d] for d in range(dim)]
                    X_temp = [pop[i][d] + delta_temp[d] for d in range(dim)]
                    space_bound(X_temp, dim, lb, ub)
                    X_temp_fit = fun(X_temp)[0]

                    if n_iter == 1:
                        quasi_reflected = [0. for _ in range(dim)]
                        if isinstance(lb, list):
                            for d in range(dim):
                                quasi_reflected[d] = random.uniform((lb[d] + ub[d]) / 2, X_temp[d])
                        else:
                            for d in range(dim):
                                quasi_reflected[d] = random.uniform((lb + ub) / 2, X_temp[d])
                        quasi_reflected_fit = fun(quasi_reflected)[0]
                        if quasi_reflected_fit < X_temp_fit:
                            X_temp = quasi_reflected
                            X_temp_fit = quasi_reflected_fit
                    else:
                        quasi_opposite = [0. for _ in range(dim)]
                        if isinstance(lb, list):
                            for d in range(dim):
                                opposite = lb[d] + ub[d] - X_temp[d]
                                quasi_opposite[d] = random.uniform((lb[d] + ub[d]) / 2, opposite)
                        else:
                            for d in range(dim):
                                opposite = lb + ub - X_temp[d]
                                quasi_opposite[d] = random.uniform((lb + ub) / 2, opposite)
                        quasi_opposite_fit = fun(quasi_opposite)[0]
                        if quasi_opposite_fit < X_temp_fit:
                            X_temp = quasi_opposite
                            X_temp_fit = quasi_opposite_fit
                    FEs += 1
                    XB.append(X_temp)
                    FB.append(X_temp_fit)
                    XB_delta_X.append(delta_temp)

        costs, X, delta_X_temp = omit_extra(fitness_list, pop, delta_X)
        FL, XL, XL_delta_X = omit_extra(FL, XL, XL_delta_X)
        FB, XB, XB_delta_X = omit_extra(FB, XB, XB_delta_X)

        NL = min(math.ceil(sigma_iter * n), len(XL))
        u = n - NL
        NB = min(math.ceil(u * 0.9), len(XB))
        NE = min(u - NB, len(X))
        if NE == 0:
            if NB > 0:
                NB = NB - 1
            else:
                NL = NL - 1
            NE = 1

        EX = []
        E_costs = []
        E_delta = []

        for i in range(n - (NB + NE + NL)):
            if isinstance(lb, list):
                EX.append([random.uniform(lb[d], ub[d]) for d in range(dim)])
            else:
                EX.append([random.uniform(lb, ub) for _ in range(dim)])
            E_costs.append(fun(EX[i])[0])
            FEs = FEs + 1
            E_delta.append([0. for _ in range(dim)])

        pop[:NE] = X[:NE]
        pop[NE:NE + NB] = XB[:NB]
        pop[NE + NB: NE + NB + NL] = XL[:NL]
        pop[NE + NB + NL: NE + NB + NL + len(EX)] = EX
        fitness_list[:NE] = costs[:NE]
        fitness_list[NE:NE + NB] = FB[:NB]
        fitness_list[NE + NB: NE + NB + NL] = FL[:NL]
        fitness_list[NE + NB + NL: NE + NB + NL + len(EX)] = E_costs
        delta_X[:NE] = delta_X[:NE]
        delta_X[NE:NE + NB] = XB_delta_X[:NB]
        delta_X[NE + NB: NE + NB + NL] = XL_delta_X[:NL]
        delta_X[NE + NB + NL: NE + NB + NL + len(EX)] = E_delta

        index = np.argmin(fitness_list)

        if fitness_list[index] < gbest_fitness_value:
            gbest = pop[index].copy()
            gbest_fitness_value = fitness_list[index]
            stagnation_num = 0
        else:
            stagnation_num += 1

        best_fitness_list.append(gbest_fitness_value)
        FEs_counts.append(FEs)

        it = it + 1
        iter_current += 1

        if FEs >= max_FEs:
            break

        if gbest_fitness_value == minimum:
            break

    if iter_current < num_it:
        FEs_counts.extend([FEs for _ in range(num_it - iter_current)])
        best_fitness_list.extend([gbest_fitness_value for _ in range(num_it - iter_current)])
    else:
        L = len(best_fitness_list)
        for i in range(num_it):
            ind = round(i * L / num_it)
            best_fitness_list[i] = best_fitness_list[ind]
            FEs_counts[i] = FEs_counts[ind]
    best_fitness_list = best_fitness_list[:num_it]
    FEs_counts = FEs_counts[:num_it]
    # return best_fitness_list, gbest
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
