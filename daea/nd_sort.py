import numpy as np
from platemo_ndsort import ndsort  # we'll define this below


def fast_nd_sort(fitness, rho):
    """
    fitness: shape (lambda,)
    rho:     shape (lambda,)

    Returns sorted_indices where:
      - primarily by nondominated rank (rho=0 points form rank 1 or similar)
      - then by -fitness (desc)
    Mirrors Fast_ND_SORT.m
    """

    fitness = np.asarray(fitness, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # obj(1,:) = -fitness
    # obj(2,:) = rho
    obj = np.vstack([-fitness, rho])  # shape (2, N)
    N = obj.shape[1]

    # split by rho==0 vs !=0 like MATLAB
    mask_rho0 = (obj[1, :] == 0)
    obj1 = obj[:, mask_rho0]
    obj2 = obj[:, ~mask_rho0]

    # init front ranks
    F = np.ones((N,), dtype=float)

    if obj2.shape[1] > 0:
        # NDSort on obj2^T
        F2 = ndsort(obj2.T, nSort=obj2.shape[1])
        # place F2+1 into F where rho!=0
        F[~mask_rho0] = F[~mask_rho0] + F2

    # build sortable keys:
    # sort by (F, obj1) where obj1 = -fitness = obj[0,:]
    # ascending F, ascending obj[0,:] means best F, highest fitness first
    sort_mat = np.vstack([F, obj[0, :]])  # (2,N)
    # lexsort uses last row as primary, so reverse order
    # We want primary: F asc, then obj[0] asc
    idx = np.lexsort((sort_mat[1, :], sort_mat[0, :]))
    return idx


def ndsort(pop_obj, nSort=None):
    """
    Lightweight ENS-SS style NDSort for 2 objectives.
    pop_obj: (N, M) where M=2
    returns frontNo for each row
    """
    pop_obj = np.asarray(pop_obj, dtype=float)
    N, M = pop_obj.shape
    if nSort is None:
        nSort = np.inf

    # This is a simple O(N^2) nondom sort for small lambda.
    # frontNo[i] = front rank starting at 1.
    frontNo = np.full(N, np.inf, dtype=float)
    maxFNo = 0

    selected_count = 0
    while selected_count < min(nSort, N):
        maxFNo += 1
        for i in range(N):
            if frontNo[i] < np.inf:
                continue
            dominated = False
            for j in range(N):
                if i == j:
                    continue
                if frontNo[j] == maxFNo:
                    # check if pop_obj[j] dominates pop_obj[i]
                    # domination for minimization:
                    # j dominates i if j <= i on all dims AND j < i on at least one
                    a = pop_obj[j, :]
                    b = pop_obj[i, :]
                    if np.all(a <= b) and np.any(a < b):
                        dominated = True
                        break
            if not dominated:
                frontNo[i] = maxFNo
                selected_count += 1
    return frontNo
