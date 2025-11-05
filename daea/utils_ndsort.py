import numpy as np


def select_two_groups_for_next_iter(groups, bestmem_set):
    """
    Simplified version of MATLAB logic for choosing first_idx and second_idx.
    groups: list of group dicts, each having "bestval", "bestmem"
    return (idx_best, idx_second or None)
    """
    if len(groups) == 0:
        return None, None
    # sort groups by bestval desc
    order = np.argsort([-g["bestval"] for g in groups])
    idx_best = order[0]
    if len(order) > 1:
        idx_second = order[1]
    else:
        idx_second = None
    return idx_best, idx_second
