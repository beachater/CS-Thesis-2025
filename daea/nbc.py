import numpy as np
from scipy.spatial.distance import cdist


def nbc_cluster_species(init_pop, fitness, min_popsize):
    """
    Python port of NBC.m
    init_pop: (N, D)
    fitness:  (N,)
    returns:
      species_list: list of dicts { "seed": int, "idx": np.array([...]), "len": int }
      species_order: order by species size desc (roughly groups)
      species_sizes: sizes
      species_best_vals: best value per species
    """

    pop = np.asarray(init_pop, dtype=float)
    N, D = pop.shape

    # nbc array [start, endParent, distanceToParent]
    # In MATLAB:
    # nbc(i,1)=i; nbc(i,2)=parent idx OR -1 if seed; nbc(i,3)=distance
    nbc_info = np.zeros((N, 3), dtype=float)
    nbc_info[:, 0] = np.arange(N)
    nbc_info[0, 1] = -1
    nbc_info[0, 2] = 0.0

    if D == 5:
        # vectorized path
        arrdis = cdist(pop, pop)  # (N,N)
        arrdis = np.tril(arrdis, k=0)
        arrdis = arrdis + np.triu(np.full((N, N), np.inf), k=1)
        # for each row, choose min of existing prior rows
        u = np.min(arrdis, axis=1)
        v = np.argmin(arrdis, axis=1)
        nbc_info[1:, 1] = v[1:]
        nbc_info[1:, 2] = u[1:]
    else:
        # incremental nearest-better
        for i in range(1, N):
            dist_i = cdist(pop[i : i + 1, :], pop[:i, :])[0]  # length i
            j = np.argmin(dist_i)
            nbc_info[i, 1] = j
            nbc_info[i, 2] = dist_i[j]

    # factor = 4 - log(dim)
    factor = 4.0 - np.log(D)
    mean_dis = factor * np.mean(nbc_info[1:, 2])
    min_num_edge = min(10, N)

    # choose threshold
    # if enough edges above mean_dis
    mask_big = nbc_info[:, 2] > mean_dis
    if np.sum(mask_big) >= min_num_edge:
        nbc_info[mask_big, 1] = -1
        nbc_info[mask_big, 2] = 0.0
    else:
        # sort descending by distance, pick min_num_edge-th largest
        order_desc = np.argsort(-nbc_info[:, 2])
        cutoff = nbc_info[order_desc[min_num_edge - 1], 2]
        mask_big2 = nbc_info[:, 2] >= cutoff
        nbc_info[mask_big2, 1] = -1
        nbc_info[mask_big2, 2] = 0.0

    # "seeds" are rows whose parent = -1
    seeds = np.where(nbc_info[:, 1] == -1)[0]

    # for each point i, follow parents until seed
    # record membership
    seed_of = np.zeros(N, dtype=int)
    for i in range(N):
        j = int(nbc_info[i, 1])
        k = j
        while j != -1:
            k = j
            j = int(nbc_info[j, 1])
        if k == -1:
            seed_of[i] = i
        else:
            seed_of[i] = k

    # build species structs
    species_list = []
    for s in seeds:
        idxs = np.where(seed_of == s)[0]
        species_list.append(
            {
                "seed": int(s),
                "idx": idxs,
                "len": len(idxs),
            }
        )

    # sort species by length desc (this is like species_arr + sort_index in MATLAB)
    sizes = [sp["len"] for sp in species_list]
    sort_order = np.argsort(-np.array(sizes))
    species_list = [species_list[i] for i in sort_order]

    # also compute best fitness in each species (take fitness of seed)
    species_best_vals = [float(fitness[sp["seed"]]) for sp in species_list]
    species_sizes = [sp["len"] for sp in species_list]

    return species_list, sort_order, species_sizes, species_best_vals
