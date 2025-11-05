import numpy as np


class TrackRecord:
    """
    Holds history of group centers ("xmean") across time,
    like track_record.m and the tracker array in Main.m
    """

    def __init__(self):
        self.history = []  # list of arrays (n_groups, D) snapshots

    def add_snapshot(self, groups):
        if groups is None or len(groups) == 0:
            return
        xs = []
        for g in groups:
            xs.append(np.asarray(g["xmean"], dtype=float))
        if len(xs) > 0:
            self.history.append(np.vstack(xs))

    def to_array(self):
        if len(self.history) == 0:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(self.history)
