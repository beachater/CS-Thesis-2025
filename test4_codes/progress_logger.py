# import time
# from typing import Any
# import numpy as np


# class ProgressLogger:
#     """
#     Collects per-generation metrics
#       best_fitness         convergence curve
#       mean_fitness         average fitness
#       pop_xy               population positions in first two dims for search history
#       evals                evaluation count if provided
#       time_sec             wall clock since run start
#     """
#     def __init__(self, want_positions: bool = False, max_store_xy: int = 400):
#         self.want_positions = want_positions
#         self.max_store_xy = max_store_xy
#         self.t0 = time.time()
#         self.metrics = {
#             "gen": [],
#             "best_fitness": [],
#             "mean_fitness": [],
#             "evals": [],
#             "time_sec": [],
#             # list of arrays with shape (k, 2) per generation, only when dim==2 and pop provided
#             "pop_xy": []
#         }

#     def __call__(self, **kwargs: Any) -> None:
#         # robustly fetch fields without boolean-coercing NumPy arrays
        
#         gen = kwargs.get("gen")
#         pop = kwargs.get("pop")
#         fit = kwargs.get("fitness")
#         if fit is None:
#             fit = kwargs.get("fitnesses")
#         gbest = kwargs.get("gbest")
#         fbest = kwargs.get("fbest")
#         if fbest is None:
#             fbest = kwargs.get("best_fitness")
#         evals = kwargs.get("evals")
#         if evals is None:
#             evals = kwargs.get("fevals")
#         if evals is None:
#             evals = kwargs.get("fes")

#         wrote_xy = False  # <--- define it here
        
#         # generation index fallback
#         if gen is None:
#             gen = len(self.metrics["gen"])

#         # compute best and mean from fitness array if present
#         best_val = fbest
#         mean_val = None
#         if fit is not None:
#             try:
#                 fit_arr = np.asarray(fit, dtype=float)
#                 if fit_arr.size > 0:
#                     mean_val = float(np.mean(fit_arr))
#                     if best_val is None:
#                         best_val = float(np.min(fit_arr))
#             except Exception:
#                 pass

#         # record scalars
#         self.metrics["gen"].append(int(gen))
#         self.metrics["best_fitness"].append(float(best_val) if best_val is not None else float("nan"))
#         self.metrics["mean_fitness"].append(float(mean_val) if mean_val is not None else float("nan"))
#         self.metrics["evals"].append(int(evals) if evals is not None else -1)
#         self.metrics["time_sec"].append(float(time.time() - self.t0))

#         # capture 2D positions for search history (no boolean use on arrays)
#         if self.want_positions and pop is not None:
#             try:
#                 pop_arr = np.asarray(pop, dtype=float)
#                 if pop_arr.ndim == 2 and pop_arr.shape[1] >= 2 and pop_arr.shape[0] > 0:
#                     if pop_arr.shape[0] > self.max_store_xy:
#                         idx = np.random.choice(pop_arr.shape[0], self.max_store_xy, replace=False)
#                         pop_xy = pop_arr[idx, :2]
#                     else:
#                         pop_xy = pop_arr[:, :2]
#                     self.metrics["pop_xy"].append(pop_xy.tolist())
#                 else:
#                     self.metrics["pop_xy"].append(None)
#             except Exception:
#                 self.metrics["pop_xy"].append(None)
#         else:
#             self.metrics["pop_xy"].append(None)

#          # DEBUG
#         print(f"[ProgressLogger] gen={gen} wrote_xy={wrote_xy} want_positions={self.want_positions} "
#             f"pop_shape={None if pop is None else pop_arr.shape}")
import time
from typing import Any
import numpy as np


class ProgressLogger:
    """
    Collects per-generation metrics
      best_fitness         convergence curve
      mean_fitness         average fitness
      pop_xy               2D snapshots for search-history (list of arrays or None per gen)
      evals                evaluation count if provided
      time_sec             wall clock since run start

    Combo capture policy for search-history (dim==2):
      - gens < dense_until: record FULL population every gen
      - else if gen % record_every == 0: record DOWNSAMPLED population
      - else if gen % best_only_every == 0: record BEST-ONLY point
      - else: skip (append None)
    """

    def __init__(
        self,
        want_positions: bool = False,
        max_store_xy=100,      # ✅ only 1 point per generation (usually best)
        dense_until=100,       # ✅ record full population only for the first 5 generations
        record_every=200,    # ✅ only every 200th gen gets a full snapshot
        best_only_every=20   # ✅ otherwise, only record the best every 20 generations    # otherwise, record best-only every M gens
    ):
        self.want_positions = want_positions
        self.max_store_xy = int(max_store_xy)
        self.dense_until = int(dense_until)
        self.record_every = int(record_every)
        self.best_only_every = int(best_only_every)

        self.t0 = time.time()
        self.metrics = {
            "gen": [],
            "best_fitness": [],
            "mean_fitness": [],
            "evals": [],
            "time_sec": [],
            # list of arrays (shape (k,2)) or None per generation
            "pop_xy": [],
        }

    def __call__(self, **kwargs: Any) -> None:
        # Read flexible inputs
        gen   = kwargs.get("gen")
        pop   = kwargs.get("pop")

        if "fitness" in kwargs:
            fit = kwargs["fitness"]
        elif "fitnesses" in kwargs:
            fit = kwargs["fitnesses"]
        else:
            fit = None

        if "gbest" in kwargs:
            gbest = kwargs["gbest"]
        else:
            gbest = None

        if "fbest" in kwargs:
            fbest = kwargs["fbest"]
        elif "best_fitness" in kwargs:
            fbest = kwargs["best_fitness"]
        else:
            fbest = None

        if "evals" in kwargs:
            evals = kwargs["evals"]
        elif "fevals" in kwargs:
            evals = kwargs["fevals"]
        elif "fes" in kwargs:
            evals = kwargs["fes"]
        else:
            evals = None

        # generation fallback
        if gen is None:
            gen = len(self.metrics["gen"])
        gen = int(gen)

        # Generation index fallback
        if gen is None:
            gen = len(self.metrics["gen"])
        gen = int(gen)

        # Derive best / mean fitness if possible
        best_val = fbest
        mean_val = None
        if fit is not None:
            try:
                fit_arr = np.asarray(fit, dtype=float)
                if fit_arr.size > 0:
                    mean_val = float(np.mean(fit_arr))
                    if best_val is None:
                        best_val = float(np.min(fit_arr))
            except Exception:
                pass

        self.metrics["gen"].append(gen)
        self.metrics["best_fitness"].append(float(best_val) if best_val is not None else float("nan"))
        self.metrics["mean_fitness"].append(float(mean_val) if mean_val is not None else float("nan"))
        self.metrics["evals"].append(int(evals) if evals is not None else -1)
        self.metrics["time_sec"].append(float(time.time() - self.t0))

        # ----- Search-history capture (combo) -----
        # We only store if asked, and we only need x1/x2.
        wrote_xy = False
        if self.want_positions:
            # Try full/partial population first
            if pop is not None:
                try:
                    pop_arr = np.asarray(pop, dtype=float)
                    if pop_arr.ndim == 2 and pop_arr.shape[1] >= 2 and pop_arr.shape[0] > 0:
                        if gen < self.dense_until:
                            # dense: full population (with optional downsample cap)
                            if pop_arr.shape[0] > self.max_store_xy:
                                idx = np.random.choice(pop_arr.shape[0], self.max_store_xy, replace=False)
                                pop_xy = pop_arr[idx, :2]
                            else:
                                pop_xy = pop_arr[:, :2]
                            self.metrics["pop_xy"].append(pop_xy.tolist())
                            wrote_xy = True
                        elif self.record_every > 0 and (gen % self.record_every == 0):
                            # sparse: every N gens, downsampled population
                            if pop_arr.shape[0] > self.max_store_xy:
                                idx = np.random.choice(pop_arr.shape[0], self.max_store_xy, replace=False)
                                pop_xy = pop_arr[idx, :2]
                            else:
                                pop_xy = pop_arr[:, :2]
                            self.metrics["pop_xy"].append(pop_xy.tolist())
                            wrote_xy = True
                except Exception:
                    # fall through to best-only branch
                    pass

            # If we didn't write a population frame, consider best-only
            if not wrote_xy and (self.best_only_every > 0) and (gen % self.best_only_every == 0):
                try:
                    if gbest is not None:
                        gbest = np.asarray(gbest, dtype=float)
                        if gbest.size >= 2:
                            self.metrics["pop_xy"].append([[float(gbest[0]), float(gbest[1])]])
                            wrote_xy = True
                except Exception:
                    pass

        if not wrote_xy:
            # Keep alignment with gens, but no snapshot
            self.metrics["pop_xy"].append(None)
