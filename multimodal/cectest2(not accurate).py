import numpy as np
import math
from typing import List, Tuple, Optional, Dict

# =========================
# Base benchmark functions
# =========================

def sphere(z: np.ndarray) -> float:
    return float(np.sum(z**2))

def rastrigin(z: np.ndarray) -> float:
    D = z.size
    return float(10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))

def ackley(z: np.ndarray) -> float:
    D = z.size
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    s1 = np.sum(z**2)
    s2 = np.sum(np.cos(c * z))
    return float(
        -a * math.exp(-b * math.sqrt(s1 / D))
        - math.exp(s2 / D)
        + a
        + math.e
    )

def griewank(z: np.ndarray) -> float:
    D = z.size
    return float(
        np.sum(z**2) / 4000.0
        - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
        + 1.0
    )

def weierstrass(z: np.ndarray, a=0.5, b=3.0, kmax=20) -> float:
    D = z.size
    s = 0.0
    for i in range(D):
        xi = z[i]
        for k in range(kmax + 1):
            s += a**k * math.cos(2 * math.pi * b**k * (xi + 0.5))
    s2 = 0.0
    for k in range(kmax + 1):
        s2 += a**k * math.cos(2 * math.pi * b**k * 0.5)
    return float(s - D * s2)

def eggr(z: np.ndarray) -> float:
    # Hybrid Rosenbrock + Griewank-style mix
    D = z.size
    s = 0.0
    for i in range(D - 1):
        xi = z[i]
        xnext = z[i + 1]
        s += 100.0 * (xnext - xi**2) ** 2 + (xi - 1.0) ** 2
    return float(
        s
        + np.sum(z**2) / 4000.0
        - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
        + 1.0
    )

BASE_FUNS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "griewank": griewank,
    "weierstrass": weierstrass,
    "eggr": eggr,
}

# =========================
# Rotation utility
# =========================

def block_rotation_matrix(D: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build a block rotation matrix by pairing dimensions into 2x2 planar rotations
    using the SAME angle theta for every pair.

    Matches the spec: at each environment change, randomly group dimensions into
    2D planes (ignore the leftover dim if D is odd), then rotate those planes by theta.
    """
    idx = np.arange(D)
    rng.shuffle(idx)
    R = np.eye(D)
    m = (D // 2) * 2  # largest even count
    c = math.cos(theta)
    s = math.sin(theta)
    for k in range(0, m, 2):
        i = int(idx[k])
        j = int(idx[k + 1])
        R[np.ix_([i, j], [i, j])] = np.array([[c, -s], [s, c]])
    return R

# =========================
# DFGenerator for F1-F4
# =========================

class DFGenerator:
    DPEAKS = 0.1  # min distance between any two peaks (spec dpeaks = 0.1)

    def __init__(
        self,
        D: int,
        g: int,
        l: int,
        rng: np.random.Generator,
        fixed_centers: Optional[List[float]] = None,
        fixed_width: Optional[float] = None,
        fixed_height: Optional[float] = None,
    ):
        """
        DF landscape (simple multimodal functions in the spec).

        D : dimension
        g : number of global peaks (gn_max)
        l : number of local peaks
        fixed_centers : if provided, we take those scalars and replicate across D
                        to define global optima coordinates
        fixed_width   : global peak width if fixed
        fixed_height  : global peak height if fixed (usually 75.0)
        """
        self.D = D
        self.g_max = g
        self.l = l
        self.rng = rng

        # self.peaks[i] = {
        #   'center': np.ndarray(D),
        #   'height': float,
        #   'width' : float,
        #   'is_global': bool
        # }

        self.peaks: List[Dict] = []

        # Globals first (important for C7 / C8 logic later)
        if fixed_centers is not None:
            centers_list = [np.full(D, v, dtype=float) for v in fixed_centers]
        else:
            centers_list = []

        num_global = len(centers_list) if fixed_centers is not None else self.g_max

        for i in range(num_global):
            if fixed_centers is not None:
                c = centers_list[i]
            else:
                c = self._rand_center()
            c = self._ensure_min_dist(c)

            self.peaks.append(
                {
                    "center": c,
                    "height": float(
                        fixed_height if fixed_height is not None else 75.0
                    ),
                    "width": float(
                        fixed_width
                        if fixed_width is not None
                        else self.rng.uniform(1.0, 12.0)
                    ),
                    "is_global": True,
                }
            )

        # Locals (never considered globals unless algorithm mislabels them, which we will prevent)
        for _ in range(self.l):
            c = self._rand_center()
            c = self._ensure_min_dist(c)
            self.peaks.append(
                {
                    "center": c,
                    "height": float(self.rng.uniform(30.0, 70.0)),
                    "width": float(self.rng.uniform(1.0, 12.0)),
                    "is_global": False,
                }
            )

        self._enforce_dpeaks()

    def _rand_center(self) -> np.ndarray:
        return self.rng.uniform(-5, 5, size=self.D)

    def _ensure_min_dist(self, center: np.ndarray) -> np.ndarray:
        # Retry random centers until we are >= DPEAKS from all existing peaks
        for _ in range(100):
            ok = True
            for p in self.peaks:
                if np.linalg.norm(center - p["center"]) < self.DPEAKS:
                    ok = False
                    break
            if ok:
                return center
            center = self._rand_center()
        return center

    def _enforce_dpeaks(self):
        # After construction or movement, push peaks apart if too close
        for i in range(len(self.peaks)):
            p = self.peaks[i]
            for _ in range(10):
                violated = False
                for j in range(len(self.peaks)):
                    if i == j:
                        continue
                    q = self.peaks[j]
                    if np.linalg.norm(p["center"] - q["center"]) < self.DPEAKS:
                        r = self.rng.normal(size=self.D)
                        r /= (np.linalg.norm(r) + 1e-12)
                        p["center"] = np.clip(
                            p["center"] + r * self.DPEAKS, -5, 5
                        )
                        violated = True
                        break
                if not violated:
                    break

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Fitness = max_i [ height_i - width_i * dist_i ],
        where dist_i is Euclidean distance to peak i.
        We maximize this.
        """
        X = np.atleast_2d(X)
        n = X.shape[0]
        vals = np.full(n, -np.inf, dtype=float)
        for p in self.peaks:
            diff = X - p["center"]
            d = np.sqrt(np.sum(diff * diff, axis=1))
            cand = p["height"] - p["width"] * d
            vals = np.maximum(vals, cand)
        return vals

    def get_global_centers(self) -> List[np.ndarray]:
        return [p["center"] for p in self.peaks if p["is_global"]]

    def apply_rotation(self, theta: float):
        """
        Rotate all peak centers by a random block rotation with angle theta,
        clip to [-5,5], then re-enforce min distance (dpeaks).
        """
        R = block_rotation_matrix(self.D, theta, self.rng)
        for p in self.peaks:
            p["center"] = np.clip(R @ p["center"], -5, 5)
        self._enforce_dpeaks()

# =========================
# CompositionFunction for F5-F8
# =========================

class CompositionFunction:
    def __init__(
        self,
        D: int,
        centers: List[np.ndarray],
        lambdas: List[float],
        sigmas: List[float],
        base_names: List[str],
        rng: np.random.Generator,
    ):
        self.D = D
        self.centers = [np.array(o, dtype=float) for o in centers]
        self.lambdas = list(lambdas)
        self.sigmas = list(sigmas)
        self.base_names = list(base_names)
        self.n = len(centers)
        self.rng = rng

        # per-subcomponent rotation bases
        self.thetas = [rng.uniform(-math.pi, math.pi) for _ in range(self.n)]
        self.Ms = [block_rotation_matrix(D, th, rng) for th in self.thetas]

        # normalizer so that each subfunction is roughly same scale
        self.norms = []
        for nm in self.base_names:
            fi = BASE_FUNS[nm]
            v = fi(np.zeros(self.D))
            if abs(v) < 1e-12:
                v = 1.0
            self.norms.append(v)

        # track which subfunctions are considered "global" this environment
        # initially all are global (matches "maximum number of global optima")
        self.is_global = [True] * self.n

    def get_centers(self) -> List[np.ndarray]:
        return [c.copy() for c in self.centers]

    def get_global_centers(self) -> List[np.ndarray]:
        return [
            c.copy() for (c, flag) in zip(self.centers, self.is_global) if flag
        ]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Composition function:
        w_i = exp( - ||x - o_i||^2 / (2 * sigma_i^2) ), normalized
        f(x) = - sum_i w_i * ( base_i( ((x-o_i)/lambda_i) * M_i ) / norm_i )
        We return negative so that best value is 0 and we can still 'maximize'.
        """
        X = np.atleast_2d(X)
        npts = X.shape[0]

        # compute weights
        w = np.zeros((self.n, npts), dtype=float)
        for i in range(self.n):
            diff = X - self.centers[i]
            sq = np.sum(diff * diff, axis=1)
            denom = 2.0 * (self.sigmas[i] ** 2)
            if denom <= 1e-12:
                denom = 1.0
            w[i, :] = np.exp(-sq / denom)

        w_sum = np.sum(w, axis=0)
        w_sum[w_sum == 0] = 1.0
        w /= w_sum  # normalize rows

        total = np.zeros(npts, dtype=float)
        for i in range(self.n):
            o = self.centers[i]
            lam = self.lambdas[i]
            M = self.Ms[i]
            Z = ((X - o) / lam) @ M.T
            fi = BASE_FUNS[self.base_names[i]]
            vals = np.apply_along_axis(fi, 1, Z)
            vals /= self.norms[i]
            total += w[i, :] * vals

        return -total  # higher is better, best ~0

    def apply_dynamics(self, theta_delta: float, R_t: np.ndarray):
        """
        Move each sub-center with shared rotation R_t,
        then gently update each internal basis rotation.
        """
        # move centers
        for i in range(self.n):
            self.centers[i] = np.clip(R_t @ self.centers[i], -5, 5)

        # update internal rotations
        for i in range(self.n):
            self.thetas[i] += theta_delta
            R_delta = block_rotation_matrix(self.D, self.thetas[i], self.rng)
            self.Ms[i] = R_delta @ self.Ms[i]

    def set_num_globals_linear(self, dir_state: int, min_g: int = 2) -> int:
        """
        Change Mode C7 for composition functions:
        Linearly increase or decrease how many subfunctions are marked global.
        dir_state 2 means go up, 1 means go down.
        """
        current_g = sum(self.is_global)
        gmax = self.n

        if dir_state == 2:
            next_g = current_g + 1
            if next_g > gmax:
                next_g = gmax
        else:
            next_g = current_g - 1
            if next_g < min_g:
                next_g = min_g

        # flip direction at edges
        if next_g >= gmax:
            dir_state = 1
        if next_g <= min_g:
            dir_state = 2

        # mark first next_g as global
        for idx in range(gmax):
            self.is_global[idx] = (idx < next_g)

        return dir_state

    def set_num_globals_random(self, rng: np.random.Generator, min_g: int = 2):
        """
        Change Mode C8 for composition functions:
        Random number of global optima between 2 and n.
        """
        gmax = self.n
        newg = int(rng.integers(min_g, gmax + 1))
        for idx in range(gmax):
            self.is_global[idx] = (idx < newg)

# =========================
# DMMOProblem = (Fi, Cj)
# =========================

class DMMOProblem:
    # severity / control constants from spec
    # α = 0.04, α_max = 0.01, A = 3.67, P = 12, ns = 0.8
    ALPHA = 0.04
    ALPHA_MAX = 0.01
    A = 3.67
    P = 12
    NS = 0.8

    # scale factors for each numeric parameter
    ES = {
        "height": 7.0,   # height severity
        "width": 1.0,    # width severity
        "theta": 1.0,    # rotation angle severity
    }

    # allowed ranges for each numeric parameter
    EMIN = {
        "height": 30.0,
        "width": 1.0,
        "theta": -math.pi,
    }
    EMAX = {
        "height": 70.0,
        "width": 12.0,
        "theta": math.pi,
    }

    # CF definitions for F5..F8
    LAMBDA_SIGMA = {
        "F5": {
            "lambda": [1, 1, 8, 8, 1 / 5, 1 / 5],
            "sigma":  [1, 1, 1, 1, 1, 1],
            "bases":  [
                "rastrigin",
                "rastrigin",
                "weierstrass",
                "weierstrass",
                "sphere",
                "sphere",
            ],
        },
        "F6": {
            "lambda": [1, 1, 10, 10, 1 / 10, 1 / 10, 1 / 7, 1 / 7],
            "sigma":  [1, 1, 1, 1, 1, 1, 1, 1],
            "bases":  [
                "rastrigin",
                "rastrigin",
                "weierstrass",
                "weierstrass",
                "griewank",
                "griewank",
                "sphere",
                "sphere",
            ],
        },
        "F7": {
            "lambda": [1 / 4, 1 / 10, 2, 1, 2, 5],
            "sigma":  [1, 1, 2, 2, 2, 2],
            "bases":  [
                "ackley",
                "ackley",
                "weierstrass",
                "weierstrass",
                "griewank",
                "griewank",
            ],
        },
        "F8": {
            "lambda": [4, 1, 4, 1, 1 / 10, 1 / 5, 1 / 10, 1 / 40],
            "sigma":  [1, 1, 1, 1, 1, 2, 2, 2],
            "bases":  [
                "rastrigin",
                "rastrigin",
                "eggr",
                "eggr",
                "weierstrass",
                "weierstrass",
                "griewank",
                "griewank",
            ],
        },
    }

    def __init__(self, func_id: str, change_mode: str = "C1", dim: int = 5, seed: int = 1):
        """
        func_id: 'F1'..'F8'
        change_mode: 'C1'..'C8'
        dim: typically 5 or 10
        seed: run seed (1..30 in official runs)
        """
        self.func_id = func_id
        self.change_mode = change_mode
        self.D = dim
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.t = 0  # environment counter (0..59)
        self.phi = self.rng.uniform(0, 2 * math.pi)  # phase for C5/C6 recurrence
        self.C7_dir = 2  # 2 means increase, 1 means decrease (like is_add in MATLAB spec)

        # Build static base landscape for t=0
        if func_id in ["F1", "F2", "F3", "F4"]:
            if func_id == "F1":
                g, l, centers, w, h = 4, 4, None, None, None
            elif func_id == "F2":
                g, l, centers, w, h = 4, 0, [-3, -2, 2, 3], 12, 75
            elif func_id == "F3":
                g, l, centers, w, h = 4, 0, [-2.5, -1.5, 0.5, 4.5], 5, 75
            else:  # F4
                g, l, centers, w, h = 4, 0, [-3, -1, 1, 3], 5, 75

            self.df = DFGenerator(
                D=dim,
                g=g,
                l=l,
                rng=self.rng,
                fixed_centers=centers,
                fixed_width=w,
                fixed_height=h,
            )
            self.evaluate = self.df.evaluate
            self._get_global_centers = self.df.get_global_centers
            self.df_gmax = g  # original number of global peaks (for C7/C8 masking)

        else:
            info = self.LAMBDA_SIGMA[func_id]
            n_sub = len(info["lambda"])
            centers = [self.rng.uniform(-5, 5, size=dim) for _ in range(n_sub)]
            self.comp = CompositionFunction(
                D=dim,
                centers=centers,
                lambdas=info["lambda"],
                sigmas=info["sigma"],
                base_names=info["bases"],
                rng=self.rng,
            )
            self.evaluate = self.comp.evaluate
            self._get_global_centers = self.comp.get_global_centers

        # precompute numeric ranges Er = Emax - Emin
        self.ER = {k: self.EMAX[k] - self.EMIN[k] for k in self.EMIN}

        # track the last theta we applied so we can compute rotation deltas
        self.theta_prev = 0.0

    def _rand_r(self) -> float:
        return float(self.rng.uniform(-1.0, 1.0))

    def _theta_bounds_for_mode(self) -> Tuple[float, float]:
        # theta range [0, pi/6] for C5,C6, otherwise [-pi,pi]
        if self.change_mode in ("C5", "C6"):
            return (0.0, math.pi / 6.0)
        return (-math.pi, math.pi)

    def _E_update_numeric(self, E_val: float, key: str) -> float:
        """
        Update a scalar parameter (theta, width, height) according to the selected
        change mode (C1..C8). Mirrors formulas (4)-(11) in the CEC 2022 spec.
        """
        cm = self.change_mode
        r = self._rand_r()

        if cm == "C1":
            # small step
            delta_E = self.ALPHA * self.ER[key] * r * self.ES[key]
            return float(
                np.clip(E_val + delta_E, self.EMIN[key], self.EMAX[key])
            )

        if cm == "C2":
            # large step
            sgn = 0.0
            if r > 0:
                sgn = 1.0
            elif r < 0:
                sgn = -1.0
            delta_E = self.ER[key] * (
                self.ALPHA * sgn + (self.ALPHA_MAX - self.ALPHA) * r
            ) * self.ES[key]
            return float(
                np.clip(E_val + delta_E, self.EMIN[key], self.EMAX[key])
            )

        if cm == "C3":
            # random Gaussian
            delta_E = self.ES[key] * self.rng.normal()
            return float(
                np.clip(E_val + delta_E, self.EMIN[key], self.EMAX[key])
            )

        if cm == "C4":
            # chaotic
            Emin = self.EMIN[key]
            Esv = self.ES[key]
            E_new = Emin + self.A * (E_val - Emin) * (1.0 - (E_val - Emin) / Esv)
            return float(
                np.clip(E_new, self.EMIN[key], self.EMAX[key])
            )

        if cm in ("C5", "C6"):
            # recurrent (periodic), plus noise for C6
            base = (
                self.EMIN[key]
                + self.ER[key]
                * (math.sin(2.0 * math.pi * self.t / self.P + self.phi) + 1.0)
                / 2.0
            )
            noise = self.NS * self.rng.normal() if cm == "C6" else 0.0
            E_new = base + noise
            return float(
                np.clip(E_new, self.EMIN[key], self.EMAX[key])
            )

        if cm in ("C7", "C8"):
            # same numeric update style as C1 for continuous params
            delta_E = self.ALPHA * self.ER[key] * r * self.ES[key]
            return float(
                np.clip(E_val + delta_E, self.EMIN[key], self.EMAX[key])
            )

        # default no change
        return float(E_val)

    def _set_globals_linear_df(self):
        """
        Change Mode C7 for DFGenerator:
        linearly increase/decrease number of global peaks between 2 and df_gmax.
        Only consider the first df_gmax peaks as 'candidates' for globals.
        """
        # gather mask in first df_gmax
        mask = [self.df.peaks[i]["is_global"] for i in range(self.df_gmax)]
        current_g = sum(mask)

        # flip direction at edges
        if current_g >= self.df_gmax:
            self.C7_dir = 1
        if current_g <= 2:
            self.C7_dir = 2

        if self.C7_dir == 2:
            next_g = min(current_g + 1, self.df_gmax)
        else:
            next_g = max(current_g - 1, 2)

        # apply new mask: first next_g are global, rest (within first df_gmax) off
        for i in range(self.df_gmax):
            self.df.peaks[i]["is_global"] = (i < next_g)

        # locals after df_gmax always non-global
        for i in range(self.df_gmax, len(self.df.peaks)):
            self.df.peaks[i]["is_global"] = False

    def _set_globals_random_df(self):
        """
        Change Mode C8 for DFGenerator:
        random number of global peaks between 2 and df_gmax.
        """
        newg = int(self.rng.integers(2, self.df_gmax + 1))

        for i in range(self.df_gmax):
            self.df.peaks[i]["is_global"] = (i < newg)

        for i in range(self.df_gmax, len(self.df.peaks)):
            self.df.peaks[i]["is_global"] = False

    def update_environment(self):
        """
        Advance to the next environment (t -> t+1):

        1. Update numeric params (theta, width, height) using chosen change mode.
        2. Rotate / move the optima in the landscape with that theta.
        3. For C7 / C8 also update how many peaks are 'global'.
        """
        cm = self.change_mode

        # update theta using mode rules
        theta_min, theta_max = self._theta_bounds_for_mode()
        theta_new = self._E_update_numeric(self.theta_prev, "theta")
        theta_new = float(np.clip(theta_new, theta_min, theta_max))
        theta_delta = theta_new - self.theta_prev
        self.theta_prev = theta_new

        # rotation matrix for center movement
        R_t = block_rotation_matrix(self.D, theta_new, self.rng)

        if hasattr(self, "df"):
            # DF branch F1..F4:
            for p_idx, p in enumerate(self.df.peaks):
                # local peak heights can change, global peak heights (75) generally fixed by design
                if not p["is_global"]:
                    p["height"] = float(
                        self._E_update_numeric(p["height"], "height")
                    )

                # all widths can change
                p["width"] = float(
                    self._E_update_numeric(p["width"], "width")
                )

            # move centers by rotation
            self.df.apply_rotation(theta_new)

            # Handle global peak count in C7 / C8
            if cm == "C7":
                self._set_globals_linear_df()
            elif cm == "C8":
                self._set_globals_random_df()

        else:
            # Composition branch F5..F8:
            self.comp.apply_dynamics(theta_delta, R_t)

            if cm == "C7":
                self.C7_dir = self.comp.set_num_globals_linear(
                    self.C7_dir, min_g=2
                )
            elif cm == "C8":
                self.comp.set_num_globals_random(self.rng, min_g=2)

        # done with this environment, advance t
        self.t += 1

    def get_global_centers(self) -> List[np.ndarray]:
        return self._get_global_centers()

    def is_peak_found(
        self,
        individual: np.ndarray,
        eps_d: float = 0.05,
        eps_f: float = 1e-4,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if individual is within eps_d in position AND eps_f in fitness
        from ANY current GLOBAL optimum.

        This matches Algorithm 1 for counting NPF (number of peaks found) in the
        official metric.
        """
        x = individual.reshape(1, -1)
        f_x = float(self.evaluate(x)[0])

        peaks = self.get_global_centers()
        if len(peaks) == 0:
            return False, None

        dists = [np.linalg.norm(x - p) for p in peaks]
        idx = int(np.argmin(dists))
        bestd = dists[idx]

        o = peaks[idx].reshape(1, -1)
        f_o = float(self.evaluate(o)[0])

        if (bestd < eps_d) and (abs(f_x - f_o) < eps_f):
            return True, idx
        return False, None

    def count_found_peaks(
        self,
        population: np.ndarray,
        eps_d: float = 0.05,
        eps_f: float = 1e-4,
    ) -> int:
        """
        Count how many DISTINCT global peaks were located in this population.
        """
        found = set()
        for i in range(population.shape[0]):
            ok, idx = self.is_peak_found(population[i], eps_d=eps_d, eps_f=eps_f)
            if ok:
                found.add(idx)
        return len(found)

# =========================
# Simple DE/rand/1/bin (maximize)
# =========================

def de_epoch(
    pop: np.ndarray,
    fitness: np.ndarray,
    problem: DMMOProblem,
    F: float = 0.5,
    CR: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One DE/rand/1/bin generation for maximization
    """
    if rng is None:
        rng = np.random.default_rng()
    NP, D = pop.shape
    new_pop = pop.copy()
    new_fit = fitness.copy()

    for i in range(NP):
        # choose a, b, c != i
        idxs = [idx for idx in range(NP) if idx != i]
        if len(idxs) < 3:
            continue
        a, b, c = rng.choice(idxs, size=3, replace=False)

        mutant = pop[a] + F * (pop[b] - pop[c])

        trial = pop[i].copy()
        jrand = rng.integers(0, D)
        for j in range(D):
            if rng.random() < CR or j == jrand:
                trial[j] = mutant[j]

        # bound handling: clip to [-5,5]
        trial = np.clip(trial, -5, 5)

        fv = float(problem.evaluate(trial.reshape(1, -1))[0])
        if fv > fitness[i]:
            new_pop[i] = trial
            new_fit[i] = fv

    return new_pop, new_fit

# =========================
# Experiment harness
# =========================

def run_experiment(
    problem_id: str,
    change_mode: str,
    dim: int,
    runs: int = 30,
    envs: int = 60,
    fe_per_env_factor: int = 5000,
    pop_size: int = 100,
    F: float = 0.6,
    CR: float = 0.9,
    seed_start: int = 1,
    eps_f_levels: List[float] = [1e-3, 1e-4, 1e-5],
    quick: bool = False,
) -> Dict:
    """
    Execute 'runs' independent runs of DE on the specified (Fi, Cj),
    track Peak Ratio (PR) across 60 environments,
    consistent with the official metric.

    NOTE:
    The DE here is just a placeholder searcher.
    We'll swap this with AMLP-RS-CMSA-ESII later.
    """
    total_NPF = {eps: 0 for eps in eps_f_levels}  # sum over all runs/envs
    total_Peaks = 0  # sum over all runs/envs
    total_fe = 0

    # each environment budget = 5000 * D per spec
    fe_per_env = (100 * dim) if quick else (fe_per_env_factor * dim)

    print(
        f"Running P({problem_id}, {change_mode}) D={dim}. "
        f"FE/Env={fe_per_env}. Runs={runs}"
    )

    for run in range(runs):
        seed = seed_start + run
        rng = np.random.default_rng(seed)
        prob = DMMOProblem(
            problem_id, change_mode=change_mode, dim=dim, seed=seed
        )

        # initialize pop in [-5,5]
        pop = rng.uniform(-5, 5, size=(pop_size, dim))
        fit = prob.evaluate(pop)
        fe_count = pop_size

        # loop over environments
        for e in range(envs):
            # spend FE budget in THIS environment before changing
            fe_in_env = 0
            while fe_in_env < fe_per_env:
                pop, fit = de_epoch(pop, fit, prob, F=F, CR=CR, rng=rng)
                fe_in_env += pop_size
                fe_count += pop_size

            # after finishing evaluations for this environment,
            # count how many peaks exist and how many we caught
            peaks_now = len(prob.get_global_centers())
            total_Peaks += peaks_now

            for eps in eps_f_levels:
                found = prob.count_found_peaks(
                    pop, eps_d=0.05, eps_f=eps
                )
                total_NPF[eps] += found

            if (e + 1) % 10 == 0 or e == envs - 1:
                print(
                    f"  Run {run+1}/{runs} | "
                    f"Env {e+1}/{envs} | "
                    f"Peaks: {peaks_now} | "
                    f"Max Fit: {fit.max():.4f}"
                )

            # advance environment to next state for next loop (unless this was final env,
            # but calling once more doesn't affect scoring now because loop will end)
            prob.update_environment()

        total_fe += fe_count

    # Peak Ratio for each eps_f level
    PRs = {
        eps: (total_NPF[eps] / total_Peaks) if total_Peaks > 0 else 0.0
        for eps in eps_f_levels
    }

    return {
        "PRs": PRs,
        "total_NPF": total_NPF,
        "total_Peaks": total_Peaks,
        "Total_FE": total_fe,
    }


if __name__ == "__main__":
    # quick sanity run for F1/C1 (P1-like but tiny budget)
    res_p1 = run_experiment(
        "F1",
        "C1",
        dim=5,
        runs=2,
        envs=5,
        pop_size=30,
        quick=True,
        seed_start=1,
    )
    print("Smoke F1/C1:", res_p1["PRs"])

    # quick sanity run for F8/C8 (P16-like behavior) just to see globals fluctuate
    res_p16 = run_experiment(
        "F8",
        "C8",
        dim=5,
        runs=2,
        envs=5,
        pop_size=30,
        quick=True,
        seed_start=1,
    )
    print("Smoke F8/C8:", res_p16["PRs"])
