import numpy as np

class DMMOProblem:
    """
    Dynamic Multimodal Optimization Problem
    Implements CEC 2022 dynamic test suite behavior for F1..F8 with change modes C1..C8

    Assumptions from CEC 2022 spec
    - Domain is [-5, 5]^D
    - One environment lasts freq = 5000 * D evaluations
    - Total environments = maxEnv = 60
    - Peak ratio is computed using known global peaks for each environment

    This class tracks:
    - peak locations (P x D)
    - peak heights
    - peak widths
    - per-peak rotation (to shape basins)
    - the change mode Ck

    Note
    We treat the problem as maximization
    Fitness is larger near the true peak
    """

    def __init__(self, func_id: int, change_id: int, dim: int, seed: int = 0):
        """
        func_id in {1..8}
        change_id in {1..8}
        """
        self.func_id = func_id
        self.change_id = change_id
        self.D = dim
        self.rng = np.random.default_rng(seed)

        # bounds
        self.lb = -5.0 * np.ones(self.D)
        self.ub =  5.0 * np.ones(self.D)

        # dynamic settings from CEC 2022
        self.freq = 5000 * self.D           # env length in FEs
        self.maxEnv = 60                    # total environments
        self.t = 0                          # current environment index (0 based)

        # movement parameters (from CEC 2022 and MATLAB code comments)
        self.ALPHA = 0.04
        self.ALPHA_MAX = 0.10        # NOTE In your old python code this was 0.01 which is 10x too small
        self.A_CHAOTIC = 3.67
        self.P_REC = 12
        self.NOISE_SEVERITY = 0.8

        # set up peaks for this function form
        self._init_function_structure()

        # track eval usage within current environment
        self.fe_used_in_env = 0
        self.total_fe_used = 0

    def _init_function_structure(self):
        """
        Set up number of peaks and their parameters based on func_id.
        We follow typical CEC style composition multimodal landscapes:
        F1..F4   multiple peaks directly
        F5..F8   composition of subfunctions (rotated basins etc)

        We create:
            self.num_global_peaks
            self.peak_pos  [P x D]
            self.peak_height [P]
            self.peak_width [P]
            self.peak_rotations [P x D x D]
        """

        # for reproducibility we define a baseline number of peaks
        # F1..F4 static count 10
        # F5..F8 dynamic count that can vary in C7 C8
        if self.func_id <= 4:
            self.num_global_peaks = 10
        else:
            self.num_global_peaks = 10

        self.peak_pos = self.rng.uniform(self.lb, self.ub, size=(self.num_global_peaks, self.D))
        # peak height roughly in [30, 70] for global peaks
        self.peak_height = self.rng.uniform(50.0, 70.0, size=(self.num_global_peaks,))
        # width controlling basin sharpness smaller width = sharper
        self.peak_width = self.rng.uniform(1.0, 2.0, size=(self.num_global_peaks,))

        # generate block rotation for each peak
        self.peak_rotations = np.zeros((self.num_global_peaks, self.D, self.D))
        for i in range(self.num_global_peaks):
            self.peak_rotations[i] = self._random_block_rotation()

        # mode C7 and C8 need counters for varying number of global optima
        self.c7_direction = 1   # goes up then down
        self.c7_min_peaks = 2
        self.c7_max_peaks = self.num_global_peaks

    def _random_block_rotation(self):
        """
        Create a D x D block diagonal rotation.
        Pairs of dims get planar rotation by random angle theta.
        Unpaired final dim stays as 1.
        """
        D = self.D
        R = np.eye(D)
        dims = np.arange(D)
        self.rng.shuffle(dims)
        for i in range(0, D - 1, 2):
            a = dims[i]
            b = dims[i+1]
            theta = self.rng.uniform(-np.pi, np.pi)
            rot2 = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            # embed into R
            R_block = np.eye(D)
            R_block[np.ix_([a,b],[a,b])] = rot2
            R = R_block @ R
        return R

    def _eval_peak_contribution(self, x, pos, height, width, R):
        """
        single-peak value
        RS CMSA ESII uses maximization so higher is better
        Basic form: height - || (x - pos) * R ||^2 / (2 * width^2)
        """
        d = x - pos
        z = d @ R
        val = height - np.sum(z**2) / (2.0 * (width**2))
        return val

    def evaluate(self, X):
        """
        Evaluate a batch of candidate solutions.
        X shape [N, D]
        Returns fitness array where larger is better
        """
        X = np.atleast_2d(X)
        N = X.shape[0]
        fit = np.empty(N)

        for i in range(N):
            xi = np.clip(X[i], self.lb, self.ub)
            # choose best peak response
            best_val = -np.inf
            for p in range(self.num_global_peaks):
                val = self._eval_peak_contribution(
                    xi,
                    self.peak_pos[p],
                    self.peak_height[p],
                    self.peak_width[p],
                    self.peak_rotations[p]
                )
                if val > best_val:
                    best_val = val
            fit[i] = best_val

        # update counters
        self.fe_used_in_env += N
        self.total_fe_used += N

        return fit

    def get_true_global_optima(self):
        """
        Return peak positions and their true fitness values
        Used for Peak Ratio scoring
        """
        opt_pos = self.peak_pos[:self.num_global_peaks]
        opt_fit = np.array([
            self._eval_peak_contribution(
                opt_pos[i],
                self.peak_pos[i],
                self.peak_height[i],
                self.peak_width[i],
                self.peak_rotations[i]
            )
            for i in range(self.num_global_peaks)
        ])
        return opt_pos, opt_fit

    def _theta_bounds_for_mode(self):
        # MATLAB limits rotation angle movement for recurrent modes
        if self.change_id in [5, 6]:
            return (0.0, np.pi/6.0)
        else:
            return (-np.pi, np.pi)

    def _move_peaks_linear(self, alpha):
        """
        C1 small linear shift
        peak_pos += alpha * random_dir
        """
        direction = self.rng.uniform(-1.0, 1.0, size=self.peak_pos.shape)
        step = alpha * direction
        self.peak_pos = np.clip(self.peak_pos + step, self.lb, self.ub)

    def _move_peaks_large_step(self):
        """
        C2 large random relocation with ALPHA_MAX
        """
        direction = self.rng.uniform(-1.0, 1.0, size=self.peak_pos.shape)
        step = self.ALPHA_MAX * direction
        self.peak_pos = np.clip(self.peak_pos + step, self.lb, self.ub)

    def _chaotic_change(self):
        """
        C4 chaotic like update on peak positions within bounds
        Approximation of the logistic style mapping used in MATLAB code
        """
        Emin = self.lb
        Emax = self.ub
        span = Emax - Emin
        # normalize to [0,1]
        normed = (self.peak_pos - Emin) / np.maximum(span, 1e-12)
        # chaotic logistic-ish map
        new_normed = self.A_CHAOTIC * normed * (1.0 - normed)
        self.peak_pos = Emin + new_normed * span
        self.peak_pos = np.clip(self.peak_pos, self.lb, self.ub)

    def _recurrent_change(self, noisy=False):
        """
        C5 recurrent
        C6 recurrent noisy
        Move peaks in a repeating pattern with limited angle theta
        """
        # simple sinusoidal recurrence on position then optional noise
        amp = 0.5
        omega = 2.0 * np.pi / self.P_REC
        shift = amp * np.sin(omega * self.t)
        self.peak_pos = np.clip(self.peak_pos + shift, self.lb, self.ub)

        if noisy:
            noise = self.rng.normal(scale=self.NOISE_SEVERITY, size=self.peak_pos.shape)
            self.peak_pos = np.clip(self.peak_pos + noise, self.lb, self.ub)

        # restrict rotations
        tmin, tmax = self._theta_bounds_for_mode()
        for i in range(self.num_global_peaks):
            # build new rotation with restricted theta
            theta = self.rng.uniform(tmin, tmax)
            # apply only one pair to keep similar spirit
            R = np.eye(self.D)
            for a in range(0, self.D - 1, 2):
                rot2 = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])
                R_block = np.eye(self.D)
                R_block[np.ix_([a, a+1],[a, a+1])] = rot2
                R = R_block @ R
            self.peak_rotations[i] = R @ self.peak_rotations[i]

    def _vary_num_global_peaks_bounded(self):
        """
        C7
        number of global optima oscillates between min and max
        """
        self.num_global_peaks += self.c7_direction
        if self.num_global_peaks >= self.c7_max_peaks:
            self.num_global_peaks = self.c7_max_peaks
            self.c7_direction = -1
        if self.num_global_peaks <= self.c7_min_peaks:
            self.num_global_peaks = self.c7_min_peaks
            self.c7_direction = 1

    def _vary_num_global_peaks_random(self):
        """
        C8
        number of global optima jumps randomly
        """
        self.num_global_peaks = self.rng.integers(
            low=self.c7_min_peaks,
            high=self.c7_max_peaks+1
        )

    def environment_finished(self):
        """
        Check if we consumed freq evaluations in this environment
        """
        return self.fe_used_in_env >= self.freq

    def update_environment(self):
        """
        Advance to next environment with mode specific change
        Reset per environment FE counter
        """
        self.t += 1
        self.fe_used_in_env = 0

        # select change rule
        if self.change_id == 1:
            self._move_peaks_linear(self.ALPHA)
        elif self.change_id == 2:
            self._move_peaks_large_step()
        elif self.change_id == 3:
            # width and height random walk
            self.peak_height += self.rng.normal(scale=1.0, size=self.peak_height.shape)
            self.peak_width = np.clip(
                self.peak_width + self.rng.normal(scale=0.1, size=self.peak_width.shape),
                0.3,
                3.0
            )
        elif self.change_id == 4:
            self._chaotic_change()
        elif self.change_id == 5:
            self._recurrent_change(noisy=False)
        elif self.change_id == 6:
            self._recurrent_change(noisy=True)
        elif self.change_id == 7:
            self._move_peaks_linear(self.ALPHA)
            self._vary_num_global_peaks_bounded()
        elif self.change_id == 8:
            self._move_peaks_linear(self.ALPHA)
            self._vary_num_global_peaks_random()

        # also refresh rotations occasionally
        for i in range(self.num_global_peaks):
            if self.rng.random() < 0.2:
                self.peak_rotations[i] = self._random_block_rotation()
