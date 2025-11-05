import numpy as np


def compute_env_peaks_summary(pro):
    """
    Wrap pro.get_peak() which should behave like MATLAB GetPeak():
      - returns (catchpeak, allpeak)
        catchpeak shape (3, maxEnv)
        allpeak   shape (1, maxEnv)
    We only care about current env just finished.
    """

    catchpeak, allpeak = pro.get_peak()
    # in MATLAB after the full run they do final reporting across all envs
    # here we try to get the latest env row
    # catchpeak: (3, maxEnv); allpeak: (maxEnv,) or (1,maxEnv)
    # we'll slice env index pro.env (after increment)
    env_idx = pro.env  # assume pro.env incremented after CheckChange

    peak_env = catchpeak[:, env_idx]  # shape (3,)
    allpeak_env = allpeak[env_idx] if allpeak.ndim > 1 else allpeak[env_idx]
    return peak_env, allpeak_env
