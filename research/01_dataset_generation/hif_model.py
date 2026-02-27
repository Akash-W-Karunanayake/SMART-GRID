"""
HIF (High-Impedance Fault) Emanuel Arc Model Approximation.
Simulates nonlinear arc behaviour by oscillating fault resistance at power frequency.
IT22577924 — Karunanayake K.P.A.W.
"""
import math
from config import HIF_R_MIN, HIF_R_MAX, GRID_FREQUENCY_HZ


def compute_arc_resistance(t: float,
                            f: float = GRID_FREQUENCY_HZ,
                            r_min: float = HIF_R_MIN,
                            r_max: float = HIF_R_MAX) -> float:
    """
    Emanuel arc model: resistance oscillates at power frequency.

    R(t) = R_min + (R_max - R_min) * |sin(π·f·t)|

    This produces:
    - Minimum resistance at current peak (zero crossing of sin term)
    - Maximum resistance at current zero-crossing
    - Asymmetric distortion → characteristic HIF signature

    Parameters
    ----------
    t      : simulation time in seconds
    f      : grid frequency (default 50 Hz)
    r_min  : minimum arc resistance (Ω), default 100 Ω
    r_max  : maximum arc resistance (Ω), default 500 Ω

    Returns
    -------
    float  : arc resistance at time t (Ω)
    """
    return r_min + (r_max - r_min) * abs(math.sin(math.pi * f * t))


def get_hif_resistance_sequence(n_steps: int,
                                 step_duration: float,
                                 t_start: float = 0.0,
                                 r_min: float = HIF_R_MIN,
                                 r_max: float = HIF_R_MAX) -> list:
    """
    Pre-compute arc resistance for each simulation step.

    Parameters
    ----------
    n_steps       : number of time steps (fault window)
    step_duration : duration of each step in seconds (e.g. 0.02 s for 1 cycle)
    t_start       : start time of fault in seconds
    r_min, r_max  : arc resistance bounds

    Returns
    -------
    list[float]   : resistance values for each step
    """
    return [
        compute_arc_resistance(t_start + i * step_duration, r_min=r_min, r_max=r_max)
        for i in range(n_steps)
    ]
