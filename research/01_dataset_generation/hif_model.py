"""
HIF (High-Impedance Fault) Emanuel Arc Model Approximation.
Simulates nonlinear arc behaviour by oscillating fault resistance at power frequency.

Bug fix (v2): Original sampling at exact cycle boundaries (t = i * 0.02s) always
evaluates sin(π·50·t) = sin(π·i) = 0, producing constant R = R_min.
Fix: sample at random sub-cycle offsets within each step to capture arc oscillation.

IT22577924 — Karunanayake K.P.A.W.
"""
import math
import random
from config import HIF_R_MIN, HIF_R_MAX, GRID_FREQUENCY_HZ


def compute_arc_resistance(t: float,
                            f: float = GRID_FREQUENCY_HZ,
                            r_min: float = HIF_R_MIN,
                            r_max: float = HIF_R_MAX) -> float:
    """
    Emanuel arc model: resistance oscillates at power frequency.

    R(t) = R_min + (R_max - R_min) * |sin(π·f·t)|

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
    Pre-compute arc resistance for each simulation step using sub-cycle
    random offsets to avoid degenerate sampling at exact cycle boundaries.

    Each step samples a random point within the cycle, producing realistic
    per-cycle resistance variation between R_min and R_max.

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
    resistances = []
    for i in range(n_steps):
        # Add random sub-cycle offset to avoid sin(π·integer) = 0
        offset = random.uniform(0.0, step_duration)
        t = t_start + i * step_duration + offset
        resistances.append(
            compute_arc_resistance(t, r_min=r_min, r_max=r_max)
        )
    return resistances
