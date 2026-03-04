"""
Fault Injector — Add/remove OpenDSS Fault elements during Dynamic simulation.
IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import opendssdirect as dss

from config import PHASE_COUNT, PHASE_NODES

logger = logging.getLogger(__name__)

FAULT_ELEMENT_NAME = "FLT_RESEARCH"


def inject_fault(bus: str, fault_type: str, phase: str, resistance: float) -> bool:
    """
    Add a Fault element to the active OpenDSS circuit.

    Parameters
    ----------
    bus        : target bus name (no node suffix — suffix added from phase)
    fault_type : one of LG, LL, LLG, LLL, HIF
    phase      : phase combination string (e.g. 'A', 'AB', 'ABC')
    resistance : fault resistance in Ohms

    Returns
    -------
    bool : True if command issued without error
    """
    node_suffix = PHASE_NODES.get(phase, "")
    n_phases    = PHASE_COUNT.get(phase, 3)

    # For ground faults (LG, LLG): connect bus to ground (bus2 default = ground)
    # For phase faults (LL):        specify both nodes explicitly via bus1 node suffix
    bus_spec = f"{bus}{node_suffix}"

    cmd = (
        f"New Fault.{FAULT_ELEMENT_NAME} "
        f"bus1={bus_spec} "
        f"phases={n_phases} "
        f"r={resistance:.6f} "
        f"enabled=yes"
    )
    dss.Text.Command(cmd)

    err = dss.Error.Description()
    if err:
        logger.warning(f"Fault injection warning at {bus}: {err}")
        return False

    logger.debug(f"Fault injected: {fault_type}/{phase} @ {bus}, R={resistance:.3f}Ω")
    return True


def update_hif_resistance(resistance: float) -> None:
    """Update resistance of existing HIF fault element (called each dynamic step)."""
    dss.Text.Command(f"Edit Fault.{FAULT_ELEMENT_NAME} r={resistance:.6f}")


def remove_fault() -> None:
    """Disable the active fault element (simulates breaker clearing)."""
    dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=no")
    logger.debug("Fault removed (disabled)")


def delete_fault() -> None:
    """Permanently delete fault element from circuit (clean-up after sample)."""
    # OpenDSS does not expose a 'Delete' command for fault; disabling is sufficient.
    # The element will be overwritten by the next 'New Fault.FLT_RESEARCH' command.
    remove_fault()
