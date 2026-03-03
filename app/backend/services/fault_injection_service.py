"""
Fault Injection Service — manages fault lifecycle and sub-cycle snapshot capture.

When a fault is injected:
  1. Fault element is added to OpenDSS circuit
  2. 20 sub-cycle snapshots are captured (5 pre-fault + 10 fault + 5 post-fault)
  3. Phasor sequences are fed to FaultDetectionService for inference
  4. Fault stays active until manually cleared

Design decisions (from Phase 1):
  - One fault at a time (Q5)
  - Queue for next solve step (Q4)
  - Only when simulation running (Q6)
  - Persist until manually cleared (Q7)
  - Sub-cycle snapshots for model accuracy — Option B (Q17)
  - Retain last 20 fault events in memory (Q15)
"""
import opendssdirect as dss
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Constants (mirror research/01_dataset_generation/config.py) ──
PRE_FAULT_CYCLES = 5
FAULT_CYCLES = 10
POST_FAULT_CYCLES = 5
TOTAL_CYCLES = PRE_FAULT_CYCLES + FAULT_CYCLES + POST_FAULT_CYCLES  # 20
DYN_STEPSIZE_S = 0.02  # 1 cycle at 50 Hz

FAULT_ELEMENT_NAME = "FLT_LIVE"

PHASE_NODES = {
    "A": ".1", "B": ".2", "C": ".3",
    "AB": ".1.2", "BC": ".2.3", "CA": ".3.1",
    "ABG": ".1.2.4", "BCG": ".2.3.4", "CAG": ".3.1.4",
    "ABC": ".1.2.3",
}

PHASE_COUNT = {
    "A": 1, "B": 1, "C": 1,
    "AB": 2, "BC": 2, "CA": 2,
    "ABG": 2, "BCG": 2, "CAG": 2,
    "ABC": 3,
}

# Fault type → valid phases
FAULT_TYPE_PHASES = {
    "LG":  ["A", "B", "C"],
    "LL":  ["AB", "BC", "CA"],
    "LLG": ["ABG", "BCG", "CAG"],
    "LLL": ["ABC"],
    "HIF": ["A", "B", "C"],
}


@dataclass
class ActiveFault:
    """Tracks the currently active fault."""
    bus: str
    fault_type: str          # LG, LL, LLG, LLL, HIF
    phase: str               # A, AB, ABC, etc.
    resistance: float
    step_injected: int       # simulation step number when injected
    timestamp: float         # wall-clock time of injection
    cleared: bool = False


@dataclass
class SubCycleCapture:
    """Result of a sub-cycle snapshot capture."""
    voltage_seq: np.ndarray   # [20, N_buses, 6]
    current_seq: np.ndarray   # [20, N_branches, 6]
    bus_names: List[str]
    branch_names: List[str]


class FaultInjectionService:
    """Manages fault injection, sub-cycle capture, and fault clearing."""

    def __init__(self):
        self._active_fault: Optional[ActiveFault] = None
        self._fault_history: List[Dict[str, Any]] = []
        self._max_history = 20   # Q15: retain last 20 events
        self._queued_fault: Optional[Dict[str, Any]] = None
        self._last_capture: Optional['SubCycleCapture'] = None

    @property
    def has_active_fault(self) -> bool:
        return self._active_fault is not None and not self._active_fault.cleared

    @property
    def active_fault(self) -> Optional[ActiveFault]:
        return self._active_fault

    @property
    def last_capture(self) -> Optional['SubCycleCapture']:
        return self._last_capture

    @property
    def has_queued_fault(self) -> bool:
        return self._queued_fault is not None

    def queue_fault(self, bus: str, fault_type: str, phase: str,
                    resistance: float, current_step: int) -> Dict[str, Any]:
        """
        Queue a fault for injection at the next simulation step.

        Args:
            bus: Target bus name
            fault_type: LG, LL, LLG, LLL, HIF
            phase: A, B, C, AB, BC, CA, ABG, BCG, CAG, ABC
            resistance: Fault resistance (ohms)
            current_step: Current simulation step number

        Returns:
            Status dict
        """
        if self.has_active_fault:
            return {"success": False, "error": "A fault is already active. Clear it first."}

        if self._queued_fault is not None:
            return {"success": False, "error": "A fault is already queued."}

        if fault_type not in FAULT_TYPE_PHASES:
            return {"success": False, "error": f"Invalid fault_type: {fault_type}"}

        if phase not in PHASE_NODES:
            return {"success": False, "error": f"Invalid phase: {phase}"}

        if phase not in FAULT_TYPE_PHASES[fault_type]:
            return {"success": False, "error": f"Phase {phase} is not valid for {fault_type}. "
                    f"Valid: {FAULT_TYPE_PHASES[fault_type]}"}

        self._queued_fault = {
            "bus": bus,
            "fault_type": fault_type,
            "phase": phase,
            "resistance": resistance,
            "current_step": current_step,
        }

        logger.info(f"Fault queued: {fault_type} on {bus} phase={phase} R={resistance}")
        return {"success": True, "message": f"Fault queued for next step: {fault_type} at {bus}"}

    def apply_queued_fault(self, current_step: int) -> Optional[ActiveFault]:
        """
        Called by simulation loop at each step. If a fault is queued, inject it.

        Returns:
            ActiveFault if a fault was just applied, else None.
        """
        if self._queued_fault is None:
            return None

        q = self._queued_fault
        self._queued_fault = None

        # Inject into OpenDSS — remove any stale fault element first
        node_suffix = PHASE_NODES[q["phase"]]
        n_phases = PHASE_COUNT[q["phase"]]
        bus_spec = f"{q['bus']}{node_suffix}"

        try:
            dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=no")
        except Exception:
            pass  # element may not exist yet — that's fine
        try:
            dss.Text.Command(f"disable Fault.{FAULT_ELEMENT_NAME}")
        except Exception:
            pass

        cmd = (
            f"New Fault.{FAULT_ELEMENT_NAME} "
            f"bus1={bus_spec} "
            f"phases={n_phases} "
            f"r={q['resistance']:.6f} "
            f"enabled=yes"
        )
        try:
            dss.Text.Command(cmd)
        except Exception:
            # If "New" fails due to duplicate, edit the existing element
            logger.warning(f"Fault element already exists, editing instead of creating new")
            dss.Text.Command(f"Edit Fault.{FAULT_ELEMENT_NAME} "
                             f"bus1={bus_spec} "
                             f"phases={n_phases} "
                             f"r={q['resistance']:.6f} "
                             f"enabled=yes")

        self._active_fault = ActiveFault(
            bus=q["bus"],
            fault_type=q["fault_type"],
            phase=q["phase"],
            resistance=q["resistance"],
            step_injected=current_step,
            timestamp=time.time(),
        )

        logger.info(f"Fault applied at step {current_step}: "
                    f"{q['fault_type']} on {q['bus']} phase={q['phase']}")
        return self._active_fault

    def capture_subcycle_snapshots(self) -> SubCycleCapture:
        """
        Run 20 sub-cycle snapshot solves and capture voltage/current phasors.

        Sequence: 5 pre-fault → 10 fault → 5 post-fault.
        Pre-fault: fault disabled, solve 5 cycles.
        Fault: fault enabled, solve 10 cycles.
        Post-fault: fault disabled, solve 5 cycles.

        Returns:
            SubCycleCapture with voltage_seq [20, N_buses, 6] and
            current_seq [20, N_branches, 6].
        """
        bus_names = dss.Circuit.AllBusNames()
        n_buses = len(bus_names)

        # Collect branch names (lines + transformers as circuit elements)
        branch_names = []
        dss.Lines.First()
        while True:
            name = dss.Lines.Name()
            if not name:
                break
            branch_names.append(f"Line.{name}")
            if not dss.Lines.Next():
                break
        dss.Transformers.First()
        while True:
            name = dss.Transformers.Name()
            if not name:
                break
            branch_names.append(f"Transformer.{name}")
            if not dss.Transformers.Next():
                break
        n_branches = len(branch_names)

        voltage_seq = np.zeros((TOTAL_CYCLES, n_buses, 6), dtype=np.float32)
        current_seq = np.zeros((TOTAL_CYCLES, n_branches, 6), dtype=np.float32)

        # Save current mode, switch to snapshot for sub-cycle captures
        dss.Text.Command("Set mode=snapshot")

        # ── Pre-fault cycles (fault disabled) ──
        if self.has_active_fault:
            dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=no")

        for cycle in range(PRE_FAULT_CYCLES):
            dss.Solution.Solve()
            self._capture_one_cycle(cycle, bus_names, branch_names,
                                    voltage_seq, current_seq)

        # ── Fault cycles (fault enabled) ──
        if self.has_active_fault:
            dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=yes")

        for cycle in range(PRE_FAULT_CYCLES, PRE_FAULT_CYCLES + FAULT_CYCLES):
            dss.Solution.Solve()
            self._capture_one_cycle(cycle, bus_names, branch_names,
                                    voltage_seq, current_seq)

        # ── Post-fault cycles (fault disabled for observation) ──
        if self.has_active_fault:
            dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=no")

        for cycle in range(PRE_FAULT_CYCLES + FAULT_CYCLES, TOTAL_CYCLES):
            dss.Solution.Solve()
            self._capture_one_cycle(cycle, bus_names, branch_names,
                                    voltage_seq, current_seq)

        # Re-enable fault (it persists until cleared)
        if self.has_active_fault:
            dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=yes")

        return SubCycleCapture(
            voltage_seq=voltage_seq,
            current_seq=current_seq,
            bus_names=list(bus_names),
            branch_names=branch_names,
        )

    def _capture_one_cycle(self, cycle_idx: int,
                           bus_names: list, branch_names: list,
                           voltage_seq: np.ndarray,
                           current_seq: np.ndarray):
        """Capture voltage and current phasors for one cycle."""
        # ── Voltages ──
        for i, bus in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bus)
            phasor = dss.Bus.puVmagAngle()  # [mag_a, ang_a, mag_b, ang_b, mag_c, ang_c, ...]
            n_nodes = dss.Bus.NumNodes()
            if phasor and n_nodes >= 3:
                voltage_seq[cycle_idx, i, 0] = phasor[0]  # Va_mag
                voltage_seq[cycle_idx, i, 1] = phasor[1]  # Va_ang
                voltage_seq[cycle_idx, i, 2] = phasor[2]  # Vb_mag
                voltage_seq[cycle_idx, i, 3] = phasor[3]  # Vb_ang
                voltage_seq[cycle_idx, i, 4] = phasor[4]  # Vc_mag
                voltage_seq[cycle_idx, i, 5] = phasor[5]  # Vc_ang
            elif phasor and n_nodes < 3:
                # Pad single/two-phase buses
                for j in range(min(n_nodes, 3)):
                    voltage_seq[cycle_idx, i, j*2] = phasor[j*2]
                    voltage_seq[cycle_idx, i, j*2+1] = phasor[j*2+1]

        # ── Currents ──
        for j, branch in enumerate(branch_names):
            dss.Circuit.SetActiveElement(branch)
            mag_ang = dss.CktElement.CurrentsMagAng()
            if mag_ang and len(mag_ang) >= 6:
                current_seq[cycle_idx, j, 0] = mag_ang[0]  # Ia_mag
                current_seq[cycle_idx, j, 1] = mag_ang[1]  # Ia_ang
                current_seq[cycle_idx, j, 2] = mag_ang[2]  # Ib_mag
                current_seq[cycle_idx, j, 3] = mag_ang[3]  # Ib_ang
                current_seq[cycle_idx, j, 4] = mag_ang[4]  # Ic_mag
                current_seq[cycle_idx, j, 5] = mag_ang[5]  # Ic_ang

    def clear_fault(self) -> Dict[str, Any]:
        """Clear the active fault and record in history."""
        if not self.has_active_fault:
            return {"success": False, "error": "No active fault to clear."}

        # Disable and remove from OpenDSS
        dss.Text.Command(f"Fault.{FAULT_ELEMENT_NAME}.enabled=no")

        fault = self._active_fault
        fault.cleared = True

        # Record in history
        event = {
            "bus": fault.bus,
            "fault_type": fault.fault_type,
            "phase": fault.phase,
            "resistance": fault.resistance,
            "step_injected": fault.step_injected,
            "injected_at": fault.timestamp,
            "cleared_at": time.time(),
        }
        self._fault_history.append(event)
        if len(self._fault_history) > self._max_history:
            self._fault_history.pop(0)

        self._active_fault = None
        logger.info(f"Fault cleared: {fault.fault_type} on {fault.bus}")
        return {"success": True, "message": f"Fault cleared: {fault.fault_type} at {fault.bus}"}

    def get_fault_history(self) -> List[Dict[str, Any]]:
        """Return the last N fault events."""
        return list(self._fault_history)


# Singleton
fault_injection_service = FaultInjectionService()
