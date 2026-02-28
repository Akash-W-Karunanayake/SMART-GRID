"""
Simulation Engine — Core OpenDSS Dynamic simulation for fault dataset generation.
Implements the 20-cycle (pre/fault/post) data capture window.
IT22577924 — Karunanayake K.P.A.W.
"""
import logging
import numpy as np
import opendssdirect as dss

from config import (
    MASTER_DSS,
    PRE_FAULT_CYCLES, FAULT_CYCLES, POST_FAULT_CYCLES, TOTAL_CYCLES,
    DYN_STEPSIZE_S,
    SNAPSHOT_MAXITER, SNAPSHOT_TOLERANCE,
    GRID_FREQUENCY_HZ,
    LOAD_MULTIPLIERS, DER_PENETRATIONS,
)
from graph_builder import build_graph, save_graph, load_graph
from fault_injector import inject_fault, update_hif_resistance, remove_fault, delete_fault
from hif_model import get_hif_resistance_sequence

logger = logging.getLogger(__name__)


class SimEngine:
    """
    Wraps OpenDSS Dynamic mode simulation for fault data capture.

    Usage
    -----
    engine = SimEngine()
    engine.compile()
    engine.build_graph_once()

    sample = engine.run_sample(
        fault_type='LG', phase='A', bus='f06_mv_bus',
        resistance=1.0, load_mult=1.0, der_fraction=0.8
    )
    """

    def __init__(self):
        self._compiled = False
        self._graph = None
        self._bus_names: list = []
        self._bus_index: dict = {}
        self._branch_names: list = []    # ordered list of (element_name, bus1, bus2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Compile Master.dss and apply snapshot base settings."""
        dss.Basic.ClearAll()
        dss.Text.Command(f'Compile "{MASTER_DSS}"')
        err = dss.Error.Description()
        if err:
            raise RuntimeError(f"OpenDSS compile error: {err}")

        # Snapshot solve settings
        dss.Text.Command(f"Set maxiterations={SNAPSHOT_MAXITER}")
        dss.Text.Command(f"Set tolerance={SNAPSHOT_TOLERANCE}")
        dss.Text.Command("Set controlmode=static")
        dss.Text.Command("Set algorithm=newton")

        self._compiled = True
        logger.info(f"Compiled: {dss.Circuit.Name()}, "
                    f"{dss.Circuit.NumBuses()} buses, "
                    f"{dss.Circuit.NumCktElements()} elements")

    def build_graph_once(self, force: bool = False) -> dict:
        """Extract topology and cache to disk. Load from cache if available."""
        from config import GRAPH_DIR
        cache = GRAPH_DIR / "graph_structure.pkl"
        if cache.exists() and not force:
            self._graph = load_graph(cache)
            logger.info("Graph loaded from cache")
        else:
            self._ensure_compiled()
            # Snapshot solve to populate element data
            dss.Text.Command("Set mode=snapshot")
            dss.Solution.Solve()
            self._graph = build_graph()
            save_graph(self._graph, cache)

        self._bus_names  = self._graph["bus_names"]
        self._bus_index  = self._graph["bus_index"]
        self._branch_names = self._collect_branch_names()
        return self._graph

    def run_sample(self,
                   fault_type: str,
                   phase: str,
                   bus: str,
                   resistance: float,
                   load_mult: float,
                   der_fraction: float,
                   start_time_hr: float = 12.0) -> dict:
        """
        Run one 20-cycle Dynamic simulation and return captured signals + labels.

        Parameters
        ----------
        fault_type   : 'normal' | 'LG' | 'LL' | 'LLG' | 'LLL' | 'HIF'
        phase        : phase string (e.g. 'A', 'AB', 'ABC', 'none')
        bus          : target bus name
        resistance   : fault resistance (Ω)
        load_mult    : load scaling factor
        der_fraction : PV/Wind output fraction (0–1)
        start_time_hr: time of day (hours) for LoadShape positioning

        Returns
        -------
        dict with keys: voltage_seq, current_seq, labels, metadata, graph_snapshot
        """
        self._ensure_compiled()

        # --- Step 1: Re-compile to clean state ---
        dss.Text.Command(f'Compile "{MASTER_DSS}"')

        # --- Step 2: Apply operating conditions ---
        self._set_operating_conditions(load_mult, der_fraction, start_time_hr)

        # --- Step 3: Snapshot solve for initial steady-state ---
        dss.Text.Command("Set mode=snapshot")
        dss.Text.Command(f"Set maxiterations={SNAPSHOT_MAXITER}")
        dss.Text.Command(f"Set tolerance={SNAPSHOT_TOLERANCE}")
        dss.Solution.Solve()

        if not dss.Solution.Converged():
            logger.warning(f"Snapshot did not converge (bus={bus}, type={fault_type})")

        # Refresh branch/bus lists after compile
        self._bus_names    = list(dss.Circuit.AllBusNames())
        self._bus_index    = {n: i for i, n in enumerate(self._bus_names)}
        self._branch_names = self._collect_branch_names()

        N_buses    = len(self._bus_names)
        N_branches = len(self._branch_names)

        # Pre-allocate output arrays
        voltage_seq = np.zeros((TOTAL_CYCLES, N_buses, 6),    dtype=np.float32)
        current_seq = np.zeros((TOTAL_CYCLES, N_branches, 6), dtype=np.float32)

        # --- Step 4: Repeated Snapshot solves (one per cycle) ---
        # Dynamic mode (EMT integration) diverges on this distribution network
        # because there are no rotating machine dynamic models — only static loads,
        # PV voltage-sources, and a Vsource at the substation. True Dynamic mode
        # requires machine swing equations to stay numerically stable; without them
        # the integrator diverges and puVmagAngle() returns values ~10^15 pu.
        # Quasi-static Snapshot solves at 1-cycle resolution are physically correct
        # for power-frequency fault detection and classification at this grid scale.
        dss.Text.Command("Set mode=snapshot")
        dss.Text.Command(f"Set maxiterations={SNAPSHOT_MAXITER}")
        dss.Text.Command(f"Set tolerance={SNAPSHOT_TOLERANCE}")
        dss.Text.Command("Set controlmode=static")

        # HIF arc resistance sequence — time-indexed over the fault window
        hif_resistances = []
        if fault_type == "HIF":
            hif_resistances = get_hif_resistance_sequence(
                FAULT_CYCLES, DYN_STEPSIZE_S
            )

        # --- Step 5: Pre-fault window ---
        for t in range(PRE_FAULT_CYCLES):
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                logger.debug(f"Pre-fault step {t} did not converge (bus={bus})")
            voltage_seq[t], current_seq[t] = self._capture_signals()

        # --- Step 6: Inject fault (skip for normal) ---
        is_fault = (fault_type != "normal")
        if is_fault:
            inject_fault(bus, fault_type, phase, resistance)

        # --- Step 7: Fault window ---
        for t in range(FAULT_CYCLES):
            if fault_type == "HIF":
                update_hif_resistance(hif_resistances[t])
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                logger.debug(f"Fault step {t} did not converge (bus={bus}, type={fault_type})")
            tidx = PRE_FAULT_CYCLES + t
            voltage_seq[tidx], current_seq[tidx] = self._capture_signals()

        # --- Step 8: Remove fault ---
        if is_fault:
            remove_fault()

        # --- Step 9: Post-fault window ---
        for t in range(POST_FAULT_CYCLES):
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                logger.debug(f"Post-fault step {t} did not converge (bus={bus})")
            tidx = PRE_FAULT_CYCLES + FAULT_CYCLES + t
            voltage_seq[tidx], current_seq[tidx] = self._capture_signals()

        # --- Step 10: Assemble sample ---
        from config import FAULT_TYPE_LABEL, FAULT_PHASE_LABEL
        bus_idx = self._bus_index.get(bus.lower(), -1)

        labels = {
            "label_detection": 0 if fault_type == "normal" else 1,
            "label_type":      FAULT_TYPE_LABEL[fault_type],
            "label_phase":     FAULT_PHASE_LABEL.get(phase, 0),
            "label_location":  bus_idx,
        }

        metadata = {
            "fault_type":   fault_type,
            "phase":        phase,
            "bus":          bus,
            "resistance":   resistance,
            "load_mult":    load_mult,
            "der_fraction": der_fraction,
            "start_time_hr": start_time_hr,
        }

        graph_snapshot = {
            "edge_index": self._graph["edge_index"] if self._graph else np.array([[], []], dtype=np.int64),
            "edge_attr":  self._graph["edge_attr"]  if self._graph else np.zeros((0, 3), dtype=np.float32),
        }

        return {
            "voltage_seq":   voltage_seq,
            "current_seq":   current_seq,
            "labels":        labels,
            "metadata":      metadata,
            "graph_snapshot": graph_snapshot,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_compiled(self):
        if not self._compiled:
            self.compile()

    def _set_operating_conditions(self, load_mult: float, der_fraction: float,
                                   start_time_hr: float) -> None:
        """Scale loads and DER generation to target operating point."""
        # Load scaling via OpenDSS global multiplier
        dss.Solution.LoadMult(load_mult)

        # DER scaling: set irradiance for PV systems
        irradiance = der_fraction * 1000.0   # W/m² equivalent
        dss.PVsystems.First()
        while True:
            name = dss.PVsystems.Name()
            if not name:
                break
            dss.Text.Command(f"PVSystem.{name}.irradiance={irradiance:.1f}")
            if not dss.PVsystems.Next():
                break

        # Wind/generator scaling
        dss.Generators.First()
        while True:
            name = dss.Generators.Name()
            if not name:
                break
            # Scale kW proportionally — generators declared at rated output
            dss.Text.Command(f"Generator.{name}.kW={dss.Generators.kW() * der_fraction:.2f}")
            if not dss.Generators.Next():
                break

    def _capture_signals(self):
        """
        Read all bus voltages and branch currents from OpenDSS after a solve step.

        Returns
        -------
        v_row : np.ndarray [N_buses, 6]    — [Va_mag, Va_ang, Vb_mag, Vb_ang, Vc_mag, Vc_ang]
        i_row : np.ndarray [N_branches, 6] — [Ia_mag, Ia_ang, Ib_mag, Ib_ang, Ic_mag, Ic_ang]
        """
        N_buses    = len(self._bus_names)
        N_branches = len(self._branch_names)
        v_row = np.zeros((N_buses, 6),    dtype=np.float32)
        i_row = np.zeros((N_branches, 6), dtype=np.float32)

        # Voltages
        for i, bname in enumerate(self._bus_names):
            dss.Circuit.SetActiveBus(bname)
            phasor = dss.Bus.puVmagAngle()  # alternating [mag0, ang0, mag1, ang1, ...]
            n_nodes = dss.Bus.NumNodes()
            for ph in range(min(3, n_nodes)):
                v_row[i, ph * 2]     = phasor[ph * 2]     if len(phasor) > ph * 2     else 0.0
                v_row[i, ph * 2 + 1] = phasor[ph * 2 + 1] if len(phasor) > ph * 2 + 1 else 0.0

        # Currents — capture both magnitude and angle per phase
        for j, bname in enumerate(self._branch_names):
            dss.Circuit.SetActiveElement(bname)
            mag_ang = dss.CktElement.CurrentsMagAng()  # [mag0, ang0, mag1, ang1, ...]
            for ph in range(3):
                mag_idx = ph * 2
                ang_idx = ph * 2 + 1
                i_row[j, ph * 2]     = mag_ang[mag_idx] if mag_ang and len(mag_ang) > mag_idx else 0.0
                i_row[j, ph * 2 + 1] = mag_ang[ang_idx] if mag_ang and len(mag_ang) > ang_idx else 0.0

        return v_row, i_row

    def _collect_branch_names(self) -> list:
        """Ordered list of branch element names for current extraction."""
        branches = []
        dss.Lines.First()
        while True:
            name = dss.Lines.Name()
            if not name:
                break
            branches.append(f"Line.{name}")
            if not dss.Lines.Next():
                break

        dss.Transformers.First()
        while True:
            name = dss.Transformers.Name()
            if not name:
                break
            branches.append(f"Transformer.{name}")
            if not dss.Transformers.Next():
                break

        return branches

    @property
    def bus_names(self) -> list:
        return self._bus_names

    @property
    def n_buses(self) -> int:
        return len(self._bus_names)

    @property
    def n_branches(self) -> int:
        return len(self._branch_names)

    @property
    def graph(self) -> dict:
        return self._graph
