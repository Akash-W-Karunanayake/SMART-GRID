"""
OpenDSS Engine
==============
Thin wrapper around opendssdirect.py.

CASING CONVENTION:
  All bus names returned by this engine are LOWERCASE.
  All element names (Line.X, Load.X) keep their original OpenDSS casing
  because OpenDSS element lookups are case-insensitive.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import opendssdirect as dss
from loguru import logger

from config import (
    settings, ALL_SWITCH_NAMES, TIE_SWITCH_NAMES,
    ALL_LOAD_NAMES, V_MIN_PU, V_MAX_PU, LINE_OVERLOAD_THRESHOLD_PCT,
)


def _normalize_bus(raw: str) -> str:
    """Strip node suffixes and lowercase: 'F09_Node3.1.2.3' → 'f09_node3'"""
    return raw.split(".")[0].lower()


class OpenDSSEngine:

    def __init__(self):
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def load_model(self, master_path=None) -> bool:
        path = Path(master_path) if master_path else settings.master_dss_path
        if not path.exists():
            logger.error(f"Master.dss not found at {path}")
            return False

        os.chdir(path.parent)
        dss.run_command(f"Compile [{path}]")

        err = dss.Error.Description()
        if err:
            logger.error(f"OpenDSS compile error: {err}")
            return False

        self._loaded = True
        logger.info(
            f"Model loaded: {dss.Circuit.Name()}, "
            f"{dss.Circuit.NumBuses()} buses, "
            f"{dss.Circuit.NumCktElements()} elements"
        )
        return True

    def reset(self) -> bool:
        return self.load_model()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── switch control ──────────────────────────────────────────────────

    def open_switch(self, name: str) -> None:
        dss.run_command(f"open Line.{name} 1")
        dss.run_command(f"Line.{name}.enabled=no")

    def close_switch(self, name: str) -> None:
        dss.run_command(f"Line.{name}.enabled=yes")
        dss.run_command(f"close Line.{name} 1")

    def set_switch(self, name: str, closed: bool) -> None:
        if closed:
            self.close_switch(name)
        else:
            self.open_switch(name)

    def get_all_switch_states(self) -> Dict[str, bool]:
        """Return {name: is_closed} for all 21 switches."""
        states: Dict[str, bool] = {}
        for sw_name in ALL_SWITCH_NAMES:
            dss.Circuit.SetActiveElement(f"Line.{sw_name}")
            states[sw_name] = bool(dss.CktElement.Enabled())
        return states

    def get_tie_switch_states(self) -> Dict[str, bool]:
        """Return {name: is_closed} for the 5 tie switches only."""
        states: Dict[str, bool] = {}
        for sw_name in TIE_SWITCH_NAMES:
            dss.Circuit.SetActiveElement(f"Line.{sw_name}")
            states[sw_name] = bool(dss.CktElement.Enabled())
        return states

    # ── power flow ──────────────────────────────────────────────────────

    def solve(self) -> bool:
        dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    def solve_at_step(self, step: int) -> bool:
        dss.Solution.Mode(1)      # daily
        dss.Solution.Number(1)
        dss.Solution.StepSize(900) # 15 min
        for _ in range(step):
            dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    # ── state extraction (ALL bus names are lowercase) ──────────────────

    def get_bus_voltages_pu(self) -> Dict[str, float]:
        """Return {bus_name_lower: avg_voltage_pu}."""
        result: Dict[str, float] = {}
        for bname in dss.Circuit.AllBusNames():
            key = bname.lower()
            dss.Circuit.SetActiveBus(bname)
            pu_vals = dss.Bus.puVmagAngle()
            mags = pu_vals[0::2] if len(pu_vals) > 0 else []
            positive = [m for m in mags if m > 0]
            result[key] = float(np.mean(positive)) if positive else 0.0
        return result

    def get_voltage_violations(self) -> List[str]:
        violations = []
        for bus, vpu in self.get_bus_voltages_pu().items():
            if vpu > 0.01 and (vpu < V_MIN_PU or vpu > V_MAX_PU):
                violations.append(bus)
        return violations

    def get_load_powers(self) -> Dict[str, float]:
        """Return {load_name: kW}."""
        powers: Dict[str, float] = {}
        flag = dss.Loads.First()
        while flag > 0:
            powers[dss.Loads.Name()] = dss.Loads.kW()
            flag = dss.Loads.Next()
        return powers

    def get_total_load_kw(self) -> float:
        return sum(self.get_load_powers().values())

    def _load_bus_map(self) -> Dict[str, str]:
        """Return {load_name: bus_name_lower}."""
        mapping: Dict[str, str] = {}
        flag = dss.Loads.First()
        while flag > 0:
            name = dss.Loads.Name()
            dss.Circuit.SetActiveElement(f"Load.{name}")
            buses = dss.CktElement.BusNames()
            if buses:
                mapping[name] = _normalize_bus(buses[0])
            flag = dss.Loads.Next()
        return mapping

    def get_energized_loads(self) -> List[str]:
        """Loads whose bus has voltage > 0.1 pu.
        Returns canonical names from ALL_LOAD_NAMES (preserving original casing).
        """
        bus_v = self.get_bus_voltages_pu()
        load_bus = self._load_bus_map()
        # OpenDSS returns lowercase load names; map back to canonical casing
        dss_energized = {
            name.lower() for name, bus in load_bus.items()
            if bus_v.get(bus, 0.0) > 0.1
        }
        return [ld for ld in ALL_LOAD_NAMES if ld.lower() in dss_energized]

    def get_de_energized_loads(self) -> List[str]:
        energized = set(self.get_energized_loads())
        return [ld for ld in ALL_LOAD_NAMES if ld not in energized]

    def get_line_loading_pct(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        flag = dss.Lines.First()
        while flag > 0:
            name = dss.Lines.Name()
            dss.Circuit.SetActiveElement(f"Line.{name}")
            norm_amps = dss.CktElement.NormalAmps()
            if norm_amps > 0:
                currents = dss.CktElement.CurrentsMagAng()
                max_i = max(currents[0::2]) if currents else 0
                result[name] = (max_i / norm_amps) * 100
            flag = dss.Lines.Next()
        return result

    def get_overloaded_lines(self) -> List[str]:
        return [
            n for n, pct in self.get_line_loading_pct().items()
            if pct > LINE_OVERLOAD_THRESHOLD_PCT
        ]

    def get_grid_state(self) -> Dict:
        """Full snapshot — all bus names are lowercase."""
        return {
            "converged": bool(dss.Solution.Converged()),
            "all_switch_states": self.get_all_switch_states(),
            "tie_switch_states": self.get_tie_switch_states(),
            "bus_voltages_pu": self.get_bus_voltages_pu(),
            "load_powers_kw": self.get_load_powers(),
            "energized_loads": self.get_energized_loads(),
            "de_energized_loads": self.get_de_energized_loads(),
            "voltage_violations": self.get_voltage_violations(),
            "overloaded_lines": self.get_overloaded_lines(),
            "total_load_kw": self.get_total_load_kw(),
        }


engine = OpenDSSEngine()
