"""
OpenDSS Service - Core interface for OpenDSS power system simulation.
Handles loading, running, and extracting data from the OpenDSS model.
"""
import opendssdirect as dss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from config import get_dss_master_path, get_dss_file_path, settings

logger = logging.getLogger(__name__)


@dataclass
class BusData:
    """Data class for bus information."""
    name: str
    base_kv: float
    voltage_pu: List[float]
    voltage_angle: List[float]
    coordinates: Optional[Tuple[float, float]] = None
    num_nodes: int = 3


@dataclass
class LineData:
    """Data class for line information."""
    name: str
    bus1: str
    bus2: str
    length: float
    current_amps: List[float]
    power_kw: float
    power_kvar: float
    losses_kw: float
    enabled: bool = True


@dataclass
class TransformerData:
    """Data class for transformer information."""
    name: str
    buses: List[str]
    kva: float
    loading_percent: float
    power_kw: float
    power_kvar: float


@dataclass
class LoadData:
    """Data class for load information."""
    name: str
    bus: str
    kw: float
    kvar: float
    voltage_pu: float


@dataclass
class GeneratorData:
    """Data class for generator/PV information."""
    name: str
    bus: str
    kw: float
    kvar: float
    type: str  # 'generator', 'pvsystem', etc.


@dataclass
class GridState:
    """Complete grid state at a point in time."""
    timestamp: float = 0.0
    converged: bool = False
    total_power_kw: float = 0.0
    total_power_kvar: float = 0.0
    total_losses_kw: float = 0.0
    total_generation_kw: float = 0.0
    total_solar_kw: float = 0.0
    total_load_kw: float = 0.0
    buses: Dict[str, BusData] = field(default_factory=dict)
    lines: Dict[str, LineData] = field(default_factory=dict)
    transformers: Dict[str, TransformerData] = field(default_factory=dict)
    loads: Dict[str, LoadData] = field(default_factory=dict)
    generators: Dict[str, GeneratorData] = field(default_factory=dict)
    voltage_violations: List[str] = field(default_factory=list)
    overloaded_elements: List[str] = field(default_factory=list)


class OpenDSSService:
    """
    Service class for OpenDSS operations.
    Provides high-level interface for power system simulation.
    """

    def __init__(self):
        self._model_loaded = False
        self._circuit_name = ""
        self._base_frequency = 50  # Sri Lankan grid
        self._current_load_mult = 1.0  # Track current load multiplier

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def load_model(self, master_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load the OpenDSS model from Master.dss file.

        Returns:
            Dict with loading status and circuit information.
        """
        try:
            if master_file is None:
                master_file = get_dss_master_path()

            logger.info(f"Loading OpenDSS model from: {master_file}")

            # Clear any existing circuit
            dss.Basic.ClearAll()

            # Set the data path to the model directory
            dss.Basic.DataPath(str(master_file.parent))

            # Compile the master file
            dss.Text.Command(f'Compile "{master_file}"')

            # Check for OpenDSS errors after compile
            error_desc = dss.Error.Description()
            if error_desc:
                raise RuntimeError(error_desc)

            # Check if circuit was created
            if dss.Circuit.Name() == "":
                raise RuntimeError("Failed to load circuit - no circuit name found")

            self._circuit_name = dss.Circuit.Name()
            self._model_loaded = True

            # Get circuit info
            circuit_info = self._get_circuit_info()

            # Log element counts for debugging
            logger.info(f"Successfully loaded circuit: {self._circuit_name}")
            logger.info(f"  - Buses: {dss.Circuit.NumBuses()}")
            logger.info(f"  - Elements: {dss.Circuit.NumCktElements()}")

            # Count specific element types
            num_lines = 0
            dss.Lines.First()
            while dss.Lines.Name():
                num_lines += 1
                if not dss.Lines.Next():
                    break
            logger.info(f"  - Lines: {num_lines}")

            num_transformers = 0
            dss.Transformers.First()
            while dss.Transformers.Name():
                num_transformers += 1
                if not dss.Transformers.Next():
                    break
            logger.info(f"  - Transformers: {num_transformers}")

            num_loads = 0
            dss.Loads.First()
            while dss.Loads.Name():
                num_loads += 1
                if not dss.Loads.Next():
                    break
            logger.info(f"  - Loads: {num_loads}")

            return {
                "success": True,
                "circuit_name": self._circuit_name,
                "info": circuit_info
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            return {
                "success": False,
                "error": str(e)
            }

    def _get_circuit_info(self) -> Dict[str, Any]:
        """Get basic circuit information."""
        return {
            "name": dss.Circuit.Name(),
            "num_buses": dss.Circuit.NumBuses(),
            "num_nodes": dss.Circuit.NumNodes(),
            "num_elements": dss.Circuit.NumCktElements(),
            "base_frequency": dss.Solution.Frequency(),
            "total_power": {
                "kw": dss.Circuit.TotalPower()[0],
                "kvar": dss.Circuit.TotalPower()[1]
            }
        }

    def solve(self) -> bool:
        """
        Run power flow solution with robust settings for high-DG scenarios.

        Returns:
            True if solution converged, False otherwise.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use robust solver settings for high solar penetration scenarios
        dss.Text.Command("Set mode=snapshot")  # Snapshot mode for single time point
        dss.Text.Command("Set controlmode=static")  # Allow regulators/capacitor controls
        dss.Text.Command("Set algorithm=newton")  # Newton-Raphson is more robust
        dss.Text.Command("Set maxiterations=300")  # Adequate iterations
        dss.Text.Command("Set tolerance=0.0001")  # Standard tolerance

        dss.Solution.Solve()
        converged = dss.Solution.Converged()

        # If first attempt fails, try with Normal current injection method
        if not converged:
            dss.Text.Command("Set algorithm=normal")  # Switch to Normal method
            dss.Text.Command("Set maxiterations=500")
            dss.Solution.Solve()
            converged = dss.Solution.Converged()

            if not converged:
                # Fall back to Newton with relaxed tolerance
                dss.Text.Command("Set algorithm=newton")
                dss.Text.Command("Set tolerance=0.001")  # Relaxed tolerance
                dss.Text.Command("Set maxiterations=500")
                dss.Solution.Solve()
                converged = dss.Solution.Converged()

                if not converged:
                    # Final attempt with very relaxed settings
                    dss.Text.Command("Set tolerance=0.01")
                    dss.Text.Command("Set maxiterations=1000")
                    dss.Solution.Solve()
                    converged = dss.Solution.Converged()

                    if converged:
                        logger.warning("Power flow converged with very relaxed tolerance (0.01)")
                    else:
                        logger.warning(
                            f"Power flow did NOT converge after all attempts "
                            f"(load_mult={self._current_load_mult:.3f}). "
                            f"Using last iteration values."
                        )
                else:
                    logger.info("Power flow converged with relaxed tolerance (0.001)")
            else:
                logger.info("Power flow converged with Normal algorithm")

        return converged

    def get_grid_state(self) -> GridState:
        """
        Get complete grid state after solving.

        Returns:
            GridState object with all system data.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Solve first
        converged = self.solve()

        state = GridState(
            converged=converged,
            total_power_kw=-dss.Circuit.TotalPower()[0],  # Negative = generation
            total_power_kvar=-dss.Circuit.TotalPower()[1],
            total_losses_kw=dss.Circuit.Losses()[0] / 1000,  # Convert to kW
        )

        # Collect data
        state.buses = self._get_all_buses()
        state.lines = self._get_all_lines()
        state.transformers = self._get_all_transformers()
        state.loads = self._get_all_loads()
        state.generators = self._get_all_generators()

        # Calculate totals (apply load multiplier to get actual load)
        nominal_load = sum(load.kw for load in state.loads.values())
        state.total_load_kw = nominal_load * self._current_load_mult
        state.total_generation_kw = sum(gen.kw for gen in state.generators.values())
        # Calculate solar-only generation (PV systems only)
        state.total_solar_kw = sum(gen.kw for gen in state.generators.values() if gen.type == "pvsystem")

        # Check for violations
        state.voltage_violations = self._check_voltage_violations(state.buses)
        state.overloaded_elements = self._check_overloads()

        return state

    def _get_all_buses(self) -> Dict[str, BusData]:
        """Get all bus data."""
        buses = {}
        bus_names = dss.Circuit.AllBusNames()

        for name in bus_names:
            dss.Circuit.SetActiveBus(name)

            # Get voltage magnitudes and angles
            voltages = dss.Bus.puVmagAngle()
            num_nodes = dss.Bus.NumNodes()

            # Parse voltage data (alternating magnitude, angle)
            v_mag = voltages[0::2][:num_nodes] if voltages else [0.0]
            v_ang = voltages[1::2][:num_nodes] if voltages else [0.0]

            buses[name] = BusData(
                name=name,
                base_kv=dss.Bus.kVBase(),
                voltage_pu=list(v_mag),
                voltage_angle=list(v_ang),
                num_nodes=num_nodes,
                coordinates=(dss.Bus.X(), dss.Bus.Y()) if dss.Bus.X() != 0 else None
            )

        return buses

    def _get_all_lines(self) -> Dict[str, LineData]:
        """Get all line data."""
        lines = {}

        dss.Lines.First()
        while True:
            name = dss.Lines.Name()
            if not name:
                break

            # Set as active circuit element to get powers
            dss.Circuit.SetActiveElement(f"Line.{name}")
            powers = dss.CktElement.Powers()
            currents = dss.CktElement.CurrentsMagAng()
            losses = dss.CktElement.Losses()

            lines[name] = LineData(
                name=name,
                bus1=dss.Lines.Bus1(),
                bus2=dss.Lines.Bus2(),
                length=dss.Lines.Length(),
                current_amps=list(currents[0::2][:3]) if currents else [0.0],
                power_kw=powers[0] if powers else 0.0,
                power_kvar=powers[1] if powers else 0.0,
                losses_kw=losses[0] / 1000 if losses else 0.0,
                enabled=dss.CktElement.Enabled()
            )

            if not dss.Lines.Next():
                break

        return lines

    def _get_all_transformers(self) -> Dict[str, TransformerData]:
        """Get all transformer data."""
        transformers = {}

        dss.Transformers.First()
        while True:
            name = dss.Transformers.Name()
            if not name:
                break

            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            powers = dss.CktElement.Powers()

            # Get loading percentage
            dss.Text.Command(f"? Transformer.{name}.%loadloss")

            transformers[name] = TransformerData(
                name=name,
                buses=[dss.Transformers.Wdg(), dss.Transformers.Wdg()],  # Simplified
                kva=dss.Transformers.kVA(),
                loading_percent=self._calculate_transformer_loading(name),
                power_kw=abs(powers[0]) if powers else 0.0,
                power_kvar=abs(powers[1]) if powers else 0.0
            )

            if not dss.Transformers.Next():
                break

        return transformers

    def _calculate_transformer_loading(self, name: str) -> float:
        """Calculate transformer loading percentage."""
        try:
            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            powers = dss.CktElement.Powers()
            kva_rating = dss.Transformers.kVA()

            if powers and kva_rating > 0:
                # Calculate apparent power
                p = abs(powers[0])
                q = abs(powers[1])
                s = np.sqrt(p**2 + q**2)
                return (s / kva_rating) * 100
            return 0.0
        except:
            return 0.0

    def _get_all_loads(self) -> Dict[str, LoadData]:
        """Get all load data."""
        loads = {}

        dss.Loads.First()
        while True:
            name = dss.Loads.Name()
            if not name:
                break

            # Set load as active element to get bus names
            dss.Circuit.SetActiveElement(f"Load.{name}")
            bus_names = dss.CktElement.BusNames()
            bus = bus_names[0].split('.')[0] if bus_names else ""  # Remove node specification
            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.puVmagAngle()
            v_pu = voltages[0] if voltages else 1.0

            loads[name] = LoadData(
                name=name,
                bus=bus,
                kw=dss.Loads.kW() * self._current_load_mult,
                kvar=dss.Loads.kvar() * self._current_load_mult,
                voltage_pu=v_pu
            )

            if not dss.Loads.Next():
                break

        return loads

    def _get_all_generators(self) -> Dict[str, GeneratorData]:
        """Get all generators and PV systems."""
        generators = {}

        # Get regular generators
        dss.Generators.First()
        while True:
            name = dss.Generators.Name()
            if not name:
                break

            dss.Circuit.SetActiveElement(f"Generator.{name}")
            powers = dss.CktElement.Powers()

            generators[name] = GeneratorData(
                name=name,
                bus=dss.Generators.Bus1().split('.')[0],
                kw=-powers[0] if powers else 0.0,  # Generation is negative in OpenDSS
                kvar=-powers[1] if powers else 0.0,
                type="generator"
            )

            if not dss.Generators.Next():
                break

        # Get PV systems
        dss.PVsystems.First()
        while True:
            name = dss.PVsystems.Name()
            if not name:
                break

            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            powers = dss.CktElement.Powers()

            generators[f"PV_{name}"] = GeneratorData(
                name=name,
                bus=dss.PVsystems.Bus1().split('.')[0] if hasattr(dss.PVsystems, 'Bus1') else "",
                kw=-powers[0] if powers else 0.0,
                kvar=-powers[1] if powers else 0.0,
                type="pvsystem"
            )

            if not dss.PVsystems.Next():
                break

        return generators

    def _check_voltage_violations(self, buses: Dict[str, BusData],
                                   v_min: float = 0.90,
                                   v_max: float = 1.10) -> List[str]:
        """Check for voltage violations outside acceptable range.

        Args:
            buses: Dictionary of bus data
            v_min: Minimum acceptable voltage in per-unit (default 0.90 = -10%)
            v_max: Maximum acceptable voltage in per-unit (default 1.10 = +10%)

        Returns:
            List of violation descriptions, limited to worst 10 violations
        """
        violations = []
        for name, bus in buses.items():
            # Skip the source bus (typically has nominal voltage)
            if 'source' in name.lower():
                continue
            for i, v in enumerate(bus.voltage_pu):
                # Only check valid voltage readings (> 0.1 to filter noise)
                if v > 0.1 and (v < v_min or v > v_max):
                    deviation = abs(v - 1.0) * 100  # Percentage deviation from nominal
                    violations.append((deviation, f"{name} (node {i+1}): {v:.4f} pu"))

        # Sort by severity (worst first) and return top 10
        violations.sort(reverse=True, key=lambda x: x[0])
        return [v[1] for v in violations[:10]]

    def _check_overloads(self, threshold: float = 100.0) -> List[str]:
        """Check for overloaded elements."""
        overloads = []

        # Check lines
        dss.Lines.First()
        while True:
            name = dss.Lines.Name()
            if not name:
                break

            # Get emergency amps rating
            norm_amps = dss.Lines.NormAmps()
            if norm_amps > 0:
                dss.Circuit.SetActiveElement(f"Line.{name}")
                currents = dss.CktElement.CurrentsMagAng()
                if currents:
                    max_current = max(currents[0::2][:3])
                    loading = (max_current / norm_amps) * 100
                    if loading > threshold:
                        overloads.append(f"Line.{name}: {loading:.1f}%")

            if not dss.Lines.Next():
                break

        return overloads

    def read_current_state(self) -> GridState:
        """
        Read the current grid state WITHOUT re-solving.

        Use this after a pipeline simulation to read the state left by the
        last Solve() call, preserving daily-mode results.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        converged = dss.Solution.Converged()

        state = GridState(
            converged=converged,
            total_power_kw=-dss.Circuit.TotalPower()[0],
            total_power_kvar=-dss.Circuit.TotalPower()[1],
            total_losses_kw=dss.Circuit.Losses()[0] / 1000,
        )

        state.buses = self._get_all_buses()
        state.lines = self._get_all_lines()
        state.transformers = self._get_all_transformers()
        state.loads = self._get_all_loads()
        state.generators = self._get_all_generators()

        state.total_load_kw = sum(load.kw for load in state.loads.values())
        state.total_generation_kw = sum(gen.kw for gen in state.generators.values())
        state.total_solar_kw = sum(
            gen.kw for gen in state.generators.values() if gen.type == "pvsystem"
        )

        state.voltage_violations = self._check_voltage_violations(state.buses)
        state.overloaded_elements = self._check_overloads()

        return state

    def get_topology(self) -> Dict[str, Any]:
        """
        Get network topology for visualization.

        Returns:
            Dict with nodes (buses) and edges (lines/transformers).
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        nodes = []
        edges = []

        # Get bus nodes
        bus_names = dss.Circuit.AllBusNames()
        for name in bus_names:
            dss.Circuit.SetActiveBus(name)
            nodes.append({
                "id": name,
                "label": name,
                "type": "bus",
                "kv": dss.Bus.kVBase(),
                "x": dss.Bus.X() if dss.Bus.X() != 0 else None,
                "y": dss.Bus.Y() if dss.Bus.Y() != 0 else None
            })

        # Get line edges
        dss.Lines.First()
        while True:
            name = dss.Lines.Name()
            if not name:
                break

            bus1 = dss.Lines.Bus1().split('.')[0]
            bus2 = dss.Lines.Bus2().split('.')[0]

            edges.append({
                "id": f"Line.{name}",
                "source": bus1,
                "target": bus2,
                "type": "line",
                "label": name
            })

            if not dss.Lines.Next():
                break

        # Get transformer edges
        dss.Transformers.First()
        while True:
            name = dss.Transformers.Name()
            if not name:
                break

            dss.Circuit.SetActiveElement(f"Transformer.{name}")
            buses = dss.CktElement.BusNames()

            if len(buses) >= 2:
                bus1 = buses[0].split('.')[0]
                bus2 = buses[1].split('.')[0]

                edges.append({
                    "id": f"Transformer.{name}",
                    "source": bus1,
                    "target": bus2,
                    "type": "transformer",
                    "label": name
                })

            if not dss.Transformers.Next():
                break

        return {
            "nodes": nodes,
            "edges": edges
        }

    def set_load_multiplier(self, multiplier: float):
        """Set global load multiplier for all loads."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        dss.Solution.LoadMult(multiplier)
        self._current_load_mult = multiplier
        logger.info(f"Load multiplier set to: {multiplier}")

    def set_generation_multiplier(self, multiplier: float):
        """Set generation multiplier for PV systems.

        Args:
            multiplier: Value between 0.0 and 1.0 representing solar intensity.
                       Will be scaled to irradiance (0-1000 W/m²).
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        # Scale multiplier (0-1) to irradiance (0-1000 W/m²)
        # Using 1000 W/m² as peak solar irradiance
        irradiance = multiplier * 1000.0

        dss.PVsystems.First()
        while True:
            name = dss.PVsystems.Name()
            if not name:
                break

            # Adjust irradiance to simulate generation changes
            dss.Text.Command(f"PVSystem.{name}.irradiance={irradiance}")

            if not dss.PVsystems.Next():
                break

        logger.info(f"Generation multiplier set to: {multiplier} (irradiance: {irradiance} W/m²)")

    def inject_fault(self, bus: str, fault_type: str = "3phase",
                     resistance: float = 0.0001) -> Dict[str, Any]:
        """
        Inject a fault at specified bus.

        Args:
            bus: Bus name where fault occurs
            fault_type: Type of fault ('3phase', 'lg', 'll', 'llg')
            resistance: Fault resistance in ohms

        Returns:
            Fault simulation results
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            # Define fault
            dss.Text.Command(f'New Fault.TestFault Bus1={bus} phases=3 r={resistance}')

            # Solve with fault
            self.solve()

            # Get fault current
            dss.Circuit.SetActiveElement("Fault.TestFault")
            currents = dss.CktElement.CurrentsMagAng()

            result = {
                "success": True,
                "bus": bus,
                "fault_type": fault_type,
                "fault_current_amps": currents[0] if currents else 0.0,
                "resistance": resistance
            }

            # Remove fault
            dss.Text.Command("Fault.TestFault.enabled=no")

            return result

        except Exception as e:
            logger.error(f"Fault injection failed: {e}")
            return {"success": False, "error": str(e)}

    def get_voltage_profile(self) -> pd.DataFrame:
        """Get voltage profile for all buses."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        self.solve()

        data = []
        bus_names = dss.Circuit.AllBusNames()

        for name in bus_names:
            dss.Circuit.SetActiveBus(name)
            voltages = dss.Bus.puVmagAngle()
            kv_base = dss.Bus.kVBase()

            if voltages:
                v_avg = np.mean(voltages[0::2][:dss.Bus.NumNodes()])
                data.append({
                    "bus": name,
                    "voltage_pu": v_avg,
                    "kv_base": kv_base
                })

        return pd.DataFrame(data)

    def run_time_series(self, hours: int = 24, step_minutes: int = 60) -> List[GridState]:
        """
        Run time-series simulation.

        Args:
            hours: Number of hours to simulate
            step_minutes: Time step in minutes

        Returns:
            List of GridState objects for each time step
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded.")

        results = []
        total_steps = int((hours * 60) / step_minutes)

        # Set up time-series mode
        dss.Text.Command("Set Mode=Daily")
        dss.Text.Command(f"Set Stepsize={step_minutes}m")
        dss.Text.Command("Set Number=1")

        for step in range(total_steps):
            dss.Solution.Solve()
            state = self.get_grid_state()
            state.timestamp = step * step_minutes / 60  # Convert to hours
            results.append(state)

        return results

    def run_daily_simulation(self, steps: int = 96) -> List[GridState]:
        """
        Run daily simulation using OpenDSS native daily mode with LoadShapes.

        The DSS model must already be configured with:
        - mode=daily, stepsize=15m, number=96 in Master.dss
        - LoadShape profiles assigned to all loads/generators

        LoadShapes drive load/generation variation automatically.
        No manual set_load_multiplier or set_generation_multiplier calls needed.

        Args:
            steps: Number of simulation steps (default 96 for 15-min over 24h)

        Returns:
            List of GridState objects, one per time step
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []

        # Configure daily mode (in case Master.dss was already compiled with snapshot)
        dss.Text.Command("Set Mode=Daily")
        dss.Text.Command("Set Stepsize=15m")
        dss.Text.Command("Set Number=1")
        dss.Text.Command("Set controlmode=static")

        for step in range(steps):
            dss.Solution.Solve()
            converged = dss.Solution.Converged()

            state = GridState(
                timestamp=step * 0.25,  # Hours
                converged=converged,
                total_power_kw=-dss.Circuit.TotalPower()[0],
                total_power_kvar=-dss.Circuit.TotalPower()[1],
                total_losses_kw=dss.Circuit.Losses()[0] / 1000,
            )

            # Collect component data
            state.buses = self._get_all_buses()
            state.lines = self._get_all_lines()
            state.transformers = self._get_all_transformers()
            state.loads = self._get_all_loads()
            state.generators = self._get_all_generators()

            # Calculate totals from actual solved values
            state.total_load_kw = sum(load.kw for load in state.loads.values())
            state.total_generation_kw = sum(gen.kw for gen in state.generators.values())
            state.total_solar_kw = sum(
                gen.kw for gen in state.generators.values() if gen.type == "pvsystem"
            )

            # Check violations
            state.voltage_violations = self._check_voltage_violations(state.buses)
            state.overloaded_elements = self._check_overloads()

            if not converged:
                logger.warning(f"Step {step} (hour {step * 0.25:.2f}): did not converge")

            results.append(state)

        logger.info(f"Daily simulation complete: {len(results)} steps, "
                     f"{sum(1 for r in results if r.converged)}/{len(results)} converged")

        return results


# Singleton instance
opendss_service = OpenDSSService()
