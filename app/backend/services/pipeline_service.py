"""
Pipeline Service - Bridges the scripts/ pipeline with the FastAPI backend.

Runs data preparation (disaggregate, generate shapes, update DSS refs)
then executes OpenDSS daily simulation, collecting per-day results.
Supports single-day and multi-day (date range) modes.
"""
import asyncio
import importlib.util
import logging
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import numpy as np

# Add scripts/ to path so we can import pipeline sub-packages
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Load scripts/config.py explicitly to avoid collision with app/backend/config.py
# (both are named "config" so sys.modules caching would return the wrong one)
_spec = importlib.util.spec_from_file_location("scripts_config", SCRIPTS_DIR / "config.py")
scripts_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scripts_config)

from .opendss_service import opendss_service

logger = logging.getLogger(__name__)


@dataclass
class SimulationTask:
    """Tracks a running simulation task."""
    task_id: str
    status: str = "pending"  # pending, preparing, running, completed, error
    mode: str = "single"     # single or range
    start_date: str = ""
    end_date: str = ""
    total_days: int = 0
    current_day: int = 0
    current_date: str = ""
    completed_days: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class PipelineService:
    """Service for running pipeline + OpenDSS simulation from the web app."""

    def __init__(self):
        self._tasks: Dict[str, SimulationTask] = {}
        self._running_task_id: Optional[str] = None

    def get_task(self, task_id: str) -> Optional[SimulationTask]:
        return self._tasks.get(task_id)

    @property
    def is_busy(self) -> bool:
        return self._running_task_id is not None

    def _prepare_date(self, target_date: str) -> dict:
        """Run pipeline phases 2-4 for a single date (synchronous).

        Returns multipliers dict from disaggregation.
        """
        # Temporarily swap sys.modules['config'] so that script sub-modules
        # (disaggregate.py, generate_*.py) which do "from config import ..."
        # resolve to scripts/config.py instead of app/backend/config.py.
        # We restore the original afterwards to avoid breaking backend imports.
        _saved_config = sys.modules.get('config')
        sys.modules['config'] = scripts_config
        try:
            from loadshape_generation.disaggregate import disaggregate
            from loadshape_generation.generate_load_shapes import generate_load_shapes
            from loadshape_generation.generate_solar_shapes import generate_solar_shapes
            from loadshape_generation.generate_wind_shapes import generate_wind_shapes
            from loadshape_generation.generate_ujps_shapes import generate_ujps_shapes
            from dss_date_updater import update_dss_references

            # Phase 2: Disaggregate
            multipliers = disaggregate(target_date)

            # Phase 3: Generate shapes
            generate_load_shapes(target_date, multipliers)
            generate_solar_shapes(target_date, multipliers)
            generate_wind_shapes(target_date, multipliers)
            generate_ujps_shapes(target_date, multipliers)

            # Phase 4: Update DSS date references
            update_dss_references(scripts_config.DSS_DATE_FILES, target_date)

            return multipliers
        finally:
            # Restore original config module
            if _saved_config is not None:
                sys.modules['config'] = _saved_config
            else:
                sys.modules.pop('config', None)

    def _run_simulation(self) -> List[Dict[str, Any]]:
        """Reload DSS model and run 96-step daily simulation.

        Returns list of per-step result dicts.
        """
        import opendssdirect as dss

        MASTER_DSS = scripts_config.MASTER_DSS

        # Reload the model (files changed on disk)
        dss.Basic.ClearAll()
        dss.Basic.DataPath(str(MASTER_DSS.parent))
        dss.Text.Command(f'Compile "{MASTER_DSS}"')

        error = dss.Error.Description()
        if error:
            raise RuntimeError(f"DSS compile error: {error}")

        # Mark opendss_service as loaded so grid state endpoints work
        opendss_service._model_loaded = True
        opendss_service._circuit_name = dss.Circuit.Name()

        # Run daily simulation
        dss.Text.Command("Set Mode=Daily")
        dss.Text.Command("Set Stepsize=15m")
        dss.Text.Command("Set Number=1")
        dss.Text.Command("Set controlmode=static")
        dss.Text.Command("Set algorithm=newton")
        dss.Text.Command("Set maxiterations=1000")
        dss.Text.Command("Set tolerance=0.001")

        FEEDER_HEAD_LINES = {
            "F06": "Line.F06_Sec2", "F07": "Line.F07_Sec2",
            "F08": "Line.F08_Sec2", "F09": "Line.F09_Sec2",
            "F10": "Line.F10_Sec2", "F11": "Line.F11_Sec2",
            "F12": "Line.F12_Sec2",
        }

        steps = []
        for step in range(96):
            dss.Solution.Solve()
            converged = dss.Solution.Converged()

            step_data = {
                "step": step,
                "hour": round(step * 0.25, 2),
                "converged": converged,
            }

            if converged:
                step_data["total_power_kw"] = round(-dss.Circuit.TotalPower()[0], 2)
                step_data["total_losses_kw"] = round(dss.Circuit.Losses()[0] / 1000, 2)

                # Per-feeder power
                for feeder, element in FEEDER_HEAD_LINES.items():
                    try:
                        dss.Circuit.SetActiveElement(element)
                        powers = dss.CktElement.Powers()
                        if powers and len(powers) >= 6:
                            step_data[f"power_{feeder}_kw"] = round(
                                powers[0] + powers[2] + powers[4], 2
                            )
                    except Exception:
                        pass

                # ── Per-step generation: PV solar + wind + thermal ──
                solar_kw = 0.0
                wind_kw = 0.0
                thermal_kw = 0.0
                try:
                    # PV systems
                    idx = dss.PVsystems.First()
                    while idx > 0:
                        name = dss.PVsystems.Name()
                        dss.Circuit.SetActiveElement(f"PVSystem.{name}")
                        p = dss.CktElement.Powers()
                        if p:
                            solar_kw += -p[0]
                        idx = dss.PVsystems.Next()

                    # Regular generators (wind + thermal)
                    idx = dss.Generators.First()
                    while idx > 0:
                        name = dss.Generators.Name()
                        dss.Circuit.SetActiveElement(f"Generator.{name}")
                        p = dss.CktElement.Powers()
                        gen_kw = -p[0] if p else 0.0
                        if "wind" in name.lower():
                            wind_kw += gen_kw
                        else:
                            thermal_kw += gen_kw
                        idx = dss.Generators.Next()
                except Exception:
                    pass

                step_data["total_solar_kw"] = round(solar_kw, 2)
                step_data["total_wind_kw"] = round(wind_kw, 2)
                step_data["total_thermal_kw"] = round(thermal_kw, 2)
                step_data["total_generation_kw"] = round(solar_kw + wind_kw + thermal_kw, 2)

                # Voltage stats + per-bus voltages for topology coloring
                bus_names = dss.Circuit.AllBusNames()
                voltages = []
                bus_voltages: dict = {}
                for bname in bus_names:
                    dss.Circuit.SetActiveBus(bname)
                    vmag = dss.Bus.puVmagAngle()
                    if vmag:
                        v_pu = vmag[0::2][:dss.Bus.NumNodes()]
                        voltages.extend(v_pu)
                        # Store mean voltage per bus (skip intermediate LV buses)
                        valid_bus_v = [v for v in v_pu if 0.1 < v < 2.0]
                        if valid_bus_v:
                            bus_voltages[bname] = round(
                                sum(valid_bus_v) / len(valid_bus_v), 4
                            )

                valid_v = [v for v in voltages if 0.1 < v < 2.0]
                step_data["min_voltage_pu"] = round(min(valid_v), 4) if valid_v else 0.0
                step_data["max_voltage_pu"] = round(max(valid_v), 4) if valid_v else 0.0
                step_data["voltage_violations"] = sum(
                    1 for v in valid_v if v < 0.95 or v > 1.05
                )
                step_data["bus_voltages"] = bus_voltages

            steps.append(step_data)

        return steps

    def _summarize_day(self, date_str: str, steps: List[Dict]) -> Dict[str, Any]:
        """Summarize a day's simulation into a single result dict."""
        converged_steps = sum(1 for s in steps if s.get("converged"))
        conv_steps = [s for s in steps if s.get("converged")]

        result = {
            "date": date_str,
            "status": "success",
            "converged_steps": converged_steps,
            "total_steps": 96,
        }

        if conv_steps:
            result["min_voltage_pu"] = round(min(s.get("min_voltage_pu", 1.0) for s in conv_steps), 4)
            result["max_voltage_pu"] = round(max(s.get("max_voltage_pu", 1.0) for s in conv_steps), 4)
            result["total_violations"] = sum(s.get("voltage_violations", 0) for s in conv_steps)
            result["avg_power_kw"] = round(
                np.mean([s.get("total_power_kw", 0) for s in conv_steps]), 2
            )
            result["peak_power_kw"] = round(
                max(s.get("total_power_kw", 0) for s in conv_steps), 2
            )
            result["min_power_kw"] = round(
                min(s.get("total_power_kw", 0) for s in conv_steps), 2
            )

            # Per-feeder averages
            for feeder in ["F06", "F07", "F08", "F09", "F10", "F11", "F12"]:
                key = f"power_{feeder}_kw"
                vals = [s[key] for s in conv_steps if key in s]
                if vals:
                    result[f"avg_{key}"] = round(np.mean(vals), 2)
                    result[f"peak_{key}"] = round(max(vals), 2)

        return result

    async def start_simulation(
        self, start_date: str, end_date: Optional[str] = None
    ) -> str:
        """Start a simulation task (runs in background).

        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD). For single-day mode, this is the only date.
        end_date : str, optional
            End date (YYYY-MM-DD, inclusive). If provided, runs multi-day.

        Returns
        -------
        str
            Task ID for polling status.
        """
        if self.is_busy:
            raise RuntimeError("A simulation is already running")

        task_id = str(uuid.uuid4())[:8]
        is_range = end_date is not None and end_date != start_date

        task = SimulationTask(
            task_id=task_id,
            mode="range" if is_range else "single",
            start_date=start_date,
            end_date=end_date or start_date,
        )

        # Calculate total days
        d_start = datetime.strptime(start_date, "%Y-%m-%d")
        d_end = datetime.strptime(end_date or start_date, "%Y-%m-%d")
        task.total_days = (d_end - d_start).days + 1

        self._tasks[task_id] = task
        self._running_task_id = task_id

        # Launch background task
        asyncio.create_task(self._run_task(task))

        return task_id

    async def _run_task(self, task: SimulationTask):
        """Execute the simulation task in background."""
        try:
            task.status = "running"
            d_start = datetime.strptime(task.start_date, "%Y-%m-%d")
            d_end = datetime.strptime(task.end_date, "%Y-%m-%d")

            current = d_start
            day_num = 0

            while current <= d_end:
                # Check for cancellation
                if task.status == "cancelled":
                    logger.info(f"[{task.task_id}] Cancelled at day {day_num}")
                    break

                day_num += 1
                date_str = current.strftime("%Y-%m-%d")
                task.current_day = day_num
                task.current_date = date_str

                logger.info(f"[{task.task_id}] Day {day_num}/{task.total_days}: {date_str}")

                day_result = await self._run_one_day(date_str)
                task.completed_days.append(day_result)

                current += timedelta(days=1)

                # Yield to event loop between days
                await asyncio.sleep(0.1)

            task.status = "completed"
            logger.info(f"[{task.task_id}] Simulation complete: {task.total_days} days")

        except Exception as e:
            task.status = "error"
            task.error = str(e)
            logger.error(f"[{task.task_id}] Simulation error: {e}")
        finally:
            self._running_task_id = None

    async def _run_one_day(self, date_str: str) -> Dict[str, Any]:
        """Prepare and simulate one day. Runs CPU-bound work in thread pool."""
        loop = asyncio.get_event_loop()

        try:
            def _sync_work():
                self._prepare_date(date_str)
                steps = self._run_simulation()
                return self._summarize_day(date_str, steps)

            result = await loop.run_in_executor(None, _sync_work)
            return result

        except Exception as e:
            logger.error(f"Day {date_str} failed: {e}")
            return {
                "date": date_str,
                "status": "error",
                "error": str(e),
            }

    def _extract_grid_state(self) -> Dict[str, Any]:
        """Extract current DSS grid state in the format the frontend expects.

        Reads bus voltages, transformers, loads, generators, violations
        from the OpenDSS model WITHOUT re-solving (preserves daily-mode state).
        """
        state = opendss_service.read_current_state()

        return {
            "timestamp": state.timestamp,
            "converged": state.converged,
            "simulation_time": "24:00",
            "summary": {
                "total_power_kw": round(state.total_power_kw, 2),
                "total_power_kvar": round(state.total_power_kvar, 2),
                "total_losses_kw": round(state.total_losses_kw, 2),
                "total_generation_kw": round(state.total_generation_kw, 2),
                "total_solar_kw": round(state.total_solar_kw, 2),
                "total_load_kw": round(state.total_load_kw, 2),
                "num_voltage_violations": len(state.voltage_violations),
                "num_overloaded_elements": len(state.overloaded_elements),
            },
            "buses": {
                name: {
                    "name": bus.name,
                    "base_kv": bus.base_kv,
                    "voltage_pu": [round(v, 4) for v in bus.voltage_pu],
                    "voltage_angle": [round(a, 2) for a in bus.voltage_angle],
                }
                for name, bus in state.buses.items()
            },
            "lines": {
                name: {
                    "name": line.name,
                    "bus1": line.bus1,
                    "bus2": line.bus2,
                    "power_kw": round(line.power_kw, 2),
                    "current_amps": [round(c, 2) for c in line.current_amps],
                    "enabled": line.enabled,
                }
                for name, line in state.lines.items()
            },
            "transformers": {
                name: {
                    "name": xfmr.name,
                    "kva": xfmr.kva,
                    "loading_percent": round(xfmr.loading_percent, 2),
                    "power_kw": round(xfmr.power_kw, 2),
                }
                for name, xfmr in state.transformers.items()
            },
            "loads": {
                name: {
                    "name": load.name,
                    "bus": load.bus,
                    "kw": round(load.kw, 2),
                    "kvar": round(load.kvar, 2),
                    "voltage_pu": round(load.voltage_pu, 4),
                }
                for name, load in state.loads.items()
            },
            "generators": {
                name: {
                    "name": gen.name,
                    "bus": gen.bus,
                    "kw": round(gen.kw, 2),
                    "kvar": round(gen.kvar, 2),
                    "type": gen.type,
                }
                for name, gen in state.generators.items()
            },
            "violations": {
                "voltage": state.voltage_violations,
                "overloads": state.overloaded_elements,
            },
        }

    async def run_single_day_detailed(self, date_str: str) -> Dict[str, Any]:
        """Run a single day and return detailed per-step results (for charts).

        This is a convenience method for single-day mode that returns
        the summary, raw 96-step data, and the final grid state.
        """
        loop = asyncio.get_event_loop()

        def _sync_work():
            self._prepare_date(date_str)
            steps = self._run_simulation()
            summary = self._summarize_day(date_str, steps)
            grid_state = self._extract_grid_state()
            return {"summary": summary, "steps": steps, "grid_state": grid_state}

        return await loop.run_in_executor(None, _sync_work)

    def cancel_task(self, task_id: str) -> bool:
        """Mark a task as cancelled (it will stop at the next day boundary)."""
        task = self._tasks.get(task_id)
        if task and task.status == "running":
            task.status = "cancelled"
            self._running_task_id = None
            return True
        return False


# Singleton
pipeline_service = PipelineService()
