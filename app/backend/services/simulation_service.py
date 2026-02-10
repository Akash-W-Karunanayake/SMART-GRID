"""
Simulation Service - Manages real-time simulation and state broadcasting.
Handles continuous simulation loops and WebSocket state updates.
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import asdict
from datetime import datetime
import logging

from .opendss_service import OpenDSSService, opendss_service, GridState

logger = logging.getLogger(__name__)


class SimulationService:
    """
    Service for managing real-time power system simulation.
    Handles simulation lifecycle, state broadcasting, and client subscriptions.
    """

    def __init__(self, dss_service: OpenDSSService = None):
        self._dss = dss_service or opendss_service
        self._running = False
        self._paused = False
        self._current_state: Optional[GridState] = None
        self._simulation_speed: float = 1.0  # 1x real-time
        self._subscribers: Set[Callable] = set()
        self._simulation_task: Optional[asyncio.Task] = None
        self._history: List[Dict] = []
        self._max_history_length = 1000
        self._current_hour: float = 0.0
        self._step_minutes: int = 15  # 15-minute intervals
        self._mode: str = "synthetic"  # "synthetic" or "real_data"

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused

    @property
    def current_state(self) -> Optional[GridState]:
        """Get current grid state."""
        return self._current_state

    def subscribe(self, callback: Callable):
        """Subscribe to state updates."""
        self._subscribers.add(callback)
        logger.info(f"New subscriber added. Total: {len(self._subscribers)}")

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from state updates."""
        self._subscribers.discard(callback)
        logger.info(f"Subscriber removed. Total: {len(self._subscribers)}")

    async def _broadcast_state(self, state: GridState):
        """Broadcast state to all subscribers."""
        state_dict = self._state_to_dict(state)

        for callback in list(self._subscribers):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state_dict)
                else:
                    callback(state_dict)
            except Exception as e:
                logger.error(f"Error broadcasting to subscriber: {e}")
                self._subscribers.discard(callback)

    def _state_to_dict(self, state: GridState) -> Dict[str, Any]:
        """Convert GridState to JSON-serializable dict."""
        return {
            "timestamp": state.timestamp,
            "simulation_time": f"{int(state.timestamp):02d}:{int((state.timestamp % 1) * 60):02d}",
            "converged": state.converged,
            "summary": {
                "total_power_kw": round(state.total_power_kw, 2),
                "total_power_kvar": round(state.total_power_kvar, 2),
                "total_losses_kw": round(state.total_losses_kw, 2),
                "total_generation_kw": round(state.total_generation_kw, 2),
                "total_solar_kw": round(state.total_solar_kw, 2),
                "total_load_kw": round(state.total_load_kw, 2),
                "num_voltage_violations": len(state.voltage_violations),
                "num_overloaded_elements": len(state.overloaded_elements)
            },
            "buses": {
                name: {
                    "name": bus.name,
                    "base_kv": bus.base_kv,
                    "voltage_pu": [round(v, 4) for v in bus.voltage_pu],
                    "voltage_angle": [round(a, 2) for a in bus.voltage_angle]
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
                    "enabled": line.enabled
                }
                for name, line in state.lines.items()
            },
            "transformers": {
                name: {
                    "name": xfmr.name,
                    "kva": xfmr.kva,
                    "loading_percent": round(xfmr.loading_percent, 2),
                    "power_kw": round(xfmr.power_kw, 2)
                }
                for name, xfmr in state.transformers.items()
            },
            "loads": {
                name: {
                    "name": load.name,
                    "bus": load.bus,
                    "kw": round(load.kw, 2),
                    "kvar": round(load.kvar, 2),
                    "voltage_pu": round(load.voltage_pu, 4)
                }
                for name, load in state.loads.items()
            },
            "generators": {
                name: {
                    "name": gen.name,
                    "bus": gen.bus,
                    "kw": round(gen.kw, 2),
                    "kvar": round(gen.kvar, 2),
                    "type": gen.type
                }
                for name, gen in state.generators.items()
            },
            "violations": {
                "voltage": state.voltage_violations,
                "overloads": state.overloaded_elements
            }
        }

    async def start(self, hours: int = 24, speed: float = 1.0,
                    mode: str = "synthetic") -> Dict[str, Any]:
        """
        Start real-time simulation.

        Args:
            hours: Total hours to simulate
            speed: Simulation speed multiplier (1.0 = real-time)
            mode: "synthetic" (manual profiles) or "real_data" (LoadShape-driven)

        Returns:
            Status dict
        """
        if not self._dss.is_loaded:
            result = self._dss.load_model()
            if not result["success"]:
                return {"success": False, "error": "Failed to load model"}

        if self._running:
            return {"success": False, "error": "Simulation already running"}

        self._running = True
        self._paused = False
        self._simulation_speed = speed
        self._current_hour = 0.0
        self._mode = mode
        self._history.clear()

        # Start simulation loop in background
        if mode == "real_data":
            self._simulation_task = asyncio.create_task(
                self._real_data_simulation_loop(hours)
            )
        else:
            self._simulation_task = asyncio.create_task(
                self._simulation_loop(hours)
            )

        logger.info(f"Simulation started: {hours} hours at {speed}x speed, mode={mode}")
        return {
            "success": True,
            "message": f"Simulation started for {hours} hours (mode={mode})",
            "speed": speed
        }

    async def _simulation_loop(self, total_hours: int):
        """Main simulation loop."""
        try:
            step_interval = (self._step_minutes * 60) / self._simulation_speed

            while self._running and self._current_hour < total_hours:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Update load multiplier based on time of day (simple daily profile)
                hour_of_day = self._current_hour % 24
                load_mult = self._get_load_profile(hour_of_day)
                self._dss.set_load_multiplier(load_mult)

                # Update solar generation based on time of day
                solar_mult = self._get_solar_profile(hour_of_day)
                self._dss.set_generation_multiplier(solar_mult)

                # Get current state
                self._current_state = self._dss.get_grid_state()
                self._current_state.timestamp = self._current_hour

                # Log warnings for non-convergence or violations
                if not self._current_state.converged:
                    logger.warning(
                        f"Hour {self._current_hour:.2f}: Power flow did not converge "
                        f"(load={load_mult:.3f}, solar={solar_mult:.3f})"
                    )
                if self._current_state.voltage_violations:
                    logger.warning(
                        f"Hour {self._current_hour:.2f}: {len(self._current_state.voltage_violations)} "
                        f"voltage violation(s) - worst: {self._current_state.voltage_violations[0]}"
                    )

                # Store in history
                self._add_to_history(self._current_state)

                # Broadcast to subscribers
                await self._broadcast_state(self._current_state)

                # Wait for next step
                await asyncio.sleep(step_interval / 60)  # Convert to simulation time

                # Advance time
                self._current_hour += self._step_minutes / 60

        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self._running = False
            self._paused = False
            self._simulation_task = None

    async def _real_data_simulation_loop(self, total_hours: int):
        """Simulation loop for real_data mode.

        LoadShapes in the DSS model drive load/generation variation.
        No manual multiplier setting needed - OpenDSS daily mode steps
        through the LoadShape automatically at each Solve().
        """
        try:
            import opendssdirect as dss

            # Configure OpenDSS for daily mode
            dss.Text.Command("Set Mode=Daily")
            dss.Text.Command("Set Stepsize=15m")
            dss.Text.Command("Set Number=1")
            dss.Text.Command("Set controlmode=static")

            total_steps = int(total_hours * 4)  # 4 steps per hour at 15-min
            step_interval = (self._step_minutes * 60) / self._simulation_speed

            for step in range(total_steps):
                if not self._running:
                    break

                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Solve one step - LoadShapes handle everything
                dss.Solution.Solve()
                converged = dss.Solution.Converged()

                # Collect grid state directly (avoid get_grid_state which resets mode to snapshot)
                from .opendss_service import GridState
                self._current_state = GridState(
                    timestamp=self._current_hour,
                    converged=converged,
                    total_power_kw=-dss.Circuit.TotalPower()[0],
                    total_power_kvar=-dss.Circuit.TotalPower()[1],
                    total_losses_kw=dss.Circuit.Losses()[0] / 1000,
                )
                self._current_state.buses = self._dss._get_all_buses()
                self._current_state.lines = self._dss._get_all_lines()
                self._current_state.transformers = self._dss._get_all_transformers()
                self._current_state.loads = self._dss._get_all_loads()
                self._current_state.generators = self._dss._get_all_generators()
                self._current_state.total_load_kw = sum(l.kw for l in self._current_state.loads.values())
                self._current_state.total_generation_kw = sum(g.kw for g in self._current_state.generators.values())
                self._current_state.total_solar_kw = sum(
                    g.kw for g in self._current_state.generators.values() if g.type == "pvsystem"
                )
                self._current_state.voltage_violations = self._dss._check_voltage_violations(self._current_state.buses)
                self._current_state.overloaded_elements = self._dss._check_overloads()

                if not converged:
                    logger.warning(
                        f"Hour {self._current_hour:.2f}: Power flow did not converge (real_data mode)"
                    )
                if self._current_state.voltage_violations:
                    logger.warning(
                        f"Hour {self._current_hour:.2f}: {len(self._current_state.voltage_violations)} "
                        f"voltage violation(s)"
                    )

                self._add_to_history(self._current_state)
                await self._broadcast_state(self._current_state)

                await asyncio.sleep(step_interval / 60)
                self._current_hour += self._step_minutes / 60

        except asyncio.CancelledError:
            logger.info("Real data simulation loop cancelled")
        except Exception as e:
            logger.error(f"Real data simulation error: {e}")
        finally:
            self._running = False
            self._paused = False
            self._simulation_task = None

    def _get_load_profile(self, hour: float) -> float:
        """
        Get load multiplier for given hour of day.
        Realistic daily load profile with natural variation.
        """
        import math
        import random

        # Base load profile
        # Peak hours: 6 PM - 10 PM
        # Off-peak: 11 PM - 5 AM
        if 18 <= hour < 22:
            base = 1.0  # Peak
        elif 6 <= hour < 18:
            base = 0.7 + 0.2 * (hour - 6) / 12  # Gradual increase
        elif 22 <= hour or hour < 6:
            base = 0.5  # Off-peak
        else:
            base = 0.7

        # Add small random variation (±3%) to make the simulation more dynamic
        # This simulates natural load fluctuations
        variation = 1.0 + random.uniform(-0.03, 0.03)

        return base * variation

    def _get_solar_profile(self, hour: float) -> float:
        """
        Get solar generation multiplier for given hour.
        Realistic solar irradiance profile with capacity factor limiting.

        Note: Peak solar is capped at 10% to account for:
        - The OpenDSS model has PV systems (47 MW total) connected through
          distribution transformers (only ~10 MVA total capacity)
        - This mismatch means only a small fraction of nameplate PV can be generated
          before overwhelming the distribution network
        - In reality, rooftop solar would be distributed across many more
          transformers; this cap simulates that physical limitation
        - Also accounts for cloud cover, panel derating, etc.
        """
        import math
        import random

        # Maximum solar capacity factor - reduced to improve convergence
        # The model struggles with higher PV penetration
        MAX_SOLAR_FACTOR = 0.10

        if 6 <= hour <= 18:
            # Bell curve centered at noon, capped at realistic maximum
            raw_solar = max(0, math.sin((hour - 6) * math.pi / 12))
            base_solar = min(raw_solar, MAX_SOLAR_FACTOR)

            # Add small cloud-like variation (±5%) for more dynamic behavior
            if base_solar > 0:
                variation = 1.0 + random.uniform(-0.05, 0.05)
                return max(0, base_solar * variation)
            return 0.0
        return 0.0

    def _add_to_history(self, state: GridState):
        """Add state to history with size limit."""
        summary = {
            "timestamp": state.timestamp,
            "total_power_kw": state.total_power_kw,
            "total_load_kw": state.total_load_kw,
            "total_generation_kw": state.total_generation_kw,
            "total_losses_kw": state.total_losses_kw,
            "converged": state.converged,
            "num_violations": len(state.voltage_violations)
        }
        self._history.append(summary)

        if len(self._history) > self._max_history_length:
            self._history.pop(0)

    async def stop(self) -> Dict[str, Any]:
        """Stop simulation and fully reset state for clean restart."""
        self._running = False
        self._paused = False
        if self._simulation_task:
            self._simulation_task.cancel()
            try:
                await self._simulation_task
            except asyncio.CancelledError:
                pass
            self._simulation_task = None

        logger.info("Simulation stopped")
        return {"success": True, "message": "Simulation stopped"}

    async def pause(self) -> Dict[str, Any]:
        """Pause simulation."""
        if not self._running:
            return {"success": False, "error": "Simulation not running"}

        self._paused = True
        logger.info("Simulation paused")
        return {"success": True, "message": "Simulation paused"}

    async def resume(self) -> Dict[str, Any]:
        """Resume simulation."""
        if not self._running:
            return {"success": False, "error": "Simulation not running"}

        self._paused = False
        logger.info("Simulation resumed")
        return {"success": True, "message": "Simulation resumed"}

    def set_speed(self, speed: float) -> Dict[str, Any]:
        """Set simulation speed multiplier."""
        if speed <= 0:
            return {"success": False, "error": "Speed must be positive"}

        self._simulation_speed = speed
        logger.info(f"Simulation speed set to {speed}x")
        return {"success": True, "speed": speed}

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get simulation history."""
        return self._history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            "running": self._running,
            "paused": self._paused,
            "current_hour": round(self._current_hour, 2),
            "speed": self._simulation_speed,
            "subscribers": len(self._subscribers),
            "history_length": len(self._history),
            "model_loaded": self._dss.is_loaded,
            "mode": self._mode,
        }

    async def step(self) -> Dict[str, Any]:
        """Execute single simulation step (for manual control)."""
        if not self._dss.is_loaded:
            result = self._dss.load_model()
            if not result["success"]:
                return {"success": False, "error": "Failed to load model"}

        # Apply load and generation profiles based on current time
        hour_of_day = self._current_hour % 24
        load_mult = self._get_load_profile(hour_of_day)
        self._dss.set_load_multiplier(load_mult)

        solar_mult = self._get_solar_profile(hour_of_day)
        self._dss.set_generation_multiplier(solar_mult)

        # Get current state
        self._current_state = self._dss.get_grid_state()
        self._current_state.timestamp = self._current_hour

        # Store in history
        self._add_to_history(self._current_state)

        # Broadcast
        await self._broadcast_state(self._current_state)

        # Advance time
        self._current_hour += self._step_minutes / 60

        return {
            "success": True,
            "state": self._state_to_dict(self._current_state)
        }


# Singleton instance
simulation_service = SimulationService()
