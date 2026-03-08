"""
GridRestorationEnv -- Gymnasium Environment
============================================
RL environment wrapping the Chunnakam OpenDSS model.

CRITICAL DESIGN:
  - Isolation is done BEFORE the episode starts (rule-based, not RL)
  - The agent's job is RESTORATION only
  - Action space = 5 tie switches (open/close)
  - Observation = post-isolation grid state (no future leakage)

Flat observation per agent:
  - tie switch states (5 binary)                              -> 5
  - all switch states (21 binary, read-only context)          -> 21
  - load served flags (22 binary)                             -> 22
  - agent's own switch index one-hot (5)                      -> 5
  Total flat: 53

Graph data (for GNN, returned in info dict):
  - bus_voltages: Dict[str, float]   (node features)
  - adjacency_edges: List[Tuple]     (edge list)

Action per agent: Discrete(2) -- 0=open, 1=close
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from config import (
    TIE_SWITCH_NAMES, NUM_TIE_SWITCHES, ALL_SWITCH_NAMES,
    ALL_LOAD_NAMES, CRITICAL_LOADS,
)
from app.services.opendss_engine import OpenDSSEngine
from app.services.grid_graph import build_grid_graph, is_radial, get_adjacency_list
from app.ml.reward import compute_reward


class GridRestorationEnv(gym.Env):
    """
    Multi-agent restoration environment.
    Each of the 5 tie-switch agents chooses open(0) or close(1).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        master_dss_path: Optional[str] = None,
        max_steps: int = 10,
    ):
        super().__init__()

        self.engine = OpenDSSEngine()
        self._master_path = master_dss_path
        self.max_steps = max_steps
        self.current_step = 0

        # Fault scenario (set per episode via options)
        self.fault_bus: Optional[str] = None
        self.fault_type: str = "lg"

        # Agents = tie switches only
        self.n_agents = NUM_TIE_SWITCHES  # 5
        self.agent_ids = TIE_SWITCH_NAMES

        self.action_space = spaces.Dict({
            name: spaces.Discrete(2) for name in TIE_SWITCH_NAMES
        })

        self._obs_size: Optional[int] = None
        self.observation_space = None

        # Tracking
        self._pre_fault_load_kw = 0.0
        self._initial_de_energized: List[str] = []
        self._de_energized_kw = 0.0
        self._prev_tie_states: Dict[str, bool] = {}

        # Baselines recorded after isolation (before agent acts)
        self._baseline_violations: int = 0
        self._baseline_overloads: int = 0
        self._prev_restored_count: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.current_step = 0

        self.engine.load_model(self._master_path)
        self.engine.solve()
        self._pre_fault_load_kw = self.engine.get_total_load_kw()

        if options:
            self.fault_bus = options.get("fault_bus", self.fault_bus)
            self.fault_type = options.get("fault_type", self.fault_type)

        # Run isolation (rule-based) -- this sets the starting state
        if self.fault_bus:
            self._run_isolation()

        self._initial_de_energized = self.engine.get_de_energized_loads()
        self._prev_tie_states = self.engine.get_tie_switch_states()

        # Record post-isolation baselines for delta-based reward
        self._baseline_violations = len(self.engine.get_voltage_violations())
        self._baseline_overloads = len(self.engine.get_overloaded_lines())
        self._prev_restored_count = 0

        # Calculate kW lost due to isolation (case-insensitive lookup)
        load_powers = self.engine.get_load_powers()
        load_powers_lower = {k.lower(): v for k, v in load_powers.items()}
        self._de_energized_kw = sum(
            load_powers_lower.get(ld.lower(), 0.0) for ld in self._initial_de_energized
        )

        obs = self._build_flat_obs()
        if self._obs_size is None:
            sample = list(obs.values())[0]
            self._obs_size = len(sample)
            self.observation_space = spaces.Dict({
                name: spaces.Box(-np.inf, np.inf, shape=(self._obs_size,), dtype=np.float32)
                for name in TIE_SWITCH_NAMES
            })

        info = {
            "fault_bus": self.fault_bus,
            "graph_data": self._build_graph_data(),
            "de_energized": self._initial_de_energized,
            "de_energized_kw": self._de_energized_kw,
            "baseline_violations": self._baseline_violations,
            "baseline_overloads": self._baseline_overloads,
        }

        return obs, info

    def step(
        self, actions: Dict[str, int],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        self.current_step += 1

        # Count how many ties actually changed
        num_toggled = 0
        for sw_name, act in actions.items():
            new_closed = bool(act)
            old_closed = self._prev_tie_states.get(sw_name, False)
            if new_closed != old_closed:
                num_toggled += 1
            self.engine.set_switch(sw_name, closed=new_closed)

        converged = self.engine.solve()

        reward = compute_reward(
            engine=self.engine,
            de_energized_kw=self._de_energized_kw,
            initial_de_energized=self._initial_de_energized,
            converged=converged,
            num_switches_toggled=num_toggled,
            baseline_violations=self._baseline_violations,
            baseline_overloads=self._baseline_overloads,
        )

        self._prev_tie_states = self.engine.get_tie_switch_states()

        # Early termination checks
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        if not converged:
            terminated = True

        # Check if all de-energized loads are now restored (success)
        energized = set(self.engine.get_energized_loads())
        current_restored = len([
            ld for ld in self._initial_de_energized if ld in energized
        ])

        if (len(self._initial_de_energized) > 0 and
                current_restored == len(self._initial_de_energized)):
            terminated = True  # Full restoration achieved

        # Check for stable state (no change from previous step)
        if (num_toggled == 0 and
                current_restored == self._prev_restored_count and
                self.current_step > 1):
            terminated = True  # No progress, stable state

        self._prev_restored_count = current_restored

        obs = self._build_flat_obs()

        info = {
            "converged": converged,
            "de_energized": self.engine.get_de_energized_loads(),
            "violations": self.engine.get_voltage_violations(),
            "overloads": self.engine.get_overloaded_lines(),
            "step": self.current_step,
            "num_toggled": num_toggled,
            "restored_count": current_restored,
            "total_de_energized": len(self._initial_de_energized),
            "graph_data": self._build_graph_data(),
        }

        return obs, reward, terminated, truncated, info

    def _run_isolation(self):
        from app.services.isolation_service import isolate_fault
        from app.api.schemas.fault_input import FaultReport

        report = FaultReport(
            fault_type=self.fault_type,
            fault_location=self.fault_bus,
        )
        result = isolate_fault(report)
        if not result.success:
            from loguru import logger
            logger.warning(f"Isolation failed for {self.fault_bus}: proceeding anyway")
        self.engine.solve()

    def _build_flat_obs(self) -> Dict[str, np.ndarray]:
        """
        Build per-agent flat observations (non-topological features).
        IMPORTANT: Only uses CURRENT state -- no future/post-restoration info.
        Bus voltages are NOT included here; they are handled by the GNN.
        """
        all_sw = self.engine.get_all_switch_states()
        tie_sw = self.engine.get_tie_switch_states()
        energized = set(self.engine.get_energized_loads())

        # Feature vectors
        all_sw_vec = np.array(
            [float(all_sw.get(n, False)) for n in ALL_SWITCH_NAMES], dtype=np.float32
        )
        tie_sw_vec = np.array(
            [float(tie_sw.get(n, False)) for n in TIE_SWITCH_NAMES], dtype=np.float32
        )
        load_vec = np.array(
            [1.0 if ld in energized else 0.0 for ld in ALL_LOAD_NAMES], dtype=np.float32
        )

        obs = {}
        for i, sw_name in enumerate(TIE_SWITCH_NAMES):
            # One-hot for which agent this is
            agent_id = np.zeros(NUM_TIE_SWITCHES, dtype=np.float32)
            agent_id[i] = 1.0
            obs[sw_name] = np.concatenate([
                tie_sw_vec, all_sw_vec, load_vec, agent_id,
            ])

        return obs

    def _build_graph_data(self) -> dict:
        """
        Build raw graph data for GNN processing.
        Returns dict with bus_voltages and adjacency_edges.
        """
        return {
            "bus_voltages": self.engine.get_bus_voltages_pu(),
            "adjacency_edges": get_adjacency_list(),
        }

    def render(self):
        pass

    def close(self):
        pass
