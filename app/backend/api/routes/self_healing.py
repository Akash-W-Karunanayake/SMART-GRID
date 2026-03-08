"""
Self-Healing API routes -- fault isolation, service restoration, switch control.

Endpoints:
  POST /self-healing/isolate       - Run fault isolation on a specified bus
  POST /self-healing/restore       - Run service restoration (heuristic/MARL/auto)
  GET  /self-healing/switches      - Get all switch states
  PUT  /self-healing/switch/{name} - Manual switch control
  GET  /self-healing/grid-state    - Current grid state for self-healing dashboard
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from models.schemas import (
    FaultReport, FaultTypeEnum, IsolationResult,
    RestorationRequest, RestorationResult,
    SwitchStateResponse, SwitchControlRequest,
    FLISRRequest, FLISRResponse, GridResetResponse,
)
from services.opendss_service import opendss_service
from services.opendss_engine import engine
from services.isolation_service import isolate_fault
from services.restoration_service import restore
from config import get_dss_master_path

router = APIRouter(prefix="/self-healing", tags=["Self-Healing"])


def _ensure_engine_loaded():
    """Ensure the self-healing engine has a loaded model (piggyback on opendss_service)."""
    if not engine.is_loaded:
        if not opendss_service.is_loaded:
            raise HTTPException(400, "OpenDSS model not loaded. Load the model first.")
        # opendss_service already compiled the DSS model into the shared COM instance
        engine._loaded = True


@router.post("/isolate", response_model=IsolationResult)
async def fault_isolation(report: FaultReport):
    """
    Isolate a faulted zone using circuit breakers and sectionalizers.

    This is a rule-based procedure:
    1. Open the feeder CB to de-energize the whole feeder.
    2. Open the sectionalizer to split into upstream/downstream zones.
    3. Re-close the CB if the fault is in the downstream zone.

    The tie switches are NOT touched here -- they are the MARL agent's domain
    during the restoration phase.
    """
    _ensure_engine_loaded()

    try:
        result = isolate_fault(report)
    except Exception as e:
        raise HTTPException(500, f"Isolation failed: {e}")

    if not result.success:
        raise HTTPException(422, result.message)

    return result


@router.post("/restore", response_model=RestorationResult)
async def service_restoration(request: RestorationRequest = RestorationRequest()):
    """
    Restore de-energized loads after fault isolation.

    Strategies:
    - 'auto': Try MARL first, fall back to heuristic on failure.
    - 'marl': Use trained MARL agent (GNN + DQN).
    - 'heuristic': Greedy tie-switch closure.
    """
    _ensure_engine_loaded()

    try:
        result = restore(strategy=request.strategy)
    except Exception as e:
        raise HTTPException(500, f"Restoration failed: {e}")

    return RestorationResult(
        strategy=result["strategy"],
        closed_switches=result["closed_switches"],
        energized_loads=result.get("energized_loads", []),
        de_energized_loads=result.get("de_energized_loads", []),
        restoration_pct=result.get("restoration_pct", 0.0),
        converged=result.get("converged", False),
        radial=result.get("radial", True),
        voltage_violations=result.get("voltage_violations", []),
        overloaded_lines=result.get("overloaded_lines", []),
    )


@router.get("/switches", response_model=SwitchStateResponse)
async def get_switches():
    """Get the current state of all 21 switches (CBs, SECs, Ties)."""
    _ensure_engine_loaded()
    states = engine.get_all_switch_states()
    return SwitchStateResponse(switches=states)


@router.put("/switch/{name}")
async def control_switch(name: str, request: SwitchControlRequest) -> Dict[str, Any]:
    """Manually open or close a specific switch."""
    _ensure_engine_loaded()

    from config.grid_config import ALL_SWITCH_NAMES
    if name not in ALL_SWITCH_NAMES:
        raise HTTPException(404, f"Unknown switch: '{name}'")

    engine.set_switch(name, closed=request.closed)
    converged = engine.solve()

    return {
        "switch": name,
        "closed": request.closed,
        "converged": converged,
        "message": f"Switch {name} {'closed' if request.closed else 'opened'}.",
    }


@router.get("/grid-state")
async def get_grid_state() -> Dict[str, Any]:
    """Get the full grid state for the self-healing dashboard."""
    _ensure_engine_loaded()
    return engine.get_grid_state()


@router.post("/flisr", response_model=FLISRResponse)
async def run_flisr(request: FLISRRequest):
    """
    Full FLISR pipeline: re-compile model -> apply fault -> isolate -> restore.

    This re-compiles the DSS model from scratch so the grid starts clean.
    The fault is applied via OpenDSS fault object, then isolation and
    restoration run sequentially.
    """
    import opendssdirect as dss

    # Step 0: Re-compile model for a clean slate
    master_path = get_dss_master_path()
    dss.Basic.DataPath(str(master_path.parent))
    dss.run_command(f"Compile [{master_path}]")
    err = dss.Error.Description()
    if err:
        raise HTTPException(500, f"Failed to compile DSS model: {err}")
    engine._loaded = True

    # Step 1: Apply fault via OpenDSS fault object
    fault_bus = request.fault_bus.lower()

    # Map FaultTypeEnum to OpenDSS fault phases
    fault_phase_map = {
        FaultTypeEnum.THREE_PHASE: 3,
        FaultTypeEnum.LINE_TO_GROUND: 1,
        FaultTypeEnum.LINE_TO_LINE: 2,
        FaultTypeEnum.LINE_TO_LINE_TO_GROUND: 2,
    }
    phases = fault_phase_map.get(request.fault_type, 3)

    dss.run_command(
        f"New Fault.selfhealing_fault Bus1={fault_bus} phases={phases} "
        f"r={request.fault_resistance}"
    )
    engine.solve()

    # Determine feeder from bus name
    feeder = ""
    for prefix in ["f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12"]:
        if fault_bus.startswith(prefix):
            feeder = prefix.upper()
            break

    # Step 2: Isolate
    report = FaultReport(
        fault_type=request.fault_type,
        fault_location=request.fault_bus,
        fault_resistance_ohms=request.fault_resistance,
    )
    isolation_result = isolate_fault(report)
    if not isolation_result.success:
        raise HTTPException(422, isolation_result.message)

    # Step 3: Restore
    restore_result = restore(strategy=request.restoration_strategy)
    restoration_result = RestorationResult(
        strategy=restore_result["strategy"],
        closed_switches=restore_result["closed_switches"],
        energized_loads=restore_result.get("energized_loads", []),
        de_energized_loads=restore_result.get("de_energized_loads", []),
        restoration_pct=restore_result.get("restoration_pct", 0.0),
        converged=restore_result.get("converged", False),
        radial=restore_result.get("radial", True),
        voltage_violations=restore_result.get("voltage_violations", []),
        overloaded_lines=restore_result.get("overloaded_lines", []),
    )

    # Step 4: Get final grid state
    grid_state = engine.get_grid_state()

    return FLISRResponse(
        fault={
            "bus": fault_bus,
            "fault_type": request.fault_type.value,
            "feeder": feeder,
        },
        isolation=isolation_result,
        restoration=restoration_result,
        grid_state=grid_state,
    )


@router.post("/reset", response_model=GridResetResponse)
async def reset_grid():
    """Re-compile the OpenDSS model to reset all switches to default state."""
    import opendssdirect as dss

    master_path = get_dss_master_path()
    dss.Basic.DataPath(str(master_path.parent))
    dss.run_command(f"Compile [{master_path}]")
    err = dss.Error.Description()
    if err:
        raise HTTPException(500, f"Failed to reset grid: {err}")

    engine._loaded = True
    engine.solve()

    return GridResetResponse(success=True, message="Grid reset to normal state")
