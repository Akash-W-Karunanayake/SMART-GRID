"""
Simulation API routes - Handles simulation control (start, stop, pause, etc.).
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from services import simulation_service
from models.schemas import (
    BaseResponse,
    StartSimulationRequest,
    SimulationStatusResponse,
    SimulationHistoryResponse
)

router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.post("/start", response_model=BaseResponse)
async def start_simulation(request: StartSimulationRequest):
    """
    Start real-time simulation.

    - **hours**: Simulation duration (1-168 hours)
    - **speed**: Speed multiplier (0.1-100x real-time)
    """
    result = await simulation_service.start(
        hours=request.hours, speed=request.speed, mode=request.mode
    )

    if not result["success"]:
        return BaseResponse(success=False, error=result.get("error"))

    return BaseResponse(success=True, message=result.get("message"))


@router.post("/stop", response_model=BaseResponse)
async def stop_simulation():
    """Stop the running simulation."""
    result = await simulation_service.stop()
    return BaseResponse(success=result["success"], message=result.get("message"))


@router.post("/pause", response_model=BaseResponse)
async def pause_simulation():
    """Pause the running simulation."""
    result = await simulation_service.pause()

    if not result["success"]:
        return BaseResponse(success=False, error=result.get("error"))

    return BaseResponse(success=True, message=result.get("message"))


@router.post("/resume", response_model=BaseResponse)
async def resume_simulation():
    """Resume a paused simulation."""
    result = await simulation_service.resume()

    if not result["success"]:
        return BaseResponse(success=False, error=result.get("error"))

    return BaseResponse(success=True, message=result.get("message"))


@router.post("/step")
async def step_simulation() -> Dict[str, Any]:
    """
    Execute a single simulation step.
    Useful for manual/controlled simulation.
    """
    result = await simulation_service.step()

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.post("/speed")
async def set_speed(speed: float) -> Dict[str, Any]:
    """
    Set simulation speed multiplier.

    - **speed**: New speed multiplier (0.1-100x)
    """
    result = simulation_service.set_speed(speed)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error"))

    return result


@router.get("/status", response_model=SimulationStatusResponse)
async def get_status():
    """Get current simulation status."""
    return simulation_service.get_status()


@router.get("/history")
async def get_history(limit: int = 100) -> Dict[str, Any]:
    """
    Get simulation history.

    - **limit**: Maximum number of history items to return (default: 100)
    """
    history = simulation_service.get_history(limit=limit)
    return {"history": history}


@router.get("/current-state")
async def get_current_state() -> Dict[str, Any]:
    """Get the current simulation state (if simulation is running)."""
    state = simulation_service.current_state

    if state is None:
        return {"state": None, "message": "No simulation state available"}

    return {
        "state": simulation_service._state_to_dict(state)
    }
