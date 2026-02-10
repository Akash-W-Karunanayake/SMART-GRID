"""
Pipeline API routes - Run data-driven OpenDSS simulations for single days or date ranges.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from services.pipeline_service import pipeline_service

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


# ============== Request / Response schemas ==============

class SimulateRequest(BaseModel):
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(
        default=None,
        description="End date (YYYY-MM-DD, inclusive). Omit for single-day mode.",
    )


class SimulateResponse(BaseModel):
    task_id: str
    mode: str
    total_days: int
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    mode: str
    total_days: int
    current_day: int
    current_date: str
    completed_count: int
    error: Optional[str] = None


class SingleDayRequest(BaseModel):
    date: str = Field(description="Target date (YYYY-MM-DD)")


# ============== Endpoints ==============

@router.post("/simulate", response_model=SimulateResponse)
async def start_simulation(request: SimulateRequest):
    """
    Start a simulation for a single date or date range.

    - Single day: only provide start_date
    - Date range: provide both start_date and end_date

    Returns a task_id for polling progress via GET /pipeline/status/{task_id}.
    """
    if pipeline_service.is_busy:
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running. Wait for it to finish or cancel it.",
        )

    try:
        task_id = await pipeline_service.start_simulation(
            start_date=request.start_date,
            end_date=request.end_date,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    task = pipeline_service.get_task(task_id)
    return SimulateResponse(
        task_id=task_id,
        mode=task.mode,
        total_days=task.total_days,
        message=f"Simulation started: {task.total_days} day(s)",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll simulation progress."""
    task = pipeline_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        mode=task.mode,
        total_days=task.total_days,
        current_day=task.current_day,
        current_date=task.current_date,
        completed_count=len(task.completed_days),
        error=task.error,
    )


@router.get("/results/{task_id}")
async def get_task_results(task_id: str) -> Dict[str, Any]:
    """
    Get simulation results.

    Returns completed_days: list of per-day summary dicts.
    """
    task = pipeline_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.task_id,
        "status": task.status,
        "mode": task.mode,
        "start_date": task.start_date,
        "end_date": task.end_date,
        "total_days": task.total_days,
        "completed_days": task.completed_days,
    }


@router.post("/simulate-day")
async def simulate_single_day(request: SingleDayRequest) -> Dict[str, Any]:
    """
    Run a single-day simulation synchronously and return detailed results.

    Returns both a summary and raw 96-step data for charts.
    This is the endpoint to use when you want immediate detailed results
    for a single day (the frontend's "single day" mode).
    """
    if pipeline_service.is_busy:
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running.",
        )

    try:
        result = await pipeline_service.run_single_day_detailed(request.date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """Cancel a running simulation."""
    success = pipeline_service.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail="Task not found or not running")
    return {"success": True, "message": "Task cancelled"}
