from fastapi import APIRouter, HTTPException
from app.api.schemas.fault_input import FaultReport
from app.api.schemas.isolation_output import IsolationResult
from app.services.opendss_engine import engine
from app.services.isolation_service import isolate_fault

router = APIRouter(tags=["Fault Isolation"])


@router.post("/isolate", response_model=IsolationResult,
             summary="Isolate a detected fault (rule-based)")
async def isolate_endpoint(report: FaultReport):
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded. POST /load-model first.")
    return isolate_fault(report)
