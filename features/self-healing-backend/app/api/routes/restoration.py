from fastapi import APIRouter, HTTPException, Query
from app.api.schemas.fault_input import FaultReport
from app.api.schemas.restoration_output import RestorationResult
from app.services.opendss_engine import engine
from app.services.isolation_service import isolate_fault
from app.services.restoration_service import restore_service

router = APIRouter(tags=["Power Restoration"])


@router.post("/restore", response_model=RestorationResult,
             summary="Full FLISR: isolate + restore")
async def restore_endpoint(
    report: FaultReport,
    strategy: str = Query("auto", enum=["auto", "marl", "heuristic"]),
):
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded.")

    engine.reset()
    if report.timestamp_step is not None:
        engine.solve_at_step(report.timestamp_step)

    isolation = isolate_fault(report)
    if not isolation.success:
        return RestorationResult(
            success=False, strategy=strategy,
            message=f"Isolation failed: {isolation.message}",
        )

    return restore_service(isolation, strategy=strategy)
