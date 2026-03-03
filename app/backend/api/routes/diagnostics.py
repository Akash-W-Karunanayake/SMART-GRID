"""
Diagnostics API routes — real fault injection, detection, and history.

Replaces all mock endpoints with live model-backed fault diagnostics.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from models.schemas import (
    FaultInjectionRequest,
    FaultPredictionResponse,
    FaultStatusResponse,
    ActiveFaultResponse,
)
from services.simulation_service import simulation_service
from services.fault_injection_service import fault_injection_service
from services.fault_detection_service import fault_detection_service

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.post("/inject-fault")
async def inject_fault(request: FaultInjectionRequest) -> Dict[str, Any]:
    """
    Queue a fault for injection at the next simulation step.
    Only available when the simulation is running (Q6).
    One fault at a time (Q5). Queued for next solve step (Q4).
    """
    if not simulation_service.is_running:
        raise HTTPException(400, "Simulation must be running to inject faults.")

    if simulation_service.is_paused:
        raise HTTPException(400, "Cannot inject fault while simulation is paused.")

    result = fault_injection_service.queue_fault(
        bus=request.bus,
        fault_type=request.fault_type,
        phase=request.phase,
        resistance=request.resistance,
        current_step=simulation_service._current_step,
    )

    if not result["success"]:
        raise HTTPException(409, result["error"])

    return result


@router.post("/clear-fault")
async def clear_fault() -> Dict[str, Any]:
    """Clear the currently active fault (Q7: persists until manually cleared)."""
    result = fault_injection_service.clear_fault()
    if not result["success"]:
        raise HTTPException(404, result["error"])

    # Clear the latest prediction since fault is gone
    simulation_service._latest_prediction = None

    return result


@router.get("/fault-status", response_model=FaultStatusResponse)
async def get_fault_status():
    """Get current fault status including active fault and latest prediction."""
    fi = fault_injection_service
    pred = simulation_service._latest_prediction

    active = None
    if fi.has_active_fault and fi.active_fault:
        af = fi.active_fault
        active = ActiveFaultResponse(
            bus=af.bus,
            fault_type=af.fault_type,
            phase=af.phase,
            resistance=af.resistance,
            step_injected=af.step_injected,
        )

    prediction_resp = None
    latency = None
    if pred is not None:
        prediction_resp = FaultPredictionResponse(
            is_fault=pred.is_fault,
            detection_confidence=pred.detection_confidence,
            fault_type=pred.fault_type,
            type_probabilities=pred.type_probabilities,
            fault_phase=pred.fault_phase,
            phase_probabilities=pred.phase_probabilities,
            fault_location_bus=pred.fault_location_bus,
            location_probabilities=pred.location_probabilities,
            step_injected=pred.step_injected,
            step_detected=pred.step_detected,
        )
        if pred.step_injected is not None and pred.step_detected is not None:
            latency = pred.step_detected - pred.step_injected

    return FaultStatusResponse(
        has_active_fault=fi.has_active_fault,
        active_fault=active,
        latest_prediction=prediction_resp,
        detection_latency_steps=latency,
    )


@router.get("/fault-history")
async def get_fault_history() -> List[Dict[str, Any]]:
    """Get the last 20 fault events (Q15)."""
    return fault_injection_service.get_fault_history()


@router.get("/model-status")
async def get_model_status() -> Dict[str, Any]:
    """Check if the fault detection model is loaded."""
    return {
        "loaded": fault_detection_service.is_loaded,
        "n_buses": fault_detection_service._n_buses if fault_detection_service.is_loaded else None,
        "n_features_cnn": fault_detection_service._in_features_cnn if fault_detection_service.is_loaded else None,
    }


@router.post("/load-model")
async def load_model() -> Dict[str, Any]:
    """Manually load the fault detection model."""
    success = fault_detection_service.load()
    if not success:
        raise HTTPException(500, "Failed to load fault detection model.")
    return {"success": True, "message": "Fault detection model loaded."}


@router.get("/fault-waveform")
async def get_fault_waveform() -> Dict[str, Any]:
    """Return the latest sub-cycle capture data for waveform visualization."""
    capture = fault_injection_service.last_capture
    if capture is None:
        raise HTTPException(404, "No sub-cycle capture data available.")

    return {
        "voltage_seq": capture.voltage_seq.tolist(),
        "current_seq": capture.current_seq.tolist(),
        "bus_names": capture.bus_names,
        "branch_names": capture.branch_names,
        "total_cycles": len(capture.voltage_seq),
    }
