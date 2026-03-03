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
from services.opendss_service import opendss_service

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.post("/inject-fault")
async def inject_fault(request: FaultInjectionRequest) -> Dict[str, Any]:
    """
    Inject a fault into the OpenDSS circuit and run model inference.

    Two modes:
      - Live simulation running: queue fault for next solve step (Q4/Q6).
      - No live simulation but DSS model loaded (e.g. after pipeline run):
        apply fault immediately + run sub-cycle capture + inference.

    One fault at a time (Q5). Fault persists until manually cleared (Q7).
    """
    if not opendss_service.model_loaded:
        raise HTTPException(400, "OpenDSS model not loaded. Run a simulation first.")

    if fault_injection_service.has_active_fault:
        raise HTTPException(409, "A fault is already active. Clear it first.")

    if simulation_service.is_running:
        # Live simulation: queue for next solve step
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
    else:
        # No live simulation: inject immediately + run inference
        result = fault_injection_service.queue_fault(
            bus=request.bus,
            fault_type=request.fault_type,
            phase=request.phase,
            resistance=request.resistance,
            current_step=0,
        )
        if not result["success"]:
            raise HTTPException(409, result["error"])

        # Apply the queued fault immediately
        fault_injection_service.apply_queued_fault(0)

        # Run sub-cycle capture + model inference
        prediction = simulation_service._run_fault_inference(0)
        simulation_service._latest_prediction = prediction

        return {
            "success": True,
            "message": f"Fault applied immediately: {request.fault_type} at {request.bus}",
            "prediction_available": prediction is not None,
        }


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
    """Check if the fault detection model and DSS model are loaded."""
    return {
        "loaded": fault_detection_service.is_loaded,
        "dss_model_loaded": opendss_service.model_loaded,
        "can_inject": opendss_service.model_loaded and not fault_injection_service.has_active_fault,
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
