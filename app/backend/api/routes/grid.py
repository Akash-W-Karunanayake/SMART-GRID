"""
Grid API routes - Handles grid model loading, state, and topology.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from services import opendss_service
from models.schemas import (
    BaseResponse,
    LoadModelResponse,
    GridStateResponse,
    TopologyResponse,
    SetLoadMultiplierRequest,
    SetGenerationMultiplierRequest,
    InjectFaultRequest,
    FaultResponse
)

router = APIRouter(prefix="/grid", tags=["Grid"])


@router.post("/load", response_model=LoadModelResponse)
async def load_model():
    """
    Load the OpenDSS power system model.
    Must be called before any other grid operations.
    """
    result = opendss_service.load_model()

    if not result["success"]:
        return LoadModelResponse(
            success=False,
            error=result.get("error", "Unknown error loading model")
        )

    return LoadModelResponse(
        success=True,
        message="Model loaded successfully",
        circuit_name=result["circuit_name"],
        info=result["info"]
    )


@router.get("/state")
async def get_grid_state() -> Dict[str, Any]:
    """
    Get current grid state including all buses, lines, transformers, loads, and generators.
    Performs a power flow solution before returning state.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        state = opendss_service.get_grid_state()

        return {
            "timestamp": state.timestamp,
            "converged": state.converged,
            "summary": {
                "total_power_kw": round(state.total_power_kw, 2),
                "total_power_kvar": round(state.total_power_kvar, 2),
                "total_losses_kw": round(state.total_losses_kw, 2),
                "total_generation_kw": round(state.total_generation_kw, 2),
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current-state")
async def get_current_grid_state() -> Dict[str, Any]:
    """
    Get current grid state WITHOUT re-solving.
    Use after a pipeline simulation to read the state from the last solve step.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Run a simulation first.")

    try:
        state = opendss_service.read_current_state()

        return {
            "timestamp": state.timestamp,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topology", response_model=TopologyResponse)
async def get_topology():
    """
    Get network topology for visualization.
    Returns nodes (buses) and edges (lines, transformers).
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        topology = opendss_service.get_topology()
        return TopologyResponse(**topology)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voltage-profile")
async def get_voltage_profile():
    """
    Get voltage profile for all buses.
    Returns DataFrame-like structure with bus names and voltage magnitudes.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        df = opendss_service.get_voltage_profile()
        return {
            "buses": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-multiplier", response_model=BaseResponse)
async def set_load_multiplier(request: SetLoadMultiplierRequest):
    """
    Set global load multiplier for all loads.
    Use this to simulate different loading conditions.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        opendss_service.set_load_multiplier(request.multiplier)
        return BaseResponse(
            success=True,
            message=f"Load multiplier set to {request.multiplier}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generation-multiplier", response_model=BaseResponse)
async def set_generation_multiplier(request: SetGenerationMultiplierRequest):
    """
    Set generation multiplier for PV systems.
    Simulates different solar irradiance conditions.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        opendss_service.set_generation_multiplier(request.multiplier)
        return BaseResponse(
            success=True,
            message=f"Generation multiplier set to {request.multiplier}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inject-fault", response_model=FaultResponse)
async def inject_fault(request: InjectFaultRequest):
    """
    Inject a fault at specified bus for testing fault detection/self-healing.
    """
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        result = opendss_service.inject_fault(
            bus=request.bus,
            fault_type=request.fault_type.value,
            resistance=request.resistance
        )

        if not result["success"]:
            return FaultResponse(success=False, error=result.get("error"))

        return FaultResponse(
            success=True,
            message="Fault injected successfully",
            bus=result["bus"],
            fault_type=result["fault_type"],
            fault_current_amps=result["fault_current_amps"],
            resistance=result["resistance"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_circuit_info():
    """Get basic circuit information."""
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    try:
        return opendss_service._get_circuit_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/buses")
async def get_all_buses():
    """Get list of all bus names."""
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    import opendssdirect as dss
    return {"buses": dss.Circuit.AllBusNames()}


@router.get("/elements")
async def get_all_elements():
    """Get list of all circuit elements."""
    if not opendss_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /grid/load first.")

    import opendssdirect as dss
    return {"elements": dss.Circuit.AllElementNames()}
