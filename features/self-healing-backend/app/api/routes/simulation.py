from fastapi import APIRouter, HTTPException, Path, Query
from app.services.opendss_engine import engine
from config import ALL_SWITCH_NAMES

router = APIRouter(tags=["Simulation & Grid"])


@router.post("/load-model", summary="Load or reload the OpenDSS model")
async def load_model():
    ok = engine.load_model()
    if not ok:
        raise HTTPException(500, "Failed to load OpenDSS model")
    return {"success": True, "message": "Model loaded"}


@router.get("/grid-state", summary="Full grid state snapshot")
async def grid_state():
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return engine.get_grid_state()


@router.post("/solve", summary="Run a single power-flow")
async def solve():
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return {"converged": engine.solve()}


@router.post("/switch/{name}", summary="Toggle a switch")
async def toggle_switch(
    name: str = Path(...),
    action: str = Query(..., enum=["open", "close"]),
):
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded")
    if name not in ALL_SWITCH_NAMES:
        raise HTTPException(404, f"Unknown switch: {name}")

    engine.set_switch(name, closed=(action == "close"))
    engine.solve()
    return {
        "switch": name, "action": action,
        "converged": engine.solve(),
        "de_energized_loads": engine.get_de_energized_loads(),
    }


@router.get("/switches", summary="All switch states")
async def get_switches():
    if not engine.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return engine.get_all_switch_states()
