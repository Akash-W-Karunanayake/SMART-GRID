"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Any, Optional
from enum import Enum


# Base model with disabled protected namespace for 'model_' prefix
class BaseModelNoProtected(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ============== Enums ==============

class SimulationStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ComponentType(str, Enum):
    BUS = "bus"
    LINE = "line"
    TRANSFORMER = "transformer"
    LOAD = "load"
    GENERATOR = "generator"
    PVSYSTEM = "pvsystem"
    SWITCH = "switch"


# ============== Base Response ==============

class BaseResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


# ============== Grid/Circuit Schemas ==============

class CircuitInfo(BaseModel):
    name: str
    num_buses: int
    num_nodes: int
    num_elements: int
    base_frequency: float
    total_power: Dict[str, float]


class LoadModelResponse(BaseResponse):
    circuit_name: Optional[str] = None
    info: Optional[CircuitInfo] = None


class BusSchema(BaseModel):
    name: str
    base_kv: float
    voltage_pu: List[float]
    voltage_angle: List[float]


class LineSchema(BaseModel):
    name: str
    bus1: str
    bus2: str
    power_kw: float
    current_amps: List[float]
    current_angle: List[float]
    enabled: bool


class TransformerSchema(BaseModel):
    name: str
    kva: float
    loading_percent: float
    power_kw: float


class LoadSchema(BaseModel):
    name: str
    bus: str
    kw: float
    kvar: float
    voltage_pu: float


class GeneratorSchema(BaseModel):
    name: str
    bus: str
    kw: float
    kvar: float
    type: str


class ViolationsSchema(BaseModel):
    voltage: List[str] = []
    overloads: List[str] = []


class GridSummary(BaseModel):
    total_power_kw: float
    total_power_kvar: float
    total_losses_kw: float
    total_generation_kw: float
    total_load_kw: float
    num_voltage_violations: int
    num_overloaded_elements: int


class GridStateResponse(BaseModel):
    timestamp: float
    simulation_time: str
    converged: bool
    summary: GridSummary
    buses: Dict[str, BusSchema]
    lines: Dict[str, LineSchema]
    transformers: Dict[str, TransformerSchema]
    loads: Dict[str, LoadSchema]
    generators: Dict[str, GeneratorSchema]
    violations: ViolationsSchema


# ============== Topology Schemas ==============

class TopologyNode(BaseModel):
    id: str
    label: str
    type: str
    kv: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None


class TopologyEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str


class TopologyResponse(BaseModel):
    nodes: List[TopologyNode]
    edges: List[TopologyEdge]


# ============== Simulation Control Schemas ==============

class StartSimulationRequest(BaseModel):
    hours: int = Field(default=24, ge=1, le=168, description="Simulation duration in hours")
    speed: float = Field(default=1.0, ge=0.1, le=100.0, description="Speed multiplier")
    mode: str = Field(
        default="synthetic",
        description="Simulation mode: 'synthetic' (manual profiles) or 'real_data' (LoadShape-driven)"
    )
    target_date: Optional[str] = Field(
        default=None,
        description="Target date for real_data mode (YYYY-MM-DD). Default: 2025-08-01"
    )


class SimulationStatusResponse(BaseModelNoProtected):
    running: bool
    paused: bool
    current_hour: float
    speed: float
    subscribers: int
    history_length: int
    model_loaded: bool
    mode: str = "synthetic"


class SimulationHistoryItem(BaseModel):
    timestamp: float
    total_power_kw: float
    total_load_kw: float
    total_generation_kw: float
    total_losses_kw: float
    converged: bool
    num_violations: int


class SimulationHistoryResponse(BaseModel):
    history: List[SimulationHistoryItem]


# ============== Control Action Schemas ==============

class SetLoadMultiplierRequest(BaseModel):
    multiplier: float = Field(ge=0.0, le=2.0, description="Load multiplier (0-2)")


class SetGenerationMultiplierRequest(BaseModel):
    multiplier: float = Field(ge=0.0, le=1.5, description="Generation multiplier (0-1.5)")


# ============== Forecasting Schemas (for future ML integration) ==============

class ForecastRequest(BaseModel):
    horizon_hours: int = Field(default=24, ge=1, le=168)
    include_uncertainty: bool = Field(default=True)


class ForecastPoint(BaseModel):
    timestamp: float
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResponse(BaseModelNoProtected):
    forecast_type: str  # 'load', 'generation', 'net_load'
    horizon_hours: int
    points: List[ForecastPoint]
    model_info: Optional[Dict[str, Any]] = None


# ============== Solar ML Forecasting Schemas ==============

class SolarPredictionPoint(BaseModel):
    timestamp: str
    predicted_mw: float
    actual_mw: Optional[float] = None
    rf_pred: Optional[float] = None
    lstm_pred: Optional[float] = None


class SolarDayDataResponse(BaseModel):
    date: str
    next_date: str
    actual_mw: List[float]
    predicted_next_day_mw: List[float]


class SolarForecastResponse(BaseModelNoProtected):
    forecast_type: str = "solar_ml"
    target_date: str
    points: List[SolarPredictionPoint]
    installed_capacity_mw: float
    model_info: Optional[Dict[str, Any]] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


# ============== Self-Healing Schemas ==============

class FaultTypeEnum(str, Enum):
    THREE_PHASE = "3phase"
    LINE_TO_GROUND = "lg"
    LINE_TO_LINE = "ll"
    LINE_TO_LINE_TO_GROUND = "llg"


class FaultReport(BaseModel):
    """Input: fault detected by diagnostics or manually specified."""
    fault_type: FaultTypeEnum = Field(..., description="Type of fault")
    fault_location: str = Field(..., description="Bus name (e.g. 'F09_Node3')")
    fault_resistance_ohms: float = Field(default=0.01, ge=0)


class SwitchAction(BaseModel):
    switch_name: str
    action: str = Field(..., description="'open' or 'close'")
    reason: str = ""


class IsolationResult(BaseModel):
    success: bool
    fault_location: str
    fault_type: str
    feeder: str = ""
    isolated_zone_buses: List[str] = Field(default_factory=list)
    switch_actions: List[SwitchAction] = Field(default_factory=list)
    de_energized_loads: List[str] = Field(default_factory=list)
    num_de_energized_loads: int = 0
    critical_loads_affected: List[str] = Field(default_factory=list)
    message: str = ""


class RestorationRequest(BaseModel):
    strategy: str = Field(default="auto", description="'auto', 'marl', or 'heuristic'")


class RestorationResult(BaseModel):
    strategy: str
    closed_switches: List[str]
    energized_loads: List[str]
    de_energized_loads: List[str]
    restoration_pct: float
    converged: bool
    radial: bool
    voltage_violations: List[str] = Field(default_factory=list)
    overloaded_lines: List[str] = Field(default_factory=list)


class SwitchStateResponse(BaseModel):
    switches: Dict[str, bool]


class SwitchControlRequest(BaseModel):
    closed: bool = Field(..., description="True to close, False to open")


class FLISRRequest(BaseModel):
    """Input for full FLISR pipeline (fault -> isolate -> restore)."""
    fault_bus: str = Field(..., description="Bus name where the fault occurs")
    fault_type: FaultTypeEnum = Field(default=FaultTypeEnum.THREE_PHASE)
    fault_resistance: float = Field(default=0.01, ge=0)
    restoration_strategy: str = Field(
        default="auto",
        description="'auto', 'marl', or 'heuristic'",
    )


class FLISRResponse(BaseModel):
    """Combined result of the full FLISR pipeline."""
    fault: Dict[str, str]
    isolation: IsolationResult
    restoration: RestorationResult
    grid_state: Dict[str, Any]


class GridResetResponse(BaseModel):
    success: bool
    message: str


# ============== Fault Injection Schemas ==============

class FaultInjectionRequest(BaseModel):
    bus: str = Field(description="Target bus name")
    fault_type: str = Field(description="LG, LL, LLG, LLL, or HIF")
    phase: str = Field(description="Phase: A, B, C, AB, BC, CA, ABG, BCG, CAG, ABC")
    resistance: float = Field(default=1.0, ge=0.0001, le=1000.0,
                              description="Fault resistance in ohms")


class FaultPredictionResponse(BaseModel):
    is_fault: bool
    detection_confidence: float

    fault_type: str
    type_probabilities: Dict[str, float]

    fault_phase: str
    phase_probabilities: Dict[str, float]

    fault_location_bus: str
    location_probabilities: Dict[str, float]

    step_injected: Optional[int] = None
    step_detected: Optional[int] = None


class ActiveFaultResponse(BaseModel):
    bus: str
    fault_type: str
    phase: str
    resistance: float
    step_injected: int


class FaultStatusResponse(BaseModel):
    has_active_fault: bool
    active_fault: Optional[ActiveFaultResponse] = None
    latest_prediction: Optional[FaultPredictionResponse] = None
    detection_latency_steps: Optional[int] = None


# ============== Diagnostics Schemas (updated for real model integration) ==============

class DiagnosticResult(BaseModel):
    fault_detected: bool
    fault_type: Optional[str] = None
    fault_phase: Optional[str] = None
    fault_location: Optional[str] = None
    confidence: float
    timestamp: float
    type_probabilities: Optional[Dict[str, float]] = None
    phase_probabilities: Optional[Dict[str, float]] = None
    location_probabilities: Optional[Dict[str, float]] = None


class DiagnosticRequest(BaseModel):
    use_live_data: bool = Field(default=True)


# ============== WebSocket Message Schemas ==============

class WSMessage(BaseModel):
    type: str  # 'state_update', 'control', 'error', 'info'
    data: Dict[str, Any]
    timestamp: Optional[float] = None


class WSControlMessage(BaseModel):
    action: str  # 'start', 'stop', 'pause', 'resume', 'step', 'set_speed'
    params: Optional[Dict[str, Any]] = None
