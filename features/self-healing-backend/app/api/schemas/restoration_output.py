"""Schemas for restoration planning results."""
from pydantic import BaseModel, Field
from typing import List, Optional


class RestorationStep(BaseModel):
    step: int
    switch_name: str
    action: str = Field(..., description="'open' or 'close'")
    reason: str = ""


class RestorationResult(BaseModel):
    success: bool
    strategy: str = Field("marl", description="'marl' | 'heuristic' | 'manual'")
    restoration_steps: List[RestorationStep] = Field(default_factory=list)
    loads_restored: List[str] = Field(default_factory=list)
    loads_still_offline: List[str] = Field(default_factory=list)
    critical_loads_restored: List[str] = Field(default_factory=list)
    pct_load_restored: float = Field(0.0, description="0-100")
    voltage_violations: List[str] = Field(default_factory=list)
    overloaded_lines: List[str] = Field(default_factory=list)
    is_radial: bool = True
    num_switching_ops: int = 0
    reward: Optional[float] = None
    message: str = ""
