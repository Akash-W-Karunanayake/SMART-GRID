"""Schemas for fault isolation results."""
from pydantic import BaseModel, Field
from typing import List


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
