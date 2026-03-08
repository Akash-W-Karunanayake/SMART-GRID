"""
Schemas for fault reports received from the Fault Diagnostics component.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class FaultTypeEnum(str, Enum):
    THREE_PHASE = "3phase"
    LINE_TO_GROUND = "lg"
    LINE_TO_LINE = "ll"
    LINE_TO_LINE_TO_GROUND = "llg"


class FaultReport(BaseModel):
    """
    Input contract with the fault-diagnostics teammate.

    Example:
    {
        "fault_type": "lg",
        "fault_location": "F09_Node3",
        "faulted_element": "F09_Sec3",
        "fault_resistance_ohms": 0.01,
        "timestamp_step": 42
    }
    """
    fault_type: FaultTypeEnum = Field(..., description="Type of fault detected")
    fault_location: str = Field(
        ..., description="Bus name where fault was detected (e.g. 'F09_Node3')"
    )
    faulted_element: Optional[str] = Field(
        None, description="Line or element name if known"
    )
    fault_resistance_ohms: float = Field(0.01, ge=0)
    timestamp_step: Optional[int] = Field(
        None, description="Simulation timestep (0-95 for daily 15-min mode)"
    )
