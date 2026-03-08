"""
Chunnakam Grid Topology Configuration
======================================
Single source of truth for all network element names.
All bus/element names must match the OpenDSS .dss files exactly.

KEY DESIGN DECISION:
  - Isolation uses CBs + Sectionalizers  (rule-based, NOT agents)
  - Restoration uses Tie Switches ONLY   (MARL agents)
  - Therefore MARL action space = 5 tie switches, NOT 21
"""
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


# Voltage limits (Sri Lankan CEB standards)
V_MIN_PU = 0.94
V_MAX_PU = 1.06

# Line thermal limit (% loading above which = overloaded)
LINE_OVERLOAD_THRESHOLD_PCT = 100.0


class SwitchRole(str, Enum):
    CIRCUIT_BREAKER = "cb"
    SECTIONALIZER = "sec"
    TIE_SWITCH = "tie"


@dataclass(frozen=True)
class SwitchInfo:
    name: str
    role: SwitchRole
    bus_from: str       # lowercase
    bus_to: str         # lowercase
    feeder: str
    normal_closed: bool


# All 21 switches
CIRCUIT_BREAKERS: List[SwitchInfo] = [
    SwitchInfo("CB_F05", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_a", "f05_node1", "F05", True),
    SwitchInfo("CB_F06", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_a", "f06_node1", "F06", True),
    SwitchInfo("CB_F07", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_a", "f07_node1", "F07", True),
    SwitchInfo("CB_F08", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_a", "f08_node1", "F08", True),
    SwitchInfo("CB_F09", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_b", "f09_node1", "F09", True),
    SwitchInfo("CB_F10", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_b", "f10_node1", "F10", True),
    SwitchInfo("CB_F11", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_b", "f11_node1", "F11", True),
    SwitchInfo("CB_F12", SwitchRole.CIRCUIT_BREAKER, "bus_33kv_b", "f12_node1", "F12", True),
]

SECTIONALIZERS: List[SwitchInfo] = [
    SwitchInfo("SEC_F05", SwitchRole.SECTIONALIZER, "f05_node2",  "f05_ujps",  "F05", True),
    SwitchInfo("SEC_F06", SwitchRole.SECTIONALIZER, "f06_node2",  "f06_node3", "F06", True),
    SwitchInfo("SEC_F07", SwitchRole.SECTIONALIZER, "f07_node2",  "f07_node3", "F07", True),
    SwitchInfo("SEC_F08", SwitchRole.SECTIONALIZER, "f08_node2",  "f08_node3", "F08", True),
    SwitchInfo("SEC_F09", SwitchRole.SECTIONALIZER, "f09_node3",  "f09_node4", "F09", True),
    SwitchInfo("SEC_F10", SwitchRole.SECTIONALIZER, "f10_node2",  "f10_node3", "F10", True),
    SwitchInfo("SEC_F11", SwitchRole.SECTIONALIZER, "f11_node2",  "f11_node3", "F11", True),
    SwitchInfo("SEC_F12", SwitchRole.SECTIONALIZER, "f12_node2",  "f12_node3", "F12", True),
]

TIE_SWITCHES: List[SwitchInfo] = [
    SwitchInfo("Tie_F06_F07", SwitchRole.TIE_SWITCH, "f06_node4", "f07_node4", "cross", False),
    SwitchInfo("Tie_F07_F08", SwitchRole.TIE_SWITCH, "f07_node4", "f08_node4", "cross", False),
    SwitchInfo("Tie_F09_F10", SwitchRole.TIE_SWITCH, "f09_node5", "f10_node4", "cross", False),
    SwitchInfo("Tie_F10_F11", SwitchRole.TIE_SWITCH, "f10_node4", "f11_node5", "cross", False),
    SwitchInfo("BusCoupler",  SwitchRole.TIE_SWITCH, "bus_33kv_a", "bus_33kv_b", "cross", False),
]

# Derived lookups
ALL_SWITCHES: List[SwitchInfo] = CIRCUIT_BREAKERS + SECTIONALIZERS + TIE_SWITCHES
ALL_SWITCH_NAMES: List[str] = [s.name for s in ALL_SWITCHES]
SWITCH_BY_NAME: Dict[str, SwitchInfo] = {s.name: s for s in ALL_SWITCHES}

TIE_SWITCH_NAMES: List[str] = [s.name for s in TIE_SWITCHES]
NUM_TIE_SWITCHES = len(TIE_SWITCHES)

# Feeder -> load mapping
FEEDER_LOADS: Dict[str, List[str]] = {
    "F06": ["F06_Industrial_Factory", "F06_SmallIndustry", "F06_Residential"],
    "F07": ["F07_Village_Center", "F07_Agricultural", "F07_Rural_Houses"],
    "F08": ["F08_Commercial", "F08_Residential", "F08_Mixed"],
    "F09": ["F09_Town_Center", "F09_Village", "F09_Rural_Agri", "F09_Remote"],
    "F10": ["F10_Town", "F10_Fishing", "F10_Coastal_Res"],
    "F11": ["F11_Hospital", "F11_Commercial", "F11_Apartments", "F11_MixedRes"],
    "F12": ["F12_Residential1", "F12_Residential2"],
}

ALL_LOAD_NAMES: List[str] = [ld for loads in FEEDER_LOADS.values() for ld in loads]
CRITICAL_LOADS: List[str] = ["F11_Hospital"]

SUPPORTED_FAULT_TYPES = ["3phase", "lg", "ll", "llg"]
