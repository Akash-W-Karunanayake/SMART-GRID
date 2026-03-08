"""
Tests for grid config and design invariants.
Run: pytest tests/ -v
"""
from config.grid_config import (
    SWITCH_BY_NAME, FEEDER_LOADS, CRITICAL_LOADS, TIE_SWITCHES,
    CIRCUIT_BREAKERS, SECTIONALIZERS, TIE_SWITCH_NAMES,
    NUM_TIE_SWITCHES, SwitchRole,
)


def test_total_switch_count():
    assert len(SWITCH_BY_NAME) == 21


def test_marl_action_space_is_5_tie_switches():
    """Agents control ONLY the 5 tie switches, not all 21."""
    assert NUM_TIE_SWITCHES == 5
    assert len(TIE_SWITCH_NAMES) == 5
    for sw in TIE_SWITCHES:
        assert sw.role == SwitchRole.TIE_SWITCH


def test_isolation_switches_are_not_ties():
    for cb in CIRCUIT_BREAKERS:
        assert cb.role == SwitchRole.CIRCUIT_BREAKER
        assert cb.name not in TIE_SWITCH_NAMES
    for sec in SECTIONALIZERS:
        assert sec.role == SwitchRole.SECTIONALIZER
        assert sec.name not in TIE_SWITCH_NAMES


def test_tie_switches_normally_open():
    for sw in TIE_SWITCHES:
        assert sw.normal_closed is False


def test_cbs_normally_closed():
    for cb in CIRCUIT_BREAKERS:
        assert cb.normal_closed is True


def test_bus_names_are_lowercase():
    """All bus names in config must be lowercase."""
    for sw in CIRCUIT_BREAKERS + SECTIONALIZERS + TIE_SWITCHES:
        assert sw.bus_from == sw.bus_from.lower(), f"{sw.name}: bus_from not lowercase"
        assert sw.bus_to == sw.bus_to.lower(), f"{sw.name}: bus_to not lowercase"


def test_hospital_is_critical():
    assert "F11_Hospital" in CRITICAL_LOADS
    assert "F11_Hospital" in FEEDER_LOADS["F11"]


def test_all_feeders_have_cb_and_sec():
    for f in ["F06", "F07", "F08", "F09", "F10", "F11", "F12"]:
        assert f"CB_{f}" in SWITCH_BY_NAME
        assert f"SEC_{f}" in SWITCH_BY_NAME
