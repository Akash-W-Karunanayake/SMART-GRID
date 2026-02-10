"""
============================================================================
CHUNNAKAM GRID - SWITCH & GRID BEHAVIOR VERIFICATION SCRIPT
============================================================================
Purpose: Test and validate the switch infrastructure and grid behavior
         for Self-Healing (FLISR) research using MARL+GNN

Student ID: IT22053350
Date: January 2026

Tests Performed:
1. Normal state power flow validation
2. Circuit breaker operation tests
3. Sectionalizer operation tests
4. Tie switch operation tests
5. Fault isolation and load restoration scenarios
6. Voltage profile analysis
7. Stochastic simulation for realistic behavior

============================================================================
"""

import opendssdirect as dss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# Configuration
MASTER_FILE = Path(__file__).parent / "Master.dss"
RESULTS_DIR = Path(__file__).parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Voltage limits (per unit) - CEB Sri Lanka standards
V_MIN = 0.95  # -5%
V_MAX = 1.05  # +5%
V_CRITICAL_LOW = 0.90   # -10% (critical undervoltage)
V_CRITICAL_HIGH = 1.10  # +10% (critical overvoltage)

# MARL Action Space Definition (21 switches)
SWITCH_CONFIG = {
    # Circuit Breakers (Index 0-7) - Normal state: CLOSED (1)
    'CB_F05': {'index': 0, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F05'},
    'CB_F06': {'index': 1, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F06'},
    'CB_F07': {'index': 2, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F07'},
    'CB_F08': {'index': 3, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F08'},
    'CB_F09': {'index': 4, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F09'},
    'CB_F10': {'index': 5, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F10'},
    'CB_F11': {'index': 6, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F11'},
    'CB_F12': {'index': 7, 'type': 'CB', 'normal': 1, 'element': 'Line.CB_F12'},

    # Sectionalizers (Index 8-15) - Normal state: CLOSED (1)
    'SEC_F05': {'index': 8, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F05'},
    'SEC_F06': {'index': 9, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F06'},
    'SEC_F07': {'index': 10, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F07'},
    'SEC_F08': {'index': 11, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F08'},
    'SEC_F09': {'index': 12, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F09'},
    'SEC_F10': {'index': 13, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F10'},
    'SEC_F11': {'index': 14, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F11'},
    'SEC_F12': {'index': 15, 'type': 'SEC', 'normal': 1, 'element': 'Line.SEC_F12'},

    # Tie Switches (Index 16-20) - Normal state: OPEN (0)
    'Tie_F06_F07': {'index': 16, 'type': 'TIE', 'normal': 0, 'element': 'Line.Tie_F06_F07'},
    'Tie_F07_F08': {'index': 17, 'type': 'TIE', 'normal': 0, 'element': 'Line.Tie_F07_F08'},
    'Tie_F09_F10': {'index': 18, 'type': 'TIE', 'normal': 0, 'element': 'Line.Tie_F09_F10'},
    'Tie_F10_F11': {'index': 19, 'type': 'TIE', 'normal': 0, 'element': 'Line.Tie_F10_F11'},
    'BusCoupler': {'index': 20, 'type': 'TIE', 'normal': 0, 'element': 'Line.BusCoupler'},
}


class GridTester:
    """Test harness for Chunnakam Grid switch and power flow verification."""

    def __init__(self):
        self.results = {}
        self.voltage_violations = []

    def compile_circuit(self):
        """Compile the OpenDSS circuit."""
        dss.Text.Command(f'Compile "{MASTER_FILE}"')
        if not dss.Circuit.Name():
            raise RuntimeError("Failed to compile circuit")
        print(f"Circuit compiled: {dss.Circuit.Name()}")
        return True

    def solve(self):
        """Solve power flow and return convergence status."""
        dss.Solution.Solve()
        return dss.Solution.Converged()

    def get_switch_state(self, switch_name: str) -> int:
        """Get current state of a switch (1=closed, 0=open)."""
        config = SWITCH_CONFIG.get(switch_name)
        if not config:
            return -1

        dss.Circuit.SetActiveElement(config['element'])
        return 1 if dss.CktElement.IsOpen(1, 0) == 0 else 0

    def set_switch_state(self, switch_name: str, state: int):
        """Set switch state (1=close, 0=open)."""
        config = SWITCH_CONFIG.get(switch_name)
        if not config:
            return False

        element = config['element']
        if state == 1:
            dss.Text.Command(f'Edit {element} enabled=yes')
        else:
            dss.Text.Command(f'Edit {element} enabled=no')
        return True

    def get_all_switch_states(self) -> list:
        """Get state vector for all 21 switches (MARL action space)."""
        states = [0] * 21
        for name, config in SWITCH_CONFIG.items():
            states[config['index']] = self.get_switch_state(name)
        return states

    def set_normal_state(self):
        """Reset all switches to normal operating state."""
        for name, config in SWITCH_CONFIG.items():
            self.set_switch_state(name, config['normal'])
        return self.get_all_switch_states()

    def get_bus_voltages(self) -> dict:
        """Get voltage magnitude at all buses."""
        voltages = {}
        bus_names = dss.Circuit.AllBusNames()

        for bus in bus_names:
            dss.Circuit.SetActiveBus(bus)
            v_pu = dss.Bus.puVmagAngle()[::2]  # Get magnitudes only
            if v_pu:
                voltages[bus] = {
                    'v_pu': np.mean(v_pu),
                    'v_min': min(v_pu),
                    'v_max': max(v_pu),
                    'phases': len(v_pu)
                }
        return voltages

    def analyze_voltage_violations(self, voltages: dict) -> dict:
        """Analyze voltage violations."""
        violations = {
            'undervoltage': [],
            'overvoltage': [],
            'critical_low': [],
            'critical_high': [],
            'total_count': 0,
            'summary': {}
        }

        for bus, data in voltages.items():
            v = data['v_pu']
            if v < V_CRITICAL_LOW:
                violations['critical_low'].append((bus, v))
                violations['total_count'] += 1
            elif v < V_MIN:
                violations['undervoltage'].append((bus, v))
                violations['total_count'] += 1
            elif v > V_CRITICAL_HIGH:
                violations['critical_high'].append((bus, v))
                violations['total_count'] += 1
            elif v > V_MAX:
                violations['overvoltage'].append((bus, v))
                violations['total_count'] += 1

        violations['summary'] = {
            'total': violations['total_count'],
            'undervoltage': len(violations['undervoltage']),
            'overvoltage': len(violations['overvoltage']),
            'critical_low': len(violations['critical_low']),
            'critical_high': len(violations['critical_high']),
        }

        return violations

    def get_power_flow(self) -> dict:
        """Get power flow through all elements."""
        results = {
            'total_load_kw': 0,
            'total_generation_kw': 0,
            'total_losses_kw': 0,
            'feeder_power': {},
            'switch_power': {}
        }

        # Get total losses
        losses = dss.Circuit.Losses()
        results['total_losses_kw'] = losses[0] / 1000  # Convert W to kW

        # Get power at each circuit breaker (feeder head)
        for name, config in SWITCH_CONFIG.items():
            if config['type'] == 'CB':
                dss.Circuit.SetActiveElement(config['element'])
                powers = dss.CktElement.Powers()
                if powers:
                    # Sum of phases (P values are at even indices)
                    p_kw = sum(powers[::2][:3])  # First 3 phases
                    results['feeder_power'][name] = p_kw

        return results

    def get_total_load(self) -> float:
        """Get total load in kW."""
        total = 0.0
        load_names = dss.Loads.AllNames()
        if not load_names:
            return 0.0

        for load_name in load_names:
            dss.Loads.Name(load_name)
            total += dss.Loads.kW()
        return total

    def get_total_generation(self) -> float:
        """Get total generation in kW."""
        total = 0.0

        # Generators
        gen_names = dss.Generators.AllNames()
        if gen_names:
            for gen_name in gen_names:
                dss.Generators.Name(gen_name)
                total += dss.Generators.kW()

        # PVSystems
        pv_names = dss.PVsystems.AllNames()
        if pv_names:
            for pv_name in pv_names:
                dss.PVsystems.Name(pv_name)
                total += dss.PVsystems.kW()

        return total


def test_1_normal_state(tester: GridTester) -> dict:
    """Test 1: Normal State Power Flow Validation"""
    print("\n" + "="*70)
    print("TEST 1: NORMAL STATE POWER FLOW VALIDATION")
    print("="*70)

    tester.compile_circuit()
    tester.set_normal_state()
    converged = tester.solve()

    results = {
        'converged': converged,
        'switch_states': tester.get_all_switch_states(),
        'voltages': tester.get_bus_voltages(),
        'power_flow': tester.get_power_flow(),
        'total_load_kw': tester.get_total_load(),
        'total_generation_kw': tester.get_total_generation(),
    }

    violations = tester.analyze_voltage_violations(results['voltages'])
    results['violations'] = violations

    print(f"\nConvergence: {'YES' if converged else 'NO'}")
    print(f"Switch State Vector: {results['switch_states']}")
    print(f"  Expected Normal:   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]")
    print(f"\nTotal Load: {results['total_load_kw']:.1f} kW")
    print(f"Total Generation: {results['total_generation_kw']:.1f} kW")
    print(f"Total Losses: {results['power_flow']['total_losses_kw']:.1f} kW")

    print(f"\nVoltage Violations Summary:")
    print(f"  Total: {violations['summary']['total']}")
    print(f"  Critical Low (<{V_CRITICAL_LOW}): {violations['summary']['critical_low']}")
    print(f"  Undervoltage (<{V_MIN}): {violations['summary']['undervoltage']}")
    print(f"  Overvoltage (>{V_MAX}): {violations['summary']['overvoltage']}")
    print(f"  Critical High (>{V_CRITICAL_HIGH}): {violations['summary']['critical_high']}")

    if violations['critical_low']:
        print(f"\n  Critical Low Buses (sample):")
        for bus, v in violations['critical_low'][:5]:
            print(f"    {bus}: {v:.4f} pu")

    if violations['critical_high']:
        print(f"\n  Critical High Buses (sample):")
        for bus, v in violations['critical_high'][:5]:
            print(f"    {bus}: {v:.4f} pu")

    return results


def test_2_circuit_breaker_operation(tester: GridTester) -> dict:
    """Test 2: Circuit Breaker Operation"""
    print("\n" + "="*70)
    print("TEST 2: CIRCUIT BREAKER OPERATION")
    print("="*70)

    results = {'tests': []}

    # Test each CB
    for cb_name in ['CB_F05', 'CB_F06', 'CB_F07', 'CB_F08', 'CB_F09', 'CB_F10', 'CB_F11', 'CB_F12']:
        tester.compile_circuit()
        tester.set_normal_state()

        # Open the CB
        tester.set_switch_state(cb_name, 0)
        converged = tester.solve()

        voltages = tester.get_bus_voltages()
        violations = tester.analyze_voltage_violations(voltages)

        # Find de-energized buses (voltage near 0)
        deenergized = [bus for bus, data in voltages.items() if data['v_pu'] < 0.1]

        test_result = {
            'cb': cb_name,
            'converged': converged,
            'deenergized_buses': len(deenergized),
            'violations': violations['summary']['total'],
            'sample_deenergized': deenergized[:5]
        }
        results['tests'].append(test_result)

        print(f"\n{cb_name} OPEN:")
        print(f"  Converged: {'YES' if converged else 'NO'}")
        print(f"  De-energized buses: {len(deenergized)}")
        if deenergized:
            print(f"  Sample: {deenergized[:5]}")

    return results


def test_3_tie_switch_restoration(tester: GridTester) -> dict:
    """Test 3: Tie Switch Load Restoration"""
    print("\n" + "="*70)
    print("TEST 3: TIE SWITCH LOAD RESTORATION (FLISR)")
    print("="*70)

    results = {'scenarios': []}

    # Scenario: F07 fault - isolate with CB, restore via tie switch
    scenarios = [
        {
            'name': 'F07 Fault - Restore via Tie_F06_F07',
            'fault_cb': 'CB_F07',
            'restore_tie': 'Tie_F06_F07',
            'expected_restoration': ['F07_Node3', 'F07_Node4']
        },
        {
            'name': 'F09 Fault - Restore via Tie_F09_F10',
            'fault_cb': 'CB_F09',
            'restore_tie': 'Tie_F09_F10',
            'expected_restoration': ['F09_Node4', 'F09_Node5']
        },
        {
            'name': 'F11 Fault - Restore via Tie_F10_F11',
            'fault_cb': 'CB_F11',
            'restore_tie': 'Tie_F10_F11',
            'expected_restoration': ['F11_Node4', 'F11_Node5']
        },
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")

        # Step 1: Normal state
        tester.compile_circuit()
        tester.set_normal_state()
        tester.solve()
        voltages_normal = tester.get_bus_voltages()

        # Step 2: Open fault CB (isolate fault)
        tester.set_switch_state(scenario['fault_cb'], 0)
        tester.solve()
        voltages_isolated = tester.get_bus_voltages()

        # Count de-energized buses after isolation
        deenergized_isolated = [
            bus for bus, data in voltages_isolated.items()
            if data['v_pu'] < 0.1
        ]

        print(f"  After isolation ({scenario['fault_cb']} OPEN):")
        print(f"    De-energized buses: {len(deenergized_isolated)}")

        # Step 3: Close tie switch (restore downstream)
        tester.set_switch_state(scenario['restore_tie'], 1)
        converged = tester.solve()
        voltages_restored = tester.get_bus_voltages()

        # Count still de-energized buses
        still_deenergized = [
            bus for bus, data in voltages_restored.items()
            if data['v_pu'] < 0.1
        ]

        # Count restored buses
        restored_buses = [
            bus for bus in deenergized_isolated
            if bus not in still_deenergized
        ]

        print(f"  After restoration ({scenario['restore_tie']} CLOSED):")
        print(f"    Converged: {'YES' if converged else 'NO'}")
        print(f"    Still de-energized: {len(still_deenergized)}")
        print(f"    Restored buses: {len(restored_buses)}")
        if restored_buses:
            print(f"    Sample restored: {restored_buses[:5]}")

        results['scenarios'].append({
            'name': scenario['name'],
            'deenergized_after_fault': len(deenergized_isolated),
            'restored_buses': len(restored_buses),
            'still_deenergized': len(still_deenergized),
            'converged': converged
        })

    return results


def test_4_sectionalizer_zones(tester: GridTester) -> dict:
    """Test 4: Sectionalizer Zone Isolation"""
    print("\n" + "="*70)
    print("TEST 4: SECTIONALIZER ZONE ISOLATION")
    print("="*70)

    results = {'tests': []}

    # Test F09 feeder with sectionalizer
    print("\n--- F09 Feeder Zone Isolation ---")

    tester.compile_circuit()
    tester.set_normal_state()
    tester.solve()

    # Normal state - count all F09 energized buses
    voltages = tester.get_bus_voltages()
    f09_buses_normal = [bus for bus in voltages.keys() if 'f09' in bus.lower()]
    print(f"Normal state - F09 buses energized: {len(f09_buses_normal)}")

    # Open sectionalizer - isolate downstream zone
    tester.set_switch_state('SEC_F09', 0)
    tester.solve()
    voltages = tester.get_bus_voltages()

    f09_deenergized = [
        bus for bus in voltages.keys()
        if 'f09' in bus.lower() and voltages[bus]['v_pu'] < 0.1
    ]
    f09_energized = [
        bus for bus in voltages.keys()
        if 'f09' in bus.lower() and voltages[bus]['v_pu'] >= 0.1
    ]

    print(f"After SEC_F09 OPEN:")
    print(f"  Upstream zone (energized): {len(f09_energized)} buses")
    print(f"  Downstream zone (de-energized): {len(f09_deenergized)} buses")

    results['tests'].append({
        'feeder': 'F09',
        'normal_buses': len(f09_buses_normal),
        'upstream_after_sec': len(f09_energized),
        'downstream_after_sec': len(f09_deenergized)
    })

    return results


def test_5_24h_simulation_variance(tester: GridTester) -> dict:
    """Test 5: 24-Hour Simulation Variance Analysis"""
    print("\n" + "="*70)
    print("TEST 5: 24-HOUR SIMULATION VARIANCE ANALYSIS")
    print("="*70)
    print("\nThis test analyzes if the simulation repeats identically every 24 hours")

    tester.compile_circuit()
    tester.set_normal_state()

    # Configure for 48-hour simulation
    dss.Text.Command('Set Mode=Daily')
    dss.Text.Command('Set Stepsize=1h')
    dss.Text.Command('Set Number=48')  # 48 hours

    results = {
        'hours': [],
        'load_kw': [],
        'generation_kw': [],
        'losses_kw': [],
        'violations': [],
        'analysis': {}
    }

    print("\nRunning 48-hour simulation (1-hour steps)...")

    for hour in range(48):
        dss.Solution.Solve()

        results['hours'].append(hour)
        results['load_kw'].append(tester.get_total_load() * dss.Solution.LoadMult())

        losses = dss.Circuit.Losses()
        results['losses_kw'].append(losses[0] / 1000)

        voltages = tester.get_bus_voltages()
        violations = tester.analyze_voltage_violations(voltages)
        results['violations'].append(violations['summary']['total'])

        if hour < 24:
            print(f"  Hour {hour:2d}: Load={results['load_kw'][-1]:.1f} kW, "
                  f"Losses={results['losses_kw'][-1]:.1f} kW, "
                  f"Violations={results['violations'][-1]}")

    # Compare Day 1 vs Day 2
    day1_load = results['load_kw'][:24]
    day2_load = results['load_kw'][24:]

    day1_losses = results['losses_kw'][:24]
    day2_losses = results['losses_kw'][24:]

    load_diff = [abs(a - b) for a, b in zip(day1_load, day2_load)]
    losses_diff = [abs(a - b) for a, b in zip(day1_losses, day2_losses)]

    results['analysis'] = {
        'day1_avg_load': np.mean(day1_load),
        'day2_avg_load': np.mean(day2_load),
        'max_load_diff': max(load_diff),
        'avg_load_diff': np.mean(load_diff),
        'day1_avg_violations': np.mean(results['violations'][:24]),
        'day2_avg_violations': np.mean(results['violations'][24:]),
        'identical_pattern': max(load_diff) < 0.1  # If max diff < 0.1 kW, identical
    }

    print(f"\n--- Day 1 vs Day 2 Comparison ---")
    print(f"Day 1 Avg Load: {results['analysis']['day1_avg_load']:.1f} kW")
    print(f"Day 2 Avg Load: {results['analysis']['day2_avg_load']:.1f} kW")
    print(f"Max Load Difference: {results['analysis']['max_load_diff']:.4f} kW")
    print(f"Day 1 Avg Violations: {results['analysis']['day1_avg_violations']:.1f}")
    print(f"Day 2 Avg Violations: {results['analysis']['day2_avg_violations']:.1f}")

    if results['analysis']['identical_pattern']:
        print("\n[WARNING] Day 2 is IDENTICAL to Day 1!")
        print("   The simulation repeats the same 24-hour pattern.")
        print("   This is NOT realistic for real-world scenarios.")

    return results


def diagnose_voltage_issues(tester: GridTester) -> dict:
    """Diagnose root causes of voltage violations."""
    print("\n" + "="*70)
    print("VOLTAGE VIOLATION DIAGNOSIS")
    print("="*70)

    tester.compile_circuit()
    tester.set_normal_state()
    tester.solve()

    voltages = tester.get_bus_voltages()

    # Group by voltage level (33kV, 0.4kV, etc.)
    hv_buses = {}  # 33kV
    lv_buses = {}  # 0.4kV

    for bus, data in voltages.items():
        dss.Circuit.SetActiveBus(bus)
        base_kv = dss.Bus.kVBase()

        if base_kv > 10:  # 33kV level
            hv_buses[bus] = data
        else:  # LV level
            lv_buses[bus] = data

    # Analyze by voltage level
    print(f"\n33kV Level Analysis ({len(hv_buses)} buses):")
    hv_voltages = [d['v_pu'] for d in hv_buses.values()]
    if hv_voltages:
        print(f"  Min: {min(hv_voltages):.4f} pu")
        print(f"  Max: {max(hv_voltages):.4f} pu")
        print(f"  Avg: {np.mean(hv_voltages):.4f} pu")

        hv_violations = [b for b, d in hv_buses.items() if d['v_pu'] < V_MIN or d['v_pu'] > V_MAX]
        print(f"  Violations: {len(hv_violations)}")

    print(f"\n0.4kV Level Analysis ({len(lv_buses)} buses):")
    lv_voltages = [d['v_pu'] for d in lv_buses.values()]
    if lv_voltages:
        print(f"  Min: {min(lv_voltages):.4f} pu")
        print(f"  Max: {max(lv_voltages):.4f} pu")
        print(f"  Avg: {np.mean(lv_voltages):.4f} pu")

        lv_violations = [b for b, d in lv_buses.items() if d['v_pu'] < V_MIN or d['v_pu'] > V_MAX]
        print(f"  Violations: {len(lv_violations)}")

        # Find worst LV buses
        worst_low = sorted(lv_buses.items(), key=lambda x: x[1]['v_pu'])[:5]
        worst_high = sorted(lv_buses.items(), key=lambda x: x[1]['v_pu'], reverse=True)[:5]

        print(f"\n  Worst Undervoltage LV Buses:")
        for bus, data in worst_low:
            print(f"    {bus}: {data['v_pu']:.4f} pu")

        print(f"\n  Worst Overvoltage LV Buses:")
        for bus, data in worst_high:
            print(f"    {bus}: {data['v_pu']:.4f} pu")

    # Check transformer tap positions
    print(f"\nTransformer Analysis:")
    xfmr_names = dss.Transformers.AllNames()
    if xfmr_names:
        for xfmr in xfmr_names:
            dss.Transformers.Name(xfmr)
            tap = dss.Transformers.Tap()
            print(f"  {xfmr}: Tap = {tap:.4f}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS TO FIX VOLTAGE VIOLATIONS")
    print("="*70)

    recommendations = []

    if lv_voltages and min(lv_voltages) < V_MIN:
        recommendations.append(
            "1. UNDERVOLTAGE ISSUE:\n"
            "   - Check transformer tap settings (raise LV voltage)\n"
            "   - Add capacitor banks for reactive power support\n"
            "   - Check if line impedances are too high\n"
            "   - Verify load values are realistic for Sri Lankan households"
        )

    if lv_voltages and max(lv_voltages) > V_MAX:
        recommendations.append(
            "2. OVERVOLTAGE ISSUE:\n"
            "   - Check if PV generation is too high relative to load\n"
            "   - Reduce PV system ratings or add voltage control\n"
            "   - Check transformer tap settings (lower them)\n"
            "   - Verify transformer kVA ratings match load"
        )

    recommendations.append(
        "3. REALISTIC SIMULATION:\n"
        "   - Add stochastic variation to load shapes (±10-20%)\n"
        "   - Use weekly/monthly load profiles instead of 24h\n"
        "   - Add weather-based solar variation\n"
        "   - Consider load growth factors"
    )

    recommendations.append(
        "4. VOLTAGE REGULATION:\n"
        "   - Add voltage regulators at 33kV/0.4kV substations\n"
        "   - Configure transformer tap changers (LTC)\n"
        "   - Add capacitor banks at key nodes"
    )

    for rec in recommendations:
        print(f"\n{rec}")

    return {
        'hv_buses': len(hv_buses),
        'lv_buses': len(lv_buses),
        'hv_violations': len([b for b, d in hv_buses.items() if d['v_pu'] < V_MIN or d['v_pu'] > V_MAX]),
        'lv_violations': len([b for b, d in lv_buses.items() if d['v_pu'] < V_MIN or d['v_pu'] > V_MAX]),
        'recommendations': recommendations
    }


def create_stochastic_simulation_example():
    """Show how to add randomness for realistic simulation."""
    print("\n" + "="*70)
    print("EXAMPLE: STOCHASTIC SIMULATION FOR REALISTIC BEHAVIOR")
    print("="*70)

    code_example = '''
# Example: Add stochastic variation to make simulation realistic

import random
import numpy as np

def add_load_variation(base_mult: float, hour: int, day: int) -> float:
    """
    Add realistic variation to load multiplier.

    Args:
        base_mult: Base load multiplier from LoadShape
        hour: Hour of day (0-23)
        day: Day number

    Returns:
        Modified load multiplier with realistic variation
    """
    # Day-of-week effect (weekends are different)
    day_of_week = day % 7
    if day_of_week in [5, 6]:  # Saturday, Sunday
        weekend_factor = 0.85  # 15% less load on weekends
    else:
        weekend_factor = 1.0

    # Random daily variation (±10%)
    daily_random = random.uniform(0.9, 1.1)

    # Temperature effect (hotter = more AC load)
    # Assume temp peaks at 14:00 (hour 14)
    temp_factor = 1.0 + 0.1 * np.sin((hour - 6) * np.pi / 12)

    # Random noise (±5%)
    noise = random.gauss(1.0, 0.025)

    return base_mult * weekend_factor * daily_random * temp_factor * noise


def add_solar_variation(base_irradiance: float, hour: int) -> float:
    """
    Add realistic cloud/weather variation to solar irradiance.

    Args:
        base_irradiance: Base irradiance from LoadShape (W/m²)
        hour: Hour of day

    Returns:
        Modified irradiance with cloud effects
    """
    if base_irradiance < 10:  # Nighttime
        return 0.0

    # Random cloud passing (can reduce output by 30-80%)
    cloud_probability = 0.3  # 30% chance of clouds
    if random.random() < cloud_probability:
        cloud_factor = random.uniform(0.2, 0.7)
    else:
        cloud_factor = random.uniform(0.9, 1.0)

    # Haze/dust effect
    haze_factor = random.uniform(0.85, 1.0)

    return base_irradiance * cloud_factor * haze_factor


# Usage in simulation loop:
# for day in range(365):
#     for hour in range(24):
#         load_mult = add_load_variation(base_load_mult[hour], hour, day)
#         dss.Solution.LoadMult(load_mult)
#
#         solar = add_solar_variation(base_solar[hour], hour)
#         dss.Text.Command(f'Edit LoadShape.SolarShape mult=({solar/1000})')
#
#         dss.Solution.Solve()
'''

    print(code_example)

    # Save example code
    example_file = RESULTS_DIR / "stochastic_simulation_example.py"
    with open(example_file, 'w') as f:
        f.write(code_example)

    print(f"\nExample code saved to: {example_file}")


def generate_report(all_results: dict):
    """Generate comprehensive test report."""
    report_file = RESULTS_DIR / f"switch_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nFull results saved to: {report_file}")

    # Generate summary report
    summary_file = RESULTS_DIR / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CHUNNAKAM GRID - SWITCH AND GRID BEHAVIOR TEST SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if 'test_1' in all_results:
            f.write("TEST 1: Normal State\n")
            f.write(f"  Converged: {all_results['test_1']['converged']}\n")
            f.write(f"  Total Violations: {all_results['test_1']['violations']['summary']['total']}\n\n")

        if 'test_5' in all_results:
            analysis = all_results['test_5']['analysis']
            f.write("TEST 5: 24h Pattern Analysis\n")
            f.write(f"  Pattern Repeats: {analysis['identical_pattern']}\n")
            f.write(f"  Max Day-to-Day Difference: {analysis['max_load_diff']:.4f} kW\n\n")

        if 'diagnosis' in all_results:
            f.write("VOLTAGE DIAGNOSIS:\n")
            f.write(f"  HV Violations: {all_results['diagnosis']['hv_violations']}\n")
            f.write(f"  LV Violations: {all_results['diagnosis']['lv_violations']}\n")

    print(f"Summary saved to: {summary_file}")


def main():
    """Run all tests."""
    print("="*70)
    print("CHUNNAKAM GRID - SWITCH AND GRID BEHAVIOR VERIFICATION")
    print("="*70)
    print(f"\nMaster File: {MASTER_FILE}")
    print(f"Results Directory: {RESULTS_DIR}")

    tester = GridTester()
    all_results = {}

    try:
        # Test 1: Normal State
        all_results['test_1'] = test_1_normal_state(tester)

        # Test 2: Circuit Breaker Operation
        all_results['test_2'] = test_2_circuit_breaker_operation(tester)

        # Test 3: Tie Switch Restoration
        all_results['test_3'] = test_3_tie_switch_restoration(tester)

        # Test 4: Sectionalizer Zones
        all_results['test_4'] = test_4_sectionalizer_zones(tester)

        # Test 5: 24h Simulation Variance
        all_results['test_5'] = test_5_24h_simulation_variance(tester)

        # Voltage Diagnosis
        all_results['diagnosis'] = diagnose_voltage_issues(tester)

        # Show stochastic simulation example
        create_stochastic_simulation_example()

        # Generate report
        generate_report(all_results)

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
