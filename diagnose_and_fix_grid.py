"""
============================================================================
CHUNNAKAM GRID - COMPREHENSIVE DIAGNOSTIC AND FIX SCRIPT
============================================================================
This script diagnoses the voltage violation and simulation issues, and
demonstrates how to fix them.

ROOT CAUSE ANALYSIS:
1. Generation >> Load at snapshot (91 MW vs 73 MW)
2. Solar irradiance=1 in snapshot mode means peak generation always
3. LoadShapes not applied in snapshot mode
4. No voltage regulation (static tap settings)

Student ID: IT22053350
Date: January 2026
============================================================================
"""

import opendssdirect as dss
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

MASTER_FILE = Path(__file__).parent / "Master.dss"
RESULTS_DIR = Path(__file__).parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Voltage limits
V_MIN, V_MAX = 0.95, 1.05

def compile_circuit():
    """Compile the OpenDSS circuit."""
    dss.Text.Command('Clear')  # Clear any existing circuit first
    dss.Text.Command(f'Compile "{MASTER_FILE}"')
    # Suppress any show commands that might fail
    try:
        dss.Text.Command('Set ShowExport=NO')
    except:
        pass
    return dss.Circuit.Name()

def get_totals():
    """Get total load and generation."""
    total_load = 0.0
    load_names = dss.Loads.AllNames()
    if load_names:
        for name in load_names:
            dss.Loads.Name(name)
            total_load += dss.Loads.kW()

    total_gen = 0.0
    gen_names = dss.Generators.AllNames()
    if gen_names:
        for name in gen_names:
            dss.Generators.Name(name)
            total_gen += dss.Generators.kW()

    total_pv = 0.0
    pv_names = dss.PVsystems.AllNames()
    if pv_names:
        for name in pv_names:
            dss.PVsystems.Name(name)
            total_pv += dss.PVsystems.Pmpp()  # Use Pmpp for rated capacity

    return total_load, total_gen, total_pv

def count_violations():
    """Count voltage violations."""
    violations = {'under': 0, 'over': 0, 'total': 0}
    bus_names = dss.Circuit.AllBusNames()

    for bus in bus_names:
        dss.Circuit.SetActiveBus(bus)
        v_pu = dss.Bus.puVmagAngle()[::2]
        if v_pu:
            avg_v = np.mean(v_pu)
            if avg_v < V_MIN:
                violations['under'] += 1
            elif avg_v > V_MAX:
                violations['over'] += 1

    violations['total'] = violations['under'] + violations['over']
    return violations

def get_losses():
    """Get total losses in kW."""
    losses = dss.Circuit.Losses()
    return losses[0] / 1000 if losses else 0

print("="*70)
print("CHUNNAKAM GRID - COMPREHENSIVE DIAGNOSTIC")
print("="*70)

# ============================================================================
# DIAGNOSIS 1: SNAPSHOT MODE ISSUES
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS 1: SNAPSHOT MODE (Current Default)")
print("="*70)

compile_circuit()
total_load, total_gen, total_pv = get_totals()

print(f"\nNAMED CAPACITY IN MODEL:")
print(f"  Total Load (kW):        {total_load:,.0f}")
print(f"  UJPS + Wind (kW):       {total_gen:,.0f}")
print(f"  Solar PV Pmpp (kW):     {total_pv:,.0f}")
print(f"  TOTAL GENERATION (kW):  {total_gen + total_pv:,.0f}")
print(f"\n  Generation - Load:      {(total_gen + total_pv) - total_load:+,.0f} kW")

print("\n  [!] PROBLEM: At irradiance=1.0, generation exceeds load by ~18 MW!")
print("      This causes massive reverse power flow and overvoltage.")

# Solve and check
dss.Solution.Solve()
converged = dss.Solution.Converged()
violations = count_violations()

print(f"\nSNAPSHOT SOLVE RESULTS:")
print(f"  Converged: {'YES' if converged else 'NO'}")
print(f"  Undervoltage buses: {violations['under']}")
print(f"  Overvoltage buses:  {violations['over']}")
print(f"  Total violations:   {violations['total']}")

# ============================================================================
# DIAGNOSIS 2: WHY LOAD IS STATIC
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS 2: WHY LOAD APPEARS STATIC")
print("="*70)

print("""
In SNAPSHOT mode:
- LoadShapes are NOT automatically applied
- Loads use their base kW values (as defined in Households.dss)
- Solution.LoadMult() multiplier is 1.0 by default

In DAILY mode:
- LoadShapes ARE applied via the 'daily=...' property
- Each hour uses the appropriate multiplier from LoadShape
- But you must iterate through time steps with Solution.Solve()

The 'daily=' property ONLY works when Mode=Daily or Mode=Yearly!
""")

# ============================================================================
# DIAGNOSIS 3: 24-HOUR REPETITION
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS 3: 24-HOUR REPETITION")
print("="*70)

print("""
LoadShapes in this model are 24-point arrays (one per hour).
When simulation exceeds 24 hours, OpenDSS WRAPS AROUND:
- Hour 25 = Hour 1
- Hour 48 = Hour 24

This is by DESIGN in OpenDSS - LoadShapes repeat cyclically.

TO ADD REALISM:
1. Create 8760-point LoadShapes (full year, hourly)
2. Add stochastic noise to each day
3. Use different profiles for weekdays/weekends
4. Vary solar irradiance based on weather
""")

# ============================================================================
# FIX 1: REDUCE SOLAR AT SNAPSHOT TO MATCH TYPICAL MID-DAY
# ============================================================================
print("\n" + "="*70)
print("FIX 1: Scale Solar Irradiance to Realistic Level")
print("="*70)

compile_circuit()

# At mid-day, irradiance is ~0.85 per the Solar_Default_Daily shape
# But loads are also different at mid-day
# Let's simulate mid-day conditions properly

# Set irradiance to 0.7 (typical partly cloudy day)
pv_names = dss.PVsystems.AllNames()
if pv_names:
    for pv_name in pv_names:
        dss.PVsystems.Name(pv_name)
        dss.PVsystems.Irradiance(0.5)  # 50% of peak (typical real-world)

# Also reduce UJPS to typical mid-day operation
gen_names = dss.Generators.AllNames()
if gen_names:
    for gen_name in gen_names:
        if 'UJPS' in gen_name:
            dss.Generators.Name(gen_name)
            dss.Generators.kW(dss.Generators.kW() * 0.25)  # 25% during solar peak

dss.Solution.Solve()
converged = dss.Solution.Converged()
violations = count_violations()
losses = get_losses()

print(f"\nWith irradiance=0.5 and UJPS at 25%:")
print(f"  Converged: {'YES' if converged else 'NO'}")
print(f"  Undervoltage: {violations['under']}")
print(f"  Overvoltage:  {violations['over']}")
print(f"  Total losses: {losses:,.1f} kW")

# ============================================================================
# FIX 2: USE DAILY MODE PROPERLY
# ============================================================================
print("\n" + "="*70)
print("FIX 2: Use Daily Mode with LoadShapes")
print("="*70)

compile_circuit()

# Configure for daily simulation
dss.Text.Command('Set Mode=Daily')
dss.Text.Command('Set Stepsize=1h')
dss.Text.Command('Set Number=1')  # Just solve one step

results_by_hour = []

print("\n24-Hour Simulation with LoadShapes Applied:")
print("-" * 60)
print(f"{'Hour':>4} | {'Load Mult':>9} | {'Converged':>9} | {'Violations':>10} | {'Losses kW':>10}")
print("-" * 60)

for hour in range(24):
    dss.Solution.Solve()

    converged = dss.Solution.Converged()
    violations = count_violations()
    losses = get_losses()
    load_mult = dss.Solution.LoadMult()

    results_by_hour.append({
        'hour': hour,
        'load_mult': load_mult,
        'converged': converged,
        'violations': violations['total'],
        'losses': losses
    })

    print(f"{hour:>4} | {load_mult:>9.3f} | {'YES' if converged else 'NO':>9} | {violations['total']:>10} | {losses:>10,.1f}")

print("-" * 60)

# ============================================================================
# FIX 3: TRANSFORMER TAP ADJUSTMENT
# ============================================================================
print("\n" + "="*70)
print("FIX 3: Adjust Transformer Taps for Voltage Regulation")
print("="*70)

compile_circuit()

# The 33/0.4 kV distribution transformers need tap adjustment
# If LV voltage is low, we need to RAISE the tap (increase secondary voltage)
# If LV voltage is high, we need to LOWER the tap

# Current taps are 1.0 - let's adjust based on typical loading

# For this grid with high solar penetration:
# - During day: Solar causes overvoltage at LV -> Lower taps
# - During night: Heavy load causes undervoltage -> Normal/higher taps

print("\nCurrent transformer tap settings:")
xfmr_names = dss.Transformers.AllNames()
for xfmr in xfmr_names:
    dss.Transformers.Name(xfmr)
    tap = dss.Transformers.Tap()
    print(f"  {xfmr}: {tap:.4f}")

print("\nAdjusting distribution transformer taps...")
for xfmr in xfmr_names:
    if xfmr.startswith('DT_'):  # Distribution transformers
        dss.Transformers.Name(xfmr)
        # Lower tap to reduce LV voltage (help with overvoltage from solar)
        dss.Transformers.Tap(0.975)  # -2.5% to reduce overvoltage

# Solve and check
dss.Solution.Mode(0)  # Snapshot
dss.Solution.Solve()
converged = dss.Solution.Converged()
violations = count_violations()

print(f"\nAfter tap adjustment (all DTs to 0.975):")
print(f"  Converged: {'YES' if converged else 'NO'}")
print(f"  Violations: {violations['total']}")

# ============================================================================
# FIX 4: STOCHASTIC SIMULATION FRAMEWORK
# ============================================================================
print("\n" + "="*70)
print("FIX 4: Stochastic Simulation Framework")
print("="*70)

print("""
To make simulation realistic (no two days identical):

1. ADD RANDOM NOISE TO LOADS:
   def get_load_with_noise(base_mult, noise_std=0.05):
       return base_mult * (1 + np.random.normal(0, noise_std))

2. VARY SOLAR BASED ON WEATHER:
   weather = random.choice(['sunny', 'partly_cloudy', 'cloudy'])
   solar_mult = {'sunny': 1.0, 'partly_cloudy': 0.6, 'cloudy': 0.3}[weather]

3. DIFFERENT WEEKEND PATTERNS:
   if day_of_week in [5, 6]:  # Sat, Sun
       load_mult *= 0.85  # 15% less on weekends

4. SEASONAL VARIATION:
   month_factor = [0.85, 0.88, 0.95, 1.0, 0.98, 0.90, ...][month]

5. RANDOM OUTAGES/EVENTS:
   if random.random() < 0.01:  # 1% chance per hour
       simulate_minor_fault()
""")

# ============================================================================
# SWITCH BEHAVIOR TEST
# ============================================================================
print("\n" + "="*70)
print("SWITCH BEHAVIOR TEST")
print("="*70)

compile_circuit()

# Set normal state
# CBs and SECs should be CLOSED (enabled=yes)
# Tie switches should be OPEN (enabled=no)

switches_config = [
    ('CB_F05', 'yes'), ('CB_F06', 'yes'), ('CB_F07', 'yes'), ('CB_F08', 'yes'),
    ('CB_F09', 'yes'), ('CB_F10', 'yes'), ('CB_F11', 'yes'), ('CB_F12', 'yes'),
    ('SEC_F05', 'yes'), ('SEC_F06', 'yes'), ('SEC_F07', 'yes'), ('SEC_F08', 'yes'),
    ('SEC_F09', 'yes'), ('SEC_F10', 'yes'), ('SEC_F11', 'yes'), ('SEC_F12', 'yes'),
    ('Tie_F06_F07', 'no'), ('Tie_F07_F08', 'no'),
    ('Tie_F09_F10', 'no'), ('Tie_F10_F11', 'no'),
    ('BusCoupler', 'no'),
]

print("\nSetting switches to normal state...")
for switch, state in switches_config:
    dss.Text.Command(f'Edit Line.{switch} enabled={state}')

dss.Solution.Solve()
print(f"Converged after normal state: {dss.Solution.Converged()}")

# Test F07 fault isolation and restoration
print("\n--- TEST: F07 Fault Isolation and Restoration ---")

# Step 1: Open CB_F07 to isolate fault
print("\n1. Opening CB_F07 (simulating fault on F07)...")
dss.Text.Command('Edit Line.CB_F07 enabled=no')
dss.Solution.Solve()

# Check F07 buses
f07_voltages = []
for bus in dss.Circuit.AllBusNames():
    if 'f07' in bus.lower():
        dss.Circuit.SetActiveBus(bus)
        v = dss.Bus.puVmagAngle()[::2]
        if v:
            f07_voltages.append((bus, np.mean(v)))

deenergized = [b for b, v in f07_voltages if v < 0.1]
print(f"   F07 buses de-energized: {len(deenergized)}")

# Step 2: Close Tie_F06_F07 to restore downstream
print("\n2. Closing Tie_F06_F07 (restoring F07 via F06)...")
dss.Text.Command('Edit Line.Tie_F06_F07 enabled=yes')
dss.Solution.Solve()

# Check F07 buses again
f07_voltages_after = []
for bus in dss.Circuit.AllBusNames():
    if 'f07' in bus.lower():
        dss.Circuit.SetActiveBus(bus)
        v = dss.Bus.puVmagAngle()[::2]
        if v:
            f07_voltages_after.append((bus, np.mean(v)))

restored = [b for b, v in f07_voltages_after if v >= 0.9]
print(f"   F07 buses restored to >0.9 pu: {len(restored)}")

converged = dss.Solution.Converged()
print(f"   Solution converged: {converged}")

# ============================================================================
# SUMMARY RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: ISSUES AND RECOMMENDED FIXES")
print("="*70)

print("""
ISSUE 1: Circuit doesn't converge in snapshot mode
-------------------------------------------------
CAUSE: Generation (91 MW) >> Load (73 MW) at irradiance=1.0
FIX:
  a) Use Daily mode with LoadShapes (realistic time-of-day)
  b) Reduce default irradiance or UJPS output for snapshot testing
  c) The model is DESIGNED for time-varying simulation, not snapshot

ISSUE 2: 120+ voltage violations
--------------------------------
CAUSE:
  - High solar penetration causes overvoltage at LV buses
  - Long feeders cause voltage drop at remote buses
  - No automatic tap changers or voltage regulators
FIX:
  a) Adjust distribution transformer taps (lower for overvoltage areas)
  b) Add voltage regulators at key points
  c) Add capacitor banks for reactive power support
  d) The violations may be REALISTIC for high-PV networks!

ISSUE 3: Load is static (73,500 kW always)
-----------------------------------------
CAUSE: Snapshot mode doesn't apply LoadShapes
FIX:   Use 'Set Mode=Daily' and iterate with Solution.Solve()

ISSUE 4: 24h simulation repeats identically
-------------------------------------------
CAUSE: LoadShapes are 24-point arrays that repeat
FIX:
  a) Add stochastic noise to load multipliers
  b) Vary solar irradiance based on weather probability
  c) Use different profiles for weekdays/weekends
  d) Create 8760-point annual LoadShapes

ISSUE 5: High losses (13+ MW)
-----------------------------
CAUSE: Unrealistic power flow from excessive generation
FIX:   Balance generation with load (use realistic scenarios)

REAL-WORLD CONTEXT:
------------------
In a real grid with 47 MW of solar on a 73 MW load:
- Voltage violations ARE expected during high solar periods
- Utilities use OLTC, voltage regulators, and PV curtailment
- Self-healing (your research) helps restore after faults
- The model behavior is actually REALISTIC for high-PV grids!
""")

# Save results
results = {
    'diagnosis_time': datetime.now().isoformat(),
    'named_load_kw': total_load,
    'named_gen_kw': total_gen,
    'named_pv_kw': total_pv,
    'hourly_results': results_by_hour
}

results_file = RESULTS_DIR / f"grid_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
