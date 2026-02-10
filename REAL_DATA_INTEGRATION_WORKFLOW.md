# Real Data Integration Workflow for Chunnakam Grid OpenDSS Simulation

## Executive Summary

This document provides a detailed, step-by-step methodology to replace synthetic load profiles, solar irradiance curves, and wind speed data with actual historical data in your Chunnakam Grid OpenDSS simulation. The workflow addresses the unique challenges of bidirectional power flow, time resolution mismatches, and load disaggregation.

---

## 1. Data Inventory & Analysis

### 1.1 Available Data Sources

| Data Type | Source | Format | Resolution | Period | Key Column(s) |
|-----------|--------|--------|------------|--------|---------------|
| **Feeder Load Profiles** | CEB Metering | TSV (.xls) | 15-minute | May-Aug 2025 | MW, Mvar, MVA |
| **Solar Irradiance** | NASA POWER | CSV | Hourly | May 1 - Aug 31, 2025 | ALLSKY_SFC_SW_DWN (Wh/m²) |
| **Wind Speed** | NASA POWER | CSV | Hourly | May 1 - Aug 31, 2025 | WS50M (m/s @ 50m) |

### 1.2 Critical Data Characteristics

**Load Profile Data Structure:**
```
Timestamp           MW      Mvar    MVA
01-08-2025 00:00    5.094   -0.184  5.097   ← Positive MW = Grid → Customer
01-08-2025 10:30   -0.317   -1.300  1.365   ← Negative MW = Customer → Grid (DER Export)
01-08-2025 19:30    6.364   -0.635  6.395   ← Evening Peak
```

**Key Insight:** The load profile data represents **NET LOAD at feeder level**, not gross load:
```
Net Load = Gross Load - PV Generation - Other DER
```

When MW is negative, DER generation exceeds consumption, indicating reverse power flow.

---

## 2. Methodology Overview

### 2.1 Approach Selection


**Option A: Decomposed Load/Generation Approach (Advanced)**
- Separate gross load from DER generation
- Requires assumptions about load/PV correlation
- Better for studying individual component behavior

**Recommendation:** Start with Option A

### 2.2 Time Resolution Strategy

| Data Source | Native Resolution | Target Resolution | Action |
|-------------|-------------------|-------------------|--------|
| Load Profiles | 15-minute | 15-minute | Use directly |
| Solar Irradiance | Hourly | 15-minute | Interpolate |
| Wind Speed | Hourly | 15-minute | Interpolate |

---

## 3. Phase 1: Data Preprocessing

### 3.1 Load Profile Processing

**Step 1.1: Parse All Feeder Files**

Create a Python script to parse all 32 load profile files:

Load profile data is available on "C:\Users\D E L L\Desktop\Data pre-processing for Machine Learning in Python\GRID_SIMULATION\data\raw\Load profiles" directory. Each file named by the relevent feeder and the month. Example: "F05_May.xlx" is represent the load profile of feeder F05 in the month of May. 

```python
# data_preprocessing/parse_load_profiles.py
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_load_profile(filepath):
    """Parse CEB load profile TSV file."""
    df = pd.read_csv(filepath, sep='\t', skiprows=0)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
    
    # Extract relevant columns
    df = df[['Timestamp', 'MW', 'Mvar', 'MVA']].copy()
    
    # Convert to numeric (handle any string values)
    df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    df['Mvar'] = pd.to_numeric(df['Mvar'], errors='coerce')
    df['MVA'] = pd.to_numeric(df['MVA'], errors='coerce')
    
    return df

def load_all_feeders(base_path, feeder_ids=['F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12']):
    """Load all feeder data into a dictionary."""
    feeder_data = {}
    
    for fid in feeder_ids:
        feeder_data[fid] = {}
        for month in ['May', 'June', 'July', 'August']:
            # Adjust filename pattern based on your actual file naming
            pattern = f"*{fid}*{month}*.xls"
            files = list(Path(base_path).glob(pattern))
            if files:
                feeder_data[fid][month] = parse_load_profile(files[0])
    
    return feeder_data
```

**Step 1.2: Handle Missing/Invalid Data**

```python
def clean_load_data(df):
    """Clean and validate load profile data."""
    # Remove rows with all NaN values
    df = df.dropna(how='all', subset=['MW', 'Mvar'])
    
    # Forward-fill small gaps (up to 4 intervals = 1 hour)
    df['MW'] = df['MW'].interpolate(method='linear', limit=4)
    df['Mvar'] = df['Mvar'].interpolate(method='linear', limit=4)
    
    # Flag remaining gaps for review
    df['data_quality'] = df['MW'].notna().astype(int)
    
    return df
```

**Step 1.3: Compute Net Load Sign Convention**

The MW column already follows the convention:
- **Positive MW**: Power flowing from grid to customer (consumption > generation)
- **Negative MW**: Power flowing from customer to grid (generation > consumption)

No conversion needed—this directly maps to OpenDSS load modeling with negative values.

### 3.2 Solar Irradiance Processing

**Step 2.1: Parse NASA Data**

Solar irradiance data is available on "C:\Users\D E L L\Desktop\Data pre-processing for Machine Learning in Python\GRID_SIMULATION\data\raw\Environmental data\" directory. The file name is "Solar irridiance (May-August).csv".

```python
# data_preprocessing/parse_solar.py
import pandas as pd

def parse_nasa_solar(filepath):
    """Parse NASA POWER solar irradiance CSV."""
    # Skip header lines
    df = pd.read_csv(filepath, skiprows=9)
    
    # Create datetime column
    df['Timestamp'] = pd.to_datetime(
        df['YEAR'].astype(str) + '-' + 
        df['MO'].astype(str).str.zfill(2) + '-' + 
        df['DY'].astype(str).str.zfill(2) + ' ' + 
        df['HR'].astype(str).str.zfill(2) + ':00:00'
    )
    
    # Rename irradiance column
    df = df.rename(columns={'ALLSKY_SFC_SW_DWN': 'GHI_Wh_m2'})
    
    return df[['Timestamp', 'GHI_Wh_m2']]
```

**Step 2.2: Interpolate to 15-Minute Resolution**

```python
def interpolate_to_15min(df, value_col):
    """Interpolate hourly data to 15-minute intervals."""
    # Set timestamp as index
    df = df.set_index('Timestamp')
    
    # Resample to 15-minute intervals
    df_15min = df.resample('15T').interpolate(method='linear')
    
    return df_15min.reset_index()
```

**Step 2.3: Convert Irradiance to PV Power Output**

```python
def irradiance_to_pv_power(ghi_wh_m2, pv_capacity_kw, panel_efficiency=0.18, 
                            system_efficiency=0.85, stc_irradiance=1000):
    """
    Convert GHI irradiance to PV power output.
    
    Parameters:
    - ghi_wh_m2: Global Horizontal Irradiance (Wh/m²)
    - pv_capacity_kw: Installed PV capacity in kW
    - panel_efficiency: Panel efficiency (default 18%)
    - system_efficiency: System efficiency including inverter, wiring losses
    - stc_irradiance: Standard Test Conditions irradiance (1000 W/m²)
    
    Returns:
    - Power output in kW
    """
    # GHI in Wh/m² is essentially average W/m² over the hour
    # For 15-min intervals after interpolation, treat as instantaneous power
    
    # Normalized irradiance (0-1 scale where 1 = STC conditions)
    normalized_ghi = ghi_wh_m2 / stc_irradiance
    
    # PV output = Capacity × Normalized GHI × System Efficiency
    pv_power_kw = pv_capacity_kw * normalized_ghi * system_efficiency
    
    return pv_power_kw
```

### 3.3 Wind Speed Processing

**Step 3.1: Parse Wind Data**

Wind speed data is available on "C:\Users\D E L L\Desktop\Data pre-processing for Machine Learning in Python\GRID_SIMULATION\data\raw\Environmental data\" directory. The file name is "Wind speed (May-August).csv".

```python
def parse_nasa_wind(filepath):
    """Parse NASA POWER wind speed CSV."""
    df = pd.read_csv(filepath, skiprows=9)
    
    df['Timestamp'] = pd.to_datetime(
        df['YEAR'].astype(str) + '-' + 
        df['MO'].astype(str).str.zfill(2) + '-' + 
        df['DY'].astype(str).str.zfill(2) + ' ' + 
        df['HR'].astype(str).str.zfill(2) + ':00:00'
    )
    
    df = df.rename(columns={'WS50M': 'WindSpeed_ms'})
    
    return df[['Timestamp', 'WindSpeed_ms']]
```

**Step 3.2: Convert Wind Speed to Power (Wind Turbine Power Curve)**

```python
def wind_speed_to_power(wind_speed_ms, rated_power_kw=20000, 
                         cut_in=3.5, rated_speed=12.0, cut_out=25.0):
    """
    Convert wind speed to wind farm power output using power curve.
    
    Typical large wind turbine parameters:
    - Cut-in: 3.5 m/s
    - Rated: 12 m/s
    - Cut-out: 25 m/s
    """
    import numpy as np
    
    power = np.zeros_like(wind_speed_ms, dtype=float)
    
    # Below cut-in: no power
    mask_below = wind_speed_ms < cut_in
    power[mask_below] = 0
    
    # Between cut-in and rated: cubic relationship
    mask_ramp = (wind_speed_ms >= cut_in) & (wind_speed_ms < rated_speed)
    power[mask_ramp] = rated_power_kw * ((wind_speed_ms[mask_ramp] - cut_in) / 
                                          (rated_speed - cut_in)) ** 3
    
    # At rated or above (but below cut-out): full power
    mask_rated = (wind_speed_ms >= rated_speed) & (wind_speed_ms <= cut_out)
    power[mask_rated] = rated_power_kw
    
    # Above cut-out: shutdown
    mask_cutout = wind_speed_ms > cut_out
    power[mask_cutout] = 0
    
    return power
```

---

## 4. Phase 2: Load Disaggregation Strategy

### 4.1 The Challenge

Current OpenDSS model has **multiple loads per feeder**, but the actual data is **feeder-level aggregate**:

**Model Structure (Example - F06):**
| Load | Peak (kW) | Type |
|------|-----------|------|
| F06_Industrial_Factory | 4,000 | Industrial |
| F06_SmallIndustry | 1,500 | Industrial |
| F06_Residential | 700 | Residential |
| **Total** | **6,200** | |

**Actual Data:** Single net MW value at feeder head.

### 4.2 Disaggregation Methodology

**Method 2: Profile-Based Scaling (More Realistic)**

Apply different temporal profiles to different load types:

```python
# Load type profiles from your OpenDSS LoadShapes
LOAD_PROFILES = {
    'Industrial': 'Industrial_Daily',      # Relatively flat, 24/7
    'Residential': 'Residential_Daily',    # Morning/evening peaks
    'Commercial': 'Commercial_Daily',      # Daytime peak
    'Agricultural': 'Agricultural_Daily',  # Early morning peak
}

def disaggregate_with_profiles(feeder_net_load_kw, load_info, hour):
    """
    Disaggregate using load type profiles.
    
    load_info: dict like {
        'F06_Industrial_Factory': {'capacity': 4000, 'type': 'Industrial'},
        ...
    }
    """
    # Get profile multipliers for current hour
    multipliers = {}
    for load_name, info in load_info.items():
        profile = LOAD_PROFILES[info['type']]
        multipliers[load_name] = get_profile_multiplier(profile, hour) * info['capacity']
    
    # Normalize multipliers
    total_weighted = sum(multipliers.values())
    
    disaggregated = {}
    for load_name, weighted_cap in multipliers.items():
        ratio = weighted_cap / total_weighted
        disaggregated[load_name] = feeder_net_load_kw * ratio
    
    return disaggregated
```

### 4.3 Handling Negative Net Load (DER Export)

When net load is negative, the load components should still consume power, but distributed PV generation exceeds consumption:

**Approach:** Separate handling for positive vs negative net load scenarios:

```python
def handle_bidirectional_flow(feeder_net_load_kw, load_capacities, pv_capacities, solar_multiplier):
    """
    Handle bidirectional power flow by separating load and generation.
    
    For OpenDSS:
    - Loads always consume (positive kW)
    - PV systems generate based on irradiance
    - Net effect is the feeder measurement
    """
    # Estimate gross load (always positive)
    # Use typical load factor or regression model
    total_load_capacity = sum(load_capacities.values())
    total_pv_capacity = sum(pv_capacities.values())
    
    # Estimate PV generation from solar data
    estimated_pv_gen = total_pv_capacity * solar_multiplier
    
    # Estimate gross load = net load + PV generation
    estimated_gross_load = feeder_net_load_kw + estimated_pv_gen
    
    # Sanity check: gross load should be positive during daytime
    estimated_gross_load = max(estimated_gross_load, total_load_capacity * 0.1)  # Min 10% base load
    
    return {
        'gross_load_kw': estimated_gross_load,
        'pv_generation_kw': estimated_pv_gen,
        'net_load_kw': feeder_net_load_kw
    }
```

---

## 5. Phase 3: OpenDSS LoadShape Generation

### 5.1 LoadShape File Format

OpenDSS LoadShapes use multipliers (0.0 to 1.0+) applied to base load values:

```dss
! Example LoadShape definition
New Loadshape.F06_Industrial_Aug1 
~   npts=96 
~   interval=0.25  ! 15-minute intervals (0.25 hours)
~   mult=[0.82, 0.81, 0.80, ..., 0.85]  ! 96 values for 24 hours
```

### 5.2 Generate LoadShapes from Real Data

```python
def generate_opendss_loadshape(load_name, daily_data, base_kw):
    """
    Generate OpenDSS LoadShape definition from daily data.
    
    daily_data: DataFrame with 'Timestamp' and 'MW' columns for one day
    base_kw: Base load value (typically peak load) in kW
    """
    # Ensure 96 points (15-min intervals × 24 hours)
    assert len(daily_data) == 96, f"Expected 96 points, got {len(daily_data)}"
    
    # Convert MW to kW
    kw_values = daily_data['MW'].values * 1000
    
    # Calculate multipliers
    multipliers = kw_values / base_kw
    
    # Format for OpenDSS
    mult_str = ', '.join([f'{m:.4f}' for m in multipliers])
    
    dss_code = f"""New Loadshape.{load_name}
~   npts=96
~   interval=0.25
~   mult=[{mult_str}]
"""
    return dss_code
```

### 5.3 Generate Solar Irradiance Shapes

```python
def generate_solar_loadshape(date, solar_data, shape_name):
    """
    Generate solar irradiance LoadShape for a specific date.
    """
    # Filter data for the specific date
    day_data = solar_data[solar_data['Timestamp'].dt.date == date]
    
    # Normalize to peak irradiance
    max_ghi = day_data['GHI_Wh_m2'].max()
    if max_ghi > 0:
        multipliers = day_data['GHI_Wh_m2'] / max_ghi
    else:
        multipliers = day_data['GHI_Wh_m2'] * 0
    
    mult_str = ', '.join([f'{m:.4f}' for m in multipliers.values])
    
    dss_code = f"""New Loadshape.{shape_name}
~   npts=96
~   interval=0.25
~   mult=[{mult_str}]
"""
    return dss_code
```

---

## 6. Phase 4: OpenDSS Model Modification

### 6.1 Modify LoadShapes.dss

Replace synthetic shapes with real data shapes:

```dss
! LoadShapes.dss - Modified for Real Data

! === LOAD SHAPES FROM ACTUAL DATA ===
! Generated from CEB feeder measurements

! F06 Industrial Feeder - August 1, 2025
Redirect LoadShapes_RealData/F06_Aug01.dss

! F07 Mixed/Agriculture Feeder - August 1, 2025
Redirect LoadShapes_RealData/F07_Aug01.dss

! ... (repeat for all feeders and dates)

! === SOLAR IRRADIANCE SHAPES ===
! Generated from NASA POWER data
Redirect LoadShapes_RealData/Solar_Aug01.dss

! === WIND SHAPES ===
Redirect LoadShapes_RealData/Wind_Aug01.dss
```

### 6.2 Modify Households.dss (Loads)

Link loads to the new LoadShapes:

```dss
! Households.dss - Modified to use real data shapes

! F06 Industrial Feeder
New Load.F06_Industrial_Factory bus1=F06_HH1
~   phases=3 conn=wye model=1
~   kV=0.4 kW=4000 kvar=2400
~   daily=F06_Industrial_Aug01

New Load.F06_SmallIndustry bus1=F06_HH2
~   phases=3 conn=wye model=1
~   kV=0.4 kW=1500 kvar=750
~   daily=F06_SmallIndustry_Aug01
```

### 6.3 Modify RooftopSolar.dss

Link PV systems to real irradiance data:

```dss
! RooftopSolar.dss - Modified for real irradiance

New PVSystem.PV_F06_Factory bus1=F06_HH1
~   phases=3 conn=wye
~   kVA=5500 Pmpp=5000
~   irradiance=1.0  ! Base irradiance
~   daily=Solar_Aug01  ! Real irradiance shape
~   %cutin=5 %cutout=5
~   effcurve=InverterEff
~   Tpmpp=35

! Repeat for all PV systems
```

### 6.4 Modify Generators.dss (Wind Farm)

```dss
! Generators.dss - Wind farm with real wind data

New Generator.WindFarm bus1=F11_WindFarm
~   phases=3
~   kV=33 kW=20000 kvar=0 pf=0.9
~   model=1
~   daily=Wind_Aug01  ! Real wind power shape
```

---

## 7. Phase 5: Simulation Execution

### 7.1 Master.dss Configuration

```dss
! Master.dss - Configured for Real Data Simulation

Clear

Set DefaultBaseFrequency=50

! Define the circuit
New Circuit.Chunnakam_GSS basekv=132 pu=1.0 phases=3 Mvasc3=1500 Mvasc1=1200

! Include modified components
Redirect LineCodes.dss
Redirect Transformers.dss
Redirect Lines.dss
Redirect Switches.dss
Redirect LoadShapes_RealData.dss  ! Real data shapes
Redirect Households.dss
Redirect RooftopSolar.dss
Redirect Generators.dss
Redirect Monitors.dss

! Solve mode configuration
Set mode=daily
Set stepsize=15m
Set number=96  ! 24 hours × 4 steps/hour
```

### 7.2 Batch Simulation Script

```python
# run_simulation.py
import opendssdirect as dss
from datetime import date, timedelta

def run_daily_simulation(simulation_date):
    """Run simulation for a specific date using real data."""
    
    # Generate LoadShape files for this date
    generate_loadshapes_for_date(simulation_date)
    
    # Compile OpenDSS model
    dss.run_command('Clear')
    dss.run_command('Compile Master.dss')
    
    # Set simulation parameters
    dss.run_command('Set mode=daily')
    dss.run_command('Set stepsize=15m')
    dss.run_command('Set number=96')
    
    # Solve
    dss.run_command('Solve')
    
    # Extract results
    results = extract_monitor_data()
    
    return results

def run_period_simulation(start_date, end_date):
    """Run simulations for a date range."""
    results = {}
    
    current_date = start_date
    while current_date <= end_date:
        print(f"Simulating {current_date}")
        results[current_date] = run_daily_simulation(current_date)
        current_date += timedelta(days=1)
    
    return results
```

---

## 8. Validation Framework

### 8.1 Validation Metrics

Compare simulation results against measured data:

```python
def validate_simulation(measured_data, simulated_data):
    """Calculate validation metrics."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    metrics = {
        'MAE': mean_absolute_error(measured_data, simulated_data),
        'RMSE': np.sqrt(mean_squared_error(measured_data, simulated_data)),
        'MAPE': np.mean(np.abs((measured_data - simulated_data) / measured_data)) * 100,
        'Correlation': np.corrcoef(measured_data, simulated_data)[0, 1]
    }
    
    return metrics
```

### 8.2 Validation Checkpoints

| Checkpoint | Expected Outcome | Tolerance |
|------------|-----------------|-----------|
| Feeder net power matches input | Measured ≈ Simulated | < 2% |
| Transformer loading realistic | < 100% normal, < 120% emergency | - |
| Voltage within limits | 0.95 ≤ V ≤ 1.05 pu | - |
| Power factor reasonable | 0.85 ≤ PF ≤ 1.0 | - |
| Reverse power flow timing | During high solar hours | - |

---



## 9. File Structure Recommendation

```
GRID_SIMULATION/
├── data/
│   ├── raw/
│   │   ├── load_profiles/
│   │   │   ├── F05_May.xls
│   │   │   ├── F05_June.xls
│   │   │   └── ...
│   │   ├── Solar_irradiance_MayAug.csv
│   │   └── Wind_speed_MayAug.csv
│   └── processed/
│       ├── load_profiles_cleaned.csv
│       ├── solar_15min.csv
│       └── wind_15min.csv
│
├── opendss_model/
│   ├── Master.dss
│   ├── LineCodes.dss
│   ├── Transformers.dss
│   ├── Lines.dss
│   ├── Switches.dss
│   ├── Households.dss
│   ├── RooftopSolar.dss
│   ├── Generators.dss
│   ├── Monitors.dss
│   └── LoadShapes_RealData/
│       ├── F06_Aug01.dss
│       ├── Solar_Aug01.dss
│       └── ...
│
├── scripts/
│   ├── data_preprocessing/
│   │   ├── parse_load_profiles.py
│   │   ├── parse_solar.py
│   │   ├── parse_wind.py
│   │   └── interpolate.py
│   ├── loadshape_generation/
│   │   ├── generate_load_shapes.py
│   │   ├── generate_solar_shapes.py
│   │   └── generate_wind_shapes.py
│   ├── simulation/
│   │   ├── run_simulation.py
│   │   └── batch_simulation.py
│   └── validation/
│       └── validate_results.py
│
├── results/
│   └── (simulation outputs)
│
└── README.md
```

---

## 10. Feeder-to-Load Mapping Reference

Based on current grid topology, here's the complete mapping for load disaggregation:

### F05 - UJPS Generator Feeder
- No loads (generator interconnect only)
- UJPS_Gen1, UJPS_Gen2, UJPS_Gen3: 8 MW each (24 MW total)

### F06 - KKS Industrial
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F06_Industrial_Factory | 4,000 | Industrial | 5,000 |
| F06_SmallIndustry | 1,500 | Industrial | 3,500 |
| F06_Residential | 700 | Residential | 1,640 |
| **Total** | **6,200** | | **10,140** |

### F07 - Vaddu Mixed
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F07_Village_Center | 2,500 | Mixed | 4,000 |
| F07_Agricultural | 3,000 | Agricultural | 3,500 |
| F07_Rural_Houses | 2,300 | Residential | 1,670 |
| **Total** | **7,800** | | **9,170** |

### F08 - Kompanai Distribution
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F08_Commercial | 5,000 | Commercial | 2,500 |
| F08_Residential | 4,500 | Residential | 2,000 |
| F08_Mixed | 3,100 | Mixed | 940 |
| **Total** | **12,600** | | **5,440** |

### F09 - Chavakachcheri Rural
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F09_Town_Center | 2,200 | Mixed | 200 |
| F09_Village | 1,800 | Residential | 150 |
| F09_Rural_Agri | 2,000 | Agricultural | 0 |
| F09_Remote | 1,200 | Residential | 0 |
| **Total** | **7,200** | | **350** |

### F10 - Point Pedro Coastal
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F10_Town | 5,500 | Commercial | 3,500 |
| F10_Fishing | 4,000 | Industrial | 2,000 |
| F10_Coastal_Res | 3,100 | Residential | 1,500 |
| **Total** | **12,600** | | **7,000** |

### F11 - Jaffna Town Urban
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F11_Hospital | 2,500 | Critical | 1,500 |
| F11_Commercial | 4,500 | Commercial | 3,500 |
| F11_Apartments | 4,000 | Residential | 3,000 |
| F11_MixedRes | 3,400 | Residential | 2,220 |
| **Total** | **14,400** | | **10,220** |
| + WindFarm | | | 20,000 |

### F12 - Tie Feeder
| Load | kW | Type | PV (kW) |
|------|-----|------|---------|
| F12_Residential1 | 7,000 | Residential | 3,000 |
| F12_Residential2 | 5,700 | Residential | 2,000 |
| **Total** | **12,700** | | **5,000** |

---

## 12. Summary

This workflow provides a systematic approach to integrate real historical data into your Chunnakam Grid OpenDSS simulation:

1. **Data Preprocessing**: Parse, clean, and interpolate all data sources to 15-minute resolution
2. **Load Disaggregation**: Distribute feeder-level measurements to individual load components
3. **LoadShape Generation**: Create OpenDSS-compatible LoadShape definitions
4. **Model Modification**: Update DSS files to reference real data shapes
5. **Simulation Execution**: Run daily/period simulations
6. **Validation**: Compare simulation outputs against measured data

The key innovation is handling **bidirectional power flow** by either:
- Using net load directly (simpler)
- Decomposing into gross load + DER generation (more realistic)

Start with Option A (direct net load) for initial validation, then migrate to Option B for detailed component-level analysis.

---

*Document Version: 1.0*  
*Generated: January 2026*  
*For: Chunnakam Grid OpenDSS Simulation Project*
