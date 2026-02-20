"""
Configuration for real data integration pipeline.
Contains file paths, feeder-load mappings, and constants.
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data
RAW_LOAD_PROFILES_DIR = PROJECT_ROOT / "data" / "raw" / "load_profiles"
RAW_SOLAR_FILE = PROJECT_ROOT / "data" / "raw" / "Environmental data" / "Solar irridiance (May-August).csv"
RAW_WIND_FILE = PROJECT_ROOT / "data" / "raw" / "Environmental data" / "Wind speed (May-August).csv"

# Processed data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DISAGGREGATED_DIR = PROCESSED_DIR / "disaggregated"
LOAD_PROFILES_CLEANED = PROCESSED_DIR / "load_profiles_cleaned.csv"
SOLAR_15MIN = PROCESSED_DIR / "solar_15min.csv"
WIND_15MIN = PROCESSED_DIR / "wind_15min.csv"

# Generated DSS shapes
LOADSHAPES_DIR = PROJECT_ROOT / "LoadShapes_RealData"

# DSS files to modify
MASTER_DSS = PROJECT_ROOT / "Master.dss"
LOADSHAPES_DSS = PROJECT_ROOT / "LoadShapes.dss"
HOUSEHOLDS_DSS = PROJECT_ROOT / "Households.dss"
ROOFTOPSOLAR_DSS = PROJECT_ROOT / "RooftopSolar.dss"
GENERATORS_DSS = PROJECT_ROOT / "Generators.dss"
DSS_DATE_FILES = [HOUSEHOLDS_DSS, ROOFTOPSOLAR_DSS, GENERATORS_DSS, LOADSHAPES_DSS]
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# DEFAULT SIMULATION DATE
# ============================================================================

DEFAULT_DATE = "2025-08-01"

# ============================================================================
# NASA DATA PARSING
# ============================================================================

NASA_HEADER_SKIP = 9  # Number of header lines to skip in NASA CSV files
STC_IRRADIANCE = 1000.0  # W/m^2 Standard Test Conditions irradiance

# ============================================================================
# WIND TURBINE PARAMETERS
# ============================================================================

WIND_CUT_IN = 3.5    # m/s
WIND_RATED = 12.0    # m/s
WIND_CUT_OUT = 25.0  # m/s
WIND_FARM_CAPACITY_KW = 20000  # 20 MW

# ============================================================================
# PV SYSTEM EFFICIENCY
# ============================================================================

PV_SYSTEM_EFFICIENCY = 0.85  # Combined inverter + wiring + derating

# ============================================================================
# FEEDER LOADS MAPPING
# ============================================================================
# Maps each feeder to its loads with rated kW, load type, and synthetic profile name.
# The synthetic_profile key references the 24-hour multiplier arrays below.

FEEDER_LOADS = {
    "F06": [
        {"name": "F06_Industrial_Factory", "kw": 4000, "type": "industrial",
         "synthetic_profile": "Industrial_Daily"},
        {"name": "F06_SmallIndustry", "kw": 1500, "type": "light_industrial",
         "synthetic_profile": "LightIndustrial_Daily"},
        {"name": "F06_Residential", "kw": 700, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
    ],
    "F07": [
        {"name": "F07_Village_Center", "kw": 2500, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
        {"name": "F07_Agricultural", "kw": 3000, "type": "agricultural",
         "synthetic_profile": "Agricultural_Daily"},
        {"name": "F07_Rural_Houses", "kw": 2300, "type": "rural_residential",
         "synthetic_profile": "Rural_Residential_Daily"},
    ],
    "F08": [
        {"name": "F08_Commercial", "kw": 5000, "type": "commercial",
         "synthetic_profile": "Commercial_Daily"},
        {"name": "F08_Residential", "kw": 4500, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
        {"name": "F08_Mixed", "kw": 3100, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
    ],
    "F09": [
        {"name": "F09_Town_Center", "kw": 2200, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
        {"name": "F09_Village", "kw": 1800, "type": "rural_residential",
         "synthetic_profile": "Rural_Residential_Daily"},
        {"name": "F09_Rural_Agri", "kw": 2000, "type": "agricultural",
         "synthetic_profile": "Agricultural_Daily"},
        {"name": "F09_Remote", "kw": 1200, "type": "rural_residential",
         "synthetic_profile": "Rural_Residential_Daily"},
    ],
    "F10": [
        {"name": "F10_Town", "kw": 5500, "type": "commercial",
         "synthetic_profile": "Commercial_Daily"},
        {"name": "F10_Fishing", "kw": 4000, "type": "cold_storage",
         "synthetic_profile": "ColdStorage_Daily"},
        {"name": "F10_Coastal_Res", "kw": 3100, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
    ],
    "F11": [
        {"name": "F11_Hospital", "kw": 2500, "type": "hospital",
         "synthetic_profile": "Hospital_Daily"},
        {"name": "F11_Commercial", "kw": 4500, "type": "commercial",
         "synthetic_profile": "Commercial_Daily"},
        {"name": "F11_Apartments", "kw": 4000, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
        {"name": "F11_MixedRes", "kw": 3400, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
    ],
    "F12": [
        {"name": "F12_Residential1", "kw": 7000, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
        {"name": "F12_Residential2", "kw": 5700, "type": "residential",
         "synthetic_profile": "Residential_Daily"},
    ],
}

# ============================================================================
# FEEDER PV CAPACITY MAPPING (kW)
# ============================================================================

FEEDER_PV = {
    "F06": [
        {"name": "PV_F06_Factory", "kw": 5000},
        {"name": "PV_F06_SmallInd", "kw": 3500},
        {"name": "PV_F06_Residential", "kw": 1640},
    ],
    "F07": [
        {"name": "PV_F07_Village", "kw": 4000},
        {"name": "PV_F07_Agricultural", "kw": 3500},
        {"name": "PV_F07_Rural", "kw": 1670},
    ],
    "F08": [
        {"name": "PV_F08_Commercial", "kw": 2500},
        {"name": "PV_F08_Residential", "kw": 2000},
        {"name": "PV_F08_Mixed", "kw": 940},
    ],
    "F09": [
        {"name": "PV_F09_Town", "kw": 200},
        {"name": "PV_F09_Village", "kw": 150},
    ],
    "F10": [
        {"name": "PV_F10_Town", "kw": 3500},
        {"name": "PV_F10_Fishing", "kw": 2000},
        {"name": "PV_F10_Coastal", "kw": 1500},
    ],
    "F11": [
        {"name": "PV_F11_Hospital", "kw": 1500},
        {"name": "PV_F11_Commercial", "kw": 3500},
        {"name": "PV_F11_Apartments", "kw": 3000},
        {"name": "PV_F11_MixedRes", "kw": 2220},
    ],
    "F12": [
        {"name": "PV_F12_Res1", "kw": 3000},
        {"name": "PV_F12_Res2", "kw": 2000},
    ],
}

# ============================================================================
# SYNTHETIC LOAD SHAPE MULTIPLIERS (24 hourly values, hour 0-23)
# ============================================================================
# Used for disaggregation weighting: determines how each load type
# distributes across the day relative to others.

SYNTHETIC_PROFILES = {
    "Residential_Daily": [
        0.35, 0.30, 0.28, 0.27, 0.28, 0.45, 0.70, 0.75,
        0.55, 0.40, 0.35, 0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.75, 0.95, 1.00, 0.90, 0.75, 0.55, 0.42,
    ],
    "Rural_Residential_Daily": [
        0.40, 0.35, 0.32, 0.30, 0.35, 0.55, 0.80, 0.65,
        0.45, 0.35, 0.30, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.70, 1.00, 0.95, 0.85, 0.70, 0.55, 0.45,
    ],
    "Commercial_Daily": [
        0.20, 0.15, 0.15, 0.15, 0.15, 0.20, 0.35, 0.55,
        0.80, 0.95, 1.00, 0.95, 0.90, 0.90, 0.85, 0.80,
        0.85, 0.90, 0.75, 0.50, 0.35, 0.30, 0.25, 0.22,
    ],
    "Industrial_Daily": [
        0.60, 0.55, 0.52, 0.50, 0.50, 0.55, 0.62, 0.65,
        0.55, 0.45, 0.35, 0.30, 0.30, 0.32, 0.35, 0.45,
        0.60, 0.80, 0.95, 1.00, 0.98, 0.90, 0.78, 0.68,
    ],
    "LightIndustrial_Daily": [
        0.25, 0.20, 0.18, 0.18, 0.20, 0.30, 0.50, 0.75,
        0.90, 0.95, 0.98, 1.00, 0.95, 0.90, 0.85, 0.80,
        0.75, 0.60, 0.45, 0.40, 0.35, 0.30, 0.28, 0.26,
    ],
    "Hospital_Daily": [
        0.70, 0.65, 0.62, 0.60, 0.62, 0.70, 0.85, 0.95,
        1.00, 1.00, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85,
        0.82, 0.85, 0.88, 0.85, 0.80, 0.78, 0.75, 0.72,
    ],
    "Agricultural_Daily": [
        0.10, 0.08, 0.08, 0.10, 0.25, 0.70, 0.95, 1.00,
        0.80, 0.40, 0.20, 0.15, 0.15, 0.18, 0.25, 0.55,
        0.85, 0.90, 0.70, 0.35, 0.20, 0.15, 0.12, 0.10,
    ],
    # 24-hour cold storage / refrigeration (fishing industry, ice plants)
    # Near-flat profile: refrigeration compressors run continuously
    # Slight peak at midday (product loading) and evening (increased demand)
    "ColdStorage_Daily": [
        0.82, 0.80, 0.79, 0.78, 0.80, 0.83, 0.88, 0.92,
        0.95, 0.98, 1.00, 0.99, 0.97, 0.96, 0.95, 0.93,
        0.92, 0.93, 0.95, 0.94, 0.91, 0.88, 0.85, 0.83,
    ],
}

# ============================================================================
# FEEDER LIST (excluding F05 which is UJPS-only)
# ============================================================================

LOAD_FEEDERS = ["F06", "F07", "F08", "F09", "F10", "F11", "F12"]
ALL_FEEDERS = ["F05", "F06", "F07", "F08", "F09", "F10", "F11", "F12"]

# ============================================================================
# MINIMUM GROSS LOAD FRACTION
# ============================================================================
# When decomposing net load + PV to get gross load, clamp gross load
# to at least this fraction of total rated capacity (prevents negative/zero).

MIN_GROSS_LOAD_FRACTION = 0.10
