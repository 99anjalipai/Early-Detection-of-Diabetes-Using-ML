import os
from pathlib import Path

# ----------------------------
# GLOBAL CONFIGURATION
# ----------------------------

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Default raw dataset filename (update if needed)
RAW_DATA_FILE = DATA_RAW / "diabetes_012_health_indicators_BRFSS2015.csv"

# Random state for reproducibility
RANDOM_STATE = 42

# Target column name
TARGET_COL = "Diabetes_012"
