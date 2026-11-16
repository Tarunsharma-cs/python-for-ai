"""Shared paths and settings used by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "logistics_delivery_data.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "clean_data.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

TARGET = "delayed"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# delivery_time_min is only known AFTER the delivery finishes, and the target
# is defined from it. Using it as a feature would be data leakage, so we leave
# it out when training.
LEAK_COLUMNS = ["delivery_time_min"]

NUMERIC_FEATURES = ["distance_km", "driver_rating"]
CATEGORICAL_FEATURES = ["vehicle_type", "traffic_level", "weather_condition", "delivery_priority"]

for _p in (CLEAN_DATA.parent, FIG_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)
