from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Pakwheel_car_data.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "cars.csv"
MODEL_PATH = ROOT / "models" / "valuation.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"

REFERENCE_YEAR = 2024
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MIN_YEAR = 1985
MAX_YEAR = 2024
RARE_MIN_COUNT = 20

# A displayed figure at or above this is already in lakh. Nothing in this
# market is 22 crore, while plenty of ordinary cars are 22 lakh.
ANCHOR_LAKH = 22.0
# Crore readings in the scrape sit below this. The column itself caps at 100.
CRORE_READING_MAX = 16.5

PREMIUM_MAKES = {
    "Mercedes Benz",
    "BMW",
    "Audi",
    "Lexus",
    "Porsche",
    "Range Rover",
    "Land Rover",
    "Jaguar",
    "Tesla",
    "Bentley",
    "Rolls Royce",
    "Maserati",
}

MULTI_WORD_MAKES = (
    "Mercedes Benz",
    "Range Rover",
    "Land Rover",
    "Alfa Romeo",
    "Rolls Royce",
)

FUEL_TYPES = ("Petrol", "Hybrid", "Diesel", "CNG", "Electric", "LPG")
TRANSMISSIONS = ("Automatic", "Manual")

FEATURE_COLUMNS = [
    "car_age",
    "log_km",
    "km_per_year",
    "engine_cc",
    "battery_kwh",
    "is_hybrid",
    "is_electric",
    "transmission_auto",
    "make",
    "model_name",
    "fuel",
    "variant_tier",
]

NUMERIC_FEATURES = [
    "car_age",
    "log_km",
    "km_per_year",
    "engine_cc",
    "battery_kwh",
    "is_hybrid",
    "is_electric",
    "transmission_auto",
]
CATEGORICAL_FEATURES = ["make", "model_name", "fuel", "variant_tier"]
