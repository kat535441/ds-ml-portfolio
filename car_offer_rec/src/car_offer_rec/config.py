import os
from typing import List
from dotenv import load_dotenv

from ml_utils.paths import find_workspace_root


def ensure_directories():
    for dir_path in [
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_INTERIM_DIR,
        DATA_EXTERNAL_DIR,
        MODELS_DIR,
        METRICS_DIR,
        OUTPUT_DIR,
        ARTIFACTS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. PATHS AND DIRECTORIES
# =============================================================================

load_dotenv()

WORKSPACE_ROOT = find_workspace_root(__file__)

TARGET_METRIC = "AUC"

DATA_DIR = WORKSPACE_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = WORKSPACE_ROOT / "models"
METRICS_DIR = WORKSPACE_ROOT / "metrics"

OUTPUT_DIR = WORKSPACE_ROOT / "results"
ARTIFACTS_DIR = WORKSPACE_ROOT / "predictions"


# =============================================================================
# 2. AZURE (MODEL STORAGE)
# =============================================================================

ONELAKE_ACCOUNT_URL = "https://onelake.dfs.fabric.microsoft.com"
WORKSPACE = "AnalyticsLake"
LAKEHOUSE = "MLDataUpload"

FABRIC_NOTEBOOK_ID = ""
FABRIC_WORKSPACE_ID = ""

SPN_SECRET = os.getenv("SPN_SECRET")
SPN_ID = os.getenv("SPN_ID")
SPN_TENANT_ID = os.getenv("SPN_TENANT_ID")

AZURE_CONTAINER_NAME = "caroffer"
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")

AZURE_OUTPUT_URI = (
    f"azure://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER_NAME}"
)


# =============================================================================
# 3. CLEARML CONNECTION
# =============================================================================

CLEARML_API_HOST = os.getenv("CLEARML_API_HOST", "https://api.clear.ml")
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST", "https://app.clear.ml")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST", "https://files.clear.ml")
CLEARML_ACCESS_KEY = os.getenv("CLEARML_ACCESS_KEY")
CLEARML_SECRET_KEY = os.getenv("CLEARML_SECRET_KEY")


# =============================================================================
# 4. CLEARML DATASETS
# =============================================================================

CLEARML_DATASET_PROJECT_NAME = "MLPlatform/CarOfferRec"
CLEARML_DATASET_NAME_HEX = "hex"
CLEARML_DATASET_NAME_TRIPS = "Trips"
CLEARML_DATASET_NAME_USERS = "Users"
CLEARML_DATASET_NAME_OFFERS = "Offers"


# =============================================================================
# 5. METRICS LOGGING TASK
# =============================================================================

CLEARML_TRACKER_TASK_ID = ""


# =============================================================================
# 6. PREDICTION OUTPUT PATHS
# =============================================================================

PREDICTION_BLOB_PATH = "caroffer/predictions/predictions.csv"


# =============================================================================
# 7. MODEL ARTIFACTS
# =============================================================================

MODEL_ARTIFACT_NAMES: List[str] = [
    "car_offer_full_model",
    "car_offer_geo_models",
    "car_offer_als_data",
    "car_offer_pca_models",
    "offers_features",
    "car_offer_models_metadata",
]

MODEL_ARTIFACTS: dict = {
    "full_model": "car_offer_full_model",
    "geo_models": "car_offer_geo_models",
    "als_data": "car_offer_als_data",
    "pca_models": "car_offer_pca_models",
    "offers_features": "offers_features",
    "models_metadata": "car_offer_models_metadata",
}

MODEL_FILENAMES: dict = {
    "full_model": "full_model.cbm",
    "geo_models": "geo_models.pkl",
    "als_data": "als_data.pkl",
    "pca_models": "pca_models.pkl",
    "offers_features": "offers_features.pkl",
    "models_metadata": "models_metadata.pkl",
}


# =============================================================================
# 8. DATA PROCESSING
# =============================================================================

NUMERIC_COLUMNS_FLOAT64: List[str] = [
    "distance_km",
    "duration_min",
    "trip_total",
    "trip_base",
    "origin_lat",
    "origin_lng",
    "destination_lat",
    "destination_lng",
]

ID_COLUMNS: List[str] = [
    "user_id",
    "offer_id",
    "trip_id",
]


# =============================================================================
# 9. BUILD PARAMETERS
# =============================================================================

PROJECT = "MLPlatform"
BUILD_CONF_ID = "Train pipeline - CarOfferRec"
EXEC_QUEUE = "cpu"


ensure_directories()