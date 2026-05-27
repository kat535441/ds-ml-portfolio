import pandas as pd
import duckdb
from typing import List
from pathlib import Path

from loguru import logger

from ml_pipelines.data_ingestion.base_ingestion import BaseDataPipeline, DatasetConfig
from car_offer_rec.config import (
    CLEARML_DATASET_PROJECT_NAME,
    CLEARML_DATASET_NAME_HEX,
    CLEARML_DATASET_NAME_TRIPS,
    CLEARML_DATASET_NAME_USERS,
    CLEARML_DATASET_NAME_OFFERS,
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    SPN_ID,
    SPN_SECRET,
    SPN_TENANT_ID,
    ONELAKE_ACCOUNT_URL,
    WORKSPACE,
    LAKEHOUSE,
    FABRIC_NOTEBOOK_ID,
    FABRIC_WORKSPACE_ID,
    NUMERIC_COLUMNS_FLOAT64,
)


# =============================================================================
# DATA PIPELINE (BaseDataPipeline subclass)
# =============================================================================

class CarOfferDataPipeline(BaseDataPipeline):
    """Data ingestion pipeline for Car Offer Recommendation Service.

    Uses Azure Fabric Lakehouse as the primary data source (source="lakehouse").
    """

    @property
    def notebook_id(self) -> str:
        return FABRIC_NOTEBOOK_ID

    @property
    def workspace_id(self) -> str:
        return FABRIC_WORKSPACE_ID

    @property
    def dataset_project(self) -> str:
        return CLEARML_DATASET_PROJECT_NAME

    @property
    def datasets(self) -> List[DatasetConfig]:
        return [
            DatasetConfig(
                name=CLEARML_DATASET_NAME_HEX,
                lakehouse_pattern=CLEARML_DATASET_NAME_HEX,
                source="clearml"
            ),
            DatasetConfig(
                name=CLEARML_DATASET_NAME_TRIPS,
                lakehouse_pattern=CLEARML_DATASET_NAME_TRIPS,
                source="lakehouse"
            ),
            DatasetConfig(
                name=CLEARML_DATASET_NAME_USERS,
                lakehouse_pattern=CLEARML_DATASET_NAME_USERS,
                source="lakehouse"
            ),
            DatasetConfig(
                name=CLEARML_DATASET_NAME_OFFERS,
                lakehouse_pattern=CLEARML_DATASET_NAME_OFFERS,
                source="lakehouse"
            ),
        ]

    @property
    def data_raw_dir(self) -> Path:
        return DATA_RAW_DIR

    @property
    def spn_tenant_id(self) -> str:
        return SPN_TENANT_ID

    @property
    def spn_client_id(self) -> str:
        return SPN_ID

    @property
    def spn_client_secret(self) -> str:
        return SPN_SECRET

    @property
    def onelake_account_url(self) -> str:
        return ONELAKE_ACCOUNT_URL

    @property
    def workspace_name(self) -> str:
        return WORKSPACE

    @property
    def lakehouse_name(self) -> str:
        return LAKEHOUSE


# =============================================================================
# BACKWARD COMPATIBLE WRAPPER
# =============================================================================

def run_data_pipeline(pipeline_type: str):
    """
    Backward-compatible wrapper for the data pipeline.

    Returns:
        Tuple of paths: (hex_path, trips_path, users_path, offers_path)
    """
    # Import inside function for ClearML pipeline serialization
    from car_offer_rec.dataset import CarOfferDataPipeline

    if pipeline_type not in ["train", "predict"]:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    pipeline = CarOfferDataPipeline()
    paths = pipeline.run(pipeline_type)

    return (
        paths[CLEARML_DATASET_NAME_HEX],
        paths[CLEARML_DATASET_NAME_TRIPS],
        paths[CLEARML_DATASET_NAME_USERS],
        paths[CLEARML_DATASET_NAME_OFFERS],
    )


# =============================================================================
# CORE PREPROCESSING LOGIC (shared by train and predict)
# =============================================================================

def _preprocess_core(df_user: pd.DataFrame, df_trip: pd.DataFrame) -> pd.DataFrame:
    """
    Core preprocessing logic: joins users and trips via DuckDB, drops NaN coordinates.
    
    This function contains the shared logic used by both preprocess_data (training)
    and data_preprocessing (prediction). Keep changes here to avoid divergence.
    """
    # Convert numeric columns to float64 to avoid DuckDB DECIMAL overflow
    df_trip = df_trip.copy()
    for col in NUMERIC_COLUMNS_FLOAT64:
        if col in df_trip.columns:
            df_trip[col] = df_trip[col].astype('float64')

    con = duckdb.connect()
    con.register("df_user", df_user)
    con.register("df_trip", df_trip)

    result = con.execute("""
        SELECT
            t.trip_id, t.created_at AS trip_created_at, t.completed_at,
            t.distance_km, t.duration_min, t.payment_method_id, t.trip_base,
            t.trip_total, t.origin_lat, t.origin_lng, t.destination_lat, t.destination_lng,
            t.offer_id, u.user_id, u.created_at AS user_created_at,
            u.last_active_at, u.client_type
        FROM df_trip t
        INNER JOIN df_user u ON t.user_id = u.user_id
    """).df()

    initial_count = len(result)
    result = result.dropna(subset=['destination_lat', 'destination_lng'])
    logger.info(f"Deleted {initial_count - len(result)} trips with missing coordinates")

    return result


# =============================================================================
# PREPROCESS DATA (for pipeline step)
# =============================================================================

def preprocess_data(
    trips_path: str,
    users_path: str,
) -> str:
    """
    Joins users and trips via duckdb, drops NaN coordinates.
    Saves result to data/interim and returns the path.
    """
    logger.info("Preprocessing: joining users and trips...")

    df_trip = pd.read_parquet(trips_path) if trips_path.endswith(".parquet") else pd.read_csv(trips_path)
    df_user = pd.read_parquet(users_path) if users_path.endswith(".parquet") else pd.read_csv(users_path)

    result = _preprocess_core(df_user, df_trip)

    out_path = DATA_INTERIM_DIR / "preprocessed.parquet"
    result.to_parquet(out_path, index=False)
    logger.info(f"Preprocessed data saved to {out_path}. Shape: {result.shape}")

    return str(out_path)


# =============================================================================
# PREDICTION-TIME PREPROCESSING
# =============================================================================

def data_preprocessing(df_user, df_trip):
    """Prediction-time preprocessing: joins users and trips."""
    logger.info("Data preprocessing...")
    return _preprocess_core(df_user, df_trip)