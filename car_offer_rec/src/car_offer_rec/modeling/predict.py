import pickle
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger

from car_offer_rec.config import (
    CLEARML_API_HOST,
    CLEARML_WEB_HOST,
    CLEARML_FILES_HOST,
    CLEARML_ACCESS_KEY,
    CLEARML_SECRET_KEY,
    MODELS_DIR,
    ARTIFACTS_DIR,
    MODEL_ARTIFACTS,
    MODEL_FILENAMES,
)
from car_offer_rec.dataset import (
    data_preprocessing,
    run_data_pipeline,
)
from car_offer_rec.features import (
    generate_temporal_features,
    generate_geo_features,
    generate_user_trip_aggregates,
    generate_offer_specific_features,
    agg_user,
    add_embeddings_to_df,
    join_with_agg,
    prepare_final_dataset,
)

from ml_common.utils import configure_clearml

_clearml_configured = False


def _ensure_clearml():
    """Lazy initialization of ClearML. Called on first use, not at import time."""
    global _clearml_configured
    if not _clearml_configured:
        configure_clearml(
            clearml_api_host=CLEARML_API_HOST,
            clearml_web_host=CLEARML_WEB_HOST,
            clearml_files_host=CLEARML_FILES_HOST,
            clearml_access_key=CLEARML_ACCESS_KEY,
            clearml_secret_key=CLEARML_SECRET_KEY,
        )
        _clearml_configured = True


class CarOfferPredictor:
    """
    MLOps wrapper for car offer recommendation system prediction.

    Loads production models from ClearML Model Registry and data from Lakehouse.

    Usage:
        predictor = CarOfferPredictor()
        predictions_path = predictor.run_prediction()
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the predictor - loads all models once.

        Args:
            models_dir: Path to directory with model artifacts.
                        Defaults to MODELS_DIR from config.
        """
        _ensure_clearml()

        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        # Cached artifacts
        self.artifacts: Dict[str, object] = {}
        self.model: Optional[CatBoostClassifier] = None
        self.categorical_features: list = []
        self.model_version: str = "production"

        # Load all models on init
        self._load_all_artifacts()

        if not self.artifacts:
            raise RuntimeError(f"No models loaded from {self.models_dir}")

        logger.success(f"Predictor initialized. Loaded {len(self.artifacts)} artifacts")

    def _load_all_artifacts(self):
        """Load all model artifacts once at startup"""
        logger.info(f"Loading models from {self.models_dir}")

        # Check if local models exist
        full_model_path = self.models_dir / MODEL_FILENAMES["full_model"]
        if not full_model_path.exists():
            logger.warning(
                f"No models found in {self.models_dir}. Trying ClearML production."
            )
            self._download_production_models_from_clearml()

        # Load each artifact
        for artifact_name, filename in MODEL_FILENAMES.items():
            artifact_path = self.models_dir / filename
            if not artifact_path.exists():
                logger.warning(f"Artifact not found: {artifact_path}")
                continue

            try:
                if artifact_name == "full_model":
                    model = CatBoostClassifier()
                    model.load_model(str(artifact_path))
                    self.artifacts[artifact_name] = model
                    self.model = model
                    logger.debug(f"Loaded CatBoostClassifier: {artifact_name}")

                elif artifact_name == "offers_features":
                    self.artifacts[artifact_name] = pd.read_pickle(artifact_path)
                    logger.debug(f"Loaded DataFrame: {artifact_name}")

                elif artifact_name == "models_metadata":
                    with open(artifact_path, "rb") as f:
                        metadata = pickle.load(f)
                    self.artifacts[artifact_name] = metadata
                    self.categorical_features = metadata.get("categorical_features", [])
                    logger.debug(f"Loaded pickle: {artifact_name}")

                else:
                    self.artifacts[artifact_name] = joblib.load(artifact_path)
                    logger.debug(f"Loaded joblib: {artifact_name}")

            except Exception as e:
                logger.error(f"Failed to load {artifact_path}: {e}")
                continue

    def _download_production_models_from_clearml(self) -> None:
        """Download production models from ClearML Registry"""
        from clearml import Model

        logger.info("Downloading production models from ClearML")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        for artifact_name, model_name in MODEL_ARTIFACTS.items():
            try:
                models = Model.query_models(
                    model_name=model_name,
                    tags=["production"],
                )
                if not models:
                    logger.warning(f"No production model found for {model_name}")
                    continue

                local_path = Path(models[0].get_local_copy())
                if local_path.is_dir():
                    if artifact_name == "full_model":
                        candidates = list(local_path.glob("*.cbm"))
                    else:
                        candidates = list(local_path.glob("*.pkl"))
                    if not candidates:
                        logger.warning(f"No artifact files in {local_path} for {model_name}")
                        continue
                    local_path = candidates[0]

                target_filename = MODEL_FILENAMES.get(artifact_name)
                target_path = self.models_dir / target_filename

                if local_path.resolve() != target_path.resolve():
                    shutil.copy2(local_path, target_path)

                logger.info(f"Downloaded production model {model_name} -> {target_path}")

            except Exception as e:
                logger.error(f"Failed to download model {model_name}: {e}")

    def prepare_data(self) -> tuple:
        """
        Prepare data for prediction using data pipeline.

        Returns:
            Tuple of (df_user, df_trip) DataFrames
        """
        logger.info("Preparing data for prediction...")

        try:
            hex_path, trips_path, users_path, offers_path = run_data_pipeline(
                pipeline_type="predict"
            )

            df_user = (
                pd.read_parquet(users_path)
                if users_path.endswith(".parquet")
                else pd.read_csv(users_path)
            )
            df_trip = (
                pd.read_parquet(trips_path)
                if trips_path.endswith(".parquet")
                else pd.read_csv(trips_path)
            )

            logger.info(f"Data prepared: users={df_user.shape}, trips={df_trip.shape}")
            return df_user, df_trip

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def processing(self, df_user: pd.DataFrame, df_trip: pd.DataFrame) -> pd.DataFrame:
        """Process data and generate features for prediction."""
        result = data_preprocessing(df_user=df_user, df_trip=df_trip)

        df_processed = generate_temporal_features(result)

        # Extract geo models
        geo_models = self.artifacts.get("geo_models", {})
        all_hexagons = geo_models.get("all_hexagons")
        kmeans_origin = geo_models.get("kmeans_origin")
        kmeans_destination = geo_models.get("kmeans_destination")
        tree = geo_models.get("tree")

        df_processed = generate_geo_features(
            df=df_processed,
            all_hexagons=all_hexagons,
            kmeans_origin=kmeans_origin,
            kmeans_destination=kmeans_destination,
            tree=tree,
        )

        df_processed = generate_user_trip_aggregates(df_processed)
        df_processed = generate_offer_specific_features(df_processed)

        # Filter to churn risk users
        churn_users = df_processed[df_processed["is_churn_risk"] == 1]
        churn_user_ids = churn_users["user_id"].unique()
        logger.info(f"Found {len(churn_user_ids)} churn-risk users")

        user_agg = agg_user(churn_users)

        # Extract ALS and PCA models
        als_data = self.artifacts.get("als_data", {})
        pca_models = self.artifacts.get("pca_models", {})

        churn_users = add_embeddings_to_df(
            df_with_vectors=churn_users,
            user_mapping=als_data.get("user_mapping", {}),
            offer_mapping=als_data.get("offer_mapping", {}),
            user_emb_32d=als_data.get("user_embeddings_32d"),
            offer_emb_32d=als_data.get("offer_embeddings_32d"),
            pca_user=pca_models.get("pca_user"),
            pca_offer=pca_models.get("pca_offer"),
        )

        offer_specific_cols = [
            "avg_distance_km_offer_user",
            "avg_duration_min_offer_user",
            "avg_discount_offer_user",
            "avg_trip_base_offer_user",
            "avg_trip_total_offer_user",
            "offer_used_times_offer_user",
            "offer_completed_times_offer_user",
            "offer_conversion_rate_offer_user",
        ]

        for col in offer_specific_cols:
            if col not in churn_users.columns:
                churn_users[col] = 0
            else:
                churn_users[col] = churn_users[col].fillna(0)

        offer_agg = self.artifacts.get("offers_features")
        result_for_model = join_with_agg(
            churn_users=churn_users,
            offer_agg=offer_agg,
            user_agg=user_agg,
        )

        return result_for_model

    def run_prediction(self, data: Optional[pd.DataFrame] = None) -> str:
        """
        Main prediction method.

        Args:
            data: Optional preprocessed data. If None, loads from pipeline.

        Returns:
            Path to the saved predictions CSV file.
        """
        logger.info("Starting predictions...")

        # Prepare data if not provided
        if data is None:
            df_user, df_trip = self.prepare_data()
            result_for_model = self.processing(df_user, df_trip)
        else:
            result_for_model = data

        if result_for_model.empty:
            logger.warning("No data available for prediction")
            return ""

        final_dataset = prepare_final_dataset(result_for_model)

        # Convert categorical features to proper types
        for col in self.categorical_features:
            if col in final_dataset.columns:
                final_dataset[col] = final_dataset[col].fillna("missing").astype(str)

        predict_pool = Pool(final_dataset, cat_features=self.categorical_features)
        predictions = self.model.predict_proba(predict_pool)[:, 1]
        final_dataset["retention_probability"] = predictions

        # Get top recommendation per user
        top_recommendations = (
            final_dataset.groupby("user_id")
            .apply(
                lambda x: x.nlargest(1, "retention_probability")[
                    ["offer_id", "retention_probability"]
                ]
            )
            .reset_index()
        )

        results_df = top_recommendations[["user_id", "offer_id", "retention_probability"]]

        if not results_df.empty:
            logger.success(f"Predictions ready: {len(results_df)} rows")

            # Save to file
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = (
                ARTIFACTS_DIR / f"predictions_{uuid.uuid4().hex[:8]}.csv"
            )
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

            # Statistics
            avg_prob = results_df["retention_probability"].mean()
            logger.info(f"Average retention probability: {avg_prob:.4f}")

            return str(output_path)
        else:
            logger.warning("No prediction results")
            return ""


if __name__ == "__main__":
    # Initialize predictor
    predictor = CarOfferPredictor(models_dir=str(MODELS_DIR))

    # Run prediction
    predictions_path = predictor.run_prediction()

    if predictions_path:
        print(f"Predictions saved to: {predictions_path}")