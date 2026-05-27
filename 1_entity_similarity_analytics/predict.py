from pathlib import Path
from typing import Tuple
import pickle
import uuid

import os
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from dataset import load_data, clean_transactions_data, download_from_s3
from features import create_features
from config import (
    CATEGORY_LABELS,
    FEATURE_COLUMNS_TO_EXCLUDE,
    EMBEDDING_FEATURE_COLUMNS_TO_EXCLUDE,
    COMPARISON_METRICS,
    ALWAYS_INCREASE,
    ALWAYS_DECREASE,
    FOLLOW_BENCHMARK,
    NUMERIC_COLUMNS,
    MODELS_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    S3_BUCKET_NAME,
)


class EntitySimilarityPredictor:
    def __init__(
        self, models_dir: str = None, output_dir: str = None
    ) -> None:
        self.models_dir = models_dir or str(MODELS_DIR)
        self.output_dir = output_dir or str(OUTPUT_DIR)
        self._models = {}

    def load_production_models(self) -> dict:
        segment_keys = ["A", "B", "C", "D_E_F"]
        loaded = {}

        for segment_key in segment_keys:
            model_path = Path(self.models_dir) / f"model_segment_{segment_key}.pkl"
            if not model_path.exists():
                raise ValueError(f"No production model found for segment_{segment_key}")

            with model_path.open("rb") as f:
                bundle = pickle.load(f)

            loaded[segment_key] = {
                "model": bundle["model"],
                "scaler": bundle["scaler"],
                "segment": bundle.get("segment", segment_key),
                "label_encoder": bundle.get("label_encoder"),
            }

        self._models = loaded
        return loaded

    def download_models_from_s3(self) -> dict:
        segment_keys = ["A", "B", "C", "D_E_F"]
        loaded = {}

        os.makedirs(self.models_dir, exist_ok=True)

        for segment_key in segment_keys:
            s3_key = f"models/model_segment_{segment_key}.pkl"
            local_path = f"{self.models_dir}/model_segment_{segment_key}.pkl"

            try:
                download_from_s3(s3_key, local_path)
                logger.info(f"Downloaded model from S3: {s3_key}")
            except Exception as e:
                logger.warning(f"Model not found in S3: {s3_key} ({e})")
                continue

            with open(local_path, "rb") as f:
                bundle = pickle.load(f)

            loaded[segment_key] = {
                "model": bundle["model"],
                "scaler": bundle["scaler"],
                "segment": bundle.get("segment", segment_key),
                "label_encoder": bundle.get("label_encoder"),
            }

        self._models = loaded
        return loaded

    def predict_categories_with_models(
        self, final_with_metrics_ml: pd.DataFrame, models: dict
    ) -> pd.DataFrame:
        df_for_prediction = final_with_metrics_ml.copy()
        feature_columns = [
            column
            for column in df_for_prediction.columns
            if column not in FEATURE_COLUMNS_TO_EXCLUDE
        ]

        predictions = []

        for idx, row in df_for_prediction.iterrows():
            segment_raw = row["entity_segment"]

            if pd.isna(segment_raw):
                predictions.append("Unclassified")
                continue

            segment = str(segment_raw)

            if segment in ["D", "E", "F"]:
                model_key = "D_E_F"
            else:
                model_key = segment

            if model_key not in models:
                if "A" in models:
                    model_key = "A"
                    logger.warning(
                        f"For segment {segment} no model found. "
                        "Using segment A as fallback"
                    )
                else:
                    predictions.append("Unclassified")
                    continue

            model_data = models[model_key]
            model = model_data["model"]
            scaler = model_data["scaler"]

            if model_data.get("label_encoder") is not None:
                label_encoder = model_data["label_encoder"]
            else:
                label_encoder = LabelEncoder()
                label_encoder.classes_ = np.array(CATEGORY_LABELS)

            missing_features = [
                feature_column
                for feature_column in feature_columns
                if feature_column not in row.index
            ]

            if missing_features:
                predictions.append("Unclassified")
                continue

            features = row[feature_columns].values.reshape(1, -1)

            expected_n = scaler.n_features_in_
            actual_n = features.shape[1]
            if actual_n != expected_n:
                logger.warning(
                    f"Feature count mismatch for entity {idx}: "
                    f"scaler expects {expected_n}, got {actual_n}. "
                    "Marking as Unclassified."
                )
                predictions.append("Unclassified")
                continue

            features_scaled = scaler.transform(features)

            encoded_prediction = model.predict(features_scaled)[0]

            try:
                category = label_encoder.inverse_transform([encoded_prediction])[0]
                predictions.append(category)
            except:
                predictions.append("Unclassified")

        df_for_prediction["category"] = predictions

        category_counts = df_for_prediction["category"].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(df_for_prediction) * 100
            logger.info(f"  {category}: {count} entities ({percentage:.1f}%)")

        return df_for_prediction

    def prepare_normalized_dataset(
        self, final_with_metrics_ml: pd.DataFrame
    ) -> Tuple[pd.DataFrame, list]:
        embedding_columns = [
            column
            for column in final_with_metrics_ml.columns
            if column not in EMBEDDING_FEATURE_COLUMNS_TO_EXCLUDE
            and final_with_metrics_ml[column].dtype in ["float64", "int64", "bool"]
        ]

        numeric_columns = [
            column
            for column in embedding_columns
            if final_with_metrics_ml[column].dtype in ["float64", "int64"]
        ]

        final_with_metrics_ml_norm = final_with_metrics_ml.copy()

        scaler = StandardScaler()
        final_with_metrics_ml_norm[numeric_columns] = scaler.fit_transform(
            final_with_metrics_ml[numeric_columns]
        )

        return final_with_metrics_ml_norm, embedding_columns

    def find_benchmark_and_calculate_kpi(
        self,
        entity_data_norm: pd.Series,
        entity_data_original: pd.Series,
        high_performer_data_norm: pd.DataFrame,
        high_performer_data_original: pd.DataFrame,
        embedding_columns: list,
        comparison_metrics: list,
    ) -> list:
        kpi_recommendations = []
        entity_id = entity_data_original["entity_id"]
        entity_category = entity_data_original["category"]
        entity_segment = entity_data_original["entity_segment"]
        entity_period = entity_data_original["preferred_period"]

        same_segment_mask = (
            high_performer_data_original["entity_segment"] == entity_segment
        )
        same_segment_hp_norm = high_performer_data_norm[same_segment_mask]
        same_segment_hp_original = high_performer_data_original[same_segment_mask]

        used_fallback = False
        if len(same_segment_hp_original) == 0:
            same_segment_hp_norm = high_performer_data_norm
            same_segment_hp_original = high_performer_data_original
            used_fallback = True

        if len(same_segment_hp_original) == 0:
            return kpi_recommendations

        if entity_category == "C":
            higher_performers_mask = (
                same_segment_hp_original["revenue_per_month"]
                > entity_data_original["revenue_per_month"] * 1.1
            )
            higher_performers_norm = same_segment_hp_norm[higher_performers_mask]
            higher_performers_original = same_segment_hp_original[higher_performers_mask]

            if len(higher_performers_original) > 0:
                target_group_norm = higher_performers_norm
                target_group_original = higher_performers_original
                search_type = "higher_performer"
            else:
                for metric in comparison_metrics:
                    if metric in entity_data_original.index and not pd.isna(
                        entity_data_original[metric]
                    ):
                        kpi_recommendations.append(
                            {
                                "entity_id": entity_id,
                                "entity_category": entity_category,
                                "entity_period": entity_period,
                                "benchmark_id": entity_id,
                                "benchmark_period": entity_period,
                                "entity_segment": entity_segment,
                                "cosine_similarity": 1.0,
                                "metric": metric,
                                "entity_value": entity_data_original[metric],
                                "benchmark_value": entity_data_original[metric],
                                "difference": 0,
                                "improvement_direction": "maintain",
                                "improvement_category": "top_performer",
                                "target_value": entity_data_original[metric],
                                "search_type": "top_performer",
                            }
                        )
                return kpi_recommendations
        else:
            target_group_norm = same_segment_hp_norm
            target_group_original = same_segment_hp_original
            search_type = "same_level"

        if used_fallback:
            search_type = f"fallback_{search_type}"

        entity_vector = entity_data_norm[embedding_columns].values.reshape(1, -1)

        best_similarity = -1
        best_hp_idx = None

        for hp_idx, hp_norm in target_group_norm.iterrows():
            hp_vector = hp_norm[embedding_columns].values.reshape(1, -1)
            similarity = cosine_similarity(entity_vector, hp_vector)[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_hp_idx = hp_idx

        if best_hp_idx is not None:
            best_hp_original = target_group_original.loc[best_hp_idx]
            hp_period = best_hp_original["preferred_period"]
            hp_id = best_hp_original["entity_id"]

            for metric in comparison_metrics:
                if (
                    metric in entity_data_original.index
                    and metric in best_hp_original.index
                ):
                    entity_val = entity_data_original[metric]
                    hp_val = best_hp_original[metric]

                    if not pd.isna(entity_val) and not pd.isna(hp_val):
                        difference = entity_val - hp_val

                        if metric in ALWAYS_INCREASE:
                            improvement_category = "always_increase"
                            improvement_direction = "increase"
                            target_value = hp_val if difference < 0 else entity_val

                        elif metric in ALWAYS_DECREASE:
                            improvement_category = "always_decrease"
                            improvement_direction = "decrease"
                            target_value = hp_val if difference > 0 else entity_val

                        elif metric in FOLLOW_BENCHMARK:
                            improvement_category = "follow_benchmark"
                            improvement_direction = (
                                "increase" if difference < 0 else "decrease"
                            )
                            target_value = hp_val

                        else:
                            improvement_category = "informative"
                            improvement_direction = "maintain"
                            target_value = entity_val

                        kpi_recommendations.append(
                            {
                                "entity_id": entity_id,
                                "entity_category": entity_category,
                                "entity_period": entity_period,
                                "benchmark_id": hp_id,
                                "benchmark_period": hp_period,
                                "entity_segment": entity_segment,
                                "cosine_similarity": round(best_similarity, 3),
                                "metric": metric,
                                "entity_value": entity_val,
                                "benchmark_value": hp_val,
                                "difference": difference,
                                "improvement_direction": improvement_direction,
                                "improvement_category": improvement_category,
                                "target_value": target_value,
                                "search_type": search_type,
                            }
                        )

        return kpi_recommendations

    def calculate_kpi_improvement(self, row: pd.Series) -> float:
        improvement = 0

        if row["benchmark_value"] == 0:
            gap_percent = 100 if row["difference"] != 0 else 0
        else:
            gap_percent = (abs(row["difference"]) / abs(row["benchmark_value"])) * 100

        if gap_percent < 5:
            return 0

        if row["improvement_category"] in ["informative", "top_performer"]:
            return 0

        gap = row["difference"]

        if row["improvement_category"] == "always_increase":
            if gap < 0:
                if gap_percent > 70:
                    improvement = abs(gap) * 0.15
                elif gap_percent > 50:
                    improvement = abs(gap) * 0.25
                elif gap_percent > 25:
                    improvement = abs(gap) * 0.35
                else:
                    improvement = abs(gap) * 0.45
            else:
                return 0

        elif row["improvement_category"] == "always_decrease":
            if gap > 0:
                if gap_percent > 70:
                    improvement = -abs(gap) * 0.15
                elif gap_percent > 50:
                    improvement = -abs(gap) * 0.25
                elif gap_percent > 25:
                    improvement = -abs(gap) * 0.35
                else:
                    improvement = -abs(gap) * 0.45
            else:
                return 0

        elif row["improvement_category"] == "follow_benchmark":
            if gap < 0:
                if gap_percent > 70:
                    improvement = abs(gap) * 0.15
                elif gap_percent > 50:
                    improvement = abs(gap) * 0.25
                elif gap_percent > 25:
                    improvement = abs(gap) * 0.35
                else:
                    improvement = abs(gap) * 0.45
            else:
                if gap_percent > 70:
                    improvement = -abs(gap) * 0.15
                elif gap_percent > 50:
                    improvement = -abs(gap) * 0.25
                elif gap_percent > 25:
                    improvement = -abs(gap) * 0.35
                else:
                    improvement = -abs(gap) * 0.45

        return improvement

    def calculate_rounded_improvements(self, row: pd.Series) -> Tuple[int, int]:
        if row["kpi_improvement"] == 0:
            return 0, 0

        current = row["entity_value"]
        improvement = row["kpi_improvement"]

        if current != 0:
            percent_kpi = (improvement / current) * 100
        else:
            percent_kpi = 100 if improvement > 0 else 0

        metric = row["metric"]

        if metric in [
            "transactions_per_hour",
            "total_bonuses",
            "successful_bonuses",
            "avg_hours_per_day",
            "revenue_per_hour",
        ]:
            amount_kpi = round(improvement)
        elif metric in [
            "success_rate",
            "abandonment_rate",
            "bonus_success_rate",
            "small_activity_rate",
        ]:
            amount_kpi = round(improvement, 2)
        elif metric in ["avg_processing_time_sec"]:
            amount_kpi = round(improvement / 30) * 30
        elif metric in ["avg_activity_volume"]:
            if abs(improvement) <= 0.5:
                amount_kpi = 0.5 if improvement > 0 else -0.5
            elif abs(improvement) <= 1.0:
                amount_kpi = 1.0 if improvement > 0 else -1.0
            else:
                amount_kpi = round(improvement)
        else:
            amount_kpi = round(improvement, 1)

        percent_kpi = round(percent_kpi)

        return percent_kpi, amount_kpi

    def calculate_kpi_for_all_entities(
        self, final_with_metrics_ml: pd.DataFrame
    ) -> pd.DataFrame:
        final_with_metrics_ml_norm, embedding_columns = self.prepare_normalized_dataset(
            final_with_metrics_ml
        )

        high_performer_mask = final_with_metrics_ml["category"] == "C"
        high_performers_original = final_with_metrics_ml[high_performer_mask]
        high_performers_norm = final_with_metrics_ml_norm[high_performer_mask]

        all_kpi_recommendations = []

        for entity_idx in final_with_metrics_ml.index:
            entity_original = final_with_metrics_ml.loc[entity_idx]
            entity_norm = final_with_metrics_ml_norm.loc[entity_idx]
            recommendations = self.find_benchmark_and_calculate_kpi(
                entity_data_norm=entity_norm,
                entity_data_original=entity_original,
                high_performer_data_norm=high_performers_norm,
                high_performer_data_original=high_performers_original,
                embedding_columns=embedding_columns,
                comparison_metrics=COMPARISON_METRICS,
            )
            all_kpi_recommendations.extend(recommendations)

        if not all_kpi_recommendations:
            return pd.DataFrame()

        kpi_df = pd.DataFrame(all_kpi_recommendations)

        kpi_df["gap_percent"] = kpi_df.apply(
            lambda row: (abs(row["difference"]) / abs(row["benchmark_value"])) * 100
            if row["benchmark_value"] != 0
            else (100 if row["difference"] != 0 else 0),
            axis=1,
        )

        kpi_df["kpi_improvement"] = kpi_df.apply(self.calculate_kpi_improvement, axis=1)
        kpi_df["kpi_target"] = kpi_df["entity_value"] + kpi_df["kpi_improvement"]

        kpi_df[["percent_kpi", "amount_kpi"]] = kpi_df.apply(
            lambda row: pd.Series(self.calculate_rounded_improvements(row)), axis=1
        )

        for col in NUMERIC_COLUMNS:
            if col in kpi_df.columns:
                kpi_df[col] = kpi_df[col].round(3)

        return kpi_df

    def run_prediction(self):
        logger.info("Starting Entity Similarity prediction pipeline")
        logger.debug(f"Using models_dir={self.models_dir} output_dir={self.output_dir}")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Loading data from S3")
        df_transactions, df_sessions, df_bonuses = load_data()
        logger.info(f"Loaded: transactions={df_transactions.shape}, sessions={df_sessions.shape}, bonuses={df_bonuses.shape}")

        logger.info("Cleaning transactions data")
        completed_transactions, abandoned_transactions, df_transactions_clean = clean_transactions_data(df_transactions)

        logger.info("Saving intermediate files")
        os.makedirs(DATA_DIR, exist_ok=True)
        completed_transactions.to_parquet(f"{DATA_DIR}/completed_transactions.parquet", index=False)
        abandoned_transactions.to_parquet(f"{DATA_DIR}/abandoned_transactions.parquet", index=False)
        df_transactions_clean.to_parquet(f"{DATA_DIR}/df_transactions_clean.parquet", index=False)
        df_sessions.to_parquet(f"{DATA_DIR}/df_sessions.parquet", index=False)
        df_bonuses.to_parquet(f"{DATA_DIR}/df_bonuses.parquet", index=False)

        logger.info("Building features for prediction")
        features_path = create_features(
            df_sessions_path=f"{DATA_DIR}/df_sessions.parquet",
            completed_transactions_path=f"{DATA_DIR}/completed_transactions.parquet",
            abandoned_transactions_path=f"{DATA_DIR}/abandoned_transactions.parquet",
            df_transactions_clean_path=f"{DATA_DIR}/df_transactions_clean.parquet",
            df_bonuses_path=f"{DATA_DIR}/df_bonuses.parquet",
        )
        final_with_metrics_ml = pd.read_parquet(features_path)
        logger.info(
            f"Features ready: rows={len(final_with_metrics_ml)} cols={len(final_with_metrics_ml.columns)}"
        )

        logger.info("Loading production models from S3")
        models_dict = self.download_models_from_s3()
        logger.info(f"Loaded {len(models_dict)} production models")

        logger.info("Predicting entity categories")
        final_with_metrics_ml_predicted = self.predict_categories_with_models(
            final_with_metrics_ml=final_with_metrics_ml,
            models=models_dict,
        )
        logger.info(
            f"Predictions complete: rows={len(final_with_metrics_ml_predicted)}"
        )

        unique_id = uuid.uuid4().hex[:8]
        prediction_path = os.path.join(
            self.output_dir, f"entities_with_predictions_{unique_id}.csv"
        )
        final_with_metrics_ml_predicted.to_csv(prediction_path, index=False)
        logger.info(f"Saved dataset with predictions: {prediction_path}")

        entities_before = len(final_with_metrics_ml_predicted)
        final_with_metrics_ml_filtered = final_with_metrics_ml_predicted[
            final_with_metrics_ml_predicted["category"] != "B"
        ].copy()
        entities_after = len(final_with_metrics_ml_filtered)

        high_risk_count = entities_before - entities_after
        logger.info(f"Removed {high_risk_count} high risk entities (category B)")
        logger.info(f"Remaining {entities_after} entities for KPI calculation")

        if entities_after == 0:
            logger.error("No entities for KPI calculation after filtering")
            return prediction_path, None

        filtered_path = os.path.join(
            self.output_dir, f"entities_without_high_risk_{unique_id}.csv"
        )
        final_with_metrics_ml_filtered.to_csv(filtered_path, index=False)
        logger.info(f"Saved filtered dataset: {filtered_path}")

        logger.info("Calculating KPI recommendations")
        kpi_df = self.calculate_kpi_for_all_entities(final_with_metrics_ml_filtered)

        kpi_path = None

        if kpi_df.empty:
            logger.error("No KPI data after filtering")
            return prediction_path, None

        kpi_path = os.path.join(
            self.output_dir, f"entity_kpi_recommendations_{unique_id}.csv"
        )
        kpi_df.to_csv(kpi_path, index=False)
        logger.info(f"Saved KPI recommendations: {kpi_path}")

        logger.info("Prediction pipeline completed successfully")

        return prediction_path, kpi_path