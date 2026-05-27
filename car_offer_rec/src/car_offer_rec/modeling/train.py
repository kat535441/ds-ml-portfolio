import pandas as pd
import numpy as np
import joblib
import os
import re
import pickle
from pathlib import Path

from clearml import Logger, Task, OutputModel, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from catboost import CatBoostClassifier, Pool
from loguru import logger
from typing import Dict

from ml_common.utils.clearml_functions import get_best_builds_by_series
from car_offer_rec.config import (
    AZURE_OUTPUT_URI,
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    ARTIFACTS_DIR,
    MODEL_ARTIFACT_NAMES,
)


# =============================================================================
# UNIFIED PUBLISH HELPER
# =============================================================================

def _publish_artifact(
    artifact_path: str,
    artifact_name: str,
    model_name: str,
    tags: list,
    framework: str = "custom",
    azure_output_uri: str = None,
) -> str:
    """
    Unified helper for publishing artifacts to ClearML Model Registry.

    Args:
        artifact_path: Path to the artifact file
        artifact_name: Human-readable name for logging
        model_name: Name for the model in ClearML Model Registry (used for querying)
        tags: List of tags for the model
        framework: Framework name (e.g., "catboost", "custom")
        azure_output_uri: Azure blob URI for upload (defaults to AZURE_OUTPUT_URI)

    Returns:
        Model ID if successful, None otherwise
    """
    task = Task.current_task()
    log = Logger.current_logger()

    if azure_output_uri is None:
        azure_output_uri = AZURE_OUTPUT_URI

    out = OutputModel(task=task, framework=framework, name=model_name, tags=tags)

    try:
        out.update_weights(weights_filename=artifact_path, upload_uri=azure_output_uri)
        out.publish()
        log.report_text(f"[publish] {artifact_name} uploaded to Azure & published as '{model_name}'")
        return out.id
    except Exception as e:
        log.report_text(f"[warn] Azure upload failed for {artifact_name}: {e} — kept as local artifact")
        return None


# =============================================================================
# SPLIT DATA
# =============================================================================

def split_data(
    clean_df,
    test_size,
):
    """Splits clean_df into train/test by USERS (no leakage through duplicates)."""

    # Remove target and user_id from features
    columns_to_drop = ['user_F_retained_by_offer', 'user_id']
    X = clean_df.drop(columns=[c for c in columns_to_drop if c in clean_df.columns], axis=1)
    y = clean_df['user_F_retained_by_offer']

    if test_size and test_size > 0:
        # Split by unique users
        unique_users = clean_df['user_id'].unique()
        train_users, test_users = train_test_split(
            unique_users, test_size=test_size, random_state=42, stratify=None
        )

        train_mask = clean_df['user_id'].isin(train_users).values
        test_mask = clean_df['user_id'].isin(test_users).values

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    categorical_features_indices = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            categorical_features_indices.append(col)
        if col in ['user_F_mode_trip_weekday', 'user_F_payment_method_id', 'user_F_client_type',
                    'user_F_mode_trip_hour', 'user_F_mode_trip_hour_bucket',
                    'user_F_mode_used_midday', 'user_F_is_churn_risk', 'user_F_mode_origin_area_id',
                    'user_F_mode_destination_area_id', 'user_F_mode_trips_cross_area', 'user_F_mode_distance_category',
                    'user_F_mode_hex_origin', 'user_F_mode_hex_destination', 'user_F_has_used_any_discount', 'user_F_has_used_any_offer',
                    'user_F_has_any_offer_success', 'offer_F_type_id', 'offer_F_car_category']:
            if col not in categorical_features_indices:
                categorical_features_indices.append(col)

    return X_train, X_test, y_train, y_test, categorical_features_indices


# =============================================================================
# TRAIN MODEL (test split)
# =============================================================================

def train_model(
    buildNum,
    split_df,
    cb_iterations: int,
    cb_learning_rate: float,
    cb_early_stopping_rounds: int,
    cb_depth: int,
):
    task = Task.current_task()
    log = Logger.current_logger()

    X_train, X_test, y_train, y_test, categorical_features_indices = split_df

    for col in categorical_features_indices:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0

    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=cb_iterations,
        random_seed=42,
        learning_rate=cb_learning_rate,
        early_stopping_rounds=cb_early_stopping_rounds,
        verbose=100,
        scale_pos_weight=scale_pos_weight_value,
        depth=cb_depth,
        allow_writing_files=False,
    )

    task.connect(
        {
            "cb_iterations": cb_iterations,
            "cb_learning_rate": cb_learning_rate,
            "cb_early_stopping_rounds": cb_early_stopping_rounds,
            "cb_depth": cb_depth,
            "scale_pos_weight_value": scale_pos_weight_value
        },
        name="catboost_hyparams"
    )

    model.fit(
        Pool(X_train, y_train, cat_features=categorical_features_indices),
        eval_set=Pool(X_test, y_test, cat_features=categorical_features_indices),
        plot=False
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    feature_importances = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=categorical_features_indices))
    feature_names = X_test.columns

    metrics_df = pd.DataFrame({
        'metric': ['AUC', 'Precision', 'Recall', 'F1', 'Accuracy'],
        'value': [auc, precision, recall, f1, class_report['accuracy']],
        'description': [
            'Area Under ROC Curve', 'Precision score',
            'Recall score', 'F1-score', 'Overall accuracy'
        ]
    })

    class_metrics = []
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            class_metrics.append({
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })
    class_metrics_df = pd.DataFrame(class_metrics)

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    log.report_table(title="metrics_df", series="train_model", iteration=0, table_plot=metrics_df)
    log.report_table(title="class_metrics_df", series="train_model", iteration=0, table_plot=class_metrics_df)
    log.report_table(title="feature_importance_df", series="train_model", iteration=0, table_plot=feature_importance_df)

    params = dict(
        loss_function='Logloss', eval_metric='AUC', iterations=cb_iterations,
        random_seed=42, learning_rate=cb_learning_rate,
        early_stopping_rounds=cb_early_stopping_rounds,
        scale_pos_weight=float(scale_pos_weight_value),
    )
    schema = dict(
        cat_features_idx=[X_train.columns.get_loc(col) for col in categorical_features_indices if col in X_train.columns],
        n_features=int(X_train.shape[1]),
    )
    task.connect(params, name="catboost")
    task.connect(schema, name="data_schema")

    # Report metrics on single chart "metrics" with each metric as a series
    log.report_scalar("metrics", "AUC", value=float(auc), iteration=buildNum)
    log.report_scalar("metrics", "Precision", value=float(precision), iteration=buildNum)
    log.report_scalar("metrics", "Recall", value=float(recall), iteration=buildNum)
    log.report_scalar("metrics", "F1", value=float(f1), iteration=buildNum)
    log.report_scalar("metrics", "Accuracy", value=float(class_report['accuracy']), iteration=buildNum)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(ARTIFACTS_DIR / "metrics_df.csv", index=False)
    class_metrics_df.to_csv(ARTIFACTS_DIR / "class_metrics_df.csv", index=False)
    feature_importance_df.to_csv(ARTIFACTS_DIR / "feature_importance_df.csv", index=False)

    task.upload_artifact("train__metrics_df_csv", str(ARTIFACTS_DIR / "metrics_df.csv"))
    task.upload_artifact("train__class_metrics_df_csv", str(ARTIFACTS_DIR / "class_metrics_df.csv"))
    task.upload_artifact("train__feature_importances_df_csv", str(ARTIFACTS_DIR / "feature_importance_df.csv"))

    model_path = str(ARTIFACTS_DIR / f"model_buildNum-{buildNum}.cbm")
    model.save_model(model_path)

    return {
        "metrics_df": metrics_df,
        "class_metrics_df": class_metrics_df,
        "feature_importance_df": feature_importance_df,
        "model_path": model_path,
    }


# =============================================================================
# REPORT METRICS
# =============================================================================

def report_metrics(train_out, buildNum, controller_id: str, step_name: str, chart_title: str = "metrics", tracker_task_id: str | None = None):
    """
    Report metrics to ClearML loggers.

    Creates a single chart with title=chart_title and each metric as a separate series.
    This allows get_best_builds_by_series to find metrics by metric name (e.g., "AUC").

    Structure: scalars[chart_title][metric_name] = {x: [builds], y: [values]}
    """
    task = Task.current_task()

    s = str(buildNum).strip().strip('"\'')
    try:
        bnum = int(s)
    except Exception:
        parts = re.findall(r"\d+", s)
        bnum = int(parts[-1])

    metrics_df = train_out["metrics_df"]

    df: pd.DataFrame = metrics_df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df[["metric", "value"]]
    df["metric"] = df["metric"].astype(str)
    df["value"] = df["value"].astype(float)

    log = Logger.current_logger()
    # Report to current task: title=chart_title, series=metric_name
    for _, r in df.iterrows():
        metric_name = r["metric"]
        log.report_scalar(f"{chart_title}_{step_name}", metric_name, iteration=bnum, value=float(r["value"]))

    controller = Task.get_task(task_id=controller_id)
    ctrl_logger = controller.get_logger()
    # Report to controller: title=chart_title, series=metric_name
    for _, r in df.iterrows():
        metric_name = r["metric"]
        ctrl_logger.report_scalar(chart_title, metric_name, iteration=bnum, value=float(r["value"]))

    if tracker_task_id:
        trk_logger = Task.get_task(task_id=tracker_task_id).get_logger()
        # Report to tracker: title=chart_title, series=metric_name
        # This is what get_best_builds_by_series reads
        for _, r in df.iterrows():
            metric_name = r["metric"]
            trk_logger.report_scalar(chart_title, metric_name, iteration=bnum, value=float(r["value"]))


# =============================================================================
# TRAIN FULL MODEL
# =============================================================================

def train_full_model(
    buildNum,
    split_df,
    cb_iterations: int,
    cb_learning_rate: float,
    cb_depth: int,
):
    task = Task.current_task()
    log = Logger.current_logger()

    X, _, y, _, categorical_features = split_df

    if len(categorical_features) > 0 and isinstance(categorical_features[0], int):
        cat_idx = [i for i in categorical_features if 0 <= i < X.shape[1]]
        cat_cols = [X.columns[i] for i in cat_idx]
    else:
        cat_cols = [c for c in categorical_features if c in X.columns]
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    for col in cat_cols:
        X[col] = X[col].astype(str)

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight_value = (neg_count / pos_count) if pos_count > 0 else 1.0

    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=cb_iterations,
        random_seed=42,
        learning_rate=cb_learning_rate,
        verbose=100,
        scale_pos_weight=scale_pos_weight_value,
        depth=cb_depth,
        allow_writing_files=False,
    )

    task.connect(
        {
            "cb_iterations": cb_iterations,
            "cb_learning_rate": cb_learning_rate,
            "cb_depth": cb_depth,
            "scale_pos_weight_value": float(scale_pos_weight_value)
        },
        name="catboost_hyparams_full"
    )

    model.fit(Pool(X, y, cat_features=cat_idx))

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    metrics_df = pd.DataFrame({
        'metric': ['AUC', 'Precision', 'Recall', 'F1'],
        'value': [auc, precision, recall, f1],
        'description': ['Area Under ROC Curve', 'Precision score', 'Recall score', 'F1-score']
    })

    log.report_table(title="full_metrics_df", series="train_full_model", iteration=0, table_plot=metrics_df)
    # Report metrics on single chart "full_model_metrics" with each metric as a series
    log.report_scalar("full_model_metrics", "AUC", value=float(auc), iteration=buildNum)
    log.report_scalar("full_model_metrics", "Precision", value=float(precision), iteration=buildNum)
    log.report_scalar("full_model_metrics", "Recall", value=float(recall), iteration=buildNum)
    log.report_scalar("full_model_metrics", "F1", value=float(f1), iteration=buildNum)

    params = dict(
        loss_function='Logloss', eval_metric='AUC', iterations=cb_iterations,
        random_seed=42, learning_rate=cb_learning_rate, verbose=100,
        scale_pos_weight=float(scale_pos_weight_value),
    )
    schema = dict(cat_features_idx=cat_idx, n_features=int(X.shape[1]))
    task.connect(params, name="catboost")
    task.connect(schema, name="data_schema")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(ARTIFACTS_DIR / "full_metrics_df.csv", index=False)
    task.upload_artifact("train__full_metrics_df_csv", str(ARTIFACTS_DIR / "full_metrics_df.csv"))

    model_path = str(ARTIFACTS_DIR / f"full_model_buildNum-{buildNum}.cbm")
    model.save_model(model_path)

    return {
        "metrics_df": metrics_df,
        "full_model_path": model_path,
    }


# =============================================================================
# VISUALIZE METRICS
# =============================================================================

def visualize_metrics(train_out):
    import matplotlib.pyplot as plt

    log = Logger.current_logger()

    metrics_df = pd.DataFrame(train_out["metrics_df"])
    class_metrics_df = pd.DataFrame(train_out["class_metrics_df"])
    feature_importance_df = pd.DataFrame(train_out["feature_importance_df"])

    plt.figure(figsize=(10, 6))
    feature_importance_df.sort_values("importance", ascending=True).plot(
        x="feature", y="importance", kind="barh", legend=False
    )
    plt.title("Feature Importances"); plt.xlabel("Importance"); plt.tight_layout()
    log.report_matplotlib_figure("Feature Importances", "barplot", plt.gcf(), iteration=0)
    plt.close()

    class_only = class_metrics_df[class_metrics_df["class"].isin(['0', '1'])].copy()
    melted = class_only.melt(id_vars="class", value_vars=["precision", "recall", "f1-score"])
    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(["precision", "recall", "f1-score"]):
        vals0 = melted[(melted["class"] == '0') & (melted["variable"] == metric)]["value"].values
        vals1 = melted[(melted["class"] == '1') & (melted["variable"] == metric)]["value"].values
        x = [i - 0.15, i + 0.15]
        y_vals = [vals0[0] if len(vals0) else 0, vals1[0] if len(vals1) else 0]
        plt.bar(x, y_vals, width=0.28)
    plt.xticks([0, 1, 2], ["precision", "recall", "f1-score"])
    plt.ylim(0.9, 1.01)
    plt.title("Class-wise Metrics"); plt.tight_layout()
    log.report_matplotlib_figure("Class-wise Metrics", "bars", plt.gcf(), iteration=0)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(metrics_df["metric"], metrics_df["value"])
    plt.ylim(0.9, 1.01)
    plt.title("Overall Metrics"); plt.tight_layout()
    log.report_matplotlib_figure("Overall Metrics", "bars", plt.gcf(), iteration=0)
    plt.close()


# =============================================================================
# PUBLISH FUNCTIONS
# =============================================================================

def publish_model(train_out, build_num: int = None, azure_output_uri: str = None):
    """Publish the test-split trained model."""
    model_path = train_out.get("model_path")
    tags = ["CarOfferRec", "CatBoost"]
    if build_num is not None:
        tags.append(f"build_{build_num}")

    return _publish_artifact(
        artifact_path=model_path,
        artifact_name="model",
        model_name="car_offer_model",
        tags=tags,
        framework="catboost",
        azure_output_uri=azure_output_uri,
    )


def publish_full_model(train_full_out, build_num: int = None, azure_output_uri: str = None):
    """Publish the full-data trained model."""
    model_path = train_full_out.get("full_model_path")
    tags = ["CarOfferRec", "CatBoost", "FullTrain"]
    if build_num is not None:
        tags.append(f"build_{build_num}")

    return _publish_artifact(
        artifact_path=model_path,
        artifact_name="full_model",
        model_name="car_offer_full_model",
        tags=tags,
        framework="catboost",
        azure_output_uri=azure_output_uri,
    )


def publish_geo_models(geo_models, build_num: int, azure_output_uri: str = None):
    """Publish geo models (KMeans, cKDTree, hexagons)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    geo_models_path = ARTIFACTS_DIR / f"geo_models_buildNum-{build_num}.pkl"
    joblib.dump(geo_models, geo_models_path)

    return _publish_artifact(
        artifact_path=str(geo_models_path),
        artifact_name="geo_models",
        model_name="car_offer_geo_models",
        tags=["CarOfferRec", "GeoModels", f"build_{build_num}"],
        framework="custom",
        azure_output_uri=azure_output_uri,
    )


def publish_offer_features(models_pkl, build_num: int, azure_output_uri: str = None):
    """Publish offer feature aggregates."""
    offer_features = models_pkl['offer_features']

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    offer_features_path = ARTIFACTS_DIR / f"offer_features_buildNum-{build_num}.pkl"
    offer_features.to_pickle(offer_features_path)

    return _publish_artifact(
        artifact_path=str(offer_features_path),
        artifact_name="offer_features",
        model_name="car_offer_offer_features",
        tags=["CarOfferRec", "OfferFeatures", f"build_{build_num}"],
        framework="custom",
        azure_output_uri=azure_output_uri,
    )


def publish_als_data(models_pkl, build_num: int, azure_output_uri: str = None):
    """Publish ALS model and embeddings."""
    als_data = models_pkl['als_data']

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    als_data_path = ARTIFACTS_DIR / f"als_data_buildNum-{build_num}.pkl"
    joblib.dump(als_data, als_data_path)

    return _publish_artifact(
        artifact_path=str(als_data_path),
        artifact_name="als_data",
        model_name="car_offer_als_data",
        tags=["CarOfferRec", "ALSData", f"build_{build_num}"],
        framework="custom",
        azure_output_uri=azure_output_uri,
    )


def publish_pca_models(models_pkl, build_num: int, azure_output_uri: str = None):
    """Publish PCA models for embeddings compression."""
    pca_models = models_pkl['pca_models']

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pca_models_path = ARTIFACTS_DIR / f"pca_models_buildNum-{build_num}.pkl"
    joblib.dump(pca_models, pca_models_path)

    return _publish_artifact(
        artifact_path=str(pca_models_path),
        artifact_name="pca_models",
        model_name="car_offer_pca_models",
        tags=["CarOfferRec", "PCAModels", f"build_{build_num}"],
        framework="custom",
        azure_output_uri=azure_output_uri,
    )


def publish_model_metadata(models_pkl, split_df, build_num: int, azure_output_uri: str = None):
    """Publish model metadata (feature columns, categorical features, offer IDs)."""
    offer_features = models_pkl['offer_features']
    X_train, _, _, _, categorical_features_indices = split_df

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models_metadata_path = ARTIFACTS_DIR / f"models_metadata_buildNum-{build_num}.pkl"

    with open(models_metadata_path, 'wb') as file:
        pickle.dump({
            'feature_columns': list(X_train.columns),
            'categorical_features': categorical_features_indices,
            'all_offer_ids': offer_features['offer_id'].unique()
        }, file)

    return _publish_artifact(
        artifact_path=str(models_metadata_path),
        artifact_name="models_metadata",
        model_name="car_offer_models_metadata",
        tags=["CarOfferRec", "ModelsMetadata", f"build_{build_num}"],
        framework="custom",
        azure_output_uri=azure_output_uri,
    )


# =============================================================================
# TRAIN AND PUBLISH ALL (pipeline entry point)
# =============================================================================

def train_and_publish_all(
    final_features_path: str,
    build_num: int,
    azure_output_uri: str,
    controller_task_id: str,
    tracker_task_id: str,
    test_size: float = 0.2,
    cb_iterations: int = 500,
    cb_learning_rate: float = 0.05,
    cb_early_stopping_rounds: int = 50,
    cb_depth: int = 6,
) -> dict:
    """
    Pipeline entry point: reads features, trains models, publishes artifacts.

    1. Read final features parquet
    2. Split data
    3. Train test model + report metrics
    4. Train full model + report metrics
    5. Visualize metrics
    6. Publish all artifacts (model, full_model, geo_models, offer_features, als_data, pca_models, metadata)
    """
    logger.info(f"train_and_publish_all: reading features from {final_features_path}")
    clean_df = pd.read_parquet(final_features_path)
    clean_df = clean_df.drop_duplicates(subset=['user_id', 'offer_id'])

    # Load feature model artifacts saved by create_features step
    models_pkl_path = DATA_PROCESSED_DIR / "feature_models.pkl"
    geo_models_path = DATA_PROCESSED_DIR / "geo_models.pkl"
    models_pkl = joblib.load(models_pkl_path)
    geo_models = joblib.load(geo_models_path)

    # Split for test model (80/20)
    split_df = split_data(clean_df, test_size=test_size)

    # Train test model
    train_out = train_model(
        buildNum=build_num,
        split_df=split_df,
        cb_iterations=cb_iterations,
        cb_learning_rate=cb_learning_rate,
        cb_early_stopping_rounds=cb_early_stopping_rounds,
        cb_depth=cb_depth,
    )

    # Report metrics
    report_metrics(
        train_out=train_out,
        buildNum=build_num,
        controller_id=controller_task_id,
        step_name="train_model",
        tracker_task_id=tracker_task_id,
    )

    # Split for full model (100% of data)
    split_df_full = split_data(clean_df, test_size=0.0)

    # Train full model on entire dataset
    train_full_out = train_full_model(
        buildNum=build_num,
        split_df=split_df_full,
        cb_iterations=cb_iterations,
        cb_learning_rate=cb_learning_rate,
        cb_depth=cb_depth,
    )

    # Note: Full model metrics are NOT reported to tracker.
    # Only validation (test-split) metrics are used for model selection via get_best_builds_by_series.

    # Visualize
    visualize_metrics(train_out)

    # Publish all artifacts
    publish_model(train_out, build_num, azure_output_uri)
    publish_full_model(train_full_out, build_num, azure_output_uri)
    publish_geo_models(geo_models, build_num, azure_output_uri)
    publish_offer_features(models_pkl, build_num, azure_output_uri)
    publish_als_data(models_pkl, build_num, azure_output_uri)
    publish_pca_models(models_pkl, build_num, azure_output_uri)
    publish_model_metadata(models_pkl, split_df, build_num, azure_output_uri)

    logger.info("train_and_publish_all completed successfully")

    return {"results": "success", "build_num": build_num}


# =============================================================================
# PROMOTE BEST MODELS TO PRODUCTION
# =============================================================================

def promote_best_models_to_production(
    tracker_task_id: str,
    metric_name: str,
) -> Dict[str, str]:
    """
    Finds the best build by metric and promotes all models from that build to production.

    Uses MODEL_ARTIFACT_NAMES from config to identify which models to promote.
    """
    if not tracker_task_id:
        raise ValueError("tracker_task_id is required")
    if not metric_name:
        raise ValueError("metric_name is required")

    logger.info(f"Promoting best models from tracker task {tracker_task_id}")
    logger.info(f"Target metric: {metric_name}")

    # Get best build number from metrics
    best_builds = get_best_builds_by_series(
        tracker_task_id=tracker_task_id,
        metric_name=metric_name,
    )

    if not best_builds:
        logger.warning("No best builds found from metrics")
        return {}

    # Get the best build number (should be same across all series)
    best_build = None
    for series_name, build_info in best_builds.items():
        best_build = build_info["best_build"]
        logger.info(f"Best build from series '{series_name}': {best_build}")
        break  # Use first series

    if best_build is None:
        logger.warning("Could not determine best build number")
        return {}

    promoted_models = {}
    required_tag = f"build_{best_build}"

    # Promote each model from MODEL_ARTIFACT_NAMES
    for model_name in MODEL_ARTIFACT_NAMES:
        try:
            all_models = Model.query_models(model_name=model_name)

            # Filter by build tag
            models = [
                m for m in all_models
                if required_tag in (m.tags or [])
            ]

            logger.info(f"Found {len(all_models)} models with name '{model_name}', {len(models)} with tag '{required_tag}'")

            if not models:
                logger.warning(f"No model found for '{model_name}' with tag '{required_tag}'")
                continue

            target_model = models[0]

            # Demote old production models
            try:
                old_prod_models = Model.query_models(
                    model_name=model_name,
                    tags=["production"]
                )
                for old_model in old_prod_models:
                    if old_model.id != target_model.id:
                        old_tags = set(old_model.tags or [])
                        old_tags.discard("production")
                        old_tags.add("archived")
                        old_model.tags = list(old_tags)
                        logger.info(f"Demoted old production model: {old_model.id}")
            except Exception as e:
                logger.warning(f"Failed to demote old models for '{model_name}': {e}")

            # Promote new model
            new_tags = set(target_model.tags or [])
            new_tags.add("production")
            new_tags.discard("archived")
            target_model.tags = list(new_tags)

            promoted_models[model_name] = target_model.id
            logger.info(f"Promoted model '{model_name}' (id={target_model.id}) to production")

        except Exception as e:
            logger.error(f"Failed to promote model '{model_name}': {e}")

    logger.info(f"Promotion complete. Promoted {len(promoted_models)} models.")
    return promoted_models