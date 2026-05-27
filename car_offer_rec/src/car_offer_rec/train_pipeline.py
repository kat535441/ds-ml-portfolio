import sys
import argparse
from loguru import logger

from car_offer_rec.config import (
    CLEARML_TRACKER_TASK_ID,
    AZURE_CONTAINER_NAME,
    AZURE_OUTPUT_URI,
    CLEARML_API_HOST,
    CLEARML_WEB_HOST,
    CLEARML_FILES_HOST,
    CLEARML_ACCESS_KEY,
    CLEARML_SECRET_KEY,
    TARGET_METRIC,
    PROJECT,
    BUILD_CONF_ID,
    EXEC_QUEUE,
)

from car_offer_rec.dataset import (
    run_data_pipeline,
    preprocess_data,
    _preprocess_core,
)

from car_offer_rec.features import (
    create_features_for_pipeline,
    _ensure_float64_numeric,
    compute_cutoff_date_for_each_user,
    compute_target_on_historical,
    take_only_historical_data,
    generate_temporal_features_train,
    generate_geo_features_train,
    generate_user_trip_aggregates_train,
    generate_offer_specific_features_train,
    agg_user_train,
    generate_offer_features_train,
    generate_aggregates_for_offer_user,
    generate_and_save_embeddings,
    final_feature_join,
    data_join_for_pipeline,
    feature_engineering_for_pipeline,
)

from car_offer_rec.modeling.train import (
    train_and_publish_all,
    promote_best_models_to_production,
    _publish_artifact,
    split_data,
    train_model,
    train_full_model,
    report_metrics,
    visualize_metrics,
    publish_model,
    publish_full_model,
    publish_geo_models,
    publish_offer_features,
    publish_als_data,
    publish_pca_models,
    publish_model_metadata,
)

from ml_common.utils import configure_clearml, get_best_builds_by_series

configure_clearml(
    clearml_api_host=CLEARML_API_HOST,
    clearml_web_host=CLEARML_WEB_HOST,
    clearml_files_host=CLEARML_FILES_HOST,
    clearml_access_key=CLEARML_ACCESS_KEY,
    clearml_secret_key=CLEARML_SECRET_KEY,
)

from clearml import PipelineController, Task


def build_pipeline(
    project_name: str,
    build_conf_id: str,
    build_num: int,
    execution_queue: str,
):
    """
    Builds the ClearML PipelineController for Car Offer Recommendation training.

    Pipeline steps:
    1. data_ingestion - Load datasets from ClearML
    2. preprocess_data - Join users and trips, clean data
    3. create_features - Feature engineering (temporal split, geo, ALS, PCA, etc.)
    4. train_and_publish - Train models and publish to ClearML Model Registry
    5. promote_best_models - Promote best model to production
    """
    logger.info(
        f"Building pipeline for project: {project_name}, "
        f"build_conf_id: {build_conf_id}, "
        f"build_num: {build_num}, "
        f"execution_queue: {execution_queue}"
    )

    if not project_name:
        logger.error("Project name is required")
        raise ValueError("Project name is required")

    if not build_conf_id:
        logger.error("Build conf id is required")
        raise ValueError("Build conf id is required")

    if build_num is None:
        logger.error("Build num is required")
        raise ValueError("Build num is required")

    if not execution_queue:
        logger.error("Execution queue is required")
        raise ValueError("Execution queue is required")

    pipe = PipelineController(
        project=f"{project_name}",
        name=f"{build_conf_id}",
        version=f"0.0.{build_num}",
        add_pipeline_tags=True,
        abort_on_failure=True
    )
    pipe.set_default_execution_queue(execution_queue)

    controller_task = Task.current_task()
    if not controller_task:
        raise RuntimeError("ClearML Task must be initialized before calling build_pipeline")

    # Step 1: Data Ingestion
    pipe.add_function_step(
        name="data_ingestion",
        function=run_data_pipeline,
        function_kwargs=dict(
            pipeline_type="train"
        ),
        function_return=[
            "hex_path",
            "trips_path",
            "users_path",
            "offers_path",
        ]
    )

    # Step 2: Preprocess Data (join users + trips)
    pipe.add_function_step(
        name="preprocess_data",
        parents=["data_ingestion"],
        function=preprocess_data,
        function_kwargs=dict(
            trips_path="${data_ingestion.trips_path}",
            users_path="${data_ingestion.users_path}",
        ),
        function_return=["preprocessed_path"],
        helper_functions=[_preprocess_core],
    )

    # Step 3: Create Features (with temporal split — no data leaks)
    pipe.add_function_step(
        name="create_features",
        parents=["data_ingestion", "preprocess_data"],
        function=create_features_for_pipeline,
        function_kwargs=dict(
            preprocessed_path="${preprocess_data.preprocessed_path}",
            hex_path="${data_ingestion.hex_path}",
            offer_path="${data_ingestion.offers_path}",
            kmeans_n_clusters=10,
            pca_n_components=1,
            als_factors=32,
            als_iterations=15,
            als_reg=0.1,
        ),
        function_return=["final_features_path"],
        helper_functions=[
            _ensure_float64_numeric,
            compute_cutoff_date_for_each_user,
            compute_target_on_historical,
            take_only_historical_data,
            generate_temporal_features_train,
            generate_geo_features_train,
            generate_user_trip_aggregates_train,
            generate_offer_specific_features_train,
            agg_user_train,
            generate_offer_features_train,
            generate_aggregates_for_offer_user,
            generate_and_save_embeddings,
            final_feature_join,
            data_join_for_pipeline,
            feature_engineering_for_pipeline,
        ]
    )

    # Step 4: Train and Publish Models
    pipe.add_function_step(
        name="train_and_publish",
        parents=["create_features"],
        function=train_and_publish_all,
        function_kwargs=dict(
            final_features_path="${create_features.final_features_path}",
            build_num=build_num,
            azure_output_uri=AZURE_OUTPUT_URI,
            controller_task_id=controller_task.id,
            tracker_task_id=CLEARML_TRACKER_TASK_ID,
            test_size=0.2,
            cb_iterations=500,
            cb_learning_rate=0.05,
            cb_early_stopping_rounds=50,
            cb_depth=6,
        ),
        function_return=["results"],
        helper_functions=[
            _publish_artifact,
            split_data,
            train_model,
            train_full_model,
            report_metrics,
            visualize_metrics,
            publish_model,
            publish_full_model,
            publish_geo_models,
            publish_offer_features,
            publish_als_data,
            publish_pca_models,
            publish_model_metadata,
        ]
    )

    # Step 5: Promote Best Models to Production
    pipe.add_function_step(
        name="promote_best_models",
        parents=["train_and_publish"],
        function=promote_best_models_to_production,
        function_kwargs=dict(
            tracker_task_id=CLEARML_TRACKER_TASK_ID,
            metric_name=TARGET_METRIC,
        ),
        function_return=["promoted_model_ids"],
        helper_functions=[get_best_builds_by_series],
    )

    return pipe


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run Car Offer Recommendation training pipeline"
    )

    parser.add_argument(
        "--buildNum",
        type=int,
        required=True,
        help="Build number"
    )

    args = parser.parse_args()

    BUILD_NUM = args.buildNum

    logger.info("Starting pipeline:")
    logger.info(f"   Project: {PROJECT}")
    logger.info(f"   BuildConfId: {BUILD_CONF_ID}")
    logger.info(f"   BuildNum: {BUILD_NUM}")
    logger.info(f"   Queue: {EXEC_QUEUE}")
    logger.info(f"   Azure Container: {AZURE_CONTAINER_NAME}")

    try:
        pipe = build_pipeline(
            project_name=PROJECT,
            build_conf_id=BUILD_CONF_ID,
            build_num=BUILD_NUM,
            execution_queue=EXEC_QUEUE,
        )

        logger.info("Pipeline created successfully!")

        pipe.start_locally(run_pipeline_steps_locally=True)
        pipe.wait()

        logger.info("All pipeline steps completed!")

        controller_task = pipe.task

        if controller_task:
            status = controller_task.get_status()
            logger.info(f"Pipeline status: {status}")

            if status not in ("completed", "published", "closed"):
                logger.error(f"Pipeline failed. Status: {status}")
                try:
                    pipe.stop(mark_failed=True, mark_aborted=False)
                except Exception as e:
                    logger.warning(f"Failed to stop pipeline: {e}")

                sys.exit(1)
            else:
                logger.info("Pipeline completed successfully!")
        else:
            logger.warning("Failed to get controller task")

    except Exception as e:
        logger.error(f"Critical error during pipeline execution: {e}")
        raise RuntimeError(f"[pipeline execution] error: {e}")