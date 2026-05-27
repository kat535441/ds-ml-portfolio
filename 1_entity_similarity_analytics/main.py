# main.py
from loguru import logger
import os

from config import DATA_DIR, OUTPUT_DIR, MODELS_DIR, S3_BUCKET_NAME
from dataset import load_data, clean_transactions_data
from features import create_features
from train import train_all_models

def main():
    logger.info("=== Starting Entity Similarity Analytics pipeline ===")
    
    # Step 1: Load data from S3
    logger.info("Step 1: Loading data from S3...")
    df_transactions, df_sessions, df_bonuses = load_data()
    logger.info(f"Loaded: transactions={df_transactions.shape}, sessions={df_sessions.shape}, bonuses={df_bonuses.shape}")
    
    # Step 2: Clean transactions data
    logger.info("Step 2: Cleaning transactions data...")
    completed_transactions, abandoned_transactions, df_transactions_clean = clean_transactions_data(df_transactions)
    logger.info(f"Completed: {completed_transactions.shape}, Abandoned: {abandoned_transactions.shape}")
    
    # Step 3: Save intermediate files (features.py expects paths)
    logger.info("Step 3: Saving intermediate files...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    completed_transactions.to_parquet(f"{DATA_DIR}/completed_transactions.parquet", index=False)
    abandoned_transactions.to_parquet(f"{DATA_DIR}/abandoned_transactions.parquet", index=False)
    df_transactions_clean.to_parquet(f"{DATA_DIR}/df_transactions_clean.parquet", index=False)
    df_sessions.to_parquet(f"{DATA_DIR}/df_sessions.parquet", index=False)
    df_bonuses.to_parquet(f"{DATA_DIR}/df_bonuses.parquet", index=False)
    
    # Step 4: Feature engineering
    logger.info("Step 4: Creating features...")
    features_path = create_features(
        df_sessions_path=f"{DATA_DIR}/df_sessions.parquet",
        completed_transactions_path=f"{DATA_DIR}/completed_transactions.parquet",
        abandoned_transactions_path=f"{DATA_DIR}/abandoned_transactions.parquet",
        df_transactions_clean_path=f"{DATA_DIR}/df_transactions_clean.parquet",
        df_bonuses_path=f"{DATA_DIR}/df_bonuses.parquet",
    )
    logger.info(f"Features saved to: {features_path}")
    
    # Step 5: Train models
    logger.info("Step 5: Training models...")
    results = train_all_models(features_path)
    
    models_trained = len([r for r in results.values() if r.get('saved')])
    logger.info(f"=== Pipeline complete! Models trained: {models_trained} ===")
    
    summary_path = f"{OUTPUT_DIR}/training_summary.json"
    logger.info(f"Metrics saved to: {summary_path}")

    from dataset import get_s3_client

    # Step 6: Upload models to S3
    logger.info("Step 6: Uploading models to S3...")
    s3 = get_s3_client()
    for segment_id in ["A", "B", "C", "D_E_F"]:
        local_path = f"{MODELS_DIR}/model_segment_{segment_id}.pkl"
        if os.path.exists(local_path):
            s3_key = f"models/model_segment_{segment_id}.pkl"
            s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
            logger.info(f"Uploaded {s3_key}")
        else:
            logger.warning(f"Skipping {local_path} (not trained)")
    
    return summary_path


if __name__ == "__main__":
    main()