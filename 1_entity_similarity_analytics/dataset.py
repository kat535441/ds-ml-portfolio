import os
import boto3
import pandas as pd
from pathlib import Path
from typing import Tuple

from config import (
    S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
    S3_BUCKET_NAME, DATA_TRANSACTIONS_KEY, DATA_SESSIONS_KEY, DATA_BONUSES_KEY,
    DATA_DIR
)

def get_s3_client():
    """Create S3 client with credentials from env"""
    return boto3.client(
        service_name='s3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )

def download_from_s3(s3_key: str, local_path: str) -> str:
    """Download file from S3, return local path"""
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
    return local_path

def load_data():
    """Download all data from S3 and return DataFrames"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    transactions_path = download_from_s3(DATA_TRANSACTIONS_KEY, f"{DATA_DIR}/transactions.csv")
    sessions_path = download_from_s3(DATA_SESSIONS_KEY, f"{DATA_DIR}/sessions.csv")
    bonuses_path = download_from_s3(DATA_BONUSES_KEY, f"{DATA_DIR}/bonuses.csv")
    
    df_transactions = pd.read_csv(transactions_path)
    df_sessions = pd.read_csv(sessions_path)
    df_bonuses = pd.read_csv(bonuses_path)
    
    return df_transactions, df_sessions, df_bonuses

def clean_transactions_data(df_transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean transactions data — same logic as before, but takes DataFrame instead of path"""
    
    df_transactions_clean = df_transactions.dropna(subset=["entity_id"]).copy()
    df_transactions_clean = df_transactions_clean.drop(columns=["region_id", "abandon_reason"], errors="ignore")

    clean_df = df_transactions_clean[
        (df_transactions_clean["transaction_completed"].notna() ^ df_transactions_clean["transaction_abandoned"].notna())
    ].copy()

    completed_transactions = clean_df[clean_df["transaction_completed"].notna()].copy()
    abandoned_transactions = clean_df[clean_df["transaction_abandoned"].notna()].copy()

    completed_transactions["registration_date"] = pd.to_datetime(completed_transactions["registration_date"])
    completed_transactions["transaction_created"] = pd.to_datetime(completed_transactions["transaction_created"])

    return completed_transactions, abandoned_transactions, clean_df