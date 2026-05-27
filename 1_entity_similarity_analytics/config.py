import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

TARGET_METRIC = "accuracy"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

def ensure_directories():
    for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
ensure_directories()

# ============================================================================
# S3 CONFIG
# ============================================================================
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# ============================================================================
# DATA PATHS (in S3 bucket)
# ============================================================================
DATA_TRANSACTIONS_KEY = "transactions.csv"
DATA_SESSIONS_KEY = "sessions.csv"
DATA_BONUSES_KEY = "bonuses.csv"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
CATEGORY_LABELS = ["A", "B", "C", "D", "E", "F"]

FEATURE_COLUMNS_TO_EXCLUDE = [
    'entity_id',
    'category',
    'registration_date',
    'processing_method',
    'preferred_period',
    'category_encoded'
]

EMBEDDING_FEATURE_COLUMNS_TO_EXCLUDE = [
    'entity_id',
    'category',
    'registration_date',
    'processing_method',
    'preferred_period',
    'avg_processing_time_original'
]

COMPARISON_METRICS = [
    'avg_activity_volume',
    'min_activity_volume',
    'min_processing_time',
    'small_activity_rate',
    'avg_revenue_per_volume',
    'revenue_per_month',
    'success_rate',
    'abandonment_rate',
    'avg_processing_time_sec',
    'revenue_per_hour',
    'revenue_growth',
    'revenue_per_day',
    'avg_hours_per_day',
    'total_bonuses',
    'total_bonus_transactions',
    'total_bonus_rewards',
    'successful_bonuses',
    'bonus_success_rate',
    'avg_bonus_reward',
    'avg_transactions_per_bonus',
    'transactions_per_hour'
]

ALWAYS_INCREASE = [
    'revenue_per_hour',
    'successful_bonuses',
    'transactions_per_hour',
    'success_rate',
    'avg_hours_per_day',
    'avg_transactions_per_bonus',
    'total_bonuses',
    'avg_revenue_per_volume',
    'revenue_per_month',
    'revenue_growth',
    'revenue_per_day',
]

ALWAYS_DECREASE = [
    'abandonment_rate',
    'avg_processing_time_sec'
]

FOLLOW_BENCHMARK = [
    'avg_activity_volume',
    'small_activity_rate'
]

NUMERIC_COLUMNS = [
    'entity_value',
    'benchmark_value',
    'difference',
    'target_value',
    'kpi_improvement',
    'kpi_target',
    'gap_percent'
]