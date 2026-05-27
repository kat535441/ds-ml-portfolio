import numpy as np
import pandas as pd
import ast
import duckdb
import h3
import joblib
import os
from pathlib import Path
from itertools import product

from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import spatial
from geopy.distance import geodesic
from implicit.als import AlternatingLeastSquares

from car_offer_rec.config import (
    DATA_PROCESSED_DIR,
    ID_COLUMNS,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_float64_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all numeric columns (except int-like IDs) to float64.
    Prevents DuckDB DECIMAL overflow errors on large values.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ['float32', 'float64', 'int64', 'int32']:
            # Skip ID columns (they should stay as-is)
            if col.endswith('_id') or col in ID_COLUMNS:
                continue
            df[col] = df[col].astype('float64')
        # Handle object columns that might be numeric
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    return df


# =============================================================================
# TEMPORAL SPLIT FUNCTIONS (no data leakage)
# =============================================================================

def compute_cutoff_date_for_each_user(df):
    """
    For each user finds the cutoff date when a gap > 15 days appears between trips.
    ALL features must be computed only on data BEFORE this date.
    """
    df = df.copy()
    df['trip_created_at'] = pd.to_datetime(df['trip_created_at'])
    df = df.sort_values(['user_id', 'trip_created_at'])

    df['prev_trip_date'] = df.groupby('user_id')['trip_created_at'].shift(1)
    df['gap_days'] = (df['trip_created_at'] - df['prev_trip_date']).dt.days

    churn_points = (
        df[df['gap_days'] >= 15]
        .groupby('user_id')['trip_created_at']
        .min()
        .rename('cutoff_date')
        .reset_index()
    )

    df = df.merge(churn_points, on='user_id', how='left')

    df['is_historical'] = (
        df['cutoff_date'].isna() |
        (df['trip_created_at'] <= df['cutoff_date'])
    )

    return df


def compute_target_on_historical(df_with_cutoff):
    """
    Computes target variable retained_by_offer:
    Whether the user had a trip with offer in the 15-30 days AFTER cutoff_date.
    """
    df = df_with_cutoff.copy()

    future = df[df['cutoff_date'].notna() & ~df['is_historical']].copy()

    if len(future) == 0:
        df['retained_by_offer'] = 0
        return df

    future['days_from_cutoff'] = (future['trip_created_at'] - future['cutoff_date']).dt.days

    future['is_retention_trip'] = (
        (future['days_from_cutoff'] >= 15) &
        (future['days_from_cutoff'] <= 30) &
        (future['offer_id'].notna())
    ).astype(int)

    user_target = (
        future.groupby('user_id')['is_retention_trip']
        .max()
        .rename('retained_by_offer')
        .reset_index()
    )

    df = df.merge(user_target, on='user_id', how='left')
    df['retained_by_offer'] = df['retained_by_offer'].fillna(0).astype(int)

    return df


def take_only_historical_data(df_with_cutoff):
    """
    Keeps ONLY trips before cutoff_date.
    All subsequent features will be computed only on this data.
    """
    df = df_with_cutoff.copy()
    df_historical = df[df['is_historical']].copy()
    df_historical = df_historical[df_historical['cutoff_date'].notna()].copy()
    return df_historical


# =============================================================================
# TRAINING-TIME FEATURE FUNCTIONS
# =============================================================================

def add_retention_features(df):
    """
    DEPRECATED — replaced by compute_cutoff_date_for_each_user + compute_target_on_historical.
    Kept for backward compatibility.
    """
    return df


def filter_trips_for_retention_training(df):
    """
    DEPRECATED — replaced by take_only_historical_data.
    Kept for backward compatibility.
    """
    return df


def generate_temporal_features_train(df):
    """
    Training-time temporal features (fixed — no data leakage).
    user_lifetime_days calculated from registration to cutoff_date.
    """
    df = df.copy()
    df['trip_created_at'] = pd.to_datetime(df['trip_created_at'], errors='coerce')
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], errors='coerce')

    df['trip_weekday'] = df['trip_created_at'].dt.weekday
    df['trip_hour'] = df['trip_created_at'].dt.hour

    def get_hour_bucket(hour):
        if pd.isnull(hour): return np.nan
        if 0 <= hour < 6: return 'night'
        elif 6 <= hour < 10: return 'morning'
        elif 10 <= hour < 16: return 'midday'
        elif 16 <= hour < 22: return 'evening'
        else: return 'late_evening'

    df['trip_hour_bucket'] = df['trip_hour'].apply(get_hour_bucket)
    df['used_midday'] = df['trip_hour'].apply(lambda x: 1 if 10 <= x < 16 else 0)

    first_trip_per_user = df.groupby('user_id')['trip_created_at'].min().reset_index()
    first_trip_per_user.rename(columns={'trip_created_at': 'first_trip_date'}, inplace=True)

    user_data = df[['user_id', 'user_created_at']].drop_duplicates()
    user_data = pd.merge(user_data, first_trip_per_user, on='user_id', how='left')

    df = pd.merge(df, user_data[['user_id', 'first_trip_date']], on='user_id', how='left')
    df['reg_to_first_trip_days'] = (df['first_trip_date'] - df['user_created_at']).dt.total_seconds() / (3600 * 24)

    # Time from last trip BEFORE cutoff_date
    last_trip_before_cutoff = df.groupby('user_id')['trip_created_at'].max().reset_index()
    last_trip_before_cutoff.rename(columns={'trip_created_at': 'last_trip_before_cutoff'}, inplace=True)
    df = df.merge(last_trip_before_cutoff, on='user_id', how='left')

    df['days_since_last_trip'] = (df['cutoff_date'] - df['last_trip_before_cutoff']).dt.days
    df['is_churn_risk'] = (df['days_since_last_trip'] >= 15).astype(int)
    df['user_lifetime_days'] = (df['cutoff_date'] - df['user_created_at']).dt.total_seconds() / (3600 * 24)

    return df


def generate_geo_features_train(df, all_hexagons, n_clusters):
    """Training-time geo features: fits KMeans and builds cKDTree."""
    coords_origin = df[['origin_lat', 'origin_lng']].dropna()
    coords_destination = df[['destination_lat', 'destination_lng']].dropna()

    kmeans_origin = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_destination = KMeans(n_clusters=n_clusters, random_state=42)

    kmeans_origin.fit(coords_origin)
    kmeans_destination.fit(coords_destination)

    df['origin_area_id'] = kmeans_origin.predict(df[['origin_lat', 'origin_lng']])
    df['destination_area_id'] = kmeans_destination.predict(df[['destination_lat', 'destination_lng']])
    df['trips_cross_area'] = (df['origin_area_id'] != df['destination_area_id']).astype(int)

    origin_centers = kmeans_origin.cluster_centers_
    destination_centers = kmeans_destination.cluster_centers_

    cluster_distances_km = {}
    for i, j in product(range(n_clusters), repeat=2):
        coord1 = (origin_centers[i][0], origin_centers[i][1])
        coord2 = (destination_centers[j][0], destination_centers[j][1])
        dist_km = geodesic(coord1, coord2).km
        cluster_distances_km[(i, j)] = dist_km

    df['cluster_center_distance_km'] = df.apply(
        lambda row: cluster_distances_km.get(
            (int(row['origin_area_id']), int(row['destination_area_id'])),
            np.nan
        ), axis=1)

    def categorize_distance(row):
        if pd.isna(row['distance_km']) or row['distance_km'] == 0:
            return 0
        elif row['distance_km'] <= 5:
            return 1
        else:
            return 2

    df['distance_category'] = df.apply(categorize_distance, axis=1)

    # H3 hexagons
    all_hexagons = all_hexagons.copy()
    all_hexagons["center"] = all_hexagons["hex"].apply(h3.cell_to_latlng)
    all_hexagons["center_lat"] = all_hexagons["center"].apply(lambda x: x[0])
    all_hexagons["center_lng"] = all_hexagons["center"].apply(lambda x: x[1])

    tree = spatial.cKDTree(all_hexagons[["center_lat", "center_lng"]].values)

    origin_points = list(zip(df["origin_lat"], df["origin_lng"]))
    destination_points = list(zip(df["destination_lat"], df["destination_lng"]))

    origin_dists, origin_indices = tree.query(origin_points)
    destination_dists, destination_indices = tree.query(destination_points)

    df["hex_origin"] = all_hexagons.iloc[origin_indices]["hex"].values
    df["hex_destination"] = all_hexagons.iloc[destination_indices]["hex"].values

    return df, kmeans_origin, kmeans_destination, tree


def generate_user_trip_aggregates_train(df):
    """
    Training-time user-trip aggregates (fixed — on historical data only).
    """
    df = df.copy()

    completed_mask = df['completed_at'].notna()
    df['completed_mask'] = completed_mask
    df['discount'] = (df['trip_total'] - df['trip_base']).clip(lower=0)

    df['avg_distance_km'] = df.groupby('user_id')['distance_km'].transform('mean')
    df['avg_duration_min'] = df.groupby('user_id')['duration_min'].transform('mean')
    df['avg_trip_base'] = df.groupby('user_id')['trip_base'].transform('mean')
    df['avg_trip_total'] = df.groupby('user_id')['trip_total'].transform('mean')

    trip_counts = df.groupby('user_id')['trip_id'].count().rename('total_trips_count')
    completed_counts = df[completed_mask].groupby('user_id')['trip_id'].count().rename('completed_trips_count')

    user_stats = pd.concat([trip_counts, completed_counts], axis=1).fillna(0)
    user_stats['completed_trips_percent'] = (
        user_stats['completed_trips_count'] / user_stats['total_trips_count']
    ).fillna(0)

    df['avg_discount'] = df[df['offer_id'].notna()].groupby('user_id')['discount'].transform('mean')
    df = df.merge(user_stats[['completed_trips_percent']], left_on='user_id', right_index=True, how='left')

    return df


def generate_offer_specific_features_train(df):
    """
    Training-time offer-specific features.
    """
    df['discount_percent'] = ((df['trip_total'] - df['trip_base']) / df['trip_total']).clip(lower=0)
    df['used_discount_flag'] = (df['trip_base'] < df['trip_total']).astype(int)
    df['offer_used'] = (df['offer_id'].notna()).astype(int)
    df['offer_success'] = ((df['offer_id'].notna()) & (df['completed_at'].notna())).astype(int)
    return df


def agg_user_train(df):
    """
    Training-time aggregation by user_id (fixed — no data leakage).
    Uses days_since_last_trip renamed to time_since_last_activity_days.
    """
    user_features = df.groupby('user_id').agg(
        payment_method_id=('payment_method_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        client_type=('client_type', 'first'),
        mode_trip_weekday=('trip_weekday', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trip_hour=('trip_hour', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trip_hour_bucket=('trip_hour_bucket', lambda x: x.mode()[0] if not x.mode().empty else 'unknown'),
        mode_used_midday=('used_midday', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        reg_to_first_trip_days=('reg_to_first_trip_days', 'first'),
        days_since_last_trip=('days_since_last_trip', 'first'),
        is_churn_risk=('is_churn_risk', 'first'),
        user_lifetime_days=('user_lifetime_days', 'first'),
        mode_origin_area_id=('origin_area_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_destination_area_id=('destination_area_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trips_cross_area=('trips_cross_area', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mean_cluster_center_distance_km=('cluster_center_distance_km', 'mean'),
        mode_distance_category=('distance_category', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_hex_origin=('hex_origin', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_hex_destination=('hex_destination', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        avg_distance_km=('distance_km', 'mean'),
        avg_duration_min=('duration_min', 'mean'),
        avg_trip_base=('trip_base', 'mean'),
        avg_trip_total=('trip_total', 'mean'),
        completed_trips_percent=('completed_trips_percent', 'first'),
        mean_discount_percent=('discount_percent', 'mean'),
        has_used_any_discount=('used_discount_flag', 'max'),
        has_used_any_offer=('offer_used', 'max'),
        total_offer_successes=('offer_success', 'sum'),
        has_any_offer_success=('offer_success', 'max'),
        retained_by_offer=('retained_by_offer', 'first')
    ).reset_index()

    for col in user_features.columns:
        if col.startswith('mode_') and col not in ['mode_trip_hour_bucket', 'mode_hex_origin', 'mode_hex_destination']:
            user_features[col] = user_features[col].fillna(0).astype(int)
        elif col in ['mode_trip_hour_bucket', 'mode_hex_origin', 'mode_hex_destination']:
            user_features[col] = user_features[col].fillna('unknown')
        elif user_features[col].dtype == 'object':
            user_features[col] = user_features[col].fillna('unknown')
        elif user_features[col].dtype in ['float64', 'int64', 'bool']:
            user_features[col] = user_features[col].fillna(0)

    # Rename for compatibility with final_feature_join
    user_features = user_features.rename(columns={'days_since_last_trip': 'time_since_last_activity_days'})

    return user_features


def generate_offer_features_train(df):
    """Aggregates features at offer_id level (global offer characteristics)."""
    df_offer = df[df['offer_id'].notna()].copy()

    if len(df_offer) == 0:
        return pd.DataFrame()

    car_category_cols = [c for c in df_offer.columns if c.startswith('car_category_')]

    agg_dict = {
        'distance_km': ['mean'],
        'duration_min': ['mean'],
        'trip_base': ['mean'],
        'trip_total': ['mean'],
        'discount': ['mean'],
        'use_percent': ['mean'],
        'use_amount': ['mean'],
        'usage_count': ['mean'],
        'offer_type_id': 'first',
        'completed_mask': ['mean', 'sum'],
        'trip_id': 'count',
        'user_lifetime_days': 'mean',
        'completed_trips_percent': 'mean',
        'is_churn_risk': 'mean'
    }

    for col in car_category_cols:
        agg_dict[col] = 'sum'

    offer_agg = df_offer.groupby('offer_id').agg(agg_dict)
    offer_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in offer_agg.columns]
    offer_agg = offer_agg.reset_index()

    rename_map = {
        'distance_km_mean': 'offer_global_avg_distance_km',
        'duration_min_mean': 'offer_global_avg_duration_min',
        'trip_base_mean': 'offer_global_avg_trip_base',
        'trip_total_mean': 'offer_global_avg_trip_total',
        'discount_mean': 'offer_global_avg_discount',
        'use_percent_mean': 'offer_global_avg_use_percent',
        'use_amount_mean': 'offer_global_avg_use_amount',
        'usage_count_mean': 'offer_global_avg_usage_count',
        'offer_type_id_first': 'offer_type_id',
        'completed_mask_mean': 'offer_conversion_rate',
        'completed_mask_sum': 'offer_completed_count',
        'trip_id_count': 'offer_usage_count',
        'user_lifetime_days_mean': 'offer_user_avg_lifetime',
        'completed_trips_percent_mean': 'offer_user_avg_completion_rate',
        'is_churn_risk_mean': 'offer_user_churn_risk_ratio'
    }
    offer_agg = offer_agg.rename(columns=rename_map)

    car_category_sum_cols = [c for c in offer_agg.columns if c.startswith('car_category_') and c.endswith('_sum')]

    if car_category_sum_cols:
        offer_agg['car_category'] = offer_agg[car_category_sum_cols].idxmax(axis=1)
        offer_agg['car_category'] = offer_agg['car_category'].str.replace('car_category_', '').str.replace('_sum', '').astype(int)
        offer_agg = offer_agg.drop(columns=car_category_sum_cols)
    else:
        offer_agg['car_category'] = 1

    offer_agg['offer_avg_saving_per_trip'] = offer_agg['offer_global_avg_discount']
    offer_agg['offer_value_ratio'] = offer_agg['offer_global_avg_discount'] / offer_agg['offer_global_avg_trip_total'].replace(0, 1)

    churn_risk_usage = df_offer[df_offer['is_churn_risk'] == 1].groupby('offer_id').size()
    offer_agg['offer_churn_risk_usage'] = offer_agg['offer_id'].map(churn_risk_usage).fillna(0)

    healthy_users_count = df_offer[df_offer['completed_trips_percent'] > 0.7].groupby('offer_id').size()
    offer_agg['offer_healthy_user_ratio'] = (
        offer_agg['offer_id'].map(healthy_users_count).fillna(0) /
        offer_agg['offer_usage_count'].replace(0, 1)
    )

    return offer_agg


def generate_aggregates_for_offer_user(df):
    """Aggregation by user_id + offer_id (user-offer activity level)."""
    df['avg_distance_km_offer_user'] = df.groupby(['user_id', 'offer_id'])['distance_km'].transform('mean')
    df['avg_duration_min_offer_user'] = df.groupby(['user_id', 'offer_id'])['duration_min'].transform('mean')
    df['avg_discount_offer_user'] = df.groupby(['user_id', 'offer_id'])['discount'].transform('mean')
    df['avg_trip_base_offer_user'] = df.groupby(['user_id', 'offer_id'])['trip_base'].transform('mean')
    df['avg_trip_total_offer_user'] = df.groupby(['user_id', 'offer_id'])['trip_total'].transform('mean')

    offer_behavior = (
        df.groupby(['user_id', 'offer_id'])
        .agg(
            offer_used_times_offer_user=('trip_id', 'count'),
            offer_completed_times_offer_user=('completed_at', lambda x: x.notnull().sum())
        )
        .reset_index()
    )

    offer_behavior['offer_conversion_rate_offer_user'] = (
        offer_behavior['offer_completed_times_offer_user'] / offer_behavior['offer_used_times_offer_user'].replace(0, np.nan)
    )

    df = df.merge(offer_behavior, on=['user_id', 'offer_id'], how='left')

    df['offer_used_times_offer_user'] = df['offer_used_times_offer_user'].fillna(0).astype(int)
    df['offer_completed_times_offer_user'] = df['offer_completed_times_offer_user'].fillna(0).astype(int)
    df['offer_conversion_rate_offer_user'] = df['offer_conversion_rate_offer_user'].fillna(0)

    return df


def generate_and_save_embeddings(df_with_vectors, als_factors, als_iterations, als_reg, pca_n_components):
    """Generates ALS embeddings + PCA reduction. Returns df and fitted models."""
    df_inter = df_with_vectors.loc[
        (df_with_vectors['offer_id'] != -1) &
        (df_with_vectors['offer_used'] == 1),
        ['user_id', 'offer_id']
    ].copy()

    users = df_inter['user_id'].astype('category')
    offers = df_inter['offer_id'].astype('category')

    user_item_matrix = csr_matrix((
        np.ones(len(df_inter)),
        (users.cat.codes, offers.cat.codes)
    ), shape=(len(users.cat.categories), len(offers.cat.categories)))

    als_model = AlternatingLeastSquares(
        factors=als_factors,
        regularization=als_reg,
        iterations=als_iterations,
        random_state=42
    )

    als_model.fit(user_item_matrix)

    user_embeddings_32d = als_model.user_factors
    offer_embeddings_32d = als_model.item_factors

    pca_user = PCA(n_components=pca_n_components, random_state=42)
    user_embeddings_1d = pca_user.fit_transform(user_embeddings_32d).flatten()

    pca_offer = PCA(n_components=pca_n_components, random_state=42)
    offer_embeddings_1d = pca_offer.fit_transform(offer_embeddings_32d).flatten()

    user_emb_df = pd.DataFrame({
        'user_id': users.cat.categories.values,
        'user_embedding': user_embeddings_1d
    })

    offer_emb_df = pd.DataFrame({
        'offer_id': offers.cat.categories.values,
        'offer_embedding': offer_embeddings_1d
    })

    df_with_vectors = df_with_vectors.merge(user_emb_df, on='user_id', how='left')
    df_with_vectors = df_with_vectors.merge(offer_emb_df, on='offer_id', how='left')

    df_with_vectors['user_embedding'] = df_with_vectors['user_embedding'].fillna(0)
    df_with_vectors['offer_embedding'] = df_with_vectors['offer_embedding'].fillna(0)

    selected_columns = [
        'user_id', 'offer_id', 'user_embedding', 'offer_embedding',
        'avg_distance_km_offer_user', 'avg_duration_min_offer_user', 'avg_discount_offer_user',
        'avg_trip_base_offer_user', 'avg_trip_total_offer_user',
        'offer_used_times_offer_user', 'offer_completed_times_offer_user',
        'offer_conversion_rate_offer_user'
    ]

    df_with_vectors = df_with_vectors[selected_columns]

    user_mapping = dict(enumerate(users.cat.categories))
    offer_mapping = dict(enumerate(offers.cat.categories))

    return df_with_vectors, als_model, user_mapping, offer_mapping, user_embeddings_32d, offer_embeddings_32d, pca_user, pca_offer


def final_feature_join(df_with_vectors, offer_agg, user_agg):
    """Final duckdb join of user_agg + offer_agg + embedding vectors."""
    df_with_vectors = df_with_vectors.dropna(subset=['offer_id'])

    df_with_vectors = _ensure_float64_numeric(df_with_vectors)
    offer_agg = _ensure_float64_numeric(offer_agg)
    user_agg = _ensure_float64_numeric(user_agg)

    con = duckdb.connect()
    con.register("df_with_vectors", df_with_vectors)
    con.register("offer_agg", offer_agg)
    con.register("user_agg", user_agg)

    clean_df = con.execute("""
        SELECT
            v.user_id,
            v.offer_id,
            v.user_embedding,
            v.offer_embedding,
            v.avg_distance_km_offer_user,
            v.avg_duration_min_offer_user,
            v.avg_discount_offer_user,
            v.avg_trip_base_offer_user,
            v.avg_trip_total_offer_user,
            v.offer_used_times_offer_user,
            v.offer_completed_times_offer_user,
            v.offer_conversion_rate_offer_user,

            u.payment_method_id AS user_F_payment_method_id,
            u.client_type AS user_F_client_type,
            u.mode_trip_weekday AS user_F_mode_trip_weekday,
            u.mode_trip_hour AS user_F_mode_trip_hour,
            u.mode_trip_hour_bucket AS user_F_mode_trip_hour_bucket,
            u.mode_used_midday AS user_F_mode_used_midday,
            u.reg_to_first_trip_days AS user_F_reg_to_first_trip_days,
            u.is_churn_risk AS user_F_is_churn_risk,
            u.user_lifetime_days AS user_F_user_lifetime_days,
            u.mode_origin_area_id AS user_F_mode_origin_area_id,
            u.mode_destination_area_id AS user_F_mode_destination_area_id,
            u.mode_trips_cross_area AS user_F_mode_trips_cross_area,
            u.mean_cluster_center_distance_km AS user_F_mean_cluster_center_distance_km,
            u.mode_distance_category AS user_F_mode_distance_category,
            u.mode_hex_origin AS user_F_mode_hex_origin,
            u.mode_hex_destination AS user_F_mode_hex_destination,
            u.avg_distance_km AS user_F_avg_distance_km,
            u.avg_duration_min AS user_F_avg_duration_min,
            u.avg_trip_base AS user_F_avg_trip_base,
            u.avg_trip_total AS user_F_avg_trip_total,
            u.completed_trips_percent AS user_F_completed_trips_percent,
            u.mean_discount_percent AS user_F_mean_discount_percent,
            u.has_used_any_discount AS user_F_has_used_any_discount,
            u.has_used_any_offer AS user_F_has_used_any_offer,
            u.total_offer_successes AS user_F_total_offer_successes,
            u.has_any_offer_success AS user_F_has_any_offer_success,
            u.retained_by_offer AS user_F_retained_by_offer,

            p.offer_global_avg_distance_km AS offer_F_global_avg_distance_km,
            p.offer_global_avg_duration_min AS offer_F_global_avg_duration_min,
            p.offer_global_avg_trip_base AS offer_F_global_avg_trip_base,
            p.offer_global_avg_trip_total AS offer_F_global_avg_trip_total,
            p.offer_global_avg_discount AS offer_F_global_avg_discount,
            p.offer_global_avg_use_percent AS offer_F_global_avg_use_percent,
            p.offer_global_avg_use_amount AS offer_F_global_avg_use_amount,
            p.offer_global_avg_usage_count AS offer_F_global_avg_usage_count,
            p.offer_type_id AS offer_F_type_id,
            p.offer_conversion_rate AS offer_F_conversion_rate,
            p.offer_completed_count AS offer_F_completed_count,
            p.offer_usage_count AS offer_F_usage_count,
            p.offer_user_avg_lifetime AS offer_F_user_avg_lifetime,
            p.offer_user_avg_completion_rate AS offer_F_user_avg_completion_rate,
            p.offer_user_churn_risk_ratio AS offer_F_user_churn_risk_ratio,
            p.car_category AS offer_F_car_category,
            p.offer_avg_saving_per_trip AS offer_F_avg_saving_per_trip,
            p.offer_value_ratio AS offer_F_value_ratio,
            p.offer_churn_risk_usage AS offer_F_churn_risk_usage,
            p.offer_healthy_user_ratio AS offer_F_healthy_user_ratio

        FROM df_with_vectors v
        INNER JOIN user_agg u
            ON v.user_id = u.user_id
        LEFT JOIN offer_agg p
            ON v.offer_id = p.offer_id
    """).df()

    return clean_df


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def data_join_for_pipeline(
    preprocessed_path: str,
    hex_path: str,
    offer_path: str,
    kmeans_n_clusters: int
):
    """
    Takes preprocessed (user+trip) join, adds offer data,
    generates features with TEMPORAL SPLIT (no leaks).
    Returns general_df and geo_models dict.
    """
    from clearml import Task, Logger

    task = Task.current_task()
    log = Logger.current_logger()

    df = pd.read_parquet(preprocessed_path)
    hex_df = pd.read_parquet(hex_path) if hex_path.endswith(".parquet") else pd.read_csv(hex_path)
    offer_df = pd.read_parquet(offer_path) if offer_path.endswith(".parquet") else pd.read_csv(offer_path)

    # Prepare offer data
    offer_df['car_category'] = offer_df['car_category'].fillna('unknown')

    df_car_onehot = pd.get_dummies(
        offer_df[['offer_id', 'car_category']],
        columns=['car_category'],
        prefix='car_category'
    )
    df_car_onehot.columns = df_car_onehot.columns.str.replace('.0', '', regex=False)
    df_car_onehot = df_car_onehot.groupby('offer_id').max().reset_index()

    df_offer_base = offer_df.groupby('offer_id').agg({
        'use_percent': 'first',
        'use_amount': 'first',
        'offer_type_id': 'first',
        'usage_count': 'first'
    }).reset_index()

    df_offer = df_offer_base.merge(df_car_onehot, on='offer_id', how='left')

    df = _ensure_float64_numeric(df)
    df_offer = _ensure_float64_numeric(df_offer)

    result = df.merge(df_offer, on='offer_id', how='left')

    cols_to_drop = [c for c in result.columns if c.startswith('car_category_')
                    and c in ['car_category_15', 'car_category_unknown']]
    result = result.drop(columns=cols_to_drop, errors='ignore')

    # Temporal split (no leaks)
    df_with_cutoff = compute_cutoff_date_for_each_user(result)
    df_with_target = compute_target_on_historical(df_with_cutoff)
    df_historical = take_only_historical_data(df_with_target)

    # Apply feature generation on historical data only
    df_1 = generate_temporal_features_train(df_historical)
    df_1, kmeans_origin, kmeans_destination, tree = generate_geo_features_train(df_1, hex_df, kmeans_n_clusters)
    df_1 = generate_user_trip_aggregates_train(df_1)
    df_1 = generate_offer_specific_features_train(df_1)

    general_df = df_1

    geo_models = {
        'kmeans_origin': kmeans_origin,
        'kmeans_destination': kmeans_destination,
        'general_df': general_df,
        'all_hexagons': hex_df,
        'tree': tree
    }

    if log:
        log.report_text(f"[data_join] general_df shape={general_df.shape}")

    return geo_models


def feature_engineering_for_pipeline(
    geo_models,
    pca_n_components: int,
    als_factors: int,
    als_iterations: int,
    als_reg: float
):
    """
    User agg, offer features, ALS embeddings, PCA, final join.
    """
    from clearml import Task, Logger
    task = Task.current_task()
    log = Logger.current_logger()

    general_df = geo_models["general_df"]

    user_agg = agg_user_train(general_df)
    offer_agg = generate_offer_features_train(general_df)
    offer_features = offer_agg.copy()

    df_2 = general_df.copy()
    df_2 = generate_aggregates_for_offer_user(df_2)

    df_with_vectors, als_model, user_mapping, offer_mapping, user_emb_32d, offer_emb_32d, pca_user, pca_offer = \
        generate_and_save_embeddings(df_2, als_factors, als_iterations, als_reg, pca_n_components)

    als_data = {
        'model': als_model,
        'user_mapping': user_mapping,
        'offer_mapping': offer_mapping,
        'user_embeddings_32d': user_emb_32d,
        'offer_embeddings_32d': offer_emb_32d,
    }

    pca_models = {
        'pca_user': pca_user,
        'pca_offer': pca_offer
    }

    clean_df = final_feature_join(df_with_vectors, offer_agg, user_agg)

    if log:
        log.report_text(f"[feature_engineering] clean_df shape={clean_df.shape}")

    models_pkl = {
        'offer_features': offer_features,
        'als_data': als_data,
        'pca_models': pca_models
    }

    feature_engineering_output = {
        'clean_df': clean_df,
        'models_pkl': models_pkl
    }

    return feature_engineering_output


def create_features_for_pipeline(
    preprocessed_path: str,
    hex_path: str,
    offer_path: str,
    kmeans_n_clusters: int = 10,
    pca_n_components: int = 1,
    als_factors: int = 32,
    als_iterations: int = 15,
    als_reg: float = 0.1,
) -> str:
    """
    Pipeline wrapper: runs ALL feature engineering.
    """
    logger.info("Starting create_features_for_pipeline...")

    geo_models = data_join_for_pipeline(
        preprocessed_path=preprocessed_path,
        hex_path=hex_path,
        offer_path=offer_path,
        kmeans_n_clusters=kmeans_n_clusters,
    )

    feature_engineering_output = feature_engineering_for_pipeline(
        geo_models=geo_models,
        pca_n_components=pca_n_components,
        als_factors=als_factors,
        als_iterations=als_iterations,
        als_reg=als_reg,
    )

    clean_df = feature_engineering_output['clean_df']
    out_path = DATA_PROCESSED_DIR / "final_features.parquet"
    clean_df.to_parquet(out_path, index=False)
    logger.info(f"Final features saved to {out_path}. Shape: {clean_df.shape}")

    models_pkl = feature_engineering_output['models_pkl']
    models_pkl_path = DATA_PROCESSED_DIR / "feature_models.pkl"
    joblib.dump(models_pkl, models_pkl_path)

    geo_models_save = {k: v for k, v in geo_models.items() if k != 'general_df'}
    geo_models_path = DATA_PROCESSED_DIR / "geo_models.pkl"
    joblib.dump(geo_models_save, geo_models_path)

    logger.info(f"Feature model artifacts saved to {DATA_PROCESSED_DIR}")

    return str(out_path)


# =============================================================================
# PREDICTION-TIME FEATURE FUNCTIONS
# =============================================================================

def prepare_final_dataset(result_for_model):

    result_for_model = result_for_model.fillna(0)

    user_offer_check = result_for_model.groupby('user_id')['offer_id'].apply(lambda x: (x > 0).any())
    users_without_any_offer = (~user_offer_check).sum()
    logger.info(users_without_any_offer)

    users_with_offer_ids = set(result_for_model[result_for_model['offer_id'] > 0]['user_id'].unique())
    filtered_with_offer = result_for_model[
        ~((result_for_model['user_id'].isin(users_with_offer_ids)) & (result_for_model['offer_id'] == 0))
    ].copy()

    all_user_ids = set(result_for_model['user_id'].unique())
    users_no_offer_ids = all_user_ids - users_with_offer_ids
    users_no_offer = result_for_model[result_for_model['user_id'].isin(users_no_offer_ids)].copy()

    logger.info(f"Users with offer: {len(users_with_offer_ids)}")
    logger.info(f"Users without offer: {len(users_no_offer_ids)}")
    logger.info(f"Rows in filtered_with_offer: {len(filtered_with_offer)}")
    logger.info(f"Rows in users_no_offer: {len(users_no_offer)}")

    if len(users_with_offer_ids) == 0:

        logger.info("No users with offer available for comparison. Using fallback strategy.")

        offer_counts = result_for_model[result_for_model['offer_id'] > 0]['offer_id'].value_counts()
        if len(offer_counts) > 0:
            most_common_offer = offer_counts.index[0]
            logger.info(f"Most common offer ID: {most_common_offer}")
            offer_features = result_for_model[result_for_model['offer_id'] == most_common_offer].iloc[0]
            offer_columns = [col for col in result_for_model.columns if col.startswith('offer_F_')]

            for col in offer_columns:
                users_no_offer[col] = offer_features[col]
            users_no_offer['offer_id'] = most_common_offer

            final_dataset = pd.concat([filtered_with_offer, users_no_offer], ignore_index=True)
        else:
            logger.info("No available offers at all! Returning original dataset.")
            final_dataset = result_for_model.copy()

    else:
        def safe_parse_embedding(embedding):
            try:
                if isinstance(embedding, str):
                    return np.array(ast.literal_eval(embedding))
                elif isinstance(embedding, (list, np.ndarray)):
                    return np.array(embedding)
                elif isinstance(embedding, (int, float)):
                    return np.array([embedding])
            except:
                return np.nan
            return np.nan

        users_no_offer['embedding_vector'] = users_no_offer['user_embedding'].apply(safe_parse_embedding)
        filtered_with_offer['embedding_vector'] = filtered_with_offer['user_embedding'].apply(safe_parse_embedding)

        users_no_offer = users_no_offer[users_no_offer['embedding_vector'].notna()].copy()
        filtered_with_offer = filtered_with_offer[filtered_with_offer['embedding_vector'].notna()].copy()

        logger.info(f"After embedding parsing:")
        logger.info(f"Rows in filtered_with_offer: {len(filtered_with_offer)}")
        logger.info(f"Rows in users_no_offer: {len(users_no_offer)}")

        if len(filtered_with_offer) == 0:
            logger.info("No valid embeddings for users with offer! Using random offer assignment.")

            available_offers = result_for_model[result_for_model['offer_id'] > 0]['offer_id'].unique()
            if len(available_offers) > 0:
                offer_columns = [col for col in result_for_model.columns if col.startswith('offer_F_')]
                for idx, row in users_no_offer.iterrows():
                    random_offer = np.random.choice(available_offers)
                    offer_features = result_for_model[result_for_model['offer_id'] == random_offer].iloc[0]
                    users_no_offer.loc[idx, 'offer_id'] = random_offer
                    for col in offer_columns:
                        users_no_offer.loc[idx, col] = offer_features[col]
                final_dataset = pd.concat(
                    [result_for_model[result_for_model['offer_id'] > 0], users_no_offer],
                    ignore_index=True
                )
            else:
                logger.info("No available offers at all! Returning original dataset.")
                final_dataset = result_for_model.copy()

        else:
            offer_columns = [col for col in result_for_model.columns if col.startswith('offer_F_')]
            offer_vectors = np.vstack(filtered_with_offer['embedding_vector'].values)

            filled_rows = []
            for idx, row in users_no_offer.iterrows():
                user_vec = row['embedding_vector'].reshape(1, -1)
                similarities = cosine_similarity(user_vec, offer_vectors).flatten()
                best_idx = similarities.argmax()
                similar_user = filtered_with_offer.iloc[best_idx]

                filled_row = row.copy()
                filled_row['offer_id'] = similar_user['offer_id']
                for col in offer_columns:
                    filled_row[col] = similar_user[col]
                filled_rows.append(filled_row)

            users_no_offer_filled = pd.DataFrame(filled_rows)
            final_dataset = pd.concat([filtered_with_offer, users_no_offer_filled], ignore_index=True)

    logger.info(f"Final dataset size: {final_dataset.shape}")
    logger.info(f"Unique users: {final_dataset['user_id'].nunique()}")
    logger.info(f"Unique offers: {final_dataset['offer_id'].nunique()}")
    logger.info(f"Rows with offer_id == 0: {(final_dataset['offer_id'] == 0).sum()}")

    final_dataset = final_dataset.drop(columns=['embedding_vector'], errors='ignore')

    return final_dataset


def generate_temporal_features(df):
    """
    Temporal feature generation (prediction-time) — FIXED.
    """
    df = df.copy()
    df['trip_created_at'] = pd.to_datetime(df['trip_created_at'], errors='coerce')
    df['user_created_at'] = pd.to_datetime(df['user_created_at'], errors='coerce')

    df['trip_weekday'] = df['trip_created_at'].dt.weekday
    df['trip_hour'] = df['trip_created_at'].dt.hour

    def get_hour_bucket(hour):
        if pd.isnull(hour): return np.nan
        if 0 <= hour < 6: return 'night'
        elif 6 <= hour < 10: return 'morning'
        elif 10 <= hour < 16: return 'midday'
        elif 16 <= hour < 22: return 'evening'
        else: return 'late_evening'

    df['trip_hour_bucket'] = df['trip_hour'].apply(get_hour_bucket)
    df['used_midday'] = df['trip_hour'].apply(lambda x: 1 if 10 <= x < 16 else 0)

    first_trip_per_user = df.groupby('user_id')['trip_created_at'].min().reset_index()
    first_trip_per_user.rename(columns={'trip_created_at': 'first_trip_date'}, inplace=True)

    user_data = df[['user_id', 'user_created_at']].drop_duplicates()
    user_data = pd.merge(user_data, first_trip_per_user, on='user_id', how='left')

    df = pd.merge(df, user_data[['user_id', 'first_trip_date']], on='user_id', how='left')
    df['reg_to_first_trip_days'] = (df['first_trip_date'] - df['user_created_at']).dt.total_seconds() / (3600 * 24)

    # FIXED: gaps and last trip
    df = df.sort_values(['user_id', 'trip_created_at'])
    df['prev_trip_date'] = df.groupby('user_id')['trip_created_at'].shift(1)
    df['gap_days'] = (df['trip_created_at'] - df['prev_trip_date']).dt.days

    last_trip = df.groupby('user_id')['trip_created_at'].max()
    df['days_since_last_trip'] = (pd.Timestamp.now() - df['user_id'].map(last_trip)).dt.days

    df['is_churn_risk'] = (df['days_since_last_trip'] >= 15).astype(int)
    df['user_lifetime_days'] = (df['user_id'].map(last_trip) - df['user_created_at']).dt.total_seconds() / (3600 * 24)

    return df


def generate_geo_features(df, all_hexagons, kmeans_origin, kmeans_destination, tree):
    """Geo feature generation using saved models (prediction-time)."""
    df = df.copy()

    df['origin_area_id'] = kmeans_origin.predict(df[['origin_lat', 'origin_lng']])
    df['destination_area_id'] = kmeans_destination.predict(df[['destination_lat', 'destination_lng']])
    df['trips_cross_area'] = (df['origin_area_id'] != df['destination_area_id']).astype(int)

    origin_centers = kmeans_origin.cluster_centers_
    destination_centers = kmeans_destination.cluster_centers_

    cluster_distances_km = {}
    n_clusters = len(origin_centers)
    for i, j in product(range(n_clusters), repeat=2):
        coord1 = (origin_centers[i][0], origin_centers[i][1])
        coord2 = (destination_centers[j][0], destination_centers[j][1])
        dist_km = geodesic(coord1, coord2).km
        cluster_distances_km[(i, j)] = dist_km

    df['cluster_center_distance_km'] = df.apply(
        lambda row: cluster_distances_km.get(
            (int(row['origin_area_id']), int(row['destination_area_id'])),
            np.nan
        ), axis=1)

    def categorize_distance(row):
        if pd.isna(row['distance_km']) or row['distance_km'] == 0:
            return 0
        elif row['distance_km'] <= 5:
            return 1
        else:
            return 2

    df['distance_category'] = df.apply(categorize_distance, axis=1)

    origin_points = list(zip(df["origin_lat"], df["origin_lng"]))
    destination_points = list(zip(df["destination_lat"], df["destination_lng"]))

    origin_dists, origin_indices = tree.query(origin_points)
    destination_dists, destination_indices = tree.query(destination_points)

    df["hex_origin"] = all_hexagons.iloc[origin_indices]["hex"].values
    df["hex_destination"] = all_hexagons.iloc[destination_indices]["hex"].values

    return df


def generate_user_trip_aggregates(df):
    """
    User aggregation (prediction-time).
    """
    df = df.copy()

    completed_mask = df['completed_at'].notna()
    df['completed_mask'] = completed_mask
    df['discount'] = (df['trip_total'] - df['trip_base']).clip(lower=0)

    df['avg_distance_km'] = df.groupby('user_id')['distance_km'].transform('mean')
    df['avg_duration_min'] = df.groupby('user_id')['duration_min'].transform('mean')
    df['avg_trip_base'] = df.groupby('user_id')['trip_base'].transform('mean')
    df['avg_trip_total'] = df.groupby('user_id')['trip_total'].transform('mean')

    trip_counts = df.groupby('user_id')['trip_id'].count().rename('total_trips_count')
    completed_counts = df[completed_mask].groupby('user_id')['trip_id'].count().rename('completed_trips_count')

    user_stats = pd.concat([trip_counts, completed_counts], axis=1).fillna(0)
    user_stats['completed_trips_percent'] = (
        user_stats['completed_trips_count'] / user_stats['total_trips_count']
    ).fillna(0)

    df['avg_discount'] = df[df['offer_id'].notna()].groupby('user_id')['discount'].transform('mean')
    df = df.merge(user_stats[['completed_trips_percent']], left_on='user_id', right_index=True, how='left')

    return df


def generate_offer_specific_features(df):
    """
    Offer-specific feature generation (prediction-time).
    """
    df = df.copy()
    df['discount_percent'] = ((df['trip_total'] - df['trip_base']) / df['trip_total']).clip(lower=0)
    df['used_discount_flag'] = (df['trip_base'] < df['trip_total']).astype(int)
    df['offer_used'] = (df['offer_id'].notna()).astype(int)
    df['offer_success'] = ((df['offer_id'].notna()) & (df['completed_at'].notna())).astype(int)
    return df


def agg_user(df):
    """
    User aggregation (prediction-time) — FIXED.
    Uses days_since_last_trip with renaming.
    """
    user_features = df.groupby('user_id').agg(
        payment_method_id=('payment_method_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        client_type=('client_type', 'first'),
        mode_trip_weekday=('trip_weekday', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trip_hour=('trip_hour', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trip_hour_bucket=('trip_hour_bucket', lambda x: x.mode()[0] if not x.mode().empty else 'unknown'),
        mode_used_midday=('used_midday', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        reg_to_first_trip_days=('reg_to_first_trip_days', 'first'),
        days_since_last_trip=('days_since_last_trip', 'first'),
        is_churn_risk=('is_churn_risk', 'first'),
        user_lifetime_days=('user_lifetime_days', 'first'),
        mode_origin_area_id=('origin_area_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_destination_area_id=('destination_area_id', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_trips_cross_area=('trips_cross_area', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mean_cluster_center_distance_km=('cluster_center_distance_km', 'mean'),
        mode_distance_category=('distance_category', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_hex_origin=('hex_origin', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        mode_hex_destination=('hex_destination', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        avg_distance_km=('distance_km', 'mean'),
        avg_duration_min=('duration_min', 'mean'),
        avg_trip_base=('trip_base', 'mean'),
        avg_trip_total=('trip_total', 'mean'),
        completed_trips_percent=('completed_trips_percent', 'first'),
        mean_discount_percent=('discount_percent', 'mean'),
        has_used_any_discount=('used_discount_flag', 'max'),
        has_used_any_offer=('offer_used', 'max'),
        total_offer_successes=('offer_success', 'sum'),
        has_any_offer_success=('offer_success', 'max')
    ).reset_index()

    for col in user_features.columns:
        if col.startswith('mode_') and col not in ['mode_trip_hour_bucket', 'mode_hex_origin', 'mode_hex_destination']:
            user_features[col] = user_features[col].fillna(0).astype(int)
        elif col in ['mode_trip_hour_bucket', 'mode_hex_origin', 'mode_hex_destination']:
            user_features[col] = user_features[col].fillna('unknown')
        elif user_features[col].dtype == 'object':
            user_features[col] = user_features[col].fillna('unknown')
        elif user_features[col].dtype in ['float64', 'int64', 'bool']:
            user_features[col] = user_features[col].fillna(0)

    user_features = user_features.rename(columns={'days_since_last_trip': 'time_since_last_activity_days'})

    return user_features


def add_embeddings_to_df(df_with_vectors, user_mapping, offer_mapping, user_emb_32d, offer_emb_32d, pca_user, pca_offer):
    """Adds user and offer embeddings (prediction-time, using saved models)."""
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_offer_mapping = {v: k for k, v in offer_mapping.items()}

    def get_user_emb(user_id):
        idx = reverse_user_mapping.get(user_id)
        if idx is not None:
            return pca_user.transform([user_emb_32d[idx]])[0][0]
        return 0.0

    def get_offer_emb(offer_id):
        idx = reverse_offer_mapping.get(offer_id)
        if idx is not None:
            return pca_offer.transform([offer_emb_32d[idx]])[0][0]
        return 0.0

    df_with_vectors['user_embedding'] = df_with_vectors['user_id'].apply(get_user_emb)
    df_with_vectors['offer_embedding'] = df_with_vectors['offer_id'].apply(get_offer_emb)

    return df_with_vectors


def join_with_agg(churn_users, offer_agg, user_agg):
    """Prediction-time final join."""
    churn_users = _ensure_float64_numeric(churn_users)
    offer_agg = _ensure_float64_numeric(offer_agg)
    user_agg = _ensure_float64_numeric(user_agg)

    con = duckdb.connect()
    con.register("df_with_vectors", churn_users)
    con.register("offer_agg", offer_agg)
    con.register("user_agg", user_agg)

    result_for_model = con.execute("""
        SELECT
            v.user_id,
            v.offer_id,
            v.user_embedding,
            v.offer_embedding,
            v.avg_distance_km_offer_user,
            v.avg_duration_min_offer_user,
            v.avg_discount_offer_user,
            v.avg_trip_base_offer_user,
            v.avg_trip_total_offer_user,
            v.offer_used_times_offer_user,
            v.offer_completed_times_offer_user,
            v.offer_conversion_rate_offer_user,

            u.payment_method_id AS user_F_payment_method_id,
            u.client_type AS user_F_client_type,
            u.mode_trip_weekday AS user_F_mode_trip_weekday,
            u.mode_trip_hour AS user_F_mode_trip_hour,
            u.mode_trip_hour_bucket AS user_F_mode_trip_hour_bucket,
            u.mode_used_midday AS user_F_mode_used_midday,
            u.reg_to_first_trip_days AS user_F_reg_to_first_trip_days,
            u.is_churn_risk AS user_F_is_churn_risk,
            u.user_lifetime_days AS user_F_user_lifetime_days,
            u.mode_origin_area_id AS user_F_mode_origin_area_id,
            u.mode_destination_area_id AS user_F_mode_destination_area_id,
            u.mode_trips_cross_area AS user_F_mode_trips_cross_area,
            u.mean_cluster_center_distance_km AS user_F_mean_cluster_center_distance_km,
            u.mode_distance_category AS user_F_mode_distance_category,
            u.mode_hex_origin AS user_F_mode_hex_origin,
            u.mode_hex_destination AS user_F_mode_hex_destination,
            u.avg_distance_km AS user_F_avg_distance_km,
            u.avg_duration_min AS user_F_avg_duration_min,
            u.avg_trip_base AS user_F_avg_trip_base,
            u.avg_trip_total AS user_F_avg_trip_total,
            u.completed_trips_percent AS user_F_completed_trips_percent,
            u.mean_discount_percent AS user_F_mean_discount_percent,
            u.has_used_any_discount AS user_F_has_used_any_discount,
            u.has_used_any_offer AS user_F_has_used_any_offer,
            u.total_offer_successes AS user_F_total_offer_successes,
            u.has_any_offer_success AS user_F_has_any_offer_success,

            p.offer_global_avg_distance_km AS offer_F_global_avg_distance_km,
            p.offer_global_avg_duration_min AS offer_F_global_avg_duration_min,
            p.offer_global_avg_trip_base AS offer_F_global_avg_trip_base,
            p.offer_global_avg_trip_total AS offer_F_global_avg_trip_total,
            p.offer_global_avg_discount AS offer_F_global_avg_discount,
            p.offer_global_avg_use_percent AS offer_F_global_avg_use_percent,
            p.offer_global_avg_use_amount AS offer_F_global_avg_use_amount,
            p.offer_global_avg_usage_count AS offer_F_global_avg_usage_count,
            p.offer_type_id AS offer_F_type_id,
            p.offer_conversion_rate AS offer_F_conversion_rate,
            p.offer_completed_count AS offer_F_completed_count,
            p.offer_usage_count AS offer_F_usage_count,
            p.offer_user_avg_lifetime AS offer_F_user_avg_lifetime,
            p.offer_user_avg_completion_rate AS offer_F_user_avg_completion_rate,
            p.offer_user_churn_risk_ratio AS offer_F_user_churn_risk_ratio,
            p.car_category AS offer_F_car_category,
            p.offer_avg_saving_per_trip AS offer_F_avg_saving_per_trip,
            p.offer_value_ratio AS offer_F_value_ratio,
            p.offer_churn_risk_usage AS offer_F_churn_risk_usage,
            p.offer_healthy_user_ratio AS offer_F_healthy_user_ratio

        FROM df_with_vectors v
        INNER JOIN user_agg u
            ON v.user_id = u.user_id
        LEFT JOIN offer_agg p
            ON v.offer_id = p.offer_id
    """).df()

    return result_for_model