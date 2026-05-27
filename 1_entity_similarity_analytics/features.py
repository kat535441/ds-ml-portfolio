import pandas as pd
from loguru import logger
from typing import Tuple, Any
from pathlib import Path

from config import DATA_DIR

def correct_registration_dates(
    df_sessions: pd.DataFrame,
    completed_transactions: pd.DataFrame
) -> dict:
    if df_sessions is None or df_sessions.empty:
        raise ValueError("df_sessions is required")
    
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    
    try:
        df_sessions['registration_date'] = pd.to_datetime(df_sessions['registration_date'])
        df_sessions['session_start'] = pd.to_datetime(df_sessions['session_start'])
        
        registration_dates_from_sessions = df_sessions.groupby('entity_id')['registration_date'].first()
        first_session_start = df_sessions.groupby('entity_id')['session_start'].min()
        
        our_entities = set(completed_transactions['entity_id'])
        
        comparison_data = []
        for entity_id in our_entities:
            registration_date = registration_dates_from_sessions.get(entity_id)
            first_session_date = first_session_start.get(entity_id)
            
            comparison_data.append({
                'entity_id': entity_id,
                'registration_date': registration_date,
                'first_session_date': first_session_date
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        def get_correct_registration_date(registration_date, first_session_date):
            current_date = pd.Timestamp.now()
            min_valid_date = pd.Timestamp('2020-01-01')
            
            if pd.notna(registration_date):
                registration_valid = (
                    registration_date <= current_date and 
                    registration_date >= min_valid_date and
                    (pd.isna(first_session_date) or registration_date <= first_session_date)
                )
                if registration_valid:
                    return registration_date
            
            if pd.notna(first_session_date):
                session_valid = (
                    first_session_date <= current_date and 
                    first_session_date >= min_valid_date
                )
                if session_valid:
                    return first_session_date
            
            return None
        
        correct_registration_dates_list = []
        for _, row in comparison_df.iterrows():
            correct_date = get_correct_registration_date(row['registration_date'], row['first_session_date'])
            correct_registration_dates_list.append(correct_date)
        
        comparison_df['correct_registration_date'] = correct_registration_dates_list
        
        correct_registration_dict = comparison_df.set_index('entity_id')['correct_registration_date'].to_dict()
        
        return correct_registration_dict
    
    except Exception as e:
        logger.error(f"Error correcting registration dates: {e}")
        raise
    

def calculate_entity_metrics(
    completed_transactions: pd.DataFrame,
    abandoned_transactions: pd.DataFrame,
    correct_registration_dict: dict
) -> pd.DataFrame:
    
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    if abandoned_transactions is None:
        raise ValueError("abandoned_transactions is required")
    if correct_registration_dict is None:
        raise ValueError("correct_registration_dict is required")
    try:
        entity_metrics = []
        
        last_transaction_date = completed_transactions['transaction_created'].max()
        
        for entity_id in completed_transactions['entity_id'].unique():
            entity_data = completed_transactions[completed_transactions['entity_id'] == entity_id]
            entity_abandoned = abandoned_transactions[abandoned_transactions['entity_id'] == entity_id]
            
            total_transactions = len(entity_data)
            if total_transactions == 0:
                continue
            
            registration_date = correct_registration_dict.get(entity_id)
            
            if pd.notna(registration_date):
                
                if not isinstance(registration_date, pd.Timestamp):
                    logger.warning(f"Registration date for entity {entity_id} is not a Timestamp: {registration_date}")
                    logger.warning(f"Converting to Timestamp...")
                    registration_date = pd.to_datetime(registration_date)
                
                if not isinstance(last_transaction_date, pd.Timestamp):
                    logger.warning(f"Last transaction date for entity {entity_id} is not a Timestamp: {last_transaction_date}")
                    logger.warning(f"Converting to Timestamp...")
                    last_transaction_date = pd.to_datetime(last_transaction_date)
                
                days_active = max(0, (last_transaction_date - registration_date).days)
            else:
                days_active = 0
            
            avg_activity_volume = float(entity_data['activity_volume'].mean())
            min_activity_volume = float(entity_data['activity_volume'].min())
            min_processing_time = float(entity_data['processing_time_sec'].min()) / 60
            small_activities = len(entity_data[entity_data['activity_volume'] <= 2500])
            small_activity_rate = small_activities / total_transactions
            
            high_volume_activities = len(entity_data[
                (entity_data['activity_volume'] > 8000) &
                (entity_data['processing_time_sec'] < 300)
            ])
            
            total_revenue = float(entity_data['revenue'].sum())
            volume_sum = float(entity_data['activity_volume'].sum())
            avg_revenue_per_volume = total_revenue / volume_sum if volume_sum > 0 else 0
            
            if days_active > 0:
                revenue_per_month = total_revenue / (days_active / 30)
            else:
                revenue_per_month = 0
            
            very_small_activities = len(entity_data[
                (entity_data['activity_volume'] <= 500) | 
                (entity_data['processing_time_sec'] <= 120)
            ])
            very_small_activity_rate = very_small_activities / total_transactions
            
            small_high_value_activities = len(entity_data[
                (entity_data['activity_volume'] <= 1000) & 
                (entity_data['revenue'] >= 20)
            ])
            
            abandonment_count = len(entity_abandoned)
            
            entity_metrics.append({
                'entity_id': entity_id,
                'total_transactions': total_transactions,
                'registration_date': registration_date,
                'days_active': days_active,
                'revenue_per_month': revenue_per_month,
                'avg_activity_volume': avg_activity_volume,
                'min_activity_volume': min_activity_volume,
                'min_processing_time': min_processing_time,
                'small_activity_rate': small_activity_rate,
                'total_revenue': total_revenue,
                'avg_revenue_per_volume': avg_revenue_per_volume,
                'very_small_activities': very_small_activities,
                'very_small_activity_rate': very_small_activity_rate,
                'small_high_value_activities': small_high_value_activities,
                'abandonment_count': abandonment_count,
                'high_volume_activities': high_volume_activities  
            })
        
        entity_df = pd.DataFrame(entity_metrics)
    
        return entity_df
    except Exception as e:
        logger.error(f"Error calculating entity metrics: {e}")
        raise


def classify_high_risk_entity(
    row: pd.Series, 
    very_small_activity_rate_threshold=0.1,
    abandonment_threshold=11,
    abandonment_rate_threshold=0.15,
    high_revenue_threshold=80,
    small_high_value_threshold=3,
    extreme_volume=0.1,
    extreme_time=1,
    weight_abandonment_absolute=1,
    weight_abandonment_percent=1,
    weight_very_small_rate=2,
    weight_very_small_count=1,
    weight_high_revenue=1,
    weight_small_high_value=0.5,
    weight_extreme=3,
    weight_high_volume=4,     
    high_risk_threshold=6,
    medium_risk_threshold=2
) -> Tuple[str, float, list]:
    
    if row is None or row.empty:
        raise ValueError("row is required")
    try:
        risk_score = 0
        risk_reasons = []
        
        total_activities = row['total_transactions'] + row['abandonment_count']
        abandon_rate = row['abandonment_count'] / total_activities if total_activities > 0 else 0
        
        if row['abandonment_count'] > abandonment_threshold:
            risk_score += weight_abandonment_absolute
            risk_reasons.append(f"High abandonment count ({row['abandonment_count']})")
        
        if abandon_rate > abandonment_rate_threshold:
            risk_score += weight_abandonment_percent
            risk_reasons.append(f"High abandonment rate ({abandon_rate:.1%})")
        
        if row['very_small_activity_rate'] > very_small_activity_rate_threshold:
            risk_score += weight_very_small_rate
            risk_reasons.append(f"High very small activity rate ({row['very_small_activity_rate']:.1%})")
        
        if row['total_transactions'] < 50:
            count_threshold = 3
        elif row['total_transactions'] < 500:  
            count_threshold = 10
        else:
            count_threshold = 50

        if row['very_small_activities'] > count_threshold:
            risk_score += weight_very_small_count
            risk_reasons.append(f"High absolute very small activities ({row['very_small_activities']})")
        
        if row['avg_revenue_per_volume'] > high_revenue_threshold:
            risk_score += weight_high_revenue
            risk_reasons.append(f"High revenue per volume ({row['avg_revenue_per_volume']:.1f})")
        
        if row['small_high_value_activities'] > small_high_value_threshold:
            risk_score += weight_small_high_value
            risk_reasons.append(f"Small high-value activities ({row['small_high_value_activities']})")
        
        if (row['min_activity_volume'] < extreme_volume or row['min_processing_time'] < extreme_time) and row['very_small_activities'] > 2:
            risk_score += weight_extreme
            risk_reasons.append(f"Extreme small (min volume: {row['min_activity_volume']:.2f}, min time: {row['min_processing_time']:.1f} min)")
        
        if row.get('high_volume_activities', 0) > 0:
            risk_score += weight_high_volume
            risk_reasons.append(f"High volume activities ({row.get('high_volume_activities', 0)} times)")
        
        if risk_score >= high_risk_threshold:
            return 'B', risk_score, risk_reasons
        elif risk_score >= medium_risk_threshold:
            return 'Medium Risk', risk_score, risk_reasons
        else:
            return 'Low Risk', risk_score, risk_reasons
    
    except Exception as e:
        logger.error(f"Error classifying high risk entity: {e}")
        raise
    
def classify_entity_priority(
    entity_row: pd.Series, 
    high_risk_ids: set, 
    low_activity_ids: set
) -> str:
    if entity_row is None or entity_row.empty:
        raise ValueError("entity_row is required")
    if high_risk_ids is None:
        raise ValueError("high_risk_ids is required")
    if low_activity_ids is None:
        raise ValueError("low_activity_ids is required")
    try:
        entity_id = entity_row['entity_id']
        
        if entity_id in high_risk_ids:
            return 'B'
        
        if entity_id in low_activity_ids:
            return 'F'
        
        if entity_row['days_active'] < 30:
            return 'E'
        
        if entity_row['days_active'] >= 30 and entity_row['revenue_per_month'] < 1750:
            return 'D'

        if entity_row['days_active'] >= 30 and entity_row['revenue_per_month'] >= 10000:
            return 'C'
        
        if entity_row['days_active'] >= 30 and 1750 <= entity_row['revenue_per_month'] < 10000:
            return 'A'
        
        return 'Unclassified'
    except Exception as e:
        logger.error(f"Error classifying entity priority: {e}")
        raise
    
def classify_all_entities(
    entity_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if entity_df is None or entity_df.empty:
        raise ValueError("entity_df is required")
    try:
        high_risk_results = []
        
        for _, row in entity_df.iterrows():
            category, score, reasons = classify_high_risk_entity(
                row,
                very_small_activity_rate_threshold=0.1,
                abandonment_threshold=11,
                abandonment_rate_threshold=0.15,
                high_revenue_threshold=80,
                small_high_value_threshold=3,
                extreme_volume=0.1,
                extreme_time=1,
                weight_abandonment_absolute=1,
                weight_abandonment_percent=1,
                weight_very_small_rate=2,
                weight_very_small_count=1,
                weight_high_revenue=1,
                weight_small_high_value=0.5,
                weight_extreme=2,
                weight_high_volume=4,
                high_risk_threshold=7,
                medium_risk_threshold=2
            )
            
            if category == 'B':
                high_risk_results.append({
                    'entity_id': row['entity_id'],
                    'risk_score': score,
                    'reasons': reasons
                })
        
        high_risk_df = pd.DataFrame(high_risk_results)
        if high_risk_df.empty:
            high_risk_ids = set()
        else:
            high_risk_ids = set(high_risk_df['entity_id'])
        
        low_activity_entities = entity_df[
            (entity_df['small_activity_rate'] <= 0.01) &
            (entity_df['total_transactions'] >= 10) &
            (entity_df['days_active'] > 100) &
            (~entity_df['entity_id'].isin(high_risk_ids))
        ]
        low_activity_ids = set(low_activity_entities['entity_id'])
        
        categories = []
        for _, entity_row in entity_df.iterrows():
            category = classify_entity_priority(entity_row, high_risk_ids, low_activity_ids)
            categories.append(category)
        
        final_df = entity_df.copy()
        final_df['category'] = categories

        
        return final_df, high_risk_df
    except Exception as e:
        logger.error(f"Error classifying all entities: {e}")
        raise


def calculate_robust_metrics_for_ml(
    entity_id: str, 
    completed_transactions: pd.DataFrame
) -> dict:
    
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    try:
        entity_transactions = completed_transactions[completed_transactions['entity_id'] == entity_id]
        
        if len(entity_transactions) == 0:
            return {
                'median_activity_volume': 0.0,
                'median_revenue_per_transaction': 0.0,
                'median_processing_time': 0.0
            }
        
        median_activity_volume = entity_transactions['activity_volume'].median()
        median_revenue_per_transaction = entity_transactions['revenue'].median()
        median_processing_time = entity_transactions['processing_time_sec'].median() / 60
        
        return {
            'median_activity_volume': median_activity_volume,
            'median_revenue_per_transaction': median_revenue_per_transaction,
            'median_processing_time': median_processing_time
        }
    except Exception as e:
        logger.error(f"Error calculating robust metrics for ml: {e}")
        raise

def add_median_features(
    final_with_metrics: pd.DataFrame, 
    completed_transactions: pd.DataFrame
) -> pd.DataFrame:
    
    if final_with_metrics is None or final_with_metrics.empty:
        raise ValueError("final_with_metrics is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    try:
        robust_metrics_list = []
        
        for entity_id in final_with_metrics['entity_id']:
            metrics = calculate_robust_metrics_for_ml(entity_id, completed_transactions)
            robust_metrics_list.append(metrics)
        
        robust_metrics_df = pd.DataFrame(robust_metrics_list)
        final_with_metrics_ml = final_with_metrics.copy()
        
        for col in robust_metrics_df.columns:
            final_with_metrics_ml[col] = robust_metrics_df[col]
    
        return final_with_metrics_ml
    except Exception as e:
        logger.error(f"Error adding median features: {e}")
        raise


def get_main_segment(
    entity_id: str, 
    completed_transactions: pd.DataFrame
) -> Any:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    try:
        entity_segments = completed_transactions[completed_transactions['entity_id'] == entity_id]['segment']
        if len(entity_segments) == 0:
            return None
        mode_result = entity_segments.mode()
        if len(mode_result) == 0:
            return None
        return mode_result.iloc[0]
    except Exception as e:
        logger.error(f"Error getting main segment: {e}")
        raise

def add_segment_features(
    final_with_metrics_ml: pd.DataFrame, 
    completed_transactions: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    
    try:
        main_segment_dict = {}
        for entity_id in final_with_metrics_ml['entity_id']:
            main_segment_dict[entity_id] = get_main_segment(entity_id, completed_transactions)
        
        final_with_metrics_ml['entity_segment'] = final_with_metrics_ml['entity_id'].map(main_segment_dict)
        
        segment_per_entity = completed_transactions.groupby('entity_id')['segment'].nunique()
        segment_count_dict = segment_per_entity.to_dict()
        final_with_metrics_ml['segment_count'] = final_with_metrics_ml['entity_id'].map(segment_count_dict)
        
        return final_with_metrics_ml
    except Exception as e:
        logger.error(f"Error adding segment features: {e}")
        raise


def calculate_efficiency_rates(
    entity_id: str, 
    completed_transactions: pd.DataFrame, 
    abandoned_transactions: pd.DataFrame
) -> Tuple[float, float]:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    if abandoned_transactions is None:
        raise ValueError("abandoned_transactions is required")
    try:
        entity_completed = len(completed_transactions[completed_transactions['entity_id'] == entity_id])
        entity_abandoned = len(abandoned_transactions[abandoned_transactions['entity_id'] == entity_id])
        
        total_activities = entity_completed + entity_abandoned
        
        if total_activities > 0:
            success_rate = entity_completed / total_activities
            abandonment_rate = entity_abandoned / total_activities
        else:
            success_rate = 0.0
            abandonment_rate = 0.0
        
        return success_rate, abandonment_rate
    except Exception as e:
        logger.error(f"Error calculating efficiency rates: {e}")
        raise

def add_efficiency_rates(
    final_with_metrics_ml: pd.DataFrame, 
    completed_transactions: pd.DataFrame, 
    abandoned_transactions: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    if abandoned_transactions is None:
        raise ValueError("abandoned_transactions is required")
    success_rates = []
    abandonment_rates = []
    
    for entity_id in final_with_metrics_ml['entity_id']:
        success_rate, abandonment_rate = calculate_efficiency_rates(entity_id, completed_transactions, abandoned_transactions)
        success_rates.append(success_rate)
        abandonment_rates.append(abandonment_rate)
    
    final_with_metrics_ml['success_rate'] = success_rates
    final_with_metrics_ml['abandonment_rate'] = abandonment_rates
    
    return final_with_metrics_ml


def calculate_improved_hybrid_processing_time(
    entity_id: str, 
    df_transactions_clean: pd.DataFrame
) -> Tuple[float, str]:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        entity_transactions = df_transactions_clean[df_transactions_clean['entity_id'] == entity_id].copy()
        
        valid_transactions = entity_transactions[
            entity_transactions['assigned_at'].notna() & 
            entity_transactions['processing_started_at'].notna()
        ].copy()
        
        calculated_result = 0.0
        used_calculation = False
        
        if len(valid_transactions) > 0:
            valid_transactions['assigned_dt'] = pd.to_datetime(valid_transactions['assigned_at'])
            valid_transactions['started_dt'] = pd.to_datetime(valid_transactions['processing_started_at'])
            valid_transactions['calculated_processing_sec'] = (valid_transactions['started_dt'] - valid_transactions['assigned_dt']).dt.total_seconds()
            
            reasonable_times = valid_transactions[valid_transactions['calculated_processing_sec'] <= 28800]
            
            if len(reasonable_times) > 0:
                calculated_result = reasonable_times['calculated_processing_sec'].mean()
                used_calculation = True
        
        if calculated_result == 0 or not used_calculation:
            non_zero_processing = entity_transactions[entity_transactions['processing_time_sec'] > 0]
            if len(non_zero_processing) > 0:
                return float(non_zero_processing['processing_time_sec'].mean()), 'processing_time_fallback'
        
        if calculated_result == 0:
            return 0.0, 'zero_result'
        else:
            return calculated_result, 'calculated_from_metrics'
    except Exception as e:
        logger.error(f"Error calculating improved hybrid processing time: {e}")
        raise

def add_processing_time_features(
    final_with_metrics_ml: pd.DataFrame, 
    df_transactions_clean: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        improved_processing_times = []
        method_used = []
        
        for entity_id in final_with_metrics_ml['entity_id']:
            avg_processing_time, method = calculate_improved_hybrid_processing_time(entity_id, df_transactions_clean)
            improved_processing_times.append(avg_processing_time)
            method_used.append(method)
        
        final_with_metrics_ml['avg_processing_time_sec'] = improved_processing_times
        final_with_metrics_ml['processing_method'] = method_used
        final_with_metrics_ml['avg_processing_time_original'] = final_with_metrics_ml['avg_processing_time_sec']
        
        all_non_zero_processing = df_transactions_clean[df_transactions_clean['processing_time_sec'] > 0]
        median_processing_time = all_non_zero_processing['processing_time_sec'].median()
        
        zeros_replaced_mask = final_with_metrics_ml['avg_processing_time_sec'] == 0
        final_with_metrics_ml.loc[zeros_replaced_mask, 'avg_processing_time_sec'] = median_processing_time
        final_with_metrics_ml.loc[zeros_replaced_mask, 'processing_method'] = 'median_replacement'
        
        return final_with_metrics_ml
    except Exception as e:
        logger.error(f"Error adding processing time features: {e}")
        raise


def calculate_revenue_per_hour(
    entity_id: str, 
    df_transactions_clean: pd.DataFrame
) -> float:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        entity_transactions = df_transactions_clean[df_transactions_clean['entity_id'] == entity_id].copy()
        
        if len(entity_transactions) == 0:
            return 0.0
        
        entity_transactions['transaction_created'] = pd.to_datetime(entity_transactions['transaction_created'])
        entity_transactions['transaction_completed'] = pd.to_datetime(entity_transactions['transaction_completed'])
        
        completed_transactions = entity_transactions[
            (entity_transactions['status_id'] == 7) & 
            (entity_transactions['transaction_created'].notna()) & 
            (entity_transactions['transaction_completed'].notna())
        ].copy()
        
        if len(completed_transactions) == 0:
            return 0.0
        
        total_revenue = float(completed_transactions['revenue'].sum())
        
        first_transaction = completed_transactions['transaction_created'].min()
        last_transaction = completed_transactions['transaction_completed'].max()
        total_work_hours = (last_transaction - first_transaction).total_seconds() / 3600
        
        if total_work_hours < len(completed_transactions) * 0.1:
            total_processing_hours = float(completed_transactions['processing_time_sec'].sum()) / 3600
            total_work_hours = total_processing_hours * 1.2
        
        if total_work_hours > 0:
            return total_revenue / total_work_hours
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating revenue per hour: {e}")
        raise
    
def calculate_revenue_growth(
    entity_id: str, 
    df_transactions_clean: pd.DataFrame
) -> float:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        entity_transactions = df_transactions_clean[df_transactions_clean['entity_id'] == entity_id].copy()
        
        if len(entity_transactions) < 2:
            return 0.0
        
        entity_transactions['transaction_created'] = pd.to_datetime(entity_transactions['transaction_created'])
        entity_transactions = entity_transactions.sort_values('transaction_created')
        
        first_transaction_date = entity_transactions['transaction_created'].min()
        last_transaction_date = entity_transactions['transaction_created'].max()
        entity_experience_days = (last_transaction_date - first_transaction_date).days
        
        if entity_experience_days < 28:
            return 0.0
        
        first_two_weeks_end = first_transaction_date + pd.Timedelta(days=14)
        last_two_weeks_start = last_transaction_date - pd.Timedelta(days=14)
        
        first_period = entity_transactions[
            (entity_transactions['transaction_created'] >= first_transaction_date) & 
            (entity_transactions['transaction_created'] <= first_two_weeks_end)
        ]
        
        last_period = entity_transactions[
            (entity_transactions['transaction_created'] >= last_two_weeks_start) & 
            (entity_transactions['transaction_created'] <= last_transaction_date)
        ]
        
        if len(first_period) > 0 and len(last_period) > 0:
            first_avg = float(first_period['revenue'].mean())
            last_avg = float(last_period['revenue'].mean())
            
            if first_avg > 0:
                growth = (last_avg - first_avg) / first_avg
                return growth
            else:
                return 0.0
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating revenue growth: {e}")
        raise

def calculate_revenue_per_day(
    entity_id: str, 
    df_transactions_clean: pd.DataFrame
) -> float:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        entity_transactions = df_transactions_clean[df_transactions_clean['entity_id'] == entity_id].copy()
        
        if len(entity_transactions) == 0:
            return 0.0
        
        entity_transactions['transaction_created'] = pd.to_datetime(entity_transactions['transaction_created'])
        
        completed_transactions = entity_transactions[entity_transactions['status_id'] == 7]
        
        if len(completed_transactions) == 0:
            return 0.0
        
        total_revenue = float(completed_transactions['revenue'].sum())
        work_dates = completed_transactions['transaction_created'].dt.date.nunique()
        
        if work_dates > 0:
            return total_revenue / work_dates
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating revenue per day: {e}")
        raise

def add_revenue_features(
    final_with_metrics_ml: pd.DataFrame, 
    df_transactions_clean: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        revenue_per_hour_list = []
        revenue_growth_list = []
        revenue_per_day_list = []
        
        for entity_id in final_with_metrics_ml['entity_id']:
            revenue_per_hour_list.append(calculate_revenue_per_hour(entity_id, df_transactions_clean))
            revenue_growth_list.append(calculate_revenue_growth(entity_id, df_transactions_clean))
            revenue_per_day_list.append(calculate_revenue_per_day(entity_id, df_transactions_clean))
        
        final_with_metrics_ml['revenue_per_hour'] = revenue_per_hour_list
        final_with_metrics_ml['revenue_growth'] = revenue_growth_list
        final_with_metrics_ml['revenue_per_day'] = revenue_per_day_list
        
        return final_with_metrics_ml
    
    except Exception as e:
        logger.error(f"Error adding revenue features: {e}")
        raise


def get_time_period(
    hour: int
) -> str:
    try:
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'day'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    except Exception as e:
        logger.error(f"Error getting time period: {e}")
        raise

def add_behavioral_features(
    final_with_metrics_ml: pd.DataFrame, 
    df_sessions: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if df_sessions is None or df_sessions.empty:
        raise ValueError("df_sessions is required")
    try:
        df_sessions_clean = df_sessions.copy()
        df_sessions_clean = df_sessions_clean[df_sessions_clean['session_end'].notna()]
        df_sessions_clean['session_start'] = pd.to_datetime(df_sessions_clean['session_start'])
        df_sessions_clean['session_end'] = pd.to_datetime(df_sessions_clean['session_end'])
        df_sessions_clean = df_sessions_clean[df_sessions_clean['session_end'] > df_sessions_clean['session_start']]
        
        df_sessions_clean['session_duration_hours'] = (
            df_sessions_clean['session_end'] - df_sessions_clean['session_start']
        ).dt.total_seconds() / 3600
        
        df_sessions_clean = df_sessions_clean[
            (df_sessions_clean['session_duration_hours'] >= 0.083) & 
            (df_sessions_clean['session_duration_hours'] <= 24)
        ]
        
        df_sessions_clean['session_start_hour'] = df_sessions_clean['session_start'].dt.hour
        df_sessions_clean['start_period'] = df_sessions_clean['session_start_hour'].apply(get_time_period)
        df_sessions_clean['session_date'] = df_sessions_clean['session_start'].dt.date
        
        entity_behavior_features = df_sessions_clean.groupby('entity_id').agg({
            'session_date': 'nunique'
        }).round(2)
        entity_behavior_features.columns = ['active_days']
        
        total_hours = df_sessions_clean.groupby('entity_id')['session_duration_hours'].sum()
        entity_behavior_features['avg_hours_per_day'] = (total_hours / entity_behavior_features['active_days']).round(2)
        
        period_counts = df_sessions_clean.groupby(['entity_id', 'start_period']).size().unstack(fill_value=0)
        expected_periods = ['morning', 'day', 'evening', 'night']
        for period in expected_periods:
            if period not in period_counts.columns:
                period_counts[period] = 0
        period_counts['preferred_period'] = period_counts[expected_periods].idxmax(axis=1)
        entity_behavior_features = entity_behavior_features.merge(
            period_counts[['preferred_period']], 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        entity_behavior_features_reset = entity_behavior_features.reset_index()
        final_with_metrics_ml = final_with_metrics_ml.merge(
            entity_behavior_features_reset,
            left_on='entity_id',
            right_on='entity_id',
            how='left'
        )
        
        final_with_metrics_ml['active_days'] = final_with_metrics_ml['active_days'].fillna(final_with_metrics_ml['active_days'].median())
        final_with_metrics_ml['avg_hours_per_day'] = final_with_metrics_ml['avg_hours_per_day'].fillna(final_with_metrics_ml['avg_hours_per_day'].median())
        final_with_metrics_ml['preferred_period'] = final_with_metrics_ml['preferred_period'].fillna('morning')
        
        return final_with_metrics_ml
    
    except Exception as e:
        logger.error(f"Error adding behavioral features: {e}")
        raise


def add_bonus_features(
    final_with_metrics_ml: pd.DataFrame, 
    df_bonuses: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if df_bonuses is None or df_bonuses.empty:
        raise ValueError("df_bonuses is required")
    try:
        df_bonuses_clean = df_bonuses.copy()
        df_bonuses_clean['bonus_date'] = pd.to_datetime(df_bonuses_clean['bonus_date'])
        df_bonuses_clean['bonus_start'] = pd.to_datetime(df_bonuses_clean['bonus_start'])
        df_bonuses_clean['bonus_end'] = pd.to_datetime(df_bonuses_clean['bonus_end'])
        
        bonus_features = df_bonuses_clean.groupby('entity_id').agg({
            'bonus_id': 'count',
            'completed_transactions': 'sum',
            'reward_amount': 'sum',
            'reward_received': 'sum'
        }).reset_index()
        
        bonus_features.columns = [
            'entity_id',
            'total_bonuses',
            'total_bonus_transactions',
            'total_bonus_rewards',
            'successful_bonuses'
        ]
        
        for col in ['total_bonuses', 'total_bonus_transactions', 'total_bonus_rewards', 'successful_bonuses']:
            bonus_features[col] = bonus_features[col].astype(float)
        
        bonus_features['bonus_participation'] = (bonus_features['total_bonuses'] > 0).astype(int)
        bonus_features['bonus_success_rate'] = (
            bonus_features['successful_bonuses'] / bonus_features['total_bonuses']
        ).round(3)
        bonus_features['avg_bonus_reward'] = (
            bonus_features['total_bonus_rewards'] / bonus_features['total_bonuses']
        ).round(2)
        bonus_features['avg_transactions_per_bonus'] = (
            bonus_features['total_bonus_transactions'] / bonus_features['total_bonuses']
        ).round(1)
        
        bonus_features['bonus_success_rate'] = bonus_features['bonus_success_rate'].fillna(0)
        
        final_with_metrics_ml = final_with_metrics_ml.merge(
            bonus_features,
            on='entity_id',
            how='left'
        )
        
        bonus_fill_values = {
            'total_bonuses': 0,
            'total_bonus_transactions': 0,
            'total_bonus_rewards': 0,
            'successful_bonuses': 0,
            'bonus_participation': 0,
            'bonus_success_rate': 0,
            'avg_bonus_reward': 0,
            'avg_transactions_per_bonus': 0
        }
        
        final_with_metrics_ml = final_with_metrics_ml.fillna(bonus_fill_values)
        
        return final_with_metrics_ml
    
    except Exception as e:
        logger.error(f"Error adding bonus features: {e}")
        raise


def calculate_transactions_per_hour(
    entity_id: str, 
    df_transactions_clean: pd.DataFrame
) -> float:
    if entity_id is None or entity_id == '' or entity_id == 'nan':
        raise ValueError("entity_id is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        entity_transactions = df_transactions_clean[df_transactions_clean['entity_id'] == entity_id].copy()
        
        if len(entity_transactions) == 0:
            return 0.0
        
        entity_transactions['transaction_created'] = pd.to_datetime(entity_transactions['transaction_created'])
        entity_transactions['transaction_completed'] = pd.to_datetime(entity_transactions['transaction_completed'])
        
        completed_transactions = entity_transactions[
            (entity_transactions['status_id'] == 7) & 
            (entity_transactions['transaction_created'].notna()) & 
            (entity_transactions['transaction_completed'].notna())
        ].copy()
        
        if len(completed_transactions) == 0:
            return 0.0
        
        total_transactions = len(completed_transactions)
        
        first_transaction = completed_transactions['transaction_created'].min()
        last_transaction = completed_transactions['transaction_completed'].max()
        total_work_hours = (last_transaction - first_transaction).total_seconds() / 3600
        
        if total_work_hours < len(completed_transactions) * 0.1:
            total_processing_hours = float(completed_transactions['processing_time_sec'].sum()) / 3600
            total_work_hours = total_processing_hours * 1.5
        
        if total_work_hours > 0:
            return total_transactions / total_work_hours
        else:
            return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating transactions per hour: {e}")
        raise

def add_productivity_features(
    final_with_metrics_ml: pd.DataFrame, 
    df_transactions_clean: pd.DataFrame
) -> pd.DataFrame:
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    try:
        transactions_per_hour_list = []
        
        for entity_id in final_with_metrics_ml['entity_id']:
            productivity = calculate_transactions_per_hour(entity_id, df_transactions_clean)
            transactions_per_hour_list.append(productivity)
        
        final_with_metrics_ml['transactions_per_hour'] = transactions_per_hour_list
        return final_with_metrics_ml
    except Exception as e:
        logger.error(f"Error adding productivity features: {e}")
        raise


def add_one_hot_encoding(
    final_with_metrics_ml: pd.DataFrame
) -> pd.DataFrame:
    
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is required")
    try:
        expected_periods = ['morning', 'day', 'evening', 'night']
        final_with_metrics_ml['preferred_period'] = pd.Categorical(
            final_with_metrics_ml['preferred_period'],
            categories=expected_periods,
        )
        
        columns_to_encode = ['preferred_period']
        one_hot_encoded = pd.get_dummies(
            final_with_metrics_ml[columns_to_encode],
            prefix=columns_to_encode,
            dtype=int,
        )
        final_with_metrics_ml = pd.concat(
            [
                final_with_metrics_ml, 
                one_hot_encoded
            ], 
            axis=1,
        )
        
        return final_with_metrics_ml
    except Exception as e:
        logger.error(f"Error adding one-hot encoding: {e}")
        raise


def create_ml_features(    
    final_with_metrics: pd.DataFrame,
    completed_transactions: pd.DataFrame,
    abandoned_transactions: pd.DataFrame,
    df_transactions_clean: pd.DataFrame,
    df_sessions: pd.DataFrame,
    df_bonuses: pd.DataFrame
) -> pd.DataFrame:
    
    if final_with_metrics is None or final_with_metrics.empty:
        raise ValueError("final_with_metrics is required")
    if completed_transactions is None or completed_transactions.empty:
        raise ValueError("completed_transactions is required")
    if abandoned_transactions is None:
        raise ValueError("abandoned_transactions is required")
    if df_transactions_clean is None or df_transactions_clean.empty:
        raise ValueError("df_transactions_clean is required")
    if df_sessions is None or df_sessions.empty:
        raise ValueError("df_sessions is required")
    if df_bonuses is None or df_bonuses.empty:
        raise ValueError("df_bonuses is required")
    try:
        logger.info("Creating ml features...")
        
        logger.info("  Adding median features...")
        final_with_metrics_ml = add_median_features(final_with_metrics, completed_transactions)
        
        logger.info("  Adding segment features...")
        final_with_metrics_ml = add_segment_features(final_with_metrics_ml, completed_transactions)
        
        logger.info("  Adding success and abandonment rates...")
        final_with_metrics_ml = add_efficiency_rates(final_with_metrics_ml, completed_transactions, abandoned_transactions)
        
        logger.info("  Adding processing time...")
        final_with_metrics_ml = add_processing_time_features(final_with_metrics_ml, df_transactions_clean)
        
        logger.info("  Adding revenue features...")
        final_with_metrics_ml = add_revenue_features(final_with_metrics_ml, df_transactions_clean)
        
        logger.info("  Adding behavioral features...")
        final_with_metrics_ml = add_behavioral_features(final_with_metrics_ml, df_sessions)
        
        logger.info("  Adding bonus features...")
        final_with_metrics_ml = add_bonus_features(final_with_metrics_ml, df_bonuses)
        
        logger.info("  Adding productivity...")
        final_with_metrics_ml = add_productivity_features(final_with_metrics_ml, df_transactions_clean)
        
        logger.info("  Adding one-hot encoding...")
        final_with_metrics_ml = add_one_hot_encoding(final_with_metrics_ml)
        
        return final_with_metrics_ml
    
    except Exception as e:
        logger.error(f"Error creating ml features: {e}")
        raise
    
def create_features(
    df_sessions_path: str,
    completed_transactions_path: str,
    abandoned_transactions_path: str,
    df_transactions_clean_path: str,
    df_bonuses_path: str,
) -> str:
    
    try:
        df_sessions_path = Path(df_sessions_path)
        completed_transactions_path = Path(completed_transactions_path)
        abandoned_transactions_path = Path(abandoned_transactions_path)
        df_transactions_clean_path = Path(df_transactions_clean_path)
        df_bonuses_path = Path(df_bonuses_path)


        df_sessions = pd.read_parquet(df_sessions_path)
        completed_transactions = pd.read_parquet(completed_transactions_path)
        abandoned_transactions = pd.read_parquet(abandoned_transactions_path)
        df_transactions_clean = pd.read_parquet(df_transactions_clean_path)
        df_bonuses = pd.read_parquet(df_bonuses_path)

        if df_sessions.empty:
            raise ValueError("df_sessions is empty")
        if completed_transactions.empty:
            raise ValueError("completed_transactions is empty")
        if df_transactions_clean.empty:
            raise ValueError("df_transactions_clean is empty")
        if df_bonuses.empty:
            raise ValueError("df_bonuses is empty")

        correct_registration_dict = correct_registration_dates(df_sessions, completed_transactions)
        entity_df = calculate_entity_metrics(completed_transactions, abandoned_transactions, correct_registration_dict)
        final_with_metrics, _ = classify_all_entities(entity_df)
        final_with_metrics_ml = create_ml_features(
            final_with_metrics,
            completed_transactions,
            abandoned_transactions,
            df_transactions_clean,
            df_sessions,
            df_bonuses,
        )
        
        out_dir = DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "final_with_metrics_ml.parquet"
        final_with_metrics_ml.to_parquet(out_path, index=False)

        return str(out_path)

    except Exception as e:
        logger.exception(f"Error creating features: {e}")
        raise