import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
import json
from pathlib import Path
from typing import Optional, Dict, List, Union

from config import MODELS_DIR, OUTPUT_DIR

def save_classification_report(
    y_test: pd.Series,
    y_pred: pd.Series,
    segment_id: str,
    label_encoder: LabelEncoder
) -> str:
    from sklearn.metrics import classification_report
    import pandas as pd
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    classes_df = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    classes_df.index = label_encoder.inverse_transform([int(idx) for idx in classes_df.index])
    
    csv_path = OUTPUT_DIR / f'classification_report_segment_{segment_id}.csv'
    classes_df.to_csv(csv_path)
    
    return csv_path


def _train_single_model(
    data: pd.DataFrame,
    segment_id: str,
    segment_filter: Union[str, List[str]],
    feature_columns: list,
    label_encoder: LabelEncoder,
    results: dict,
    min_samples: int = 3,
) -> None:
    result_key = f"segment_{segment_id}"
    
    if isinstance(segment_filter, list):
        segment_data = data[data['entity_segment'].isin(segment_filter)]
        log_name = f"segments {', '.join(map(str, segment_filter))}"
    else:
        segment_data = data[data['entity_segment'] == segment_filter]
        log_name = f"segment {segment_filter}"
    
    logger.info(f"Training model for {log_name}")
    logger.info(f"  Samples available: {len(segment_data)}")
    
    if len(segment_data) < min_samples:
        logger.warning(f"  SKIPPED: only {len(segment_data)} samples (min required: {min_samples})")
        results[result_key] = {
            'samples': len(segment_data),
            'saved': False,
            'reason': f'Not enough data: {len(segment_data)} < {min_samples}'
        }
        return
    
    X = segment_data[feature_columns]
    y = segment_data['category_encoded']
    
    logger.info(f"  Categories: {y.nunique()}, class balance: {dict(y.value_counts())}")
    
    try:
        if y.nunique() < 2:
            logger.warning(f"  SKIPPED: only one class present")
            results[result_key] = {
                'samples': len(segment_data),
                'saved': False,
                'reason': 'Only one class present'
            }
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        save_classification_report(y_test, y_pred, segment_id, label_encoder)
        
        model_name = f"model_segment_{segment_id}"
        model_file_name = f"{model_name}.pkl"
        model_path = MODELS_DIR / model_file_name
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': rf_model,
                'scaler': scaler,
                'segment': str(segment_id),
                'label_encoder': label_encoder
            }, f)
        
        logger.info(f"  Model saved: {model_path}")
        
        results[result_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': len(segment_data),
            'saved': True,
            'model_path': str(model_path)
        }
        logger.info(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
    except Exception as e:
        logger.error(f"  Error training: {str(e)}")
        results[result_key] = {
            'samples': len(segment_data),
            'saved': False,
            'reason': str(e)
        }


def train_all_models(
    final_with_metrics_ml_path: str,
):
    metrics_file_path = Path(final_with_metrics_ml_path)
    
    if not metrics_file_path.exists():
        raise ValueError(f"final_with_metrics_ml_path not found: {metrics_file_path}")

    final_with_metrics_ml = pd.read_parquet(metrics_file_path)

    if final_with_metrics_ml.empty:
        raise ValueError("final_with_metrics_ml is empty")
    
    if final_with_metrics_ml is None or final_with_metrics_ml.empty:
        logger.error("final_with_metrics_ml is required")
        raise ValueError("final_with_metrics_ml is required or empty")
    
    try:
        
        label_encoder = LabelEncoder()
        final_with_metrics_ml['category_encoded'] = label_encoder.fit_transform(
            final_with_metrics_ml['category']
        )
        
        columns_to_exclude = [
            'entity_id', 'category', 'registration_date', 'processing_method', 
            'preferred_period', 'category_encoded'
        ]
        feature_columns = [
            col for col in final_with_metrics_ml.columns 
            if col not in columns_to_exclude
        ]
        
        results = {}
        
        segment_config = [
            {"id": "A", "filter": "A"},
            {"id": "B", "filter": "B"},
            {"id": "C", "filter": "C"},
            {"id": "D_E_F", "filter": ["D", "E", "F"]},
        ]
        
        for config in segment_config:
            _train_single_model(
                data=final_with_metrics_ml,
                segment_id=config["id"],
                segment_filter=config["filter"],
                feature_columns=feature_columns,
                label_encoder=label_encoder,
                results=results
            )
        
        summary_path = OUTPUT_DIR / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        models_trained = sum(1 for r in results.values() if r.get('saved', False))
        
        logger.info(f"Training completed. Models trained: {models_trained}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise