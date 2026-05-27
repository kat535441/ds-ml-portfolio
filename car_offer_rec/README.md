# Car Offer Recommendation System

Enterprise-grade ML service for personalized car offer recommendations using collaborative filtering, geographic context, and temporal user behavior analysis.

## Architecture

Data is stored in Azure Lakehouse. Training runs via ClearML Pipeline. Prediction runs as a FastAPI service.
Azure Lakehouse → ClearML Pipeline (train) → ClearML Model Registry → FastAPI (predict) → Azure Blob (results)

text

## Project Structure
car_offer_rec/
├── src/
│ └── car_offer_rec/
│ ├── config.py # All settings via environment variables
│ ├── dataset.py # Data loading from Azure Lakehouse
│ ├── features.py # Feature engineering (temporal split, geo, ALS, PCA)
│ ├── train_pipeline.py # ClearML pipeline orchestration
│ ├── modeling/
│ │ ├── train.py # CatBoost model training
│ │ └── predict.py # Prediction and recommendations
│ └── api/
│ └── api.py # FastAPI endpoint
├── Dockerfile.train # Training container
├── Dockerfile.api # API container
├── pyproject.toml # Python dependencies
└── README.md

text

## Key Features

| Feature | Description |
|---------|-------------|
| **Temporal Split** | No data leakage — cutoff detection with 15-day gaps |
| **Geographic Context** | H3 hexagons + KMeans clustering of origin/destination |
| **Collaborative Filtering** | ALS embeddings for user-offer interactions |
| **PCA Compression** | 32d → 1d embedding compression |
| **CatBoost Classification** | AUC-optimized model with scale_pos_weight |
| **MLOps Ready** | ClearML pipeline, model registry, Azure storage |

## Quick Start

### 1. Configure environment

Create `.env` file with your credentials:

```env
AZURE_STORAGE_ACCOUNT=your_account
AZURE_STORAGE_KEY=your_key
AZURE_CONTAINER_NAME=caroffer
SPN_ID=your_spn_id
SPN_SECRET=your_spn_secret
SPN_TENANT_ID=your_tenant_id
CLEARML_ACCESS_KEY=your_clearml_key
CLEARML_SECRET_KEY=your_clearml_secret
2. Run training pipeline
bash
docker build -f Dockerfile.train -t car-offer-rec-train .
docker run --env-file .env car-offer-rec-train --buildNum 1
3. Run API locally
bash
docker build -f Dockerfile.api -t car-offer-rec-api .
docker run -p 8000:8000 --env-file .env car-offer-rec-api
4. Get recommendations
bash
curl -X POST http://localhost:8000/get_car_offer_recommendations \
  -H "Content-Type: application/json" \
  -d '{"models_dir": "models"}'
Response:

json
{
  "status": "success",
  "prediction_path": "caroffer/predictions/predictions.csv",
  "local_path": "predictions/predictions_abc123.csv"
}

Features Computed

User Features

* trip_weekday, trip_hour, trip_hour_bucket

* origin_area_id, destination_area_id (KMeans clusters)

* hex_origin, hex_destination (H3 geohashes)

* avg_distance_km, avg_duration_min

* completed_trips_percent

* is_churn_risk

Offer Features

* offer_global_avg_distance_km, offer_global_avg_duration_min

* offer_conversion_rate, offer_usage_count

* offer_avg_saving_per_trip

* car_category

ALS Embeddings

* user_embedding (1d PCA from 32d ALS)

* offer_embedding (1d PCA from 32d ALS)

Requirements
* Python 3.12+

* Docker

* Azure subscription (Data Lake, Blob Storage)

* ClearML account