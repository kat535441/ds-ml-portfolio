# Entity Similarity Analytics Service

Enterprise-grade ML service for entity performance analytics using vector similarity and automated KPI recommendations.

## Architecture

Data is stored in S3. Training runs via GitHub Actions. Prediction runs as a FastAPI service.

S3 (data) → GitHub Actions (train) → S3 (models) → FastAPI (predict) → S3 (results)

## Project Structure

- config.py - All settings via environment variables
- dataset.py - Data loading and cleaning from S3
- features.py - Feature engineering (30+ metrics)
- train.py - Model training logic
- main.py - Entry point for training pipeline
- predict.py - Prediction and KPI recommendations
- api.py - FastAPI endpoint
- requirements.txt - Python dependencies
- Dockerfile.train - Docker image for training
- Dockerfile.api - Docker image for API
- .env.example - Environment variables template
- .github/workflows/train.yml - GitHub Actions workflow
- README.md - This file

## Quick Start

### 1. Configure environment

Copy the environment template and fill in your S3 credentials:

```bash
cp .env.example .env
Edit .env with your values:

env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_ENDPOINT_URL=https://your-s3-endpoint.com
S3_BUCKET_NAME=your-bucket-name

2. Upload data to S3
Upload three CSV files to your S3 bucket:

* transactions.csv - Entity transactions with volume, revenue, timestamps

* sessions.csv - Entity activity sessions

* bonuses.csv - Bonus/incentive completion data

3. Train models via GitHub Actions
Go to repository Settings → Secrets and variables → Actions

Add the following secrets:

* AWS_ACCESS_KEY_ID

* AWS_SECRET_ACCESS_KEY

* S3_ENDPOINT_URL

* S3_BUCKET_NAME

Then:

* Go to Actions tab

* Select "Train Entity Similarity Model"

* Click "Run workflow"

Models will be saved to s3://your-bucket/models/

4. Run prediction API locally
bash
docker build -f Dockerfile.api -t entity-similarity-api .
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e S3_ENDPOINT_URL=https://your-s3-endpoint.com \
  -e S3_BUCKET_NAME=your-bucket-name \
  entity-similarity-api
API will be available at http://localhost:8000

5. Get predictions
bash
curl -X POST http://localhost:8000/get_entity_recommendations \
  -H "Content-Type: application/json" \
  -d '{"models_dir": "models"}'
Response:

json
{
  "status": "success",
  "prediction_path": "predictions/entities_with_predictions_abc123.csv",
  "kpi_path": "predictions/entity_kpi_recommendations_abc123.csv"
}
Results are also uploaded to your S3 bucket under predictions/

6. Health check
bash
curl http://localhost:8000/health
json
{"status": "ok"}
Key Features
Entity Classification - 6 performance categories (A-F) based on behavioral metrics

Vector-based Benchmarking - Finds most similar high-performing entity using cosine similarity

KPI Recommendations - Automatic suggestions with improvement targets

Multi-segment Models - Separate RandomForest models for different entity segments

Production Ready - Docker, CI/CD, S3 integration, FastAPI

Metrics Computed
The service computes 20+ behavioral metrics per entity:

Activity

* activity_volume

* processing_time

* transactions_per_hour

Revenue

* revenue_per_month

* revenue_per_hour

* revenue_growth

Quality

* success_rate

* abandonment_rate

Engagement

* avg_hours_per_day

* preferred_period

* Incentives

* total_bonuses

* bonus_success_rate

* avg_bonus_reward

Requirements
* Python 3.10+

* Docker

* S3-compatible storage

* GitHub repository with Actions enabled

Deployment

Docker Hub (recommended)
bash
docker pull yourname/entity-similarity-api:latest
docker run -p 8000:8000 --env-file .env yourname/entity-similarity-api:latest