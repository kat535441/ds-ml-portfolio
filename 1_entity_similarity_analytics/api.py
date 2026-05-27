from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger
import os

from predict import EntitySimilarityPredictor
from config import MODELS_DIR, S3_BUCKET_NAME
from dataset import get_s3_client

router = APIRouter(
    tags=["Entity Similarity Analytics"],
)


class EntitySimilarityRequest(BaseModel):
    models_dir: str = str(MODELS_DIR)


class EntitySimilarityResponse(BaseModel):
    status: str
    prediction_path: str
    kpi_path: Optional[str] = None


@router.post("/get_entity_recommendations", response_model=EntitySimilarityResponse)
def get_entity_recommendations(request: EntitySimilarityRequest):
    try:
        logger.info("[api] Entity Similarity prediction request received")
        logger.info(f"[api] Using models_dir={request.models_dir}")

        predictor = EntitySimilarityPredictor(models_dir=request.models_dir)
        prediction_path, kpi_path = predictor.run_prediction()

        if not prediction_path:
            logger.error("[api] Prediction output path is empty")
            raise RuntimeError("Prediction output is empty")

        logger.info(f"[api] Prediction output saved: {prediction_path}")
        if kpi_path:
            logger.info(f"[api] KPI recommendations saved: {kpi_path}")

        # Upload results to S3
        s3 = get_s3_client()
        s3_prediction_key = f"predictions/{os.path.basename(prediction_path)}"
        s3.upload_file(prediction_path, S3_BUCKET_NAME, s3_prediction_key)
        logger.info(f"[api] Predictions uploaded to S3: {s3_prediction_key}")

        s3_kpi_key = None
        if kpi_path:
            s3_kpi_key = f"predictions/{os.path.basename(kpi_path)}"
            s3.upload_file(kpi_path, S3_BUCKET_NAME, s3_kpi_key)
            logger.info(f"[api] KPI uploaded to S3: {s3_kpi_key}")

        return EntitySimilarityResponse(
            status="success",
            prediction_path=s3_prediction_key,
            kpi_path=s3_kpi_key,
        )
    except Exception as e:
        logger.error(f"[api] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok"}


app = FastAPI(title="Entity Similarity Analytics API")
app.include_router(router)