from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from ml_common.clients import AzureBlobClient
from car_offer_rec.modeling.predict import CarOfferPredictor
from car_offer_rec.config import (
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_KEY,
    AZURE_CONTAINER_NAME,
    MODELS_DIR,
    PREDICTION_BLOB_PATH,
)

router = APIRouter(
    tags=["Car Offer Recommendation"],
)


class CarOfferRequest(BaseModel):
    models_dir: str = str(MODELS_DIR)


class CarOfferResponse(BaseModel):
    status: str
    prediction_path: str
    local_path: Optional[str] = None


@router.post("/get_car_offer_recommendations", response_model=CarOfferResponse)
async def get_car_offer_recommendations(request: CarOfferRequest):
    try:
        logger.info("[api] Car Offer Recommendation prediction request received")
        logger.info(f"[api] Using models_dir={request.models_dir}")

        predictor = CarOfferPredictor(models_dir=request.models_dir)
        prediction_path = predictor.run_prediction()

        if not prediction_path:
            logger.error("[api] Prediction output path is empty")
            raise RuntimeError("Prediction output is empty")

        logger.info(f"[api] Prediction output saved: {prediction_path}")

        azure_client = AzureBlobClient(
            account_name=AZURE_STORAGE_ACCOUNT,
            account_key=AZURE_STORAGE_KEY,
            container_name=AZURE_CONTAINER_NAME,
        )

        azure_client.upload_file(
            local_path=prediction_path,
            blob_path=PREDICTION_BLOB_PATH,
        )

        logger.info(f"[api] Prediction uploaded to: {PREDICTION_BLOB_PATH}")

        return CarOfferResponse(
            status="success",
            prediction_path=PREDICTION_BLOB_PATH,
            local_path=prediction_path,
        )
    except Exception as e:
        logger.error(f"[api] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok"}


app = FastAPI(title="Car Offer Recommendation API")
app.include_router(router)