import os
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from doc_verify.config import MODELS_DIR
from doc_verify.predict import DocumentVerifier


class VerificationRequest(BaseModel):
    document_type: int
    image_path: str
    name: str
    surname: str
    date_of_birth: str
    gender: str
    expiration_date: str


router = APIRouter(
    tags=["Document Verification"],
)


@router.post("/verify")
def verify_document(user_info: VerificationRequest):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        document_type = user_info.document_type
        cls_model_path = MODELS_DIR / f"best_model_type_{document_type}_YOLO.pt"

        user_info_dict = {
            "document_type": user_info.document_type,
            "image_path": user_info.image_path,
            "name": user_info.name,
            "surname": user_info.surname,
            "date_of_birth": user_info.date_of_birth,
            "gender": user_info.gender,
            "expiration_date": user_info.expiration_date,
        }

        verifier = DocumentVerifier(
            user_info=user_info_dict, cls_model_path=cls_model_path
        )

        is_valid = verifier.verify_image()

        return {
            "valid_status": is_valid,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok"}


app = FastAPI(title="Document Verification Service API")
app.include_router(router)