README для третьего проекта
markdown
# Document Verification Service

ML-powered document verification service using YOLO for document classification and OCR for text extraction.

## Architecture
Input Image → YOLO Classification → OCR Text Extraction → Content Verification → Verification Result

text

## Project Structure
doc_verify/
├── config.py # Configuration and environment variables
├── dataset.py # Document type mappings
├── predict.py # DocumentVerifier class with YOLO + OCR pipeline
├── api.py # FastAPI endpoint
└── pyproject.toml # Dependencies

text

## Key Features

| Feature | Description |
|---------|-------------|
| **Document Classification** | YOLO-based classifier for 6 document types |
| **OCR Extraction** | OpenOCR with ONNX runtime |
| **Content Verification** | Name, surname, DOB, gender, expiration date validation |
| **Fuzzy Matching** | SequenceMatcher for text normalization and comparison |
| **MRZ Support** | Machine Readable Zone parsing for passports |
| **Rotation Handling** | Automatic image rotation for misoriented documents |

## Document Types

| ID | Type |
|----|------|
| 1 | Type 1 (e.g., License Front) |
| 2 | Type 2 (e.g., Permit) |
| 3 | Type 3 (e.g., Clearance) |
| 4 | Type 4 (e.g., License Back) |
| 5 | Type 5 (e.g., Passport) |
| 6 | Type 6 (e.g., Registration Certificate) |

## Quick Start

### 1. Configure environment

Create `.env` file:

```env
AZURE_STORAGE_ACCOUNT=your_account
AZURE_STORAGE_KEY=your_key
DOC_VER_CONTAINER_NAME=your_container
2. Run API
bash
docker build -f Dockerfile.api -t doc-verify-api .
docker run -p 8000:8000 --env-file .env doc-verify-api
3. Verify a document
bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": 1,
    "image_path": "/path/to/image.jpg",
    "name": "JOHN",
    "surname": "DOE",
    "date_of_birth": "01/01/1990",
    "gender": "M",
    "expiration_date": "31/12/2030"
  }'
Response:

json
{
  "valid_status": true
}
4. Health check
bash
curl http://localhost:8000/health

Requirements

* Python 3.12+

* Docker

* Azure Blob Storage (for model weights)