# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12-slim

FROM python:${PYTHON_VERSION} AS builder

WORKDIR /app

# Install system dependencies for OpenCV and OpenOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.117.1" \
    "uvicorn>=0.37.0" \
    "pydantic>=2.11.9" \
    "python-dotenv>=1.0.0" \
    "opencv-python-headless>=4.9.0" \
    "ultralytics>=8.3.192" \
    "openocr-python>=0.0.10" \
    "onnxruntime>=1.24.1" \
    "azure-storage-blob>=12.20.0" \
    "azure-core>=1.30.0" \
    "python-dateutil>=2.8.2" \
    "loguru>=0.7.0"

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "doc_verify.api:app", "--host", "0.0.0.0", "--port", "8000"]