# CPU-only image: no torch extra installed, so segmentation/embedding
# backends run the same heuristic fallback path they already use by default
# outside Docker (SAM2/DINOv2 adapters hard-fallback regardless of torch
# availability — see cropmerge/segmentation/sam3.py, features/dinov3.py).
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/vision/pyproject.toml ./
COPY apps/vision/api ./api
COPY apps/vision/cropmerge ./cropmerge
COPY apps/vision/scripts ./scripts
COPY apps/vision/configs ./configs
COPY apps/vision/samples ./samples
RUN pip install --no-cache-dir -e .

ENV CROP_MERGE_DEVICE=cpu \
    CROP_MERGE_SEGMENTATION_BACKEND=heuristic \
    CROP_MERGE_DINO_BACKEND=heuristic

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
