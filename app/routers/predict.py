import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.response import PredictResponse, TopPrediction
from app.services.image_processor import ImageProcessingError, preprocess_image
from app.services.triton_client import triton_client
from scripts.imagenet_classes import IMAGENET_CLASSES

router = APIRouter(tags=["inference"])

_ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _postprocess_resnet(raw_output: np.ndarray) -> list[TopPrediction]:
    if raw_output.shape != (1, 1000):
        raise HTTPException(status_code=502, detail="Invalid inference response")

    logits = raw_output[0].astype(np.float64)
    probs = _softmax(logits)
    top5_ids = np.argsort(probs)[::-1][:5]

    top5_predictions: list[TopPrediction] = []
    for rank, class_id in enumerate(top5_ids, start=1):
        class_index = int(class_id)
        label = IMAGENET_CLASSES[class_index] if class_index < len(IMAGENET_CLASSES) else f"class_{class_index}"
        top5_predictions.append(
            TopPrediction(
                rank=rank,
                class_id=class_index,
                label=label,
                confidence=float(probs[class_index]),
            )
        )

    return top5_predictions


@router.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    if image.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Use JPEG or PNG.",
        )

    image_bytes = await image.read()
    try:
        image_tensor = preprocess_image(image_bytes)
    except ImageProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    triton_result = await triton_client.infer(image_tensor)
    top5_predictions = _postprocess_resnet(triton_result["raw_output"])

    return PredictResponse(
        model=triton_result["model"],
        inference_time_ms=round(float(triton_result["inference_time_ms"]), 3),
        top5_predictions=top5_predictions,
        status="success",
    )
