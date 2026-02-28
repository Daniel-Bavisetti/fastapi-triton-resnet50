from typing import List

from pydantic import BaseModel


class TopPrediction(BaseModel):
    rank: int
    class_id: int
    label: str
    confidence: float


class PredictResponse(BaseModel):
    model: str
    inference_time_ms: float
    top5_predictions: List[TopPrediction]
    status: str
