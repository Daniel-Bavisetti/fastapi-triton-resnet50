import asyncio
import logging
import time
from typing import Any, Dict

import numpy as np
import tritonclient.http as httpclient
from fastapi import HTTPException
from tritonclient.utils import InferenceServerException

from app.core.config import get_settings
from app.core.metrics import inference_latency_seconds, triton_errors_total

logger = logging.getLogger("upjao.triton")


class TritonInferenceClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.MODEL_NAME
        self.model_version = settings.MODEL_VERSION
        self._client = httpclient.InferenceServerClient(
            url=f"{settings.TRITON_HOST}:{settings.TRITON_HTTP_PORT}",
            verbose=False,
        )

    async def is_server_live(self) -> bool:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._client.is_server_live)
        except Exception:
            return False

    async def infer(self, image_tensor: np.ndarray) -> Dict[str, Any]:
        if image_tensor.dtype != np.float32:
            image_tensor = image_tensor.astype(np.float32)
        if image_tensor.shape != (1, 3, 224, 224):
            raise HTTPException(status_code=400, detail="Invalid input tensor shape. Expected (1, 3, 224, 224).")

        loop = asyncio.get_running_loop()
        infer_input = httpclient.InferInput("data", image_tensor.shape, "FP32")
        infer_input.set_data_from_numpy(image_tensor)
        requested_output = httpclient.InferRequestedOutput("resnetv17_dense0_fwd")

        start = time.perf_counter()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._client.infer(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    inputs=[infer_input],
                    outputs=[requested_output],
                ),
            )
        except (InferenceServerException, OSError, ConnectionError) as exc:
            triton_errors_total.inc()
            logger.exception("Triton inference failed: %s", exc)
            raise HTTPException(status_code=503, detail="Inference service unavailable") from exc
        except Exception as exc:
            triton_errors_total.inc()
            logger.exception("Unexpected Triton error: %s", exc)
            raise HTTPException(status_code=503, detail="Inference service unavailable") from exc

        elapsed = time.perf_counter() - start
        inference_latency_seconds.observe(elapsed)
        logger.info("triton_infer_latency_ms=%.2f", elapsed * 1000)

        output = result.as_numpy("resnetv17_dense0_fwd")
        if output is None:
            triton_errors_total.inc()
            raise HTTPException(status_code=502, detail="Invalid inference response")

        return {
            "model": self.model_name,
            "inference_time_ms": elapsed * 1000,
            "raw_output": output,
        }


triton_client = TritonInferenceClient()
