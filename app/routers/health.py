from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.services.triton_client import triton_client

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    live = await triton_client.is_server_live()
    if not live:
        raise HTTPException(status_code=503, detail="Triton server is not live")

    return {"status": "ok", "triton_live": True}


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
