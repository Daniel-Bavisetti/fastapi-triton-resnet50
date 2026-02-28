import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.metrics import active_requests, request_count

logger = logging.getLogger("upjao.request")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        active_requests.inc()

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(status_code),
            ).inc()
            active_requests.dec()
            logger.info(
                "%s %s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )
