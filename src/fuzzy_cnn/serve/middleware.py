import asyncio
import time
import uuid
import logging
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fuzzy_cnn.common.config import settings
from fuzzy_cnn.serve.context import request_id_ctx

logger = logging.getLogger("fuzzy_cnn.middleware")

MAX_BODY_LOG_LENGTH = 200

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request_id_ctx.set(request_id)

        start = time.perf_counter()

        body_snippet = None
        if request.method in {"POST", "PUT", "PATCH"}:
            raw = await request.body()
            decoded = raw.decode("utf-8", errors="replace")
            body_snippet = (
                decoded[:MAX_BODY_LOG_LENGTH] + "…"
                if len(decoded) > MAX_BODY_LOG_LENGTH
                else decoded
            )

            async def receive():
                return {
                    "type": "http.request",
                    "body": raw,
                    "more_body": False,
                }

            request = Request(request.scope, receive)

        params_snippet = str(dict(request.query_params)) if request.query_params else None

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000

            parts = [
                f"request_id={request_id}",
                f"{request.method} {request.url.path}",
                "status=unhandled_exception",
                f"duration_ms={duration_ms:.1f}",
            ]
            if params_snippet:
                parts.append(f"params={params_snippet}")
            if body_snippet:
                parts.append(f"body={body_snippet}")

            logger.exception(" | ".join(parts))
            raise

        response.headers["X-Request-ID"] = request_id
        duration_ms = (time.perf_counter() - start) * 1000

        parts = [
            f"request_id={request_id}",
            f"{request.method} {request.url.path}",
            f"status={response.status_code}",
            f"duration_ms={duration_ms:.1f}",
        ]
        if params_snippet:
            parts.append(f"params={params_snippet}")
        if body_snippet:
            parts.append(f"body={body_snippet}")

        log = logger.error if response.status_code >= 500 else logger.info
        log(" | ".join(parts))

        return response

class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            async with asyncio.timeout(settings.MAX_REQ_SEC):
                return await call_next(request)
        except TimeoutError:
            return JSONResponse(status_code=504, content={"detail": "Request timeout"})