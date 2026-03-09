from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
import onnxruntime as ort
from fuzzy_cnn.serve.middleware import RequestLoggingMiddleware, TimeoutMiddleware
from fuzzy_cnn.serve.routes.health import router as health_router
from fuzzy_cnn.serve.routes.inference import router as inference_router
from fuzzy_cnn.serve.logging import configure_logging
from fuzzy_cnn.common.config import ONNX_MODEL_PATH, settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    yield
    
app = FastAPI(
    title="Fuzzy Classifier",
    description="",
    version="1.0.0",
    docs_url="/",
    lifespan=lifespan
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TimeoutMiddleware)

app.include_router(health_router)
app.include_router(inference_router)

def main() -> None:
    import os
    import uvicorn

    host = os.getenv("HOST", settings.host)
    port = int(os.getenv("PORT", settings.port))
    workers = int(os.getenv("WORKERS", settings.workers))

    uvicorn.run(
        "fuzzy_cnn.serve.api:app",
        host=host,
        port=port,
        workers=workers,
    )