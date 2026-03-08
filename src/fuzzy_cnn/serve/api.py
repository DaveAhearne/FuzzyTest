from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fuzzy_cnn.serve.deps import load_model
from fuzzy_cnn.serve.routes.health import router as health_router
from fuzzy_cnn.serve.routes.inference import router as inference_router
from fuzzy_cnn.serve.logging import setup_logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    load_model()
    yield
    
app = FastAPI(
    title="Fuzzy Classifier",
    description="",
    version="1.0.0",
    docs_url="/",
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(inference_router)

def main() -> None:
    import os
    import uvicorn

    # TODO: move this to conf, we should really centralize this
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "1234"))
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "fuzzy_cnn.serve.api:app",
        host=host,
        port=port,
        workers=workers,
    )