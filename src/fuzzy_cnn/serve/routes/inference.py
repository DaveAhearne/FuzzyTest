import logging
from fastapi import APIRouter, FastAPI, Request, Depends, HTTPException, UploadFile,status

from fuzzy_cnn.serve.deps import get_model
from fuzzy_cnn.serve.inference import get_result
from fuzzy_cnn.serve.schemas import InferenceResult
from onnxruntime import InferenceSession

router = APIRouter(prefix="/inference", tags=["inference"])

logger = logging.getLogger(__name__)

@router.post("/img", response_model=InferenceResult, status_code=status.HTTP_200_OK)
async def score(
    request: Request,
    file: UploadFile,
    onnx_model: InferenceSession = Depends(get_model)
):
    logger.info("HIT: /inference/img endpoint")
    try:
        image_bytes = await file.read()
        inference_result = get_result(onnx_model, image_bytes)
        return InferenceResult.from_domain(inference_result)
    except Exception:
        logger.exception("Issue processing user request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal evaluation error",
        )
