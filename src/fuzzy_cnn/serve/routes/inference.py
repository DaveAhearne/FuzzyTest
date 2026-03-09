import logging
from fastapi import APIRouter, Depends, Request, HTTPException, UploadFile,status
from fuzzy_cnn.serve.inference import get_result
from fuzzy_cnn.serve.schemas import InferenceResult
from fuzzy_cnn.serve.security import require_api_key

router = APIRouter(prefix="/inference", tags=["inference"])

logger = logging.getLogger(__name__)

@router.post("/img", response_model=InferenceResult, status_code=status.HTTP_200_OK)
async def score(
    request: Request,
    file: UploadFile,
    _: None = Depends(require_api_key)
):
    logger.info("HIT: /inference/img endpoint")
    try:
        onnx_session = request.app.state.onnx_session
        image_bytes = await file.read()
        inference_result = get_result(onnx_session, image_bytes)
        return InferenceResult.from_domain(inference_result)
    except Exception:
        logger.exception("Issue processing user request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal evaluation error",
        )
