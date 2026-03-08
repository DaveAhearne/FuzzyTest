
import io
from PIL import Image
from onnxruntime import InferenceSession
from fuzzy_cnn.common.postprocessing import postprocess
from fuzzy_cnn.common.preprocessing import get_inference_transforms

def get_result(onnx_session: InferenceSession, image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")

    inf_transforms = get_inference_transforms()
    tensor_result = inf_transforms(pil_image)

    input_array = tensor_result.unsqueeze(0).numpy()

    # Without the squeeze, this comes back as a 1d array of [1,10], not a 1d array
    # It's just a fact of us having the batch size during training
    onnx_res = onnx_session.run(None, {"image": input_array})[0].squeeze()

    return postprocess(onnx_res)