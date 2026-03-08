import onnxruntime as ort
from fuzzy_cnn.common.config import ONNX_MODEL_PATH

session = None

# Not a big fan of doing this globally, but it's a fine hack for now
# TODO: move this to the request instead
def load_model():
    global session
    session = ort.InferenceSession(ONNX_MODEL_PATH)

def get_model():
    return session