import numpy as np
import pytest
import torch
import onnxruntime as ort
from fuzzy_cnn.common.config import CHECKPOINT_DIR, ONNX_MODEL_PATH
from fuzzy_cnn.common.io import load_checkpoint
from fuzzy_cnn.train.model import CIFAR10ClassifierModel

def test_onnx_matches_pytorch():
    checkpoint_path = CHECKPOINT_DIR / "final.pt"
    print(checkpoint_path)

    # Skip if no trained model exists
    if not checkpoint_path.exists():
        pytest.skip("No checkpoint found — run training first")

    if not ONNX_MODEL_PATH.exists():
        pytest.skip("No ONNX model found — run export first")

    model = CIFAR10ClassifierModel()
    load_checkpoint(checkpoint_path, model)
    model.eval()

    test_input = torch.randn(1, 3, 32, 32)

    # Because we're loading a raw model we need to remember to turn gradients off
    # otherwise it's a waste
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    # ONNX can have multiple outputs, we only have 1 so we take the first with [0]
    onnx_output = session.run(None, {"image": test_input.numpy()})[0]

    # Make sure the ONNX and the trained version are close enough, won't be exact but needs to be close
    assert np.allclose(torch_output, onnx_output, atol=1e-6), \
        f"Max difference: {np.max(np.abs(torch_output - onnx_output))}"