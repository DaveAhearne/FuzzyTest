import numpy as np
import pytest
import torch
import onnxruntime as ort
from fuzzy_cnn.common.config import CHECKPOINT_DIR, ONNX_MODEL_PATH
from fuzzy_cnn.common.io import load_checkpoint
from fuzzy_cnn.train.model import CIFAR10ClassifierModel

def test_onnx_matches_pytorch():
    # Skip if no trained model exists
    checkpoint_path = CHECKPOINT_DIR / "final.pt"
    if not checkpoint_path.exists():
        pytest.skip("No checkpoint found — run training first")

    if not ONNX_MODEL_PATH.exists():
        pytest.skip("No ONNX model found — run export first")

    model = CIFAR10ClassifierModel()
    load_checkpoint(checkpoint_path, model)
    model.eval()

    test_input = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(str(ONNX_MODEL_PATH))
    onnx_output = session.run(None, {"image": test_input.numpy()})[0]

    assert np.allclose(torch_output, onnx_output, atol=1e-6), \
        f"Max difference: {np.max(np.abs(torch_output - onnx_output))}"