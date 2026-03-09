import io
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from fuzzy_cnn.serve.api import app

@pytest.fixture
def client():
    mock_session = MagicMock()
    mock_session.run.return_value = [np.zeros((1, 10), dtype=np.float32)]

    with patch("fuzzy_cnn.serve.api.ort.InferenceSession", return_value=mock_session):
        with TestClient(app) as client:
            yield client

@pytest.fixture
def test_image_bytes():
    # Create a dummy image and convert to bytes
    image = Image.new("RGB", (64, 64), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
