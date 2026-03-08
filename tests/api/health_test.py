import pytest
from fastapi.testclient import TestClient
from fuzzy_cnn.serve.api import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_returns_ok(client):
    response = client.get(
        "/health",
    )

    assert response.status_code == 200