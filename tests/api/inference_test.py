def test_predict_endpoint_returns_predictions(client, test_image_bytes):
    response = client.post(
        "/inference/img",
        files={"file": ("test.png", test_image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0

    first = data["predictions"][0]
    assert "label" in first
    assert "confidence" in first
    assert 0.0 <= first["confidence"] <= 1.0

def test_predict_endpoint_invalid_file(client):
    response = client.post(
        "/inference/img",
        files={"file": ("test.txt", b"I am the coolest book in town but im not a photo of a cat", "text/plain")},
    )

    assert response.status_code == 500