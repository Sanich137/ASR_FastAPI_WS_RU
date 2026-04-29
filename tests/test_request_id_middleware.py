import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_request_id_middleware_generates_id(client):
    """Если клиент не передал X-Request-ID, middleware должен сгенерировать UUID."""
    response = client.get("/")
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_request_id_middleware_preserves_provided_id(client):
    """Если клиент передал X-Request-ID, он должен вернуться в ответе без изменений."""
    custom_id = "my-custom-id-12345"
    response = client.get("/", headers={"X-Request-ID": custom_id})
    assert response.headers["x-request-id"] == custom_id
