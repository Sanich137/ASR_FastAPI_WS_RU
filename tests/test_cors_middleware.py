import pytest
from fastapi.testclient import TestClient
from main import app
from config import settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_cors_origins_default_config():
    """По умолчанию CORS_ORIGINS должен быть ['*']."""
    assert hasattr(settings, "CORS_ORIGINS")
    assert settings.CORS_ORIGINS == ["*"]


def test_cors_middleware_allows_any_origin(client):
    """При CORS_ORIGINS=['*'] любой Origin должен получить разрешение."""
    response = client.get("/", headers={"Origin": "http://example.com"})
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_preflight_request(client):
    """Preflight OPTIONS запрос должен возвращать 200 с CORS-заголовками."""
    response = client.options(
        "/",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
