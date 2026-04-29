import pytest
from fastapi.testclient import TestClient
from starlette.middleware.gzip import GZipMiddleware
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_gzip_middleware_is_configured():
    """GZipMiddleware должна быть зарегистрирована в приложении."""
    middleware_classes = [m.cls for m in app.user_middleware]
    assert GZipMiddleware in middleware_classes


def test_gzip_middleware_compresses_large_response(client):
    """Большие ответы (>1000 байт) должны сжиматься при Accept-Encoding: gzip."""
    response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert len(response.content) > 0


def test_gzip_middleware_skips_small_response(client):
    """Маленькие ответы (<1000 байт) не должны сжиматься."""
    response = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert "content-encoding" not in response.headers
