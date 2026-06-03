import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_proxy_headers_middleware_is_configured():
    """ProxyHeadersMiddleware должна быть зарегистрирована в приложении."""
    middleware_classes = [m.cls for m in app.user_middleware]
    assert ProxyHeadersMiddleware in middleware_classes


def test_proxy_headers_middleware_allows_forwarded_header(client):
    """При TRUSTED_PROXIES=['*'] запрос с X-Forwarded-For не должен падать."""
    response = client.get("/", headers={"X-Forwarded-For": "203.0.113.42"})
    assert response.status_code == 200
