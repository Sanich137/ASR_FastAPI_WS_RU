import pytest
from fastapi.testclient import TestClient
from main import app
from config import settings


@pytest.fixture
def restricted_client():
    """Клиент с ограниченным ALLOWED_HOSTS для проверки TrustedHostMiddleware."""
    original_hosts = settings.ALLOWED_HOSTS.copy()
    settings.ALLOWED_HOSTS[:] = ["trusted.example.com"]
    # Сбрасываем кэш middleware stack, чтобы перестроить с актуальными настройками
    app._middleware_stack = None
    with TestClient(app) as client:
        yield client
    settings.ALLOWED_HOSTS[:] = original_hosts
    app._middleware_stack = None


def test_trusted_host_middleware_rejects_invalid_host(restricted_client):
    """Запросы с недоверенным Host должны отклоняться с 400."""
    response = restricted_client.get("/", headers={"Host": "untrusted.example.com"})
    assert response.status_code == 400


def test_trusted_host_middleware_allows_valid_host(restricted_client):
    """Запросы с доверенным Host должны проходить (не 400)."""
    response = restricted_client.get("/", headers={"Host": "trusted.example.com"})
    assert response.status_code != 400
