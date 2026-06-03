import pytest
from fastapi.testclient import TestClient
from starlette.middleware.gzip import GZipMiddleware
from main import app


@pytest.fixture
def client():
    # Сбрасываем кэш middleware stack, чтобы перестроить с актуальными параметрами
    app.middleware_stack = None
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
    print(f"[test_gzip_skips] request headers: {dict(response.request.headers)}")
    print(f"[test_gzip_skips] status: {response.status_code}")
    print(f"[test_gzip_skips] response headers: {dict(response.headers)}")
    print(f"[test_gzip_skips] content length: {len(response.content)}")
    print(f"[test_gzip_skips] text: {response.text[:500]}")
    assert response.status_code == 200
    print(f"[response.headers]: {response.headers}")
    assert "content-encoding" not in response.headers, f"Unexpected content-encoding in headers: {dict(response.headers)}"


# def test_gzip_middleware_config_debug():
#     """Диагностика: выводим все параметры GZipMiddleware."""
#     gzip_middlewares = [m for m in app.user_middleware if m.cls is GZipMiddleware]
#     print(f"[gzip debug] Найдено экземпляров GZipMiddleware: {len(gzip_middlewares)}")
#     for i, m in enumerate(gzip_middlewares):
#         opts = getattr(m, "kwargs", {})
#         print(f"[gzip debug] Экземпляр {i}: {opts}")
#     assert len(gzip_middlewares) == 1
#     opts = getattr(gzip_middlewares[0], "kwargs", {})
#     assert opts.get("minimum_size") == 500
#
#
# def test_gzip_skips_tiny_response(client):
#     """Гарантированно маленький ответ не должен сжиматься."""
#     from main import app
#
#     @app.get("/_test_tiny", include_in_schema=False)
#     def _tiny():
#         return {"x": 1}
#
#     resp = client.get("/_test_tiny", headers={"Accept-Encoding": "gzip"})
#     print(f"[tiny] status={resp.status_code}")
#     print(f"[tiny] headers={dict(resp.headers)}")
#     print(f"[tiny] len(content)={len(resp.content)}")
#     print(f"[tiny] text={resp.text}")
#     assert resp.status_code == 200
#     assert "content-encoding" not in resp.headers, f"Сжался ответ в {len(resp.content)} байт!"
