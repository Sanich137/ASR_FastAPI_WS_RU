import asyncio
import json

import httpx
import pytest

from config import settings
from models.fast_api_models import V1BaseResponse as BaseResponse

BASE_URL = f"http://127.0.0.1:{settings.PORT}/api/v1"


async def _assert_base_response(body: dict, expect_success: bool):
    assert "success" in body, f"Ответ должен содержать поле 'success' (V1BaseResponse)"
    assert body["success"] is expect_success


@pytest.mark.asyncio
async def test_post_by_url_success():
    async with httpx.AsyncClient() as client:
        payload = {
            "AudioFileUrl": "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav",
            "keep_raw": True,
            "do_echo_clearing": False,
            "do_dialogue": False,
            "do_punctuation": False,
        }
        try:
            resp = await client.post(
                f"{BASE_URL}/asr/url", json=payload, timeout=120.0
            )
        except httpx.ConnectError:
            pytest.fail(f"Сервер не отвечает по адресу {BASE_URL}. Убедитесь, что приложение запущено.")
        print("[POST /v1/post_one_step_req] status:", resp.status_code)
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))

        await _assert_base_response(body, expect_success=True)
        return body


@pytest.mark.asyncio
async def test_post_by_url_validation_error():
    """Ожидаем ErrorResponse при 422."""
    async with httpx.AsyncClient() as client:
        payload = {"keep_raw": True}  # отсутствует AudioFileUrl
        try:
            resp = await client.post(
                f"{BASE_URL}/asr/url", json=payload, timeout=10.0
            )
        except httpx.ConnectError:
            pytest.fail(f"Сервер не отвечает по адресу {BASE_URL}. Убедитесь, что приложение запущено.")
        print("[POST /v1/post_one_step_req 422] status:", resp.status_code)
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))

        assert resp.status_code == 422
        await _assert_base_response(body, expect_success=False)
        assert body.get("error_description") is not None
        return body


@pytest.mark.asyncio
async def test_post_by_file_success():
    async with httpx.AsyncClient() as client:
        with open("./examples/orig.wav", "rb") as f:
            files = {"file": ("orig.wav", f, "audio/wav")}
            data = {
                "keep_raw": "true",
                "do_echo_clearing": "false",
                "do_dialogue": "false",
                "do_punctuation": "false",
                "do_diarization": "false",
                "diar_vad_sensity": "3",
            }
            try:
                resp = await client.post(
                    f"{BASE_URL}/asr/file", data=data, files=files, timeout=20.0
                )
            except httpx.ConnectError:
                pytest.fail(f"Сервер не отвечает по адресу {BASE_URL}. Убедитесь, что приложение запущено.")
            print("[POST /v1/post_file] status:", resp.status_code)
            body = resp.json()
            print(json.dumps(body, indent=2, ensure_ascii=False))

            await _assert_base_response(body, expect_success=True)
            return body


@pytest.mark.asyncio
async def test_post_by_file_validation_error():
    """Ожидаем ErrorResponse при 422 (нет файла)."""
    async with httpx.AsyncClient() as client:
        data = {"keep_raw": "true"}
        try:
            resp = await client.post(
                f"{BASE_URL}/asr/file", data=data, timeout=10.0
            )
        except httpx.ConnectError:
            pytest.fail(f"Сервер не отвечает по адресу {BASE_URL}. Убедитесь, что приложение запущено.")
        print("[POST /v1/post_file 422] status:", resp.status_code)
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))

        assert resp.status_code == 422
        await _assert_base_response(body, expect_success=False)
        return body


async def main():
    print("=== Запуск HTTP-тестов роутов ===\n")
    await test_post_by_url_success()
    print()
    await test_post_by_url_validation_error()
    print()
    await test_post_by_file_success()
    print()
    await test_post_by_file_validation_error()
    print("\n=== Все тесты завершены ===")


if __name__ == "__main__":
    asyncio.run(main())
