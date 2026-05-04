import pytest
from datetime import timedelta

from core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from core.exceptions import (
    CredentialsException,
    TokenExpiredException,
    InvalidTokenException,
)


def test_get_password_hash_returns_string():
    """Хеш должен быть строкой и отличаться от исходного пароля."""
    password = "super_secret_123"
    hashed = get_password_hash(password)
    assert isinstance(hashed, str)
    assert hashed != password
    assert hashed.startswith("$2")


def test_verify_password_correct():
    """Верификация правильного пароля должна возвращать True."""
    password = "my_password"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True


def test_verify_password_incorrect():
    """Верификация неправильного пароля должна возвращать False."""
    password = "correct_horse_battery_staple"
    hashed = get_password_hash(password)
    assert verify_password("wrong_password", hashed) is False


def test_hash_salting_produces_different_hashes():
    """Один и тот же пароль должен давать разные хеши из-за соли."""
    password = "same_password"
    hash1 = get_password_hash(password)
    hash2 = get_password_hash(password)
    assert hash1 != hash2
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True


def test_create_and_decode_access_token():
    """Access-токен создаётся и корректно декодируется."""
    token = create_access_token(data={"sub": "user123"})
    payload = decode_token(token, expected_type="access")
    assert payload.sub == "user123"
    assert payload.type == "access"


def test_create_and_decode_refresh_token():
    """Refresh-токен создаётся и корректно декодируется."""
    token = create_refresh_token(data={"sub": "user456"})
    payload = decode_token(token, expected_type="refresh")
    assert payload.sub == "user456"
    assert payload.type == "refresh"


def test_decode_token_wrong_type_raises():
    """Декодирование access-токена как refresh вызывает InvalidTokenException."""
    access_token = create_access_token(data={"sub": "user"})
    with pytest.raises(InvalidTokenException):
        decode_token(access_token, expected_type="refresh")


def test_expired_token_raises():
    """Просроченный токен вызывает TokenExpiredException."""
    token = create_access_token(data={"sub": "user"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(TokenExpiredException):
        decode_token(token)


def test_invalid_token_raises():
    """Совершенно невалидная строка токена вызывает InvalidTokenException."""
    with pytest.raises(InvalidTokenException):
        decode_token("totally.invalid.token")


def test_credentials_exception_properties():
    """CredentialsException должен иметь статус 401 и корректный detail."""
    exc = CredentialsException()
    assert exc.status_code == 401
    assert exc.detail == "Could not validate credentials"


def test_token_expired_exception_properties():
    """TokenExpiredException должен иметь статус 401 и корректный detail."""
    exc = TokenExpiredException()
    assert exc.status_code == 401
    assert exc.detail == "Token has expired"


def test_invalid_token_exception_properties():
    """InvalidTokenException должен иметь статус 401 и корректный detail."""
    exc = InvalidTokenException()
    assert exc.status_code == 401
    assert exc.detail == "Invalid token"
