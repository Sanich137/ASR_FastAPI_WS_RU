from fastapi import HTTPException


class CredentialsException(HTTPException):
    """401 — не удалось проверить учётные данные."""
    def __init__(self):
        super().__init__(status_code=401, detail="Could not validate credentials")


class TokenExpiredException(HTTPException):
    """401 — токен просрочен."""
    def __init__(self):
        super().__init__(status_code=401, detail="Token has expired")


class InvalidTokenException(HTTPException):
    """401 — невалидный токен."""
    def __init__(self):
        super().__init__(status_code=401, detail="Invalid token")
