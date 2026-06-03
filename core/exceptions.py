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


class PermissionDeniedException(HTTPException):
    """403 — доступ запрещён."""
    def __init__(self):
        super().__init__(status_code=403, detail="Permission denied")


class RateLimitExceededException(HTTPException):
    """429 — превышена дневная квота."""
    def __init__(self):
        super().__init__(status_code=429, detail="Daily quota exceeded")
