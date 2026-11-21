"""One error model for the whole API: {"error": CODE, "message": text} with a matching HTTP status."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    status, code = 500, "INTERNAL"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(AppError):
    status, code = 404, "NOT_FOUND"


class BadRequest(AppError):
    status, code = 400, "INVALID_REQUEST"


class Conflict(AppError):
    status, code = 409, "CONFLICT"


class UpstreamError(AppError):
    """A neighbouring service (scoring API, MLflow, Docker) failed or is not running."""
    status, code = 502, "UPSTREAM_UNAVAILABLE"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_req: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError):
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"] if p not in ("body", "query", "path"))
        return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST", "message": f"{field}: {err['msg']}"})
