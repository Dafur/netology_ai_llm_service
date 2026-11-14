"""Точка входа FastAPI-сервиса."""

import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import router
from config import configure_logging
from llm.exceptions import LLMError

# Логирование настраивается до создания приложения, чтобы все
# последующие сообщения попали и в консоль, и в файл.
configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="Учебный LLM-сервис", version="0.1.0")
app.include_router(router)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    """Логирует начало и завершение каждого HTTP-запроса с длительностью."""
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    client_host = request.client.host if request.client else "-"
    logger.info(
        "HTTP-запрос начат: method=%s, path=%s, client=%s, started_at=%s",
        request.method,
        request.url.path,
        client_host,
        started_at.isoformat(),
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception(
            "HTTP-запрос завершён ошибкой: method=%s, path=%s, elapsed=%.3f c",
            request.method,
            request.url.path,
            elapsed,
        )
        raise

    elapsed = time.perf_counter() - started
    logger.info(
        "HTTP-запрос завершён: method=%s, path=%s, status=%s, elapsed=%.3f c",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Ошибки валидации входных данных возвращаются как 400, а не 422."""
    logger.warning("Некорректные входные данные: %s", exc.errors())
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Некорректные входные данные",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(LLMError)
async def llm_exception_handler(request: Request, exc: LLMError) -> JSONResponse:
    """Страховка для необработанных ошибок LLM-слоя."""
    logger.error("Необработанная ошибка LLM-слоя: %s", exc, exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Сервис LLM временно недоступен, повторите запрос позже"
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Любая непредвиденная ошибка превращается в понятный ответ 500."""
    logger.error("Непредвиденная ошибка сервиса: %s", exc, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервиса"},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск сервиса на http://127.0.0.1:8000")
    # log_config=None: uvicorn не переопределяет настройки logging,
    # установленные configure_logging().
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_config=None,
    )
