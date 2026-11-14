"""HTTP-эндпоинты сервиса."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from api.schemas import ChatRequest, ChatResponse
from llm.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
from services.llm_service import get_answer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Простая проверка работоспособности сервиса."""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest) -> ChatResponse:
    """Принимает сообщение пользователя и возвращает ответ модели.

    ``message`` передаётся как промпт в :func:`services.llm_service.get_answer`,
    который проверяет кэш, формирует запрос к YandexGPT и пост-обрабатывает
    ответ. Текст ответа модели возвращается в поле ``reply``.

    Валидация выполняется автоматически моделью ChatRequest:
    - message обязательно к передаче;
    - message не может быть пустым (после обрезки пробелов);
    - длина message не превышает 1000 символов.

    Raises:
        HTTPException: 400 — если промпт отвергнут сервисом (например, пустой);
            502 — если LLM вернула некорректный/неразбираемый ответ;
            503 — если LLM недоступна или не настроена;
            500 — при непредвиденной внутренней ошибке.
    """
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    logger.info(
        "Запрос /chat: started_at=%s, message=%r, длина=%d",
        started_at.isoformat(),
        payload.message,
        len(payload.message),
    )

    try:
        reply = await get_answer(payload.message)
    except ValueError as exc:
        # Ошибка входных данных: пользователь может исправить запрос.
        logger.warning(
            "Некорректный запрос к /chat: %s (elapsed=%.3f c)",
            exc,
            time.perf_counter() - started,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (LLMUnavailableError, LLMConfigurationError) as exc:
        # Модель временно недоступна или сервис не настроен.
        logger.exception(
            "Сервис LLM недоступен для /chat (elapsed=%.3f c)",
            time.perf_counter() - started,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис LLM временно недоступен, повторите запрос позже",
        ) from exc
    except (LLMRequestError, LLMResponseError) as exc:
        # LLM отклонила запрос или вернула данные, которые нельзя разобрать.
        logger.exception(
            "Сервис LLM вернул некорректный ответ для /chat (elapsed=%.3f c)",
            time.perf_counter() - started,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Сервис LLM вернул некорректный ответ",
        ) from exc
    except LLMError as exc:
        # Страховка на случай новых типов ошибок LLM-слоя.
        logger.exception(
            "Не удалось получить ответ LLM для /chat (elapsed=%.3f c)",
            time.perf_counter() - started,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис LLM временно недоступен, повторите запрос позже",
        ) from exc
    except Exception as exc:
        # Непредвиденная ошибка не должна раскрывать внутренние детали.
        logger.exception(
            "Непредвиденная ошибка при обработке /chat (elapsed=%.3f c)",
            time.perf_counter() - started,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервиса",
        ) from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "Ответ /chat: status=200, длина_ответа=%d, elapsed=%.3f c",
        len(reply),
        elapsed,
    )
    logger.debug("Текст ответа /chat: %s", reply)
    return ChatResponse(reply=reply)
