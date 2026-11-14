"""Асинхронный клиент YandexGPT с retry и timeout.

Модуль инкапсулирует транспортное взаимодействие с Foundation Models API:
- конфигурация берётся из переменных окружения (и `.env`);
- каждый HTTP-запрос ограничен таймаутом (httpx.Timeout);
- повторяются только временные ошибки (таймауты, сеть, 429, 5xx)
  с экспоненциальной задержкой и джиттером;
- постоянные ошибки (4xx, неразбираемый ответ) пробрасываются сразу,
  чтобы API вернул пользователю осмысленный HTTP-статус.

Подготовка промпта и пост-обработка ответа выполняются в сервисном слое
(:mod:`services.llm_service`), поэтому клиент принимает готовое тело запроса.

Пример использования::

    from llm import YandexGPTClient, YandexGPTConfig

    config = YandexGPTConfig.from_env()
    payload = {
        "modelUri": config.model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "1000",
        },
        "messages": [{"role": "user", "text": "Привет!"}],
    }
    async with YandexGPTClient(config) as client:
        response = await client.complete(payload)
        print(response.text)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

import httpx
from dotenv import load_dotenv

from llm.exceptions import (
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

#: Эндпоинт синхронной (не потоковой) генерации YandexGPT.
COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

#: Коды ответа, при которых запрос имеет смысл повторить.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _env_str(name: str, default: str | None, env: Mapping[str, str]) -> str | None:
    """Читает строковую переменную окружения, пустое значение = ``default``."""
    value = env.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float, env: Mapping[str, str]) -> float:
    """Читает float-переменную окружения; при ошибке формата — ``default``."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"Переменная {name} должна быть числом, получено: {raw!r}"
        ) from exc


def _env_int(name: str, default: int, env: Mapping[str, str]) -> int:
    """Читает int-переменную окружения; при ошибке формата — ``default``."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"Переменная {name} должна быть целым числом, получено: {raw!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class YandexGPTConfig:
    """Настройки клиента YandexGPT.

    Значения по умолчанию рассчитаны на учебный сервис: лёгкая модель,
    умеренный таймаут и три повтора с экспоненциальной задержкой.
    """

    api_key: str
    folder_id: str
    model: str = "yandexgpt-lite"
    model_version: str = "latest"
    #: Параметры генерации по умолчанию; используются сервисным слоем
    #: при формировании тела запроса.
    temperature: float = 0.6
    max_tokens: int = 1000
    system_prompt: str | None = None
    #: Таймаут одного HTTP-запроса в секундах.
    timeout: float = 20.0
    #: Количество повторов сверх первой попытки.
    max_retries: int = 3
    #: Базовая задержка перед повтором (далее удваивается).
    backoff_base: float = 0.5
    #: Максимальная задержка между попытками в секундах.
    backoff_max: float = 8.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LLMConfigurationError("Не задан YANDEX_GPT_API_KEY")
        if not self.folder_id:
            raise LLMConfigurationError("Не задан YANDEX_GPT_FOLDER_ID")
        if self.timeout <= 0:
            raise LLMConfigurationError("timeout должен быть больше нуля")
        if self.max_retries < 0:
            raise LLMConfigurationError("max_retries не может быть отрицательным")
        if self.backoff_base < 0 or self.backoff_max < 0:
            raise LLMConfigurationError("Задержки backoff не могут быть отрицательными")
        if not 0.0 <= self.temperature <= 1.0:
            raise LLMConfigurationError("temperature должна быть в диапазоне [0, 1]")
        if self.max_tokens <= 0:
            raise LLMConfigurationError("max_tokens должен быть больше нуля")

    @property
    def model_uri(self) -> str:
        """Идентификатор модели в формате ``gpt://<folder>/<model>/<version>``."""
        return f"gpt://{self.folder_id}/{self.model}/{self.model_version}"

    @property
    def total_attempts(self) -> int:
        """Полное число попыток, включая первую."""
        return self.max_retries + 1

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> "YandexGPTConfig":
        """Собирает конфигурацию из переменных окружения.

        Поддерживаемые переменные:
        ``YANDEX_GPT_API_KEY``, ``YANDEX_GPT_FOLDER_ID``, ``YANDEX_GPT_MODEL``,
        ``YANDEX_GPT_MODEL_VERSION``, ``YANDEX_GPT_TEMPERATURE``,
        ``YANDEX_GPT_MAX_TOKENS``, ``YANDEX_GPT_TIMEOUT``,
        ``YANDEX_GPT_MAX_RETRIES``, ``YANDEX_GPT_SYSTEM_PROMPT``.
        """
        if env is None:
            if load_env_file:
                load_dotenv()
            env = os.environ

        return cls(
            api_key=_env_str("YANDEX_GPT_API_KEY", None, env) or "",
            folder_id=_env_str("YANDEX_GPT_FOLDER_ID", None, env) or "",
            model=_env_str("YANDEX_GPT_MODEL", "yandexgpt-lite", env)
            or "yandexgpt-lite",
            model_version=_env_str("YANDEX_GPT_MODEL_VERSION", "latest", env)
            or "latest",
            temperature=_env_float("YANDEX_GPT_TEMPERATURE", 0.6, env),
            max_tokens=_env_int("YANDEX_GPT_MAX_TOKENS", 1000, env),
            system_prompt=_env_str("YANDEX_GPT_SYSTEM_PROMPT", None, env),
            timeout=_env_float("YANDEX_GPT_TIMEOUT", 20.0, env),
            max_retries=_env_int("YANDEX_GPT_MAX_RETRIES", 3, env),
            backoff_base=_env_float("YANDEX_GPT_BACKOFF_BASE", 0.5, env),
            backoff_max=_env_float("YANDEX_GPT_BACKOFF_MAX", 8.0, env),
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Результат обращения к модели."""

    text: str
    model_uri: str
    attempts: int
    usage: Mapping[str, Any] | None = None


class YandexGPTClient:
    """Асинхронный клиент YandexGPT с ретраями и fallback-ответом.

    Клиент можно переиспользовать между запросами: внутри держится один
    ``httpx.AsyncClient`` с общим пулом соединений. Закрывать нужно методом
    :meth:`aclose` или через ``async with``.
    """

    def __init__(
        self,
        config: YandexGPTConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config or YandexGPTConfig.from_env()
        self._sleep = sleep or asyncio.sleep
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            transport=transport,
            headers={
                "Authorization": f"Api-Key {self._config.api_key}",
                "x-folder-id": self._config.folder_id,
                "Content-Type": "application/json",
            },
        )

    @property
    def config(self) -> YandexGPTConfig:
        """Текущая конфигурация клиента."""
        return self._config

    async def __aenter__(self) -> "YandexGPTClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Закрывает HTTP-соединения."""
        await self._client.aclose()

    async def complete(self, payload: Mapping[str, Any]) -> LLMResponse:
        """Отправляет готовое тело запроса в YandexGPT и разбирает ответ.

        Подготовка промпта (формирование ``payload``) выполняется в сервисном
        слое, клиент отвечает только за транспорт.

        Повторяет запрос только при временных ошибках: таймаутах, сетевых
        сбоях и ответах 429/5xx. Постоянные ошибки (4xx) и неразбираемый
        ответ пробрасываются сразу.

        Raises:
            LLMRequestError: API отклонил запрос по постоянной причине.
            LLMResponseError: ответ получен, но не разобран.
            LLMUnavailableError: временная ошибка не исчезла после всех попыток.
        """
        if not payload:
            raise ValueError("payload не может быть пустым")

        last_error: LLMUnavailableError | None = None
        total_attempts = self._config.total_attempts
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        logger.info(
            "YandexGPT: обращение начато, model=%s, started_at=%s, "
            "попыток=%d",
            self._config.model_uri,
            started_at.isoformat(),
            total_attempts,
        )
        logger.debug("YandexGPT: тело запроса: %s", payload)

        for attempt in range(1, total_attempts + 1):
            try:
                logger.debug(
                    "YandexGPT: попытка %s/%s", attempt, total_attempts
                )
                data = await self._request(payload)
                text = self._extract_text(data)
                result = data.get("result")
                usage = result.get("usage") if isinstance(result, dict) else None
                elapsed = time.perf_counter() - started
                logger.info(
                    "YandexGPT: ответ получен, attempts=%s, длина=%d, "
                    "elapsed=%.3f c",
                    attempt,
                    len(text),
                    elapsed,
                )
                logger.debug("YandexGPT: сырой текст ответа: %s", text)
                if attempt > 1:
                    logger.info("YandexGPT ответил успешно с попытки %s", attempt)
                return LLMResponse(
                    text=text,
                    model_uri=self._config.model_uri,
                    attempts=attempt,
                    usage=usage if isinstance(usage, dict) else None,
                )
            except LLMRequestError as exc:
                # Постоянная ошибка: повтор не поможет (неверный ключ, запрос и т.п.).
                logger.error("YandexGPT отклонил запрос без повтора: %s", exc)
                raise
            except LLMResponseError as exc:
                # Ответ получен, но не разобран: повтор того же запроса не поможет.
                logger.error("YandexGPT вернул неразбираемый ответ: %s", exc)
                raise
            except LLMUnavailableError as exc:
                # Временная ошибка (таймаут, сеть, 429/5xx): повторяем.
                last_error = exc
                if attempt >= total_attempts:
                    logger.error(
                        "YandexGPT: попытка %s/%s неудачна, повторы исчерпаны: %s",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "YandexGPT: попытка %s/%s неудачна (%s), повтор через %.2f c",
                    attempt,
                    total_attempts,
                    exc,
                    delay,
                )
                await self._sleep(delay)

        raise LLMUnavailableError(
            f"не удалось получить ответ YandexGPT после {total_attempts} попыток"
        ) from last_error

    async def ask(self, payload: Mapping[str, Any]) -> str:
        """Упрощённая обёртка: возвращает только текст ответа."""
        response = await self.complete(payload)
        return response.text

    async def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Выполняет один HTTP-запрос, классифицируя ошибки."""
        logger.debug("YandexGPT: HTTP POST %s", COMPLETION_URL)
        try:
            response = await self._client.post(COMPLETION_URL, json=payload)
        except httpx.TimeoutException as exc:
            logger.error(
                "YandexGPT: таймаут запроса (%.1f c)", self._config.timeout
            )
            raise LLMUnavailableError(
                f"таймаут запроса ({self._config.timeout:g} c)"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("YandexGPT: сетевая ошибка: %s", exc)
            raise LLMUnavailableError(f"сетевая ошибка: {exc}") from exc

        logger.debug(
            "YandexGPT: HTTP %s, длина тела=%d",
            response.status_code,
            len(response.content),
        )

        if response.status_code in RETRYABLE_STATUS_CODES:
            raise LLMUnavailableError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise LLMRequestError(
                f"HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError("ответ YandexGPT не является JSON") from exc
        if not isinstance(data, dict):
            raise LLMResponseError("неожиданный формат ответа YandexGPT")
        return data

    @staticmethod
    def _extract_text(data: Mapping[str, Any]) -> str:
        """Достаёт текст ответа из структуры ``result.alternatives[0].message``."""
        result = data.get("result")
        alternatives = result.get("alternatives") if isinstance(result, dict) else None
        if not alternatives:
            raise LLMResponseError("в ответе нет alternatives")
        first = alternatives[0]
        message = first.get("message") if isinstance(first, dict) else None
        text = message.get("text") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("в ответе пустой текст")
        return text.strip()

    def _backoff_delay(self, attempt: int) -> float:
        """Экспоненциальная задержка с джиттером: base * 2^(n-1), не выше max."""
        delay = min(
            self._config.backoff_base * (2 ** (attempt - 1)),
            self._config.backoff_max,
        )
        return delay + random.uniform(0, delay * 0.25)


_default_client: YandexGPTClient | None = None


def get_default_client() -> YandexGPTClient:
    """Возвращает общий клиент, создавая его при первом обращении."""
    global _default_client
    if _default_client is None:
        _default_client = YandexGPTClient()
    return _default_client


async def close_default_client() -> None:
    """Закрывает общий клиент (например, при остановке приложения)."""
    global _default_client
    if _default_client is not None:
        await _default_client.aclose()
        _default_client = None


async def ask_yandex_gpt(payload: Mapping[str, Any]) -> str:
    """Готовая функция обращения к YandexGPT: retry + timeout.

    Принимает готовое тело запроса и возвращает «сырой» текст ответа.
    Нормализация текста выполняется в сервисном слое. При исчерпании
    повторов пробрасывается :class:`LLMUnavailableError`.
    """
    return await get_default_client().ask(payload)


async def generate_yandex_gpt(payload: Mapping[str, Any]) -> LLMResponse:
    """Как :func:`ask_yandex_gpt`, но возвращает полный :class:`LLMResponse`."""
    return await get_default_client().complete(payload)
