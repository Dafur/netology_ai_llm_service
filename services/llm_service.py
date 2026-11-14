"""Сервисный слой ответов на промпты с кэшированием.

Модуль связывает независимые компоненты:

- :mod:`cache.llm_cache` — in-memory кэш пар «промпт — ответ»;
- :mod:`llm.prompt_builder` — подготовка промпта и тела запроса;
- :mod:`llm.post_processing` — нормализация ответа модели;
- :mod:`llm.yandex_gpt` — транспортный клиент с ретраями временных ошибок.

Логика обработки промпта:

1. Нормализованный промпт ищется в кэше.
2. Если ответ найден, он возвращается без обращения к модели
   (``from_cache=True``). Подготовка промпта при этом **не** выполняется.
3. Если ответа нет, формируется тело запроса (:func:`llm.prompt_builder.build_payload`).
4. Выполняется запрос к YandexGPT.
5. Полученный ответ проходит пост-обработку и сохраняется в кэш.
6. Ошибки модели пробрасываются вызывающему слою, который преобразует
   их в осмысленный HTTP-статус (400/502/503).

Пример использования::

    from services import get_answer

    text = await get_answer("Что такое FastAPI?")

Пример с метаданными::

    from services import answer_prompt

    answer = await answer_prompt("Что такое FastAPI?")
    print(answer.text, answer.from_cache)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

from cache import add_to_cache, get_cached_response
from llm.exceptions import LLMError, LLMUnavailableError
from llm.post_processing import postprocess_llm_response
from llm.prompt_builder import DEFAULT_SYSTEM_PROMPT, build_payload
from llm.yandex_gpt import LLMResponse, YandexGPTClient, YandexGPTConfig
from llm.yandex_gpt import get_default_client

logger = logging.getLogger(__name__)


def _preview(text: str, limit: int = 200) -> str:
    """Короткое однострочное представление текста для логов."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}… (всего {len(text)} символов)"


#: Функция генерации ответа: принимает готовое тело запроса.
Generator = Callable[[Mapping[str, Any]], Awaitable[LLMResponse]]

#: Разделитель частей составного ключа кэша (не является пробельным символом).
_KEY_SEPARATOR = "\x00"


@dataclass(frozen=True, slots=True)
class GenerationDefaults:
    """Параметры генерации по умолчанию для подготовки промпта.

    Используются, когда конкретный запрос не переопределяет параметры.
    """

    model_uri: str = ""
    system_prompt: str | None = None
    temperature: float = 0.6
    max_tokens: int = 1000


@dataclass(frozen=True, slots=True)
class LLMAnswer:
    """Ответ сервиса вместе с метаданными обработки."""

    text: str
    #: ``True``, если ответ взят из кэша без обращения к модели.
    from_cache: bool
    #: Сколько попыток сделал клиент YandexGPT (0 для ответа из кэша).
    attempts: int = 0


def _build_cache_key(
    prompt: str,
    *,
    system_prompt: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    """Строит ключ кэша.

    Если параметры генерации не заданы явно, ключом служит сам промпт —
    так кэш остаётся совместим с прямыми вызовами :mod:`cache.llm_cache`.
    Иначе параметры добавляются в ключ, чтобы разные настройки генерации
    не переиспользовали один и тот же ответ.
    """
    if system_prompt is None and temperature is None and max_tokens is None:
        return prompt
    return _KEY_SEPARATOR.join(
        (
            prompt,
            f"system={system_prompt!r}",
            f"temperature={temperature!r}",
            f"max_tokens={max_tokens!r}",
        )
    )


def _defaults_from_config(config: YandexGPTConfig) -> GenerationDefaults:
    """Преобразует конфигурацию клиента в параметры подготовки промпта."""
    return GenerationDefaults(
        model_uri=config.model_uri,
        system_prompt=config.system_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


class CachedLLMService:
    """Сервис «кэш + подготовка промпта + YandexGPT + пост-обработка».

    Args:
        generator: Функция генерации ответа, принимающая готовое тело
            запроса. По умолчанию используется ``YandexGPTClient.complete``
            общего клиента. Подмена удобна в тестах и при переходе
            на другого провайдера.
        defaults: Параметры генерации по умолчанию. Если не заданы, берутся
            из конфигурации клиента.
        client: Явный клиент YandexGPT. Используется, только если
            ``generator`` не передан.
    """

    def __init__(
        self,
        generator: Generator | None = None,
        *,
        defaults: GenerationDefaults | None = None,
        client: YandexGPTClient | None = None,
    ) -> None:
        if generator is not None:
            self._generator = generator
            self._defaults = defaults or GenerationDefaults()
            return

        client = client or get_default_client()
        self._generator = client.complete
        self._defaults = defaults or _defaults_from_config(client.config)

    async def answer(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMAnswer:
        """Возвращает ответ на промпт, используя кэш и YandexGPT.

        Args:
            prompt: Текст запроса пользователя.
            system_prompt: Переопределение системного промпта для этого запроса.
            temperature: Переопределение температуры генерации.
            max_tokens: Переопределение лимита токенов ответа.

        Returns:
            :class:`LLMAnswer` с текстом ответа и признаком попадания в кэш.

        Raises:
            ValueError: Если промпт пустой или не является строкой.
            LLMError: Если модель недоступна или ответ не удалось разобрать.
        """
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning(
                "Промпт отклонён: пустая строка или не строка (started_at=%s)",
                started_at.isoformat(),
            )
            raise ValueError("prompt не может быть пустым")

        logger.info(
            "Начало обработки промпта: started_at=%s, prompt=%r, "
            "system_prompt=%r, temperature=%s, max_tokens=%s",
            started_at.isoformat(),
            _preview(prompt, 300),
            _preview(system_prompt, 200) if system_prompt else system_prompt,
            temperature,
            max_tokens,
        )
        logger.debug("Полный текст промпта: %s", prompt)

        cache_key = _build_cache_key(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Кэш проверяется до подготовки промпта: при попадании тело запроса
        # не формируется вовсе.
        cached = get_cached_response(cache_key)
        if cached is not None:
            elapsed = time.perf_counter() - started
            logger.info(
                "Кэш: успешный поиск (попадание), cache_key=%r, elapsed=%.3f c",
                _preview(cache_key, 300),
                elapsed,
            )
            logger.debug("Ответ из кэша: %s", cached)
            return LLMAnswer(text=cached, from_cache=True)

        logger.info(
            "Кэш: неуспешный поиск (промах), cache_key=%r — "
            "готовим промпт и обращаемся к YandexGPT",
            _preview(cache_key, 300),
        )

        payload = self._build_payload(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(
            "Промпт подготовлен: model=%s, temperature=%s, max_tokens=%s, "
            "сообщений=%d",
            payload.get("modelUri"),
            payload["completionOptions"].get("temperature"),
            payload["completionOptions"].get("maxTokens"),
            len(payload.get("messages", [])),
        )
        logger.debug("Тело запроса к YandexGPT: %s", payload)

        logger.info(
            "Обращение к YandexGPT: model=%s",
            payload.get("modelUri"),
        )
        llm_started = time.perf_counter()
        try:
            response = await self._generator(payload)
        except LLMError as exc:
            # Ошибки LLM-слоя уже классифицированы клиентом.
            logger.error(
                "Ошибка YandexGPT: %s: %s (elapsed=%.3f c)",
                type(exc).__name__,
                exc,
                time.perf_counter() - llm_started,
            )
            raise
        except Exception as exc:
            # Непредвиденная ошибка транспорта/провайдера: приводим к общему
            # типу, чтобы API вернул пользователю 503, а не 500.
            logger.exception(
                "Неожиданная ошибка при обращении к LLM (elapsed=%.3f c)",
                time.perf_counter() - llm_started,
            )
            raise LLMUnavailableError(
                "не удалось получить ответ от LLM"
            ) from exc

        llm_elapsed = time.perf_counter() - llm_started
        logger.info(
            "Ответ YandexGPT получен: attempts=%s, длина=%d, elapsed=%.3f c",
            response.attempts,
            len(response.text),
            llm_elapsed,
        )
        logger.debug("Сырой ответ YandexGPT: %s", response.text)

        text = postprocess_llm_response(response.text)
        add_to_cache(cache_key, text)
        logger.info(
            "Ответ сохранён в кэш: cache_key=%r",
            _preview(cache_key, 300),
        )
        logger.debug("Итоговый ответ после пост-обработки: %s", text)

        total_elapsed = time.perf_counter() - started
        logger.info(
            "Обработка промпта завершена: from_cache=False, attempts=%s, "
            "длина_ответа=%d, elapsed=%.3f c",
            response.attempts,
            len(text),
            total_elapsed,
        )
        return LLMAnswer(
            text=text,
            from_cache=False,
            attempts=response.attempts,
        )

    async def get_answer(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Как :meth:`answer`, но возвращает только текст ответа."""
        result = await self.answer(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.text

    def _build_payload(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Формирует тело запроса, подставляя параметры по умолчанию."""
        defaults = self._defaults

        resolved_system_prompt = (
            system_prompt if system_prompt is not None else defaults.system_prompt
        )
        if resolved_system_prompt is None:
            resolved_system_prompt = DEFAULT_SYSTEM_PROMPT

        return build_payload(
            prompt,
            model_uri=defaults.model_uri,
            system_prompt=resolved_system_prompt,
            temperature=(
                temperature if temperature is not None else defaults.temperature
            ),
            max_tokens=(
                max_tokens if max_tokens is not None else defaults.max_tokens
            ),
        )


_default_service: CachedLLMService | None = None


def get_service() -> CachedLLMService:
    """Возвращает общий экземпляр сервиса, создавая его при первом обращении."""
    global _default_service
    if _default_service is None:
        _default_service = CachedLLMService()
    return _default_service


async def answer_prompt(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMAnswer:
    """Готовая функция: кэш → промпт → YandexGPT → кэш. Возвращает :class:`LLMAnswer`."""
    return await get_service().answer(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def get_answer(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Готовая функция: возвращает только текст ответа на промпт."""
    return await get_service().get_answer(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
