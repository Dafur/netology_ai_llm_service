"""Простой in-memory кэш ответов LLM с поддержкой TTL.

Кэш хранится в словаре ``{промпт: CacheEntry}`` и живёт, пока жив процесс.
Каждая запись имеет время жизни (TTL), по умолчанию 30 секунд.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Время жизни записи кэша по умолчанию, в секундах.
DEFAULT_TTL: float = 30.0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Запись кэша: ответ модели и момент истечения срока жизни."""

    #: Сохранённый ответ модели.
    response: str
    #: Момент времени (``time.monotonic()``), после которого запись устаревает.
    ttl: float

    def is_expired(self, now: float | None = None) -> bool:
        """Проверяет, истёк ли срок жизни записи.

        Args:
            now: Текущее значение ``time.monotonic()``. Передаётся явно
                в тестах, чтобы не зависеть от реальных часов.
        """
        return (time.monotonic() if now is None else now) >= self.ttl


#: Хранилище кэша: промпт -> запись кэша.
_cache: dict[str, CacheEntry] = {}


def _normalize(prompt: str) -> str:
    """Приводит промпт к единому виду, чтобы не плодить дубли."""
    return " ".join(prompt.split())


def get_cached_response(prompt: str) -> str | None:
    """Ищет ответ в кэше.

    Возвращает сохранённый ответ или ``None``, если промпта в кэше нет
    либо срок жизни записи истёк. Просроченная запись удаляется.
    """
    key = _normalize(prompt)
    entry = _cache.get(key)
    if entry is None:
        logger.debug("Кэш: промах, ключ отсутствует (%r)", key)
        return None
    if entry.is_expired():
        del _cache[key]
        logger.debug("Кэш: запись истекла и удалена (%r)", key)
        return None
    logger.debug("Кэш: попадание (%r)", key)
    return entry.response


def add_to_cache(
    prompt: str,
    response: str,
    ttl: float = DEFAULT_TTL,
) -> None:
    """Добавляет пару «промпт — ответ» в кэш (перезаписывая старую запись).

    Args:
        prompt: Текст запроса, по которому строится ключ кэша.
        response: Ответ модели, который нужно закэшировать.
        ttl: Время жизни записи в секундах. По умолчанию 30 секунд.
    """
    key = _normalize(prompt)
    _cache[key] = CacheEntry(
        response=response,
        ttl=time.monotonic() + ttl,
    )
    logger.debug(
        "Кэш: добавлена запись (%r, длина_ответа=%d, ttl=%s c)",
        key,
        len(response),
        ttl,
    )


def clear_cache() -> None:
    """Полностью очищает кэш."""
    logger.debug("Кэш: очистка (было записей: %d)", len(_cache))
    _cache.clear()


def _purge_expired() -> None:
    """Удаляет из кэша все записи с истёкшим TTL."""
    now = time.monotonic()
    expired = [key for key, entry in _cache.items() if entry.is_expired(now)]
    for key in expired:
        del _cache[key]
    if expired:
        logger.debug("Кэш: удалено просроченных записей: %d", len(expired))


def cache_size() -> int:
    """Возвращает количество актуальных (не истёкших) записей в кэше."""
    _purge_expired()
    return len(_cache)
