"""Исключения слоя обращения к LLM."""

from __future__ import annotations


class LLMError(Exception):
    """Базовое исключение для всех ошибок LLM-слоя."""


class LLMConfigurationError(LLMError):
    """Некорректная конфигурация клиента (нет ключа, папки, неверные лимиты)."""


class LLMRequestError(LLMError):
    """API отклонил запрос по постоянной причине (400/401/403/404 и т.п.).

    Такие ошибки не имеет смысла повторять: запрос нужно исправить.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMUnavailableError(LLMError):
    """Временная ошибка: таймаут, сетевая проблема или ответ 429/5xx."""


class LLMResponseError(LLMError):
    """Ответ получен, но его не удалось разобрать или он пустой."""
