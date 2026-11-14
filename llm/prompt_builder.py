"""Формирование промпта и тела запроса к LLM."""

from __future__ import annotations

from typing import Any

#: Системный промпт по умолчанию.
DEFAULT_SYSTEM_PROMPT = "Ты — полезный ассистент. Ответь кратко."


def build_messages(
    prompt: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Собирает сообщения: системное (если задано) и пользовательское."""
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "text": system_prompt.strip()})
    messages.append({"role": "user", "text": prompt})
    return messages


def build_payload(
    prompt: str,
    *,
    model_uri: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.6,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Формирует тело запроса к Foundation Models API."""
    return {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        },
        "messages": build_messages(prompt, system_prompt),
    }
