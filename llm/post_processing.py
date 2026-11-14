"""Пост-обработка текстовых ответов LLM.

Модуль приводит «сырой» текст от модели к аккуратному виду:

- нормализует переводы строк (``\r\n``/``\r`` → ``\n``);
- убирает лишние пробелы и табы внутри строк;
- схлопывает повторяющиеся пустые строки до одной;
- обрезает пробелы в начале и конце текста;
- проверяет, что после нормализации ответ не пустой.
"""

from __future__ import annotations

import re

from llm.exceptions import LLMResponseError

#: Подряд идущие пробельные символы внутри строки (кроме перевода строки).
_INLINE_SPACES_RE = re.compile(r"[ \t\f\v]+")

#: Три и более переводов строки подряд — лишние пустые строки.
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Удаляет лишние пробелы, сохраняя разбиение текста на абзацы."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        _INLINE_SPACES_RE.sub(" ", line).strip() for line in normalized.split("\n")
    ]
    joined = "\n".join(lines)
    joined = _BLANK_LINES_RE.sub("\n\n", joined)
    return joined.strip()


def postprocess_llm_response(
    text: str | None,
    *,
    field_name: str = "ответ LLM",
) -> str:
    """Нормализует ответ LLM и проверяет, что он не пустой.

    Args:
        text: «Сырой» текст ответа модели.
        field_name: Название поля для текста ошибки.

    Returns:
        Очищенный непустой текст.

    Raises:
        LLMResponseError: Если ответ отсутствует, не является строкой или
            после удаления лишних пробелов оказался пустым.
    """
    if text is None:
        raise LLMResponseError(f"{field_name} отсутствует (None)")
    if not isinstance(text, str):
        raise LLMResponseError(
            f"{field_name} должен быть строкой, получено: {type(text).__name__}"
        )

    cleaned = normalize_whitespace(text)
    if not cleaned:
        raise LLMResponseError(f"{field_name} пуст")
    return cleaned
