"""Пакет сервисного слоя.

Содержит модули, связывающие кэш и LLM-провайдеров в готовые сценарии.
"""

from services.llm_service import (
    CachedLLMService,
    GenerationDefaults,
    LLMAnswer,
    answer_prompt,
    get_answer,
    get_service,
)

__all__ = [
    "CachedLLMService",
    "GenerationDefaults",
    "LLMAnswer",
    "answer_prompt",
    "get_answer",
    "get_service",
]
