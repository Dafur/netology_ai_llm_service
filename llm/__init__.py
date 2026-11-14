"""Пакет интеграции с LLM-провайдерами.

Сейчас содержит клиент YandexGPT с повторами временных ошибок и таймаутом.
"""

from llm.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
from llm.post_processing import (
    normalize_whitespace,
    postprocess_llm_response,
)
from llm.prompt_builder import (
    DEFAULT_SYSTEM_PROMPT,
    build_messages,
    build_payload,
)
from llm.yandex_gpt import (
    LLMResponse,
    YandexGPTClient,
    YandexGPTConfig,
    ask_yandex_gpt,
    close_default_client,
    generate_yandex_gpt,
    get_default_client,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLMConfigurationError",
    "LLMError",
    "LLMRequestError",
    "LLMResponse",
    "LLMResponseError",
    "LLMUnavailableError",
    "YandexGPTClient",
    "YandexGPTConfig",
    "ask_yandex_gpt",
    "build_messages",
    "build_payload",
    "close_default_client",
    "generate_yandex_gpt",
    "get_default_client",
    "normalize_whitespace",
    "postprocess_llm_response",
]
