"""Тесты сервисного слоя кэширования ответов LLM."""

from __future__ import annotations

import unittest
from typing import Any, Mapping
from unittest.mock import patch

from cache import cache_size, clear_cache
from llm.exceptions import LLMUnavailableError
from llm.prompt_builder import build_payload
from llm.yandex_gpt import LLMResponse
from services import CachedLLMService, GenerationDefaults


def _response(text: str = "Ответ модели") -> LLMResponse:
    """Собирает тестовый ответ клиента YandexGPT."""
    return LLMResponse(
        text=text,
        model_uri="gpt://test/folder/model/latest",
        attempts=1,
    )


def _defaults() -> GenerationDefaults:
    """Параметры генерации для тестового сервиса."""
    return GenerationDefaults(
        model_uri="gpt://test/folder/model/latest",
        system_prompt="Ты — полезный ассистент. Ответь кратко.",
        temperature=0.6,
        max_tokens=1000,
    )


class CachedLLMServiceTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет порядок «кэш → промпт → модель → кэш»."""

    def setUp(self) -> None:
        clear_cache()

    def tearDown(self) -> None:
        clear_cache()

    async def test_second_request_uses_cache(self) -> None:
        calls: list[Mapping[str, Any]] = []

        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            calls.append(payload)
            return _response("  Ответ   модели ")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())

        first = await service.answer("Что такое FastAPI?")
        second = await service.answer("Что такое FastAPI?")

        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(first.text, "Ответ модели")
        self.assertEqual(second.text, "Ответ модели")
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache_size(), 1)

    async def test_cache_hit_skips_prompt_building(self) -> None:
        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            return _response("Ответ модели")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())

        with patch(
            "services.llm_service.build_payload", wraps=build_payload
        ) as build_mock:
            await service.answer("Вопрос")
            self.assertEqual(build_mock.call_count, 1)

            second = await service.answer("Вопрос")

        self.assertTrue(second.from_cache)
        # Повторный запрос обслужен кэшем: промпт повторно не строился.
        self.assertEqual(build_mock.call_count, 1)

    async def test_unavailable_error_is_not_cached(self) -> None:
        calls = 0

        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            nonlocal calls
            calls += 1
            raise LLMUnavailableError("модель недоступна")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())

        for _ in range(2):
            with self.assertRaises(LLMUnavailableError):
                await service.answer("Привет")

        self.assertEqual(calls, 2)
        self.assertEqual(cache_size(), 0)

    async def test_unexpected_generator_error_is_wrapped(self) -> None:
        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            raise RuntimeError("boom")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())

        with self.assertRaises(LLMUnavailableError):
            await service.answer("Вопрос")
        self.assertEqual(cache_size(), 0)

    async def test_generation_options_split_cache_entries(self) -> None:
        calls = 0

        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            nonlocal calls
            calls += 1
            return _response(f"Ответ #{calls}")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())

        await service.answer("Вопрос", temperature=0.1)
        await service.answer("Вопрос", temperature=0.9)
        await service.answer("Вопрос", temperature=0.1)

        self.assertEqual(calls, 2)
        self.assertEqual(cache_size(), 2)

    async def test_options_are_forwarded_to_payload(self) -> None:
        received: dict[str, Mapping[str, Any]] = {}

        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            received["payload"] = payload
            return _response()

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())
        await service.answer("Вопрос", system_prompt="Будь краток", max_tokens=100)

        payload = received["payload"]
        self.assertEqual(payload["modelUri"], "gpt://test/folder/model/latest")
        self.assertEqual(payload["completionOptions"]["maxTokens"], "100")
        self.assertEqual(payload["completionOptions"]["temperature"], 0.6)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "text": "Будь краток"},
                {"role": "user", "text": "Вопрос"},
            ],
        )

    async def test_defaults_are_used_when_options_not_passed(self) -> None:
        received: dict[str, Mapping[str, Any]] = {}

        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            received["payload"] = payload
            return _response()

        defaults = GenerationDefaults(
            model_uri="gpt://test/folder/model/latest",
            system_prompt=None,
            temperature=0.2,
            max_tokens=50,
        )
        service = CachedLLMService(generator=fake_generate, defaults=defaults)
        await service.answer("Вопрос")

        payload = received["payload"]
        self.assertEqual(payload["completionOptions"]["maxTokens"], "50")
        self.assertEqual(payload["completionOptions"]["temperature"], 0.2)
        # Если системный промпт не задан, используется DEFAULT_SYSTEM_PROMPT.
        self.assertEqual(
            payload["messages"][0],
            {
                "role": "system",
                "text": "Ты — полезный ассистент. Ответь кратко.",
            },
        )

    async def test_empty_prompt_is_rejected(self) -> None:
        async def fake_generate(payload: Mapping[str, Any]) -> LLMResponse:
            raise AssertionError("генератор не должен вызываться")

        service = CachedLLMService(generator=fake_generate, defaults=_defaults())
        with self.assertRaises(ValueError):
            await service.answer("   ")


if __name__ == "__main__":
    unittest.main()
