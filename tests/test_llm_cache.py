"""Тесты in-memory кэша ответов LLM с поддержкой TTL."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cache import (
    DEFAULT_TTL,
    add_to_cache,
    cache_size,
    clear_cache,
    get_cached_response,
)


class LLMCacheTTLTests(unittest.TestCase):
    """Проверяет срок жизни записей кэша."""

    def setUp(self) -> None:
        clear_cache()

    def tearDown(self) -> None:
        clear_cache()

    def test_default_ttl_is_30_seconds(self) -> None:
        self.assertEqual(DEFAULT_TTL, 30.0)

    def test_entry_is_returned_before_ttl_expires(self) -> None:
        with patch("cache.llm_cache.time.monotonic", return_value=100.0):
            add_to_cache("Вопрос", "Ответ")
            self.assertEqual(get_cached_response("Вопрос"), "Ответ")
            self.assertEqual(cache_size(), 1)

    def test_entry_expires_after_default_ttl(self) -> None:
        with patch("cache.llm_cache.time.monotonic", return_value=100.0):
            add_to_cache("Вопрос", "Ответ")

        # Ровно через 30 секунд запись уже просрочена.
        with patch("cache.llm_cache.time.monotonic", return_value=130.0):
            self.assertIsNone(get_cached_response("Вопрос"))
            self.assertEqual(cache_size(), 0)

    def test_custom_ttl(self) -> None:
        with patch("cache.llm_cache.time.monotonic", return_value=0.0):
            add_to_cache("Вопрос", "Ответ", ttl=5.0)

        with patch("cache.llm_cache.time.monotonic", return_value=4.9):
            self.assertEqual(get_cached_response("Вопрос"), "Ответ")

        with patch("cache.llm_cache.time.monotonic", return_value=5.0):
            self.assertIsNone(get_cached_response("Вопрос"))

    def test_prompt_is_normalized(self) -> None:
        with patch("cache.llm_cache.time.monotonic", return_value=0.0):
            add_to_cache("  Вопрос   с   пробелами ", "Ответ")
            self.assertEqual(get_cached_response("Вопрос с пробелами"), "Ответ")

    def test_expired_entries_are_not_counted(self) -> None:
        with patch("cache.llm_cache.time.monotonic", return_value=0.0):
            add_to_cache("Первый", "Ответ 1", ttl=10.0)
            add_to_cache("Второй", "Ответ 2", ttl=100.0)

        with patch("cache.llm_cache.time.monotonic", return_value=50.0):
            self.assertEqual(cache_size(), 1)


if __name__ == "__main__":
    unittest.main()
