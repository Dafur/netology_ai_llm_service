"""Тесты политики повторов клиента YandexGPT."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from llm.exceptions import LLMRequestError, LLMResponseError, LLMUnavailableError
from llm.yandex_gpt import YandexGPTClient, YandexGPTConfig


def _config(max_retries: int = 2) -> YandexGPTConfig:
    """Конфигурация без реальных задержек между попытками."""
    return YandexGPTConfig(
        api_key="test-key",
        folder_id="test-folder",
        timeout=1.0,
        max_retries=max_retries,
        backoff_base=0.0,
        backoff_max=0.0,
    )


def _ok_data(text: str = "Ответ") -> dict[str, object]:
    """Успешный ответ в формате Foundation Models API."""
    return {"result": {"alternatives": [{"message": {"text": text}}]}}


class YandexGPTRetryTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет, что повторяются только временные ошибки."""

    async def asyncSetUp(self) -> None:
        self.sleep = AsyncMock()
        self.client = YandexGPTClient(_config(), sleep=self.sleep)
        self.addAsyncCleanup(self.client.aclose)

    async def test_temporary_error_is_retried(self) -> None:
        self.client._request = AsyncMock(  # type: ignore[method-assign]
            side_effect=[LLMUnavailableError("timeout"), _ok_data("готово")]
        )

        response = await self.client.complete({"modelUri": "test"})

        self.assertEqual(response.text, "готово")
        self.assertEqual(response.attempts, 2)
        self.assertEqual(self.client._request.await_count, 2)
        self.sleep.assert_awaited_once()

    async def test_temporary_error_is_retried_until_exhausted(self) -> None:
        self.client._request = AsyncMock(  # type: ignore[method-assign]
            side_effect=LLMUnavailableError("HTTP 503")
        )

        with self.assertRaises(LLMUnavailableError):
            await self.client.complete({"modelUri": "test"})

        self.assertEqual(self.client._request.await_count, 3)

    async def test_request_error_is_not_retried(self) -> None:
        self.client._request = AsyncMock(  # type: ignore[method-assign]
            side_effect=LLMRequestError("HTTP 400", status_code=400)
        )

        with self.assertRaises(LLMRequestError):
            await self.client.complete({"modelUri": "test"})

        self.assertEqual(self.client._request.await_count, 1)
        self.sleep.assert_not_awaited()

    async def test_response_error_is_not_retried(self) -> None:
        self.client._request = AsyncMock(  # type: ignore[method-assign]
            side_effect=LLMResponseError("не JSON")
        )

        with self.assertRaises(LLMResponseError):
            await self.client.complete({"modelUri": "test"})

        self.assertEqual(self.client._request.await_count, 1)
        self.sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
