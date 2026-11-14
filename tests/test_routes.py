"""Тесты HTTP-статусов и сообщений об ошибках эндпоинта /chat."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import Response

from llm.exceptions import (
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailableError,
)
from main import app


class ChatErrorHandlingTests(unittest.TestCase):
    """Проверяет преобразование ошибок в HTTP-статусы."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def _post(self, message: str = "Привет") -> Response:
        return self.client.post("/chat", json={"message": message})

    def test_blank_message_returns_400(self) -> None:
        response = self._post("   ")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Некорректные входные данные")

    def test_value_error_returns_400(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=ValueError("prompt не может быть пустым")),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "prompt не может быть пустым")

    def test_unavailable_error_returns_503(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=LLMUnavailableError("нет связи")),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 503)
        self.assertIn("недоступен", response.json()["detail"])

    def test_configuration_error_returns_503(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=LLMConfigurationError("нет ключа")),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 503)

    def test_request_error_returns_502(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=LLMRequestError("HTTP 400", status_code=400)),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 502)

    def test_response_error_returns_502(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=LLMResponseError("не JSON")),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 502)

    def test_unexpected_error_returns_500(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Внутренняя ошибка сервиса")

    def test_successful_answer(self) -> None:
        with patch(
            "api.routes.get_answer",
            new=AsyncMock(return_value="Ответ модели"),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": "Ответ модели"})


if __name__ == "__main__":
    unittest.main()
