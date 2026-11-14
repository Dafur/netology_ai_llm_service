"""Pydantic-схемы запросов и ответов API."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Тело запроса к эндпоинту /chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Сообщение пользователя (не пустое, не длиннее 1000 символов)",
        examples=["Привет!"],
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, value: object) -> object:
        """Убирает ведущие и завершающие пробелы до проверки длины."""
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    """Ответ эндпоинта /chat."""

    reply: str
