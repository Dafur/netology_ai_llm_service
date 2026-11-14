"""Настройка логирования приложения.

Модуль настраивает корневой логгер так, чтобы сообщения одновременно
попадали в консоль и в файл. Файл ротируется по размеру, чтобы логи
не разрастались бесконечно.

Переменные окружения:

``LOG_LEVEL``
    Базовый уровень логирования (по умолчанию ``INFO``).
``LOG_CONSOLE_LEVEL``
    Уровень для консоли (по умолчанию равен ``LOG_LEVEL``).
``LOG_FILE_LEVEL``
    Уровень для файла (по умолчанию равен ``LOG_LEVEL``).
``LOG_FILE``
    Путь к файлу лога (по умолчанию ``logs/app.log``).
``LOG_MAX_BYTES``
    Максимальный размер файла до ротации (по умолчанию 5 МБ).
``LOG_BACKUP_COUNT``
    Количество хранимых резервных копий (по умолчанию 5).
``LOG_TO_CONSOLE`` / ``LOG_TO_FILE``
    Включают вывод в консоль/файл (``true``/``false``).

Пример использования::

    from config import configure_logging

    configure_logging()
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv заявлен в requirements
    load_dotenv = None  # type: ignore[assignment]

#: Формат сообщений: время, уровень, имя логгера, текст.
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Формат времени в логах.
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Путь к файлу лога по умолчанию.
DEFAULT_LOG_FILE = "logs/app.log"

#: Размер файла до ротации — 5 МБ.
DEFAULT_MAX_BYTES = 5 * 1024 * 1024

#: Количество резервных копий файла лога.
DEFAULT_BACKUP_COUNT = 5

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "n"})

#: Признак того, что логирование уже настроено (защита от дублей обработчиков).
_configured = False


def _load_env_file() -> None:
    """Загружает переменные из ``.env``, если установлен python-dotenv."""
    if load_dotenv is not None:
        load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Читает булеву переменную окружения; при ошибке возвращает ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Читает целочисленную переменную окружения; при ошибке — ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_level(value: str | int | None, default: int) -> int:
    """Преобразует строковое/числовое имя уровня в числовой уровень logging."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    name = value.strip().upper()
    if not name:
        return default
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else default


def build_formatter() -> logging.Formatter:
    """Создаёт общий форматтер для всех обработчиков."""
    return logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)


def configure_logging(
    *,
    level: str | int | None = None,
    console_level: str | int | None = None,
    file_level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
    to_console: bool | None = None,
    to_file: bool | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    force: bool = False,
) -> None:
    """Настраивает корневой логгер: вывод в консоль и в ротируемый файл.

    Функция идемпотентна: повторный вызов без ``force=True`` не добавляет
    обработчики заново. Параметры, не переданные явно, берутся из переменных
    окружения (см. docstring модуля).

    Args:
        level: Базовый уровень логирования.
        console_level: Уровень сообщений для консоли.
        file_level: Уровень сообщений для файла.
        log_file: Путь к файлу лога.
        to_console: Включать ли вывод в консоль.
        to_file: Включать ли вывод в файл.
        max_bytes: Максимальный размер файла до ротации.
        backup_count: Количество резервных копий файла лога.
        force: Удалить ранее установленные обработчики и настроить заново.
    """
    global _configured
    if _configured and not force:
        return

    # .env может содержать LOG_LEVEL/LOG_FILE и другие настройки,
    # поэтому читаем его до разбора переменных окружения.
    _load_env_file()

    base_level = _resolve_level(
        level if level is not None else os.getenv("LOG_LEVEL"),
        logging.INFO,
    )
    resolved_console_level = _resolve_level(
        console_level if console_level is not None else os.getenv("LOG_CONSOLE_LEVEL"),
        base_level,
    )
    resolved_file_level = _resolve_level(
        file_level if file_level is not None else os.getenv("LOG_FILE_LEVEL"),
        base_level,
    )

    resolved_file = (
        log_file if log_file is not None else os.getenv("LOG_FILE", DEFAULT_LOG_FILE)
    )
    resolved_to_console = (
        to_console if to_console is not None else _env_bool("LOG_TO_CONSOLE", True)
    )
    resolved_to_file = (
        to_file if to_file is not None else _env_bool("LOG_TO_FILE", True)
    )
    resolved_max_bytes = (
        max_bytes
        if max_bytes is not None
        else _env_int("LOG_MAX_BYTES", DEFAULT_MAX_BYTES)
    )
    resolved_backup_count = (
        backup_count
        if backup_count is not None
        else _env_int("LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT)
    )

    root = logging.getLogger()

    # Убираем обработчики, установленные этим модулем при прошлой настройке,
    # чтобы не дублировать строки в логах.
    for handler in list(root.handlers):
        if force or getattr(handler, "_app_handler", False):
            root.removeHandler(handler)
            handler.close()

    levels: list[int] = []
    if resolved_to_console:
        levels.append(resolved_console_level)
    if resolved_to_file:
        levels.append(resolved_file_level)
    root.setLevel(min(levels) if levels else logging.CRITICAL)

    formatter = build_formatter()

    if resolved_to_console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(resolved_console_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, "_app_handler", True)
        root.addHandler(console_handler)

    if resolved_to_file:
        path = Path(resolved_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=resolved_max_bytes,
                backupCount=resolved_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(resolved_file_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, "_app_handler", True)
            root.addHandler(file_handler)
        except OSError as exc:
            # Недоступность файла не должна ронять приложение:
            # логи продолжат выводиться в консоль.
            logging.getLogger(__name__).warning(
                "Не удалось настроить запись логов в файл %s: %s", path, exc
            )

    # Сторонние HTTP-клиенты не должны засорять лог на уровне INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True
