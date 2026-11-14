# Учебный LLM-сервис

FastAPI-сервис, который принимает сообщение пользователя, обращается к YandexGPT
и возвращает текстовый ответ. Включает in-memory кэш ответов, повторные попытки
при временных сбоях, таймауты, пост-обработку текста и логирование в консоль и файл.

## Возможности

- `POST /chat` — ответ на сообщение пользователя через YandexGPT.
- `GET /health` — проверка работоспособности сервиса.
- In-memory кэш пар «промпт — ответ» с TTL 30 секунд.
- Повторы только временных ошибок (таймауты, сеть, HTTP 429/5xx)
  с экспоненциальной задержкой и джиттером.
- Понятные HTTP-статусы ошибок: 400, 502, 503, 500.
- Логирование в консоль и в ротируемый файл `logs/app.log`.
- Интерактивная документация OpenAPI: `/docs`, `/redoc`, `/openapi.json`.

## Требования

- Python 3.11 или новее.
- Аккаунт Yandex Cloud и доступ к YandexGPT:
  - API-ключ сервисного аккаунта с ролью `ai.languageModels.user`;
  - идентификатор каталога (folder ID).

## Структура проекта

```
my_llm_service/
├── main.py                  # точка входа FastAPI, middleware и обработчики ошибок
├── requirements.txt         # зависимости
├── .env.example             # шаблон переменных окружения
├── .env                     # локальные секреты (не коммитится)
├── api/
│   ├── routes.py            # эндпоинты /health и /chat
│   └── schemas.py           # Pydantic-схемы запроса и ответа
├── cache/
│   └── llm_cache.py         # in-memory кэш с TTL
├── config/
│   └── logging_config.py    # настройка логирования
├── llm/
│   ├── yandex_gpt.py        # асинхронный клиент YandexGPT (retry, timeout)
│   ├── prompt_builder.py    # формирование промпта и тела запроса
│   ├── post_processing.py   # нормализация ответа модели
│   └── exceptions.py        # типы ошибок LLM-слоя
├── services/
│   └── llm_service.py       # кэш → промпт → YandexGPT → пост-обработка
├── tests/                   # unit-тесты (unittest)
└── logs/
    └── app.log              # файл логов (создаётся автоматически)
```

## Установка

### 1. Клонирование и виртуальное окружение

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если PowerShell блокирует активацию, разрешите выполнение локальных скриптов:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Настройка окружения и `.env`

Сервис читает переменные окружения и файл `.env` из рабочей директории
(используется `python-dotenv`). Создайте `.env` на основе шаблона:

Windows (cmd):

```bat
copy .env.example .env
```

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Откройте `.env` и заполните обязательные значения:

```dotenv
YANDEX_GPT_API_KEY=ваш-api-ключ
YANDEX_GPT_FOLDER_ID=ваш-folder-id
```

> Файл `.env` содержит секреты. Не добавляйте его в систему контроля версий
> и не публикуйте API-ключ.

#### Переменные окружения YandexGPT

| Переменная | Обязательна | По умолчанию | Описание |
|---|---|---|---|
| `YANDEX_GPT_API_KEY` | да | — | API-ключ сервисного аккаунта |
| `YANDEX_GPT_FOLDER_ID` | да | — | Идентификатор каталога Yandex Cloud |
| `YANDEX_GPT_MODEL` | нет | `yandexgpt-lite` | Имя модели |
| `YANDEX_GPT_MODEL_VERSION` | нет | `latest` | Версия модели |
| `YANDEX_GPT_TEMPERATURE` | нет | `0.6` | Температура генерации, от `0` до `1` |
| `YANDEX_GPT_MAX_TOKENS` | нет | `1000` | Максимум токенов в ответе |
| `YANDEX_GPT_TIMEOUT` | нет | `20` | Таймаут одного запроса, секунды |
| `YANDEX_GPT_MAX_RETRIES` | нет | `3` | Повторы сверх первой попытки |
| `YANDEX_GPT_SYSTEM_PROMPT` | нет | `Ты — полезный ассистент. Ответь кратко.` | Системный промпт |
| `YANDEX_GPT_BACKOFF_BASE` | нет | `0.5` | Базовая задержка перед повтором, секунды |
| `YANDEX_GPT_BACKOFF_MAX` | нет | `8.0` | Максимальная задержка между попытками, секунды |

Идентификатор модели формируется автоматически:
`gpt://<YANDEX_GPT_FOLDER_ID>/<YANDEX_GPT_MODEL>/<YANDEX_GPT_MODEL_VERSION>`.

#### Переменные окружения логирования

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Базовый уровень логирования |
| `LOG_CONSOLE_LEVEL` | равен `LOG_LEVEL` | Уровень для консоли |
| `LOG_FILE_LEVEL` | равен `LOG_LEVEL` | Уровень для файла |
| `LOG_FILE` | `logs/app.log` | Путь к файлу лога |
| `LOG_MAX_BYTES` | `5242880` | Размер файла до ротации (5 МБ) |
| `LOG_BACKUP_COUNT` | `5` | Количество резервных копий |
| `LOG_TO_CONSOLE` | `true` | Вывод в консоль |
| `LOG_TO_FILE` | `true` | Вывод в файл |

Допустимые уровни: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
Булевы значения: `true/false`, `1/0`, `yes/no`, `on/off`.

Переменные окружения имеют приоритет над значениями из `.env`.
Конфигурация YandexGPT читается при первом обращении к модели, поэтому после
изменения `.env` перезапустите сервис.

## Запуск

Рекомендуемый способ (сохраняет настройки логирования приложения):

```bash
python main.py
```

Сервис будет доступен по адресу <http://127.0.0.1:8000>.
Запуск выполняется с `reload=True`, поэтому изменения в коде применяются автоматически.

Альтернативный запуск через uvicorn:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Проверка после запуска:

- Swagger UI — <http://127.0.0.1:8000/docs>
- ReDoc — <http://127.0.0.1:8000/redoc>
- OpenAPI-схема — <http://127.0.0.1:8000/openapi.json>
- Проверка здоровья — <http://127.0.0.1:8000/health>

## API

### `GET /health`

Проверка работоспособности. Не обращается к YandexGPT и не требует валидных ключей.

Ответ `200 OK`:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Принимает сообщение пользователя и возвращает ответ модели.

Тело запроса:

| Поле | Тип | Обязательно | Ограничения |
|---|---|---|---|
| `message` | string | да | Не пустое после обрезки пробелов, не длиннее 1000 символов |

Пример ответа `200 OK`:

```json
{
  "reply": "FastAPI — это современный асинхронный веб-фреймворк для Python."
}
```

#### Примеры запросов

`curl` в Windows cmd:

```bat
curl http://127.0.0.1:8000/health

curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Привет! Что такое FastAPI?\"}"
```

`curl` в Linux/macOS/bash:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Привет! Что такое FastAPI?"}'
```

PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get

Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body (@{ message = "Привет! Что такое FastAPI?" } | ConvertTo-Json)
```

Python (`httpx` уже установлен как зависимость):

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/chat",
    json={"message": "Привет! Что такое FastAPI?"},
    timeout=30,
)
response.raise_for_status()
print(response.json()["reply"])
```

Пример ответа на запрос с пустым сообщением (`400 Bad Request`):

```json
{
  "detail": "Некорректные входные данные",
  "errors": [
    {
      "type": "string_too_short",
      "loc": ["body", "message"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": {"min_length": 1}
    }
  ]
}
```

#### Статусы ошибок

| Статус | Причина |
|---|---|
| `400` | Некорректное тело запроса, пустое сообщение или длина больше 1000 символов |
| `502` | YandexGPT вернул некорректный или неразбираемый ответ |
| `503` | YandexGPT недоступен, не настроен (`YANDEX_GPT_API_KEY`/`YANDEX_GPT_FOLDER_ID`), исчерпаны повторы |
| `500` | Непредвиденная внутренняя ошибка сервиса |

## Кэширование

- Кэш хранится в памяти процесса и очищается при перезапуске.
- Ключ кэша — нормализованный промпт (лишние пробелы схлопываются).
- Время жизни записи — 30 секунд.
- Повторный одинаковый запрос в течение TTL возвращает ответ без обращения
  к YandexGPT, что экономит токены и ускоряет ответ.
- В ответе API нет отдельного признака попадания в кэш; факт попадания
  виден в логах (`Кэш: успешный поиск (попадание)`).

## Логирование

По умолчанию сообщения пишутся в консоль и в `logs/app.log`.
Файл ротируется при достижении 5 МБ, хранится до 5 резервных копий.
Логи содержат метод, путь и длительность HTTP-запросов, попытки обращения
к YandexGPT, попадания в кэш и тексты ошибок.

Для подробной диагностики задайте в `.env`:

```dotenv
LOG_LEVEL=DEBUG
LOG_CONSOLE_LEVEL=DEBUG
LOG_FILE_LEVEL=DEBUG
```

## Тесты

Тесты используют стандартный `unittest` и не обращаются к реальному YandexGPT
(сетевые вызовы замоканы):

```bash
python -m unittest discover -s tests -v
```

Запуск отдельного модуля:

```bash
python -m unittest tests.test_routes -v
```

## Диагностика проблем

- **`503` при первом запросе к `/chat`.** Проверьте, что `.env` находится
  в рабочей директории процесса и содержит `YANDEX_GPT_API_KEY` и
  `YANDEX_GPT_FOLDER_ID`. Убедитесь, что у сервисного аккаунта есть роль
  `ai.languageModels.user` и доступ к каталогу.
- **`502`.** Модель вернула пустой текст или ответ не является ожидаемым JSON.
  Подробности — в `logs/app.log` на уровне `DEBUG`.
- **`400`.** Проверьте JSON: поле `message` обязательно, не пустое после
  обрезки пробелов и не длиннее 1000 символов.
- **Кириллица в Windows cmd отображается некорректно.** Выполните
  `chcp 65001` или используйте PowerShell либо Python-пример.
- **Изменения в `.env` не применяются.** Перезапустите сервис: настройки
  читаются при старте и при создании клиента YandexGPT.
