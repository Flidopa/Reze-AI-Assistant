# CLAUDE.md

> Это главный контекстный документ проекта. Claude Code читает его перед каждой задачей.
> Документ — живой. Когда меняется архитектура или конвенции — обновляем здесь.

---

## 🎯 О проекте

**AI-ассистент** — голосовой персональный ассистент с собственным голосом (клонированным через RVC), реагирующий на имя, понимающий русскую речь, умеющий выходить в интернет и работать с Telegram (читать входящие, отвечать вместо пользователя, делать сводки).

Проект **не учебный**. Цель — рабочий продукт для личного использования с прицелом на долгое развитие.

**Ключевые возможности:**
- Wake-word активация (отзыв на имя)
- Speech-to-Text (русский язык)
- Диалог через LLM с function calling
- Text-to-Speech с клонированным голосом + эмоции
- Долгосрочная память (vector DB)
- Telegram-интеграция (user-account, не bot)
- Web search, weather, и другие tools
- В будущем — UI-дашборд

---

## 🏗️ Архитектурные принципы

> ⚠️ **Это самый важный раздел. Не нарушай эти принципы без явного обсуждения.**

### 1. Модули общаются ТОЛЬКО через Event Bus

Никаких прямых импортов между функциональными модулями (`audio`, `brain`, `tools`).
Исключение — общие типы из `core/events.py` и `core/config.py`.

❌ Плохо:
```python
# в audio/stt.py
from assistant.brain.orchestrator import Orchestrator
```

✅ Хорошо:
```python
# в audio/stt.py
from assistant.core.event_bus import EventBus
from assistant.core.events import UserSpeechDetected

await event_bus.emit(UserSpeechDetected(text=transcribed))
```

### 2. События — это контракт между модулями

Все события определяются в `core/events.py` как dataclass'ы.
Если модулю нужен новый тип события — сначала описываем его там, потом используем.

### 3. Тяжёлые ресурсы — lazy load

Модели (Whisper, XTTS, RVC) загружаются один раз при инициализации модуля, не на каждый вызов.

### 4. Все внешние вызовы — async + timeout + retry

LLM API, web search, Telegram — всё async, с таймаутами и обработкой ошибок.
Никаких блокирующих вызовов в основном loop'е.

### 5. main.py — только wiring

`main.py` создаёт инстансы модулей, подключает их к event bus, запускает loop. Никакой бизнес-логики там.

---

## 🛠️ Технологический стек

### Базовое
- **Python 3.11+**
- **Package manager**: `uv` (быстрее poetry, сейчас стандарт)
- **Async**: `asyncio`
- **Linter + formatter**: `ruff` (заменяет black + flake8 + isort)
- **Type checker**: `mypy` (strict mode для core)
- **Tests**: `pytest` + `pytest-asyncio`

### Аудио
- **Wake word**: `pvporcupine` (Picovoice Porcupine)
- **STT**: `faster-whisper` (модель `large-v3` для качества или `medium` для скорости)
- **TTS**: `TTS` от Coqui (XTTS-v2) — позже добавим Bark для эмоций
- **Voice cloning**: RVC (отдельный inference-сервис)
- **Audio I/O**: `sounddevice` + `numpy`

### Brain
- **LLM client**: `openai` SDK с base_url на OpenRouter (унифицированный интерфейс ко многим моделям)
- **Vector DB**: `chromadb` (локально, без отдельного сервера)
- **Embeddings**: `sentence-transformers` (модель `paraphrase-multilingual-MiniLM`)

### Tools
- **Telegram**: `telethon` (user-account через MTProto)
- **Web search**: `duckduckgo-search` (бесплатно) или `tavily-python` (если нужно лучше)
- **Weather**: прямой `httpx` к OpenWeather API

### Конфигурация
- `.env` для секретов (через `python-dotenv` или `pydantic-settings`)
- `pydantic` для конфигов и валидации событий

---

## 📁 Структура проекта

```
ai-assistant/
├── CLAUDE.md                  # этот файл
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .env                       # ❌ в gitignore
├── .gitignore
│
├── docs/
│   └── architecture.drawio    # схема архитектуры
│
├── src/
│   └── assistant/
│       ├── __init__.py
│       ├── main.py            # entry point — только wiring
│       │
│       ├── core/              # инфраструктура
│       │   ├── event_bus.py   # шина событий
│       │   ├── events.py      # все типы событий (dataclasses)
│       │   ├── config.py      # pydantic settings
│       │   └── logging.py     # настройка логов
│       │
│       ├── audio/             # аудио I/O
│       │   ├── wake_word.py
│       │   ├── stt.py
│       │   ├── tts.py
│       │   ├── player.py      # воспроизведение аудио через sounddevice
│       │   └── voice_cloning.py
│       │
│       ├── brain/             # мозг
│       │   ├── orchestrator.py
│       │   ├── memory.py
│       │   ├── prompts.py
│       │   └── llm_client.py
│       │
│       └── tools/             # function calling tools
│           ├── base.py        # базовый класс Tool
│           ├── registry.py    # регистрация tools
│           ├── web_search.py
│           ├── telegram.py
│           └── weather.py
│
├── tests/                     # зеркалит структуру src/
│   ├── core/
│   ├── audio/
│   ├── brain/
│   └── tools/
│
├── scripts/                   # утилиты (тренировка RVC, миграции)
│
└── data/                      # ❌ в gitignore
    ├── models/                # веса моделей
    ├── voice_samples/         # сэмплы для клонирования
    └── chroma/                # данные vector DB
```

---

## 🎨 Code conventions

### Naming
- `snake_case` — файлы, функции, переменные
- `PascalCase` — классы, dataclass'ы, события
- `UPPER_CASE` — константы
- Async-функции называются обычно (без суффикса `_async`)
- Приватные — с `_` префиксом

### Type hints — обязательны
```python
async def transcribe(audio: np.ndarray, language: str = "ru") -> str:
    ...
```

### Docstrings
Только для публичных API (методы классов, экспортируемые функции). Внутренние функции с понятными именами не требуют docstring.

Формат — Google style:
```python
def process(text: str) -> dict:
    """Обрабатывает входной текст.

    Args:
        text: входной текст пользователя.

    Returns:
        Словарь с распарсенным результатом.
    """
```

### Imports
Сортируются `ruff`. Порядок: stdlib → third-party → local.

### Логи, не print
```python
import logging
logger = logging.getLogger(__name__)
logger.info("starting wake word detector")
```

---

## ⚙️ Команды разработки

```bash
# установка зависимостей
uv sync

# запуск
uv run python -m assistant.main

# тесты
uv run pytest

# линтер + форматтер
uv run ruff check --fix
uv run ruff format

# тайпчек
uv run mypy src/
```

---

## 🚧 Текущая фаза

**Phase 0 — Setup** ✅ ЗАВЕРШЕНА
- [x] Структура проекта, pyproject.toml, uv, ruff, mypy, pytest
- [x] EventBus + базовые события (core/event_bus.py, core/events.py)
- [x] Config через pydantic-settings (core/config.py)
- [x] Логирование (core/logging.py)
- [x] Skeleton main.py

**Phase 1 — Voice Loop MVP** ✅ ЗАВЕРШЕНА
- [x] LLM Client (brain/llm_client.py) — OpenRouter
- [x] Orchestrator (brain/orchestrator.py) — диалог с историей
- [x] TTS (audio/tts.py) — edge-tts placeholder
- [x] Audio Player (audio/player.py) — sounddevice
- [x] STT (audio/stt.py) — faster-whisper + CUDA fallback
- [x] Wake Word (audio/wake_word.py) — Porcupine (graceful degradation без ключа)
- [x] Финальный wiring (main.py)
- Результат: полная голосовая петля работает

**Phase 2 — Кастомный голос** (следующая)
- [ ] Поднять RVC локально
- [ ] Заменить edge-tts на XTTS-v2 (audio/tts.py)
- [ ] Подключить RVC для клонирования тембра (audio/voice_cloning.py)
- [ ] Натренировать модель на сэмплах выбранного голоса
- [ ] Добавить cooldown для wake word (защита от повторных срабатываний)
- Цель: ассистент говорит нужным голосом

> Этот раздел обновляем при переходе между фазами.

---

## ⚠️ DO / DON'T

### DO
- ✅ Используй Event Bus для общения между модулями
- ✅ Все новые события — сначала в `core/events.py`
- ✅ Все секреты — через `.env`, доступ через `core/config.py`
- ✅ Type hints везде
- ✅ Покрывай тестами `core/` и `brain/` (audio/tools — опционально)
- ✅ Перед коммитом: `ruff check` + `ruff format` + `mypy` + `pytest`

### DON'T
- ❌ Прямые импорты между `audio/`, `brain/`, `tools/`
- ❌ Бизнес-логика в `main.py`
- ❌ Блокирующие вызовы в async-коде (`time.sleep`, sync HTTP и т.д.)
- ❌ Хардкод секретов и путей
- ❌ Коммит файлов: `.env`, `data/`, `*.session` (Telegram), веса моделей
- ❌ Загрузка тяжёлых моделей внутри hot-path (только при init)
- ❌ Менять архитектурные принципы без обновления этого документа

---

## 📖 Глоссарий

| Термин | Определение |
|---|---|
| **Event Bus** | Внутренняя шина событий (asyncio-based), через которую общаются модули |
| **Wake word** | Кодовое слово для активации ассистента (отзыв на имя) |
| **STT** | Speech-to-Text — распознавание речи |
| **TTS** | Text-to-Speech — синтез речи |
| **RVC** | Retrieval-based Voice Conversion — клонирование тембра голоса |
| **Tool** | Функция, которую LLM может вызвать через function calling (web search, telegram и т.д.) |
| **Orchestrator** | Главный модуль в `brain/` — принимает решения, что делать с входящим event'ом |
| **Memory** | Управление контекстом: краткосрочный (последние реплики) + долгосрочный (vector DB) |

---

## 🔗 Ссылки

- Архитектурная диаграмма: `docs/architecture.drawio`
- OpenRouter docs: https://openrouter.ai/docs
- Telethon docs: https://docs.telethon.dev
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Coqui TTS: https://github.com/coqui-ai/TTS
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
