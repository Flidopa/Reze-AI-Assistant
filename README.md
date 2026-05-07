# AI Assistant

Personal voice AI assistant with custom voice cloning, wake-word activation, and Telegram integration.
<a href="https://ibb.co/yF5fMVwc"><img src="https://i.ibb.co/tpcmN37T/rezeai.png" alt="rezeai" border="0"></a>

## Установка

```bash
git clone <repo>
cd ai-assistant
uv sync
cp .env.example .env  # затем заполни реальными ключами
```

## Запуск

```bash
uv run python -m assistant.main
```

## Разработка

```bash
uv run ruff check --fix    # линтер
uv run ruff format         # форматтер
uv run mypy src/           # типы
uv run pytest              # тесты
```

Архитектура и конвенции — см. [`CLAUDE.md`](./CLAUDE.md).
