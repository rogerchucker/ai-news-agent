# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `main.py` (CLI fetching Techmeme RSS and Hacker News via hn-sdk, LLM tagging, SQLite cache).
- Tests: `tests/` holds pytest coverage (e.g., `tests/test_main.py`).
- Config: `pyproject.toml` for dependencies (`hn-sdk`, `openai`, `pytest`); `uv.lock` is managed by `uv`.
- Cache: defaults to `~/.cache/techmeme_ai_cache.sqlite3` (overridable via `--db`).

## Build, Test, and Development Commands
- Install deps (creates `.venv`):  
  ```bash
  uv sync
  ```
- Run CLI (both sources):  
  ```bash
  uv run python main.py --limit 60 --hn-limit 40
  ```
- Heuristic-only mode (no LLM calls):  
  ```bash
  uv run python main.py --heuristic-only
  ```
- Tests:  
  ```bash
  uv run pytest
  ```

## Coding Style & Naming Conventions
- Language: Python 3.13+. Use 4-space indentation.
- Prefer small, pure functions; keep side effects explicit.
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants.
- Comments: brief, only when intent isn’t obvious. Avoid noisy “what” comments.
- Formatting/linting: none enforced; follow PEP 8 where reasonable.

## Testing Guidelines
- Framework: pytest. Add tests under `tests/` with filenames like `test_*.py`.
- Cover heuristics, parsing, and classification paths; mock network/LLM calls.
- Run `uv run pytest` before submitting; keep tests deterministic (no live API calls).

## Commit & Pull Request Guidelines
- Commits: clear, imperative summaries (e.g., “Add HN fetcher tests”).
- PRs: include what changed, why, and how to verify (commands run). Link issues if applicable.
- Keep diffs minimal and focused; include tests for new behavior or bug fixes.

## Security & Configuration Tips
- Secrets: export `OPENAI_API_KEY` in your shell; do not commit keys or tokens.
- Network calls: prefer offline tests; mock OpenAI and hn-sdk in unit tests.
- Cache hygiene: `uv run python main.py --reset-db` to clear the SQLite cache when needed.
