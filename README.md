AI News CLI
============

Project Title and Description
-----------------------------
AI News CLI surfaces AI-related stories from Techmeme (RSS) and Hacker News (via hn-sdk). It applies heuristic filters plus optional LLM classification and topic tagging, clusters results, and uses SQLite to avoid showing repeats across runs.

Features
--------
- Fetches Techmeme RSS and Hacker News front-page items.
- Heuristic keyword pre-filter with optional LLM-based tagging and takeaways.
- Topic clustering with confidence scores.
- SQLite cache to skip previously seen links.
- Source toggles and recency filtering.

Installation
------------
```bash
uv sync
export OPENAI_API_KEY="sk-..."
```

Usage
-----
Default (both sources):
```bash
uv run python main.py --limit 60 --hn-limit 40
```
Heuristics only (no LLM calls):
```bash
uv run python main.py --heuristic-only
```
Reset cache:
```bash
uv run python main.py --reset-db
```
Skip a source:
```bash
uv run python main.py --no-hn
uv run python main.py --no-techmeme
```
Common flags:
- `--min-confidence`: minimum confidence for inclusion (default 0.55)
- `--borderline-low` / `--borderline-high`: heuristic thresholds controlling LLM usage
- `--since-minutes`: filter items newer than N minutes
- `--db`: override cache location (default `~/.cache/techmeme_ai_cache.sqlite3`)

Contributing
------------
- Fork the repository and create a feature branch for your change.
- Keep changes focused and add tests for new behavior.
- Run `uv run pytest` before submitting a pull request.
- See `AGENTS.md` for detailed repository guidelines and contributor expectations.

License
-------
MIT
