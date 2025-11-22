# Repository Guidelines

## Project Structure & Module Organization
- `kg/` covers ontology/extraction/alignment plus Neo4j import helpers; keep utilities in their subfolders and re-export shared APIs via `kg/__init__.py` when required.
- `retrieval/dense/` bundles `labse_encoder.py`, `build_faiss.py`, and `dense_search.py`; add new retrieval modes as sibling modules and reuse shared config/log utilities.
- `app/ui/streamlit_app.py` houses the Streamlit demo and UI helpers or assets—import backend services.
- `scripts/` keeps numbered pipelines (`01_clean_corpus.py`, `06_index_dense.py`); store knobs in `config.py`, logging defaults in `logger.py`, and stash corpora/indexes under `data/` + `artifacts/`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` — bootstrap the Python 3.10+ environment.
- `pip install -r requirements.txt` — install runtime dependencies.
- `python scripts/01_clean_corpus.py --input data/raw --output data/cleaned --lang fr` — turn raw corpora into language-tagged JSONL.
- `python scripts/06_index_dense.py --corpus-dir data/cleaned --output artifacts/faiss_fr --langs fr zh en` — encode cleaned corpora and write FAISS assets.
- `python retrieval/dense/build_faiss.py --corpus data/cleaned/corpus_fr_cleaned.jsonl --output artifacts/faiss_labse --index-type IVF` — manually rebuild indexes for ad-hoc corpora.
- `python retrieval/dense/dense_search.py --index artifacts/faiss_labse --corpus data/cleaned/corpus_fr_cleaned.jsonl --query "法语语法" --top-k 5` — sanity-check dense retrieval or pass `--queries-file`.
- `streamlit run app/ui/streamlit_app.py` — preview the UI locally.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and typed public APIs; keep functions tight and purposeful.
- Use `snake_case` modules/files, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants/env keys (e.g., `class DenseSearcher`).
- Inject dependencies through constructors, route logs via `logger.get_logger`, and document CLI entry points with `argparse` help near `__main__`.

## Testing Guidelines
- Adopt `pytest` + `pytest-cov` (install separately) and mirror the source tree under `tests/`.
- Name cases `test_<feature>_<behavior>` and keep tiny fixtures in `tests/fixtures/` so retrieval math stays deterministic.
- Run `pytest --maxfail=1 --disable-warnings -q`; for coverage or CI, extend to `pytest --cov=retrieval --cov=kg --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat: add kg alignment`, `fix: guard faiss loader`, `chore: update requirements`) so history stays searchable.
- Scope commits narrowly (code + docs + tests), mention touched modules plus validation commands, and exclude generated artifacts.
- PRs should cover motivation, data/config effects, validation commands, linked issues, and UI screenshots or retrieval metrics when behavior changes.

## Security & Configuration Tips
- Keep Neo4j credentials, FAISS paths, and API keys solely in `.env` managed by `pydantic-settings`; never commit secrets or raw corpora.
- Reference values via `config.py` or environment variables, and update this guide plus `README.md`/`.env.example` whenever knobs change.
