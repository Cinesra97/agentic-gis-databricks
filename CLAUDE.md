# CLAUDE.md — working agreement for AI coding sessions

Read README.md, HANDOFF.md and the relevant docs/ page before changing code.

## Commands

```bash
uv sync --all-extras                 # environment from uv.lock (Python 3.11–3.13)
uv run pytest                        # offline suite; must stay green
uv run ruff format . && uv run ruff check . && uv run mypy agentic_gis app.py
uv run streamlit run app.py          # UI (AGIS_LLM_PROVIDER=demo for no LLM)
uv run agentic-gis ask --provider demo -v "question"   # headless agent run
uv run python scripts/build_sample_data.py             # regenerate sample data (deterministic)
uv run agentic-gis layer-schema > docs/map_layer_schema.json
uv export --format requirements-txt --no-emit-project --no-dev --no-hashes --extra databricks-auth -o requirements.txt
```

After changing dependencies: `uv lock`, re-export `requirements.txt`, update docs/LICENSES.md.

## Layout

All code lives in the `agentic_gis` package (agent/, connectors/, spatial/, security/, ui/, imagery/,
config/). `app.py` is the Streamlit entry point; `config/datasets.yaml` is the dataset allowlist.

## Rules

* Never commit secrets, `.env`, `config/datasets.local.yaml`, `data/ingested/`, `data/cache/`, `logs/`,
  audit logs, or real agency data. Use `.env.example` for new settings (and document them).
* Every SQL path goes through `SQLGuard`; tools receive data by reference only; no tool may execute
  model-provided code, shell commands, file paths or URLs.
* New SQL-bearing tools need a `sql_renderer` so SQL is reviewed/approved before execution.
* Keep synthetic data labeled (`synthetic: true`, `is_synthetic`, titles, answers). Never describe
  complaint locations as confirmed violations; imagery results are candidates.
* Distances/areas in a projected CRS (`settings.projected_crs`, EPSG:2263 default).
* Keep graph state JSON-serializable; DataFrames go in `ResultStore`.
* UI color changes: keep roles in `ui/layers.py` and re-validate palettes (light and dark).
* Update README/HANDOFF/docs when behavior, setup or limitations change. Report untested components
  honestly in docs/TESTING.md.
* Git: commit logical units with descriptive messages; push to the working branch; do not rewrite shared
  history.
