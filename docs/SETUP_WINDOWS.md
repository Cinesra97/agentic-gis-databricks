# Windows setup (PowerShell)

Tested toolchain versions are pinned in `uv.lock` (Python 3.11–3.13 supported; 3.12 recommended).
Commands are for **Windows PowerShell 5.1 or PowerShell 7** in the repository folder unless noted.

> These instructions were written for Windows but the development environment was Linux; see
> [TESTING.md](TESTING.md#not-verified-in-the-development-environment) for what still needs a Windows check.

## 1. Install prerequisites

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e          # uv: fast Python + dependency manager
winget install --id Ollama.Ollama -e         # Ollama: local LLM runtime (or https://ollama.com/download/windows)
```

Close and reopen PowerShell so the new commands are on `PATH`, then check:

```powershell
git --version
uv --version
ollama --version
```

No `winget`? Install uv with
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` and Ollama from the
installer on ollama.com.

## 2. Get the code

```powershell
mkdir $HOME\source -Force | Out-Null
cd $HOME\source
git clone https://github.com/Cinesra97/agentic-gis-databricks.git
cd agentic-gis-databricks
git checkout claude/funny-mendel-8l1m9p      # working branch until merged into a main branch
```

## 3. Create the Python environment and install dependencies

**Recommended (uv, reproducible from `uv.lock`):**

```powershell
uv python install 3.12
uv sync --python 3.12                        # creates .venv with exact locked versions + dev tools
```

Optional extras:

```powershell
uv sync --python 3.12 --extra duckdb-spatial-wheel   # DuckDB Spatial without internet at runtime
uv sync --python 3.12 --extra databricks-auth        # OAuth machine-to-machine (service principal)
uv sync --python 3.12 --all-extras                   # + rasterio, qdrant-client (imagery Phase 2)
```

You can run tools with `uv run <command>` without activating the environment. To activate it instead:

```powershell
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Alternative (pip + venv, pinned versions from `requirements.txt`):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
pip install pytest ruff mypy               # for running tests
```

## 4. Configure

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` is git-ignored. Every setting is documented inline in `.env.example`. Values set in the
environment override `.env`. For a single PowerShell session:

```powershell
$env:AGIS_MODE = "demo"
$env:OLLAMA_MODEL = "qwen2.5:7b"
```

Secrets: prefer typing them into the session instead of saving them in files or shell history:

```powershell
$sec = Read-Host "Databricks token" -AsSecureString
$env:DATABRICKS_TOKEN = [System.Net.NetworkCredential]::new("", $sec).Password
```

## 5. Start Ollama and choose a model

The Windows installer runs Ollama in the background (system tray) on `http://localhost:11434`.
If it is not running, start the app from the Start menu or run `ollama serve` in a separate window.

```powershell
ollama pull qwen2.5:7b      # default; Apache-2.0; ~4.7 GB download; ~6 GB RAM
ollama list                 # installed models
ollama ps                   # loaded models and whether they use the GPU
```

| Hardware | Suggested model | Notes |
|---|---|---|
| NVIDIA GPU ≥ 8 GB VRAM or 32 GB RAM | `qwen2.5:7b` (default) | Good JSON/structured-output reliability; Apache-2.0 |
| Same, alternative | `qwen3:8b` | Apache-2.0; set `AGIS_OLLAMA_THINK=false` |
| 16 GB RAM, CPU only | `qwen3:4b` | Faster; set `AGIS_OLLAMA_THINK=false`; plans may need the fallback more often |
| GPU ≥ 16 GB VRAM | `gpt-oss:20b` | Apache-2.0; stronger planning |

Check each model's license before organizational use (code licenses and weights licenses differ).
On CPU, a planning call (~3,000 prompt tokens) can take 30–120 s; raise `AGIS_LLM_TIMEOUT_S` if needed.
If the model is unreachable or returns an invalid plan, the app falls back to the deterministic planner
and says so in the reasoning panel.

## 6. Verify the installation

```powershell
uv run agentic-gis check
```

Expected: datasets ready in demo mode, `DuckDB … Spatial loaded` (or a GeoPandas fallback message),
`Ollama … ready with model qwen2.5:7b`, Databricks `SKIP not configured`.

Run a real natural-language → spatial-analysis workflow with the local model, headless:

```powershell
uv run agentic-gis ask --provider ollama -v "Which boroughs have the most illegal dumping complaints?"
uv run agentic-gis ask --provider ollama -v --approve "Find repeated illegal dumping complaints in Brooklyn" "Which of those are near industrial properties?"
```

The output states whether the plan came from the LLM (`plan source: llm`) or the fallback
(`rules_fallback`). Generated SQL is printed and only runs with `--approve`.

## 7. Run the app

```powershell
uv run streamlit run app.py
```

Open http://localhost:8501. In the sidebar choose the data source and reasoning engine; type a question
or click an example. Approve or reject SQL in the chat panel when asked. Exports (CSV, GeoJSON, report,
layer JSON, ZIP) are on the **Export** tab.

Use another port with `uv run streamlit run app.py --server.port 8502`.

## 8. Connect Databricks (optional)

1. Ask your workspace admin for a **read-only** identity (see [DATABRICKS.md](DATABRICKS.md#read-only-access)).
2. In the Databricks UI, open your SQL warehouse → **Connection details** for *Server hostname* and *HTTP path*.
3. Set the variables in `.env` (or the session) and test:

```powershell
$env:DATABRICKS_SERVER_HOSTNAME = "adb-1234567890123456.7.azuredatabricks.net"
$env:DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/abcdef1234567890"
$env:DATABRICKS_CATALOG = "dsny_sandbox"
$env:DATABRICKS_SCHEMA = "egpmo_sandbox"
$sec = Read-Host "Databricks token" -AsSecureString
$env:DATABRICKS_TOKEN = [System.Net.NetworkCredential]::new("", $sec).Password

uv run agentic-gis check
uv run agentic-gis databricks-tables
```

Browser-based OAuth instead of a token: `$env:DATABRICKS_AUTH_TYPE = "oauth-u2m"` (a browser window opens
on first query).

4. Register the tables you want the agent to use in `config/datasets.local.yaml` (git-ignored) — see
   [DATABRICKS.md](DATABRICKS.md#register-unity-catalog-tables) — then choose **Databricks** in the sidebar.

## 9. Public NYC Open Data (optional)

```powershell
uv run agentic-gis ingest --list
uv run agentic-gis ingest nyc_311_sanitation --since 2024-01-01 --limit 100000
uv run agentic-gis ingest nyc_nta_2020
uv run agentic-gis ingest nyc_pluto_industrial
```

Then choose **Public NYC data** in the sidebar (click **New conversation** to reload). Dataset ids marked
“verify” in `--list` were not verifiable from the development environment; confirm them on
data.cityofnewyork.us. A free `SOCRATA_APP_TOKEN` raises rate limits.

## 10. Run the tests

```powershell
uv run pytest                                   # offline suite
uv run ruff check . ; uv run mypy agentic_gis app.py

# Live checks against your own services (opt-in):
$env:OLLAMA_LIVE_TESTS = "1"; uv run pytest -m ollama -v; Remove-Item Env:OLLAMA_LIVE_TESTS
$env:DATABRICKS_LIVE_TESTS = "1"; uv run pytest -m databricks -v; Remove-Item Env:DATABRICKS_LIVE_TESTS
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv` / `ollama` not recognized | Reopen PowerShell. For uv, check `$env:USERPROFILE\.local\bin` is on `PATH`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and use `uv run`. |
| SSL / certificate errors behind a corporate proxy during `uv sync` | `$env:UV_SYSTEM_CERTS = "1"` (older uv: `$env:UV_NATIVE_TLS = "1"`), and set `$env:HTTPS_PROXY` if your network requires it. |
| “Ollama is not reachable at http://localhost:11434” | Start Ollama from the Start menu (tray icon) or run `ollama serve`. Check `Get-Process ollama`. If another app uses port 11434, set `OLLAMA_BASE_URL`. |
| “Model … is not installed” | `ollama pull <model>`; the name must match `ollama list` (e.g. `qwen2.5:7b`). |
| Answers say “LLM unavailable or invalid; used deterministic planner” | Check the Reasoning tab for the reason. Try `qwen2.5:7b` or a larger model; raise `AGIS_LLM_TIMEOUT_S`. |
| Very slow planning on CPU | Use a smaller model (`qwen3:4b`, `AGIS_OLLAMA_THINK=false`), a GPU, or the deterministic planner. Lowering `AGIS_LLM_NUM_CTX` below 6144 can truncate the prompt. |
| “DuckDB Spatial unavailable … using GeoPandas” | Fine for most questions (spatial work runs in GeoPandas). For Spatial SQL offline: `uv sync --extra duckdb-spatial-wheel`. |
| Map area blank or gray | The map JS is served locally; gray tiles mean the basemap server is blocked. Pick **No basemap** or set `AGIS_CUSTOM_BASEMAP_URL` to an approved tile server. |
| `Port 8501 is already in use` | `uv run streamlit run app.py --server.port 8502` |
| Databricks “authentication failed” | Token expired or wrong workspace; check `DATABRICKS_SERVER_HOSTNAME` has no `https://` issues (it is stripped automatically) and recreate the token. |
| Databricks “does not have permission” | The identity lacks `USE CATALOG` / `USE SCHEMA` / `SELECT`; see [DATABRICKS.md](DATABRICKS.md#read-only-access). |
| First Databricks query is slow | The SQL warehouse is starting; wait or keep it running. Queries time out after `AGIS_QUERY_TIMEOUT_S`. |
| `ingest` fails with “Dataset not found” | The NYC Open Data id changed; update `socrata.dataset_id` in `config/datasets.yaml`. |
| `ImportError: DLL load failed` for geopandas/pyogrio | Avoid mixing Conda and pip environments; recreate `.venv` with `uv sync`. |
