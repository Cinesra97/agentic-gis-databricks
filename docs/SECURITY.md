# Security and governance

This application may eventually run in a government environment. The controls below are implemented
in code and covered by tests unless marked otherwise. They are defense in depth: **database permissions
remain the primary control** — always connect with a read-only identity.

## Controls

| Requirement | Implementation | Code / tests |
|---|---|---|
| Read-only database access | SQL guard accepts exactly one query statement (SELECT / set operation / CTE); any DDL, DML, `COPY`, `ATTACH`, `INSTALL`, `PRAGMA`, `SET`, `USE`, `GRANT`, `SELECT … INTO`, etc. anywhere in the tree is rejected. Databricks: use an identity with only `USE CATALOG`/`USE SCHEMA`/`SELECT`. DuckDB: in-memory, external access disabled and configuration locked after load. | `security/sql_guard.py`, `connectors/duckdb_engine.py`; `tests/test_sql_guard.py` |
| AST validation (not regex) | SQLGlot parse with the target dialect; table references resolved and checked against the registry allowlist (CTE names excepted); table functions (`read_csv(...)`, `glob(...)`, `IDENTIFIER(...)`) rejected; denied functions (file readers, `getenv`, DuckDB system catalogs, Databricks `secret()`, `ai_*`, `http_*`, `reflect`, `java_method`, `read_files`, …); qualified UDF calls rejected; path-like table names rejected; Databricks names rewritten to fully qualified identifiers. | same |
| Dataset allowlist | Only `enabled` datasets visible in the active mode can be listed, described, planned or queried. Example DSNY tables are disabled. | `config/registry.py` |
| Query limits and timeouts | Guard enforces `LIMIT ≤ AGIS_MAX_RESULT_ROWS`; DuckDB queries interrupted after `AGIS_QUERY_TIMEOUT_S`; Databricks `STATEMENT_TIMEOUT` session setting plus client-side cancel. | `connectors/*` |
| Approval of newly generated SQL | `AGIS_SQL_APPROVAL=generated` (default): LLM-written SQL and all Databricks queries pause for user approval (LangGraph interrupt) with the exact SQL and parameters; fingerprints approved once per session. `always` approves every query; `never` is for automated tests only. | `agent/graph.py`; `tests/test_graph.py`, `tests/test_databricks.py` |
| Parameterized SQL | Template queries bind user values as parameters (`$p0` DuckDB, `:p0` Databricks native parameters); identifiers come from the registry/schema only. | `agent/tools.py::render_filter_sql` |
| No arbitrary code execution | The LLM can only choose registered tools; arguments are Pydantic-validated. No tool runs Python, shell, file or network operations chosen by the model. | `agent/tools.py`, `agent/planner.py::validate_plan` |
| No unrestricted filesystem access | Local data paths come from the registry; DuckDB file access is disabled after load; exports are browser downloads. | |
| Sensitive data handling | Columns marked `sensitive: true` are hidden from prompts and blocked in SQL (`SELECT *` refused on such tables). Dataset `sensitivity` + `AGIS_LLM_MAX_SENSITIVITY` decide whether schema/result digests may enter prompts; non-local LLM endpoints are blocked unless `AGIS_ALLOW_EXTERNAL_LLM=true` and then only see public datasets. | `security/policy.py` |
| Prompt-injection protection | Untrusted text (records, column comments, previous results, errors) is truncated, control-stripped, scanned for instruction-like phrases and wrapped in `<untrusted_data>` tags the system prompt declares as data. Even if injected text survives, the model can only output a validated plan, SQL is guarded and needs approval. | `security/sanitize.py`; `tests/test_llm.py` |
| Secure credentials | Secrets only from environment / `.env` (git-ignored); `SecretStr` (never in `repr`); redacted from audit logs and error messages; `.env.example` contains no secrets. OAuth U2M/M2M supported. | `config/settings.py`, `security/audit.py` |
| Audit logging | JSONL log (`AGIS_AUDIT_LOG_PATH`) of questions, plans, SQL validation/approval/rejection/execution (fingerprints, row counts, timings), tool calls, LLM calls, answers and guardrail adjustments. No result rows; token patterns and configured secrets redacted. | `security/audit.py`; `tests/test_graph.py` |
| Provenance | Every answer lists datasets used (publisher, license, synthetic flag, sensitivity), methodology notes, executed SQL and the plan. Synthetic data is labeled in titles, legends, map badge, headline and limitations. | `agent/narrative.py`, `ui/results.py` |
| Graceful permission failures | Databricks permission/auth/not-found errors are translated into actionable messages (secrets redacted); failures skip dependent steps and are reported. | `connectors/databricks_sql.py` |
| Claims policy | Answers distinguish observation / inference / verified. “Verified” requires a dataset marked `verification_source: true`; otherwise relabeled. Statements that a place *is* an illegal dumping site or that violations were confirmed are rewritten. Imagery results are always “candidate”. | `agent/narrative.py`, `imagery/search.py` |

## Known gaps and deployment requirements

* **No built-in authentication.** Streamlit has no login. Deploy behind organizational SSO (e.g. Databricks
  Apps, an authenticating reverse proxy) — never expose it directly to the internet.
* **Shared Databricks identity.** All users of one deployment share the configured token/service principal,
  so Unity Catalog permissions are per deployment, not per user. A future enhancement is on-behalf-of-user
  authentication (Databricks Apps forwards the user's token) so each query runs with the viewer's grants.
* **Guard scope.** The guard is conservative but not a substitute for least-privilege grants. New database
  functions with side effects could appear in future engine versions; keep the deny list reviewed.
* **Prompt-injection detection is heuristic.** The architectural controls (validated plans, guarded SQL,
  approval) are what make injection low-impact.
* **Audit log is a local file.** For production, ship it to a central, append-only store and protect access.
* **Basemap requests** go from the user's browser to the selected tile provider (they reveal the map area
  being viewed). Use an internal tile server via `AGIS_CUSTOM_BASEMAP_URL` or “No basemap” if that matters.
* **Optional geocoder** `AGIS_GEOCODER=nyc_geosearch` sends the place text (only) to NYC Planning Labs
  GeoSearch; default is local reference data.
* **Third-party DuckDB Spatial wheel** (`duckdb-spatial-wheel` extra) repackages the official extension
  binary; DuckDB verifies the extension signature on load, but review your supply-chain policy before use.

## Reporting

Treat findings about these controls like any other security issue in your organization's process; do not
open public issues containing credentials, internal hostnames or data.
