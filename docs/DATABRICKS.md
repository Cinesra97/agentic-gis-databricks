# Databricks integration

Databricks is the primary analytical warehouse; the app also runs without it. Access uses the official
[`databricks-sql-connector`](https://pypi.org/project/databricks-sql-connector/) against a SQL warehouse,
reading Unity Catalog tables that an administrator has registered.

## Configuration

| Variable | Purpose |
|---|---|
| `DATABRICKS_SERVER_HOSTNAME` | Workspace host, e.g. `adb-123.7.azuredatabricks.net` (a leading `https://` is stripped) |
| `DATABRICKS_HTTP_PATH` | Warehouse HTTP path, e.g. `/sql/1.0/warehouses/abc123` (or `DATABRICKS_WAREHOUSE_ID`) |
| `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA` | Defaults for unqualified names and table discovery; also fill `${DATABRICKS_CATALOG:-…}` placeholders in the registry |
| `DATABRICKS_AUTH_TYPE` | `pat` (default), `oauth-u2m` (browser login), `oauth-m2m` (service principal, needs `--extra databricks-auth`) |
| `DATABRICKS_TOKEN` | Personal access token for `pat` |
| `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` | Service principal for `oauth-m2m` |

Test the connection: `uv run agentic-gis check` or **Databricks connection → Test** in the sidebar.
Discover tables in the default schema: `uv run agentic-gis databricks-tables` or the sidebar button.
Discovery lists tables; it does not make them queryable.

## Read-only access

Create (or ask for) a dedicated principal with read-only grants. Example for a service principal or group
(run by an administrator; see `notebooks/databricks_readonly_grants.sql`):

```sql
GRANT USE CATALOG ON CATALOG dsny_sandbox TO `agentic-gis-readers`;
GRANT USE SCHEMA ON SCHEMA dsny_sandbox.egpmo_sandbox TO `agentic-gis-readers`;
GRANT SELECT ON TABLE dsny_sandbox.egpmo_sandbox.catch_basins_exploded_final TO `agentic-gis-readers`;
GRANT SELECT ON TABLE dsny_sandbox.egpmo_sandbox.catch_basin_records_final TO `agentic-gis-readers`;
-- plus CAN USE on the SQL warehouse (workspace UI / permissions API)
```

Do not grant `MODIFY`, `CREATE`, or `ALL PRIVILEGES`. Row filters and column masks in Unity Catalog
continue to apply — the app never bypasses them.

## Register Unity Catalog tables

The DSNY example tables in `config/datasets.yaml` are **disabled** and have no assumed schema. To use a
table:

1. Inspect it (as a human) — e.g. `DESCRIBE TABLE` in Databricks SQL, or temporarily register it with
   `schema_discovery: true` and use `get_schema` from the agent.
2. Add an entry in `config/datasets.local.yaml` (git-ignored; merged over `datasets.yaml`):

```yaml
datasets:
  - id: dsny_catch_basins_exploded
    enabled: true                      # merges into the disabled example entry
    sensitivity: internal
    category_column: basin_status      # only if such a column exists
    spatial:
      geometry_column: geom_wkt        # the real column name
      format: wkt                      # wkt | wkb | geojson | latlon | native
      crs: "EPSG:2263"                 # CRS of the stored geometry
      geometry_type: point
    temporal: {column: last_cleaned_date}
    columns:                           # optional; omit to discover live
      - {name: basin_id, type: string, description: Catch basin identifier}
      - {name: borough, type: string, description: Borough name}
```

For a brand-new table use a full entry (see `docs/DATASETS.md`), with
`source: {type: databricks_table, table: catalog.schema.table}` and `backend: databricks`,
`modes: [databricks]`. With `AGIS_ADMIN_MODE=true` the sidebar has a registration form that writes the
same file.

3. Restart the app (or click **New conversation**) and pick **Databricks** as the data source.

### Geometry handling

| `spatial.format` | Expected column content | What the app does |
|---|---|---|
| `latlon` | numeric `lat_column` / `lon_column` | builds points client-side |
| `wkt` / `geojson` | text | parses with Shapely |
| `wkb` | binary | parses with Shapely |
| `native` | Databricks `GEOMETRY`/`GEOGRAPHY` | template SQL selects `ST_AsWKB(col)` (requires a runtime with ST functions) |

Coordinates are transformed from `spatial.crs` to WGS84 for display and to EPSG:2263 for distance work.

## How queries run

1. The planner chooses `filter_dataset` (preferred) or, rarely, `query_databricks` with raw SQL.
2. Template SQL is built with SQLGlot for the Databricks dialect, with **native named parameters**
   (`:p0`) and fully qualified table names; filters (category, borough, dates) are pushed down so only
   needed rows are transferred.
3. The guard validates it; the UI shows it for approval (every Databricks query requires approval under the
   default policy).
4. Rows are fetched as Arrow (bounded by `AGIS_MAX_RESULT_ROWS`), converted to GeoDataFrames and analyzed
   with GeoPandas; boundaries (boroughs, neighborhoods, ZIPs) are local reference data, so cross-source
   joins happen in the app.
5. Errors (permissions, missing tables, timeouts) are reported in plain language; secrets are redacted.

## Deploying as a Databricks App

`app.yaml` is a starting point (**not tested in this repository**):

1. Create a Databricks App from this repository (Git folder) or upload the files.
2. Add a **SQL warehouse** resource with key `sql-warehouse` and *Can use* permission.
3. Grant the app's service principal the read-only Unity Catalog privileges above.
4. Commit a `config/datasets.local.yaml` equivalent through your normal change process, or register datasets
   in `config/datasets.yaml` on a deployment branch (never with credentials).
5. Databricks Apps injects `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`; the
   manifest sets `DATABRICKS_AUTH_TYPE=oauth-m2m` and maps the warehouse id.
6. Dependencies install from `requirements.txt` (exported from `uv.lock` with the `databricks-auth` extra).

Ollama cannot run inside Databricks Apps; the manifest uses the deterministic planner. An organization-
approved model endpoint can be configured with `AGIS_LLM_PROVIDER=openai_compatible` and
`AGIS_ALLOW_EXTERNAL_LLM=true` after a data-handling review.

## Limits and notes

* One connection per user session; all users of one deployment share its identity (see SECURITY.md).
* The connector fetches results through the warehouse; very large results are truncated at the row limit
  with a warning — aggregate in SQL or narrow filters.
* Databricks spatial SQL (`ST_*`, H3) can be used in approved `query_databricks` SQL where your runtime
  supports it; built-in tools do not depend on it.
