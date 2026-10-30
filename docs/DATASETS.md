# Datasets and the registry

`config/datasets.yaml` is the allowlist and the metadata catalog. Administrators register datasets there
(shared, committed) or in `config/datasets.local.yaml` (git-ignored, for workspace-specific or sensitive
entries; merged over the base file). `${VAR:-default}` placeholders are filled from the environment and may
only contain identifier characters.

## Dataset entry reference

```yaml
- id: demo_311_sanitation             # lowercase identifier, unique
  name: "SYNTHETIC 311-style sanitation complaints (demo)"
  short_name: 311 sanitation complaints   # used in result titles
  description: What the dataset contains; shown to users and (policy permitting) the LLM
  backend: local                      # local (DuckDB) | databricks
  modes: [demo]                       # demo | local | databricks — where it is visible
  enabled: true                       # false = never visible or queryable
  synthetic: true                     # labels every result as synthetic
  sensitivity: public                 # public | internal | confidential | restricted
  role: events                        # events | infrastructure | parcels | streets | boundary | reference | other
  geography_role: null                # for boundaries: borough | neighborhood | community_district | zip
  name_column: null                   # label column for boundaries / streets
  tags: [311, complaints, dumping]    # discovery vocabulary
  synonyms: {abandoned car: Derelict Vehicles}   # user words -> category values
  source:
    type: file                        # file | socrata | databricks_table
    path: sample_data/synthetic_311_sanitation.parquet
    format: geoparquet                # geoparquet | parquet | geojson | csv
    # table: catalog.schema.table     # for databricks_table
    # socrata: {dataset_id: erm2-nwe9, select: ..., where: ..., limit: 150000}
  table: null                         # DuckDB table name (defaults to id)
  spatial:
    geometry_column: geometry
    geometry_type: point              # point | line | polygon | mixed
    format: wkb                       # wkb | wkt | geojson | latlon | native
    crs: "EPSG:4326"
    lat_column: latitude
    lon_column: longitude
  temporal: {column: created_date, granularity: timestamp}
  id_column: unique_key
  category_column: complaint_type
  join_keys: {borough: borough, cd_code: cd_code}
  location_columns: [incident_address, street_name, incident_zip, community_board, borough]  # most specific first
  columns:
    - {name: complaint_type, type: string, description: ..., known_values: [Illegal Dumping, ...]}
    - {name: caller_phone, type: string, sensitive: true}   # hidden from LLM, blocked in SQL
  schema_discovery: false             # true = read columns from the warehouse at run time
  verification_source: false          # true only for authoritative verified outcomes
  provenance:
    publisher: ...
    source_url: ...
    license: ...
    retrieved: ...
    notes: ...
    citation: ...
  max_rows: null                      # per-dataset cap below AGIS_MAX_RESULT_ROWS
```

`geographies:` maps roles to boundary datasets per mode, e.g. `neighborhood` → NTAs when ingested in local
mode, otherwise community districts.

## Bundled data (demo mode)

See [sample_data/README.md](../sample_data/README.md). Boundaries are real (simplified) public data;
complaints, catch basins, street corridors and industrial lots are **synthetic**.

## Public NYC Open Data (local mode)

| id | NYC Open Data | Contents |
|---|---|---|
| `nyc_311_sanitation` | `erm2-nwe9` (311 Service Requests) | DSNY requests since 2023 (configurable) |
| `nyc_nta_2020` | `9nt8-h7nd` (verify) | 2020 Neighborhood Tabulation Areas |
| `nyc_pluto_industrial` | `64uk-42ks` (verify) | PLUTO lots with land use 06 |

```powershell
uv run agentic-gis ingest --list
uv run agentic-gis ingest nyc_311_sanitation --since 2024-01-01 --limit 100000
```

Ingestion writes GeoParquet to `data/ingested/` (git-ignored) plus a `.provenance.json` sidecar (portal id,
filter, order, row limit, rows fetched, whether the limit was reached, stored date range, retrieval time).
Ingestion is an operator action; the agent never calls web APIs on its own. Add more Socrata datasets by
adding entries with `source.type: socrata`.

**Stored coverage is not the configured window.** With `order: created_date DESC` and `limit: 150000`, a
download keeps only the most recent 150,000 matching records; with the full 311 dataset that can be a few
months, not “since 2023”. At startup the app measures each local dataset's row count and earliest/latest
date from the loaded data (no extra file scan) and combines it with the sidecar:

* `uv run agentic-gis check` prints the stored range per dataset and flags truncated downloads;
* the sidebar *Data catalog* shows stored rows and dates;
* questions about a period outside the stored range get an “insufficient coverage” answer instead of an
  empty result; partial overlaps and truncated downloads are disclosed in the answer.

**Getting a historical period.** `--since` alone does not reach older records: the 311 download is ordered
newest-first and stops at the row limit, so it keeps the most recent records again. Download a **bounded
window** instead:

```powershell
uv run agentic-gis ingest nyc_311_sanitation --since 2020-01-01 --until 2023-12-31 --limit 1500000
```

The command first asks the portal how many records match the window and prints that count next to the limit;
the sidecar records it, and the app reports “only N of M records were downloaded” if the limit cut the window.
Choose a `--limit` at least as large as the count (memory: DuckDB holds the table in RAM and GeoPandas holds
each selection again, so check memory use before ingesting millions of rows), or split a long period into
smaller windows. Pages are
fetched with offset paging ordered by the configured `order` plus the row id (`:id`), so records with equal
timestamps are neither skipped nor repeated.

Ingesting **replaces** the stored copy. To keep two periods side by side, register a second dataset in
`config/datasets.local.yaml` (copy the `nyc_311_sanitation` entry, give it a new `id`, `table` and
`source.path`) and ingest the window into it. Older sidecars without the new fields are still read; the
configured limit is then assumed.

**Location columns.** `location_columns` lists the columns that say where a record is, most specific first.
They are always kept in query results, even when a plan asks for a narrower column list, so clusters and
other results can be described by address or street. Without the field they are derived from column names
(address, street, ZIP, community board, neighborhood, borough). Sensitive columns are never included.

## Adding a local file dataset

Put a GeoParquet/GeoJSON/CSV file somewhere in the project (e.g. `data/ingested/`), then:

```yaml
- id: dsny_litter_baskets
  name: DSNY litter baskets
  description: Litter basket locations
  backend: local
  modes: [local]
  role: infrastructure
  tags: [litter basket, baskets, trash can]
  source: {type: file, path: data/ingested/litter_baskets.geojson, format: geojson}
  spatial: {geometry_column: geometry, geometry_type: point, format: wkb, crs: "EPSG:4326"}
```

Files are loaded into DuckDB when the app (re)starts. Reprojection to WGS84 happens on load.

## Synthetic data rules

* `synthetic: true` on every generated dataset; records carry `is_synthetic`.
* Titles get “(synthetic)”, maps show a “SYNTHETIC DEMO DATA” badge, answers start with “Synthetic demo
  data” and include a limitation stating results are not findings about NYC.
* Never mix synthetic and real records in one dataset.
