# Third-party licenses, data terms and usage policies

This project is licensed under **Apache-2.0**. It depends on permissively licensed libraries; the
exceptions and non-code terms that need attention are listed first.

## Items that need attention

| Item | License / terms | Note |
|---|---|---|
| GEOS (bundled in Shapely, pyogrio and DuckDB Spatial binaries) | LGPL-2.1 | Weak copyleft, dynamically linked in the wheels; distributing modified GEOS triggers LGPL obligations. |
| GDAL / PROJ (bundled in pyogrio, pyproj, rasterio wheels) | MIT/X-style | Wheels also bundle other libraries (e.g. libcurl, sqlite); see each wheel's license files. |
| certifi | MPL-2.0 | Weak copyleft (file-level); used unmodified. |
| `duckdb-extension-spatial` (optional extra) | MIT (official extension binary repackaged by a third party) | Review supply-chain policy; DuckDB verifies the extension signature on load. |
| **LLM model weights** (pulled through Ollama) | Per model | `qwen2.5:7b`, `qwen3:*`, `gpt-oss:20b` are Apache-2.0; Llama models use the Llama Community License; some Qwen2.5 sizes (e.g. 3B) use a research license. Ollama itself is MIT. Check the model card before organizational use. |
| **Basemap tiles** | Provider terms | *CARTO* basemaps: attribution required; free tier intended for limited/non-commercial use — check CARTO's terms or use another provider for production. *OpenStreetMap* tiles: OSMF tile usage policy (light use, attribution, no bulk downloading). *NYC Basemap (NYC OTI)*: public NYC service; attribute NYC. Prefer an organization tile server for production via `AGIS_CUSTOM_BASEMAP_URL`. |
| NYC public data (boundaries, 311, PLUTO, NTAs) | NYC Open Data Terms of Use | Free to use; no warranty; attribute the publishing agency. |
| Clay Foundation Model (Phase 2, not bundled) | Code Apache-2.0; weights per model card | Verify before use. |
| Qdrant (optional) | Apache-2.0 | pgvector alternative: PostgreSQL License. |

## Python dependencies (runtime)

| Package | Version (uv.lock) | License |
|---|---|---|
| streamlit | 1.66.1 | Apache-2.0 |
| streamlit-folium | 0.27.4 | MIT |
| folium / branca | 0.20.0 / 0.8.2 | MIT |
| pydeck | 0.9.3 | Apache-2.0 |
| langgraph / langgraph-checkpoint / langchain-core | 1.2.16 / 4.3.0 / 1.7.0 | MIT |
| pydantic / pydantic-settings | 2.14.1 / 2.16.1 | MIT |
| PyYAML | 6.0.3 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| duckdb | 1.5.5 | MIT |
| geopandas / shapely | 1.2.0 / 2.2.0 | BSD-3-Clause |
| pyproj / pyogrio | 3.8.0 / 0.13.0 | MIT |
| pyarrow | 26.0.0 | Apache-2.0 |
| pandas / numpy | 3.1.0 / 2.5.4 | BSD-3-Clause (numpy: BSD-3-Clause and bundled permissive licenses) |
| sqlglot | 30.24.0 | MIT |
| databricks-sql-connector / thrift | 4.7.0 / 0.25.0 | Apache-2.0 |
| databricks-sdk (extra) | 0.156.0 | Apache-2.0 |
| rasterio (extra) | 1.5.2 | BSD-3-Clause |
| qdrant-client (extra) | 1.20.0 | Apache-2.0 |

Development: pytest (MIT), ruff (MIT), mypy (MIT). Full list with transitive packages: `uv.lock` /
`requirements.txt`; license files ship inside each installed package.

## Vendored JavaScript (`static/vendor/`)

| Library | Version | License |
|---|---|---|
| Leaflet | 1.9.4 | BSD-2-Clause |
| Leaflet.markercluster | 1.5.3 | MIT |
| Leaflet.heat | 0.2.0 | BSD-2-Clause |
| jQuery | 3.7.1 | MIT |

Each directory contains the upstream `LICENSE` and `VERSION`; `scripts/vendor_map_assets.py` re-downloads
them from the npm registry with integrity verification. deck.gl / MapLibre are bundled by Streamlit
(MIT / BSD-3-Clause).

## Not required

The core workflow does not use or require Databricks Genie, CARTO (as a product), LGND, Mapbox, or any paid
LLM API.
