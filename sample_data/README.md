# Bundled sample data

These files let the app run fully offline (demo mode). They are rebuilt deterministically by
`scripts/build_sample_data.py` (fixed seed `20241023`); file checksums are in `MANIFEST.json`.

## Real public reference boundaries (simplified)

| File | Source | Notes |
|---|---|---|
| `reference/nyc_boroughs.parquet` | NYC Department of City Planning, Borough Boundaries (clipped to shoreline), via the `dwillis/nyc-maps` GeoJSON conversion | 5 polygons, simplified with an 80 ft coverage-preserving tolerance |
| `reference/nyc_community_districts.parquet` | NYC DCP Community Districts, via `dwillis/nyc-maps` | 59 districts + 12 joint interest areas (parks/airports); neighborhood labels follow NYC DOHMH Community Health Profile naming |
| `reference/nyc_modzcta.parquet` | NYC DOHMH Modified ZIP Code Tabulation Areas, via `nychealth/coronavirus-data` | 177 areas |

NYC government data is published under the NYC Open Data Terms of Use. Simplification means
boundaries are approximate at street scale (~25 m).

## SYNTHETIC demonstration data — not NYC records

| File | Rows | What it imitates |
|---|---|---|
| `synthetic_311_sanitation.parquet` | 8,917 | NYC 311 DSNY service requests, 2022–2024 (illegal dumping, derelict vehicles, street sweeping, missed collection, dirty conditions, litter baskets) |
| `synthetic_catch_basins.parquet` | 5,003 | Catch basins along corridors and scattered citywide, with condition and inspection date |
| `synthetic_street_corridors.parquet` | 769 | ~900 ft segments of 23 major corridors, **hand-digitized approximations** (may be off by 50–200 m) |
| `synthetic_industrial_lots.parquet` | 625 | PLUTO-like land use 06 lots clustered in known industrial areas |

Planted patterns (so the analysis tools have something to find):

* 28 repeat-report sites for illegal dumping (6–22 reports each within ~20 ft), half near industrial lots;
* derelict-vehicle concentrations in six areas;
* street segments with frequent sweeping complaints;
* 8 community districts with strongly rising complaint counts and 5 with falling counts (2022 → 2024).

Every synthetic record has `is_synthetic = true`, synthetic ids (`900000001…`, `CB-SYN-…`,
`SYN-LOT-…`, `SEG-SYN-…`) and synthetic addresses (`SYNTHETIC SITE nnn`). The app labels every
map layer and answer computed from these files as synthetic. **Do not cite results from this data
as findings about New York City.**
