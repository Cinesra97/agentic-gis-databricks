# Map layer schema

The agent never writes UI code for a question. Tools emit `MapLayerSpec` objects (JSON) that reference a
result artifact; the Folium and PyDeck renderers draw any spec. JSON Schema:
[`docs/map_layer_schema.json`](map_layer_schema.json) (regenerate with `uv run agentic-gis layer-schema`).

## Example

```json
{
  "schema_version": "1.0",
  "id": "layer-t3.s2-1",
  "title": "Change in 311 sanitation complaints by neighborhood (synthetic)",
  "artifact_id": "t3.s2",
  "render": "choropleth",
  "geometry_kind": "polygon",
  "style": {
    "color_role": "diverging",
    "color_by": "change",
    "breaks": [-63, -11, -5, -1, 5, 14, 72],
    "categories": [],
    "radius": 5.0, "size_by": null, "size_range": [5.0, 16.0],
    "weight": 2.0, "opacity": 0.9, "fill_opacity": 0.65
  },
  "tooltip": ["label", "neighborhood", "count_a", "count_b", "change", "pct_change"],
  "legend_title": "change",
  "legend": [{"label": "-63 to -11", "role": "div:neg:0"}, {"label": "0 (no change)", "role": "div:mid"}],
  "visible": true,
  "feature_count": 59,
  "bounds": [-74.255, 40.496, -73.700, 40.915],
  "synthetic": true,
  "provenance": ["demo_311_sanitation", "nyc_community_districts"],
  "z_index": 0,
  "note": null
}
```

## Fields

| Field | Meaning |
|---|---|
| `render` | `points`, `cluster` (marker clustering), `heatmap`, `choropleth`, `lines`, `polygons` |
| `style.color_role` | `single`, `context` (muted reference layer), `categorical` (≤3 values + Other), `sequential`, `diverging`, `hotspot` (Gi* classes) |
| `style.breaks` | Upper bounds of numeric classes (quantiles; integers for counts; per-arm for diverging) |
| `style.size_by` | Numeric column scaling marker radius (square-root scaling within `size_range`) |
| `legend[].role` | Palette role (`seq:2`, `cat:0`, `div:pos:1`, `other`, `context`), resolved per basemap theme |
| `tooltip` | Columns shown on hover and in popups (click) |
| `note` | Rendering caveat, e.g. sampling of very large point layers |

## Colors

Colors are roles, resolved at render time for light or dark basemaps, using palettes checked with a
color-vision-deficiency and contrast validator:

* Categorical: three fixed slots (blue, orange, aqua) + gray “Other” — the most a map can show with
  every pair distinguishable for color-blind readers.
* Sequential: one blue hue; low values recede toward the basemap on polygons; thin marks (lines/markers)
  use a variant whose lightest/darkest step still clears 2:1 contrast.
* Diverging: blue (decrease / cold) ↔ neutral gray ↔ red (increase / hot), three steps per arm.

## Rendering rules

* Context layers and heat maps draw beneath analytical layers.
* Very small, sparse polygons (e.g. 75 ft repeat-report clusters) are drawn as size-scaled markers at their
  centroids so they remain visible citywide.
* Point layers above `AGIS_MAX_RENDER_FEATURES` (default 6,000) are sampled for display only; the table
  and exports keep every row.
* Every synthetic layer has “(synthetic)” in its title and the map shows a SYNTHETIC DEMO DATA badge.
