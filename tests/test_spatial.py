import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from agentic_gis.spatial import analysis as A
from agentic_gis.spatial import operations as O
from agentic_gis.spatial.geometry import SpatialError, ensure_gdf, to_crs_units

P = "EPSG:2263"


def pts(coords, **cols):
    return gpd.GeoDataFrame(cols, geometry=[Point(x, y) for x, y in coords], crs="EPSG:4326")


def test_units_conversion():
    assert to_crs_units(300, "ft", P) == pytest.approx(300 * 0.3048 / 0.3048006096, rel=1e-6)
    assert to_crs_units(1, "mi", P) == pytest.approx(5280, rel=1e-4)
    with pytest.raises(SpatialError):
        to_crs_units(10, "ft", "EPSG:4326")


def test_buffer_distance_is_in_feet():
    g = pts([(-73.95, 40.70)])
    b = O.buffer_features(g, 300, "ft", P)
    area_ft2 = b.to_crs(P).geometry.area.iloc[0]
    assert area_ft2 == pytest.approx(np.pi * 300**2, rel=0.01)


def test_dwithin_join_and_nearest():
    line = gpd.GeoDataFrame(
        {"street_name": ["A ST"]}, geometry=[LineString([(-73.95, 40.70), (-73.94, 40.70)])], crs="EPSG:4326"
    )
    lp = line.to_crs(P).geometry.iloc[0]
    near = (
        gpd.GeoSeries([Point(lp.interpolate(100).x, lp.interpolate(100).y + 200)], crs=P).to_crs(4326).iloc[0]
    )
    far = (
        gpd.GeoSeries([Point(lp.interpolate(100).x, lp.interpolate(100).y + 600)], crs=P).to_crs(4326).iloc[0]
    )
    g = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[near, far], crs="EPSG:4326")
    j = O.spatial_join(g, line, "dwithin", P, distance=300, units="ft", summarize=True)
    assert list(j["id"]) == [1] and j["nearest_match_ft"].iloc[0] == pytest.approx(200, abs=1)
    n = O.nearest_features(g, line, P)
    assert n.sort_values("id")["distance_ft"].round().tolist() == [200, 600]
    n2 = O.nearest_features(g, line, P, max_distance=300)
    assert len(n2) == 1


def test_match_names():
    m = O.match_names(
        ["Atlantic Ave", "Grand Concourse", "Nowhere Rd"], ["ATLANTIC AVENUE", "GRAND CONCOURSE", "BROADWAY"]
    )
    assert m["Atlantic Ave"] == ["ATLANTIC AVENUE"] and m["Grand Concourse"] == ["GRAND CONCOURSE"]
    assert m["Nowhere Rd"] == []


def test_dbscan_finds_planted_cluster():
    rng = np.random.default_rng(0)
    base = gpd.GeoSeries([Point(-73.95, 40.70)], crs="EPSG:4326").to_crs(P).iloc[0]
    cluster = [(base.x + rng.normal(0, 10), base.y + rng.normal(0, 10)) for _ in range(8)]
    noise = [(base.x + rng.uniform(2000, 9000), base.y + rng.uniform(2000, 9000)) for _ in range(20)]
    g = gpd.GeoDataFrame(
        {"t": ["Illegal Dumping"] * 28, "d": pd.date_range("2024-01-01", periods=28)},
        geometry=gpd.points_from_xy(*zip(*(cluster + noise), strict=True)),
        crs=P,
    ).to_crs(4326)
    r = A.dbscan_clusters(g, P, radius=75, min_points=5, category_column="t", date_column="d")
    assert r.stats["clusters"] == 1 and r.data["count"].iloc[0] == 8
    assert r.data["top_category"].iloc[0] == "Illegal Dumping"


def test_gi_star_detects_hotspot():
    rng = np.random.default_rng(1)
    base = gpd.GeoSeries([Point(-73.95, 40.70)], crs="EPSG:4326").to_crs(P).iloc[0]
    hot = [(base.x + rng.normal(0, 200), base.y + rng.normal(0, 200)) for _ in range(150)]
    bg = [(base.x + rng.uniform(-20000, 20000), base.y + rng.uniform(-20000, 20000)) for _ in range(300)]
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*(hot + bg), strict=True)), crs=P).to_crs(4326)
    r = A.gi_star_hotspots(g, P, cell_size=1000)
    top = r.data.iloc[0]
    assert top["hotspot_class"] == "Hot spot (99%)"
    assert top.geometry.wkt and r.stats["hot_cells_95"] >= 1


def test_aggregate_and_compare_periods():
    areas = gpd.GeoDataFrame(
        {"name": ["A", "B"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326"
    )
    dates = pd.to_datetime(
        [
            "2022-01-05",
            "2022-06-01",
            "2022-12-20",
            "2023-03-01",
            "2024-01-03",
            "2024-05-01",
            "2024-07-01",
            "2024-12-15",
        ]
    )
    g = pts(
        [(0.5, 0.5)] * 3 + [(1.5, 0.5)] + [(0.5, 0.5)] + [(1.5, 0.5)] * 3,
        d=dates,
        cat=["x", "y", "x", "x", "x", "y", "y", "z"],
    )
    r = A.aggregate_to_polygons(g, areas, "name", category_column="cat")
    assert dict(zip(r.data["name"], r.data["value"], strict=True)) == {"A": 4, "B": 4}
    c = A.compare_periods(g, areas, "d", "name", min_baseline=1)
    row = c.data.set_index("name").loc["B"]
    assert (row["count_a"], row["count_b"], row["change"]) == (0, 3, 3)
    assert c.stats["period_a"].startswith("2022") and c.stats["period_b"].startswith("2024")


def test_temporal_patterns_and_grid():
    g = pts(
        [(-73.95 + i * 1e-4, 40.7) for i in range(24)],
        d=pd.date_range("2022-01-01", periods=24, freq="MS"),
        k=["a", "b"] * 12,
    )
    r = A.temporal_patterns(g, "d", "year", "k")
    assert set(r.data["group"]) == {"a", "b"} and r.stats["full_years"] == [2022, 2023]
    gr = A.grid_aggregate(g, P, 500, measure="count_distinct", value_column="k", min_value=2)
    assert (gr.data["value"] >= 2).all()


def test_ensure_gdf_variants():
    df = pd.DataFrame({"latitude": [40.7], "longitude": [-73.9]})
    assert ensure_gdf(df).geometry.iloc[0].x == -73.9
    df2 = pd.DataFrame({"geom_wkt": ["POINT (-73.9 40.7)"]})
    assert ensure_gdf(df2).geometry.iloc[0].y == 40.7
    assert ensure_gdf(pd.DataFrame({"a": [1]})) is None
