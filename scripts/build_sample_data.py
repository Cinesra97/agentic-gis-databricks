"""Build the bundled offline sample datasets in ``sample_data/``.

Two kinds of data are produced:

1. **Reference boundaries (real, simplified)** -- NYC borough, community district and
   MODZCTA (ZIP-like) boundaries originally published by NYC Department of City Planning
   (DCP) and NYC Department of Health and Mental Hygiene (DOHMH). They are downloaded from
   public GitHub mirrors, simplified (~25 m tolerance) and saved as GeoParquet.

2. **Synthetic demonstration data (NOT real NYC records)** -- sanitation complaints shaped
   like NYC 311 records, catch basins, approximate street corridors and industrial lots. They
   are generated with a fixed random seed so the build is reproducible. Every synthetic
   record carries ``is_synthetic = True`` and every synthetic dataset is labeled synthetic in
   ``config/datasets.yaml``. Results computed from them are not findings about NYC.

Usage (PowerShell or bash)::

    uv run python scripts/build_sample_data.py            # downloads sources if not cached
    uv run python scripts/build_sample_data.py --offline  # use cached sources only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data"
REF = OUT / "reference"
CACHE = ROOT / "data" / "cache" / "sample_build"
SEED = 20241023
PROJ = "EPSG:2263"  # NAD83 / New York Long Island (ftUS)
WGS84 = "EPSG:4326"

SOURCES = {
    "boroughs": {
        "url": "https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson",
        "publisher": "NYC Department of City Planning (Borough Boundaries, clipped to shoreline)",
        "mirror": "github.com/dwillis/nyc-maps (ogr2ogr conversion of DCP Bytes of the Big Apple)",
    },
    "community_districts": {
        "url": "https://raw.githubusercontent.com/dwillis/nyc-maps/master/community_districts.geojson",
        "publisher": "NYC Department of City Planning (Community Districts)",
        "mirror": "github.com/dwillis/nyc-maps (ogr2ogr conversion of DCP Bytes of the Big Apple)",
    },
    "modzcta": {
        "url": (
            "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/"
            "Geography-resources/MODZCTA_2010_WGS1984.geo.json"
        ),
        "publisher": "NYC Department of Health and Mental Hygiene (Modified ZIP Code Tabulation Areas)",
        "mirror": "github.com/nychealth/coronavirus-data",
    },
}

BORO = {
    1: ("Manhattan", "MN"),
    2: ("Bronx", "BX"),
    3: ("Brooklyn", "BK"),
    4: ("Queens", "QN"),
    5: ("Staten Island", "SI"),
}

# Community district -> commonly used neighborhood label (DOHMH Community Health Profile naming).
CD_NAMES = {
    101: "Financial District",
    102: "Greenwich Village and Soho",
    103: "Lower East Side and Chinatown",
    104: "Clinton and Chelsea",
    105: "Midtown",
    106: "Stuyvesant Town and Turtle Bay",
    107: "Upper West Side",
    108: "Upper East Side",
    109: "Morningside Heights and Hamilton Heights",
    110: "Central Harlem",
    111: "East Harlem",
    112: "Washington Heights and Inwood",
    201: "Mott Haven and Melrose",
    202: "Hunts Point and Longwood",
    203: "Morrisania and Crotona",
    204: "Highbridge and Concourse",
    205: "Fordham and University Heights",
    206: "Belmont and East Tremont",
    207: "Kingsbridge Heights and Bedford",
    208: "Riverdale and Fieldston",
    209: "Parkchester and Soundview",
    210: "Throgs Neck and Co-op City",
    211: "Morris Park and Bronxdale",
    212: "Williamsbridge and Baychester",
    301: "Greenpoint and Williamsburg",
    302: "Fort Greene and Brooklyn Heights",
    303: "Bedford Stuyvesant",
    304: "Bushwick",
    305: "East New York and Starrett City",
    306: "Park Slope and Carroll Gardens",
    307: "Sunset Park",
    308: "Crown Heights and Prospect Heights",
    309: "South Crown Heights and Lefferts Gardens",
    310: "Bay Ridge and Dyker Heights",
    311: "Bensonhurst",
    312: "Borough Park",
    313: "Coney Island",
    314: "Flatbush and Midwood",
    315: "Sheepshead Bay",
    316: "Brownsville",
    317: "East Flatbush",
    318: "Flatlands and Canarsie",
    401: "Long Island City and Astoria",
    402: "Woodside and Sunnyside",
    403: "Jackson Heights",
    404: "Elmhurst and Corona",
    405: "Ridgewood and Maspeth",
    406: "Rego Park and Forest Hills",
    407: "Flushing and Whitestone",
    408: "Hillcrest and Fresh Meadows",
    409: "Kew Gardens and Woodhaven",
    410: "South Ozone Park and Howard Beach",
    411: "Bayside and Little Neck",
    412: "Jamaica and Hollis",
    413: "Queens Village",
    414: "Rockaway and Broad Channel",
    501: "St. George and Stapleton",
    502: "South Beach and Willowbrook",
    503: "Tottenville and Great Kills",
    # Joint interest areas (parks, airports) -- not community boards
    164: "Central Park",
    226: "Van Cortlandt Park",
    227: "Bronx Park",
    228: "Pelham Bay Park",
    355: "Prospect Park",
    356: "Gateway National Recreation Area (Brooklyn)",
    480: "LaGuardia Airport",
    481: "Flushing Meadows Corona Park",
    482: "Forest Park",
    483: "JFK International Airport",
    484: "Gateway National Recreation Area (Queens)",
    595: "Gateway National Recreation Area (Staten Island)",
}

# Approximate corridor centerlines, hand-digitized from general knowledge (lon, lat).
# SYNTHETIC / APPROXIMATE: these are not authoritative street geometries and may be off by
# 50-200 m. They exist so the offline demo can exercise street-based analysis.
CORRIDORS: list[tuple[str, list[tuple[float, float]]]] = [
    (
        "BROADWAY",
        [
            (-74.0137, 40.7056),
            (-74.0068, 40.7128),
            (-74.0018, 40.7190),
            (-73.9967, 40.7252),
            (-73.9903, 40.7359),
            (-73.9890, 40.7411),
            (-73.9877, 40.7505),
            (-73.9855, 40.7580),
            (-73.9819, 40.7681),
            (-73.9819, 40.7783),
            (-73.9722, 40.7937),
            (-73.9643, 40.8076),
            (-73.9503, 40.8265),
            (-73.9398, 40.8404),
            (-73.9258, 40.8656),
            (-73.9150, 40.8697),
        ],
    ),
    (
        "CANAL STREET",
        [
            (-74.0095, 40.7236),
            (-74.0050, 40.7215),
            (-74.0018, 40.7190),
            (-73.9958, 40.7165),
            (-73.9935, 40.7150),
        ],
    ),
    (
        "125 STREET",
        [
            (-73.9590, 40.8160),
            (-73.9500, 40.8100),
            (-73.9460, 40.8080),
            (-73.9370, 40.8041),
            (-73.9330, 40.8020),
        ],
    ),
    (
        "ATLANTIC AVENUE",
        [
            (-73.9990, 40.6905),
            (-73.9776, 40.6845),
            (-73.9535, 40.6812),
            (-73.9300, 40.6788),
            (-73.9225, 40.6779),
            (-73.9035, 40.6781),
            (-73.8960, 40.6757),
            (-73.8700, 40.6800),
        ],
    ),
    (
        "FLATBUSH AVENUE",
        [
            (-73.9838, 40.6960),
            (-73.9776, 40.6845),
            (-73.9701, 40.6743),
            (-73.9620, 40.6634),
            (-73.9590, 40.6505),
            (-73.9530, 40.6370),
            (-73.9480, 40.6325),
            (-73.9205, 40.6085),
            (-73.9030, 40.5950),
        ],
    ),
    (
        "FULTON STREET",
        [
            (-73.9880, 40.6915),
            (-73.9740, 40.6865),
            (-73.9500, 40.6805),
            (-73.9305, 40.6795),
            (-73.9040, 40.6785),
            (-73.8740, 40.6830),
        ],
    ),
    (
        "EASTERN PARKWAY",
        [
            (-73.9701, 40.6743),
            (-73.9636, 40.6712),
            (-73.9505, 40.6697),
            (-73.9310, 40.6688),
            (-73.9210, 40.6690),
            (-73.9110, 40.6720),
            (-73.9040, 40.6790),
        ],
    ),
    (
        "MYRTLE AVENUE",
        [
            (-73.9860, 40.6935),
            (-73.9700, 40.6935),
            (-73.9560, 40.6950),
            (-73.9400, 40.6970),
            (-73.9120, 40.6995),
            (-73.8960, 40.7000),
            (-73.8700, 40.7010),
        ],
    ),
    ("BUSHWICK AVENUE", [(-73.9420, 40.7150), (-73.9300, 40.7020), (-73.9190, 40.6950), (-73.9050, 40.6820)]),
    (
        "FLUSHING AVENUE",
        [
            (-73.9760, 40.6980),
            (-73.9580, 40.6990),
            (-73.9400, 40.7005),
            (-73.9250, 40.7035),
            (-73.9100, 40.7110),
        ],
    ),
    (
        "METROPOLITAN AVENUE",
        [
            (-73.9600, 40.7140),
            (-73.9400, 40.7140),
            (-73.9200, 40.7145),
            (-73.9050, 40.7110),
            (-73.8800, 40.7110),
            (-73.8500, 40.7100),
        ],
    ),
    (
        "LINDEN BOULEVARD",
        [
            (-73.9480, 40.6530),
            (-73.9200, 40.6560),
            (-73.9000, 40.6620),
            (-73.8800, 40.6670),
            (-73.8500, 40.6720),
        ],
    ),
    ("3 AVENUE", [(-73.9870, 40.6800), (-74.0000, 40.6600), (-74.0170, 40.6450), (-74.0300, 40.6280)]),
    (
        "GRAND CONCOURSE",
        [
            (-73.9271, 40.8186),
            (-73.9230, 40.8275),
            (-73.9185, 40.8357),
            (-73.9150, 40.8400),
            (-73.9055, 40.8490),
            (-73.8975, 40.8620),
            (-73.8935, 40.8680),
            (-73.8875, 40.8730),
            (-73.8805, 40.8800),
        ],
    ),
    (
        "FORDHAM ROAD",
        [
            (-73.9135, 40.8620),
            (-73.9020, 40.8625),
            (-73.8975, 40.8620),
            (-73.8915, 40.8610),
            (-73.8810, 40.8600),
            (-73.8730, 40.8575),
        ],
    ),
    (
        "WEBSTER AVENUE",
        [
            (-73.9090, 40.8270),
            (-73.9030, 40.8400),
            (-73.8950, 40.8530),
            (-73.8905, 40.8620),
            (-73.8800, 40.8780),
            (-73.8650, 40.8950),
        ],
    ),
    (
        "SOUTHERN BOULEVARD",
        [
            (-73.9170, 40.8070),
            (-73.9030, 40.8150),
            (-73.8920, 40.8210),
            (-73.8930, 40.8270),
            (-73.8870, 40.8380),
            (-73.8800, 40.8500),
            (-73.8790, 40.8610),
        ],
    ),
    ("HUNTS POINT AVENUE", [(-73.8905, 40.8205), (-73.8840, 40.8120), (-73.8780, 40.8060)]),
    (
        "QUEENS BOULEVARD",
        [
            (-73.9405, 40.7505),
            (-73.9185, 40.7435),
            (-73.9030, 40.7413),
            (-73.8780, 40.7370),
            (-73.8680, 40.7330),
            (-73.8615, 40.7290),
            (-73.8445, 40.7215),
            (-73.8310, 40.7140),
            (-73.8080, 40.7070),
        ],
    ),
    (
        "NORTHERN BOULEVARD",
        [
            (-73.9380, 40.7510),
            (-73.9100, 40.7540),
            (-73.8920, 40.7565),
            (-73.8700, 40.7590),
            (-73.8450, 40.7600),
            (-73.8320, 40.7630),
            (-73.7930, 40.7620),
            (-73.7680, 40.7630),
            (-73.7500, 40.7680),
        ],
    ),
    (
        "JAMAICA AVENUE",
        [
            (-73.8730, 40.6830),
            (-73.8510, 40.6930),
            (-73.8300, 40.7000),
            (-73.8010, 40.7020),
            (-73.7930, 40.7045),
            (-73.7600, 40.7120),
            (-73.7300, 40.7230),
        ],
    ),
    (
        "VICTORY BOULEVARD",
        [
            (-74.0785, 40.6420),
            (-74.0930, 40.6280),
            (-74.1140, 40.6185),
            (-74.1500, 40.6100),
            (-74.1700, 40.6060),
            (-74.1900, 40.5880),
        ],
    ),
    (
        "HYLAN BOULEVARD",
        [
            (-74.0710, 40.6150),
            (-74.0750, 40.5970),
            (-74.0900, 40.5850),
            (-74.1170, 40.5620),
            (-74.1550, 40.5420),
            (-74.1900, 40.5250),
            (-74.2350, 40.5050),
        ],
    ),
]

# Industrial clusters (lon, lat, spread_ft, n_lots, typical orientation in degrees).
INDUSTRIAL_AREAS = [
    ("Hunts Point", -73.8800, 40.8100, 1800, 70, 30),
    ("Port Morris", -73.9130, 40.8020, 1300, 40, 25),
    ("Maspeth", -73.9150, 40.7230, 2000, 70, 10),
    ("East Williamsburg", -73.9300, 40.7150, 1500, 55, 15),
    ("Newtown Creek / Greenpoint", -73.9420, 40.7330, 1200, 35, 20),
    ("Long Island City", -73.9380, 40.7450, 1400, 45, 30),
    ("Red Hook", -74.0080, 40.6750, 1100, 30, 30),
    ("Sunset Park waterfront", -74.0180, 40.6550, 1500, 45, 35),
    ("Gowanus", -73.9920, 40.6750, 900, 25, 25),
    ("Brooklyn Navy Yard", -73.9700, 40.7010, 900, 25, 0),
    ("East New York industrial", -73.8900, 40.6700, 1300, 35, 20),
    ("College Point", -73.8400, 40.7850, 1300, 30, 10),
    ("South Jamaica", -73.7850, 40.6900, 1300, 30, 5),
    ("Zerega", -73.8350, 40.8300, 1100, 25, 15),
    ("Staten Island West Shore", -74.1850, 40.5900, 2500, 40, 5),
    ("Port Ivory", -74.1700, 40.6350, 1600, 25, 10),
]

COMPLAINT_TYPES = {
    "Illegal Dumping": {
        "share": 0.18,
        "descriptors": ["Removal Request", "Surveillance Request"],
        "locations": ["Sidewalk", "Street", "Vacant Lot", "Curb"],
    },
    "Derelict Vehicles": {
        "share": 0.15,
        "descriptors": ["With License Plate", "Without License Plate"],
        "locations": ["Street"],
    },
    "Street Sweeping Complaint": {
        "share": 0.10,
        "descriptors": ["Missed Sweeping", "Wrong Side Swept"],
        "locations": ["Street"],
    },
    "Missed Collection": {
        "share": 0.22,
        "descriptors": ["Trash", "Recycling - Paper", "Recycling - Metal/Glass", "Organics"],
        "locations": ["Residential Building", "Sidewalk"],
    },
    "Dirty Condition": {
        "share": 0.20,
        "descriptors": ["Trash", "Debris", "Litter"],
        "locations": ["Sidewalk", "Street", "Lot"],
    },
    "Litter Basket Complaint": {
        "share": 0.08,
        "descriptors": ["Overflowing", "Missing Basket"],
        "locations": ["Sidewalk"],
    },
    "Sanitation Condition": {
        "share": 0.07,
        "descriptors": ["Spill", "Other"],
        "locations": ["Street", "Sidewalk"],
    },
}

RESOLUTIONS = [
    "SYNTHETIC RECORD: The Department visited the location and addressed the reported condition.",
    "SYNTHETIC RECORD: The Department inspected the location and did not observe the reported condition.",
    "SYNTHETIC RECORD: The complaint was referred for further inspection.",
    "SYNTHETIC RECORD: A cleanup was scheduled for the reported location.",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def fetch(name: str, offline: bool) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.geojson"
    if path.exists():
        return path
    if offline:
        sys.exit(f"Missing cached source {path}; run without --offline to download it.")
    url = SOURCES[name]["url"]
    print(f"Downloading {url}")
    r = httpx.get(url, timeout=120, follow_redirects=True)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def simplify_coverage(gdf: gpd.GeoDataFrame, tol_ft: float) -> gpd.GeoDataFrame:
    proj = gdf.to_crs(PROJ)
    geoms = shapely.make_valid(proj.geometry.values)
    try:
        simplified = shapely.coverage_simplify(geoms, tol_ft)
    except Exception:  # GEOS < 3.12 fallback
        simplified = shapely.simplify(geoms, tol_ft, preserve_topology=True)
    proj = proj.set_geometry(gpd.GeoSeries(shapely.make_valid(simplified), crs=PROJ))
    out = proj.to_crs(WGS84)
    out["geometry"] = shapely.set_precision(out.geometry.values, 1e-6)
    return out


def sample_in_polygons(
    rng: np.random.Generator, polys: gpd.GeoSeries, weights: np.ndarray, n: int
) -> np.ndarray:
    """Sample n points (projected coords) inside polygons chosen by weight."""
    idx = rng.choice(len(polys), size=n, p=weights / weights.sum())
    out = np.empty((n, 2))
    for i, k in enumerate(idx):
        poly = polys.iloc[k]
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(200):
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if shapely.contains_xy(poly, x, y):
                out[i] = (x, y)
                break
        else:
            p = poly.representative_point()
            out[i] = (p.x, p.y)
    return out


def point_along(line: LineString, d: float, offset: float) -> tuple[float, float]:
    d = min(max(d, 1.0), line.length - 1.0)
    p0, p1 = line.interpolate(d - 1.0), line.interpolate(d + 1.0)
    dx, dy = p1.x - p0.x, p1.y - p0.y
    norm = math.hypot(dx, dy) or 1.0
    p = line.interpolate(d)
    return p.x - dy / norm * offset, p.y + dx / norm * offset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def build_reference(offline: bool) -> dict[str, gpd.GeoDataFrame]:
    boros = gpd.read_file(fetch("boroughs", offline)).to_crs(WGS84)
    boros = boros.rename(columns={"BoroCode": "boro_code", "BoroName": "boro_name"})
    boros["boro_code"] = boros["boro_code"].astype(int)
    boros["borough"] = boros["boro_name"].str.upper()
    boros["boro_abbrev"] = boros["boro_code"].map(lambda c: BORO[c][1])
    boros = boros[["boro_code", "boro_name", "borough", "boro_abbrev", "geometry"]].sort_values("boro_code")
    boros = simplify_coverage(boros.reset_index(drop=True), 80)

    cds = gpd.read_file(fetch("community_districts", offline)).to_crs(WGS84)
    cds["boro_cd"] = cds["BoroCD"].astype(int)
    cds["boro_code"] = cds["boro_cd"] // 100
    cds["district"] = cds["boro_cd"] % 100
    cds["is_joint_interest_area"] = cds["district"] > 18
    cds["boro_name"] = cds["boro_code"].map(lambda c: BORO[c][0])
    cds["borough"] = cds["boro_name"].str.upper()
    cds["cd_code"] = cds.apply(lambda r: f"{BORO[r.boro_code][1]}{r.district:02d}", axis=1)
    cds["neighborhood"] = cds["boro_cd"].map(CD_NAMES).fillna("Unknown")
    cds["community_board"] = cds.apply(
        lambda r: None if r.is_joint_interest_area else f"{r.district:02d} {r.borough}", axis=1
    )
    cds["label"] = cds.apply(
        lambda r: f"{r.cd_code} {r.neighborhood}" if not r.is_joint_interest_area else r.neighborhood, axis=1
    )
    cds = (
        cds[
            [
                "boro_cd",
                "cd_code",
                "boro_code",
                "boro_name",
                "borough",
                "community_board",
                "neighborhood",
                "label",
                "is_joint_interest_area",
                "geometry",
            ]
        ]
        .sort_values("boro_cd")
        .reset_index(drop=True)
    )
    cds = simplify_coverage(cds, 80)

    z = gpd.read_file(fetch("modzcta", offline)).to_crs(WGS84)
    z = z.rename(columns={"MODZCTA": "modzcta", "label": "zip_codes"})
    z = z[z["modzcta"].astype(str).str.fullmatch(r"1\d{4}")].copy()
    z["modzcta"] = z["modzcta"].astype(str)
    reps = gpd.GeoDataFrame(z[["modzcta"]], geometry=z.geometry.representative_point(), crs=WGS84)
    j = gpd.sjoin(reps, boros[["borough", "geometry"]], how="left", predicate="within")
    z["borough"] = j.groupby(level=0)["borough"].first()
    z = z[["modzcta", "zip_codes", "borough", "geometry"]].sort_values("modzcta").reset_index(drop=True)
    z = simplify_coverage(z, 80)
    return {"boroughs": boros, "community_districts": cds, "modzcta": z}


def build_streets(rng: np.random.Generator, land_proj) -> gpd.GeoDataFrame:
    rows = []
    seg_no = 0
    for name, pts in CORRIDORS:
        line = gpd.GeoSeries([LineString(pts)], crs=WGS84).to_crs(PROJ).iloc[0]
        clipped = line.intersection(land_proj)
        parts = list(getattr(clipped, "geoms", [clipped]))
        for part in parts:
            if part.is_empty or part.geom_type != "LineString" or part.length < 200:
                continue
            n = max(1, int(round(part.length / 900)))
            step = part.length / n
            for i in range(n):
                seg = substring(part, i * step, (i + 1) * step)
                seg_no += 1
                rows.append(
                    {
                        "segment_id": f"SEG-SYN-{seg_no:05d}",
                        "street_name": name,
                        "street_class": "Major corridor (synthetic approximation)",
                        "length_ft": round(seg.length, 1),
                        "is_synthetic": True,
                        "geometry": seg,
                    }
                )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJ)


def build_industrial(rng: np.random.Generator, land_proj, boros_proj) -> gpd.GeoDataFrame:
    rows = []
    lot_no = 0
    zoning = ["M1-1", "M1-2", "M1-4", "M2-1", "M3-1"]
    bldg = ["E1", "E9", "F1", "F5", "F9", "G7"]
    for area, lon, lat, spread, n, angle in INDUSTRIAL_AREAS:
        c = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(PROJ).iloc[0]
        made = 0
        tries = 0
        while made < n and tries < n * 30:
            tries += 1
            x, y = rng.normal(c.x, spread / 2), rng.normal(c.y, spread / 2)
            w, d = rng.uniform(80, 260), rng.uniform(100, 320)
            rect = Polygon([(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)])
            rect = shapely.affinity.rotate(rect, angle + rng.normal(0, 4), origin=(0, 0))
            rect = shapely.affinity.translate(rect, x, y)
            if not land_proj.contains(rect):
                continue
            lot_no += 1
            made += 1
            rows.append(
                {
                    "lot_id": f"SYN-LOT-{lot_no:05d}",
                    "industrial_area": area,
                    "landuse": "06",
                    "landuse_desc": "Industrial & Manufacturing",
                    "zoning": str(rng.choice(zoning)),
                    "bldg_class": str(rng.choice(bldg)),
                    "lot_area_sqft": round(rect.area),
                    "year_built": int(rng.integers(1910, 2015)),
                    "is_synthetic": True,
                    "geometry": rect,
                }
            )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJ)
    reps = gdf.copy()
    reps["geometry"] = reps.geometry.representative_point()
    j = gpd.sjoin(reps, boros_proj[["borough", "geometry"]], how="left", predicate="within")
    gdf["borough"] = j.groupby(level=0)["borough"].first()
    return gdf


def build_catch_basins(rng, streets: gpd.GeoDataFrame, cds_land: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts, names = [], []
    for _, seg in streets.iterrows():
        line = seg.geometry
        d = rng.uniform(20, 120)
        side = 1
        while d < line.length:
            pts.append(point_along(line, d, side * rng.uniform(28, 42)))
            names.append(seg.street_name)
            side *= -1
            d += rng.uniform(220, 380)
    weights = cds_land.geometry.area.values
    rand = sample_in_polygons(rng, cds_land.geometry, weights, 2600)
    pts.extend(map(tuple, rand))
    names.extend([None] * len(rand))
    xy = np.array(pts)
    n = len(xy)
    cond = rng.choice(
        ["Clean", "Partially clogged", "Clogged", "Needs repair"], size=n, p=[0.62, 0.22, 0.11, 0.05]
    )
    start = date(2023, 1, 1).toordinal()
    inspected = [date.fromordinal(int(start + v)) for v in rng.integers(0, 730, size=n)]
    gdf = gpd.GeoDataFrame(
        {
            "basin_id": [f"CB-SYN-{i + 1:06d}" for i in range(n)],
            "street_name": names,
            "condition": cond,
            "last_inspected": pd.to_datetime(inspected),
            "maintaining_agency": "DEP (synthetic)",
            "is_synthetic": True,
        },
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
        crs=PROJ,
    )
    return gdf


def build_complaints(rng, cds_land, streets, industrial, n_total: int = 9000) -> gpd.GeoDataFrame:
    n_cd = len(cds_land)
    # Per-district yearly growth factor: most stable; a few rising strongly, a few falling.
    growth = rng.normal(1.0, 0.07, size=n_cd)
    rising = rng.choice(n_cd, size=8, replace=False)
    growth[rising] = rng.uniform(1.35, 1.6, size=8)
    remaining = np.setdiff1d(np.arange(n_cd), rising)
    falling = rng.choice(remaining, size=5, replace=False)
    growth[falling] = rng.uniform(0.62, 0.78, size=5)
    density = rng.lognormal(0, 0.45, size=n_cd)  # baseline intensity per district
    cd_weights = cds_land.geometry.area.values**0.5 * density

    xy_list: list[np.ndarray] = []
    meta: list[dict] = []

    def add(xy: np.ndarray, ctype: str, **extra) -> None:
        xy_list.append(xy)
        for _ in range(len(xy)):
            meta.append({"complaint_type": ctype, **extra})

    def background(ctype: str, n: int) -> None:
        add(sample_in_polygons(rng, cds_land.geometry, cd_weights, n), ctype)

    def along_streets(ctype: str, n: int, hot_segments: np.ndarray | None = None) -> None:
        segs = streets.reset_index(drop=True)
        w = np.ones(len(segs))
        if hot_segments is not None:
            w[hot_segments] = 14.0
        pick = rng.choice(len(segs), size=n, p=w / w.sum())
        pts = []
        for k in pick:
            line = segs.geometry.iloc[k]
            pts.append(point_along(line, rng.uniform(0, line.length), rng.uniform(-30, 30)))
        add(np.array(pts), ctype)

    counts = {k: int(v["share"] * n_total) for k, v in COMPLAINT_TYPES.items()}

    # Illegal dumping: repeat-location sites + near industrial lots + background.
    n_dump = counts["Illegal Dumping"]
    ind_cent = industrial.geometry.representative_point()
    n_sites = 28
    site_centers = []
    for s in range(n_sites):
        if s % 2 == 0:
            p = ind_cent.iloc[int(rng.integers(len(ind_cent)))]
            site_centers.append((p.x + rng.normal(0, 250), p.y + rng.normal(0, 250)))
        else:
            site_centers.append(tuple(sample_in_polygons(rng, cds_land.geometry, cd_weights, 1)[0]))
    used = 0
    for s, (cx, cy) in enumerate(site_centers):
        k = int(rng.integers(6, 23))
        xy = np.column_stack([rng.normal(cx, 18, k), rng.normal(cy, 18, k)])
        add(xy, "Illegal Dumping", location_id=f"SYN-SITE-{s + 1:03d}")
        used += k
    k = int((n_dump - used) * 0.45)
    near = []
    for _ in range(k):
        p = ind_cent.iloc[int(rng.integers(len(ind_cent)))]
        near.append((p.x + rng.normal(0, 450), p.y + rng.normal(0, 450)))
    add(np.array(near), "Illegal Dumping")
    background("Illegal Dumping", max(n_dump - used - k, 0))

    # Derelict vehicles: concentration blobs + background.
    n_dv = counts["Derelict Vehicles"]
    blobs = [
        (-73.8900, 40.6700),
        (-73.9150, 40.7230),
        (-73.8800, 40.8100),
        (-73.7850, 40.6900),
        (-74.1700, 40.6150),
        (-73.8600, 40.7500),
    ]
    blob_proj = gpd.GeoSeries([Point(b) for b in blobs], crs=WGS84).to_crs(PROJ)
    k_each = int(n_dv * 0.55 / len(blobs))
    for b in blob_proj:
        sd = rng.uniform(900, 1500)
        add(np.column_stack([rng.normal(b.x, sd, k_each), rng.normal(b.y, sd, k_each)]), "Derelict Vehicles")
    background("Derelict Vehicles", n_dv - k_each * len(blobs))

    # Street sweeping: along corridors with problem segments.
    n_sw = counts["Street Sweeping Complaint"]
    hot = rng.choice(len(streets), size=max(6, len(streets) // 12), replace=False)
    along_streets("Street Sweeping Complaint", int(n_sw * 0.7), hot)
    background("Street Sweeping Complaint", n_sw - int(n_sw * 0.7))

    # Litter baskets: mostly along commercial corridors.
    n_lb = counts["Litter Basket Complaint"]
    along_streets("Litter Basket Complaint", int(n_lb * 0.6))
    background("Litter Basket Complaint", n_lb - int(n_lb * 0.6))

    for ctype in ("Missed Collection", "Dirty Condition", "Sanitation Condition"):
        background(ctype, counts[ctype])

    xy = np.vstack(xy_list)
    df = pd.DataFrame(meta)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=PROJ)

    # Assign districts; drop points that fall in water or parks/airports.
    j = gpd.sjoin(
        gdf,
        cds_land[["boro_cd", "cd_code", "borough", "community_board", "geometry"]],
        how="inner",
        predicate="within",
    )
    j = j[~j.index.duplicated(keep="first")].drop(columns=["index_right"])
    cd_pos = {cd: i for i, cd in enumerate(cds_land["boro_cd"].tolist())}

    # Dates: district-specific growth across 2022-2024 and type seasonality.
    years = np.array([2022, 2023, 2024])
    created = []
    for _, r in j.iterrows():
        g = growth[cd_pos[r.boro_cd]]
        w = g ** (years - 2022)
        y = int(rng.choice(years, p=w / w.sum()))
        if r.complaint_type == "Street Sweeping Complaint":
            month = int(rng.choice(np.arange(4, 12)))
        elif r.complaint_type == "Illegal Dumping":
            mw = np.array([0.7, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.0, 0.9, 0.8])
            month = int(rng.choice(np.arange(1, 13), p=mw / mw.sum()))
        else:
            month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        hour = int(
            rng.choice(
                np.arange(24),
                p=np.r_[np.full(6, 0.01), np.full(16, 0.055), np.full(2, 0.03)]
                / np.r_[np.full(6, 0.01), np.full(16, 0.055), np.full(2, 0.03)].sum(),
            )
        )
        created.append(datetime(y, month, day, hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))))
    j["created_date"] = pd.to_datetime(created)
    open_mask = rng.random(len(j)) < np.where(j["created_date"] > pd.Timestamp("2024-11-15"), 0.6, 0.04)
    hours = rng.exponential(72, size=len(j))
    j["closed_date"] = j["created_date"] + pd.to_timedelta(hours, unit="h")
    j.loc[open_mask, "closed_date"] = pd.NaT
    j["status"] = np.where(open_mask, "Open", "Closed")
    j["resolution_description"] = [None if o else str(rng.choice(RESOLUTIONS)) for o in open_mask]
    j["descriptor"] = [str(rng.choice(COMPLAINT_TYPES[t]["descriptors"])) for t in j["complaint_type"]]
    j["location_type"] = [str(rng.choice(COMPLAINT_TYPES[t]["locations"])) for t in j["complaint_type"]]
    j["agency"] = "DSNY"
    if "location_id" not in j:
        j["location_id"] = None

    # Street names / synthetic addresses for points near the synthetic corridors.
    near = gpd.sjoin_nearest(
        j[["geometry"]], streets[["street_name", "geometry"]], how="left", max_distance=120, distance_col="_d"
    )
    near = near[~near.index.duplicated(keep="first")]
    j["street_name"] = near["street_name"]
    house = rng.integers(1, 2999, size=len(j))
    j["incident_address"] = [
        f"{h} {s}" if isinstance(s, str) else None for h, s in zip(house, j["street_name"], strict=True)
    ]
    # Repeat sites share one synthetic address.
    for site, idx in j.groupby("location_id").groups.items():
        if site:
            j.loc[idx, "incident_address"] = f"SYNTHETIC SITE {site[-3:]}"
            j.loc[idx, "location_type"] = str(rng.choice(["Vacant Lot", "Sidewalk", "Street"]))

    j = j.sort_values("created_date").reset_index(drop=True)
    j.insert(0, "unique_key", np.arange(900_000_001, 900_000_001 + len(j), dtype="int64"))
    j["is_synthetic"] = True
    return j


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="Use cached sources only")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)
    REF.mkdir(exist_ok=True)

    ref = build_reference(args.offline)
    boros, cds, zips = ref["boroughs"], ref["community_districts"], ref["modzcta"]

    boros_proj = boros.to_crs(PROJ)
    land_proj = shapely.union_all(boros_proj.geometry.values)
    cds_proj = cds.to_crs(PROJ)
    cds_land = cds_proj[~cds_proj["is_joint_interest_area"]].reset_index(drop=True)

    streets = build_streets(rng, land_proj)
    sj = gpd.sjoin(
        streets.assign(geometry=streets.geometry.interpolate(0.5, normalized=True)),
        boros_proj[["borough", "geometry"]],
        how="left",
        predicate="within",
    )
    streets["borough"] = sj.groupby(level=0)["borough"].first()
    industrial = build_industrial(rng, land_proj, boros_proj)
    basins = build_catch_basins(rng, streets, cds_land)
    jb = gpd.sjoin(basins, cds_land[["borough", "cd_code", "geometry"]], how="inner", predicate="within")
    basins = jb[~jb.index.duplicated()].drop(columns=["index_right"]).reset_index(drop=True)
    complaints = build_complaints(rng, cds_land, streets, industrial)

    zips_proj = zips.to_crs(PROJ)
    jz = gpd.sjoin(
        complaints[["geometry"]], zips_proj[["modzcta", "geometry"]], how="left", predicate="within"
    )
    complaints["incident_zip"] = jz[~jz.index.duplicated()]["modzcta"]

    complaints = complaints.to_crs(WGS84)
    complaints["latitude"] = complaints.geometry.y.round(6)
    complaints["longitude"] = complaints.geometry.x.round(6)
    complaints = complaints[
        [
            "unique_key",
            "created_date",
            "closed_date",
            "agency",
            "complaint_type",
            "descriptor",
            "location_type",
            "incident_zip",
            "incident_address",
            "street_name",
            "borough",
            "community_board",
            "cd_code",
            "status",
            "resolution_description",
            "location_id",
            "latitude",
            "longitude",
            "is_synthetic",
            "geometry",
        ]
    ]
    basins = basins.to_crs(WGS84)
    basins["latitude"] = basins.geometry.y.round(6)
    basins["longitude"] = basins.geometry.x.round(6)

    outputs = {
        REF / "nyc_boroughs.parquet": boros,
        REF / "nyc_community_districts.parquet": cds,
        REF / "nyc_modzcta.parquet": zips,
        OUT / "synthetic_street_corridors.parquet": streets.to_crs(WGS84),
        OUT / "synthetic_industrial_lots.parquet": industrial.to_crs(WGS84),
        OUT / "synthetic_catch_basins.parquet": basins,
        OUT / "synthetic_311_sanitation.parquet": complaints,
    }
    manifest = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "generator": "scripts/build_sample_data.py",
        "files": {},
        "sources": SOURCES,
    }
    for path, gdf in outputs.items():
        gdf = gdf.copy()
        gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-6)
        gdf.to_parquet(path, index=False, compression="zstd")
        manifest["files"][str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "rows": len(gdf),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "synthetic": bool(path.parent == OUT),
        }
        print(f"wrote {path.relative_to(ROOT)}: {len(gdf):,} rows, {path.stat().st_size / 1024:.0f} KB")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
