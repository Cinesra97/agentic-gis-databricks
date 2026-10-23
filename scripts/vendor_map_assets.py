"""Vendor the Leaflet JavaScript/CSS used by Folium maps into ``static/vendor/``.

Why: by default Folium loads Leaflet and plugins from public CDNs. Government and air-gapped
deployments often block CDNs, which leaves the map blank. With vendored assets (served by
Streamlit's static file server) the map works offline; only basemap tiles need network access.

Packages are fetched from the npm registry at pinned versions and verified against the
registry's SHA-512 integrity value before extraction.

    uv run python scripts/vendor_map_assets.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "static" / "vendor"

PACKAGES = {
    # name: (version, {path inside package: destination relative to DEST/<name>})
    "leaflet": (
        "1.9.4",
        {
            "dist/leaflet.js": "leaflet.js",
            "dist/leaflet.css": "leaflet.css",
            "dist/images/": "images/",
            "LICENSE": "LICENSE",
        },
    ),
    "leaflet.markercluster": (
        "1.5.3",
        {
            "dist/leaflet.markercluster.js": "leaflet.markercluster.js",
            "dist/MarkerCluster.css": "MarkerCluster.css",
            "dist/MarkerCluster.Default.css": "MarkerCluster.Default.css",
            "MIT-LICENCE.txt": "LICENSE",
        },
    ),
    "leaflet.heat": ("0.2.0", {"dist/leaflet-heat.js": "leaflet-heat.js", "LICENSE": "LICENSE"}),
    "jquery": ("3.7.1", {"dist/jquery.min.js": "jquery.min.js", "LICENSE.txt": "LICENSE"}),
}


def fetch(name: str, version: str) -> tarfile.TarFile:
    meta = httpx.get(f"https://registry.npmjs.org/{name}/{version}", timeout=60).json()
    url, integrity = meta["dist"]["tarball"], meta["dist"]["integrity"]
    data = httpx.get(url, timeout=120, follow_redirects=True).content
    algo, b64 = integrity.split("-", 1)
    digest = base64.b64encode(hashlib.new(algo, data).digest()).decode()
    if digest != b64:
        raise SystemExit(f"Integrity check failed for {name}@{version}")
    print(f"{name}@{version}: {len(data):,} bytes, integrity OK ({algo})")
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    for name, (version, files) in PACKAGES.items():
        tf = fetch(name, version)
        members = {m.name.removeprefix("package/"): m for m in tf.getmembers() if m.isfile()}
        out = DEST / name
        for src, dst in files.items():
            matches = [k for k in members if k == src or (src.endswith("/") and k.startswith(src))]
            if not matches:
                raise SystemExit(f"{src} not found in {name}@{version}")
            for k in matches:
                target = out / (dst + k[len(src) :] if src.endswith("/") else dst)
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = tf.extractfile(members[k])
                assert fh is not None
                target.write_bytes(fh.read())
        (out / "VERSION").write_text(f"{name}@{version}\n", encoding="utf-8")
    print(f"Vendored assets written to {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
