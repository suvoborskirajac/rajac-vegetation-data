#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIO Rajac vegetation indices processor for GitHub Actions.

Builds monthly + yearly JSON files for the WordPress viewer using
Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED) via
Google Earth Engine. Each pixel carries values for all configured
vegetation indices, so the front-end can switch indices without
re-fetching.

Outputs:
- public/results/index.json
- public/results/YYYY-MM.json   (monthly composites)
- public/results/YYYY.json      (yearly composites)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee

ROOT = Path(__file__).resolve().parent
BOUNDARY = ROOT / "public" / "boundaries" / "pio-rajac.geojson"
# Šume 2024 — 53 полигона, 1.035,5 ha = 58,5% ПИО (без преклапaња).
# Користи clipped верзиjу (1.032,8 ha) за чист мониторинг искључиво у ПИО грaницама.
FOREST_BOUNDARY = ROOT / "public" / "boundaries" / "pio-rajac-forests-2024-clipped.geojson"
RESULTS = ROOT / "public" / "results"
RESULTS_FORESTS = ROOT / "public" / "results-forests"  # одвоjeни директоријум за шуме

# Sentinel-2 collection — Surface Reflectance, harmonized between S2A and S2B.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Default sampling resolution for the heat-map pixels (metres).
# 100 m keeps the total pixel count under GEE's 5000-element FeatureCollection
# limit for PIO Rajac (~33 km² → ~3300 pixels). For finer detail pass --scale 60
# with sampleRegions/tiling implemented separately.
DEFAULT_SCALE_M = 100.0

# SCL classes to mask out (keep vegetation, soil, water, unclassified).
# 1 saturated, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus, 11 snow/ice
SCL_BAD = [1, 3, 8, 9, 10, 11]

# Index definitions — Earth Engine expression with surface-reflectance band names.
# Each entry: id, label, expression, scale (display range), unit/description, palette type.
INDICES: List[Dict[str, Any]] = [
    {
        "id": "ndvi",
        "label": "NDVI",
        "name_sr": "Виталност вегетације",
        "expr": "(B8 - B4) / (B8 + B4)",
        "min": -0.2, "max": 0.9,
        "palette": "vegetation",
        "desc": "Normalized Difference Vegetation Index — основни показатељ виталности и густине вегетације.",
    },
    {
        "id": "evi",
        "label": "EVI",
        "name_sr": "Побољшани вегетациони индекс",
        "expr": "2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))",
        "min": -0.2, "max": 1.0,
        "palette": "vegetation",
        "desc": "Enhanced Vegetation Index — отпорнији на засићење у густим шумама него NDVI.",
    },
    {
        "id": "ndre",
        "label": "NDRE",
        "name_sr": "Хлорофил (Red Edge)",
        "expr": "(B8 - B5) / (B8 + B5)",
        "min": -0.1, "max": 0.6,
        "palette": "chlorophyll",
        "desc": "Normalized Difference Red Edge — концентрација хлорофила; рани показатељ стреса.",
    },
    {
        "id": "ndmi",
        "label": "NDMI",
        "name_sr": "Влажност вегетације",
        "expr": "(B8 - B11) / (B8 + B11)",
        "min": -0.4, "max": 0.6,
        "palette": "moisture",
        "desc": "Normalized Difference Moisture Index — садржај воде у листу; стрес од суше.",
    },
    {
        "id": "ndwi",
        "label": "NDWI",
        "name_sr": "Водене површине",
        "expr": "(B3 - B8) / (B3 + B8)",
        "min": -0.5, "max": 0.6,
        "palette": "water",
        "desc": "Normalized Difference Water Index — отворене водене површине (потоци, акумулације).",
    },
    {
        "id": "nbr",
        "label": "NBR",
        "name_sr": "Опожарена подручја",
        "expr": "(B8 - B12) / (B8 + B12)",
        "min": -0.5, "max": 0.9,
        "palette": "burn",
        "desc": "Normalized Burn Ratio — детекција опожарених подручја и опоравка после пожара.",
    },
    {
        "id": "bsi",
        "label": "BSI",
        "name_sr": "Голо тло / ерозија",
        "expr": "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
        "min": -0.5, "max": 0.5,
        "palette": "bare",
        "desc": "Bare Soil Index — површине без вегетационог покривача; индикатор ерозије или крчења.",
    },
]


@dataclass
class Period:
    id: str
    label: str
    start: str
    end: str


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(path)


def previous_month_start(d: date) -> date:
    first = date(d.year, d.month, 1)
    prev_last = first - timedelta(days=1)
    return date(prev_last.year, prev_last.month, 1)


def month_after(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def period_from_month_start(start: date) -> Period:
    end = month_after(start)
    return Period(
        id=f"{start.year}-{start.month:02d}",
        label=f"{start.month:02d}/{start.year}",
        start=start.isoformat(),
        end=end.isoformat(),
    )


def latest_candidate_periods(n: int, extra_back: int = 8) -> List[Period]:
    cursor = date.today().replace(day=1)
    periods: List[Period] = []
    for _ in range(max(1, n + extra_back)):
        start = previous_month_start(cursor)
        periods.append(period_from_month_start(start))
        cursor = start
    return periods


def yearly_period(year: int) -> Period:
    return Period(
        id=f"{year}",
        label=f"{year}. (годишњи просек)",
        start=f"{year}-01-01",
        end=f"{year + 1}-01-01",
    )


def yearly_candidate_periods(start_year: int, end_year: int) -> List[Period]:
    return [yearly_period(y) for y in range(end_year, start_year - 1, -1)]


def geojson_to_ee_geometry(gj: Dict[str, Any]) -> ee.Geometry:
    if gj.get("type") == "FeatureCollection":
        geoms = [feat.get("geometry") for feat in gj.get("features", []) if feat.get("geometry")]
        if not geoms:
            raise ValueError("GeoJSON FeatureCollection nema geometrije.")
        if len(geoms) == 1:
            return ee.Geometry(geoms[0])
        polys: List[Any] = []
        for geom in geoms:
            if geom.get("type") == "Polygon":
                polys.append(geom["coordinates"])
            elif geom.get("type") == "MultiPolygon":
                polys.extend(geom["coordinates"])
        return ee.Geometry.MultiPolygon(polys)
    if gj.get("type") == "Feature":
        return ee.Geometry(gj["geometry"])
    return ee.Geometry(gj)


def init_ee() -> None:
    project = os.environ.get("GEE_PROJECT", "deft-epigram-414409").strip()
    secret = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    if not secret:
        raise RuntimeError("Nedostaje GitHub secret GEE_SERVICE_ACCOUNT_JSON.")
    try:
        key = json.loads(secret)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_JSON nije validan JSON tekst.") from exc
    email = key.get("client_email")
    if not email:
        raise RuntimeError("Service-account JSON nema client_email.")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(key, tmp)
        tmp_path = tmp.name
    credentials = ee.ServiceAccountCredentials(email, key_file=tmp_path)
    ee.Initialize(credentials, project=project)
    log(f"Earth Engine initialized for project: {project}; service account: {email}")


def safe_round(v: Any, ndigits: int = 4) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, ndigits)
    except Exception:
        return None


def lonlat_bbox_from_center(lon: float, lat: float, scale_m: float) -> List[float]:
    dlat = scale_m / 111_320.0
    dlon = scale_m / (111_320.0 * max(0.15, math.cos(math.radians(lat))))
    return [
        round(lon - dlon / 2, 6),
        round(lat - dlat / 2, 6),
        round(lon + dlon / 2, 6),
        round(lat + dlat / 2, 6),
    ]


def mask_s2_scl(img: ee.Image) -> ee.Image:
    """Mask out clouds/shadows/snow using the Scene Classification Layer."""
    scl = img.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_BAD:
        mask = mask.And(scl.neq(cls))
    return img.updateMask(mask)


def s2_collection_for_period(period: Period, region: ee.Geometry, max_cloud_pct: int = 80) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(period.start, period.end)
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_s2_scl)
    )


def has_images(period: Period, region: ee.Geometry) -> bool:
    try:
        size = s2_collection_for_period(period, region).size().getInfo()
        return int(size or 0) > 0
    except Exception as exc:
        log(f"WARN: ne mogu proveriti dostupnost za {period.id}: {exc}")
        return False


def build_index_image(collection: ee.ImageCollection) -> ee.Image:
    """Median composite of all bands needed by the index expressions, scaled to 0–1 reflectance."""
    # Sentinel-2 SR is scaled by 10000. Keep the bands we need.
    needed = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]
    composite = collection.select(needed).median().multiply(0.0001).rename(needed)
    images = {}
    for idx in INDICES:
        images[idx["id"]] = composite.expression(idx["expr"], {b: composite.select(b) for b in needed}).rename(idx["id"])
    out = ee.Image.cat(list(images.values()))
    return out


def reduce_stats_for_index(image: ee.Image, band: str, region: ee.Geometry, scale_m: float) -> Dict[str, Any]:
    reducer = (
        ee.Reducer.minMax()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    info = image.select(band).reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale_m,
        maxPixels=2_000_000,
        bestEffort=True,
        tileScale=4,
    ).getInfo() or {}

    def pick(suffix: str):
        keyed = f"{band}_{suffix}"
        return info.get(keyed) if keyed in info else info.get(suffix)

    return {
        "min": safe_round(pick("min")),
        "max": safe_round(pick("max")),
        "mean": safe_round(pick("mean")),
        "median": safe_round(pick("median")),
        "p10": safe_round(pick("p10")),
        "p90": safe_round(pick("p90")),
        "count": int(pick("count") or 0),
    }


def _features_to_pixels(features: List[Dict[str, Any]], band_ids: List[str], scale_m: float) -> List[Dict[str, Any]]:
    pixels = []
    for feat in features:
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        values: Dict[str, Optional[float]] = {}
        any_valid = False
        for band in band_ids:
            v = props.get(band)
            r = safe_round(v, 4) if v is not None else None
            values[band] = r
            if r is not None:
                any_valid = True
        if not any_valid:
            continue
        pixels.append({
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "bbox": lonlat_bbox_from_center(lon, lat, scale_m),
            "v": values,
        })
    return pixels


def _tile_grid(bbox: List[float], n_cols: int, n_rows: int) -> List[List[float]]:
    minLon, minLat, maxLon, maxLat = bbox
    dLon = (maxLon - minLon) / n_cols
    dLat = (maxLat - minLat) / n_rows
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            lo = minLon + c * dLon
            la = minLat + r * dLat
            tiles.append([lo, la, lo + dLon, la + dLat])
    return tiles


def _estimate_tile_count(bbox: List[float], scale_m: float, max_per_tile: int = 4500) -> Tuple[int, int]:
    minLon, minLat, maxLon, maxLat = bbox
    midLat = (minLat + maxLat) / 2
    width_m = (maxLon - minLon) * 111_320.0 * math.cos(math.radians(midLat))
    height_m = (maxLat - minLat) * 111_320.0
    total = (width_m / scale_m) * (height_m / scale_m)
    if total <= max_per_tile:
        return (1, 1)
    n_tiles = int(math.ceil(total / max_per_tile))
    n = int(math.ceil(math.sqrt(n_tiles)))
    return (n, n)


def sample_pixels(image: ee.Image, region: ee.Geometry, region_bbox: List[float], scale_m: float) -> List[Dict[str, Any]]:
    """Sample pixels for all indices. Auto-tiles when count would exceed GEE's
    5000-element FeatureCollection limit, so high-resolution (10–30 m) works."""
    band_ids = [idx["id"] for idx in INDICES]
    n_cols, n_rows = _estimate_tile_count(region_bbox, scale_m, max_per_tile=4500)

    if n_cols == 1 and n_rows == 1:
        fc = image.select(band_ids).sample(region=region, scale=scale_m, geometries=True, tileScale=4)
        data = fc.getInfo()
        features = data.get("features", []) if isinstance(data, dict) else []
        pixels = _features_to_pixels(features, band_ids, scale_m)
    else:
        log(f"  Tiling sample at {scale_m}m: {n_cols}x{n_rows} = {n_cols * n_rows} tiles")
        pixels: List[Dict[str, Any]] = []
        seen = set()
        tiles = _tile_grid(region_bbox, n_cols, n_rows)
        for ti, tb in enumerate(tiles, 1):
            tile_rect = ee.Geometry.Rectangle(tb)
            tile_geom = tile_rect.intersection(region, 1)
            try:
                area = tile_geom.area(1).getInfo() or 0
            except Exception:
                area = 0
            if area <= 0:
                continue
            try:
                fc = image.select(band_ids).sample(region=tile_geom, scale=scale_m, geometries=True, tileScale=4)
                data = fc.getInfo()
            except Exception as exc:
                log(f"  WARN tile {ti}/{len(tiles)}: {exc}")
                continue
            features = data.get("features", []) if isinstance(data, dict) else []
            chunk = _features_to_pixels(features, band_ids, scale_m)
            for p in chunk:
                key = (p["lon"], p["lat"])
                if key in seen:
                    continue
                seen.add(key)
                pixels.append(p)

    pixels.sort(key=lambda p: (p["lat"], p["lon"]))
    for i, p in enumerate(pixels, 1):
        p["id"] = f"px-{i:04d}"
    return pixels


def compute_hotspots(pixels: List[Dict[str, Any]], band: str, top_n: int = 5, low: bool = False) -> List[Dict[str, Any]]:
    """Top-N (or bottom-N if low=True) pixels for one index."""
    ranked = sorted(
        [p for p in pixels if p["v"].get(band) is not None],
        key=lambda p: float(p["v"][band]),
        reverse=not low,
    )
    out = []
    for i, p in enumerate(ranked[:top_n], 1):
        out.append({"rank": i, "lon": p["lon"], "lat": p["lat"], "value": p["v"][band]})
    return out


def build_result(period: Period, region: ee.Geometry, region_bbox: List[float], scale_m: float, kind: str = "monthly") -> Dict[str, Any]:
    coll = s2_collection_for_period(period, region)
    n = int(coll.size().getInfo() or 0)
    if n == 0:
        raise RuntimeError(f"Period {period.id} nema Sentinel-2 snimaka unutar PIO Rajac.")
    image = build_index_image(coll)
    pixels = sample_pixels(image, region, region_bbox, scale_m)
    if not pixels:
        raise RuntimeError(f"Period {period.id}: nakon cloud-mask filtera nema validnih piksela.")

    stats: Dict[str, Any] = {}
    for idx in INDICES:
        s = reduce_stats_for_index(image, idx["id"], region, scale_m)
        stats[idx["id"]] = s

    return {
        "ok": True,
        "meta": {
            "id": period.id,
            "label": period.label,
            "kind": kind,
            "source": "Copernicus Sentinel-2 SR Harmonized (COPERNICUS/S2_SR_HARMONIZED)",
            "dataset": S2_COLLECTION,
            "scale_m": scale_m,
            "date_start": period.start,
            "date_end": period.end,
            "image_count": n,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                          "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
        },
        "stats": stats,
        "pixels": pixels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12, help="Number of complete months to process.")
    parser.add_argument("--months-skip", type=int, default=0, help="Skip the first N most-recent months before processing (default 0). Useful for non-overlapping backfill batches.")
    parser.add_argument("--yearly", action="store_true", help="Also generate yearly composites.")
    parser.add_argument("--yearly-from", type=int, default=2017, help="Earliest year for yearly composites (default 2017 — full S2A+S2B).")
    parser.add_argument("--yearly-to", type=int, default=date.today().year - 1, help="Latest year (default last complete calendar year).")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_M, help="Sampling resolution in metres (default 100).")
    parser.add_argument("--hi-res-from", type=int, default=None, help="Months whose year >= this use --hi-res-scale (e.g. 2026).")
    parser.add_argument("--hi-res-scale", type=float, default=20.0, help="Resolution for hi-res periods (default 20 — native Sentinel-2 for SWIR/RedEdge).")
    parser.add_argument("--ultra-res-scale", type=float, default=10.0, help="Maximum native resolution (default 10 — full Sentinel-2 native).")
    parser.add_argument("--ultra-res-from", type=str, default=None, help="Months whose id (YYYY-MM) >= this use --ultra-res-scale. Example: 2026-06 enables 10 m from June 2026 onwards.")
    parser.add_argument("--ultra-res-months", type=int, default=0, help="Alternative threshold: apply --ultra-res-scale to the most recent N months (default 0). Use either --ultra-res-from OR --ultra-res-months.")
    parser.add_argument("--forests", action="store_true", help="Also process forest-only composites using public/boundaries/pio-rajac-forests.geojson (137 polygons, ~5383 ha).")
    parser.add_argument("--forests-only", action="store_true", help="Process ONLY the forest mask (skip the PIO-wide composites). Useful for quick forest reruns.")
    args = parser.parse_args()

    if not BOUNDARY.exists():
        raise RuntimeError(f"Nedostaje granica: {BOUNDARY}")

    init_ee()

    def compute_bbox(gj):
        lons, lats = [], []
        def _walk(c):
            if isinstance(c, list):
                if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)):
                    lons.append(c[0]); lats.append(c[1])
                else:
                    for x in c: _walk(x)
        for f in gj.get("features", []) if gj.get("type") == "FeatureCollection" else [gj]:
            _walk((f.get("geometry") or f).get("coordinates"))
        return [min(lons), min(lats), max(lons), max(lats)]

    def scale_for(period, idx):
        """Choose sampling resolution based on tiering rules."""
        year = int(period.id.split("-")[0])
        use_ultra = False
        if args.ultra_res_from and period.id >= args.ultra_res_from:
            use_ultra = True
        elif args.ultra_res_months > 0 and idx < args.ultra_res_months:
            use_ultra = True
        if use_ultra:
            return args.ultra_res_scale
        if args.hi_res_from is not None and year >= args.hi_res_from:
            return args.hi_res_scale
        return args.scale

    def process_region(boundary_path: Path, results_dir: Path, label: str, note: str) -> int:
        """Run monthly + yearly composites for a given geojson boundary.
        Used twice: once for PIO border, once for the forest mask."""
        if not boundary_path.exists():
            log(f"SKIP {label}: boundary {boundary_path} not found.")
            return 0
        results_dir.mkdir(parents=True, exist_ok=True)
        boundary = load_json(boundary_path)
        region = geojson_to_ee_geometry(boundary)
        region_bbox = compute_bbox(boundary)
        log(f"=== {label} === bbox: {region_bbox}")

        # Monthly
        # When --months-skip N is set, drop the first N most-recent candidates
        # so this batch only processes older months (non-overlapping backfill).
        skip = max(0, args.months_skip)
        all_candidates = latest_candidate_periods(args.months + skip, extra_back=4)
        target_candidates = all_candidates[skip:] if skip else all_candidates
        if skip:
            log(f"  --months-skip={skip}: skipping latest {skip} candidate months; processing window starts at {target_candidates[0].id if target_candidates else 'N/A'}")

        monthly_periods: List[Period] = []
        for p in target_candidates:
            if has_images(p, region):
                monthly_periods.append(p)
                log(f"  selected month: {p.id}")
            else:
                log(f"  skipped month (no images): {p.id}")
            if len(monthly_periods) >= args.months:
                break

        index_periods = []
        for i, period in enumerate(monthly_periods):
            scale = scale_for(period, i)
            log(f"  Processing {label} month {period.id} at {scale}m ...")
            try:
                result = build_result(period, region, region_bbox, scale, kind="monthly")
            except RuntimeError as exc:
                log(f"  WARN month {period.id}: {exc}")
                continue
            write_json(results_dir / f"{period.id}.json", result)
            index_periods.append({"id": period.id, "label": period.label, "kind": "monthly", "image_count": result["meta"]["image_count"], "scale_m": scale})
            log(f"  OK month {period.id}: pixels={len(result['pixels'])} images={result['meta']['image_count']}")

        # Yearly
        index_yearly: List[Dict[str, Any]] = []
        if args.yearly:
            for year_period in yearly_candidate_periods(args.yearly_from, args.yearly_to):
                if not has_images(year_period, region):
                    log(f"  Skipped year {year_period.id} (no images)")
                    continue
                log(f"  Processing {label} year {year_period.id} at {args.scale}m ...")
                try:
                    result = build_result(year_period, region, region_bbox, args.scale, kind="yearly")
                except RuntimeError as exc:
                    log(f"  WARN year {year_period.id}: {exc}")
                    continue
                write_json(results_dir / f"{year_period.id}.json", result)
                index_yearly.append({"id": year_period.id, "label": year_period.label, "kind": "yearly", "image_count": result["meta"]["image_count"]})
                log(f"  OK year {year_period.id}: pixels={len(result['pixels'])} images={result['meta']['image_count']}")

        # index.json
        index = {
            "ok": True,
            "label": label,
            "latest": index_periods[0]["id"] if index_periods else None,
            "periods": index_periods,
            "yearlyPeriods": index_yearly,
            "indices": [{"id": i["id"], "label": i["label"], "name_sr": i["name_sr"], "desc": i["desc"],
                          "min": i["min"], "max": i["max"], "palette": i["palette"]} for i in INDICES],
            "scale_m": args.scale,
            "source": "Copernicus Sentinel-2 SR Harmonized",
            "dataset": S2_COLLECTION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "coverage": {
                "count": len(index_periods),
                "first": index_periods[-1]["id"] if index_periods else None,
                "last": index_periods[0]["id"] if index_periods else None,
                "yearlyCount": len(index_yearly),
                "yearlyFirst": index_yearly[-1]["id"] if index_yearly else None,
                "yearlyLast": index_yearly[0]["id"] if index_yearly else None,
            },
            "note": note,
        }
        write_json(results_dir / "index.json", index)
        log(f"DONE {label}: {len(index_periods)} monthly + {len(index_yearly)} yearly periods written to {results_dir}")
        return len(index_periods) + len(index_yearly)

    total = 0
    pio_failed = False
    forests_failed = False

    if not args.forests_only:
        try:
            total += process_region(
                BOUNDARY, RESULTS, "PIO Rajac (full)",
                "Generated by GitHub Actions for PIO Rajac vegetation monitor. Sentinel-2 SR Harmonized — целокупна територија ПИО.",
            )
        except Exception as exc:
            log(f"FAIL PIO processing: {type(exc).__name__}: {exc}")
            pio_failed = True

    if args.forests or args.forests_only:
        try:
            total += process_region(
                FOREST_BOUNDARY, RESULTS_FORESTS, "PIO Rajac — само шумске површине",
                "Сателитски мониторинг ИСКЉУЧИВО шумских површина (~1.033 ha = 58,5% ПИО) — заснован на дигиталној карти шума ПИО „Рајац\" из 2024.",
            )
        except Exception as exc:
            # Forest failure must NOT break the PIO results — they are committed independently.
            log(f"FAIL forests processing — PIO results unaffected: {type(exc).__name__}: {exc}")
            forests_failed = True

    log(f"ALL DONE — total {total} periods written. (pio_failed={pio_failed}, forests_failed={forests_failed})")
    # Only fail the whole run if BOTH failed (or PIO failed when forests were not requested)
    if pio_failed and (forests_failed or not (args.forests or args.forests_only)):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
