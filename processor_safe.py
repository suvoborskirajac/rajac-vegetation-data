#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-safe runner for the Rajac vegetation processor.

This wrapper keeps the existing processor.py data model and CLI, but replaces
its tile sampling with a defensive implementation:

* every Earth Engine request is retried;
* a failing tile is recursively split into four smaller tiles;
* a tile that still fails aborts that period instead of being silently skipped;
* sampled-pixel count is checked against Earth Engine's NDVI pixel count so an
  incomplete map cannot be published as a successful result.

The goal is to prevent the large rectangular holes/artifacts that can appear
when a single high-resolution processing tile fails.
"""
from __future__ import annotations

import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ee

import processor as base

RETRY_ATTEMPTS = 3
MAX_SPLIT_DEPTH = 2
MIN_SAMPLE_COVERAGE = 0.88


def _retry_getinfo(obj: Any, label: str) -> Any:
    """Call getInfo() with bounded retries; raise instead of hiding failures."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return obj.getInfo()
        except Exception as exc:
            last_exc = exc
            base.log(
                f"  WARN {label}: attempt {attempt}/{RETRY_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{label}: all {RETRY_ATTEMPTS} attempts failed: {last_exc}")


def _area_m2(geom: ee.Geometry, label: str) -> float:
    value = _retry_getinfo(geom.area(1), f"{label} area")
    try:
        return float(value or 0.0)
    except Exception as exc:
        raise RuntimeError(
            f"{label}: invalid area returned by Earth Engine: {value!r}"
        ) from exc


def _sample_geometry(
    image: ee.Image,
    band_ids: List[str],
    geom: ee.Geometry,
    scale_m: float,
    label: str,
) -> List[Dict[str, Any]]:
    fc = image.select(band_ids).sample(
        region=geom,
        scale=scale_m,
        geometries=True,
        tileScale=4,
    )
    data = _retry_getinfo(fc, f"{label} sample")
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}: Earth Engine returned non-dict sample payload")
    features = data.get("features", [])
    if not isinstance(features, list):
        raise RuntimeError(f"{label}: Earth Engine payload has invalid features array")
    return base._features_to_pixels(features, band_ids, scale_m)


def _sample_bbox_recursive(
    image: ee.Image,
    band_ids: List[str],
    region: ee.Geometry,
    bbox: List[float],
    scale_m: float,
    label: str,
    depth: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Sample one bbox; subdivide on persistent EE failure.

    Returns (pixels, successful_leaf_tiles). Any leaf that still cannot be
    sampled raises RuntimeError, so a partially missing rectangle can never be
    accepted as a complete period.
    """
    tile_rect = ee.Geometry.Rectangle(bbox)
    tile_geom = tile_rect.intersection(region, 1)
    area = _area_m2(tile_geom, label)
    if area <= 0:
        return [], 0

    try:
        chunk = _sample_geometry(image, band_ids, tile_geom, scale_m, label)
        return chunk, 1
    except RuntimeError as exc:
        if depth >= MAX_SPLIT_DEPTH:
            raise RuntimeError(
                f"{label}: tile failed after retries and maximum subdivision; "
                "period is rejected to prevent an incomplete map"
            ) from exc

        base.log(
            f"  RETRY {label}: subdividing failed tile into 2x2 "
            f"(depth {depth + 1}/{MAX_SPLIT_DEPTH})"
        )
        pixels: List[Dict[str, Any]] = []
        leaves = 0
        for sub_i, sub_bbox in enumerate(base._tile_grid(bbox, 2, 2), 1):
            sub_pixels, sub_leaves = _sample_bbox_recursive(
                image,
                band_ids,
                region,
                sub_bbox,
                scale_m,
                f"{label}.{sub_i}",
                depth + 1,
            )
            pixels.extend(sub_pixels)
            leaves += sub_leaves
        return pixels, leaves


def sample_pixels_safe(
    image: ee.Image,
    region: ee.Geometry,
    region_bbox: List[float],
    scale_m: float,
) -> List[Dict[str, Any]]:
    """Fail-safe replacement for processor.sample_pixels."""
    band_ids = [idx["id"] for idx in base.INDICES]
    n_cols, n_rows = base._estimate_tile_count(
        region_bbox, scale_m, max_per_tile=4500
    )
    tiles = base._tile_grid(region_bbox, n_cols, n_rows)

    base.log(
        f"  Safe tiling at {scale_m}m: {n_cols}x{n_rows} = {len(tiles)} "
        f"top-level tiles; retries={RETRY_ATTEMPTS}, "
        f"split-depth={MAX_SPLIT_DEPTH}"
    )

    pixels: List[Dict[str, Any]] = []
    seen = set()
    intersecting_top = 0
    successful_leaves = 0

    for ti, tb in enumerate(tiles, 1):
        label = f"tile {ti}/{len(tiles)}"
        chunk, leaves = _sample_bbox_recursive(
            image, band_ids, region, tb, scale_m, label, depth=0
        )
        if leaves > 0:
            intersecting_top += 1
            successful_leaves += leaves
        for p in chunk:
            key = (p["lon"], p["lat"])
            if key in seen:
                continue
            seen.add(key)
            pixels.append(p)

    base.log(
        f"  Safe tiling complete: intersecting top-level tiles="
        f"{intersecting_top}, successful leaf samples={successful_leaves}, "
        f"unique pixels={len(pixels)}"
    )

    pixels.sort(key=lambda p: (p["lat"], p["lon"]))
    for i, p in enumerate(pixels, 1):
        p["id"] = f"px-{i:04d}"
    return pixels


_original_build_result = base.build_result


def build_result_safe(
    period: base.Period,
    region: ee.Geometry,
    region_bbox: List[float],
    scale_m: float,
    kind: str = "monthly",
) -> Dict[str, Any]:
    """Build a result and reject suspiciously incomplete sampling."""
    result = _original_build_result(
        period, region, region_bbox, scale_m, kind=kind
    )

    sampled = len(result.get("pixels") or [])
    stats = result.get("stats") or {}
    ndvi_count = int(((stats.get("ndvi") or {}).get("count")) or 0)

    if ndvi_count <= 0:
        raise RuntimeError(
            f"Period {period.id}: NDVI statistics report zero valid pixels; "
            "refusing to publish."
        )

    ratio = sampled / ndvi_count
    base.log(
        f"  Coverage check {period.id}: sampled={sampled}, "
        f"NDVI-count={ndvi_count}, ratio={ratio:.5f}"
    )

    if ratio < MIN_SAMPLE_COVERAGE:
        raise RuntimeError(
            f"Period {period.id}: sampled pixel coverage {ratio:.2%} is below "
            f"the required {MIN_SAMPLE_COVERAGE:.0%}; refusing to publish "
            "incomplete map."
        )

    result.setdefault("meta", {})["sampling_validation"] = {
        "sampled_pixels": sampled,
        "ndvi_count": ndvi_count,
        "coverage_ratio": round(ratio, 4),
        "minimum_required": MIN_SAMPLE_COVERAGE,
        "status": "ok",
    }
    return result


_MONTH_FILE_RE = re.compile(r"^\d{4}-\d{2}\.json$")
_YEAR_FILE_RE = re.compile(r"^\d{4}\.json$")


def _read_result_entry(path: Path, kind: str) -> Dict[str, Any] | None:
    """Read just enough metadata from one existing result JSON for index.json.

    Historical result files are the source of truth for dropdown availability.
    A partial/backfill run must never erase periods that were not processed in
    that particular invocation.
    """
    try:
        data = base.load_json(path)
    except Exception as exc:
        base.log(f"  WARN index rebuild: cannot read {path}: {exc}")
        return None

    if not isinstance(data, dict) or not data.get("ok", False):
        base.log(f"  WARN index rebuild: skipping invalid result {path}")
        return None

    meta = data.get("meta") or {}
    period_id = str(meta.get("id") or path.stem)
    label = str(meta.get("label") or (
        f"{period_id[5:7]}/{period_id[:4]}"
        if kind == "monthly" and len(period_id) == 7
        else f"{period_id}. (годишњи просек)"
    ))

    image_count = meta.get("image_count")
    try:
        image_count = int(image_count or 0)
    except Exception:
        image_count = 0

    entry: Dict[str, Any] = {
        "id": period_id,
        "label": label,
        "kind": kind,
        "image_count": image_count,
    }

    # Preserve the actual resolution of each historical period when present.
    scale = meta.get("scale_m")
    if scale is not None:
        try:
            entry["scale_m"] = float(scale)
        except Exception:
            pass

    return entry


def rebuild_complete_index(results_dir: Path) -> None:
    """Rebuild index.json from every valid result file already on disk.

    This makes partial runs (e.g. only June+July repair) safe: they can update
    selected JSON files, but cannot remove older months/years from the frontend
    dropdown merely because those periods were not reprocessed in the run.
    """
    results_dir = Path(results_dir)
    index_path = results_dir / "index.json"

    try:
        current = base.load_json(index_path) if index_path.exists() else {}
    except Exception:
        current = {}
    if not isinstance(current, dict):
        current = {}

    monthly: List[Dict[str, Any]] = []
    yearly: List[Dict[str, Any]] = []

    for path in results_dir.glob("*.json"):
        if path.name == "index.json":
            continue
        if _MONTH_FILE_RE.match(path.name):
            entry = _read_result_entry(path, "monthly")
            if entry:
                monthly.append(entry)
        elif _YEAR_FILE_RE.match(path.name):
            entry = _read_result_entry(path, "yearly")
            if entry:
                yearly.append(entry)

    # ISO-style ids sort chronologically as strings.
    monthly.sort(key=lambda x: x["id"], reverse=True)
    yearly.sort(key=lambda x: x["id"], reverse=True)

    # Deduplicate defensively by period id.
    def dedupe(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for entry in entries:
            pid = entry["id"]
            if pid in seen:
                continue
            seen.add(pid)
            out.append(entry)
        return out

    monthly = dedupe(monthly)
    yearly = dedupe(yearly)

    current["ok"] = True
    current["latest"] = monthly[0]["id"] if monthly else None
    current["periods"] = monthly
    current["yearlyPeriods"] = yearly
    current["updated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    current["coverage"] = {
        "count": len(monthly),
        "first": monthly[-1]["id"] if monthly else None,
        "last": monthly[0]["id"] if monthly else None,
        "yearlyCount": len(yearly),
        "yearlyFirst": yearly[-1]["id"] if yearly else None,
        "yearlyLast": yearly[0]["id"] if yearly else None,
    }

    base.write_json(index_path, current)
    base.log(
        f"INDEX RESTORED {results_dir}: "
        f"{len(monthly)} monthly + {len(yearly)} yearly periods"
    )


# Monkey-patch only the safety-critical functions. All existing file formats,
# statistics, period selection and forest handling remain in processor.py.
base.sample_pixels = sample_pixels_safe
base.build_result = build_result_safe


if __name__ == "__main__":
    try:
        rc = base.main()
        if rc == 0:
            # processor.py intentionally writes an index containing only the
            # periods handled by the current run. Restore the complete history
            # from all result JSONs before GitHub commits the output.
            rebuild_complete_index(base.RESULTS)
            rebuild_complete_index(base.RESULTS_FORESTS)
        raise SystemExit(rc)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
