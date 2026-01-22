

from __future__ import annotations

import argparse
import sys
import json
import os
import re
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image
import folium

try:
    from osgeo import gdal
except Exception as e:
    raise RuntimeError("GDAL import failed (from osgeo import gdal).") from e

# OpenAI for natural language queries (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Your Pydro/BlueTopo downloader hooks
try:
    from nbs.bluetopo import fetch_tiles, build_vrt
except Exception as e:
    raise RuntimeError(
        "Could not import nbs.bluetopo.fetch_tiles/build_vrt.\n"
        "This script expects the same environment as your BlueTopo Downloader UI.\n"
        f"Original error: {e!r}"
    ) from e

# Optional colormap support
try:
    import matplotlib.cm as cm  # type: ignore
except Exception:
    cm = None  # fallback handled below


# --- OpenAI Configuration (optional) ---
# If you don't want any cloud calls, simply DO NOT set OPENAI_API_KEY.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# --- Band mapping (per BlueTopo spec) ---
DEPTH_BAND = 1
UNCERTAINTY_BAND = 2
CONTRIB_BAND = 3

_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"overlay_meta": {"ready": False}, "last_error": None}


# ----------------------------
# Utilities
# ----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _in_docker() -> bool:
    # Common, simple check
    return os.getenv("DOCKER") == "1" or Path("/.dockerenv").exists()


def _extract_geometry(obj: dict[str, Any]) -> dict[str, Any]:
    t = obj.get("type")
    if t in ("Polygon", "MultiPolygon"):
        return obj
    if t == "Feature":
        return obj["geometry"]
    if t == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            raise ValueError("Empty FeatureCollection")
        return feats[0]["geometry"]
    raise ValueError(f"Unsupported GeoJSON type: {t}")


def _bbox_from_geometry(geom: dict[str, Any]) -> Tuple[float, float, float, float]:
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    lons: list[float] = []
    lats: list[float] = []

    def walk(x):
        if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], (int, float)):
            lons.append(float(x[0]))
            lats.append(float(x[1]))
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)

    if gtype not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"bbox: geometry must be Polygon/MultiPolygon, got {gtype}")

    walk(coords)
    if not lons:
        raise ValueError("bbox: no coordinates found")

    return (min(lons), min(lats), max(lons), max(lats))


def _write_geojson_feature(geom: dict[str, Any], out_path: Path) -> None:
    feat = {"type": "Feature", "properties": {}, "geometry": geom}
    fc = {"type": "FeatureCollection", "features": [feat]}
    out_path.write_text(json.dumps(fc), encoding="utf-8")


def _find_best_vrt(output_dir: Path) -> Path:
    vrts = list(output_dir.rglob("*.vrt"))
    if not vrts:
        raise FileNotFoundError(f"No .vrt found under {output_dir}")

    def score(p: Path) -> Tuple[int, float]:
        name = p.name.lower()
        s = 0
        if "modeling" in name:
            s += 10
        if "fetched" in name:
            s += 10
        if "utm" in name:
            s += 5
        if "modeling_vrt" in str(p).lower():
            s += 5
        return (s, p.stat().st_mtime)

    vrts.sort(key=score, reverse=True)
    return vrts[0]


def _pick_raster_shape(bounds: Tuple[float, float, float, float], max_dim: int = 1800) -> Tuple[int, int]:
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_span = max(1e-9, abs(max_lon - min_lon))
    lat_span = max(1e-9, abs(max_lat - min_lat))

    if lon_span >= lat_span:
        width = max_dim
        height = max(64, int(max_dim * (lat_span / lon_span)))
    else:
        height = max_dim
        width = max(64, int(max_dim * (lon_span / lat_span)))
    return width, height


def _warp_aoi(
    vrt_path: Path,
    cutline_geojson: Path,
    width: int,
    height: int,
    resample_alg: int,
) -> Any:
    warp_opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        format="MEM",
        cutlineDSName=str(cutline_geojson),
        cropToCutline=True,
        resampleAlg=resample_alg,
        width=width,
        height=height,
        multithread=True,
    )
    ds = gdal.Warp("", str(vrt_path), options=warp_opts)
    if ds is None:
        raise RuntimeError("GDAL Warp returned None (check VRT path / cutline).")
    return ds


def _overlay_bounds_from_ds(ds: Any) -> list[list[float]]:
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    min_lon = gt[0]
    max_lon = gt[0] + (cols * gt[1])
    max_lat = gt[3]
    min_lat = gt[3] + (rows * gt[5])  # gt[5] negative
    return [[min_lat, min_lon], [max_lat, max_lon]]


def _band_as_array(ds: Any, band_index: int) -> Tuple[np.ndarray, Optional[float]]:
    if band_index < 1 or band_index > ds.RasterCount:
        raise RuntimeError(f"Requested band {band_index}, but dataset has {ds.RasterCount} bands.")
    b = ds.GetRasterBand(band_index)
    arr = b.ReadAsArray()
    nodata = b.GetNoDataValue()
    return arr, nodata


def _valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    m = np.isfinite(arr)
    if nodata is not None:
        m &= arr != nodata
    return m


def _hex_to_rgba(hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
    s = hex_color.strip()
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", s):
        # fallback orange
        s = "#ff9500"
    if not s.startswith("#"):
        s = "#" + s
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return r, g, b, alpha


def _save_rgba(rgba: np.ndarray, out_png: Path) -> None:
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(out_png)


def _get_colors(n: int, cmap_name: str) -> list[str]:
    """Returns n hex colors."""
    if n <= 0:
        return []
    if cm is None:
        # fallback palette
        palette = [
            "#440154", "#3b528b", "#21918c", "#5ec962", "#fde725",
            "#482878", "#2c728e", "#28ae80", "#b5de2b", "#f9e721",
        ]
        return [palette[i % len(palette)] for i in range(n)]

    try:
        cmap = cm.get_cmap(cmap_name, n)
    except Exception:
        cmap = cm.get_cmap("viridis", n)

    # sample in [0,1]
    colors = []
    for i in range(n):
        r, g, b, _a = cmap(i / max(1, (n - 1)))
        colors.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return colors


# ----------------------------
# RAT schema + filtering
# ----------------------------
_GFT_INT = getattr(gdal, "GFT_Integer", 0)
_GFT_REAL = getattr(gdal, "GFT_Real", 1)
_GFT_STR = getattr(gdal, "GFT_String", 2)


def _rat_schema_from_vrt(vrt_path: Path) -> dict[str, Any]:
    ds = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open VRT: {vrt_path}")

    if ds.RasterCount < CONTRIB_BAND:
        raise RuntimeError(f"VRT has {ds.RasterCount} bands; expected at least {CONTRIB_BAND} for contributors/RAT.")

    band = ds.GetRasterBand(CONTRIB_BAND)
    rat = band.GetDefaultRAT()
    if rat is None:
        raise RuntimeError("No default RAT found on contributor band.")

    nrows = rat.GetRowCount()
    ncols = rat.GetColumnCount()

    cols = []
    # We'll sample a small number of rows to infer bool-ish columns
    sample_rows = min(nrows, 500)

    for c in range(ncols):
        name = rat.GetNameOfCol(c)
        ctype = rat.GetTypeOfCol(c)

        # infer "kind" used by UI
        if ctype == _GFT_STR:
            kind = "string"
        elif ctype in (_GFT_INT, _GFT_REAL):
            kind = "number"
        else:
            kind = "string"

        # bool heuristic: numeric with only {0,1} in sample
        if kind == "number":
            vals = set()
            for r in range(sample_rows):
                try:
                    v = rat.GetValueAsDouble(r, c)
                except Exception:
                    continue
                if not np.isfinite(v):
                    continue
                if len(vals) < 5:
                    vals.add(int(v))
            if vals.issubset({0, 1}) and len(vals) > 0:
                kind = "bool"

        # date heuristic: string field with "date" in the name
        if kind == "string" and "date" in name.lower():
            kind = "date"

        cols.append({"name": name, "gdal_type": int(ctype), "kind": kind})

    return {
        "vrt": str(vrt_path),
        "raster_count": ds.RasterCount,
        "rat_rows": nrows,
        "columns": cols,
        "note": "Column definitions are read from the Raster Attribute Table attached to band 3.",
    }


def _rat_value_to_row_map(rat: Any, value_col: Optional[int]) -> dict[int, int]:
    """
    Returns dict: contributor_id_int -> rat_row_index.
    BlueTopo spec includes a 'value' column listing unique cell values.
    If value column exists, we map via it; otherwise assume row index corresponds to value.
    """
    nrows = rat.GetRowCount()
    m: dict[int, int] = {}
    if value_col is None:
        for r in range(nrows):
            m[r] = r
        return m

    for r in range(nrows):
        v = rat.GetValueAsDouble(r, value_col)
        if not np.isfinite(v):
            continue
        m[int(round(v))] = r
    return m


def _rat_find_col(rat: Any, name: str) -> Optional[int]:
    ncols = rat.GetColumnCount()
    for c in range(ncols):
        if (rat.GetNameOfCol(c) or "").strip().lower() == name.strip().lower():
            return c
    return None


def _rat_eval_condition(
    *,
    rat: Any,
    id_to_row: dict[int, int],
    field: str,
    op: str,
    raw_value: str,
    ids_present: np.ndarray,
) -> np.ndarray:
    """
    Returns the subset of ids_present that satisfy the RAT condition.
    ids_present: unique int IDs.
    """
    col = _rat_find_col(rat, field)
    if col is None:
        raise RuntimeError(f"RAT field not found: {field}")

    ctype = rat.GetTypeOfCol(col)

    def get_val(row: int):
        if ctype == _GFT_STR:
            return rat.GetValueAsString(row, col)
        elif ctype == _GFT_INT:
            return rat.GetValueAsInt(row, col)
        else:
            return rat.GetValueAsDouble(row, col)

    keep = []
    # normalize op
    op = op.strip().lower()

    # Check if this is a date field based on field name (date fields are stored as strings)
    is_date_field = ctype == _GFT_STR and "date" in field.lower()

    # Handle date-specific operators
    if is_date_field and op in ("after", "before", "in_year"):
        v = (raw_value or "").strip()

        # Parse the input value - support year-only (e.g., "2021") or full date (e.g., "2021-01-01")
        if v.isdigit() and len(v) == 4:
            # Year-only input
            year = v
            if op == "after":
                # After year means >= YYYY-01-01
                compare_date = f"{year}-01-01"
            elif op == "before":
                # Before year means < YYYY-01-01
                compare_date = f"{year}-01-01"
            elif op == "in_year":
                # In year means >= YYYY-01-01 and < (YYYY+1)-01-01
                year_start = f"{year}-01-01"
                year_end = f"{int(year)+1}-01-01"
        else:
            # Assume full date format
            compare_date = v

        for cid in ids_present.tolist():
            row = id_to_row.get(int(cid))
            if row is None:
                continue
            s = str(get_val(row) or "").strip()
            if not s:
                continue

            # Perform date comparison (lexicographic comparison works for ISO date strings)
            if op == "after":
                if s >= compare_date:
                    keep.append(int(cid))
            elif op == "before":
                if s < compare_date:
                    keep.append(int(cid))
            elif op == "in_year":
                if s >= year_start and s < year_end:
                    keep.append(int(cid))

    # parse value depending on op/type
    elif ctype == _GFT_STR:
        v = (raw_value or "").strip()
        for cid in ids_present.tolist():
            row = id_to_row.get(int(cid))
            if row is None:
                continue
            s = str(get_val(row) or "")
            if op == "equals":
                if s == v:
                    keep.append(int(cid))
            elif op == "contains":
                if v.lower() in s.lower():
                    keep.append(int(cid))
            else:
                # default to contains
                if v.lower() in s.lower():
                    keep.append(int(cid))

    else:
        # numeric/bool
        # support: =, <=, >=, <, >
        vtxt = (raw_value or "").strip()
        try:
            vnum = float(vtxt) if vtxt != "" else 0.0
        except Exception:
            vnum = 0.0

        for cid in ids_present.tolist():
            row = id_to_row.get(int(cid))
            if row is None:
                continue
            x = float(get_val(row))
            if not np.isfinite(x):
                continue

            if op in ("=", "==", "equals"):
                ok = x == vnum
            elif op in ("<", "lt"):
                ok = x < vnum
            elif op in ("<=", "lte"):
                ok = x <= vnum
            elif op in (">", "gt"):
                ok = x > vnum
            elif op in (">=", "gte"):
                ok = x >= vnum
            else:
                ok = x == vnum

            if ok:
                keep.append(int(cid))

    return np.array(sorted(set(keep)), dtype=np.int64)


def _process_natural_language_query(query: str, rat_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Uses OpenAI to convert a natural language query into structured RAT filter conditions.

    Returns a dict with format:
    {
        "combiner": "AND" | "OR",
        "conditions": [{"field": str, "op": str, "value": str}, ...]
    }
    or {"error": str} if processing fails.
    """
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI library not installed. Run: pip install openai"}

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        return {"error": "Please set your OpenAI API key in OPENAI_API_KEY variable"}

    # Build context about available fields
    columns = rat_schema.get("columns", [])
    field_descriptions = []
    for col in columns:
        kind = col.get("kind", "string")
        name = col.get("name", "")

        if kind == "date":
            ops = "after (>=), before (<), in_year, equals"
        elif kind == "string":
            ops = "contains, equals"
        elif kind == "bool":
            ops = "equals (use 1 for true, 0 for false)"
        elif kind == "number":
            ops = "=, <=, >=, <, >"
        else:
            ops = "contains, equals"

        field_descriptions.append(f"  - {name} ({kind}): operators = {ops}")

    fields_text = "\n".join(field_descriptions)

    system_prompt = f"""You are a helpful assistant that translates natural language queries about bathymetry data into structured filter conditions.

Available fields and their types:
{fields_text}

IMPORTANT - BlueTopo Field Value Meanings (from official specs):

**bathy_coverage** (boolean: 0 or 1):
  - Value "1" (True): Depth is from MEASURED data (not interpolated)
    User queries: "no interpolation", "measured data", "actual measurements", "not interpolated"
  - Value "0" (False): Depth is INTERPOLATED (no depth measurement achieved)
    User queries: "interpolated", "filled in", "estimated"

**coverage** (boolean: 0 or 1):
  - Value "1" (True): Full seafloor coverage achieved in survey area
    User queries: "full coverage", "complete coverage", "fully covered"
  - Value "0" (False): Incomplete coverage (partial survey)
    User queries: "partial coverage", "incomplete", "gaps"

**significant_features** (boolean: 0 or 1):
  - Value "1" (True): Systematic method of exploring seafloor features was undertaken
    Features = objects projecting above seafloor that may endanger surface navigation
    User queries: "feature detection", "systematic survey", "hazard detection"
  - Value "0" (False): No systematic feature detection performed
    User queries: "no feature detection", "basic survey"

**data_assessment** (number: 1 or 3):
  - Value "1": Data has been assessed for quality
    User queries: "assessed", "quality checked", "verified"
  - Value "3": Data is unassessed
    User queries: "unassessed", "not verified", "unchecked"

**Interpolation context**: The elevation layer contains both measured bathymetry and interpolated values.
Some interpolation is supported by side-scan methods, some is "best guess" estimates.

Your task is to convert the user's query into a JSON object with this exact structure:
{{
  "combiner": "AND" or "OR",
  "conditions": [
    {{"field": "field_name", "op": "operator", "value": "value_string"}},
    ...
  ]
}}

Rules:
1. Use "AND" when all conditions must be met, "OR" when any condition can match
2. For date fields like survey_date_end:
   - Use "after" for ">=" semantics (e.g., "from 2020" or "since 2020" → after 2020)
   - Use "before" for "<" semantics (e.g., "before 2020" → before 2020)
   - Use "in_year" for a specific year (e.g., "in 2020" → in_year 2020)
3. For string fields, use "contains" for partial matches, "equals" for exact matches
4. For boolean fields, use value "1" for true/yes and "0" for false/no
5. Values should be strings (for dates, use year like "2020" or full date like "2020-01-01")
6. Field names must exactly match the available fields listed above
7. If the query is unclear or cannot be translated, return {{"error": "explanation"}}

Examples:
- "NOAA surveys from 2020 or newer" → {{"combiner": "AND", "conditions": [{{"field": "source_institution", "op": "contains", "value": "NOAA"}}, {{"field": "survey_date_end", "op": "after", "value": "2020"}}]}}
- "NOAA or USGS data" → {{"combiner": "OR", "conditions": [{{"field": "source_institution", "op": "contains", "value": "NOAA"}}, {{"field": "source_institution", "op": "contains", "value": "USGS"}}]}}
- "surveys from 2015 to 2019" → {{"combiner": "AND", "conditions": [{{"field": "survey_date_end", "op": "after", "value": "2015"}}, {{"field": "survey_date_end", "op": "before", "value": "2020"}}]}}
- "show me NOAA surveys where there is no interpolation" → {{"combiner": "AND", "conditions": [{{"field": "source_institution", "op": "contains", "value": "NOAA"}}, {{"field": "bathy_coverage", "op": "=", "value": "1"}}]}}
- "measured data with full coverage" → {{"combiner": "AND", "conditions": [{{"field": "bathy_coverage", "op": "=", "value": "1"}}, {{"field": "coverage", "op": "=", "value": "1"}}]}}
- "assessed data only" → {{"combiner": "AND", "conditions": [{{"field": "data_assessment", "op": "=", "value": "1"}}]}}
"""

    try:

        if not OPENAI_API_KEY:
            return {"error": "OpenAI is disabled (OPENAI_API_KEY not set)."}
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        if not result_text:
            return {"error": "Empty response from OpenAI"}

        result = json.loads(result_text)

        # Validate the response structure
        if "error" in result:
            return result

        if "combiner" not in result or "conditions" not in result:
            return {"error": "Invalid response format from LLM"}

        if not isinstance(result["conditions"], list):
            return {"error": "Conditions must be a list"}

        # Validate each condition
        for cond in result["conditions"]:
            if not all(k in cond for k in ["field", "op", "value"]):
                return {"error": "Each condition must have field, op, and value"}

        return result

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM response as JSON: {e}"}
    except Exception as e:
        return {"error": f"OpenAI API error: {e}"}


def _process_chat_assistant(query: str, rat_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Enhanced chat assistant that can control ALL visualization parameters.

    Returns:
    {
        "explanation": str,  # User-friendly explanation
        "plan": str,  # HTML execution plan
        "mode": str,  # Visualization mode
        "depth_threshold_m": float,  # Calculated depth threshold
        "rat_filter": {combiner, conditions},  # RAT filters
        "execute": bool  # Whether to auto-execute
    }
    """
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI library not installed. Run: pip install openai"}

    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        return {"error": "Please set your OpenAI API key in OPENAI_API_KEY variable"}

    # Get current year for date calculations
    import datetime
    current_year = datetime.datetime.now().year

    # Build RAT fields context
    columns = rat_schema.get("columns", [])
    field_descriptions = []
    for col in columns:
        kind = col.get("kind", "string")
        name = col.get("name", "")
        if kind == "date":
            ops = "after (>=), before (<), in_year"
        elif kind == "string":
            ops = "contains, equals"
        elif kind == "bool":
            ops = "= (use 1/0)"
        else:
            ops = "=, <=, >=, <, >"
        field_descriptions.append(f"  - {name} ({kind}): {ops}")

    fields_text = "\n".join(field_descriptions)

    system_prompt = f"""You are an expert bathymetry visualization assistant. You help mariners and researchers visualize bathymetry data based on natural language requests.

Current year: {current_year}

Available RAT fields:
{fields_text}

BlueTopo Field Meanings:
- bathy_coverage=1: Measured data (no interpolation)
- coverage=1: Full seafloor coverage
- significant_features=1: Systematic feature detection performed
- data_assessment=1: Quality assessed

Visualization modes available:
- "depth_mask": Show areas deeper than threshold (use for navigation safety or general area viewing)
- "depth_ramp": Grayscale depth visualization
- "depth_classes": Color-coded depth bins

Navigation terminology:
- Draft: Ship's depth below waterline
- UKC (Under Keel Clearance): Safety margin below keel
- Minimum safe depth = Draft + UKC
- All depths in MLLW (Mean Lower Low Water)

Unit conversions:
- 1 foot = 0.3048 meters
- 1 fathom = 1.8288 meters

Date calculations:
- "past X years" = survey_date_end after {current_year - 10} (example for 10 years)
- "last decade" = survey_date_end after {current_year - 10}
- "since YYYY" = survey_date_end after YYYY

Your task: Convert the user's request into a complete visualization configuration.

Return JSON with this structure:
{{
  "explanation": "User-friendly explanation of what you understood",
  "plan": "HTML-formatted execution plan (use <br> for newlines, <strong> for emphasis)",
  "mode": "depth_mask|depth_ramp|depth_classes",
  "depth_threshold_m": float (POSITIVE number in meters, or 0 for no depth filtering),
  "depth_op": "deeper|shallower|between" (default: "deeper"),
  "depth_min_m": float (only for "between" operator),
  "depth_max_m": float (only for "between" operator),
  "rat_filter": {{
    "combiner": "AND"|"OR",
    "conditions": [{{"field": "...", "op": "...", "value": "..."}}, ...]
  }} or null,
  "execute": true
}}

Important calculation rules:
1. For depth filtering operators:
   - "deeper": Show areas deeper than threshold (depth >= threshold)
     Example: "deeper than 10m" → depth_op="deeper", depth_threshold_m=10
   - "shallower": Show areas shallower than threshold (depth < threshold)
     Example: "shallower than 20 feet" → depth_op="shallower", depth_threshold_m=6.1
   - "between": Show areas with depth between min and max
     Example: "between 5 and 15 meters" → depth_op="between", depth_min_m=5, depth_max_m=15
   - Default to "deeper" for navigation safety queries

2. For navigation safety queries (draft + UKC):
   - Calculate: depth_threshold_m = (draft_ft + ukc_ft) * 0.3048
   - Use depth_op="deeper" and mode="depth_mask" to show safe areas
   - Depth threshold should be POSITIVE (e.g., 2.13 for 7 ft)

3. For queries WITHOUT depth requirements:
   - Set depth_threshold_m to 0 and depth_op="deeper" (this shows all depths)
   - Example: "show USACE surveys" → depth_threshold_m=0, depth_op="deeper"

4. For date queries:
   - "past 10 years" → survey_date_end after "{current_year - 10}"
   - "last 5 years" → survey_date_end after "{current_year - 5}"

5. For data quality:
   - "no interpolation" → bathy_coverage = 1
   - "measured data" → bathy_coverage = 1
   - "full coverage" → coverage = 1

6. Default visualization settings (automatically applied):
   - Area of Interest (AOI): Viewport (screen extents) - uses current map view
   - Mask color: Red (#ff0000)
   - Opacity: 0.75
   - You do not need to specify these in your response

Examples:

Query: "I'm a ship with a 5 foot draft. I only like to transit areas with an Under Keel Clearance of at least 2 feet, and also in areas that have been surveyed in the past 10 years."

Response:
{{
  "explanation": "I'll show you areas safe for your vessel (5 ft draft + 2 ft UKC = 7 ft = 2.13 m minimum depth) that have been surveyed in the last 10 years.",
  "plan": "<strong>Navigation Safety Calculation:</strong><br>• Draft: 5 feet<br>• UKC: 2 feet<br>• Minimum safe depth: 7 feet = 2.13 meters<br><br><strong>Filters Applied:</strong><br>• Depth mask: Areas deeper than 2.13m<br>• Survey date: After {current_year - 10} (past 10 years)",
  "mode": "depth_mask",
  "depth_threshold_m": 2.13,
  "depth_op": "deeper",
  "rat_filter": {{
    "combiner": "AND",
    "conditions": [
      {{"field": "survey_date_end", "op": "after", "value": "{current_year - 10}"}}
    ]
  }},
  "execute": true
}}

Query: "Show me NOAA surveys with no interpolation from 2020 or newer"

Response:
{{
  "explanation": "Showing NOAA surveys with measured data (no interpolation) from 2020 onwards.",
  "plan": "<strong>Filters:</strong><br>• Source: NOAA<br>• Data type: Measured (no interpolation)<br>• Survey date: 2020 or newer",
  "mode": "depth_mask",
  "depth_threshold_m": 0,
  "depth_op": "deeper",
  "rat_filter": {{
    "combiner": "AND",
    "conditions": [
      {{"field": "source_institution", "op": "contains", "value": "NOAA"}},
      {{"field": "bathy_coverage", "op": "=", "value": "1"}},
      {{"field": "survey_date_end", "op": "after", "value": "2020"}}
    ]
  }},
  "execute": true
}}

Query: "show me areas just surveyed by the USACE in the past 5 years"

Response:
{{
  "explanation": "Showing areas surveyed by USACE (U.S. Army Corps of Engineers) in the past 5 years.",
  "plan": "<strong>Filters:</strong><br>• Source: USACE<br>• Survey date: After {current_year - 5} (past 5 years)",
  "mode": "depth_mask",
  "depth_threshold_m": 0,
  "depth_op": "deeper",
  "rat_filter": {{
    "combiner": "AND",
    "conditions": [
      {{"field": "source_institution", "op": "contains", "value": "USACE"}},
      {{"field": "survey_date_end", "op": "after", "value": "{current_year - 5}"}}
    ]
  }},
  "execute": true
}}

Query: "show me depth in areas shallower than 20 feet, with no interpolation"

Response:
{{
  "explanation": "Showing areas shallower than 20 feet (6.1 meters) using measured data (no interpolation).",
  "plan": "<strong>Depth Filter:</strong><br>• Shallower than 20 feet (6.1 meters)<br><br><strong>Data Quality:</strong><br>• Measured data only (no interpolation)",
  "mode": "depth_mask",
  "depth_threshold_m": 6.1,
  "depth_op": "shallower",
  "rat_filter": {{
    "combiner": "AND",
    "conditions": [
      {{"field": "bathy_coverage", "op": "=", "value": "1"}}
    ]
  }},
  "execute": true
}}
"""

    try:

        if not OPENAI_API_KEY:
            return {"error": "OpenAI is disabled (OPENAI_API_KEY not set)."}
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = response.choices[0].message.content
        if not result_text:
            return {"error": "Empty response from OpenAI"}

        result = json.loads(result_text)

        if "error" in result:
            return result

        # Validate and set defaults
        if "execute" not in result:
            result["execute"] = True

        return result

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM response as JSON: {e}"}
    except Exception as e:
        return {"error": f"OpenAI API error: {e}"}


# ----------------------------
# Overlay builders
# ----------------------------
def _apply_alpha_where(rgba: np.ndarray, keep: np.ndarray, alpha: int) -> np.ndarray:
    out = rgba.copy()
    out[..., 3] = 0
    out[keep, 3] = alpha
    return out


def _depth_mask_rgba(depth_m: np.ndarray, keep: np.ndarray, mask_hex: str, alpha: int) -> np.ndarray:
    rgba = np.zeros((depth_m.shape[0], depth_m.shape[1], 4), dtype=np.uint8)
    r, g, b, a = _hex_to_rgba(mask_hex, alpha)
    rgba[keep] = [r, g, b, a]
    return rgba


def _depth_ramp_rgba(depth_m: np.ndarray, keep: np.ndarray, alpha: int) -> np.ndarray:
    # Ramp on positive depth (meters)
    depth_pos = -depth_m
    vals = depth_pos[keep]
    p2 = float(np.percentile(vals, 2))
    p98 = float(np.percentile(vals, 98))
    if p98 <= p2:
        p2, p98 = float(vals.min()), float(vals.max())
    scaled = np.clip((depth_pos - p2) / (p98 - p2 + 1e-12), 0.0, 1.0)
    # deeper -> darker (scaled high => deeper) -> use (1-scaled) for light shallow
    img = (1.0 - scaled) * 255.0
    g = img.astype(np.uint8)

    rgba = np.zeros((depth_m.shape[0], depth_m.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = g
    rgba[..., 1] = g
    rgba[..., 2] = g
    rgba[..., 3] = 0
    rgba[keep, 3] = alpha
    return rgba


def _depth_classes_rgba(
    depth_m: np.ndarray,
    keep: np.ndarray,
    bins_m: list[float],
    cmap_name: str,
    alpha: int,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    bins_m: edge list (positive depths, meters): e.g. [0, 5, 10, 20, 35]
    produces classes:
      [b0-b1), [b1-b2), ... [b_{k-2}-b_{k-1}), >= b_{k-1}
    """
    if len(bins_m) < 2:
        bins_m = [0.0, 10.0, 20.0]

    # sanitize bins
    bins = sorted(set(float(x) for x in bins_m))
    if bins[0] > 0:
        bins = [0.0] + bins

    n_classes = max(1, len(bins) - 1) + 1  # +1 for >= last
    colors = _get_colors(n_classes, cmap_name=cmap_name)

    depth_pos = -depth_m

    rgba = np.zeros((depth_m.shape[0], depth_m.shape[1], 4), dtype=np.uint8)

    legend_items = []

    # intervals
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = keep & (depth_pos >= lo) & (depth_pos < hi)
        r, g, b, a = _hex_to_rgba(colors[i], alpha)
        rgba[m] = [r, g, b, a]
        legend_items.append({"label": f"{lo:g}–{hi:g} m", "color": colors[i]})

    # last bin: >= last edge
    last = bins[-1]
    m = keep & (depth_pos >= last)
    r, g, b, a = _hex_to_rgba(colors[-1], alpha)
    rgba[m] = [r, g, b, a]
    legend_items.append({"label": f">= {last:g} m", "color": colors[-1]})

    legend = {"title": "Depth classes (m)", "items": legend_items}
    return rgba, legend


def _contributors_random_rgba(contrib: np.ndarray, keep: np.ndarray, alpha: int) -> np.ndarray:
    # deterministic random-like color per contributor ID (cast to int)
    ids = np.zeros_like(contrib, dtype=np.int64)
    ids[keep] = np.round(contrib[keep]).astype(np.int64)

    h = (ids * 1103515245 + 12345) & 0x7fffffff
    r = (h & 255).astype(np.uint8)
    g = ((h >> 8) & 255).astype(np.uint8)
    b = ((h >> 16) & 255).astype(np.uint8)

    rgba = np.zeros((contrib.shape[0], contrib.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = 0
    rgba[keep, 3] = alpha
    return rgba


# ----------------------------
# Core compute
# ----------------------------
def _compute_overlay(
    vrt_path: Path,
    cutline_geojson: Path,
    overlay_png: Path,
    overlay_meta_json: Path,
    *,
    mode: str,
    depth_threshold_m: float,
    depth_op: str,
    depth_min_m: Optional[float],
    depth_max_m: Optional[float],
    depth_bins_m: list[float],
    cmap_name: str,
    mask_hex: str,
    opacity: float,
    max_uncertainty_m: Optional[float],
    rat_filter: Optional[dict[str, str]],
    max_dim: int = 1800,
) -> dict[str, Any]:
    geom_fc = json.loads(cutline_geojson.read_text(encoding="utf-8"))
    geom = _extract_geometry(geom_fc)
    bounds = _bbox_from_geometry(geom)
    width, height = _pick_raster_shape(bounds, max_dim=max_dim)

    # Continuous warp (depth + uncertainty): bilinear
    ds_cont = _warp_aoi(vrt_path, cutline_geojson, width, height, resample_alg=gdal.GRA_Bilinear)
    bounds_ll = _overlay_bounds_from_ds(ds_cont)

    depth_m, depth_nodata = _band_as_array(ds_cont, DEPTH_BAND)
    unc, unc_nodata = _band_as_array(ds_cont, UNCERTAINTY_BAND) if ds_cont.RasterCount >= UNCERTAINTY_BAND else (None, None)

    valid = _valid_mask(depth_m, depth_nodata)
    # land values are positive (per spec); treat as invalid for "depth" visualizations
    valid &= depth_m <= 0

    if unc is not None:
        valid_unc = _valid_mask(unc, unc_nodata)
        valid &= valid_unc

    # optional uncertainty filter
    if max_uncertainty_m is not None and unc is not None:
        valid &= (unc <= max_uncertainty_m)

    # Optional RAT filter requires contributors band; warp with nearest neighbor to preserve IDs
    rat_ids_keep: Optional[np.ndarray] = None
    if rat_filter is not None:
        ds_disc = _warp_aoi(vrt_path, cutline_geojson, width, height, resample_alg=gdal.GRA_NearestNeighbour)
        if ds_disc.RasterXSize != ds_cont.RasterXSize or ds_disc.RasterYSize != ds_cont.RasterYSize:
            raise RuntimeError("Contributor warp shape mismatch; please report this AOI (grid mismatch).")
        contrib, contrib_nodata = _band_as_array(ds_disc, CONTRIB_BAND)
        valid_contrib = _valid_mask(contrib, contrib_nodata)

        ids = np.round(contrib).astype(np.int64)
        ids_present = np.unique(ids[valid & valid_contrib])

        # Load RAT from the VRT (not from the warped MEM dataset)
        src = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
        if src is None:
            raise RuntimeError("Could not open VRT for RAT filtering.")
        b = src.GetRasterBand(CONTRIB_BAND)
        rat = b.GetDefaultRAT()
        if rat is None:
            raise RuntimeError("No RAT found on band 3 (contributors).")

        value_col = _rat_find_col(rat, "value")
        id_to_row = _rat_value_to_row_map(rat, value_col)

        # Handle multi-condition filtering with AND/OR
        combiner = rat_filter.get("combiner", "AND")
        conditions = rat_filter.get("conditions", [])

        if len(conditions) == 0:
            # No conditions, skip filtering
            pass
        elif len(conditions) == 1:
            # Single condition - simple case
            cond = conditions[0]
            keep_ids = _rat_eval_condition(
                rat=rat,
                id_to_row=id_to_row,
                field=cond.get("field", ""),
                op=cond.get("op", "contains"),
                raw_value=cond.get("value", ""),
                ids_present=ids_present,
            )
            rat_ids_keep = np.isin(ids, keep_ids) & valid_contrib
            valid &= rat_ids_keep
        else:
            # Multiple conditions - combine with AND/OR
            all_keep_ids = []
            for cond in conditions:
                keep_ids = _rat_eval_condition(
                    rat=rat,
                    id_to_row=id_to_row,
                    field=cond.get("field", ""),
                    op=cond.get("op", "contains"),
                    raw_value=cond.get("value", ""),
                    ids_present=ids_present,
                )
                all_keep_ids.append(set(keep_ids.tolist()))

            # Combine results based on combiner
            if combiner == "OR":
                # Union: any condition matches
                final_keep_ids = set()
                for keep_set in all_keep_ids:
                    final_keep_ids |= keep_set
            else:  # AND
                # Intersection: all conditions must match
                final_keep_ids = all_keep_ids[0]
                for keep_set in all_keep_ids[1:]:
                    final_keep_ids &= keep_set

            final_keep_ids_arr = np.array(sorted(final_keep_ids), dtype=np.int64)
            rat_ids_keep = np.isin(ids, final_keep_ids_arr) & valid_contrib
            valid &= rat_ids_keep

    # Alpha from opacity
    alpha = int(max(0, min(1, float(opacity))) * 255)

    legend: Optional[dict[str, Any]] = None

    if mode == "depth_mask":
        # Apply depth filtering based on operator
        if depth_op == "shallower":
            # Show areas shallower than threshold (depth_m is negative, so > -threshold)
            keep = valid & (depth_m > -abs(float(depth_threshold_m)))
            legend = {"title": "Mask", "items": [{"label": f"Depth < {abs(float(depth_threshold_m)):g} m", "color": mask_hex}]}
        elif depth_op == "between":
            # Show areas between min and max depth
            min_val = abs(float(depth_min_m)) if depth_min_m is not None else 0
            max_val = abs(float(depth_max_m)) if depth_max_m is not None else float('inf')
            keep = valid & (depth_m < -min_val) & (depth_m > -max_val)
            legend = {"title": "Mask", "items": [{"label": f"Depth {min_val:g}-{max_val:g} m", "color": mask_hex}]}
        else:  # "deeper" (default)
            # Show areas deeper than threshold (original behavior)
            keep = valid & (depth_m < -abs(float(depth_threshold_m)))
            legend = {"title": "Mask", "items": [{"label": f"Depth >= {abs(float(depth_threshold_m)):g} m", "color": mask_hex}]}

        rgba = _depth_mask_rgba(depth_m, keep=keep, mask_hex=mask_hex, alpha=alpha)

    elif mode == "depth_ramp":
        rgba = _depth_ramp_rgba(depth_m, keep=valid, alpha=alpha)
        legend = {"title": "Depth ramp", "items": [{"label": "grayscale (shallow→deep)", "color": "#888888"}]}

    elif mode == "depth_classes":
        rgba, legend = _depth_classes_rgba(depth_m, keep=valid, bins_m=depth_bins_m, cmap_name=cmap_name, alpha=alpha)

    elif mode == "contributors_random":
        # contributor visualization only, but still respects uncertainty + RAT filter if enabled
        ds_disc = _warp_aoi(vrt_path, cutline_geojson, width, height, resample_alg=gdal.GRA_NearestNeighbour)
        contrib, contrib_nodata = _band_as_array(ds_disc, CONTRIB_BAND)
        valid_contrib = _valid_mask(contrib, contrib_nodata)
        keep = valid & valid_contrib
        rgba = _contributors_random_rgba(contrib=contrib, keep=keep, alpha=alpha)
        legend = {"title": "Contributors", "items": [{"label": "random color per contributor ID", "color": "#777777"}]}

    else:
        raise ValueError(f"Unknown mode: {mode}")

    _save_rgba(rgba, overlay_png)

    meta = {
        "ready": True,
        "bounds": bounds_ll,
        "url": "/overlay.png",
        "opacity": float(opacity),
        "mode": mode,
        "generated_at": time.time(),
        "bands": {"depth": DEPTH_BAND, "uncertainty": UNCERTAINTY_BAND, "contributors": CONTRIB_BAND, "raster_count": ds_cont.RasterCount},
        "params": {
            "depth_threshold_m": depth_threshold_m,
            "depth_op": depth_op,
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "depth_bins_m": depth_bins_m,
            "cmap": cmap_name,
            "mask_hex": mask_hex,
            "max_uncertainty_m": max_uncertainty_m,
            "rat_filter": rat_filter,
        },
        "legend": legend,
    }
    overlay_meta_json.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _run_pipeline(
    *,
    aoi_geojson: dict[str, Any],
    mode: str,
    depth_threshold_m: float,
    depth_op: str,
    depth_min_m: Optional[float],
    depth_max_m: Optional[float],
    depth_bins_m: list[float],
    cmap_name: str,
    mask_hex: str,
    opacity: float,
    max_uncertainty_m: Optional[float],
    rat_filter: Optional[dict[str, str]],
    download: bool,
    force_rebuild_vrt: bool,
    output_dir: Path,
    web_dir: Path,
) -> dict[str, Any]:
    _ensure_dir(output_dir)
    _ensure_dir(web_dir)

    aoi_path = web_dir / "aoi.geojson"
    overlay_png = web_dir / "overlay.png"
    overlay_meta_json = web_dir / "overlay_meta.json"

    geom = _extract_geometry(aoi_geojson)
    _write_geojson_feature(geom, aoi_path)

    did_download = False
    if download:
        succeeded, failed = fetch_tiles(str(output_dir), str(aoi_path), data_source="modeling")
        did_download = bool(succeeded)
        if failed:
            print("⚠️ Some tiles failed:", failed)
    else:
        print("ℹ️ Download skipped (using cached tiles/VRT only).")

    existing_vrt: Optional[Path]
    try:
        existing_vrt = _find_best_vrt(output_dir)
    except Exception:
        existing_vrt = None

    if force_rebuild_vrt or did_download or existing_vrt is None:
        build_vrt(str(output_dir), data_source="modeling")
        vrt_path = _find_best_vrt(output_dir)
        vrt_built = True
    else:
        vrt_path = existing_vrt
        vrt_built = False

    print(f"✅ Using VRT: {vrt_path}")

    meta = _compute_overlay(
        vrt_path=vrt_path,
        cutline_geojson=aoi_path,
        overlay_png=overlay_png,
        overlay_meta_json=overlay_meta_json,
        mode=mode,
        depth_threshold_m=depth_threshold_m,
        depth_op=depth_op,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        depth_bins_m=depth_bins_m,
        cmap_name=cmap_name,
        mask_hex=mask_hex,
        opacity=opacity,
        max_uncertainty_m=max_uncertainty_m,
        rat_filter=rat_filter,
    )

    meta["cache"] = {
        "download_requested": download,
        "did_download_any": did_download,
        "force_rebuild_vrt": force_rebuild_vrt,
        "vrt_rebuilt": vrt_built,
        "vrt_path": str(vrt_path),
    }
    return meta


# ----------------------------
# Web page
# ----------------------------
def _make_index_html(web_dir: Path) -> None:
    m = folium.Map(location=[36.9, -76.3], zoom_start=9, tiles="OpenStreetMap")

    folium.WmsTileLayer(
        url="https://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/MaritimeChartService/WMSServer",
        layers="0,1,2,3,4,5,6,7,10",
        fmt="image/png",
        name="NOAA ENC",
        attr="NOAA",
        transparent=True,
        overlay=True,
        control=True,
        show=True,
    ).add_to(m)

    # Add BlueTopo WMTS layers
    # Note: Folium doesn't support WMTS directly, so we'll add these via custom JavaScript
    # The layer control and WMTS setup will be in the JavaScript section below

    folium.LayerControl().add_to(m)

    control_html = r"""
    <div id="bt-control" style="position: fixed; top: 10px; left: 10px; z-index: 9999;
         background: white; padding: 10px; border:2px solid #666; width: 450px; opacity:0.95; font-family: sans-serif; max-height: 90vh; overflow-y: auto;">
      <div style="font-weight: 700; margin-bottom: 8px;">BlueTopo ChatBathy 💬</div>

      <!-- Tab Buttons -->
      <div style="display: flex; gap: 4px; margin-bottom: 10px; border-bottom: 2px solid #ddd;">
        <button id="tab-chat-btn" onclick="switchTab('chat')" style="flex:1; padding:8px; font-size:13px; font-weight:600; border:none; background:#4CAF50; color:white; cursor:pointer; border-radius:4px 4px 0 0;">
          💬 Chat Assistant
        </button>
        <button id="tab-manual-btn" onclick="switchTab('manual')" style="flex:1; padding:8px; font-size:13px; border:none; background:#ddd; color:#666; cursor:pointer; border-radius:4px 4px 0 0;">
          🎛️ Manual Controls
        </button>
      </div>

      <!-- CHAT TAB -->
      <div id="tab-chat" style="display:block;">
        <div style="margin-bottom:10px; font-size:11px; color:#555; background:#f9f9f9; padding:8px; border-radius:4px;">
          <strong>Pro Tip:</strong> Ask me anything! I can calculate depths, filter by date, understand navigation terminology, and more.
          <br><br>
          <strong>Examples:</strong><br>
          • "I'm a ship with 5 ft draft, show areas with 2 ft UKC"<br>
          • "NOAA surveys from last 10 years, no interpolation"<br>
          • "Show full coverage areas deeper than 10m"
        </div>

        <!-- Chat History -->
        <div id="chat-history" style="max-height: 300px; overflow-y: auto; margin-bottom: 10px; border: 1px solid #ddd; background: #fafafa; padding: 8px; border-radius: 4px; font-size: 12px;">
          <div style="color: #666; font-style: italic;">Chat history will appear here...</div>
        </div>

        <!-- Chat Input -->
        <div style="display:flex; gap:6px; margin-bottom:10px;">
          <textarea id="chat-input" placeholder="Ask me anything about bathymetry data..."
                    style="flex:1; font-size:13px; padding:8px; border:2px solid #4CAF50; border-radius:4px; font-family:sans-serif; resize:vertical;"
                    rows="3"
                    onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); sendChatMessage(); }"></textarea>
        </div>
        <div style="display:flex; gap:6px;">
          <button onclick="sendChatMessage()" style="flex:1; padding:10px; font-size:14px; font-weight:600; background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer;">
            Send
          </button>
          <button onclick="clearChatHistory()" style="padding:10px; font-size:12px; background:#ddd; color:#666; border:none; border-radius:4px; cursor:pointer;">
            Clear
          </button>
        </div>

        <!-- Execution Plan -->
        <div id="execution-plan" style="margin-top:10px; padding:8px; background:#fff3cd; border:1px solid #ffc107; border-radius:4px; font-size:11px; display:none;">
          <div style="font-weight:600; margin-bottom:4px;">🤖 Execution Plan:</div>
          <div id="execution-plan-content"></div>
        </div>

        <!-- Status -->
        <div id="chat-tab-status" style="margin-top:8px; font-size:12px; color:#333;">Ready to chat!</div>
      </div>

      <!-- MANUAL CONTROLS TAB -->
      <div id="tab-manual" style="display:none;">
        <div style="margin-top:6px; font-size: 12px;">
          AOI options:<br>
          • Draw a polygon/rectangle using the toolbar (top-right), or<br>
          • Use the current screen extent as a bbox AOI
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          AOI:
          <select id="aoi_mode" style="width: 240px;">
            <option value="draw" selected>Drawn polygon</option>
            <option value="viewport">Viewport bounds (bbox)</option>
          </select>
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          Mode:
          <select id="mode" style="width: 240px;">
            <option value="depth_mask" selected>Depth threshold mask</option>
            <option value="depth_ramp">Depth ramp (grayscale)</option>
            <option value="depth_classes">Depth classes (binned)</option>
            <option value="contributors_random">Contributor IDs (random colors)</option>
          </select>
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          Depth filter:
          <select id="depth_op" style="width: 100px;" onchange="updateDepthInputs()">
            <option value="deeper" selected>Deeper than</option>
            <option value="shallower">Shallower than</option>
            <option value="between">Between</option>
          </select>
          <span id="depth_single_label" style="margin-left:6px;">(m):</span>
          <input id="depth_m" type="number" value="10" min="0" step="0.1" style="width: 80px;">
          <span id="depth_range_label" style="margin-left:6px; display:none;">Min (m):</span>
          <input id="depth_min_m" type="number" value="5" min="0" step="0.1" style="width: 70px; display:none;">
          <span id="depth_range_max_label" style="margin-left:6px; display:none;">Max (m):</span>
          <input id="depth_max_m" type="number" value="15" min="0" step="0.1" style="width: 70px; display:none;">
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          <span>Mask color:</span>
          <input id="mask_color" type="color" value="#ff9500">
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          Depth bins (m, edges):
          <input id="bins_m" type="text" value="0,5,10,20,35" style="width: 240px;">
          <div style="font-size:11px; color:#555; margin-top:2px;">Example: 0,5,10,20,35 creates 0–5, 5–10, 10–20, 20–35, &gt;=35</div>
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          Colormap:
          <select id="cmap" style="width: 140px;">
            <option value="viridis" selected>viridis</option>
            <option value="plasma">plasma</option>
            <option value="magma">magma</option>
            <option value="inferno">inferno</option>
            <option value="turbo">turbo</option>
          </select>

          <span style="margin-left:8px;">Opacity:</span>
          <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.65" style="width: 120px;">
          <span id="opacity_label" style="font-size:11px; color:#333;">0.65</span>
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          <label><input id="unc_enable" type="checkbox"> Filter by uncertainty (<= m)</label>
          <input id="unc_m" type="number" value="1" min="0" step="0.1" style="width: 80px;">
        </div>

        <div style="margin-top:10px; padding-top:8px; border-top: 1px solid #ddd;">
          <div style="font-weight:600; font-size: 12px;">RAT filter (band 3)</div>
          <div style="font-size:11px; color:#555;">(Loads fields from cached VRT RAT)</div>

          <label style="font-size:12px;"><input id="rat_enable" type="checkbox"> Enable RAT filter</label>
          <button onclick="loadRatSchema()" style="margin-left:8px; font-size:12px;">Load RAT fields</button>

          <div style="margin-top:6px; font-size:12px;">
            <label style="margin-right:10px;">
              <input type="radio" name="rat_combiner" value="AND" checked> Match ALL (AND)
            </label>
            <label>
              <input type="radio" name="rat_combiner" value="OR"> Match ANY (OR)
            </label>
          </div>

          <div id="rat_conditions_container" style="margin-top:6px;">
            <!-- Condition rows will be added here dynamically -->
          </div>

          <button onclick="addRatCondition()" style="margin-top:6px; font-size:12px;">+ Add condition</button>

          <div id="rat_hint" style="margin-top:4px; font-size:11px; color:#666;"></div>
        </div>

        <div style="margin-top:8px; font-size: 12px;">
          <label><input id="do_download" type="checkbox" checked> Download missing/updated tiles</label><br>
          <label><input id="force_vrt" type="checkbox"> Force rebuild VRT</label>
        </div>

        <button onclick="runBluetopo()" style="margin-top:8px; width:100%; padding:6px;">
          Run
        </button>

        <div id="status" style="margin-top:8px; font-size: 12px; color:#333;">Ready.</div>

        <div id="legend" style="margin-top:8px; font-size:12px; background:#fafafa; border:1px solid #ddd; padding:6px; display:none;"></div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(control_html))

    # Leaflet.draw assets
    m.get_root().header.add_child(
        folium.Element(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css"/>'
        )
    )
    m.get_root().header.add_child(
        folium.Element(
            '<script defer src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>'
        )
    )

    map_id = m.get_name()

    glue_js = f"""
    (function() {{
      const mapName = "{map_id}";
      function getMap() {{ return window[mapName]; }}

      let ratSchema = null;
      let chatHistory = [];

      // Tab switching
      window.updateDepthInputs = function() {{
        const op = document.getElementById('depth_op').value;
        const singleLabel = document.getElementById('depth_single_label');
        const singleInput = document.getElementById('depth_m');
        const rangeLabel = document.getElementById('depth_range_label');
        const rangeMinInput = document.getElementById('depth_min_m');
        const rangeMaxLabel = document.getElementById('depth_range_max_label');
        const rangeMaxInput = document.getElementById('depth_max_m');

        if (op === 'between') {{
          // Show range inputs, hide single input
          singleLabel.style.display = 'none';
          singleInput.style.display = 'none';
          rangeLabel.style.display = 'inline';
          rangeMinInput.style.display = 'inline';
          rangeMaxLabel.style.display = 'inline';
          rangeMaxInput.style.display = 'inline';
        }} else {{
          // Show single input, hide range inputs
          singleLabel.style.display = 'inline';
          singleInput.style.display = 'inline';
          rangeLabel.style.display = 'none';
          rangeMinInput.style.display = 'none';
          rangeMaxLabel.style.display = 'none';
          rangeMaxInput.style.display = 'none';
        }}
      }}

      window.switchTab = function(tab) {{
        if (tab === 'chat') {{
          document.getElementById('tab-chat').style.display = 'block';
          document.getElementById('tab-manual').style.display = 'none';
          document.getElementById('tab-chat-btn').style.background = '#4CAF50';
          document.getElementById('tab-chat-btn').style.color = 'white';
          document.getElementById('tab-manual-btn').style.background = '#ddd';
          document.getElementById('tab-manual-btn').style.color = '#666';
        }} else {{
          document.getElementById('tab-chat').style.display = 'none';
          document.getElementById('tab-manual').style.display = 'block';
          document.getElementById('tab-chat-btn').style.background = '#ddd';
          document.getElementById('tab-chat-btn').style.color = '#666';
          document.getElementById('tab-manual-btn').style.background = '#4CAF50';
          document.getElementById('tab-manual-btn').style.color = 'white';
        }}
      }}

      // Chat functions
      window.sendChatMessage = async function() {{
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message) return;

        // Add user message to history
        addChatMessage('user', message);
        input.value = '';

        // Show thinking status
        const status = document.getElementById('chat-tab-status');
        status.innerText = '🤔 Thinking and planning...';
        status.style.color = '#666';

        try {{
          // Load RAT schema if not already loaded
          if (!ratSchema) {{
            await loadRatSchema();
          }}

          // Send to enhanced chat endpoint
          const resp = await fetch('/chat_assistant', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
              query: message,
              aoi_mode: document.getElementById('aoi_mode').value
            }})
          }});

          const result = await resp.json();

          if (result.error) {{
            addChatMessage('assistant', '❌ Error: ' + result.error);
            status.innerText = 'Error occurred';
            status.style.color = '#d00';
            return;
          }}

          // Show assistant's interpretation
          if (result.explanation) {{
            addChatMessage('assistant', result.explanation);
          }}

          // Show execution plan
          if (result.plan) {{
            showExecutionPlan(result.plan);
          }}

          // Apply settings and execute
          if (result.execute) {{
            applyVisualizationSettings(result);
            status.innerText = '⚙️ Executing visualization...';
            // Small delay to ensure all form updates complete (especially RAT filter setTimeout calls)
            await new Promise(resolve => setTimeout(resolve, 100));
            await runBluetopo();
            status.innerText = '✅ Done!';
            status.style.color = '#060';
          }} else {{
            status.innerText = 'Ready';
          }}

        }} catch (e) {{
          addChatMessage('assistant', '❌ Error: ' + e);
          status.innerText = 'Error occurred';
          status.style.color = '#d00';
        }}
      }}

      function addChatMessage(role, text) {{
        const history = document.getElementById('chat-history');
        const msg = document.createElement('div');
        msg.style.marginBottom = '8px';
        msg.style.padding = '6px';
        msg.style.borderRadius = '4px';

        if (role === 'user') {{
          msg.style.background = '#e3f2fd';
          msg.style.borderLeft = '3px solid #2196F3';
          msg.innerHTML = '<strong>You:</strong> ' + escapeHtml(text);
        }} else {{
          msg.style.background = '#f1f8e9';
          msg.style.borderLeft = '3px solid #4CAF50';
          msg.innerHTML = '<strong>Assistant:</strong> ' + text;
        }}

        // Remove placeholder if exists
        if (history.querySelector('[style*="italic"]')) {{
          history.innerHTML = '';
        }}

        history.appendChild(msg);
        history.scrollTop = history.scrollHeight;

        chatHistory.push({{ role, text }});
      }}

      function escapeHtml(text) {{
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
      }}

      function showExecutionPlan(plan) {{
        const planDiv = document.getElementById('execution-plan');
        const content = document.getElementById('execution-plan-content');
        content.innerHTML = plan;
        planDiv.style.display = 'block';

        // Hide after 10 seconds
        setTimeout(() => {{
          planDiv.style.display = 'none';
        }}, 10000);
      }}

      function applyVisualizationSettings(settings) {{
        // RESET ALL PARAMETERS TO DEFAULTS FIRST
        // This ensures old settings don't interfere with new queries

        // Reset mode to default
        document.getElementById('mode').value = 'depth_mask';

        // Reset depth threshold and operator to defaults
        document.getElementById('depth_op').value = 'deeper';
        document.getElementById('depth_m').value = '0';
        document.getElementById('depth_min_m').value = '5';
        document.getElementById('depth_max_m').value = '15';
        updateDepthInputs(); // Update UI based on operator

        // Disable and clear RAT filter
        document.getElementById('rat_enable').checked = false;
        const container = document.getElementById('rat_conditions_container');
        container.innerHTML = '';

        // Disable uncertainty filter
        document.getElementById('unc_enable').checked = false;

        // Set AOI mode to viewport (screen extents) as default for chat
        document.getElementById('aoi_mode').value = 'viewport';

        // Set red mask color as default for chat
        document.getElementById('mask_color').value = '#ff0000';

        // Set 0.75 opacity as default for chat
        document.getElementById('opacity').value = '0.75';
        document.getElementById('opacity_label').innerText = '0.75';

        // NOW APPLY LLM SETTINGS

        // Set mode if specified
        if (settings.mode) {{
          document.getElementById('mode').value = settings.mode;
        }}

        // Set depth threshold and operator if specified
        // If null or undefined, keep the default (0)
        if (settings.depth_threshold_m !== null && settings.depth_threshold_m !== undefined) {{
          document.getElementById('depth_m').value = settings.depth_threshold_m;
        }}

        // Set depth operator if specified
        if (settings.depth_op) {{
          document.getElementById('depth_op').value = settings.depth_op;
        }}

        // Set depth min/max for "between" operator
        if (settings.depth_min_m !== null && settings.depth_min_m !== undefined) {{
          document.getElementById('depth_min_m').value = settings.depth_min_m;
        }}
        if (settings.depth_max_m !== null && settings.depth_max_m !== undefined) {{
          document.getElementById('depth_max_m').value = settings.depth_max_m;
        }}

        // Update depth inputs UI to show/hide appropriate fields
        updateDepthInputs();

        // Set RAT filter if specified
        if (settings.rat_filter) {{
          document.getElementById('rat_enable').checked = true;
          const filter = settings.rat_filter;

          // Set combiner
          if (filter.combiner) {{
            const combinerRadio = document.querySelector(`input[name="rat_combiner"][value="${{filter.combiner}}"]`);
            if (combinerRadio) combinerRadio.checked = true;
          }}

          // Add conditions
          if (filter.conditions && filter.conditions.length > 0) {{
            filter.conditions.forEach(cond => {{
              addRatCondition();
              const rows = container.querySelectorAll("[id^='rat_condition_']");
              const lastRow = rows[rows.length - 1];

              if (lastRow) {{
                const fieldSel = lastRow.querySelector(".rat_field_sel");
                const opSel = lastRow.querySelector(".rat_op_sel");
                const valueInput = lastRow.querySelector(".rat_value_input");

                if (fieldSel && opSel && valueInput) {{
                  fieldSel.value = cond.field;
                  const conditionId = lastRow.id.replace("rat_condition_", "");
                  updateConditionUI(parseInt(conditionId));
                  setTimeout(() => {{
                    opSel.value = cond.op;
                    valueInput.value = cond.value;
                  }}, 10);
                }}
              }}
            }});
          }}
        }}
      }}

      window.clearChatHistory = function() {{
        const history = document.getElementById('chat-history');
        history.innerHTML = '<div style="color: #666; font-style: italic;">Chat history cleared...</div>';
        chatHistory = [];
        document.getElementById('execution-plan').style.display = 'none';
      }}

      function setOpsForKind(kind) {{
        const opSel = document.getElementById("rat_op");
        opSel.innerHTML = "";
        const ops = [];
        if (kind === "string") {{
          ops.push(["contains","contains"], ["equals","equals"]);
        }} else if (kind === "bool") {{
          ops.push(["equals","="], ["equals (true)","1"], ["equals (false)","0"]);
        }} else if (kind === "date") {{
          ops.push(["after (>=)","after"], ["before (<)","before"], ["in year","in_year"], ["equals","equals"]);
        }} else {{
          ops.push(["=","="], ["<=","<="], [">=",">="], ["<","<"], [">",">"]);
        }}
        for (const [label,val] of ops) {{
          const o = document.createElement("option");
          o.text = label;
          o.value = val;
          opSel.appendChild(o);
        }}
      }}

      let ratConditionCounter = 0;

      function createFieldSelect() {{
        if (!ratSchema) return null;
        const sel = document.createElement("select");
        sel.style.width = "150px";

        const preferred = ["survey_date_end","source_institution","source_survey_id","significant_features","coverage","bathy_coverage"];
        const cols = ratSchema.columns || [];
        const byName = new Map(cols.map(c => [c.name, c]));
        const ordered = [];
        for (const p of preferred) {{
          if (byName.has(p)) ordered.push(byName.get(p));
        }}
        for (const c of cols) {{
          if (!preferred.includes(c.name)) ordered.push(c);
        }}

        for (const c of ordered) {{
          const o = document.createElement("option");
          o.text = c.name;
          o.value = c.name;
          sel.appendChild(o);
        }}
        return sel;
      }}

      function updateConditionUI(conditionId) {{
        const row = document.getElementById(`rat_condition_${{conditionId}}`);
        if (!row) return;

        const fieldSel = row.querySelector(".rat_field_sel");
        const opSel = row.querySelector(".rat_op_sel");
        const valueSel = row.querySelector(".rat_value_input");

        if (!fieldSel || !ratSchema) return;

        const field = fieldSel.value;
        const col = ratSchema.columns.find(c => c.name === field);
        if (!col) return;

        const kind = col.kind || "string";

        // Update operators
        opSel.innerHTML = "";
        const ops = [];
        if (kind === "string") {{
          ops.push(["contains","contains"], ["equals","equals"]);
        }} else if (kind === "bool") {{
          ops.push(["equals","="], ["equals (true)","1"], ["equals (false)","0"]);
        }} else if (kind === "date") {{
          ops.push(["after (>=)","after"], ["before (<)","before"], ["in year","in_year"], ["equals","equals"]);
        }} else {{
          ops.push(["=","="], ["<=","<="], [">=",">="], ["<","<"], [">",">"]);
        }}
        for (const [label,val] of ops) {{
          const o = document.createElement("option");
          o.text = label;
          o.value = val;
          opSel.appendChild(o);
        }}

        // Update placeholder
        if (kind === "date") {{
          valueSel.placeholder = "e.g., 2021 or 2020-01-01";
        }} else if (kind === "number") {{
          valueSel.placeholder = "e.g., 1 or 2.5";
        }} else if (kind === "bool") {{
          valueSel.placeholder = "e.g., 1 or 0";
        }} else {{
          valueSel.placeholder = "e.g., NOAA";
        }}
      }}

      window.addRatCondition = function() {{
        if (!ratSchema) {{
          alert("Please load RAT fields first!");
          return;
        }}

        const container = document.getElementById("rat_conditions_container");
        const conditionId = ratConditionCounter++;

        const row = document.createElement("div");
        row.id = `rat_condition_${{conditionId}}`;
        row.style.marginTop = "6px";
        row.style.fontSize = "12px";
        row.style.padding = "4px";
        row.style.border = "1px solid #ddd";
        row.style.borderRadius = "3px";
        row.style.backgroundColor = "#f9f9f9";

        const fieldSel = createFieldSelect();
        fieldSel.className = "rat_field_sel";
        fieldSel.onchange = () => updateConditionUI(conditionId);

        const opSel = document.createElement("select");
        opSel.className = "rat_op_sel";
        opSel.style.width = "100px";
        opSel.style.marginLeft = "4px";

        const valueInput = document.createElement("input");
        valueInput.className = "rat_value_input";
        valueInput.type = "text";
        valueInput.style.width = "140px";
        valueInput.style.marginLeft = "4px";

        const removeBtn = document.createElement("button");
        removeBtn.innerText = "✕";
        removeBtn.style.marginLeft = "4px";
        removeBtn.style.fontSize = "12px";
        removeBtn.onclick = () => {{
          container.removeChild(row);
        }};

        row.appendChild(fieldSel);
        row.appendChild(opSel);
        row.appendChild(document.createTextNode(" Value: "));
        row.appendChild(valueInput);
        row.appendChild(removeBtn);

        container.appendChild(row);
        updateConditionUI(conditionId);
      }}

      function collectRatConditions() {{
        const container = document.getElementById("rat_conditions_container");
        const rows = container.querySelectorAll("[id^='rat_condition_']");
        const conditions = [];

        rows.forEach(row => {{
          const fieldSel = row.querySelector(".rat_field_sel");
          const opSel = row.querySelector(".rat_op_sel");
          const valueInput = row.querySelector(".rat_value_input");

          if (fieldSel && opSel && valueInput) {{
            const field = fieldSel.value;
            const op = opSel.value;
            const value = valueInput.value;
            if (field && value) {{
              conditions.push({{ field, op, value }});
            }}
          }}
        }});

        return conditions;
      }}

      window.loadRatSchema = async function() {{
        const hint = document.getElementById("rat_hint");
        hint.innerText = "Loading RAT schema…";
        try {{
          const r = await fetch("/rat_schema?ts=" + Date.now());
          const data = await r.json();
          if (!r.ok) {{
            hint.innerText = "RAT schema error: " + (data.error || "unknown");
            return;
          }}
          ratSchema = data;
          hint.innerText = "Loaded " + data.columns.length + " RAT fields ✅";

          // Add first condition row automatically
          const container = document.getElementById("rat_conditions_container");
          if (container.children.length === 0) {{
            addRatCondition();
          }}
        }} catch (e) {{
          hint.innerText = "RAT schema load failed: " + e;
        }}
      }}

      window.processChatQuery = async function() {{
        const queryInput = document.getElementById("chat_query");
        const statusDiv = document.getElementById("chat_status");
        const query = queryInput.value.trim();

        if (!query) {{
          statusDiv.innerText = "Please enter a query";
          statusDiv.style.color = "#d00";
          return;
        }}

        if (!ratSchema) {{
          statusDiv.innerText = "Please load RAT fields first!";
          statusDiv.style.color = "#d00";
          return;
        }}

        statusDiv.innerText = "🤔 Thinking...";
        statusDiv.style.color = "#666";

        try {{
          const resp = await fetch("/chat_query", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ query }})
          }});

          const result = await resp.json();

          if (result.error) {{
            statusDiv.innerText = "❌ " + result.error;
            statusDiv.style.color = "#d00";
            return;
          }}

          // Clear existing conditions
          const container = document.getElementById("rat_conditions_container");
          container.innerHTML = "";

          // Set combiner
          const combiner = result.combiner || "AND";
          document.querySelector(`input[name="rat_combiner"][value="${{combiner}}"]`).checked = true;

          // Add each condition
          const conditions = result.conditions || [];
          for (const cond of conditions) {{
            addRatCondition();

            // Get the last added row
            const rows = container.querySelectorAll("[id^='rat_condition_']");
            const lastRow = rows[rows.length - 1];

            if (lastRow) {{
              const fieldSel = lastRow.querySelector(".rat_field_sel");
              const opSel = lastRow.querySelector(".rat_op_sel");
              const valueInput = lastRow.querySelector(".rat_value_input");

              if (fieldSel && opSel && valueInput) {{
                fieldSel.value = cond.field;

                // Trigger update to populate operators
                const conditionId = lastRow.id.replace("rat_condition_", "");
                updateConditionUI(parseInt(conditionId));

                // Set operator and value after UI update
                setTimeout(() => {{
                  opSel.value = cond.op;
                  valueInput.value = cond.value;
                }}, 10);
              }}
            }}
          }}

          // Enable the filter
          document.getElementById("rat_enable").checked = true;

          statusDiv.innerText = `✅ Applied ${{conditions.length}} condition(s) with ${{combiner}}`;
          statusDiv.style.color = "#060";

        }} catch (e) {{
          statusDiv.innerText = "❌ Error: " + e;
          statusDiv.style.color = "#d00";
        }}
      }}

      function waitReady(cb) {{
        const t0 = Date.now();
        (function tick() {{
          const map = getMap();
          if (window.L && map) return cb(map);
          if (Date.now() - t0 > 15000) {{
            console.error("Leaflet/map not ready after 15s");
            const s = document.getElementById("status");
            if (s) s.innerText = "Error: map libraries didn't load (see console).";
            return;
          }}
          setTimeout(tick, 50);
        }})();
      }}

      function modeLabel(m) {{
        const lut = {{
          "depth_mask": "Depth mask",
          "depth_ramp": "Depth ramp",
          "depth_classes": "Depth classes",
          "contributors_random": "Contributors"
        }};
        return lut[m] || m;
      }}

      function renderLegend(legend) {{
        const box = document.getElementById("legend");
        if (!legend || !legend.items || legend.items.length === 0) {{
          box.style.display = "none";
          box.innerHTML = "";
          return;
        }}
        box.style.display = "block";
        let html = "<div style='font-weight:600; margin-bottom:4px;'>" + (legend.title || "Legend") + "</div>";
        for (const it of legend.items) {{
          const c = it.color || "#999";
          const lab = it.label || "";
          html += "<div style='display:flex; align-items:center; margin:2px 0;'>"
               +  "<span style='display:inline-block; width:14px; height:14px; background:" + c + "; border:1px solid #666; margin-right:6px;'></span>"
               +  "<span>" + lab + "</span></div>";
        }}
        box.innerHTML = html;
      }}

      waitReady(function(map) {{
        // keep opacity label updated
        const op = document.getElementById("opacity");
        const opl = document.getElementById("opacity_label");
        op.oninput = () => {{ opl.innerText = op.value; }};

        // AOI drawing
        const drawnItems = new L.FeatureGroup();
        map.addLayer(drawnItems);

        const hasDraw = (L.Control && L.Control.Draw);
        if (hasDraw) {{
          const drawControl = new L.Control.Draw({{
            position: "topright",
            edit: {{ featureGroup: drawnItems }},
            draw: {{
              polygon: true,
              rectangle: true,
              polyline: false,
              circle: false,
              circlemarker: false,
              marker: false
            }}
          }});
          map.addControl(drawControl);

          map.on(L.Draw.Event.CREATED, function (e) {{
            drawnItems.clearLayers();
            drawnItems.addLayer(e.layer);
            window._aoi = e.layer.toGeoJSON().geometry;
            const status = document.getElementById("status");
            if (status) status.innerText = "AOI captured ✅";
          }});
        }} else {{
          console.warn("Leaflet.draw did not load. Drawing AOI disabled; use viewport AOI instead.");
          const status = document.getElementById("status");
          if (status) status.innerText = "Leaflet.draw not loaded. Use AOI = Viewport bounds.";
        }}

        // Overlay layer
        let overlayLayer = null;

        async function refreshOverlay() {{
          try {{
            const r = await fetch("/overlay_meta?ts=" + Date.now());
            if (!r.ok) return;
            const meta = await r.json();
            if (!meta || !meta.ready) return;

            if (overlayLayer) {{
              map.removeLayer(overlayLayer);
              overlayLayer = null;
            }}

            const url = meta.url + "?ts=" + Date.now();
            overlayLayer = L.imageOverlay(url, meta.bounds, {{ opacity: meta.opacity || 0.65 }});
            overlayLayer.addTo(map);

            renderLegend(meta.legend);

            const status = document.getElementById("status");
            const mode = meta.mode || document.getElementById("mode").value;
            const cache = meta.cache || {{}};

            let extra = "";
            if (cache.download_requested === false) extra = " (cache-only)";
            else if (cache.did_download_any === false && cache.vrt_rebuilt === false) extra = " (reused cache)";
            else if (cache.vrt_rebuilt === true) extra = " (VRT rebuilt)";

            if (status) status.innerText = "Overlay updated ✅ [" + modeLabel(mode) + "]" + extra;
          }} catch (err) {{
            console.warn("overlay refresh error", err);
          }}
        }}

        function viewportBoundsToPolygon() {{
          const b = map.getBounds();
          const sw = b.getSouthWest();
          const ne = b.getNorthEast();
          return {{
            "type": "Polygon",
            "coordinates": [[
              [sw.lng, sw.lat],
              [ne.lng, sw.lat],
              [ne.lng, ne.lat],
              [sw.lng, ne.lat],
              [sw.lng, sw.lat]
            ]]
          }};
        }}

        function parseBins(s) {{
          const parts = (s || "").split(",").map(x => parseFloat(x.trim())).filter(x => isFinite(x));
          return parts;
        }}

        window.runBluetopo = async function() {{
          const status = document.getElementById("status");
          const mode = document.getElementById("mode").value;

          const aoiMode = document.getElementById("aoi_mode").value;
          let geom = null;
          if (aoiMode === "viewport") geom = viewportBoundsToPolygon();
          else geom = window._aoi || null;

          if (!geom) {{
            alert("Draw a polygon/rectangle first, or switch AOI to 'Viewport bounds (bbox)'.");
            return;
          }}

          const depthOp = document.getElementById("depth_op").value;
          let depthM = 0;
          let depthMinM = null;
          let depthMaxM = null;

          if (depthOp === "between") {{
            depthMinM = parseFloat(document.getElementById("depth_min_m").value);
            depthMaxM = parseFloat(document.getElementById("depth_max_m").value);
            if (!isFinite(depthMinM) || depthMinM < 0 || !isFinite(depthMaxM) || depthMaxM < 0) {{
              alert("Min and max depth must be non-negative numbers (meters).");
              return;
            }}
            if (depthMinM >= depthMaxM) {{
              alert("Min depth must be less than max depth.");
              return;
            }}
          }} else {{
            depthM = parseFloat(document.getElementById("depth_m").value);
            if (!isFinite(depthM) || depthM < 0) {{
              alert("Depth threshold must be a non-negative number (meters).");
              return;
            }}
          }}

          const bins = parseBins(document.getElementById("bins_m").value);
          const cmap = document.getElementById("cmap").value;
          const maskColor = document.getElementById("mask_color").value;
          const opacity = parseFloat(document.getElementById("opacity").value);

          const uncEnable = document.getElementById("unc_enable").checked;
          const uncM = parseFloat(document.getElementById("unc_m").value);
          const maxUnc = (uncEnable && isFinite(uncM) && uncM >= 0) ? uncM : null;

          const ratEnable = document.getElementById("rat_enable").checked;
          let ratFilter = null;
          if (ratEnable) {{
            const conditions = collectRatConditions();
            if (conditions.length > 0) {{
              const combiner = document.querySelector('input[name="rat_combiner"]:checked').value;
              ratFilter = {{ combiner, conditions }};
            }}
          }}

          const doDownload = document.getElementById("do_download").checked;
          const forceVrt = document.getElementById("force_vrt").checked;

          const payload = {{
            aoi: geom,
            mode: mode,
            depth_threshold_m: depthM,
            depth_op: depthOp,
            depth_min_m: depthMinM,
            depth_max_m: depthMaxM,
            depth_bins_m: bins,
            cmap: cmap,
            mask_hex: maskColor,
            opacity: opacity,
            max_uncertainty_m: maxUnc,
            rat_filter: ratFilter,
            download: doDownload,
            force_rebuild_vrt: forceVrt
          }};

          if (status) status.innerText = "Running… " + modeLabel(mode) + " ⏳";

          try {{
            const resp = await fetch("/run", {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify(payload)
            }});
            const data = await resp.json();

            if (!resp.ok) {{
              const msg = "Error: " + (data.error || "unknown");
              if (status) status.innerText = msg;
              alert(msg);
              return;
            }}

            await refreshOverlay();
          }} catch (err) {{
            const msg = "Error: " + err;
            if (status) status.innerText = msg;
            alert(msg);
          }}
        }}

        // try to load RAT schema on startup (if VRT already exists)
        loadRatSchema().catch(() => {{}});
        refreshOverlay();

        // ========================================
        // WMTS LAYERS AND POINT QUERY
        // ========================================

        // Add WMTS layers (Folium doesn't support WMTS, so we add them manually)
        const wmtsUrl = 'https://nowcoast.noaa.gov/geoserver/gwc/service/wmts';

        // BlueTopo Hillshade layer (visible by default, 60% opacity)
        const hillshadeLayer = L.tileLayer(
          wmtsUrl + '?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&' +
          'LAYER=bluetopo:hillshade&STYLE=&TILEMATRIX=EPSG:3857:{{z}}&' +
          'TILEMATRIXSET=EPSG:3857&FORMAT=image/png8&TILECOL={{x}}&TILEROW={{y}}',
          {{
            attribution: 'NOAA BlueTopo Hillshade',
            opacity: 0.6,
            maxZoom: 19,
            zIndex: 500  // Above ENC and base layers
          }}
        ).addTo(map);

        // BlueTopo Bathymetry layer (hidden by default, 70% opacity)
        const bathymetryLayer = L.tileLayer(
          wmtsUrl + '?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&' +
          'LAYER=bluetopo:bathymetry&STYLE=&TILEMATRIX=EPSG:3857:{{z}}&' +
          'TILEMATRIXSET=EPSG:3857&FORMAT=image/png8&TILECOL={{x}}&TILEROW={{y}}',
          {{
            attribution: 'NOAA BlueTopo Bathymetry',
            opacity: 0.7,
            maxZoom: 19,
            zIndex: 501  // Above hillshade
          }}
        );
        // Don't add bathymetry by default (user can toggle it on)

        // Add layers to the existing layer control
        setTimeout(() => {{
          const layerControl = document.querySelector('.leaflet-control-layers');
          if (layerControl) {{
            const overlaysDiv = layerControl.querySelector('.leaflet-control-layers-overlays');
            if (overlaysDiv) {{
              // Add hillshade checkbox
              const hillshadeLabel = document.createElement('label');
              hillshadeLabel.innerHTML = '<input type="checkbox" class="leaflet-control-layers-selector" checked> <span>BlueTopo Hillshade</span>';
              hillshadeLabel.querySelector('input').addEventListener('change', (e) => {{
                if (e.target.checked) {{
                  map.addLayer(hillshadeLayer);
                }} else {{
                  map.removeLayer(hillshadeLayer);
                }}
              }});
              overlaysDiv.appendChild(hillshadeLabel);

              // Add bathymetry checkbox
              const bathyLabel = document.createElement('label');
              bathyLabel.innerHTML = '<input type="checkbox" class="leaflet-control-layers-selector"> <span>BlueTopo Depth/Elevation</span>';
              bathyLabel.querySelector('input').addEventListener('change', (e) => {{
                if (e.target.checked) {{
                  map.addLayer(bathymetryLayer);
                }} else {{
                  map.removeLayer(bathymetryLayer);
                }}
              }});
              overlaysDiv.appendChild(bathyLabel);
            }}
          }}
        }}, 500);

        // ========================================
        // POINT QUERY FUNCTIONALITY
        // ========================================

        map.on('click', async function(e) {{
          const lat = e.latlng.lat;
          const lng = e.latlng.lng;

          // Show loading popup
          const loadingPopup = L.popup()
            .setLatLng(e.latlng)
            .setContent('<div style="padding:5px;">🔍 Querying data...</div>')
            .openOn(map);

          try {{
            const resp = await fetch('/query_point', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ lat: lat, lon: lng }})
            }});

            const data = await resp.json();

            if (!resp.ok || data.error) {{
              L.popup()
                .setLatLng(e.latlng)
                .setContent('<div style="padding:5px; color:#d00;">❌ ' + (data.error || 'Query failed') + '</div>')
                .openOn(map);
              return;
            }}

            // Build popup content
            let html = '<div style="font-size:12px; font-family:sans-serif; max-width:300px;">';
            html += '<div style="font-weight:700; margin-bottom:8px; padding-bottom:4px; border-bottom:2px solid #4CAF50;">📍 Point Query Results</div>';

            // Depth/Elevation
            if (data.depth !== null && data.depth !== undefined) {{
              const depthVal = parseFloat(data.depth);
              const depthLabel = depthVal < 0 ? 'Depth' : 'Elevation';
              const depthAbs = Math.abs(depthVal).toFixed(2);
              html += '<div style="margin:4px 0;"><strong>' + depthLabel + ':</strong> ' + depthAbs + ' m</div>';
            }}

            // Uncertainty
            if (data.uncertainty !== null && data.uncertainty !== undefined) {{
              html += '<div style="margin:4px 0;"><strong>Uncertainty:</strong> ' + parseFloat(data.uncertainty).toFixed(2) + ' m</div>';
            }}

            // RAT Data
            if (data.rat_data && Object.keys(data.rat_data).length > 0) {{
              html += '<div style="margin-top:8px; padding-top:8px; border-top:1px solid #ddd;">';
              html += '<div style="font-weight:600; margin-bottom:4px;">Survey Information:</div>';
              html += '<div style="max-height:200px; overflow-y:auto; font-size:11px;">';

              // Sort RAT fields for better display
              const sortedKeys = Object.keys(data.rat_data).sort();
              sortedKeys.forEach(key => {{
                const value = data.rat_data[key];
                if (value !== null && value !== undefined && value !== '') {{
                  // Format field name (convert snake_case to Title Case)
                  const fieldName = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                  html += '<div style="margin:2px 0;"><strong>' + fieldName + ':</strong> ' + value + '</div>';
                }}
              }});

              html += '</div></div>';
            }}

            html += '</div>';

            L.popup()
              .setLatLng(e.latlng)
              .setContent(html)
              .openOn(map);

          }} catch (err) {{
            console.error('Point query error:', err);
            L.popup()
              .setLatLng(e.latlng)
              .setContent('<div style="padding:5px; color:#d00;">❌ Query error: ' + err + '</div>')
              .openOn(map);
          }}
        }});

      }});
    }})();
    """
    m.get_root().script.add_child(folium.Element(glue_js))

    out = web_dir / "index.html"
    m.save(str(out))


# ----------------------------
# HTTP server
# ----------------------------
class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.startswith("/overlay_meta"):
            meta_path = Path("overlay_meta.json")
            if meta_path.exists():
                try:
                    self._send_json(json.loads(meta_path.read_text(encoding="utf-8")))
                except Exception as e:
                    self._send_json({"ready": False, "error": str(e)}, status=500)
            else:
                self._send_json({"ready": False})
            return

        if self.path.startswith("/rat_schema"):
            try:
                schema = self.server.rat_schema()  # type: ignore[attr-defined]
                self._send_json(schema, status=200)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/chat_query":
            # Handle natural language query
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))

                query = payload.get("query", "")
                if not query:
                    self._send_json({"error": "Query is required"}, status=400)
                    return

                # Get RAT schema
                rat_schema = self.server.rat_schema()  # type: ignore[attr-defined]

                # Process query with OpenAI
                result = _process_natural_language_query(query, rat_schema)

                self._send_json(result, status=200)

            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path == "/chat_assistant":
            # Enhanced chat assistant with full visualization control
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))

                query = payload.get("query", "")
                if not query:
                    self._send_json({"error": "Query is required"}, status=400)
                    return

                # Get RAT schema
                rat_schema = self.server.rat_schema()  # type: ignore[attr-defined]

                # Process query with enhanced chat assistant
                result = _process_chat_assistant(query, rat_schema)

                self._send_json(result, status=200)

            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path == "/query_point":
            # Handle point query for depth, uncertainty, and RAT data
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))

                lat = float(payload.get("lat"))
                lon = float(payload.get("lon"))

                # Find the VRT file
                output_dir = self.server._output_dir  # type: ignore[attr-defined]
                vrt_path = _find_best_vrt(output_dir)

                if vrt_path is None:
                    self._send_json({"error": "No VRT file found. Please run a visualization first."}, status=404)
                    return

                # Create a small point geometry around the click location
                # We'll create a 10-meter buffer to query a small area
                import tempfile
                from osgeo import gdal, ogr, osr
                import numpy as np

                # Create a tiny bounding box around the point (0.0001 degrees ~= 11 meters at equator)
                buffer_deg = 0.0001
                minx = lon - buffer_deg
                miny = lat - buffer_deg
                maxx = lon + buffer_deg
                maxy = lat + buffer_deg

                # Create a temporary geometry for the query point
                with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp:
                    point_geom = {
                        "type": "Polygon",
                        "coordinates": [[
                            [minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy],
                            [minx, miny]
                        ]]
                    }
                    tmp.write(json.dumps({"type": "Feature", "geometry": point_geom, "properties": {}}))
                    tmp_geojson = Path(tmp.name)

                try:
                    # Warp the VRT to a small area around the click point
                    # This ensures we can query any point, regardless of the VRT's original projection
                    ds_warped = gdal.Warp(
                        '',  # In-memory
                        str(vrt_path),
                        format='MEM',
                        dstSRS='EPSG:4326',
                        outputBounds=[minx, miny, maxx, maxy],
                        width=3,  # Small 3x3 pixel window
                        height=3,
                        resampleAlg=gdal.GRA_NearestNeighbour,
                        cutlineDSName=str(tmp_geojson),
                        cropToCutline=True
                    )

                    if ds_warped is None:
                        self._send_json({"error": "No data available at this location"}, status=404)
                        return

                    # Query the center pixel (1,1) of the 3x3 grid
                    depth_band = ds_warped.GetRasterBand(DEPTH_BAND)
                    uncertainty_band = ds_warped.GetRasterBand(UNCERTAINTY_BAND)
                    contrib_band = ds_warped.GetRasterBand(CONTRIB_BAND)

                    # Read the center pixel
                    depth_val = depth_band.ReadAsArray(1, 1, 1, 1)[0, 0]
                    uncertainty_val = uncertainty_band.ReadAsArray(1, 1, 1, 1)[0, 0]
                    contrib_val = contrib_band.ReadAsArray(1, 1, 1, 1)[0, 0]

                    # Check for nodata values
                    depth_nodata = depth_band.GetNoDataValue()
                    uncertainty_nodata = uncertainty_band.GetNoDataValue()
                    contrib_nodata = contrib_band.GetNoDataValue()

                    result = {}

                    # Add depth (convert to Python float, handle nodata)
                    if depth_nodata is not None and np.isclose(depth_val, depth_nodata):
                        result["depth"] = None
                    else:
                        result["depth"] = float(depth_val)

                    # Add uncertainty
                    if uncertainty_nodata is not None and np.isclose(uncertainty_val, uncertainty_nodata):
                        result["uncertainty"] = None
                    else:
                        result["uncertainty"] = float(uncertainty_val)

                    # Look up RAT data if contributor is valid
                    result["rat_data"] = {}
                    print(f"DEBUG: contrib_val={contrib_val}, contrib_nodata={contrib_nodata}")

                    if contrib_nodata is None or not np.isclose(contrib_val, contrib_nodata):
                        contrib_id = int(contrib_val)
                        print(f"DEBUG: Looking up RAT for contributor ID: {contrib_id}")

                        # Get RAT schema and data from the ORIGINAL VRT (not the warped one)
                        # The warped dataset doesn't preserve the RAT, so we need to read it from the source
                        ds_original = gdal.Open(str(vrt_path))
                        if ds_original is not None:
                            rat_schema = self.server.rat_schema()  # type: ignore[attr-defined]
                            print(f"DEBUG: RAT schema exists: {rat_schema is not None}")
                            print(f"DEBUG: RAT schema keys: {list(rat_schema.keys()) if rat_schema else 'None'}")

                            if rat_schema and "columns" in rat_schema:
                                print(f"DEBUG: RAT schema has {len(rat_schema['columns'])} columns")
                                # Read the RAT table from the original VRT's contributor band
                                original_contrib_band = ds_original.GetRasterBand(CONTRIB_BAND)
                                rat = original_contrib_band.GetDefaultRAT()
                                print(f"DEBUG: RAT table exists: {rat is not None}")

                                if rat is not None:
                                    # Find the row index for this contributor ID
                                    row_count = rat.GetRowCount()
                                    print(f"DEBUG: RAT has {row_count} rows")
                                    found = False
                                    for row in range(row_count):
                                        # The first column is usually the ID
                                        row_id = int(rat.GetValueAsInt(row, 0))
                                        if row_id == contrib_id:
                                            found = True
                                            print(f"DEBUG: Found contributor {contrib_id} at row {row}")
                                            # Found the matching row, read all columns
                                            for col_info in rat_schema["columns"]:
                                                field_name = col_info["name"]
                                                # Find the column index for this field
                                                col_idx = None
                                                for col in range(rat.GetColumnCount()):
                                                    if rat.GetNameOfCol(col) == field_name:
                                                        col_idx = col
                                                        break

                                                if col_idx is not None:
                                                    field_type = rat.GetTypeOfCol(col_idx)
                                                    if field_type == gdal.GFT_Integer:
                                                        result["rat_data"][field_name] = rat.GetValueAsInt(row, col_idx)
                                                    elif field_type == gdal.GFT_Real:
                                                        result["rat_data"][field_name] = rat.GetValueAsDouble(row, col_idx)
                                                    else:
                                                        result["rat_data"][field_name] = rat.GetValueAsString(row, col_idx)
                                            print(f"DEBUG: Read {len(result['rat_data'])} RAT fields")
                                            break
                                    if not found:
                                        print(f"DEBUG: Contributor ID {contrib_id} not found in RAT")
                            ds_original = None  # Close the original VRT
                    else:
                        print(f"DEBUG: Contributor value is nodata, skipping RAT lookup")

                    print(f"DEBUG: Final result has {len(result['rat_data'])} RAT fields")
                    ds_warped = None  # Close the dataset
                    self._send_json(result, status=200)

                finally:
                    # Clean up temporary file
                    if tmp_geojson.exists():
                        tmp_geojson.unlink()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if self.path != "/run":
            self._send_json({"error": "unknown endpoint"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            aoi = payload["aoi"]
            mode = str(payload.get("mode", "depth_mask"))

            depth_threshold_m = float(payload.get("depth_threshold_m", 10.0))
            depth_op = str(payload.get("depth_op", "deeper"))
            depth_min_m = payload.get("depth_min_m", None)
            if depth_min_m is not None:
                depth_min_m = float(depth_min_m)
            depth_max_m = payload.get("depth_max_m", None)
            if depth_max_m is not None:
                depth_max_m = float(depth_max_m)
            depth_bins_m = payload.get("depth_bins_m", [0, 5, 10, 20, 35])
            if not isinstance(depth_bins_m, list):
                depth_bins_m = [0, 5, 10, 20, 35]
            depth_bins_m = [float(x) for x in depth_bins_m if isinstance(x, (int, float, str))]

            cmap_name = str(payload.get("cmap", "viridis"))
            mask_hex = str(payload.get("mask_hex", "#ff9500"))
            opacity = float(payload.get("opacity", 0.65))

            max_uncertainty_m = payload.get("max_uncertainty_m", None)
            if max_uncertainty_m is not None:
                max_uncertainty_m = float(max_uncertainty_m)

            rat_filter = payload.get("rat_filter", None)
            if rat_filter is not None:
                # New multi-condition format: {combiner: "AND"|"OR", conditions: [{field, op, value}, ...]}
                combiner = str(rat_filter.get("combiner", "AND"))
                conditions = rat_filter.get("conditions", [])
                if isinstance(conditions, list) and len(conditions) > 0:
                    rat_filter = {
                        "combiner": combiner,
                        "conditions": [
                            {
                                "field": str(c.get("field", "")),
                                "op": str(c.get("op", "contains")),
                                "value": str(c.get("value", "")),
                            }
                            for c in conditions
                        ],
                    }
                else:
                    rat_filter = None

            download = bool(payload.get("download", True))
            force_rebuild_vrt = bool(payload.get("force_rebuild_vrt", False))

            meta = self.server.run_compute(  # type: ignore[attr-defined]
                aoi_geojson=aoi,
                mode=mode,
                depth_threshold_m=depth_threshold_m,
                depth_op=depth_op,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                depth_bins_m=depth_bins_m,
                cmap_name=cmap_name,
                mask_hex=mask_hex,
                opacity=opacity,
                max_uncertainty_m=max_uncertainty_m,
                rat_filter=rat_filter,
                download=download,
                force_rebuild_vrt=force_rebuild_vrt,
            )
            self._send_json(meta, status=200)

        except Exception as e:
            self._send_json({"error": str(e)}, status=500)


class _Server(ThreadingTCPServer):
    allow_reuse_address = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, *, output_dir: Path, web_dir: Path):
        super().__init__(server_address, RequestHandlerClass)
        self._output_dir = output_dir
        self._web_dir = web_dir
        self._rat_schema_cache: Optional[dict[str, Any]] = None
        self._rat_schema_cache_mtime: float = 0.0

    def _cached_vrt(self) -> Path:
        return _find_best_vrt(self._output_dir)

    def rat_schema(self) -> dict[str, Any]:
        vrt = self._cached_vrt()
        mt = vrt.stat().st_mtime
        if self._rat_schema_cache is not None and mt == self._rat_schema_cache_mtime:
            return self._rat_schema_cache

        schema = _rat_schema_from_vrt(vrt)
        self._rat_schema_cache = schema
        self._rat_schema_cache_mtime = mt
        return schema

    def run_compute(
        self,
        *,
        aoi_geojson: dict[str, Any],
        mode: str,
        depth_threshold_m: float,
        depth_op: str,
        depth_min_m: Optional[float],
        depth_max_m: Optional[float],
        depth_bins_m: list[float],
        cmap_name: str,
        mask_hex: str,
        opacity: float,
        max_uncertainty_m: Optional[float],
        rat_filter: Optional[dict[str, str]],
        download: bool,
        force_rebuild_vrt: bool,
    ) -> dict[str, Any]:
        with _STATE_LOCK:
            _STATE["last_error"] = None

        meta = _run_pipeline(
            aoi_geojson=aoi_geojson,
            mode=mode,
            depth_threshold_m=depth_threshold_m,
            depth_op=depth_op,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            depth_bins_m=depth_bins_m,
            cmap_name=cmap_name,
            mask_hex=mask_hex,
            opacity=opacity,
            max_uncertainty_m=max_uncertainty_m,
            rat_filter=rat_filter,
            download=download,
            force_rebuild_vrt=force_rebuild_vrt,
            output_dir=self._output_dir,
            web_dir=self._web_dir,
        )

        with _STATE_LOCK:
            _STATE["overlay_meta"] = meta
        return meta


def main() -> None:
    ap = argparse.ArgumentParser()

    in_docker = _in_docker()
    default_root = Path(os.getenv("BLUETOPO_DATA_ROOT", "/data" if in_docker else str(Path.cwd())))

    ap.add_argument(
        "--output-dir",
        default=str(default_root / "cache"),
        help="Directory where tiles/VRT will be downloaded/built (persistent cache).",
    )
    ap.add_argument(
        "--web-dir",
        default=str(default_root / "web"),
        help="Directory served by the local webserver (index.html + overlay files).",
    )
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0" if in_docker else "127.0.0.1"), help="Bind host.")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8777")), help="Port to serve on.")

    # Browser behavior:
    # - default: open only when NOT in docker AND host is loopback
    # - override with --open-browser or --no-browser
    ap.add_argument("--open-browser", action="store_true", help="Force-open a browser tab on start.")
    ap.add_argument("--no-browser", action="store_true", help="Never open a browser tab on start.")

    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    web_dir = Path(args.web_dir).expanduser().resolve()
    _ensure_dir(output_dir)
    _ensure_dir(web_dir)

    _make_index_html(web_dir=web_dir)
    os.chdir(str(web_dir))

    addr = (args.host, args.port)
    with _Server(addr, _Handler, output_dir=output_dir, web_dir=web_dir) as httpd:
        host_for_url = "localhost" if addr[0] in ("0.0.0.0", "::") else addr[0]
        url = f"http://{host_for_url}:{addr[1]}/index.html"
        print(f"🌊 Serving at: {url}")
        print("   Draw AOI (toolbar top-right) or use viewport bbox, set Mode + params, then Run.")

        should_open = False
        if args.no_browser:
            should_open = False
        elif args.open_browser:
            should_open = True
        else:
            should_open = (not in_docker) and (host_for_url in ("localhost", "127.0.0.1"))

        if should_open:
            try:
                webbrowser.open(url, new=1)
            except Exception:
                pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down.")


if __name__ == "__main__":
    main()

