#!/usr/bin/env python3
"""Build a versioned city-level annual PM2.5 table from ACAG SatPM2.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Iterable

import pandas as pd
import requests


SOURCE_VERSION = "V6.GL.03"
SOURCE_PRODUCT = "ACAG SatPM2.5"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_REGISTRY = "https://registry.opendata.aws/surface-pm2-5-v6gl/"
SOURCE_URL_TEMPLATE = (
    "https://satpmdata.s3.amazonaws.com/V6GL03/CoarseResolution/GL/Annual/"
    "V6GL03.CNNPM25.0p10.GL.{year}01-{year}12.nc"
)
GRID_LAT_ORIGIN = -59.95
GRID_LON_ORIGIN = -179.95
GRID_STEP = 0.1
GRID_LAT_SIZE = 1300
GRID_LON_SIZE = 3600
DEFAULT_YEARS = tuple(range(2015, 2025))
DEFAULT_OUTPUT = Path("data/air_quality/acag_v6gl03_city_annual_2015_2024.csv")
DEFAULT_MANIFEST = Path("data/air_quality/acag_v6gl03_manifest.json")
DEFAULT_DOWNLOAD_DIR = Path("data/cache/air_quality/acag_v6gl03")


def parse_years(value: str) -> tuple[int, ...]:
    """Parse a single year, comma list, or inclusive ``start:end`` range."""

    cleaned = value.strip()
    if ":" in cleaned:
        start_text, end_text = cleaned.split(":", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise argparse.ArgumentTypeError("year range end is earlier than start")
        return tuple(range(start, end + 1))
    years = tuple(sorted({int(item.strip()) for item in cleaned.split(",") if item.strip()}))
    if not years:
        raise argparse.ArgumentTypeError("at least one year is required")
    return years


def grid_window(latitude: float, longitude: float, radius_cells: int = 1) -> tuple[int, int, int, int]:
    """Return a clipped square grid window around a latitude/longitude point."""

    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"invalid coordinates: {latitude}, {longitude}")
    lat_index = round((latitude - GRID_LAT_ORIGIN) / GRID_STEP)
    lon_index = round((longitude - GRID_LON_ORIGIN) / GRID_STEP)
    lat_start = max(0, lat_index - radius_cells)
    lon_start = max(0, lon_index - radius_cells)
    lat_end = min(GRID_LAT_SIZE, lat_index + radius_cells + 1)
    lon_end = min(GRID_LON_SIZE, lon_index + radius_cells + 1)
    if lat_start >= lat_end or lon_start >= lon_end:
        raise ValueError(f"coordinates outside ACAG grid: {latitude}, {longitude}")
    return lat_start, lon_start, lat_end - lat_start, lon_end - lon_start


def download_annual_file(year: int, download_dir: Path, force: bool = False) -> tuple[Path, str]:
    """Download one official annual NetCDF file with bounded retries."""

    url = SOURCE_URL_TEMPLATE.format(year=year)
    destination = download_dir / Path(url).name
    if destination.exists() and destination.stat().st_size > 1_000_000 and not force:
        return destination, url
    download_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    final_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size <= 1_000_000:
                raise ValueError(f"downloaded file is unexpectedly small: {partial.stat().st_size} bytes")
            partial.replace(destination)
            return destination, url
        except (requests.RequestException, OSError, ValueError) as exc:
            final_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(2 ** (attempt - 1))
    assert final_error is not None
    raise RuntimeError(f"failed to download {url} after 4 attempts") from final_error


def extract_city_pm25(
    source_path: Path,
    latitude: float,
    longitude: float,
    radius_cells: int = 1,
    h5dump_command: str = "h5dump",
) -> tuple[float, int, int]:
    """Extract a finite-cell mean from one ACAG annual HDF5/NetCDF file."""

    executable = shutil.which(h5dump_command)
    if executable is None:
        raise RuntimeError("h5dump is required to regenerate the ACAG city table")
    lat_start, lon_start, lat_count, lon_count = grid_window(latitude, longitude, radius_cells)
    command = [
        executable,
        "-d",
        "/PM25",
        "-s",
        f"{lat_start},{lon_start}",
        "-c",
        f"{lat_count},{lon_count}",
        "-y",
        "-w",
        "0",
        "-m",
        "%.8f",
        str(source_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = _parse_h5dump_values(completed.stdout)
    finite = [value for value in values if math.isfinite(value) and value >= 0]
    if not finite:
        raise ValueError(f"no finite PM2.5 cells near {latitude}, {longitude} in {source_path.name}")
    return sum(finite) / len(finite), len(finite), lat_count * lon_count


def _parse_h5dump_values(output: str) -> list[float]:
    """Parse only the first dataset data block from ``h5dump`` output."""

    if "DATA {" not in output:
        raise ValueError("h5dump output has no data block")
    data_block = output.split("DATA {", 1)[1].split("}", 1)[0]
    tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|[-+]?nan|[-+]?inf", data_block, flags=re.IGNORECASE)
    return [float(token) for token in tokens]


def build_city_records(
    seed_data: pd.DataFrame,
    years: Iterable[int],
    download_dir: Path,
    radius_cells: int = 1,
    force_download: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[Path]]:
    """Download annual sources and build one PM2.5 record per city-year."""

    required = {"city", "city_zh", "city_en", "latitude", "longitude"}
    missing = required - set(seed_data.columns)
    if missing:
        raise ValueError(f"seed table missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    downloaded_paths: list[Path] = []
    for year in years:
        source_path, source_url = download_annual_file(int(year), download_dir, force=force_download)
        downloaded_paths.append(source_path)
        sources.append(
            {
                "year": int(year),
                "url": source_url,
                "filename": source_path.name,
                "bytes": source_path.stat().st_size,
                "sha256": file_sha256(source_path),
            }
        )
        for city in seed_data.itertuples(index=False):
            pm25_mean, valid_cells, total_cells = extract_city_pm25(
                source_path,
                float(city.latitude),
                float(city.longitude),
                radius_cells=radius_cells,
            )
            rows.append(
                {
                    "city": str(city.city),
                    "city_zh": str(city.city_zh),
                    "city_en": str(city.city_en),
                    "year": int(year),
                    "pm25_mean_ug_m3": round(pm25_mean, 4),
                    "valid_grid_cells": valid_cells,
                    "grid_cell_count": total_cells,
                    "source_product": SOURCE_PRODUCT,
                    "source_version": SOURCE_VERSION,
                    "spatial_method": f"mean of finite cells in {radius_cells * 2 + 1}x{radius_cells * 2 + 1} 0.1-degree window",
                }
            )
    frame = pd.DataFrame(rows).sort_values(["city", "year"]).reset_index(drop=True)
    return frame, sources, downloaded_paths


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    frame: pd.DataFrame,
    sources: list[dict[str, object]],
    output_path: Path,
    manifest_path: Path,
    radius_cells: int,
) -> None:
    """Write the annual city table and a provenance/checksum manifest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    years = sorted(int(value) for value in frame["year"].unique())
    manifest = {
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "source_registry": SOURCE_REGISTRY,
        "license": SOURCE_LICENSE,
        "accessed_on": "2026-07-31",
        "year_start": min(years),
        "year_end": max(years),
        "years": years,
        "resolution_degrees": GRID_STEP,
        "spatial_method": f"mean of finite cells in {radius_cells * 2 + 1}x{radius_cells * 2 + 1} 0.1-degree window",
        "city_count": int(frame["city"].nunique()),
        "record_count": int(len(frame)),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
        "sources": sources,
        "citation": "SatPM2.5 was accessed on 2026-07-31 from https://registry.opendata.aws/surface-pm2-5-v6gl/.",
        "method_note": "Satellite-, simulation-, and monitor-fused annual ground-level PM2.5 estimate; not a pure station observation.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_downloads(paths: Iterable[Path]) -> None:
    """Remove only source NetCDF files used during this run."""

    for path in paths:
        path.unlink(missing_ok=True)


def main() -> None:
    """Run the command-line city extraction workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=Path("data/city_seed.csv"))
    parser.add_argument("--years", type=parse_years, default=DEFAULT_YEARS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--radius-cells", type=int, default=1)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-source-files", action="store_true")
    args = parser.parse_args()
    if args.radius_cells < 0 or args.radius_cells > 5:
        parser.error("--radius-cells must be between 0 and 5")

    seed_data = pd.read_csv(args.seed)
    frame, sources, downloaded_paths = build_city_records(
        seed_data,
        args.years,
        args.download_dir,
        radius_cells=args.radius_cells,
        force_download=args.force_download,
    )
    write_outputs(frame, sources, args.output, args.manifest, args.radius_cells)
    if not args.keep_source_files:
        remove_downloads(downloaded_paths)
    print(f"wrote {len(frame)} records for {frame['city'].nunique()} cities to {args.output}")


if __name__ == "__main__":
    main()
