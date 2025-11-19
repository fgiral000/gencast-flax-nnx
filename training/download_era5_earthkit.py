#!/usr/bin/env python3
"""Download ERA5 data using Earthkit with year-range control.

This script downloads the minimal set of ERA5 variables needed by GenCast/GraphCast
for a chosen resolution and date range, saving monthly files to keep disk usage
reasonable.

It uses the CDS (Copernicus Climate Data Store) via earthkit-data. Ensure you
have valid CDS credentials in ~/.cdsapirc. See:
https://cds.climate.copernicus.eu/api-how-to

Examples:
  # Download single & pressure level data for 2019 at 1.0° resolution in NetCDF
  python scripts/download_era5_earthkit.py \
      --out-dir ./data/era5 \
      --start-year 2019 --end-year 2019 \
      --resolution 1.0 \
      --format netcdf

Notes:
  - Static fields (geopotential_at_surface, land_sea_mask) are fetched once.
  - Monthly files are written for both single-level and pressure-level datasets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import earthkit.data as ekd

# Variables in CDS naming
SINGLE_LEVEL_VARS = [
    # 2m & surface
    "2m_temperature",               # t2m
    "mean_sea_level_pressure",      # msl
    "10m_u_component_of_wind",      # u10
    "10m_v_component_of_wind",      # v10
    "total_precipitation",          # tp (we will aggregate to 12h later)
]

# Static variables from single-levels dataset (download once)
STATIC_SINGLE_LEVEL_VARS = [
    "geopotential",                 # z at surface
    "land_sea_mask",                # lsm
]

# Pressure level variables (CDS names) needed for WB13 levels
PRESSURE_LEVEL_VARS = [
    "temperature",                   # t
    "geopotential",                  # z
    "u_component_of_wind",           # u
    "v_component_of_wind",           # v
    "vertical_velocity",             # w (Pa/s)
    "specific_humidity",             # q
]

# WeatherBench13 pressure levels in hPa
WB13_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
# ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
HOURS_12H = ["00:00", "12:00"]



def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def request_single_levels(year: int, month: str, grid: str, fmt: str):
    return {
        "product_type": "reanalysis",
        "variable": SINGLE_LEVEL_VARS,
        "year": str(year),
        "month": month,
        "day": ALL_DAYS,
        "time": HOURS_12H,
        "grid": grid,  # e.g. "1.0/1.0" or "0.25/0.25"
        "format": fmt,  # "grib" or "netcdf"
    }


def request_static_single_levels(grid: str, fmt: str):
    # Download static fields for one date/time only (they're time-invariant)
    return {
        "product_type": "reanalysis",
        "variable": STATIC_SINGLE_LEVEL_VARS,
        "year": "2019",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "grid": grid,
        "format": fmt,
    }


def request_pressure_levels(year: int, month: str, grid: str, fmt: str):
    return {
        "product_type": "reanalysis",
        "variable": PRESSURE_LEVEL_VARS,
        "pressure_level": WB13_LEVELS,
        "year": str(year),
        "month": month,
        "day": ALL_DAYS,
        "time": HOURS_12H,
        "grid": grid,
        "format": fmt,
    }


def filename(prefix: str, kind: str, year: int, month: str, res: float, fmt: str) -> str:
    ext = "nc" if fmt == "netcdf" else "grib"
    return f"{prefix}_{kind}_{year}{month}_{res:.2f}deg.{ext}"


def download_month(
    out_dir: Path,
    year: int,
    month: str,
    resolution: float,
    fmt: str,
) -> List[Path]:
    grid = f"{resolution}/{resolution}"
    written: List[Path] = []

    # Single levels
    req_sl = request_single_levels(year, month, grid, fmt)
    ds_sl = ekd.from_source("cds", "reanalysis-era5-single-levels", req_sl)
    fout_sl = out_dir / filename("era5", "single_levels", year, month, resolution, fmt)
    ds_sl.save(str(fout_sl))
    written.append(fout_sl)

    # Pressure levels
    req_pl = request_pressure_levels(year, month, grid, fmt)
    ds_pl = ekd.from_source("cds", "reanalysis-era5-pressure-levels", req_pl)
    fout_pl = out_dir / filename("era5", "pressure_levels", year, month, resolution, fmt)
    ds_pl.save(str(fout_pl))
    written.append(fout_pl)

    return written


def download_static(out_dir: Path, resolution: float, fmt: str) -> Path:
    grid = f"{resolution}/{resolution}"
    req_static = request_static_single_levels(grid, fmt)
    ds_static = ekd.from_source("cds", "reanalysis-era5-single-levels", req_static)
    fout = out_dir / (f"era5_static_{resolution:.2f}deg." + ("nc" if fmt == "netcdf" else "grib"))
    ds_static.save(str(fout))
    return fout


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 via Earthkit with bounded year range.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for downloaded files.")
    parser.add_argument("--start-year", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="End year (inclusive).")
    parser.add_argument("--resolution", type=float, default=1.0, choices=[0.25, 1.0, 2.5], help="Target grid resolution in degrees.")
    parser.add_argument("--format", type=str, default="netcdf", choices=["netcdf", "grib"], help="Output format.")
    parser.add_argument("--months", type=str, nargs="*", default=None, help="Subset of months like 01 02 ... 12. Default: all months.")
    parser.add_argument("--season", type=str, default=None, choices=["DJF","MAM","JJA","SON"], help="Convenience to select meteorological season; overrides --months if set.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Resolve months: season overrides explicit months if provided
    if args.season:
        season_map = {
            "DJF": ["12","01","02"],
            "MAM": ["03","04","05"],
            "JJA": ["06","07","08"],
            "SON": ["09","10","11"],
        }
        months = season_map[args.season]
    else:
        months = args.months if args.months else ALL_MONTHS

    print(f"Downloading ERA5 {args.start_year}-{args.end_year}, months {months}, res={args.resolution}°, format={args.format}")
    print("NOTE: Ensure you have valid ~/.cdsapirc credentials.")

    # Download static once
    static_path = download_static(out_dir, args.resolution, args.format)
    print(f"Saved static fields: {static_path}")

    for year in range(args.start_year, args.end_year + 1):
        for month in months:
            print(f"Year {year}, Month {month}: downloading...")
            try:
                files = download_month(out_dir, year, month, args.resolution, args.format)
                for f in files:
                    print(f"  -> {f}")
            except Exception as e:
                print(f"Failed {year}-{month}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
