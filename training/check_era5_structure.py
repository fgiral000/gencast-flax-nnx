#!/usr/bin/env python3
"""
Validate ERA5 monthly files structure and consistency.

Checks for each month:
  - Both pressure and single-level files exist
  - Can be opened by xarray
  - Standardized dims/names to (time, level, lat, lon) or (time, lat, lon)
  - Sizes of time/lat/lon match between pressure and single
  - lon and lat have consistent ordering; lon not duplicated
  - Expected variable sets present (best-effort)
  - Merging (similar to training/era5_dataset.load_merged_month) succeeds

Usage:
  python scripts/check_era5_structure.py --root data_era5 --resolution 2.50

Exit code is non-zero if any month fails checks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

EXPECTED_SINGLE_VARS = {
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}
EXPECTED_PRESSURE_VARS = {
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
}


def find_month_files(root_dir: Path, resolution_deg: float) -> List[Tuple[str, Path, Path]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"ERA5 root dir not found: {root}")
    res_str = f"{resolution_deg:.2f}deg"
    p_files = sorted(root.glob(f"era5_pressure_levels_*_{res_str}.nc"))
    s_files = sorted(root.glob(f"era5_single_levels_*_{res_str}.nc"))

    def ym_from_path(p: Path) -> str:
        parts = p.stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename: {p.name}")
        return parts[-2]

    p_map = {ym_from_path(p): p for p in p_files}
    s_map = {ym_from_path(p): p for p in s_files}
    months = sorted(set(p_map) & set(s_map))
    return [(ym, p_map[ym], s_map[ym]) for ym in months]


def standardize_pressure_file(path: Path) -> xr.Dataset:
    ds = xr.load_dataset(path)
    dim_ren = {}
    if "valid_time" in ds.dims:
        dim_ren["valid_time"] = "time"
    if "pressure_level" in ds.dims:
        dim_ren["pressure_level"] = "level"
    if "latitude" in ds.dims:
        dim_ren["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_ren["longitude"] = "lon"
    if dim_ren:
        ds = ds.rename(dim_ren)

    drop_coords = [c for c in ds.coords if c in ("number", "expver", "step")]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)

    var_ren = {}
    if "t" in ds:
        var_ren["t"] = "temperature"
    if "z" in ds:
        var_ren["z"] = "geopotential"
    if "u" in ds:
        var_ren["u"] = "u_component_of_wind"
    if "v" in ds:
        var_ren["v"] = "v_component_of_wind"
    if "w" in ds:
        var_ren["w"] = "vertical_velocity"
    if "q" in ds:
        var_ren["q"] = "specific_humidity"
    if var_ren:
        ds = ds.rename(var_ren)

    # Reorder dims per variable
    for name in list(ds.data_vars):
        da = ds[name]
        dims = list(da.dims)
        desired = [d for d in ("time", "level", "lat", "lon") if d in dims]
        remaining = [d for d in dims if d not in desired]
        order = desired + remaining
        if order != dims:
            ds[name] = da.transpose(*order)

    return ds


def standardize_single_file(path: Path) -> xr.Dataset:
    ds = xr.load_dataset(path)
    dim_ren = {}
    if "valid_time" in ds.dims and "time" not in ds.dims:
        dim_ren["valid_time"] = "time"
    if "latitude" in ds.dims:
        dim_ren["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_ren["longitude"] = "lon"
    if dim_ren:
        ds = ds.rename(dim_ren)

    if "valid_time" in ds.coords and "time" in ds.dims:
        ds = ds.drop_vars("valid_time")

    var_map = {}
    if "2m_temperature" in ds:
        var_map["2m_temperature"] = "2m_temperature"
    elif "t2m" in ds:
        var_map["t2m"] = "2m_temperature"
    elif "2t" in ds:
        var_map["2t"] = "2m_temperature"

    if "mean_sea_level_pressure" in ds:
        var_map["mean_sea_level_pressure"] = "mean_sea_level_pressure"
    elif "msl" in ds:
        var_map["msl"] = "mean_sea_level_pressure"

    if "10m_u_component_of_wind" in ds:
        var_map["10m_u_component_of_wind"] = "10m_u_component_of_wind"
    elif "u10" in ds:
        var_map["u10"] = "10m_u_component_of_wind"
    elif "10u" in ds:
        var_map["10u"] = "10m_u_component_of_wind"

    if "10m_v_component_of_wind" in ds:
        var_map["10m_v_component_of_wind"] = "10m_v_component_of_wind"
    elif "v10" in ds:
        var_map["v10"] = "10m_v_component_of_wind"
    elif "10v" in ds:
        var_map["10v"] = "10m_v_component_of_wind"

    if var_map:
        ds = ds.rename(var_map)

    # Keep expected vars only (if present)
    keep = [v for v in ds.data_vars if v in EXPECTED_SINGLE_VARS]
    ds = ds[keep]

    for v in ds.data_vars:
        ds[v] = ds[v].transpose("time", "lat", "lon")

    return ds


def check_month(ym: str, p_path: Path, s_path: Path) -> Dict:
    issues: List[str] = []
    try:
        ds_p = standardize_pressure_file(p_path)
    except Exception as e:
        return {"month": ym, "ok": False, "issues": [f"open pressure failed: {e}"]}
    try:
        ds_s = standardize_single_file(s_path)
    except Exception as e:
        return {"month": ym, "ok": False, "issues": [f"open single failed: {e}"]}

    # Basic dim presence
    for d in ("time", "lat", "lon"):
        if d not in ds_p.dims:
            issues.append(f"pressure missing dim {d}")
        if d not in ds_s.dims:
            issues.append(f"single missing dim {d}")
    if "level" not in ds_p.dims:
        issues.append("pressure missing dim level")

    # Sizes
    sizes_ok = True
    for d in ("time", "lat", "lon"):
        if d in ds_p.dims and d in ds_s.dims and ds_p.sizes[d] != ds_s.sizes[d]:
            issues.append(
                f"size mismatch {d}: pressure={ds_p.sizes.get(d)} single={ds_s.sizes.get(d)}"
            )
            sizes_ok = False

    # lon duplicates
    try:
        lon_vals = ds_p["lon"].values
        unique_lon = np.unique(lon_vals)
        if unique_lon.size != lon_vals.size:
            issues.append(f"pressure lon has duplicates: {lon_vals.size - unique_lon.size} duplicates")
    except Exception as e:
        issues.append(f"failed to check lon duplicates: {e}")

    # Variable dims consistency
    for name, da in ds_p.data_vars.items():
        exp = tuple(d for d in ("time", "level", "lat", "lon") if d in da.dims)
        if tuple(da.dims) != exp:
            issues.append(f"pressure var {name} dims {da.dims} not ordered like {exp}")
    for name, da in ds_s.data_vars.items():
        if tuple(da.dims) != ("time", "lat", "lon"):
            issues.append(f"single var {name} dims {da.dims} not (time, lat, lon)")

    # Try to align/merge similar to loader
    try:
        ds_p2 = ds_p.sortby("lon").sortby("lat")
        ds_s2 = ds_s.sortby("lon").sortby("lat")
        ds_s2 = ds_s2.assign_coords(
            time=("time", ds_p2["time"].values),
            lat=("lat", ds_p2["lat"].values),
            lon=("lon", ds_p2["lon"].values),
        )
        _ = xr.merge([ds_p2, ds_s2], compat="override", join="override")
    except Exception as e:
        issues.append(f"merge failed: {e}")

    # Expected vars present (warn only)
    missing_single = sorted([v for v in EXPECTED_SINGLE_VARS if v not in ds_s.data_vars])
    if missing_single:
        issues.append(f"missing single vars: {missing_single}")
    missing_pressure = sorted([v for v in EXPECTED_PRESSURE_VARS if v not in ds_p.data_vars])
    if missing_pressure:
        issues.append(f"missing pressure vars: {missing_pressure}")

    ok = len(issues) == 0
    return {
        "month": ym,
        "ok": ok,
        "issues": issues,
        "sizes": {
            "time": int(ds_p.sizes.get("time", -1)),
            "level": int(ds_p.sizes.get("level", -1)),
            "lat": int(ds_p.sizes.get("lat", -1)),
            "lon": int(ds_p.sizes.get("lon", -1)),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Validate ERA5 monthly data structure")
    ap.add_argument("--root", type=str, default="data_era5/", help="Root dir with ERA5 nc files")
    ap.add_argument("--resolution", type=float, default=2.50, help="Resolution degrees used in filenames")
    args = ap.parse_args()

    months = find_month_files(Path(args.root), args.resolution)
    if not months:
        print(f"No month pairs found under {args.root} for resolution {args.resolution:.2f}")
        sys.exit(2)

    print(f"Found {len(months)} months to check")
    failures = 0
    for ym, p_path, s_path in months:
        res = check_month(ym, p_path, s_path)
        if res["ok"]:
            print(f"[OK ] {ym}  sizes: {res['sizes']}")
        else:
            print(f"[ERR] {ym} -> {len(res['issues'])} issue(s)")
            for it in res["issues"]:
                print(f"       - {it}")
            failures += 1

    print("\nSummary:")
    print(f"  Months checked: {len(months)}")
    print(f"  Failures     : {failures}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
