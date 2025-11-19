# era5_dataset.py
"""
ERA5 dataset utilities for GenCast training using Grain.

Assumptions:
- ERA5 has been downloaded with scripts/download_era5_earthkit.py
  at 2.50° resolution, producing files like:
    - era5_pressure_levels_199506_2.50deg.nc
    - era5_single_levels_199506_2.50deg.nc
    - era5_static_2.50deg.nc

Goals:
- Do NOT mix time across files: treat each monthly file independently.
- TRAIN:
    Use 3-step windows within a month:
      window: [t-1, t, t+1] → (X_{t-1}, X_t) -> Y_{t+1}
- TEST:
    Use full month sequences (for autoregressive rollouts).
- Provide Grain-based datasets that yield (inputs, targets, forcings)
  in the same xarray format as the toy GCS dataset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional, Literal, Any

import numpy as np
import xarray as xr

import grain  # google-grain
from gencast import gencast
from common import data_utils


# --------------------------------------------------------------------------------------
# File discovery helpers
# --------------------------------------------------------------------------------------


def _find_month_files(
    root_dir: str | Path,
    resolution_deg: float = 2.50,
) -> List[Tuple[str, Path, Path]]:
    """
    Find all (month_id, pressure_path, single_path) under root_dir for the given resolution.

    month_id is "YYYYMM" as a string.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"ERA5 root dir not found: {root}")

    res_str = f"{resolution_deg:.2f}deg"
    # pressure: era5_pressure_levels_YYYYMM_2.50deg.nc
    # single:   era5_single_levels_YYYYMM_2.50deg.nc
    pressure_files = sorted(root.glob(f"era5_pressure_levels_*_{res_str}.nc"))
    single_files = sorted(root.glob(f"era5_single_levels_*_{res_str}.nc"))

    # Build mapping YYYYMM -> path
    def _extract_month_id(p: Path, kind: str) -> str:
        # expects "era5_<kind>_YYYYMM_2.50deg.nc"
        stem = p.stem  # e.g. "era5_pressure_levels_199506_2.50deg"
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format for {kind}: {p.name}")
        # last 2 parts are yearmonth + res; yearmonth is parts[-2]
        ym = parts[-2]
        return ym

    pressure_map: Dict[str, Path] = { _extract_month_id(p, "pressure"): p
                                      for p in pressure_files }
    single_map: Dict[str, Path] = { _extract_month_id(p, "single"): p
                                    for p in single_files }

    # Only months where we have BOTH pressure & single level files
    common_months = sorted(set(pressure_map.keys()) & set(single_map.keys()))
    if not common_months:
        raise RuntimeError(
            f"No matching pressure+single ERA5 files found in {root} "
            f"for resolution {res_str}"
        )

    month_tuples: List[Tuple[str, Path, Path]] = []
    for ym in common_months:
        month_tuples.append((ym, pressure_map[ym], single_map[ym]))
    return month_tuples


def _load_static(
    root_dir: str | Path,
    resolution_deg: float = 2.50,
) -> xr.Dataset:
    """
    Load static ERA5 fields and force them to be 2D (lat, lon) only:
      - land_sea_mask(lat, lon)
      - geopotential_at_surface(lat, lon)  (if available)
    """

    root = Path(root_dir)
    res_str = f"{resolution_deg:.2f}deg"
    candidates = list(root.glob(f"era5_static_{res_str}.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"No static ERA5 file found in {root} (expected era5_static_{res_str}.nc)"
        )

    ds = xr.load_dataset(candidates[0])

    # --- Rename dims to lat/lon if necessary ---
    dim_ren = {}
    if "latitude" in ds.dims:
        dim_ren["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_ren["longitude"] = "lon"
    if dim_ren:
        ds = ds.rename(dim_ren)

    # --- Drop ALL non lat/lon dims (select first index) ---
    for d in list(ds.dims):
        if d not in ("lat", "lon"):
            ds = ds.isel({d: 0}, drop=True)

    # --- Rename variables to GraphCast-style names ---
    var_ren = {}
    if "geopotential" in ds:
        var_ren["geopotential"] = "geopotential_at_surface"
    if "z" in ds:
        var_ren["z"] = "geopotential_at_surface"
    if "lsm" in ds:
        var_ren["lsm"] = "land_sea_mask"
    # sometimes already "land_sea_mask" / "geopotential_at_surface"
    ds = ds.rename(var_ren)

    # --- Keep only the expected static vars ---
    keep = [v for v in ds.data_vars if v in ("geopotential_at_surface", "land_sea_mask")]
    ds = ds[keep]

    # Final sanity check: only lat/lon dims
    for v in ds.data_vars:
        if set(ds[v].dims) != {"lat", "lon"}:
            raise ValueError(
                f"Static variable {v} has wrong dims {ds[v].dims}, expected ('lat','lon')"
            )

    return ds




# --------------------------------------------------------------------------------------
# Per-file preprocessing
# --------------------------------------------------------------------------------------


def _standardize_pressure_file(path: Path) -> xr.Dataset:
    """
    Open one pressure-level file and standardize dims/var names.

    Input dims from cfgrib/Earthkit look like:
      valid_time, pressure_level, latitude, longitude

    We rename to:
      time, level, lat, lon

    And variables:
      t -> temperature
      z -> geopotential
      u -> u_component_of_wind
      v -> v_component_of_wind
      w -> vertical_velocity
      q -> specific_humidity
    """
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

    # Drop unneeded coords like "number", "expver" if present
    drop_coords = [c for c in ds.coords if c in ("number", "expver")]
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

    # Reorder dims for all vars to (time, level, lat, lon) when applicable
    for name in list(ds.data_vars):
        da = ds[name]
        dims = list(da.dims)
        desired = [d for d in ("time", "level", "lat", "lon") if d in dims]
        # Append remaining dims in original order
        remaining = [d for d in dims if d not in desired]
        new_order = desired + remaining
        if new_order != dims:
            ds[name] = da.transpose(*new_order)

    return ds


def _standardize_single_file(single_path: Path) -> xr.Dataset:
    """
    Load a monthly ERA5 *single-level* file and map it to GraphCast-style
    surface variables:

      - 2m_temperature          <- t2m or 2t
      - mean_sea_level_pressure <- msl
      - 10m_u_component_of_wind <- 10u
      - 10m_v_component_of_wind <- 10v
      - total_precipitation_12hr <- derived from tp (total_precipitation)

    Output dims: (time, lat, lon) for each variable.
    """
    ds = xr.load_dataset(single_path)

    # --- Normalize dimension names ---
    dim_ren = {}
    if "valid_time" in ds.dims and "time" not in ds.dims:
        dim_ren["valid_time"] = "time"
    if "latitude" in ds.dims:
        dim_ren["latitude"] = "lat"
    if "longitude" in ds.dims:
        dim_ren["longitude"] = "lon"
    if dim_ren:
        ds = ds.rename(dim_ren)

    # Some files have 'valid_time' as coord but 'time' as dim; ensure we use 'time'
    if "valid_time" in ds.coords and "time" in ds.dims:
        # they are usually identical; drop the extra coord
        ds = ds.drop_vars("valid_time")

    # Sanity
    for d in ("time", "lat", "lon"):
        if d not in ds.dims:
            raise ValueError(f"Single-level file {single_path.name} missing dim {d}")

    # --- Map ERA5 short names → GraphCast variable names ---
    # (try both possible spellings for safety)
    var_map: dict[str, str] = {}

    # 2m temperature
    if "2m_temperature" in ds:
        var_map["2m_temperature"] = "2m_temperature"
    elif "t2m" in ds:
        var_map["t2m"] = "2m_temperature"
    elif "2t" in ds:
        var_map["2t"] = "2m_temperature"

    # Mean sea-level pressure
    if "mean_sea_level_pressure" in ds:
        var_map["mean_sea_level_pressure"] = "mean_sea_level_pressure"
    elif "msl" in ds:
        var_map["msl"] = "mean_sea_level_pressure"

    # 10m wind components
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

    ds = ds.rename(var_map)

    # --- Derive 12-hour accumulated precipitation ---
    # ERA5's 'tp' is total_precipitation (m). We need total_precipitation_12hr
    # with the same shape/time grid as the toy dataset.
    if "total_precipitation_12hr" in ds:
        tp_12hr = ds["total_precipitation_12hr"]
    else:
        tp_src = None
        if "total_precipitation" in ds:
            tp_src = ds["total_precipitation"]
        elif "tp" in ds:
            tp_src = ds["tp"]

        if tp_src is not None:
            # Simple 12h accumulation: difference along time axis.
            # This assumes time is 12-hourly (which it is in your download).
            # First step: set accumulation to 0.
            diff = tp_src.diff("time", label="upper")
            # pad first timestep with zeros
            first = xr.zeros_like(tp_src.isel(time=0))
            tp_12hr = xr.concat([first, diff], dim="time")
            tp_12hr = tp_12hr.transpose("time", "lat", "lon")
            tp_12hr.name = "total_precipitation_12hr"
        else:
            # If you really don't have tp, we can at least create a zero-field
            tp_12hr = xr.zeros_like(ds[list(ds.data_vars)[0]].isel(time=0, drop=True))
            tp_12hr = tp_12hr.expand_dims(time=ds["time"])
            tp_12hr.name = "total_precipitation_12hr"

    # --- Subset to only the vars GraphCast cares about ---
    keep_vars = []
    for name in [
        "2m_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]:
        if name in ds:
            keep_vars.append(name)

    ds_out = ds[keep_vars]
    ds_out["total_precipitation_12hr"] = tp_12hr

    # Ensure final ordering (time, lat, lon) for all vars
    for v in ds_out.data_vars:
        ds_out[v] = ds_out[v].transpose("time", "lat", "lon")

    return ds_out



def _ensure_graphcast_time_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Make ERA5 month look like the GraphCast toy dataset in terms of time coords:

      - Add 'datetime' coord with absolute times (if missing).
      - Convert 'time' coord to relative timedelta (time since first step).
      - Ensure there is a 'batch' dimension of size 1.
      - Ensure 'datetime' has dims ('batch', 'time') so that
        common.data_utils.featurize_progress works.

    This is exactly what the toy GCS dataset looks like from the point of view
    of data_utils.extract_inputs_targets_forcings.
    """
    if "time" not in ds.coords:
        raise ValueError("'time' coordinate missing in dataset.")

    # 1) If time is absolute datetime, create 'datetime' from it and
    #    convert 'time' to relative timedelta.
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        # Absolute times → 'datetime'
        if "datetime" not in ds.coords:
            ds = ds.assign_coords(datetime=("time", ds["time"].values))

        # Relative times (since first step) → 'time'
        t0 = ds["time"].isel(time=0)
        rel = ds["time"] - t0
        ds = ds.assign_coords(time=rel)

    # If 'datetime' still missing for some reason, try to reconstruct from attributes
    if "datetime" not in ds.coords:
        raise ValueError(
            "Dataset must have 'datetime' coord or 'time' as datetime64 to "
            "build it, but neither condition holds."
        )

    # 2) Ensure a 'batch' dimension of size 1 exists
    if "batch" not in ds.dims:
        # add a new batch dim of length 1
        ds = ds.expand_dims(batch=[0])

    # 3) Ensure 'datetime' has dims ('batch', 'time')
    dt = ds["datetime"]
    if "batch" not in dt.dims:
        # Currently dims are ('time',); expand along batch
        dt = dt.expand_dims(batch=ds["batch"])
    # Now enforce ordering (batch, time)
    dt = dt.transpose("batch", "time")
    ds = ds.assign_coords(datetime=dt)

    # (Optional but harmless) If you want, you can also expand the
    # already-added progress features to include batch dim.
    for v in ("year_progress_cos", "year_progress_sin"):
        if v in ds.data_vars and "batch" not in ds[v].dims:
            ds[v] = ds[v].expand_dims(batch=ds["batch"]).transpose("batch", "time")
    for v in ("day_progress_cos", "day_progress_sin"):
        if v in ds.data_vars and "batch" not in ds[v].dims:
            ds[v] = ds[v].expand_dims(batch=ds["batch"]).transpose(
                "batch", "time", "lon"
            )

    return ds



def _add_time_features(ds: xr.Dataset) -> xr.Dataset:
    """
    Add day/year progress cos/sin features similar to the toy GCS dataset.

    Toy shapes:
      - day_progress_cos: (batch, time, lon)
      - day_progress_sin: (batch, time, lon)
      - year_progress_cos: (batch, time)
      - year_progress_sin: (batch, time)

    Here we initially create without 'batch'; 'batch' will be added later
    when we expand_dims(batch=1) before calling data_utils.extract_...
    """
    if "time" not in ds.coords:
        return ds

    time = ds["time"]
    # Use xarray's dt accessors (requires pandas)
    hour = time.dt.hour.astype("float32")
    minute = time.dt.minute.astype("float32")
    second = time.dt.second.astype("float32")
    day_of_year = time.dt.dayofyear.astype("float32")

    # Fraction of day [0, 1)
    frac_of_day = (hour * 3600 + minute * 60 + second) / (24.0 * 3600.0)
    # Fraction of year [0, 1)
    frac_of_year = (day_of_year - 1 + frac_of_day) / 365.0

    # Angles
    two_pi = 2.0 * np.pi
    day_angle = two_pi * frac_of_day
    year_angle = two_pi * frac_of_year

    # Broadcast along lon to get (time, lon) shapes
    if "lon" in ds.dims:
        lon = ds["lon"]
        day_cos = np.cos(day_angle.values)[:, None] * np.ones(
            (1, lon.sizes["lon"]), dtype=np.float32
        )
        day_sin = np.sin(day_angle.values)[:, None] * np.ones(
            (1, lon.sizes["lon"]), dtype=np.float32
        )
        da_day_cos = xr.DataArray(
            day_cos,
            dims=("time", "lon"),
            coords={"time": time, "lon": lon},
            name="day_progress_cos",
        )
        da_day_sin = xr.DataArray(
            day_sin,
            dims=("time", "lon"),
            coords={"time": time, "lon": lon},
            name="day_progress_sin",
        )
    else:
        da_day_cos = xr.DataArray(
            np.cos(day_angle.values).astype(np.float32),
            dims=("time",),
            coords={"time": time},
            name="day_progress_cos",
        )
        da_day_sin = xr.DataArray(
            np.sin(day_angle.values).astype(np.float32),
            dims=("time",),
            coords={"time": time},
            name="day_progress_sin",
        )

    da_year_cos = xr.DataArray(
        np.cos(year_angle.values).astype(np.float32),
        dims=("time",),
        coords={"time": time},
        name="year_progress_cos",
    )
    da_year_sin = xr.DataArray(
        np.sin(year_angle.values).astype(np.float32),
        dims=("time",),
        coords={"time": time},
        name="year_progress_sin",
    )

    ds = ds.assign(
        day_progress_cos=da_day_cos,
        day_progress_sin=da_day_sin,
        year_progress_cos=da_year_cos,
        year_progress_sin=da_year_sin,
    )
    return ds


def load_merged_month(
    pressure_path: Path,
    single_path: Path,
    static_ds: xr.Dataset,
) -> xr.Dataset:
    """
    Load and merge *one month* of ERA5 data into a GraphCast-style dataset.

    - pressure_path: era5_pressure_levels_YYYYMM_2.50deg.nc
    - single_path:   era5_single_levels_YYYYMM_2.50deg.nc
    - static_ds:     static fields with dims (lat, lon)

    Output dims: time, level, lat, lon
    """

    # --- Standardize individual monthly files ---
    ds_p = _standardize_pressure_file(pressure_path)   # (time, level, lat, lon)
    ds_s = _standardize_single_file(single_path)       # (time, lat, lon)

    # Basic shape sanity
    for d in ("time", "lat", "lon"):
        if d not in ds_p.dims or d not in ds_s.dims:
            raise ValueError(f"Missing dim {d} in pressure or single dataset.")
        if ds_p.sizes[d] != ds_s.sizes[d]:
            raise ValueError(
                f"Dim {d} size mismatch between pressure ({ds_p.sizes[d]}) "
                f"and single ({ds_s.sizes[d]})."
            )

    # Sort both on lon/lat in the same way (ERA5 is usually already sorted)
    ds_p = ds_p.sortby("lon")
    ds_s = ds_s.sortby("lon")
    ds_p = ds_p.sortby("lat")
    ds_s = ds_s.sortby("lat")

    # Force single-level coords to exactly match pressure-level coords
    ds_s = ds_s.assign_coords(
        time=("time", ds_p["time"].values),
        lat=("lat", ds_p["lat"].values),
        lon=("lon", ds_p["lon"].values),
    )

    # Just in case: deduplicate lon values consistently
    lon_vals = ds_p["lon"].values
    unique_lon, unique_idx = np.unique(lon_vals, return_index=True)
    if len(unique_idx) != lon_vals.size:
        unique_idx = np.sort(unique_idx)
        ds_p = ds_p.isel(lon=unique_idx)
        ds_s = ds_s.isel(lon=unique_idx)

    # --- Merge pressure + single-level vars without clever alignment ---
    ds = xr.merge(
        [ds_p, ds_s],
        compat="override",
        join="override",
    )

    # --- Broadcast static fields over time ---
    # static_ds has dims (lat, lon); we want (time, lat, lon)
    static_b = static_ds.expand_dims(time=ds["time"])
    static_b = static_b.transpose("time", "lat", "lon")

    ds = xr.merge(
        [ds, static_b],
        compat="override",
        join="left",
    )

    # --- Add time features (day/year progress) ---
    ds = _add_time_features(ds)

    # --- Make time coords compatible with GraphCast toy dataset ---
    ds = _ensure_graphcast_time_coords(ds)

    return ds



# --------------------------------------------------------------------------------------
# Indexed sample source (Sequence) for Grain
# --------------------------------------------------------------------------------------


@dataclass
class Era5SampleSource(Sequence):
    """
    A random-access collection of GenCast training/eval samples constructed from
    local ERA5 monthly files.

    Modes:
      - mode="train":
          * Each element is one 3-step window [t-1, t, t+1] within a month.
          * We call data_utils.extract_inputs_targets_forcings with
            target_lead_times="12h", matching the toy GCS setup.
      - mode="test":
          * Each element is a *whole month*; we prepare inputs/targets/forcings
            for multi-step autoregressive evaluation (like the toy script's
            eval_* extraction).

    Elements are returned as dicts:
      {
        "inputs":  xarray.Dataset,
        "targets": xarray.Dataset,
        "forcings": xarray.Dataset,
      }
    """

    era5_root_dir: str
    task_config: gencast.graphcast.TaskConfig = gencast.TASK
    mode: Literal["train", "test"] = "train"
    resolution_deg: float = 2.50
    target_lead_time: str = "12h"

    # Optional restriction on months; if empty, use all.
    months: Optional[List[str]] = None  # e.g. ["199506", "199507"]

    def __post_init__(self):
        self._month_tuples: List[Tuple[str, Path, Path]] = _find_month_files(
            self.era5_root_dir,
            resolution_deg=self.resolution_deg,
        )
        if self.months:
            allowed = set(self.months)
            self._month_tuples = [
                t for t in self._month_tuples if t[0] in allowed
            ]
            if not self._month_tuples:
                raise ValueError(
                    f"No ERA5 months matching {self.months} "
                    f"in {self.era5_root_dir}"
                )

        self._static = _load_static(
            self.era5_root_dir,
            resolution_deg=self.resolution_deg,
        )

        # Cache for current month data
        self._cached_month_id: Optional[str] = None
        self._cached_ds: Optional[xr.Dataset] = None

        # Build global index over months and time positions
        self._indices: List[Tuple[int, int]] = []  # (month_idx, time_idx_or_-1)

        if self.mode == "train":
            self._build_train_index()
        elif self.mode == "test":
            self._build_test_index()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    # --------------------------- index builders ---------------------------

    def _open_month_quick(self, month_idx: int) -> xr.Dataset:
        """Load & merge a month; used internally (cached)."""
        month_id, p_path, s_path = self._month_tuples[month_idx]
        if self._cached_month_id == month_id and self._cached_ds is not None:
            return self._cached_ds

        ds = load_merged_month(p_path, s_path, self._static)
        self._cached_month_id = month_id
        self._cached_ds = ds
        return ds

    def _build_train_index(self):
        """
        For mode='train': create index over all valid central time positions
        within each month.

        We want 3-step windows [t-1, t, t+1], so valid central indices are
        1..(n_time-2) for each month.
        """
        self._indices.clear()
        for mi, (month_id, p_path, s_path) in enumerate(self._month_tuples):
            # Quick open to get time length
            ds = self._open_month_quick(mi)
            n_time = ds.sizes["time"]
            if n_time < 3:
                continue
            for t in range(1, n_time - 1):
                self._indices.append((mi, t))

        if not self._indices:
            raise RuntimeError(
                "No valid training windows found (need at least 3 timesteps "
                "per month)."
            )

    def _build_test_index(self):
        """
        For mode='test': each element is one whole month.
        We just index by month, with time_idx = -1 as sentinel.
        """
        self._indices = [(mi, -1) for mi in range(len(self._month_tuples))]
        if not self._indices:
            raise RuntimeError("No ERA5 months available for test mode.")

    # ------------------------ Sequence interface -------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, xr.Dataset]:
        month_idx, t_idx = self._indices[idx]
        ds = self._open_month_quick(month_idx)

        if self.mode == "train":
            return self._make_train_sample(ds, t_idx)
        else:
            return self._make_test_sample(ds)

    # ------------------------ sample construction ------------------------

    def _make_train_sample(
        self,
        month_ds: xr.Dataset,
        center_t: int,
    ) -> Dict[str, xr.Dataset]:
        """
        Single training sample:
          window = month_ds.isel(time=[center_t-1, center_t, center_t+1]),
          then use GraphCast's extract_inputs_targets_forcings
          with target_lead_times="12h".
        """
        # 3-step window: t-1, t, t+1
        window = month_ds.isel(time=slice(center_t - 1, center_t + 2))

        # Add batch dim like toy GCS dataset (batch=1)
        if "batch" not in window.dims:
            window = window.expand_dims(batch=[0])

        tc = self.task_config
        train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
            window,
            target_lead_times=slice(self.target_lead_time, self.target_lead_time),
            input_variables=tc.input_variables,
            target_variables=tc.target_variables,
            forcing_variables=tc.forcing_variables,
            pressure_levels=tc.pressure_levels,
            input_duration=tc.input_duration,
        )

        return {
            "inputs": train_inputs,
            "targets": train_targets,
            "forcings": train_forcings,
        }

    def _make_test_sample(
        self,
        month_ds: xr.Dataset,
    ) -> Dict[str, xr.Dataset]:
        """
        Test sample for autoregressive evaluation.

        Mimics the toy example's eval_* extraction:
          target_lead_times=slice("12h", f"{(T-2)*12}h"), where T = #time steps
        """
        # Ensure we have a batch dimension of size 1, but don't duplicate it.
        if "batch" in month_ds.dims:
            ds = month_ds
        else:
            ds = month_ds.expand_dims(batch=[0])

        T = ds.sizes["time"]
        if T < 3:
            raise ValueError(
                f"Month has too few timesteps for eval (got {T}, need >=3)."
            )

        last_lead_hours = (T - 2) * 12
        lead_slice = slice("12h", f"{last_lead_hours}h")

        tc = self.task_config
        eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=lead_slice,
            input_variables=tc.input_variables,
            target_variables=tc.target_variables,
            forcing_variables=tc.forcing_variables,
            pressure_levels=tc.pressure_levels,
            input_duration=tc.input_duration,
        )

        return {
            "inputs": eval_inputs,
            "targets": eval_targets,
            "forcings": eval_forcings,
        }


# --------------------------------------------------------------------------------------
# Grain helpers
# --------------------------------------------------------------------------------------


def create_era5_grain_dataset(
    era5_root_dir: str,
    *,
    split: Literal["train", "test"],
    task_config: gencast.graphcast.TaskConfig = gencast.TASK,
    resolution_deg: float = 2.50,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
) -> grain.IterDataset[Dict[str, xr.Dataset]]:
    """
    Create a Grain dataset that yields (inputs, targets, forcings) batches.

    - For split="train":
        Dataset elements are *single samples* (3-step windows).
        We shuffle and batch them.
    - For split="test":
        Dataset elements are full-month sequences, *not* batched
        (batch_size is ignored).

    Returns:
      - train: IterDataset where each element is a batch dict:
          {"inputs": Dataset, "targets": Dataset, "forcings": Dataset}
      - test:  IterDataset over months, unbatched.
    """
    src = Era5SampleSource(
        era5_root_dir=era5_root_dir,
        task_config=task_config,
        mode=split,
        resolution_deg=resolution_deg,
    )

    ds = grain.MapDataset.source(src)

    if split == "train":
        if shuffle:
            ds = ds.shuffle(seed=seed)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        iter_ds = ds.to_iter_dataset()
    else:
        # test: keep one month per element; no batching
        if shuffle:
            ds = ds.shuffle(seed=seed)
        iter_ds = ds.to_iter_dataset()

    return iter_ds


# --------------------------------------------------------------------------------------
# Simple manual debug if run as a script
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick ERA5 local debug / comparison to toy GCS format."
    )
    parser.add_argument(
        "--era5_root",
        type=str,
        required=True,
        help="Directory containing era5_*_YYYYMM_2.50deg.nc files.",
    )
    args = parser.parse_args()

    print("Discovering months...")
    months = _find_month_files(args.era5_root, resolution_deg=2.50)
    print(f"Found {len(months)} months.")
    ym0, p0, s0 = months[0]
    print(f"First month: {ym0}")
    static_ds = _load_static(args.era5_root, resolution_deg=2.50)
    print("Static fields:", list(static_ds.data_vars))

    merged = load_merged_month(p0, s0, static_ds)
    print("\nMerged month dataset:")
    print(merged)

    print("\nCreating one training sample via Era5SampleSource...")
    src_train = Era5SampleSource(
        era5_root_dir=args.era5_root,
        mode="train",
    )
    sample = src_train[0]
    print("Train sample inputs dims:", sample["inputs"].dims)
    print("Train sample targets dims:", sample["targets"].dims)
    print("Train sample forcings dims:", sample["forcings"].dims)

    print("\nCreating one test sample (full month)...")
    src_test = Era5SampleSource(
        era5_root_dir=args.era5_root,
        mode="test",
    )
    test_sample = src_test[0]
    print("Test inputs dims:", test_sample["inputs"].dims)
    print("Test targets dims:", test_sample["targets"].dims)
    print("Test forcings dims:", test_sample["forcings"].dims)

    print("\nDone.")
