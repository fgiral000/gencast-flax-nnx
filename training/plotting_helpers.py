"""
Plotting utilities for GenCast ERA5 evaluation using earthkit-plots.

Main features:
    - plot_variable_triptych(): Pred / GT / Error
    - plot_2m_temperature_triptych(): convenience wrapper
    - rollout_to_gif(): create a GIF from autoregressive rollout (global domain)

earthkit-plots (ekp) is required.
"""

from __future__ import annotations
import numpy as np
import xarray as xr
import earthkit.plots as ekp
import imageio.v3 as iio
from typing import Optional, Sequence


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _extract_single_2d(da: xr.DataArray, level: Optional[int] = None) -> xr.DataArray:
    """
    For a DataArray with dims [batch?, time?, level?, lat/lon], select one sample,
    timestep and level, and make it suitable for earthkit-plots Map.quickplot.

    Also drops any timedelta64 coordinates to avoid wrapper warnings.
    """
    sel = da

    # Grab first batch element if present
    if "batch" in sel.dims:
        sel = sel.isel(batch=0, drop=True)

    # Collapse time
    if "time" in sel.dims:
        sel = sel.isel(time=0, drop=True)

    # Handle level dimension
    if "level" in sel.dims:
        if level is not None:
            sel = sel.isel(level=level, drop=True)
        else:
            sel = sel.isel(level=0, drop=True)

    # Drop any timedelta64 coords (e.g. lead_time, step)
    drop_coords = []
    for c in sel.coords:
        try:
            if np.issubdtype(sel[c].dtype, np.timedelta64):
                drop_coords.append(c)
        except TypeError:
            pass
    if drop_coords:
        sel = sel.drop_vars(drop_coords)

    # Normalise spatial dims
    if "lat" in sel.dims and "lon" in sel.dims:
        sel = sel.transpose("lat", "lon", ...)
    elif "latitude" in sel.dims and "longitude" in sel.dims:
        sel = sel.transpose("latitude", "longitude", ...)
    else:
        raise ValueError(
            "DataArray must have ('lat','lon') or ('latitude','longitude') dims."
        )

    return sel


def _lat_lon(ds: xr.Dataset):
    if "lat" not in ds.coords and "latitude" not in ds.coords:
        raise ValueError("Dataset must have 'lat' or 'latitude' coords.")
    if "lon" not in ds.coords and "longitude" not in ds.coords:
        raise ValueError("Dataset must have 'lon' or 'longitude' coords.")
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    return np.asarray(ds[lat_name]), np.asarray(ds[lon_name])


# ---------------------------------------------------------------------
# Triptych plotting
# ---------------------------------------------------------------------

def plot_variable_triptych(
    preds: xr.Dataset,
    targets: xr.Dataset,
    variable: str,
    *,
    units: str = "",
    domain: str = "global",
    level: Optional[int] = None,
):
    """
    Create a 1x3 Pred / GT / Error map using earthkit-plots.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure suitable for wandb.Image, or None if variable not present.
    """
    if variable not in preds or variable not in targets:
        return None

    pred_da = _extract_single_2d(preds[variable], level=level)
    targ_da = _extract_single_2d(targets[variable], level=level)
    err_da = pred_da - targ_da

    # Sanity check: ensure lat/lon exist
    _lat_lon(preds)

    # Figure with 3 map subplots
    figure = ekp.Figure(rows=1, columns=3, size=[14, 4], domain=domain)

    m_pred = figure.add_map(domain=domain)
    m_targ = figure.add_map(domain=domain)
    m_err = figure.add_map(domain=domain)

    # Use Map.quickplot as per docs – no dict style object!
    m_pred.quickplot(pred_da, units=units)
    m_targ.quickplot(targ_da, units=units)
    m_err.quickplot(err_da, units=units)

    m_pred.title(f"{variable} - Predicted")
    m_targ.title(f"{variable} - Ground truth")
    m_err.title(f"{variable} - Error (Pred - GT)")

    # Common map layers
    figure.coastlines()
    figure.gridlines()

    # Correct legend API: keyword argument 'location'
    figure.legend(location="bottom")

    # Ensure the underlying Matplotlib figure is instantiated
    figure.draw()

    # earthkit-plots Figure.fig is a Matplotlib Figure
    return figure.fig


def plot_2m_temperature_triptych(
    preds: xr.Dataset,
    targets: xr.Dataset,
    *,
    domain: str = "Europe",
):
    """Convenience wrapper for common 2m temperature names."""
    for name in ["2m_temperature", "t2m", "2t"]:
        if name in preds:
            return plot_variable_triptych(preds, targets, name, units="K", domain=domain)
    return None


# ---------------------------------------------------------------------
# GIF rendering for autoregressive rollout
# ---------------------------------------------------------------------

def rollout_to_gif(
    rollout: xr.Dataset,
    *,
    variable: str,
    outfile: str,
    domain: str = "global",
    fps: int = 4,
    level: Optional[int] = None,
    title_prefix: Optional[str] = None,
):
    """
    Create a GIF from a multi-step autoregressive rollout.

    We skip frames where the selected variable is entirely NaN/Inf,
    because earthkit-plots (via SciPy's griddata) cannot interpolate
    with zero valid points.
    """
    if variable not in rollout:
        raise ValueError(f"Variable '{variable}' not found in rollout dataset!")

    da = rollout[variable]
    if "time" not in da.dims:
        raise ValueError("Rollout must have 'time' dimension.")

    frames: Sequence[np.ndarray] = []
    nt = da.sizes["time"]

    for t in range(nt):
        da_t = da.isel(time=t)
        da_2d = _extract_single_2d(da_t, level=level)

        # Check if there is any finite data at this timestep
        vals = np.asarray(da_2d.values)
        if not np.isfinite(vals).any():
            print(
                f"[rollout_to_gif] frame {t}: all values are NaN/Inf for "
                f"variable '{variable}', skipping this frame."
            )
            continue

        figure = ekp.Figure(rows=1, columns=1, size=[8, 4], domain=domain)
        m = figure.add_map(domain=domain)
        m.quickplot(da_2d)

        # Title
        if title_prefix is None:
            m.title(f"{variable} - t={t}")
        else:
            m.title(f"{title_prefix} - t={t}")

        figure.coastlines()
        figure.gridlines()
        figure.legend(location="bottom")

        # Render & convert to RGB array
        figure.draw()
        canvas = figure.fig.canvas
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)

        # Clean up matplotlib figure to avoid memory build-up
        import matplotlib.pyplot as plt
        plt.close(figure.fig)

    if not frames:
        raise ValueError(
            f"rollout_to_gif: No valid frames for variable '{variable}'. "
            "All timesteps appear to be NaN/Inf. "
            "Check your autoregressive_rollout outputs and training stability."
        )

    iio.imwrite(outfile, frames, fps=fps)
    print(f"✔ GIF saved to: {outfile}")

