#!/usr/bin/env python3
"""
Inference / plotting script for a trained GenCast model.

- Loads a GenCast model and restores weights from a checkpoint directory.
- Takes one sample from the ERA5 test split.
- Runs full_testing(), which now performs true forecast-time
  autoregressive rollout using single-step predictions.
- Plots user-selected variables and creates a GIF from the AR rollout.

Example
-------
python run_gencast_inference.py \
  --load_checkpoint ./checkpoints/gencast_model/checkpoint_step_5000 \
  --era5_root_dir data_era5 \
  --variables 2m_temperature \
  --gif_variable 2m_temperature \
  --max_rollout_steps 12 \
  --sample_index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt

import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp
import numpy as np
import xarray as xr

from gencast import gencast

from training.era5_dataset import Era5SampleSource
from training.train_helpers import (
    create_gencast_model,
    maybe_wrap_with_nan_cleaner,
    maybe_wrap_with_normalization,
    autoregressive_rollout,
    to_jax_dataset,
)
from training.plotting_helpers import (
    plot_variable_triptych,
    rollout_to_gif,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def build_model_from_flags(args: argparse.Namespace) -> gencast.GenCast:
    """Create the GenCast model and wrap with optional cleaners/normalization."""
    # rngs = nnx.Rngs(args.seed)

    model, _, _, _, _ = create_gencast_model(
        task_config=gencast.TASK,
        rngs=nnx.Rngs(args.seed),
        mesh_size=args.mesh_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        stochastic_churn_rate=0.0,
    )

    model = maybe_wrap_with_nan_cleaner(
        model, clean_sst_nans=args.clean_sst_nans
    )
    model = maybe_wrap_with_normalization(
        model,
        apply_normalization=args.apply_normalization,
        normalization_stats_dir=args.normalization_stats_dir,
    )
    return model


def parse_variable_list(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


# def restore_model_from_checkpoint(model, ckpt_path: str):
#     """
#     Restore parameters into `model` using the user-provided pattern.

#     ckpt_path should point to the checkpoint directory itself, e.g.
#       ./checkpoints/gencast_model/checkpoint_step_5000
#     """
#     checkpoint_dir = Path(ckpt_path)

#     abs_graphdef, abs_rng_state, abs_other_state = nnx.split(
#         model, nnx.RngState, ...
#     )
#     ckptr = ocp.PyTreeCheckpointer()
#     ckpt_dir = checkpoint_dir.expanduser().resolve()

#     restored_state = ckptr.restore(str(ckpt_dir), item=abs_other_state)

#     # --- DEBUGGING BLOCK ---
#     print("Restored State Type:", type(restored_state))
#     # Look at the leaves. Are they arrays or tuples of arrays?
#     first_leaf = jax.tree_util.tree_leaves(restored_state)[0]
#     print(f"First leaf type: {type(first_leaf)}")
#     if isinstance(first_leaf, (tuple, list)):
#          print("MISMATCH DETECTED: Leaves are wrapped in tuples/lists.")
#     # -----------------------

#     nnx.update(model, restored_state)
#     print(f"✔ Restored parameters from {ckpt_dir}")

#     model.eval()
#     return model

def restore_model_from_checkpoint(model, ckpt_path: str):
    """
    Restore parameters with specific structural unwrapping for GenCast.
    """
    checkpoint_dir = Path(ckpt_path)

    # 1. Create abstract state
    abs_graphdef, abs_rng_state, abs_other_state = nnx.split(
        model, nnx.RngState, ...
    )
    
    ckptr = ocp.PyTreeCheckpointer()
    ckpt_dir = checkpoint_dir.expanduser().resolve()

    print(f"Loading raw state from {ckpt_dir}...")
    restored_state = ckptr.restore(str(ckpt_dir), item=abs_other_state)

    # --- CLEANING LOGIC ---
    def clean_state(obj):
        if isinstance(obj, (dict, nnx.State)):
            # 1. Drop Normalization Stats (if looks like a dataset)
            if "10m_u_component_of_wind" in obj or "2m_temperature" in obj:
                return None

            # 2. STRUCTURAL FIX: Unwrap 'graph_network'
            # If this dict contains 'graph_network', we assume it's an extra wrapper.
            # We extract the content and process THAT instead.
            if 'graph_network' in obj:
                print("  ⚠ Unwrap: Detected 'graph_network' wrapper. Hoisting contents up one level...")
                return clean_state(obj['graph_network'])

            new_dict = {}
            for k, v in obj.items():
                # 3. Drop static/private buffers (starting with _)
                if isinstance(k, str) and k.startswith('_'):
                    continue
                
                # Recurse
                cleaned_v = clean_state(v)
                if cleaned_v is not None:
                    new_dict[k] = cleaned_v
            
            # 4. Unwrap Singletons {0: ...} -> ...
            # This handles the case where a Layer is wrapped in a single-key dict
            keys = list(new_dict.keys())
            if len(keys) == 1 and (keys[0] == 0 or keys[0] == '0'):
                # Only unwrap if the content is likely a Variable (leaf array)
                # If it's a complex dict, it might be a Layer in a List, so we keep the 0.
                val = new_dict[keys[0]]
                if not isinstance(val, (dict, nnx.State, list, tuple)):
                     return val
            
            return new_dict

        # Handle Lists/Tuples
        if isinstance(obj, (list, tuple)) and len(obj) == 1:
            return clean_state(obj[0])

        return obj

    print("Cleaning state...")
    final_state = clean_state(restored_state)

    print("Updating model weights...")
    nnx.update(model, final_state)
    
    print(f"Restored parameters successfully from {ckpt_dir}")
    model.eval()
    return model


# ---------------------------------------------------------------------
# NetCDF Saving Helper
# ---------------------------------------------------------------------

def save_rollout_to_netcdf(
    preds,
    targets_future,
    outfile: Path,
    mse: float,
    context_steps: int,
    compression_level: int = 4,
):
    """Persist autoregressive rollout predictions (and matching targets) to NetCDF.

    Parameters
    ----------
    preds : xarray.DataArray | xarray.Dataset
        Autoregressive forecast horizon produced by `autoregressive_rollout`.
    targets_future : xarray.DataArray | xarray.Dataset
        Matching future targets horizon (same time length ideally).
    outfile : Path
        Destination .nc file path.
    mse : float
        Forecast MSE for logging in attributes.
    context_steps : int
        Number of initial context steps used for rollout.
    compression_level : int, default 4
        zlib compression level (0-9). Trade-off between size and speed.
    """

    # Normalize to Dataset form.
    if isinstance(preds, xr.DataArray):
        ds_pred = preds.to_dataset(name="prediction")
    else:
        ds_pred = preds

    if isinstance(targets_future, xr.DataArray):
        ds_tgt = targets_future.to_dataset(name="target")
    else:
        # rename each target var to avoid collision if same names as preds
        rename_map = {v: f"target_{v}" for v in targets_future.data_vars}
        ds_tgt = targets_future.rename(rename_map)

    # Ensure numpy-backed memory (NetCDF writer may not like JAX device arrays).
    def _ensure_numpy(ds: xr.Dataset) -> xr.Dataset:
        for v in ds.data_vars:
            ds[v].data = np.asarray(ds[v].data)
        return ds

    ds_pred = _ensure_numpy(ds_pred)
    ds_tgt = _ensure_numpy(ds_tgt)

    # Merge datasets.
    combined = xr.merge([ds_pred, ds_tgt], compat="override")

    # Global attrs.
    combined.attrs.update(
        dict(
            description="GenCast autoregressive rollout",
            mse=f"{mse:.6e}",
            context_steps=context_steps,
        )
    )

    # Compression encoding per variable.
    encoding = {
        var: {"zlib": True, "complevel": compression_level} for var in combined.data_vars
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(outfile, encoding=encoding)
    # Optional lightweight verification (avoid heavy printing).
    opened = xr.open_dataset(outfile)
    print(
        f"✔ Saved rollout NetCDF to {outfile} | vars={list(opened.data_vars)} | time={opened.dims.get('time', 'N/A')}"
    )
    opened.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GenCast inference + plotting")

    # --- Checkpoint / model config ------------------------------------------
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        required=True,
        help=(
            "Path to checkpoint directory "
            "(e.g. ./checkpoints/gencast_model/checkpoint_step_5000)"
        ),
    )

    parser.add_argument("--mesh_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    # --- Preprocessing wrappers ---------------------------------------------
    parser.add_argument(
        "--clean_sst_nans", action="store_true",
        help="Use same setting as in training."
    )
    parser.add_argument(
        "--apply_normalization", action="store_true",
        help="Use same setting as in training."
    )
    parser.add_argument(
        "--normalization_stats_dir", type=str,
        default="./normalization_stats"
    )

    # --- Data source --------------------------------------------------------
    parser.add_argument(
        "--era5_root_dir", type=str, default="data_era5",
        help="Root directory with ERA5 monthly NetCDF files."
    )
    parser.add_argument(
        "--resolution_deg", type=float, default=2.5,
        help="ERA5 resolution in degrees (must match training)."
    )
    parser.add_argument(
        "--sample_index", type=int, default=0,
        help="Which test sample (month) to use."
    )

    # --- Rollout / plotting options ----------------------------------------
    parser.add_argument(
        "--context_steps", type=int, default=2,
        help="Number of context steps (e.g. model trained on 2)."
    )
    parser.add_argument(
        "--max_rollout_steps", type=int, default=10,
        help="Max future steps to roll out autoregressively."
    )
    parser.add_argument(
        "--variables", type=str, default="2m_temperature",
        help="Comma-separated list of variables to plot (must exist in preds/targets)."
    )
    parser.add_argument(
        "--domain", type=str, default="Europe",
        help="earthkit-plots domain (e.g. 'global', 'Europe')."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_plots",
        help="Where to write PNGs and GIFs."
    )

    parser.add_argument(
        "--gif_variable", type=str, default=None,
        help="Single variable to use for GIF (must exist in preds)."
    )
    parser.add_argument(
        "--gif_out", type=str, default="rollout.gif",
        help="Filename for GIF inside output_dir."
    )
    parser.add_argument(
        "--gif_fps", type=int, default=4,
        help="Frames per second for the GIF."
    )

    # --- NetCDF saving options ---------------------------------------------
    parser.add_argument(
        "--save_rollout_nc", action="store_true",
        help="If set, save the full autoregressive rollout (preds + targets) to NetCDF."
    )
    parser.add_argument(
        "--rollout_nc_out", type=str, default="rollout.nc",
        help="Filename for saved NetCDF (placed inside output_dir)."
    )
    parser.add_argument(
        "--netcdf_compression_level", type=int, default=4,
        help="zlib compression level (0-9) for NetCDF output."
    )

    args = parser.parse_args()

    # ------------------------- Build + restore model ------------------------
    print("Building GenCast model...")
    model = build_model_from_flags(args)

    # --- Build a dataset and do a warm-up forward pass (like training) -----
    print("Creating ERA5 source for warm-up...")
    warmup_source = Era5SampleSource(
        era5_root_dir=args.era5_root_dir,
        task_config=gencast.TASK,
        mode="train",          # or "test" – just needs to match training config
        resolution_deg=args.resolution_deg,
    )
    first_sample = warmup_source[0]
    warm_inp = to_jax_dataset(first_sample["inputs"])
    warm_tgt = to_jax_dataset(first_sample["targets"])
    warm_frc = to_jax_dataset(first_sample["forcings"])

    print("Running warm-up forward pass to initialize model params...")
    loss_da, _ = model.loss(
        inputs=warm_inp,
        targets=warm_tgt,
        forcings=warm_frc,
    )
    _ = jax.device_get(loss_da)
    print("Warm-up forward pass completed.")

    # --- NOW restore from checkpoint, after graph is fully instantiated -----
    print(f"Restoring model from checkpoint: {args.load_checkpoint}")
    model = restore_model_from_checkpoint(model, args.load_checkpoint)
    print("Model restored. Devices:", jax.devices())

    # ------------------------- Test dataset ---------------------------------
    print("Creating ERA5 test source...")
    test_source = Era5SampleSource(
        era5_root_dir=args.era5_root_dir,
        task_config=gencast.TASK,
        mode="test",
        resolution_deg=args.resolution_deg,
    )

    if args.sample_index < 0 or args.sample_index >= len(test_source):
        raise IndexError(
            f"sample_index {args.sample_index} out of range "
            f"(0..{len(test_source) - 1})"
        )

    batch = test_source[args.sample_index]
    print(
        f"Using test sample index {args.sample_index}: "
        f"{batch['inputs'].time.values}"
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------- Autoregressive full_testing ------------------
    print(
        f"Running full_testing (single-step autoregressive) with "
        f"max_rollout_steps={args.max_rollout_steps} ..."
    )
    mse, preds, targets_future = autoregressive_rollout(
        model,
        batch=batch,
        context_steps=args.context_steps,
        max_rollout_steps=args.max_rollout_steps,
        teacher_forcing=True,
    )
    print(f"[full_testing] Future-horizon MSE (AR rollout): {mse:.6e}")
    print(
        f"Preds horizon: {preds.sizes.get('time', 'N/A')} steps, "
        f"Targets horizon: {targets_future.sizes.get('time', 'N/A')} steps"
    )

    # ------------------------- Optional NetCDF save ------------------------
    if args.save_rollout_nc:
        nc_path = outdir / args.rollout_nc_out
        print(f"Saving rollout to NetCDF: {nc_path}")
        save_rollout_to_netcdf(
            preds=preds,
            targets_future=targets_future,
            outfile=nc_path,
            mse=mse,
            context_steps=args.context_steps,
            compression_level=args.netcdf_compression_level,
        )
    else:
        print("Skipping NetCDF save (use --save_rollout_nc to enable).")

    # ------------------------- Triptych plots -------------------------------
    var_list = parse_variable_list(args.variables)
    print(f"Creating triptych plots for variables: {var_list}")

    for var in var_list:
        fig = plot_variable_triptych(
            preds=preds,
            targets=targets_future,
            variable=var,
            domain=args.domain,
        )
        if fig is None:
            print(f"  [skip] Variable '{var}' not found in preds/targets.")
            continue

        png_path = outdir / f"triptych_{var}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved triptych for '{var}' to {png_path}")

    # ------------------------- GIF rollout ---------------------------------
    if args.gif_variable is not None:
        gif_var = args.gif_variable
        gif_path = outdir / args.gif_out
        print(f"Creating GIF for variable '{gif_var}' -> {gif_path}")

        rollout_to_gif(
            rollout=preds,             # AR rollout from full_testing
            variable=gif_var,
            outfile=str(gif_path),
            domain=args.domain,
            fps=args.gif_fps,
            level=None,
            title_prefix=f"{gif_var} rollout",
        )
    else:
        print("No --gif_variable specified; skipping GIF creation.")

    print("Inference + plotting completed.")


if __name__ == "__main__":
    main()
