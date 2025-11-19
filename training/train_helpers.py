"""
Helper utilities for training GenCast on ERA5 data.

This module factors out:
  * Model construction and optional wrappers (NaN cleaning, normalization)
  * Loss / JITted train & eval steps
  * Optimizer + LR schedule
  * Checkpoint I/O
  * Full-testing (full denoising / sampling) on a test batch
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax
import xarray as xr
import orbax.checkpoint as ocp

from gencast import gencast
from gencast import denoiser as denoiser_mod
from gencast import nan_cleaning as nan_cleaning_mod
from common import normalization as normalization_mod
from common import xarray_jax
from common import data_utils

# -----------------------------------------------------------------------------
# Basic xarray / JAX helpers
# -----------------------------------------------------------------------------


def to_jax_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Convert all data_vars to JAX arrays, keep coords static (NumPy).

    This is useful to ensure everything inside the Dataset is on-device and
    compatible with jit / pmap.
    """
    data_vars = {
        name: (da.dims, jax.numpy.asarray(xarray_jax.unwrap_data(da)))
        for name, da in ds.data_vars.items()
    }
    return xarray_jax.Dataset(data_vars, coords=ds.coords, attrs=ds.attrs)


def _pack_ds_to_arrays(ds: xr.Dataset) -> Dict[str, jax.Array]:
    """Extract JAX arrays from a (NumPy-backed) Dataset for JIT-friendly arguments.

    Unlike the original implementation, we don't require that `ds` is already
    JAX-backed. We simply take the underlying data and wrap it with
    `jnp.asarray`, which works whether it is NumPy or JAX.
    """
    return {name: jnp.asarray(da.data) for name, da in ds.data_vars.items()}




def _unpack_arrays_to_ds(
    arrs: Dict[str, jax.Array], template: xr.Dataset
) -> xr.Dataset:
    """Rebuild a Dataset from JAX arrays using dims from template.

    We avoid passing per-variable coords to xr.DataArray to sidestep
    strict coord/shape validation; global coords from `template.coords`
    are enough for the model + xarray_jax.
    """
    data_vars: Dict[str, Any] = {}
    for name, tmpl_da in template.data_vars.items():
        dims = tmpl_da.dims
        data_vars[name] = xr.DataArray(
            arrs[name],
            dims=dims,
            attrs=tmpl_da.attrs,
        )

    # Attach the same global coords/attrs as in the template.
    ds = xr.Dataset(data_vars, coords=template.coords, attrs=template.attrs)
    return ds




# -----------------------------------------------------------------------------
# Model construction & wrappers
# -----------------------------------------------------------------------------


def create_gencast_model(
    task_config: gencast.graphcast.TaskConfig,
    rngs: nnx.Rngs,
    *,
    mesh_size: int = 3,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    stochastic_churn_rate: float = 0.0,
) -> tuple[
    gencast.GenCast,
    denoiser_mod.DenoiserArchitectureConfig,
    gencast.SamplerConfig,
    gencast.NoiseConfig,
    Optional[denoiser_mod.NoiseEncoderConfig],
]:
    """Create a GenCast model with a valid architecture and configs."""

    sampler_config = gencast.SamplerConfig(
        max_noise_level=80.0,
        min_noise_level=0.03,
        num_noise_levels=20,
        rho=7.0,
        stochastic_churn_rate=float(stochastic_churn_rate),
    )

    noise_config = gencast.NoiseConfig(
        training_noise_level_rho=7.0,
        training_max_noise_level=88.0,
        training_min_noise_level=0.02,
    )

    transformer_cfg = denoiser_mod.SparseTransformerConfig(
        attention_k_hop=8,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    denoiser_architecture_config = denoiser_mod.DenoiserArchitectureConfig(
        sparse_transformer_config=transformer_cfg,
        mesh_size=mesh_size,
        latent_size=d_model,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )

    noise_encoder_config = None  # use default inside GenCast

    model = gencast.GenCast(
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        sampler_config=sampler_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
        rngs=rngs,
    )

    return (
        model,
        denoiser_architecture_config,
        sampler_config,
        noise_config,
        noise_encoder_config,
    )


def maybe_wrap_with_nan_cleaner(
    model: gencast.GenCast,
    *,
    clean_sst_nans: bool = False,
) -> gencast.GenCast:
    """Optionally wrap the model with NaNCleaner for SST fields."""
    if not clean_sst_nans:
        return model

    sst_fill = xr.Dataset(
        {"sea_surface_temperature": ((), np.array(0.0, dtype=np.float32))}
    )
    model = nan_cleaning_mod.NaNCleaner(
        predictor=model,
        var_to_clean="sea_surface_temperature",
        fill_value=sst_fill,
        reintroduce_nans=False,
    )
    return model


def maybe_wrap_with_normalization(
    model: gencast.GenCast,
    *,
    apply_normalization: bool = False,
    normalization_stats_dir: str = "./normalization_stats",
) -> gencast.GenCast:
    """Optionally wrap model with InputsAndResiduals normalization."""
    if not apply_normalization:
        return model

    stats_dir = Path(normalization_stats_dir)
    stddev_path = stats_dir / "gencast_stats_stddev_by_level.nc"
    mean_path = stats_dir / "gencast_stats_mean_by_level.nc"
    diffs_stddev_path = stats_dir / "gencast_stats_diffs_stddev_by_level.nc"

    if not (stddev_path.exists() and mean_path.exists() and diffs_stddev_path.exists()):
        raise FileNotFoundError(
            f"Normalization stats not found in {normalization_stats_dir}. Expected:\n"
            f"  {stddev_path.name}, {mean_path.name}, {diffs_stddev_path.name}"
        )

    stddev_ds = xr.load_dataset(stddev_path)
    mean_ds = xr.load_dataset(mean_path)
    diffs_stddev_ds = xr.load_dataset(diffs_stddev_path)

    model = normalization_mod.InputsAndResiduals(
        predictor=model,
        stddev_by_level=stddev_ds,
        mean_by_level=mean_ds,
        diffs_stddev_by_level=diffs_stddev_ds,
    )
    return model


# -----------------------------------------------------------------------------
# Loss + JITted train/eval steps
# -----------------------------------------------------------------------------


def compute_loss(
    model: gencast.GenCast,
    inputs: xr.Dataset,
    targets: xr.Dataset,
    forcings: Optional[xr.Dataset],
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute mean loss over the batch and return diagnostics."""
    loss_da, diagnostics = model.loss(inputs=inputs, targets=targets, forcings=forcings)
    loss_scalar = jnp.mean(xarray_jax.jax_data(loss_da))
    diag_out: Dict[str, Any] = {}
    for k, v in diagnostics.items():
        try:
            diag_out[k] = jnp.mean(xarray_jax.jax_data(v))
        except Exception:
            diag_out[k] = v
    return loss_scalar, diag_out


def make_jitted_steps(
    template_inputs: xr.Dataset,
    template_targets: xr.Dataset,
    template_forcings: Optional[xr.Dataset],
):
    """Factory to build JITted step fns that take only PyTrees of JAX arrays.

    The packed functions work purely on dicts of arrays with the same shapes
    as those in the templates, which makes them easy to feed from a Grain
    iterator.
    """

    @nnx.jit
    def train_step_packed(
        model: gencast.GenCast,
        optimizer: nnx.Optimizer,
        inputs_arrs: Dict[str, jax.Array],
        targets_arrs: Dict[str, jax.Array],
        forcings_arrs: Optional[Dict[str, jax.Array]],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        inputs = _unpack_arrays_to_ds(inputs_arrs, template_inputs)
        targets = _unpack_arrays_to_ds(targets_arrs, template_targets)
        forcings = (
            None
            if forcings_arrs is None
            else _unpack_arrays_to_ds(forcings_arrs, template_forcings)  # type: ignore[arg-type]
        )

        def loss_fn(model: gencast.GenCast):
            return compute_loss(model, inputs, targets, forcings)

        (loss, diagnostics), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        optimizer.update(grads)
        return loss, diagnostics

    @nnx.jit
    def eval_step_packed(
        model: gencast.GenCast,
        inputs_arrs: Dict[str, jax.Array],
        targets_arrs: Dict[str, jax.Array],
        forcings_arrs: Optional[Dict[str, jax.Array]],
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        inputs = _unpack_arrays_to_ds(inputs_arrs, template_inputs)
        targets = _unpack_arrays_to_ds(targets_arrs, template_targets)
        forcings = (
            None
            if forcings_arrs is None
            else _unpack_arrays_to_ds(forcings_arrs, template_forcings)  # type: ignore[arg-type]
        )
        return compute_loss(model, inputs, targets, forcings)

    return train_step_packed, eval_step_packed


# -----------------------------------------------------------------------------
# Optimizer / schedule
# -----------------------------------------------------------------------------


def create_optimizer(
    model: gencast.GenCast,
    num_steps: int,
    learning_rate: float,
    *,
    warmup_fraction: float = 0.1,
    max_warmup_steps: int = 500,
    grad_clip_norm: float = 1.0,
) -> Tuple[nnx.Optimizer, optax.Schedule]:
    """Create AdamW optimizer + warmup+cosine schedule wrapped for nnx."""
    warmup_steps = max(1, min(int(num_steps * warmup_fraction), max_warmup_steps))
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(1, num_steps - warmup_steps),
        alpha=0.0,
    )
    schedule = optax.join_schedules(
        [warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.999, weight_decay=0.1
        ),
    )
    optimizer = nnx.ModelAndOptimizer(model, tx)
    return optimizer, schedule


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------


def save_checkpoint(
    step: int,
    model: gencast.GenCast,
    checkpoint_root: Path,
) -> None:
    """Save parameters using Orbax PyTreeCheckpointer."""
    model.eval()
    checkpoint_path = checkpoint_root / f"checkpoint_step_{step}"

    if checkpoint_path.exists():
        import shutil

        shutil.rmtree(checkpoint_path)

    # Only save non-RNG state; RNG state is regenerated from seed in practice.
    _, rng_state, other_state = nnx.split(model, nnx.RngState, ...)
    del rng_state
    other_state = jax.device_get(other_state)
    ocp.PyTreeCheckpointer().save(str(checkpoint_path), other_state)
    print(f"✔ Checkpoint saved at step {step}")
    model.train()


# -----------------------------------------------------------------------------
# WandB helper
# -----------------------------------------------------------------------------


def init_wandb(use_flag: Optional[bool], config: Dict[str, Any]) -> bool:
    """Initialize wandb if requested / enabled via env, return bool use_wandb."""
    if use_flag is not None:
        use_wandb = bool(use_flag)
    else:
        env_val = os.environ.get("USE_WANDB", "0").strip().lower()
        try:
            use_wandb = bool(int(env_val))
        except Exception:
            use_wandb = env_val in {"1", "true", "yes", "on"}

    if not use_wandb:
        return False

    try:
        import wandb  # type: ignore

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "GenCast"),
            name=os.environ.get("WANDB_RUN_NAME", None),
            config=config,
        )
        return True
    except Exception as e:
        print(f"[wandb] init failed, proceeding without wandb: {e}")
        return False


# -----------------------------------------------------------------------------
# Full testing / denoising (sampling-based evaluation)
# -----------------------------------------------------------------------------


def autoregressive_rollout(
    model,
    *,
    batch: Dict[str, xr.Dataset],
    context_steps: int = 2,
    max_rollout_steps: Optional[int] = None,
    teacher_forcing: bool = False,
) -> Tuple[float, xr.Dataset, xr.Dataset]:
    """
    Single-step rollout on a **test batch**.

    Assumes Era5SampleSource(mode="test") produced:

      - batch["inputs"]  : context timesteps (e.g., time=2: X_{t-1}, X_t)
      - batch["targets"] : future timesteps (time = forecast horizon)
      - batch["forcings"]: forcings for the future timesteps

    Modes
    -----
    teacher_forcing = False  (default, current behavior)
        True forecast-time autoregression:
            context = [ ... previous context_steps frames ... ]
            context is updated with model predictions:
                X_{t-1}, X_t -> X̂_{t+1}
                X_t,   X̂_{t+1} -> X̂_{t+2}
                ...

    teacher_forcing = True   (new behavior)
        Teacher-forced rollout:
            context = ground-truth frames (inputs + GT targets)
            We STILL generate predictions for each future time, but we
            never feed them back to the model:
                X_0, X_1    -> X̂_2
                X_1, X_2    -> X̂_3
                X_2, X_3    -> X̂_4
                ...
        This is useful to see how predictions behave *without* error
        accumulation from feeding back bad forecasts.

    Returns
    -------
    mse : float
        MSE over all variables, timesteps, lat/lon/level for the evaluated
        horizon (may be shorter than targets if max_rollout_steps < horizon).
    preds_rollout : xr.Dataset
        Dataset with dim 'time' = number of predicted steps.
    targets_future : xr.Dataset
        Ground-truth targets for the same horizon as preds_rollout.
    """

    # Convert to JAX-backed xarray Datasets
    inputs_ds = to_jax_dataset(batch["inputs"])
    targets_ds = to_jax_dataset(batch["targets"])
    forcings_ds = to_jax_dataset(batch["forcings"])

    if "time" not in inputs_ds.dims:
        raise ValueError("Expected 'time' dim in inputs for testing batch.")
    if "time" not in targets_ds.dims:
        raise ValueError("Expected 'time' dim in targets for testing batch.")

    # Horizon we will roll out over
    horizon_total = targets_ds.sizes["time"]
    if max_rollout_steps is not None:
        horizon = min(max_rollout_steps, horizon_total)
    else:
        horizon = horizon_total

    # Ensure we have enough context steps
    if inputs_ds.sizes["time"] < context_steps:
        raise ValueError(
            f"Not enough context steps in inputs (have {inputs_ds.sizes['time']}, "
            f"need at least {context_steps})"
        )

    preds_per_step: list[xr.Dataset] = []

    # Variable partitions from task config
    try:
        task_cfg = getattr(model, "_task_config", gencast.TASK)
    except Exception:
        task_cfg = gencast.TASK
    input_vars = set(task_cfg.input_variables)
    target_vars = set(task_cfg.target_variables)
    forcing_vars = set(task_cfg.forcing_variables)
    input_only_vars = input_vars - target_vars - forcing_vars

    def _compose_next_frame(
        target_like: xr.Dataset,
        forcings_like: xr.Dataset,
        prev_context: xr.Dataset,
    ) -> xr.Dataset:
        """Compose a full input frame (time=1) with all input variables.

        - target-like variables come from target_like (predicted or GT),
        - forcing variables come from forcings_like (aligned time=1),
        - input-only variables (e.g., statics) are carried forward from prev_context
          by copying the last available time slice and reindexing to the new time.
        """
        data_vars: Dict[str, xr.DataArray] = {}

        # time coordinate for the new frame (use target_like's time coord)
        if "time" in target_like.coords:
            new_time = target_like.coords["time"]
        else:
            # Fallback: take from forcings_like
            new_time = forcings_like.coords.get("time", None)

        # 1) targets
        for v in target_vars:
            if v in target_like:
                data_vars[v] = target_like[v]

        # 2) forcings
        for v in forcing_vars:
            if v in forcings_like:
                data_vars[v] = forcings_like[v]

        # 3) input-only (e.g., statics or other inputs not predicted)
        last = prev_context.isel(time=-1)
        for v in input_only_vars:
            if v in last:
                da = last[v]
                # expand to time=1 using the new time coord if available
                if new_time is not None:
                    da = da.expand_dims(time=new_time)
                else:
                    da = da.expand_dims(time=[0])
                data_vars[v] = da

        # Build dataset; preserve global coords from target_like where possible
        coords = {}
        # Carry common coords if present
        for c in ["batch", "time", "lat", "lon", "level", "latitude", "longitude"]:
            if c in target_like.coords:
                coords[c] = target_like.coords[c]
            elif c in prev_context.coords:
                # copy and if needed adjust time coord length to 1
                cc = prev_context.coords[c]
                if c == "time" and cc.size != 1:
                    # replace with new_time if available
                    if new_time is not None:
                        coords[c] = new_time
                        continue
                    else:
                        coords[c] = xr.DataArray([cc.values[-1]], dims=("time",))
                        continue
                coords[c] = cc

        return xr.Dataset(data_vars=data_vars, coords=coords)

    if teacher_forcing:
        # ------------------------------------------------------------------
        # TEACHER-FORCED ROLLOUT (use GT for context updates)
        # ------------------------------------------------------------------
        # Start from the actual input context, as in standard AR
        current_context = inputs_ds.isel(time=slice(-context_steps, None))

        horizon_effective = min(horizon, targets_ds.sizes["time"])

        for k in range(horizon_effective):
            # Template for single-step prediction
            target_k_template = targets_ds.isel(time=slice(k, k + 1)) * 0.0

            # Forcings for this future step
            if "time" in forcings_ds.dims:
                forcings_k = forcings_ds.isel(time=slice(k, k + 1))
            else:
                forcings_k = forcings_ds

            # Generate prediction (for logging), but do NOT feed it back
            pred_k = model.full_sampling(
                inputs=current_context,
                targets_template=target_k_template,
                forcings=forcings_k,
            )
            preds_per_step.append(pred_k)

            # Build the next context using GROUND-TRUTH targets for this step
            gt_target_k = targets_ds.isel(time=slice(k, k + 1))
            tail = current_context.isel(time=slice(1, None))  # drop oldest
            next_frame_gt = _compose_next_frame(
                target_like=gt_target_k,
                forcings_like=forcings_k,
                prev_context=current_context,
            )
            current_context = xr.concat([tail, next_frame_gt], dim="time")

        preds_rollout = xr.concat(preds_per_step, dim="time")
        targets_future = targets_ds.isel(time=slice(0, horizon_effective))

    else:
        # ------------------------------------------------------------------
        # STANDARD FORECAST-TIME AUTOREGRESSIVE ROLLOUT (existing behavior)
        # ------------------------------------------------------------------
        # Initial context: last `context_steps` frames of inputs (e.g. X_{t-1}, X_t)
        current_context = inputs_ds.isel(time=slice(-context_steps, None))

        for k in range(horizon):
            # Single-step target template (time=1, same vars/coords as GT)
            target_k_template = targets_ds.isel(time=slice(k, k + 1)) * 0.0

            # Forcings for this step
            if "time" in forcings_ds.dims:
                forcings_k = forcings_ds.isel(time=slice(k, k + 1))
            else:
                forcings_k = forcings_ds

            # One-step forecast (predict target variables)
            pred_k = model.full_sampling(
                inputs=current_context,
                targets_template=target_k_template,
                forcings=forcings_k,
            )
            preds_per_step.append(pred_k)

            # Compose a full input frame for the new time using
            # predicted targets + provided forcings + carried input-only vars.
            tail = current_context.isel(time=slice(1, None))  # drop oldest
            next_frame = _compose_next_frame(
                target_like=pred_k,
                forcings_like=forcings_k,
                prev_context=current_context,
            )
            current_context = xr.concat([tail, next_frame], dim="time")

        preds_rollout = xr.concat(preds_per_step, dim="time")
        targets_future = targets_ds.isel(time=slice(0, horizon))

    # ----------------------------------------------------------------------
    # MSE over all variables, timesteps, lat/lon/level
    # ----------------------------------------------------------------------
    pred_da = preds_rollout.to_array()
    targ_da = targets_future.to_array()
    mse = float(
        jnp.mean(
            (xarray_jax.jax_data(pred_da) - xarray_jax.jax_data(targ_da)) ** 2
        )
    )

    return mse, preds_rollout, targets_future
