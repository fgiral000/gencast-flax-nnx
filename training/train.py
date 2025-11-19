#!/usr/bin/env python3
"""
training script for GenCast using Grain-based ERA5 datasets.

The heavy lifting (model creation, jitted steps, optimizer, full_testing, etc.)
is implemented in `train_helpers.py`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless cluster
import wandb


from gencast import gencast

# Local helpers
from training.era5_dataset import Era5SampleSource
from training.train_helpers import (
    create_gencast_model,
    maybe_wrap_with_nan_cleaner,
    maybe_wrap_with_normalization,
    make_jitted_steps,
    create_optimizer,
    to_jax_dataset,
    _pack_ds_to_arrays,  
    init_wandb,
    save_checkpoint,
    autoregressive_rollout,
)

from training.plotting_helpers import plot_2m_temperature_triptych

def make_simple_iterator(
    source: Era5SampleSource,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Simple Python iterator over Era5SampleSource with xarray-based batching.

    - source[i] returns a dict: {"inputs": ds, "targets": ds, "forcings": ds}
      where each ds has a 'batch' dim of size 1.
    - This iterator yields the same dict structure, but with 'batch' size
      equal to batch_size (for train) or 1 (for test).
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(source))

    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idxs = indices[start : start + batch_size]
            samples = [source[int(i)] for i in idxs]

            if len(samples) == 1:
                yield samples[0]
            else:
                inputs = xr.concat(
                    [s["inputs"] for s in samples], dim="batch"
                )
                targets = xr.concat(
                    [s["targets"] for s in samples], dim="batch"
                )
                forcings = xr.concat(
                    [s["forcings"] for s in samples], dim="batch"
                )
                yield {
                    "inputs": inputs,
                    "targets": targets,
                    "forcings": forcings,
                }



def _arrays_from_batch(
    batch: Dict[str, xr.Dataset]
) -> tuple[Dict[str, jax.Array], Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Convert a batch of xarray.Datasets into dicts of jax.Array.

    batch structure:
      {
        "inputs":  Dataset,
        "targets": Dataset,
        "forcings": Dataset,
      }
    """
    inputs_arrs = _pack_ds_to_arrays(batch["inputs"])
    targets_arrs = _pack_ds_to_arrays(batch["targets"])
    forcings_arrs = _pack_ds_to_arrays(batch["forcings"])
    return inputs_arrs, targets_arrs, forcings_arrs


def main():
    parser = argparse.ArgumentParser(description="Train GenCast diffusion model on ERA5 using Grain datasets.")

    # --- Data source options -------------------------------------------------
    parser.add_argument("--data_source", type=str, default="local_era5", choices=["gcs_toy", "local_single", "local_era5"], help=("Which data source to use: 'gcs_toy' for original toy dataset on GCS, 'local_single' for a single local NetCDF file, 'local_era5' for a directory of monthly ERA5 files."))
    parser.add_argument("--dataset_file", type=str, default="source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc", help=("Dataset file name. For gcs_toy: object name under prefix/dataset/. For local_single: path to local NetCDF file. Ignored for local_era5."))
    parser.add_argument("--era5_root_dir", type=str, default="data_era5/", help="Root directory containing era5_pressure_levels_*.nc and era5_single_levels_*.nc (for local_era5).")
    parser.add_argument("--gcs_bucket", type=str, default="dm_graphcast", help="GCS bucket for public datasets (for gcs_toy).")
    parser.add_argument("--gcs_dir_prefix", type=str, default="gencast/", help="GCS directory prefix (ends with '/') for gcs_toy.")

    # --- Training hyperparameters -------------------------------------------
    parser.add_argument("--num_steps", type=int, default=5000, help="Number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Base learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size over constructed samples.")
    parser.add_argument("--log_every", type=int, default=100, help="Per-step logging frequency (0 to disable; epoch summaries always shown).")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint save frequency in steps (0 to disable).")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--model_name", type=str, default="gencast_model", help="Name of the model for checkpoints.")

    # --- Model configuration -------------------------------------------------
    parser.add_argument("--mesh_size", type=int, default=4, help="Icosahedral mesh refinement level.")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer model width.")
    parser.add_argument("--num_layers", type=int, default=16, help="Transformer blocks depth.")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # --- Evaluation / testing ------------------------------------------------
    parser.add_argument("--eval_every", type=int, default=100, help="Eval loss frequency in steps (0 to disable). Uses train data distribution.")
    parser.add_argument("--do_sampling_eval", action="store_true", help="Run full sampling eval on a test batch at eval steps.")

    # --- Preprocessing wrappers ---------------------------------------------
    parser.add_argument("--clean_sst_nans", action="store_true", help="Fill NaNs in sea_surface_temperature before model/loss.")
    parser.add_argument("--apply_normalization", action="store_true", help="Apply InputsAndResiduals normalization wrapper (uses residual-diffs stats).")
    parser.add_argument("--normalization_stats_dir", type=str, default="./normalization_stats", help="Directory containing normalization stats .nc files.")

    # --- WandB ---------------------------------------------------------------
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging (or set env USE_WANDB=1).")

    args = parser.parse_args()

    # Resolve data source logic if user used old flags
    if args.data_source == "local_era5" and not args.era5_root_dir:
        raise ValueError(
            "data_source='local_era5' requires --era5_root_dir pointing to ERA5 monthly NetCDF files."
        )

    print("JAX devices:", jax.devices())

    # ----------------------------- Model ------------------------------------
    rngs = nnx.Rngs(args.seed)
    model, den_arch_cfg, sampler_cfg, noise_cfg, noise_enc_cfg = create_gencast_model(
        task_config=gencast.TASK,
        rngs=rngs,
        mesh_size=args.mesh_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        stochastic_churn_rate=0.0,
    )

    model = maybe_wrap_with_nan_cleaner(model, clean_sst_nans=args.clean_sst_nans)
    model = maybe_wrap_with_normalization(
        model,
        apply_normalization=args.apply_normalization,
        normalization_stats_dir=args.normalization_stats_dir,
    )

    # ----------------------------- Datasets ---------------------------------
    source_kind = args.data_source  # directly map to Era5SampleSource.SourceKind

    # ----------------------------- Datasets ---------------------------------
    # Right now Era5SampleSource only supports local ERA5 monthly files.
    # We ignore data_source/dataset_file/gcs_* here and just use era5_root_dir.
    train_source = Era5SampleSource(
        era5_root_dir=args.era5_root_dir,
        task_config=gencast.TASK,
        mode="train",
        resolution_deg=2.50,   # or expose as a flag if you like
    )

    test_source = Era5SampleSource(
        era5_root_dir=args.era5_root_dir,
        task_config=gencast.TASK,
        mode="test",
        resolution_deg=2.50,
    )

    print(f"[Dataset] train_source: {train_source}")
    print(f"[Dataset] test_source : {test_source}")

    train_iter = make_simple_iterator(
        train_source,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    test_iter = make_simple_iterator(
        test_source,
        batch_size=1,      # one month/sample per eval
        shuffle=False,
        seed=args.seed + 1,
    )


    # Get one sample from the training source to build templates
    first_sample = train_source[0]
    train_tpl_inp = to_jax_dataset(first_sample["inputs"])
    train_tpl_tgt = to_jax_dataset(first_sample["targets"])
    train_tpl_frc = to_jax_dataset(first_sample["forcings"])


    print("Preparing model with a warm-up forward pass…")
    loss_da, _ = model.loss(
        inputs=train_tpl_inp,
        targets=train_tpl_tgt,
        forcings=train_tpl_frc,
    )
    _ = jax.device_get(loss_da)
    print("Model init completed via loss().")



    # ------------------------- Optimizer & steps ----------------------------
    optimizer, schedule = create_optimizer(
        model, num_steps=args.num_steps, learning_rate=args.learning_rate
    )
    train_step_packed, eval_step_packed = make_jitted_steps(
        template_inputs=train_tpl_inp,
        template_targets=train_tpl_tgt,
        template_forcings=train_tpl_frc,
    )

    # ------------------------- WandB setup ----------------------------------
    use_wandb = init_wandb(
        args.use_wandb,
        config=dict(
            num_steps=args.num_steps,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            mesh_size=args.mesh_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ),
    )

    # ------------------------- Checkpoint dir -------------------------------
    checkpoint_root = Path(args.checkpoint_dir).resolve() / args.model_name
    checkpoint_root.mkdir(exist_ok=True, parents=True)

    # ------------------------- Training loop --------------------------------
    print(f"Starting training for {args.num_steps} steps…")
    print(f"Learning rate: {args.learning_rate}; batch_size: {args.batch_size}")
    print(f"Logging every {args.log_every} steps")

    start_time = time.time()
    step = 0
    epoch = 0

    # We'll track epoch stats roughly as (#samples / batch_size) steps per epoch.
    steps_per_epoch = max(1, len(train_source) // max(1, args.batch_size))
    epoch_loss_sum = 0.0
    epoch_diag_sums: Dict[str, float] = {}
    epoch_step_count = 0

    def finalize_and_log_epoch(epoch_idx: int):
        nonlocal epoch_loss_sum, epoch_diag_sums, epoch_step_count
        if epoch_step_count == 0:
            return
        avg_loss = epoch_loss_sum / epoch_step_count
        print(
            f"Epoch {epoch_idx:4d} finished: "
            f"avg_loss={avg_loss:.6f} over {epoch_step_count} steps"
        )
        if use_wandb:
            try:
                log_dict = {
                    "train/epoch": epoch_idx,
                    "train/epoch_loss": float(avg_loss),
                }
                for k, v in epoch_diag_sums.items():
                    log_dict[f"train/epoch_{k}"] = float(v / epoch_step_count)
                wandb.log(log_dict)
            except Exception:
                pass
        epoch_loss_sum = 0.0
        epoch_diag_sums = {}
        epoch_step_count = 0

    # Main loop
    while step < args.num_steps:
        # Recycle over train_iter indefinitely
        batch = next(train_iter)
        inputs_arrs, targets_arrs, forcings_arrs = _arrays_from_batch(batch)

        loss, diagnostics = train_step_packed(
            model,
            optimizer,
            inputs_arrs,
            targets_arrs,
            forcings_arrs if forcings_arrs else None,
        )

        epoch_loss_sum += float(loss)
        for k, v in diagnostics.items():
            try:
                epoch_diag_sums[k] = epoch_diag_sums.get(k, 0.0) + float(v)
            except Exception:
                pass
        epoch_step_count += 1

        # Logging
        if args.log_every and (step % args.log_every == 0):
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
            lr_now = float(schedule(step)) if callable(schedule) else args.learning_rate
            print(
                f"Step {step:6d}: loss={float(loss):.6f}, "
                f"lr={lr_now:.6f}, steps/sec={steps_per_sec:.2f}"
            )
            for k, v in diagnostics.items():
                try:
                    print(f"  {k}: {float(v):.6f}")
                except Exception:
                    print(f"  {k}: {v}")
            if use_wandb:
                try:
                    wandb.log(
                        {"train/loss": float(loss), "lr": lr_now, "step": step}
                    )
                except Exception:
                    pass

        # "Epoch" bookkeeping
        if (step + 1) % steps_per_epoch == 0:
            finalize_and_log_epoch(epoch)
            epoch += 1

        # Checkpointing
        if args.save_every and step > 0 and (step % args.save_every == 0):
            save_checkpoint(step, model, checkpoint_root)

        # Simple eval (loss only, on train distribution)
        if args.eval_every and step > 0 and (step % args.eval_every == 0):
            eval_batch = next(train_iter)
            eval_inp_arrs, eval_tgt_arrs, eval_frc_arrs = _arrays_from_batch(eval_batch)
            eval_loss, _ = eval_step_packed(
                model,
                eval_inp_arrs,
                eval_tgt_arrs,
                eval_frc_arrs if eval_frc_arrs else None,
            )
            print(f"  Eval @ step {step}: loss={float(eval_loss):.6f}")
            if use_wandb:
                try:
                    wandb.log({"eval/loss": float(eval_loss), "step": step})
                except Exception:
                    pass

            # Full sampling-based eval on separate test dataset
            if args.do_sampling_eval:
                test_batch = next(test_iter)
                mse, preds, targets_future = autoregressive_rollout(
                    model,
                    batch=test_batch,
                    context_steps=2,
                    max_rollout_steps=1,   # single-step eval to keep it cheap
                )

                print(f"  Sampling eval MSE (future horizon): {mse:.6f}")
                if use_wandb:
                    try:
                        # Log scalar MSE first
                        wandb.log(
                            {
                                "eval/sample_mse_future": mse,
                                "step": step,
                            }
                        )

                        fig = plot_2m_temperature_triptych(
                            preds, targets_future, domain="Europe"
                        )
                        if fig is not None:
                            wandb.log(
                                {"eval/2m_temperature_map": wandb.Image(fig)},
                                step=step,
                            )
                            plt.close(fig)  # important on headless systems
                        else:
                            print("[wandb] Triptych figure is None, nothing to log.")
                    except Exception as e:
                        print(f"[wandb image logging error] {e}", flush=True)



        step += 1

    finalize_and_log_epoch(epoch)

    if args.save_every:
        save_checkpoint(step, model, checkpoint_root)

    print(f"Training completed! Total time: {time.time() - start_time:.2f} seconds")
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    print("Training run completed.")


if __name__ == "__main__":
    main()
