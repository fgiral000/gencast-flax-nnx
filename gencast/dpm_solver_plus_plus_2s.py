#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threads an explicit PRNGKey pulled from nnx.Rngs through lax.fori_loop,
executes purely on raw JAX arrays, and rebuilds final Dataset.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import xarray as xr
from typing import Optional, Tuple
from graphcast import casting
from gencast import samplers_utils as utils
from common import xarray_jax
import gencast.samplers_base as sampler_base



class Sampler(sampler_base.Sampler):
    def __init__(
        self,
        denoiser: nnx.Module,
        max_noise_level: float,
        min_noise_level: float,
        num_noise_levels: int,
        rho: float,
        stochastic_churn_rate: float,
        churn_min_noise_level: float,
        churn_max_noise_level: float,
        noise_level_inflation_factor: float,
    ):
        self._noise_levels = utils.noise_schedule(
            max_noise_level, min_noise_level, num_noise_levels, rho
        )
        self._stochastic_churn = stochastic_churn_rate > 0.0
        self._per_step_churn_rates = utils.stochastic_churn_rate_schedule(
            self._noise_levels,
            stochastic_churn_rate,
            churn_min_noise_level,
            churn_max_noise_level,
        )
        self._noise_level_inflation_factor = noise_level_inflation_factor
        self._denoiser = denoiser
        self.sigma_data = 1.0

    def __call__(
        self,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: Optional[xr.Dataset] = None,
        rngs: nnx.Rngs = None,
    ) -> xr.Dataset:
        if rngs is None:
            raise ValueError("Must pass rngs: nnx.Rngs(...) to Sampler")
        key = rngs.noise()  # pull once, outside any JAX trace

        # 1) pull out raw JAX array from the targets_template
        da_tmpl = targets_template.to_array()            # dims e.g. ('variable','batch','node')
        arr_tmpl = xarray_jax.unwrap(da_tmpl)           # pure JAX DeviceArray
        dims   = da_tmpl.dims
        coords = da_tmpl.coords

        # 2) schedules
        dtype  = casting.infer_floating_dtype(targets_template)
        sigmas = jnp.array(self._noise_levels, dtype=dtype)
        churns = jnp.array(self._per_step_churn_rates, dtype=dtype)

        # --- Precompute initial spherical noise on the sphere ---
        # Use the xarray-based spherical noise on the targets_template grid.
        spherical_noise_ds = utils.spherical_white_noise_like(
            targets_template, rngs=rngs
        )
        spherical_noise_da = spherical_noise_ds.to_array()
        spherical_noise_arr = xarray_jax.unwrap_data(spherical_noise_da)

        # Scale by the maximum noise level σ_0 to get the initial x_σ0
        init_x = spherical_noise_arr * sigmas[0]


        # 3) denoiser wrapper on raw arrays
        def denoise_arr(sigma: jnp.ndarray, x_arr: jnp.ndarray) -> jnp.ndarray:
            # Handle the edge case where sigma=0 which causes NaNs in the denoiser
            eps = 1e-6  # small epsilon to avoid sigma=0
            sigma_safe = jnp.maximum(sigma, eps)
            
            # rebuild a tiny Dataset for the denoiser
            da = xarray_jax.DataArray(x_arr, dims=dims, coords=coords)
            ds = da.to_dataset(dim="variable")

            for v, tmpl_var in targets_template.data_vars.items():
                # Unconditionally squeeze artificial broadcast 'level' for single-level template vars.
                # The earlier safety check caused TracerBoolConversion warnings inside the JAX loop.
                if v in ds.data_vars and ('level' not in tmpl_var.dims) and ('level' in ds[v].dims):
                    ds[v] = ds[v].isel(level=0).drop("level")

            # Build noise_levels as a (batch,) DataArray
            batch_coord = coords["batch"]
            batch_size  = batch_coord.size
            sigma_arr   = jnp.full((batch_size,), sigma_safe, dtype=sigma_safe.dtype)
            nl_da       = xarray_jax.DataArray(
                sigma_arr,
                dims=("batch",),
                coords={"batch": batch_coord},
            )
            
            out_ds = self._preconditioned_denoiser(
                inputs=inputs,
                noisy_targets=ds,
                noise_levels=nl_da,
                forcings=forcings
            )
            # extract back to raw array - ensure it's a pure JAX array with correct dtype
            out_arr = out_ds.to_array()
            result = xarray_jax.unwrap_data(out_arr)
            
            # Cast back to original dtype to maintain consistency
            return result.astype(x_arr.dtype)

        def body_fn(i: jnp.ndarray,
                    state: Tuple[jnp.ndarray, jnp.ndarray]
                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            x_arr, key = state

            # 2) Current noise level
            sigma = sigmas[i]

            # 3) (optional) stochastic churn – this part can stay,
            #    as long as utils.apply_stochastic_churn_arr is pure JAX-array.
            if self._stochastic_churn:
                x_arr, sigma, key = utils.apply_stochastic_churn_arr(
                    x_arr,
                    sigma,
                    churns[i],
                    self._noise_level_inflation_factor,
                    key
                )

            # 4) Denoiser steps (2S update) – unchanged
            sigma_next = sigmas[i + 1]
            sigma_mid  = jnp.sqrt(sigma * sigma_next)

            x_denoised = denoise_arr(sigma, x_arr)

            alpha_mid = sigma_mid / sigma
            x_mid = alpha_mid * x_arr + (1 - alpha_mid) * x_denoised

            x_mid_denoised = denoise_arr(sigma_mid, x_mid)

            alpha_next = sigma_next / sigma
            x_next = alpha_next * x_arr + (1 - alpha_next) * x_mid_denoised

            x_out = jnp.where(sigma_next == 0, x_denoised, x_next)
            return x_out, key

        # Start the loop from spherical noise at σ_0 instead of zeros
        init = (init_x, key)
        final_arr, _ = jax.lax.fori_loop(0, len(sigmas) - 1, body_fn, init)
        # 5) wrap back to Dataset using proper xarray_jax functions
        da_final = xarray_jax.DataArray(final_arr, dims=dims, coords=coords)
        result_ds = da_final.to_dataset(dim="variable")

        # Apply the same single-level restoration on the final output.
        for v, tmpl_var in targets_template.data_vars.items():
            if v in result_ds.data_vars and ('level' not in tmpl_var.dims) and ('level' in result_ds[v].dims):
                result_ds[v] = result_ds[v].isel(level=0).drop("level")

        # Report restored width (count levels only for multi-level vars).
        expected_width = 0
        for v, tmpl_var in targets_template.data_vars.items():
            expected_width += tmpl_var.sizes.get('level', 1)
        actual_width = 0
        for v, out_var in result_ds.data_vars.items():
            actual_width += out_var.sizes.get('level', 1)
        # print(f"[INFO] Restored runtime data width: {actual_width} (expected {expected_width})")
        
        return result_ds
    

    # preconditioning helpers
    def _c_in(self, sigma: xr.DataArray) -> xr.DataArray:
        return (sigma**2 + self.sigma_data**2) ** -0.5

    def _c_out(self, sigma: xr.DataArray) -> xr.DataArray:
        return (sigma*self.sigma_data) / ((sigma**2 + self.sigma_data**2) ** 0.5)

    def _c_skip(self, sigma: xr.DataArray) -> xr.DataArray:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def _preconditioned_denoiser(
        self,
        inputs: xr.Dataset,
        noisy_targets: xr.Dataset,
        noise_levels: xr.DataArray,
        forcings: Optional[xr.Dataset] = None,
        **kwargs) -> xr.Dataset:
        """The preconditioned denoising function D from the paper (Eqn 7)."""
        raw_predictions = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs)
        return (raw_predictions * self._c_out(noise_levels) +
                noisy_targets * self._c_skip(noise_levels))