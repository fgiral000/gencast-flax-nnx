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

        # 3) denoiser wrapper on raw arrays
        def denoise_arr(sigma: jnp.ndarray, x_arr: jnp.ndarray) -> jnp.ndarray:
            # Handle the edge case where sigma=0 which causes NaNs in the denoiser
            eps = 1e-6  # small epsilon to avoid sigma=0
            sigma_safe = jnp.maximum(sigma, eps)
            
            # rebuild a tiny Dataset for the denoiser
            da = xarray_jax.DataArray(x_arr, dims=dims, coords=coords)
            ds = da.to_dataset(dim="variable")
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
            """
            One iteration of the 2S sampler, mirroring the DeepMind implementation
            but on raw JAX arrays plus RNG key.
            """
            x_arr, key = state
            # ——————————————————————————————————————————————————————————————
            # 1) Initial noise injection at i == 0
            # ——————————————————————————————————————————————————————————————
            # sample i.i.d. normal noise
            init_noise, key = utils.spherical_white_noise_like(x_arr, key)
            # only apply on the very first step
            is_first = (i == 0).astype(dtype)
            x_arr = x_arr + init_noise * sigmas[0] * is_first

            # ——————————————————————————————————————————————————————————————
            # 2) Current noise level
            # ——————————————————————————————————————————————————————————————
            sigma = sigmas[i]

            # ——————————————————————————————————————————————————————————————
            # 3) Stochastic churn (if enabled)
            # ——————————————————————————————————————————————————————————————
            if self._stochastic_churn:
                x_arr, sigma, key = utils.apply_stochastic_churn_arr(
                    x_arr,
                    sigma,
                    churns[i],
                    self._noise_level_inflation_factor,
                    key
                )

            # ——————————————————————————————————————————————————————————————
            # 4) ODE‐solver (2S) update
            # ——————————————————————————————————————————————————————————————
            # compute next and midpoint noise levels
            sigma_next = sigmas[i + 1]
            sigma_mid  = jnp.sqrt(sigma * sigma_next)

            # first denoise at σᵢ
            x_denoised = denoise_arr(sigma, x_arr)

            # midpoint update: x_mid = (σ_mid/σ) * x + (1 − σ_mid/σ) * x_denoised
            alpha_mid = sigma_mid / sigma
            x_mid = alpha_mid * x_arr + (1 - alpha_mid) * x_denoised

            # second denoise at σ_mid
            x_mid_denoised = denoise_arr(sigma_mid, x_mid)

            # full step update: x_next = (σₙₑₓₜ/σ) * x + (1 − σₙₑₓₜ/σ) * x_mid_denoised
            alpha_next = sigma_next / sigma
            x_next = alpha_next * x_arr + (1 - alpha_next) * x_mid_denoised

            # ——————————————————————————————————————————————————————————————
            # 5) Final‐step correction (avoid a second denoiser call at σ=0)
            # ——————————————————————————————————————————————————————————————
            x_out = jnp.where(sigma_next == 0, x_denoised, x_next)

            return x_out, key


        init = (jnp.zeros_like(arr_tmpl), key)
        final_arr, _ = jax.lax.fori_loop(0, len(sigmas) - 1, body_fn, init)

        # 5) wrap back to Dataset using proper xarray_jax functions
        da_final = xarray_jax.DataArray(final_arr, dims=dims, coords=coords)
        result_ds = da_final.to_dataset(dim="variable")
        
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