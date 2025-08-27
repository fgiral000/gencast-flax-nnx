#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPM-Solver++ 2S sampler for GenSynth CFD diffusion.

Threads an explicit PRNGKey pulled from nnx.Rngs through lax.fori_loop,
executes purely on raw JAX arrays. Now works with JAX arrays instead of xarray.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict
from graphcast import casting
from gencast import samplers_utils as utils


class Sampler:
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
        self._noise_levels = utils.edm_noise_schedule(
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
        noisy_inputs: jnp.ndarray,
        forcings: Dict[str, jnp.ndarray],
        rngs: nnx.Rngs = None,
    ) -> jnp.ndarray:
        """
        Args:
            noisy_inputs: Shape (batch, node, channels)
            forcings: Dict with keys like 'case_number', 'U_mag_prev'
            rngs: Random number generator state
        
        Returns:
            Denoised output with same shape as noisy_inputs
        """
        if rngs is None:
            raise ValueError("Must pass rngs: nnx.Rngs(...) to Sampler")
        key = rngs.noise()  # pull once, outside any JAX trace

        # Get array template and dtype
        arr_tmpl = noisy_inputs
        dtype = arr_tmpl.dtype

        # 2) schedules
        sigmas = jnp.array(self._noise_levels, dtype=dtype)
        churns = jnp.array(self._per_step_churn_rates, dtype=dtype)

        # 3) denoiser wrapper on raw arrays
        def denoise_arr(sigma: jnp.ndarray, x_arr: jnp.ndarray) -> jnp.ndarray:
            # Handle the edge case where sigma=0 which causes NaNs in the denoiser
            eps = 1e-6  # small epsilon to avoid sigma=0
            sigma_safe = jnp.maximum(sigma, eps)
            
            # Build noise_levels as a (batch,) array
            batch_size = x_arr.shape[0]
            sigma_arr = jnp.full((batch_size,), sigma_safe, dtype=sigma_safe.dtype)
            
            return self._preconditioned_denoiser(
                noisy_inputs=x_arr,
                noise_levels=sigma_arr,
                forcings=forcings
            )
            
            # # Cast back to original dtype to maintain consistency
            # return result.astype(x_arr.dtype)

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
                x_arr, sigma, key = utils.apply_stochastic_churn(
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

        # Return the final JAX array directly
        return final_arr
    

    # preconditioning helpers
    def _c_in(self, sigma: jnp.ndarray) -> jnp.ndarray:
        return (sigma**2 + self.sigma_data**2) ** -0.5

    def _c_out(self, sigma: jnp.ndarray) -> jnp.ndarray:
        return (sigma*self.sigma_data) / ((sigma**2 + self.sigma_data**2) ** 0.5)

    def _c_skip(self, sigma: jnp.ndarray) -> jnp.ndarray:
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def _preconditioned_denoiser(
        self,
        noisy_inputs: jnp.ndarray,
        noise_levels: jnp.ndarray,
        forcings: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply preconditioning to the denoiser input and output."""
        c_in = self._c_in(noise_levels)
        # Expand dimensions for broadcasting (batch,) -> (batch, 1, ..., 1)
        c_in_expanded = c_in.reshape((-1,) + (1,) * (noisy_inputs.ndim - 1))
        
        y = noisy_inputs * c_in_expanded
        raw = self._denoiser(noisy_inputs=y, noise_levels=noise_levels, forcings=forcings)
        
        c_out = self._c_out(noise_levels)
        c_skip = self._c_skip(noise_levels)
        c_out_expanded = c_out.reshape((-1,) + (1,) * (noisy_inputs.ndim - 1))
        c_skip_expanded = c_skip.reshape((-1,) + (1,) * (noisy_inputs.ndim - 1))
        
        return raw * c_out_expanded + noisy_inputs * c_skip_expanded
