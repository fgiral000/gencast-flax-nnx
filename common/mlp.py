# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constructors for MLPs.

To enable multi-GPU sharding, create a mesh at the top level and pass it to all modules:

import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Example mesh creation for 4-way data and 2-way model parallelism:
mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))

# Pass mesh=mesh to all model/module constructors.
"""
import jax
import jax.numpy as jnp
import jraph
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec as P

import functools
from typing import Optional, Sequence, Callable
from common.model_utils import fourier_features



### Flax NNX modules ###
class LinearNormConditioning(nnx.Module):
  def __init__(self, feature_size: int, rngs: nnx.Rngs, mesh, conditioning_dim: int = 16):
    self.feature_size = feature_size
    kernel_init = nnx.with_partitioning(
        nnx.initializers.truncated_normal(stddev=1e-8),
        P(None, 'model')
    )
    bias_init = nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        P('model')
    )
    self.conditional_linear_layer = nnx.Linear(
      in_features=conditioning_dim,
      out_features=2 * feature_size,
      kernel_init=kernel_init,
      bias_init=bias_init,
      rngs=rngs,
    )
  
  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):
    # inputs: (..., C)
    # norm_conditioning: (..., D) broadcastable to inputs[..., None, :]
    cond = self.conditional_linear_layer(norm_conditioning)  # (..., 2*C)
    scale_minus_one, offset = jnp.split(cond, 2, axis=-1)
    scale = scale_minus_one + 1.
    return inputs * scale + offset
  

class MLPWithNormConditioning(nnx.Module):
  def __init__(self,
               mlp_input_size: int,
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               mlp_output_size: int,
               activation,
               *,
               use_layer_norm: bool,
               use_norm_conditioning: bool,
               rngs: nnx.Rngs,
               mesh,
               norm_conditioning_dim: Optional[int] = 16):
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning

    self.network = MLP(
        mlp_input_size=mlp_input_size,
        mlp_hidden_size=mlp_hidden_size,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_output_size=mlp_output_size,
        activation=activation,
        rngs=rngs,
        mesh=mesh,
    )

    if self._use_layer_norm:
      self.layer_norm = nnx.LayerNorm(
          num_features=mlp_output_size,
          use_scale=not use_norm_conditioning,
          use_bias=not use_norm_conditioning,
          feature_axes=-1,
          scale_init=nnx.initializers.ones_init() if not use_norm_conditioning else None,
          bias_init=nnx.initializers.zeros_init() if not use_norm_conditioning else None,
          rngs=rngs
      )

    if self._use_norm_conditioning:
      if norm_conditioning_dim is None:
        raise ValueError("norm_conditioning_dim must be set when norm conditioning is enabled.")
      self.norm_conditioning_layer = LinearNormConditioning(
          feature_size=mlp_output_size,
          conditioning_dim=norm_conditioning_dim,
          rngs=rngs,
          mesh=mesh
      )

  def __call__(self, inputs: jax.Array, global_norm_conditioning: Optional[jax.Array] = None):
    if self._use_norm_conditioning and global_norm_conditioning is None:
      raise ValueError("global_norm_conditioning must be provided when norm conditioning is enabled.")

    x = self.network(inputs)

    if self._use_layer_norm:
      x = self.layer_norm(x)

    if self._use_norm_conditioning:
      # Expect global_norm_conditioning of shape (B, D)
      # Match to inputs of shape (N, B, C) or (B, N, C)
      if x.ndim == 3:
        # Case (N,B,C)
        if x.shape[1] == global_norm_conditioning.shape[0]:
          cond = global_norm_conditioning[None, :, :]  # (1,B,D)
        # Case (B,N,C)
        elif x.shape[0] == global_norm_conditioning.shape[0]:
          cond = global_norm_conditioning[:, None, :]  # (B,1,D)
        else:
          raise ValueError(f"Cannot align conditioning {global_norm_conditioning.shape} with x {x.shape}")
      elif x.ndim == 2:
        # Case (B,C)
        if x.shape[0] == global_norm_conditioning.shape[0]:
          cond = global_norm_conditioning
        else:
          raise ValueError(f"Cannot align conditioning {global_norm_conditioning.shape} with x {x.shape}")
      else:
        raise ValueError(f"Unsupported input shape {x.shape} for norm conditioning")

      x = self.norm_conditioning_layer(x, cond)

    return x

  


class MLP(nnx.Module): 
  """A simple MLP module."""
  def __init__(self,
               mlp_input_size: int,
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               mlp_output_size: int,
               activation,
               *,
               rngs: nnx.Rngs,
               mesh):
    """Initializes the MLP module."""
    layers = []
    feature_size = mlp_input_size
    for _ in range(mlp_num_hidden_layers):
      kernel_init = nnx.with_partitioning(
          nnx.initializers.xavier_uniform(),
          P(None, 'model')
      )
      bias_init = nnx.with_partitioning(
          nnx.initializers.zeros_init(),
          P('model')
      )
      layers.append(nnx.Linear(
          in_features=feature_size, 
          out_features=mlp_hidden_size, 
          kernel_init=kernel_init,
          bias_init=bias_init,
          rngs=rngs
      ))
      feature_size = mlp_hidden_size
      layers.append(activation)
    # Final layer
    kernel_init = nnx.with_partitioning(
        nnx.initializers.xavier_uniform(),
        P(None, 'model')
    )
    bias_init = nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        P('model')
    )
    layers.append(nnx.Linear(
        in_features=mlp_hidden_size, 
        out_features=mlp_output_size, 
        kernel_init=kernel_init,
        bias_init=bias_init,
        rngs=rngs
    ))
    self.network = nnx.Sequential(*layers)
  
  def __call__(self, inputs: jax.Array):
    return self.network(inputs)  
  


class FourierFeaturesMLP(nnx.Module):
    def __init__(
        self,
        base_period: float,
        num_frequencies: int,
        output_sizes: Sequence[int],
        apply_log_first: bool = False,
        w_init: Optional[nnx.Initializer] = None,
        activation: Callable = jax.nn.gelu,
        rngs: nnx.Rngs = nnx.Rngs(0),
        mesh=None,
        **mlp_kwargs,
    ):
        self.base_period = base_period
        self.num_frequencies = num_frequencies
        self.apply_log_first = apply_log_first
        self.activation = activation

        if w_init is None:
            w_init = nnx.initializers.variance_scaling(
                2.0, mode="fan_in", distribution="uniform"
            )

        in_ch = 2 * num_frequencies
        self.linears = []
        for out_ch in output_sizes:
            if mesh is not None:
                from jax.sharding import PartitionSpec as P
                kernel_init = nnx.with_partitioning(w_init, P(None, "model"))
                bias_init = nnx.with_partitioning(
                    nnx.initializers.zeros_init(), P("model")
                )
            else:
                kernel_init = w_init
                bias_init = nnx.initializers.zeros_init()

            lin = nnx.Linear(
                in_features=in_ch,
                out_features=out_ch,
                kernel_init=kernel_init,
                bias_init=bias_init,
                rngs=rngs,
                **mlp_kwargs,
            )
            setattr(self, f"linear_{len(self.linears)}", lin)
            self.linears.append(lin)
            in_ch = out_ch

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.apply_log_first:
            x = jnp.log(x)
        feats = fourier_features(
            x, self.base_period, self.num_frequencies
        )
        for i, lin in enumerate(self.linears):
            feats = lin(feats)
            if i < len(self.linears) - 1:
                feats = self.activation(feats)
        return feats