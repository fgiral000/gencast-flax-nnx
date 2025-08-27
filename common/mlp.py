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
from typing import Optional


### Flax NNX modules ###
class LinearNormConditioning(nnx.Module):
  """Module for norm conditioning.

  Conditions the normalization of "inputs" by applying a linear layer to the
  "norm_conditioning" which produces the scale and variance which are applied to
  each channel (across the last dim) of "inputs".

  NOTE: This is a reimplementation of the Haiku module using Flax NNX.
  """

  def __init__(self, 
               feature_size: int,
               rngs: nnx.Rngs,
               mesh,
               conditioning_dim: Optional[int] = None):
    self.feature_size = feature_size
    # mesh is now required
    kernel_init = nnx.with_partitioning(
        nnx.initializers.truncated_normal(stddev=1e-8),
        P(None, 'model')
    )
    bias_init = nnx.with_partitioning(
        nnx.initializers.zeros_init(),
        P('model')
    )
    self.conditional_linear_layer = nnx.Linear(
      in_features=conditioning_dim if conditioning_dim is not None else 64,  # Default to 16 if not provided
      out_features=2 * feature_size,
      kernel_init=kernel_init,
      bias_init=bias_init,
      rngs=rngs,
    )
  
  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):
    conditional_scale_offset = self.conditional_linear_layer(norm_conditioning)
    scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
    scale = scale_minus_one + 1.
    return inputs * scale + offset
  


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


class MLPWithNormConditioning(nnx.Module):
  """An MLP with optional LayerNorm and optional external norm conditioning."""

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
               norm_conditioning_dim: Optional[int] = None,
               use_gradient_checkpointing: bool = False):
    """Initializes the MLP with optional norm conditioning."""
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning
    self._use_gradient_checkpointing = use_gradient_checkpointing

    self.network = MLP(
        mlp_input_size=mlp_input_size,
        mlp_hidden_size=mlp_hidden_size,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_output_size=mlp_output_size,
        activation=activation,
        rngs=rngs,
        mesh=mesh,
    )
    
    # Apply gradient checkpointing to the network if requested and it's large enough
    if use_gradient_checkpointing and mlp_num_hidden_layers >= 2:
        self.network = nnx.remat(self.network)

    if self._use_layer_norm:
      self.layer_norm = nnx.LayerNorm(
          num_features=mlp_output_size,
          use_scale=not use_norm_conditioning,   # <- FIXED
          use_bias=not use_norm_conditioning,    # <- FIXED
          feature_axes=-1,
          scale_init=nnx.initializers.ones_init() if not use_norm_conditioning else None,
          bias_init=nnx.initializers.zeros_init() if not use_norm_conditioning else None,
          rngs=rngs
      )

    if self._use_norm_conditioning:
      # if norm_conditioning_dim is None:
      #   raise ValueError("norm_conditioning_dim must be provided when norm conditioning is enabled.")
      self.norm_conditioning_layer = LinearNormConditioning(
          feature_size=mlp_output_size,
          conditioning_dim=norm_conditioning_dim if norm_conditioning_dim is not None else 16,  # Default to 16 if not provided
          rngs=rngs,
          mesh=mesh
      )

  def __call__(self, inputs: jax.Array, global_norm_conditioning: Optional[jax.Array] = None):
    if self._use_norm_conditioning and global_norm_conditioning is None:
      raise ValueError("global_norm_conditioning must be provided when norm conditioning is enabled.")
    if not self._use_norm_conditioning and global_norm_conditioning is not None:
      raise ValueError("global_norm_conditioning was provided, but norm conditioning is disabled.")

    x = self.network(inputs)

    if self._use_layer_norm:
      x = self.layer_norm(x)

      if self._use_norm_conditioning:
        # Ensure broadcast: global_norm_conditioning shape is (B, D), match x
        if inputs.shape[0] == global_norm_conditioning.shape[0]:
          global_norm_conditioning = global_norm_conditioning[:, None, :]  # (B, 1, D)
        else:
          global_norm_conditioning = global_norm_conditioning[None, ...]   # (1, B, D)
        x = self.norm_conditioning_layer(x, global_norm_conditioning)

    return x



    
    




if __name__ == "__main__":
  # Example usage
  rngs = nnx.Rngs(0)

  mlp = MLPWithNormConditioning(
      mlp_input_size=10,
      mlp_hidden_size=20,
      mlp_num_hidden_layers=3,
      mlp_output_size=5,
      activation=nnx.relu,
      use_layer_norm=True,
      use_norm_conditioning=True,
      rngs=rngs
  )
  # Visualize the model architecture
  # nnx.display(mlp)
  # This will print the structure of the MLP.

  # Let's use some dummy data to test the MLP
  dummy_input = jnp.ones((1,10))  # Batch size of 1, input size of 10
  global_norm_conditioning = jnp.ones((5,))  # Example global norm conditioning
  


  output = mlp(dummy_input, global_norm_conditioning=global_norm_conditioning)
  print("Output shape:", output.shape)  # Should be (1, 5) for the output size of 5