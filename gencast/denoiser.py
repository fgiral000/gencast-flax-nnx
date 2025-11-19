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
"""Support for wrapping a general Predictor to act as a Denoiser."""

import dataclasses
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import chex
from common import deep_typed_graph_net
from gencast import denoisers_base as base
from common import grid_mesh_connectivity
from common import icosahedral_mesh
from common import model_utils
from gencast import sparse_transformer
from gencast import transformer
from common import typed_graph
from common import xarray_jax
from common import mlp as mlp_builder
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import xarray

import flax.nnx as nnx

Kwargs = Mapping[str, Any]
NoiseLevelEncoder = Callable[[jnp.ndarray], jnp.ndarray]



  



@chex.dataclass(frozen=True, eq=True)
class NoiseEncoderConfig:
  """Configures the noise level encoding.

  Properties:
    apply_log_first: Whether to take the log of the inputs before computing
      Fourier features.
    base_period: The base period to use. This should be greater or equal to the
      range of the values, or to the period if the values have periodic
      semantics (e.g. 2pi if they represent angles). Frequencies used will be
      integer multiples of 1/base_period.
    num_frequencies: The number of frequencies to use, we will use integer
      multiples of 1/base_period from 1 up to num_frequencies inclusive. (We
      don't include a zero frequency as this would just give constant features
      which are redundant if a bias term is present).
    output_sizes: Layer sizes for the MLP.
  """
  apply_log_first: bool = True
  base_period: float = 16.0
  num_frequencies: int = 32
  # 2-layer MLP applied to Fourier features
  output_sizes: tuple[int, int] = (32, 16)


@chex.dataclass(eq=True)
class SparseTransformerConfig:
  """Sparse Transformer config."""
  # Neighbours to attend to.
  attention_k_hop: int
  # Primary width, the number of channels on the carrier path.
  d_model: int
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  num_layers: int = 16
  # Number of heads for self-attention.
  num_heads: int = 4
  # Attention type.
  attention_type: str = "triblockdiag_mha"
  # mask type if splash attention being used.
  mask_type: str = "lazy"
  block_q: int = 1024
  block_kv: int = 512
  block_kv_compute: int = 256
  block_q_dkv: int = 512
  block_kv_dkv: int = 1024
  block_kv_dkv_compute: int = 1024
  # Init scale for final ffw layer (divided by depth)
  ffw_winit_final_mult: float = 0.0
  # Init scale for mha w (divided by depth).
  attn_winit_final_mult: float = 0.0
  # Number of hidden units in the MLP blocks. Defaults to 4 * d_model.
  ffw_hidden: int = 2048


@chex.dataclass(eq=True)
class DenoiserArchitectureConfig:
  """Defines the GenCast architecture.

  Properties:
    sparse_transformer_config: Config for the mesh transformer.
    mesh_size: How many refinements to do on the multi-mesh.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
      length of the longest edge of the finest mesh to define the radius of
      connectivity to use in the Grid2Mesh graph. Reasonable values are
      between 0.6 and 1. 0.6 reduces the number of grid points feeding into
      multiple mesh nodes and therefore reduces edge count and memory use, but
      1 gives better predictions.
    norm_conditioning_features: List of feature names which will be used to
      condition the GNN via norm_conditioning, rather than as regular
      features. If this is provided, the GNN has to support the
      `global_norm_conditioning` argument. For now it only supports global
      norm conditioning (e.g. the same vector conditions all edges and nodes
      normalization), which means features passed here must not have "lat" or
      "lon" axes. In the future it may support node level norm conditioning
      too.
    grid2mesh_aggregate_normalization: Optional constant to normalize the output
      of aggregate_edges_for_nodes_fn in the mesh2grid GNN. This can be used to
        reduce the shock the model undergoes when switching resolution, which
        increases the number of edges connected to a node.
    node_output_size: Size of the output node representations for
        each node type. For node types not specified here, the latent node
        representation from the output of the processor will be returned.
  """

  sparse_transformer_config: SparseTransformerConfig
  mesh_size: int
  latent_size: int = 512
  hidden_layers: int = 1
  radius_query_fraction_edge_length: float = 0.6
  norm_conditioning_features: tuple[str, ...] = ("noise_level_encodings",)
  grid2mesh_aggregate_normalization: Optional[float] = None
  node_output_size: Optional[int] = None


class Denoiser(nnx.Module):
  """Wraps a general deterministic Predictor to act as a Denoiser.

  This passes an encoding of the noise level as an additional input to the
  Predictor as an additional input 'noise_level_encodings' with shape
  ('batch', 'noise_level_encoding_channels'). It passes the noisy_targets as
  additional forcings (since they are also per-target-timestep data that the
  predictor needs to condition on) with the same names as the original target
  variables.
  """

  def __init__(
      self,
      noise_encoder_config: Optional[NoiseEncoderConfig],
      denoiser_architecture_config: DenoiserArchitectureConfig,
      rngs: nnx.Rngs,
      gpu_mesh,
  ):
    self.rngs = rngs
    self.predictor = DenoiserArchitecture(
        denoiser_architecture_config=denoiser_architecture_config,
        rngs=self.rngs,
        gpu_mesh=gpu_mesh
    )
    # Use default values if not specified.
    if noise_encoder_config is None:
      noise_encoder_config = NoiseEncoderConfig()
    self.noise_level_encoder = mlp_builder.FourierFeaturesMLP(**noise_encoder_config,
                                                   rngs=self.rngs)

  def __call__(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    

    if forcings is None: forcings = xarray.Dataset()
    
    ################################################
    forcings = forcings.assign(noisy_targets)
    ################################################

    
    if noise_levels.dims != ("batch",):
      raise ValueError("noise_levels expected to be shape (batch,).")
    noise_level_encodings = self.noise_level_encoder(
        xarray_jax.unwrap_data(noise_levels)
    )
    noise_level_encodings = xarray_jax.Variable(
        ("batch", "noise_level_encoding_channels"), noise_level_encodings
    )
    inputs = inputs.assign(noise_level_encodings=noise_level_encodings)

    return self.predictor(
        inputs=inputs,
        targets_template=noisy_targets,
        forcings=forcings,
        **kwargs)


class DenoiserArchitecture(nnx.Module):
  """GenCast Predictor.

  The model works on graphs that take into account:
  * Mesh nodes: nodes for the vertices of the mesh.
  * Grid nodes: nodes for the points of the grid.
  * Nodes: When referring to just "nodes", this means the joint set of
    both mesh nodes, concatenated with grid nodes.

  The model works with 3 graphs:
  * Grid2Mesh graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from grid nodes to mesh nodes using a
    fixed radius query. The grid2mesh_gnn will operate in this graph. The output
    of this stage will be a latent representation for the mesh nodes, and a
    latent representation for the grid nodes.
  * Mesh graph: Graph that contains mesh nodes only. The mesh_gnn will
    operate in this graph. It will update the latent state of the mesh nodes
    only.
  * Mesh2Grid graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from mesh nodes to grid nodes such that each grid
    node is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.
  """

  def __init__(
      self,
      denoiser_architecture_config: DenoiserArchitectureConfig,
      rngs: nnx.Rngs,
      gpu_mesh,
  ):
    """Initializes the predictor."""
    self._spatial_features_kwargs = dict(
        add_node_positions=False,
        add_node_latitude=True,
        add_node_longitude=True,
        add_relative_positions=True,
        relative_longitude_local_coordinates=True,
        relative_latitude_local_coordinates=True,
    )
    self.rngs = rngs
    # Construct the mesh.
    mesh = icosahedral_mesh.get_last_triangular_mesh_for_sphere(
        splits=denoiser_architecture_config.mesh_size
    )
    # Permute the mesh to a banded structure so we can run sparse attention
    # operations.
    self._mesh = _permute_mesh_to_banded(mesh=mesh)

    # Defer layer initialization until we have graph templates (built in _maybe_init).
    # Store frequently used config and handles.
    self._denoiser_architecture_config = denoiser_architecture_config
    self._gpu_mesh = gpu_mesh
    self.grid2mesh_gnn = None

    # Processor - performs multiple rounds of message passing on the mesh.
    self.mesh_gnn = None

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    self.mesh2grid_gnn = None

    self._norm_conditioning_features = (
        denoiser_architecture_config.norm_conditioning_features
    )
    # Obtain the query radius in absolute units for the unit-sphere for the
    # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
    self._query_radius = (
        _get_max_edge_distance(self._mesh)
        * denoiser_architecture_config.radius_query_fraction_edge_length
    )

    # Other initialization is delayed until the first call (`_maybe_init`)
    # when we get some sample data so we know the lat/lon values.
    self._initialized = False

    # A "_init_mesh_properties":
    # This one could be initialized at init but we delay it for consistency too.
    self._num_mesh_nodes = None  # num_mesh_nodes
    self._mesh_nodes_lat = None  # [num_mesh_nodes]
    self._mesh_nodes_lon = None  # [num_mesh_nodes]

    # A "_init_grid_properties":
    self._grid_lat = None  # [num_lat_points]
    self._grid_lon = None  # [num_lon_points]
    self._num_grid_nodes = None  # num_lat_points * num_lon_points
    self._grid_nodes_lat = None  # [num_grid_nodes]
    self._grid_nodes_lon = None  # [num_grid_nodes]

    # A "_init_{grid2mesh,processor,mesh2grid}_graph"
    self._grid2mesh_graph_structure = None
    self._mesh_graph_structure = None
    self._mesh2grid_graph_structure = None

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               ) -> xarray.Dataset:
    # Initialize graph structures and networks with correct input feature sizes
    # based on the actual inputs/forcings present in this call.
    self._maybe_init(inputs, forcings)

    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    grid_node_features, global_norm_conditioning = (
        self._inputs_to_grid_node_features_and_norm_conditioning(
            inputs, forcings
        )
    )

    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(
        grid_node_features, global_norm_conditioning
    )

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    updated_latent_mesh_nodes = self._run_mesh_gnn(
        latent_mesh_nodes, global_norm_conditioning
    )
    # Transfer data from the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    output_grid_nodes = self._run_mesh2grid_gnn(
        updated_latent_mesh_nodes, latent_grid_nodes, global_norm_conditioning
    )
    # Convert output flat vectors for the grid nodes to the format of the
    # output. [num_grid_nodes, batch, output_size] -> xarray (batch, one time
    # step, lat, lon, level, multiple vars)
    return self._grid_node_outputs_to_prediction(
        output_grid_nodes, targets_template
    )

  def _maybe_init(self, sample_inputs: xarray.Dataset, sample_forcings: xarray.Dataset):
    if not self._initialized:
        self._init_mesh_properties()
        self._init_grid_properties(grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)

        # Compute data channel width seen by grid nodes to size the encoder correctly.
        sample_grid_feats, _ = self._inputs_to_grid_node_features_and_norm_conditioning(
            sample_inputs, sample_forcings
        )
        self._data_feature_size = int(sample_grid_feats.shape[-1])

        # Build graph templates. For grid2mesh, include dummy data channels so the
        # embedder input size = (struct + data) is fixed at init.
        self._grid2mesh_graph_structure = self._init_grid2mesh_graph(
            extra_node_features_size=self._data_feature_size
        )
        self._mesh_graph_structure = self._init_mesh_graph()
        self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

        # Now it’s safe to create networks (NNX eager init needs known input sizes).
        cfg = self._denoiser_architecture_config

        self.grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            aggregate_normalization=(cfg.grid2mesh_aggregate_normalization),
            edge_latent_size=dict(grid2mesh=cfg.latent_size),
            embed_edges=True,
            embed_nodes=True,
            f32_aggregation=True,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=cfg.latent_size,
            mlp_num_hidden_layers=cfg.hidden_layers,
            node_latent_size=dict(
                grid_nodes=cfg.latent_size,
                mesh_nodes=cfg.latent_size,
            ),
            node_output_size=None,
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=True,
            rngs=self.rngs,
            gpu_mesh=self._gpu_mesh,
            graph_template=self._grid2mesh_graph_structure,
        )

        self.mesh_gnn = transformer.MeshTransformer(
            transformer_kwargs=dataclasses.asdict(cfg.sparse_transformer_config),
            rngs=self.rngs,
            gpu_mesh=self._gpu_mesh,
            graph_template=self._mesh_graph_structure,
        )

        self.mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            activation="swish",
            edge_latent_size=dict(mesh2grid=cfg.latent_size),
            embed_nodes=False,
            f32_aggregation=False,
            include_sent_messages_in_node_update=False,
            mlp_hidden_size=cfg.latent_size,
            mlp_num_hidden_layers=cfg.hidden_layers,
            node_latent_size=dict(
                grid_nodes=cfg.latent_size,
                mesh_nodes=cfg.latent_size,
            ),
            node_output_size=dict(grid_nodes=cfg.node_output_size),
            num_message_passing_steps=1,
            use_layer_norm=True,
            use_norm_conditioning=True,
            rngs=self.rngs,
            gpu_mesh=self._gpu_mesh,
            graph_template=self._mesh2grid_graph_structure,
        )

        self._initialized = True


  def _init_mesh_properties(self):
    """Inits static properties that have to do with mesh nodes."""
    self._num_mesh_nodes = self._mesh.vertices.shape[0]
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        self._mesh.vertices[:, 0],
        self._mesh.vertices[:, 1],
        self._mesh.vertices[:, 2])
    mesh_nodes_lat, mesh_nodes_lon = model_utils.spherical_to_lat_lon(phi=mesh_phi, theta=mesh_theta)
    # Convert to f32 to ensure the lat/lon features aren't in f64.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Inits static properties that have to do with grid nodes."""
    self._grid_lat = grid_lat.astype(np.float32)
    self._grid_lon = grid_lon.astype(np.float32)
    # Initialized the counters.
    self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

    # Initialize lat and lon for the grid.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

  def _init_grid2mesh_graph(self, *, extra_node_features_size: int = 0) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""

    # Create some edges according to distance between mesh and grid nodes.
    assert self._grid_lat is not None and self._grid_lon is not None
    (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=self._grid_lat,
        grid_longitude=self._grid_lon,
        mesh=self._mesh,
        radius=self._query_radius)

    # Edges sending info from grid to mesh.
    senders = grid_indices
    receivers = mesh_indices

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._grid_nodes_lat,
         senders_node_lon=self._grid_nodes_lon,
         receivers_node_lat=self._mesh_nodes_lat,
         receivers_node_lon=self._mesh_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([mesh_indices.shape[0]])

    # If requested, append dummy zeros to node structural features so that
    # the embedder MLP is initialized with (struct + data) input width.
    if extra_node_features_size > 0:
      senders_node_features = np.concatenate(
          [senders_node_features,
           np.zeros((senders_node_features.shape[0], extra_node_features_size),
                    dtype=senders_node_features.dtype)],
          axis=-1,
      )
      receivers_node_features = np.concatenate(
          [receivers_node_features,
           np.zeros((receivers_node_features.shape[0], extra_node_features_size),
                    dtype=receivers_node_features.dtype)],
          axis=-1,
      )

    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=senders_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=receivers_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")):
            edge_set
    }
    grid2mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return grid2mesh_graph

  def _init_mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh graph."""
    # Work simply on the mesh edges.
    # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
    # should be reflected in the other 2 calls to faces_to_edges in this file.
    senders, receivers = icosahedral_mesh.faces_to_edges(self._mesh.faces)

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features(
        node_lat=self._mesh_nodes_lat,
        node_lon=self._mesh_nodes_lon,
        senders=senders,
        receivers=receivers,
        **self._spatial_features_kwargs,
    )

    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    assert n_mesh_node == len(node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set
    }
    mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    (grid_indices,
     mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
         grid_latitude=self._grid_lat,
         grid_longitude=self._grid_lon,
         mesh=self._mesh)

    # Edges sending info from mesh to grid.
    senders = mesh_indices
    receivers = grid_indices

    # Precompute structural node and edge features according to config options.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._mesh_nodes_lat,
         senders_node_lon=self._mesh_nodes_lon,
         receivers_node_lat=self._grid_nodes_lat,
         receivers_node_lon=self._grid_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=receivers_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=senders_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
            edge_set
    }
    mesh2grid_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return mesh2grid_graph

  def _run_grid2mesh_gnn(self, grid_node_features: chex.Array,
                        global_norm_conditioning: Optional[chex.Array] = None
                        ) -> tuple[chex.Array, chex.Array]:

    batch_size = grid_node_features.shape[1]


    #################################################################################
    # C_data = int(self._data_feature_size)
    # assert C_data == int(grid_node_features.shape[-1]), "Runtime data width changed."


    ###################################################################################
    C_data = grid_node_features.shape[-1]
    if self._data_feature_size is None:
        print(">>> [DEBUG] Setting _data_feature_size =", C_data)
        self._data_feature_size = C_data
    else:
        if C_data != self._data_feature_size:
            print(">>> [ERROR] Runtime data width changed!")
            print(">>> [ERROR] Expected:", self._data_feature_size)
            print(">>> [ERROR] Got:", C_data)

            # Also show variable-by-variable count
            print(">>> [ERROR] Variables included in grid_node_features:")
            if hasattr(self, "_last_variable_list"):
                print("    ", self._last_variable_list)

            raise AssertionError(
                f"Runtime data width changed: expected {self._data_feature_size}, got {C_data}"
            )


    grid2mesh_graph = self._grid2mesh_graph_structure
    assert grid2mesh_graph is not None

    grid_nodes_tpl = grid2mesh_graph.nodes["grid_nodes"]
    mesh_nodes_tpl = grid2mesh_graph.nodes["mesh_nodes"]

    # Template features are [STRUCT | DUMMY_DATA]
    grid_tpl = jnp.asarray(grid_nodes_tpl.features)
    mesh_tpl = jnp.asarray(mesh_nodes_tpl.features)
    Cg = int(grid_tpl.shape[-1]); Cm = int(mesh_tpl.shape[-1])
    Cg_struct = Cg - C_data
    Cm_struct = Cm - C_data
    if Cg_struct < 0 or Cm_struct < 0:
        raise ValueError("Template for grid2mesh must contain reserved dummy data channels.")

    # Build runtime features that EXACTLY match the template width = struct + data
    grid_struct = grid_tpl[:, :Cg_struct]
    mesh_struct = mesh_tpl[:, :Cm_struct]

    new_grid_nodes = grid_nodes_tpl._replace(
        features=jnp.concatenate([
            _add_batch_second_axis(grid_struct.astype(grid_node_features.dtype), batch_size),
            grid_node_features
        ], axis=-1)
    )

    # Mesh nodes carry zeros for the data block (same width)
    dummy_mesh_data = jnp.zeros((self._num_mesh_nodes, batch_size, C_data), dtype=grid_node_features.dtype)
    new_mesh_nodes = mesh_nodes_tpl._replace(
        features=jnp.concatenate([
            _add_batch_second_axis(mesh_struct.astype(grid_node_features.dtype), batch_size),
            dummy_mesh_data
        ], axis=-1)
    )

    # Broadcast edge features to batch
    grid2mesh_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
    edges_tpl = grid2mesh_graph.edges[grid2mesh_key]
    new_edges = edges_tpl._replace(
        features=_add_batch_second_axis(edges_tpl.features.astype(grid_node_features.dtype), batch_size)
    )

    # Assemble input graph
    input_graph = self._grid2mesh_graph_structure._replace(
        edges={grid2mesh_key: new_edges},
        nodes={"grid_nodes": new_grid_nodes, "mesh_nodes": new_mesh_nodes}
    )

    # Sanity: width must match template (avoids dot_general shape errors)
    assert new_grid_nodes.features.shape[-1] == grid_tpl.shape[-1]
    assert new_mesh_nodes.features.shape[-1] == mesh_tpl.shape[-1]

    out = self.grid2mesh_gnn(input_graph, global_norm_conditioning)
    return out.nodes["mesh_nodes"].features, out.nodes["grid_nodes"].features


  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array,
                    global_norm_conditioning: Optional[chex.Array] = None
                    ) -> chex.Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = latent_mesh_nodes.shape[1]

    mesh_graph = self._mesh_graph_structure
    assert mesh_graph is not None
    mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
    edges = mesh_graph.edges[mesh_edges_key]

    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named "mesh" for the edges and that it uses a single set of nodes named
    # "mesh_nodes"
    msg = ("The setup currently requires to only have one kind of edge in the"
           " mesh GNN.")
    assert len(mesh_graph.edges) == 1, msg

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_mesh_nodes.dtype), batch_size))

    nodes = mesh_graph.nodes["mesh_nodes"]
    nodes = nodes._replace(features=latent_mesh_nodes)

    input_graph = mesh_graph._replace(
        edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes})

    # Run the GNN.
    return self.mesh_gnn(input_graph,
                          global_norm_conditioning=global_norm_conditioning
                          ).nodes["mesh_nodes"].features

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
                         global_norm_conditioning: Optional[chex.Array] = None,
                         ) -> chex.Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = updated_latent_mesh_nodes.shape[1]

    mesh2grid_graph = self._mesh2grid_graph_structure
    assert mesh2grid_graph is not None
    mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
    grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
    mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
    edges = mesh2grid_graph.edges[mesh2grid_key]

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_grid_nodes.dtype), batch_size))

    input_graph = mesh2grid_graph._replace(
        edges={mesh2grid_key: new_edges},
        nodes={
            "mesh_nodes": new_mesh_nodes,
            "grid_nodes": new_grid_nodes
        })

    # Run the GNN.
    output_graph = self.mesh2grid_gnn(input_graph, global_norm_conditioning)
    output_grid_nodes = output_graph.nodes["grid_nodes"].features

    return output_grid_nodes

  def _inputs_to_grid_node_features_and_norm_conditioning(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> Tuple[chex.Array, Optional[chex.Array]]:
    """xarray ->[n_grid_nodes, batch, n_channels], [batch, n_cond channels]."""

    if self._norm_conditioning_features:
      norm_conditioning_inputs = inputs[list(self._norm_conditioning_features)]
      inputs = inputs.drop_vars(list(self._norm_conditioning_features))

      if "lat" in norm_conditioning_inputs or "lon" in norm_conditioning_inputs:
        raise ValueError("Features with lat or lon dims are not currently "
                         "supported for norm conditioning.")
      global_norm_conditioning = xarray_jax.unwrap_data(
          model_utils.dataset_to_stacked(norm_conditioning_inputs,
                                         preserved_dims=("batch",),
                                         ).transpose("batch", ...))

    else:
      global_norm_conditioning = None

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat(
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
        stacked_inputs)
    # ["node", "batch", "features"]
    grid_node_features = xarray_jax.unwrap(
        grid_xarray_lat_lon_leading.data
    ).reshape((-1,) + grid_xarray_lat_lon_leading.data.shape[2:])
    return grid_node_features, global_norm_conditioning

  def _grid_node_outputs_to_prediction(
      self,
      grid_node_outputs: chex.Array,
      targets_template: xarray.Dataset,
  ) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""

    # numpy array with shape [lat_lon_node, batch, channels]
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
    grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
        grid_shape + grid_node_outputs.shape[1:])
    dims = ("lat", "lon", "batch", "channels")
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)


def _add_batch_second_axis(data, batch_size):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([batch_size, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]


def _get_max_edge_distance(mesh):
  # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
  # should be reflected in the other 2 calls to faces_to_edges in this file.
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max()


def _permute_mesh_to_banded(mesh):
  """Permutes the mesh nodes such that adjacency matrix has banded structure."""
  # Build adjacency matrix.
  # N.B.To make sure ordering is preserved, any changes to faces_to_edges here
  # should be reflected in the other 2 calls to faces_to_edges in this file.
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
  num_mesh_nodes = mesh.vertices.shape[0]
  adj_mat = sparse.csr_matrix((num_mesh_nodes, num_mesh_nodes))
  adj_mat[senders, receivers] = 1
  # Permutation to banded (this algorithm is deterministic, a given sparse
  # adjacency matrix will yield the same permutation every time this is run).
  mesh_permutation = sparse.csgraph.reverse_cuthill_mckee(
      adj_mat, symmetric_mode=True
  )
  vertex_permutation_map = {j: i for i, j in enumerate(mesh_permutation)}
  permute_func = np.vectorize(lambda x: vertex_permutation_map[x])
  return icosahedral_mesh.TriangularMesh(
      vertices=mesh.vertices[mesh_permutation], faces=permute_func(mesh.faces)
  )
