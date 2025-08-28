"""A Transformer model for weather predictions.

This model wraps the transformer model and swaps the leading two axes of the
nodes in the input graph prior to evaluating the model to make it compatible
with a [nodes, batch, ...] ordering of the inputs.
"""

from typing import Any, Mapping, Optional, Type

from common import typed_graph
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from gencast import sparse_transformer

Kwargs = Mapping[str, Any]


def _get_adj_matrix_for_edge_set(
    graph: typed_graph.TypedGraph,
    edge_set_name: str,
    add_self_edges: bool,
):
  """Returns the adjacency matrix for the given graph and edge set."""
  # Get nodes and edges of the graph.
  edge_set_key = graph.edge_key_by_name(edge_set_name)
  sender_node_set, receiver_node_set = edge_set_key.node_sets

  # Compute number of sender and receiver nodes.
  sender_n_node = graph.nodes[sender_node_set].n_node[0]
  receiver_n_node = graph.nodes[receiver_node_set].n_node[0]

  # Build adjacency matrix.
  adj_mat = sparse.lil_matrix((sender_n_node, receiver_n_node), dtype=np.int32)
  edge_set = graph.edges[edge_set_key]
  # s, r = edge_set.indices
  s = edge_set.indices.senders
  r = edge_set.indices.receivers
  adj_mat[s, r] = True
  if add_self_edges:
    # Should only do this if we are certain the adjacency matrix is square.
    assert sender_node_set == receiver_node_set
    adj_mat.setdiag(True)
  adj_mat = adj_mat.tocsr()           # back to CSR for downstream use
  return adj_mat


class MeshTransformer(nnx.Module):
  """A Transformer for inputs with ordering [nodes, batch, ...]."""

  def __init__(self,
               transformer_kwargs: Kwargs,
               *, # rngs must be a keyword argument
               rngs: nnx.Rngs,
               mesh = None,
               graph_template: Optional[typed_graph.TypedGraph] = None,
               precomputed_adj_mat: Optional[Any] = None,
               ):
    """Initialises the Transformer model.

    Args:
      transformer_ctor: Constructor for transformer (the NNX Transformer class).
      transformer_kwargs: Kwargs to pass to the transformer module.
      rngs: The PRNG key for initializing any submodules.
      mesh: Optional mesh for model sharding across multiple devices.
      graph_template: Optional TypedGraph template for eager initialization.
        If provided, transformer will be initialized immediately. If None,
        transformer will be initialized lazily on first call.
      precomputed_adj_mat: Pre-computed adjacency matrix to avoid JIT issues.
      name: Optional name for nnx module.
    """
    self._transformer_kwargs = transformer_kwargs
    self._mesh = mesh
    
    if precomputed_adj_mat is not None:
      self.adj_mat = precomputed_adj_mat
    else:
      self.adj_mat = _get_adj_matrix_for_edge_set(
              graph=graph_template,
              edge_set_name='mesh',
              add_self_edges=True,
          )

    # If graph_template is provided, initialize transformer eagerly
    self.batch_first_transformer = sparse_transformer.Transformer(
        adj_mat=self.adj_mat,
        rngs=rngs,
        mesh=self._mesh,
        **self._transformer_kwargs,
    )

  def __call__(
      self, x: typed_graph.TypedGraph,
      global_norm_conditioning: jax.Array
  ) -> typed_graph.TypedGraph:
    """Applies the model to the input graph and returns graph of same shape."""

    if set(x.nodes.keys()) != {'mesh_nodes'}:
      raise ValueError(
          f'Expected x.nodes to have key `mesh_nodes`, got {x.nodes.keys()}.'
      )
    features = x.nodes['mesh_nodes'].features
    if features.ndim != 3:
      raise ValueError(
          'Expected `x.nodes["mesh_nodes"].features` to be 3, got'
          f' {features.ndim}.'
      )

    y = jnp.transpose(features, axes=[1, 0, 2])
    y = self.batch_first_transformer(y, global_norm_conditioning)
    y = jnp.transpose(y, axes=[1, 0, 2])
    x = x._replace(
        nodes={
            'mesh_nodes': x.nodes['mesh_nodes']._replace(
                features=y.astype(features.dtype)
            )
        }
    )
    return x