# Copyright 2023 DeepMind Technologies Limited.
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
"""A library of typed Graph Neural Networks."""

from typing import Callable, Mapping, Optional, Union

from common import typed_graph
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import chex
import flax.nnx as nnx

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = (
    jraph.ArrayTree)

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, Mapping[str, SenderFeatures], Mapping[str, ReceiverFeatures],
     Globals],
    NodeFeatures]

GNUpdateGlobalFn = Callable[
    [Mapping[str, NodeFeatures], Mapping[str, EdgeFeatures], Globals],
    Globals]


class GraphNetwork(nnx.Module):  # pylint: disable=invalid-name
  """A Graph Network that applies updates to edges, nodes, and globals.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
  extended to Typed Graphs with multiple edge sets and node sets and extended to
  allow aggregating not only edges received by the nodes, but also edges sent by
  the nodes.

  Example usage::

    gn = GraphNetwork(update_edge_fns, update_node_fns, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)
  """

  def __init__(self,
               update_edge_fns: Mapping[str, nnx.Module],
               update_node_fns: Mapping[str, nnx.Module],
               update_global_fn: Optional[nnx.Module] = None,
               aggregate_edges_for_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum,
               aggregate_nodes_for_globals_fn: jraph.AggregateNodesToGlobalsFn = jraph.segment_sum,
               aggregate_edges_for_globals_fn: jraph.AggregateEdgesToGlobalsFn = jraph.segment_sum):
    """Initializes the GraphNetwork.

    Args:
      update_edge_fns: mapping of NNX modules used to update a subset of the edge
        types, indexed by edge type name.
      update_node_fns: mapping of NNX modules used to update a subset of the node
        types, indexed by node type name.
      update_global_fn: NNX module used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
    """
    self.update_edge_fns = update_edge_fns
    self.update_node_fns = update_node_fns
    self.update_global_fn = update_global_fn
    self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
    self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
    self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn

  def __call__(self,
               graph: typed_graph.TypedGraph, 
               global_norm_conditioning: Optional[chex.Array] = None
               ) -> typed_graph.TypedGraph:
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
    extended to Typed Graphs with multiple edge sets and node sets and extended
    to allow aggregating not only edges received by the nodes, but also edges
    sent by the nodes.

    Args:
      graph: a `TypedGraph` containing the graph.
      global_norm_conditioning: Optional global norm conditioning.

    Returns:
      Updated `TypedGraph`.
    """

    updated_graph = graph

    # Edge update.
    updated_edges = dict(updated_graph.edges)
    for edge_set_name, edge_fn in self.update_edge_fns.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      updated_edges[edge_set_key] = self._edge_update(
          updated_graph, edge_fn, edge_set_key, global_norm_conditioning)
    updated_graph = updated_graph._replace(edges=updated_edges)

    # Node update.
    updated_nodes = dict(updated_graph.nodes)
    for node_set_key, node_fn in self.update_node_fns.items():
      updated_nodes[node_set_key] = self._node_update(
          updated_graph, node_fn, node_set_key, self.aggregate_edges_for_nodes_fn, global_norm_conditioning)
    updated_graph = updated_graph._replace(nodes=updated_nodes)

    # Global update.
    if self.update_global_fn:
      updated_context = self._global_update(
          updated_graph, self.update_global_fn,
          self.aggregate_edges_for_globals_fn,
          self.aggregate_nodes_for_globals_fn)
      updated_graph = updated_graph._replace(context=updated_context)

    return updated_graph

  def _edge_update(self, graph, edge_fn, edge_set_key, global_norm_conditioning):  # pylint: disable=invalid-name
    """Updates an edge set of a given key."""

    sender_nodes = graph.nodes[edge_set_key.node_sets[0]]
    receiver_nodes = graph.nodes[edge_set_key.node_sets[1]]
    edge_set = graph.edges[edge_set_key]
    senders = edge_set.indices.senders  # pytype: disable=attribute-error
    receivers = edge_set.indices.receivers  # pytype: disable=attribute-error

    sent_attributes = tree.tree_map(
        lambda n: n[senders], sender_nodes.features)
    received_attributes = tree.tree_map(
        lambda n: n[receivers], receiver_nodes.features)

    n_edge = edge_set.n_edge
    sum_n_edge = senders.shape[0]
    global_features = tree.tree_map(
        lambda g: jnp.repeat(g, n_edge, axis=0, total_repeat_length=sum_n_edge),
        graph.context.features)
    new_features = edge_fn(
        edge_set.features, 
        sent_attributes, 
        received_attributes,
        global_features, 
        global_norm_conditioning=global_norm_conditioning)
    return edge_set._replace(features=new_features)

  def _node_update(self, graph, node_fn, node_set_key, aggregation_fn, global_norm_conditioning):  # pylint: disable=invalid-name
    """Updates a node set of a given key."""
    node_set = graph.nodes[node_set_key]
    sum_n_node = tree.tree_leaves(node_set.features)[0].shape[0]

    sent_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      sender_node_set_key = edge_set_key.node_sets[0]
      if sender_node_set_key == node_set_key:
        assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
        senders = edge_set.indices.senders
        sent_features[edge_set_key.name] = tree.tree_map(
            lambda e: aggregation_fn(e, senders, sum_n_node), edge_set.features)  # pylint: disable=cell-var-from-loop

    received_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      receiver_node_set_key = edge_set_key.node_sets[1]
      if receiver_node_set_key == node_set_key:
        assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
        receivers = edge_set.indices.receivers
        received_features[edge_set_key.name] = tree.tree_map(
            lambda e: aggregation_fn(e, receivers, sum_n_node), edge_set.features)  # pylint: disable=cell-var-from-loop

    n_node = node_set.n_node
    global_features = tree.tree_map(
        lambda g: jnp.repeat(g, n_node, axis=0, total_repeat_length=sum_n_node),
        graph.context.features)

    new_features = node_fn(
        node_set.features, 
        sent_features, 
        received_features, 
        global_features, 
        global_norm_conditioning=global_norm_conditioning)
    return node_set._replace(features=new_features)

  def _global_update(self, graph, global_fn, edge_aggregation_fn, node_aggregation_fn):  # pylint: disable=invalid-name
    """Updates the global features of a graph."""
    n_graph = graph.context.n_graph.shape[0]
    graph_idx = jnp.arange(n_graph)

    edge_features = {}
    for edge_set_key, edge_set in graph.edges.items():
      assert isinstance(edge_set.indices, typed_graph.EdgesIndices)
      sum_n_edge = edge_set.indices.senders.shape[0]
      edge_gr_idx = jnp.repeat(
          graph_idx, edge_set.n_edge, axis=0, total_repeat_length=sum_n_edge)
      edge_features[edge_set_key.name] = tree.tree_map(
          lambda e: edge_aggregation_fn(e, edge_gr_idx, n_graph),   # pylint: disable=cell-var-from-loop
          edge_set.features)

    node_features = {}
    for node_set_key, node_set in graph.nodes.items():
      sum_n_node = tree.tree_leaves(node_set.features)[0].shape[0]
      node_gr_idx = jnp.repeat(
          graph_idx, node_set.n_node, axis=0, total_repeat_length=sum_n_node)
      node_features[node_set_key] = tree.tree_map(
          lambda n: node_aggregation_fn(n, node_gr_idx, n_graph),   # pylint: disable=cell-var-from-loop
          node_set.features)

    new_features = global_fn(node_features, edge_features, graph.context.features)
    return graph.context._replace(features=new_features)


InteractionUpdateNodeFn = Callable[
    [jraph.NodeFeatures,
     Mapping[str, SenderFeatures],
     Mapping[str, ReceiverFeatures]],
    jraph.NodeFeatures]


InteractionUpdateNodeFnNoSentEdges = Callable[
    [jraph.NodeFeatures,
     Mapping[str, ReceiverFeatures]],
    jraph.NodeFeatures]


class InteractionNetwork(nnx.Module):  # pylint: disable=invalid-name
  """An Interaction Network for computing interactions on graphs.

  An interaction network computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incoming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This implementation extends the behavior to `TypedGraphs` adding an option
  to include edge features for which a node is a sender in the arguments to
  the node update function.
  """

  def __init__(self,
               update_edge_fns: Mapping[str, nnx.Module],
               update_node_fns: Mapping[str, nnx.Module],
               aggregate_edges_for_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum,
               include_sent_messages_in_node_update: bool = False):
    """Initializes the InteractionNetwork.

    Args:
      update_edge_fns: mapping of NNX modules used to update a subset of the edge
        types, indexed by edge type name.
      update_node_fns: mapping of NNX modules used to update a subset of the node
        types, indexed by node type name.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      include_sent_messages_in_node_update: pass edge features for which a node is
        a sender to the node update function.
    """
    # An InteractionNetwork is a GraphNetwork without globals features,
    # so we implement the InteractionNetwork as a configured GraphNetwork.

    # An InteractionNetwork edge function does not have global feature inputs,
    # so we filter the passed global argument in the GraphNetwork.
    wrapped_update_edge_fns = {}
    for edge_name, fn in update_edge_fns.items():
      wrapped_update_edge_fns[edge_name] = EdgeWrapper(fn)

    # Similarly, we wrap the update_node_fn to ensure only the expected
    # arguments are passed to the Interaction net.
    wrapped_update_node_fns = {}
    for node_name, fn in update_node_fns.items():
      wrapped_update_node_fns[node_name] = NodeWrapper(fn, include_sent_messages_in_node_update)

    self.graph_network = GraphNetwork(
        update_edge_fns=wrapped_update_edge_fns,
        update_node_fns=wrapped_update_node_fns,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)

  def __call__(self,
               graph: typed_graph.TypedGraph,
               global_norm_conditioning: Optional[chex.Array] = None
               ) -> typed_graph.TypedGraph:
    return self.graph_network(graph, global_norm_conditioning)


class EdgeWrapper(nnx.Module):
  """Wrapper for edge update functions in InteractionNetwork."""
  
  def __init__(self, edge_fn: nnx.Module):
    self.edge_fn = edge_fn
  
  def __call__(self, e, s, r, g, global_norm_conditioning=None):
    return self.edge_fn(
        jnp.concatenate([e, s, r], axis=-1),
        global_norm_conditioning
    )


class NodeWrapper(nnx.Module):
  """Wrapper for node update functions in InteractionNetwork."""
  
  def __init__(self, node_fn: nnx.Module, include_sent_messages: bool):
    self.node_fn = node_fn
    self.include_sent_messages = include_sent_messages
  
  def __call__(self, n, s, r, g, global_norm_conditioning=None):
    if self.include_sent_messages:
      inputs = jnp.concatenate(
          tree.tree_leaves(n) + tree.tree_leaves(s) + tree.tree_leaves(r), 
          axis=-1
      )
    else:
      inputs = jnp.concatenate(
          tree.tree_leaves(n) + tree.tree_leaves(r), 
          axis=-1
      )
    return self.node_fn(inputs, global_norm_conditioning)


class GraphMapFeatures(nnx.Module):  # pylint: disable=invalid-name
  """Module which embeds the components of a graph independently."""

  def __init__(self,
               embed_edge_fns: Optional[Mapping[str, nnx.Module]] = None,
               embed_node_fns: Optional[Mapping[str, nnx.Module]] = None,
               embed_global_fn: Optional[nnx.Module] = None):
    """Initializes the GraphMapFeatures module.

    Args:
      embed_edge_fns: mapping of NNX modules used to embed each edge type,
        indexed by edge type name.
      embed_node_fns: mapping of NNX modules used to embed each node type,
        indexed by node type name.
      embed_global_fn: NNX module used to embed the globals.
    """
    self.embed_edge_fns = embed_edge_fns or {}
    self.embed_node_fns = embed_node_fns or {}
    self.embed_global_fn = embed_global_fn

  def __call__(self,
               graph: typed_graph.TypedGraph,
               global_norm_conditioning: Optional[chex.Array] = None
               ) -> typed_graph.TypedGraph:
    """Embeds the components of a graph independently.

    Args:
      graph: a `TypedGraph` containing the graph.
      global_norm_conditioning: Optional global norm conditioning.

    Returns:
      Updated `TypedGraph` with embedded features.
    """

    updated_edges = dict(graph.edges)
    if self.embed_edge_fns:
      for edge_set_name, embed_fn in self.embed_edge_fns.items():
        edge_set_key = graph.edge_key_by_name(edge_set_name)
        edge_set = graph.edges[edge_set_key]
        updated_edges[edge_set_key] = edge_set._replace(
            features=embed_fn(edge_set.features, global_norm_conditioning))

    updated_nodes = dict(graph.nodes)
    if self.embed_node_fns:
      for node_set_key, embed_fn in self.embed_node_fns.items():
        node_set = graph.nodes[node_set_key]
        updated_nodes[node_set_key] = node_set._replace(
            features=embed_fn(node_set.features, global_norm_conditioning))

    updated_context = graph.context
    if self.embed_global_fn:
      updated_context = updated_context._replace(
          features=self.embed_global_fn(updated_context.features))

    return graph._replace(edges=updated_edges, nodes=updated_nodes,
                          context=updated_context)