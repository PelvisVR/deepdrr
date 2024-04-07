from __future__ import annotations
import os
from pathlib import Path

import networkx as nx
from typing import Callable, Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import nrrd
import numpy as np
import h5py
import logging
import tempfile
import hashlib
import shutil

from .. import load_dicom
from .volume_processing import *

from deepdrr import geo

log = logging.getLogger(__name__)

class TransformDriver(ABC):
    # drives the transform graph

    @abstractmethod
    def _add_to(self, node: TransformNode): ...
    # add self as a child to node

    @abstractmethod
    def add(self, node: TransformNode): ...
    # add a node as a child to self


class TransformNode:
    transform: geo.FrameTransform # parent to self transform
    contents: List[Any]
    _tree: TransformTree

    def __init__(self, transform: Optional[geo.FrameTransform] = None, contents: Optional[List[Any]] = None):
        assert transform is None or isinstance(transform, geo.FrameTransform)
        assert contents is None or isinstance(contents, list)

        if transform is None:
            transform = geo.FrameTransform.identity()
        if contents is None:
            contents = []
            
        self.transform = transform
        self.contents = contents
        self._tree = None    

    def _get_tree(self) -> TransformTree:
        if self._tree is None:
            raise ValueError("Node does not belong to a tree")
        return self._tree
    
    def _set_tree(self, value: TransformTree):
        if self._tree is not None and self._tree != value:
            raise ValueError("Node already belongs to a different tree")
        self._tree = value

    def add(self, node: TransformNode):
        # add a node as a child to self
        self._get_tree().add(node=node, parent=self)

    def __str__(self):
        return self.contents[0] if len(self.contents) > 0 else "TransformTreeNode"
        # return f"TransformTreeNode({self.transform}, {self.contents})"

    def __repr__(self):
        return self.__str__()


class TransformTree:
    def __init__(self):
        self._g = nx.DiGraph()
        self._root_node = TransformNode(contents=["root"])
        self._root_node._set_tree(self)

    def _add_tree_edge(self, parent: TransformNode, child: TransformNode):
        self._g.add_edge(parent, child, forward=True)
        self._g.add_edge(child, parent, forward=False)

    def _remove_tree_edge(self, parent: TransformNode, child: TransformNode):
        self._g.remove_edge(parent, child)
        self._g.remove_edge(child, parent)

    def get_parent(self, node: TransformNode) -> Optional[TransformNode]:
        try:
            return next((x for x in self._g.neighbors(node) if not self._g[node][x]['forward']))
        except StopIteration:
            return None

    def get_children(self, node: TransformNode) -> List[TransformNode]:
        return [x for x in self._g.neighbors(node) if self._g[node][x]['forward']]

    def _forward_view(self) -> nx.DiGraph:
        return nx.subgraph_view(self._g, filter_edge=lambda u, v: self._g[u][v]['forward'])

    def _reverse_view(self) -> nx.DiGraph:
        return nx.reverse_view(self._forward_view())

    def is_ancestor(self, ancestor: TransformNode, descendant: TransformNode) -> bool:
        return nx.has_path(self._reverse_view(), descendant, ancestor)

    def add(self, node: Union[TransformNode, TransformDriver], parent: Optional[TransformNode] = None):
        if parent is None:
            parent = self._root_node

        if isinstance(node, TransformDriver):
            node._add_to(parent)
            return

        self._add_node(node, parent)

    def _add_node(self, node: TransformNode, parent: TransformNode):
        if not isinstance(node, TransformNode):
            raise ValueError(f"Node must be a TransformTreeNode not {type(node)}")
        if not isinstance(parent, TransformNode):
            raise ValueError(f"Parent must be a TransformTreeNode not {type(parent)}")

        # if parent is self, raise an error
        if parent == node:
            raise ValueError("Node cannot be its own parent")

        # is the node already in the tree?
        if node in self._g:
            old_parent = self.get_parent(node)
            # if the parent is the same, do nothing
            if old_parent == parent:
                return
            # if the new parent is a descendant of the node, raise an error
            if self.is_ancestor(ancestor=node, descendant=parent):
                raise ValueError("Cannot make a node a descendant of itself")
            # the new parent is not a descendant of the node
            # remove the parent edge
            self._remove_tree_edge(parent=old_parent, child=node)
            # add the new parent edge
            self._add_tree_edge(parent=parent, child=node)
            return

        node._set_tree(self)
        self._g.add_node(node)
        self._add_tree_edge(parent, node)

    def remove(self, node: TransformNode):
        if node == self._root_node:
            raise ValueError("Cannot remove root node")

        if node not in self._g:
            raise ValueError("Node not in tree")

        # remove all children
        for child in self.get_children(node):
            self.remove(child)

        node._tree = None
        self._g.remove_node(node)

    def get_transform(self, source: Union[TransformTree, TransformNode], target: Union[TransformTree, TransformNode]) -> geo.FrameTransform:
        if source is None or isinstance(source, TransformTree):
            source = source._root_node
        if target is None or isinstance(target, TransformTree):
            target = target._root_node

        if source == target:
            return geo.FrameTransform.identity()

        if source not in self._g:
            raise ValueError("Source not in tree")
        if target not in self._g:
            raise ValueError("Target not in tree")

        try:
            path = nx.shortest_path(self._g, source, target)
        except nx.NetworkXNoPath:
            raise ValueError("No path between source and target")

        transform = geo.FrameTransform.identity()

        for i in range(len(path) - 1):
            forward = self._g[path[i]][path[i+1]]['forward']
            if forward:
                transform = transform @ path[i+1].transform
            else:
                transform = transform @ path[i].transform.inverse()

        return transform

    def __iter__(self):
        return self._g.__iter__()

    def draw(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self._g)
        g_forward = self._forward_view()
        nx.draw(g_forward, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", linewidths=1, font_color="black")
        plt.show()
