from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Any, Set, Union

import networkx as nx

from .volume_processing import *

log = logging.getLogger(__name__)


class TransformDriver(ABC):

    @abstractmethod
    def _add_as_child_of(self, node: TransformNode):
        """
        Add self as a child of node
        """
        ...

    @abstractmethod
    def add(self, node: TransformNode):
        """
        Add a node as a child to self
        """
        ...

    @property
    @abstractmethod
    def base_node(self) -> TransformNode:
        """
        Get the base node of the driver
        """
        ...


class TransformNodeContent(ABC):
    _node: Optional[TransformNode] = None

    def _set_node(self, node: Optional[TransformNode]):
        self._node = node

    def _get_node(self) -> Optional[TransformNode]:
        return self._node

    @property
    def node(self) -> Optional[TransformNode]:
        return self._get_node()

    @property
    def has_node(self) -> bool:
        return self._node is not None


class TransformNode:
    transform: geo.FrameTransform  # parent to self transform
    _contents: Set[TransformNodeContent]
    _tree: Optional[TransformTree]

    def __init__(
        self,
        transform: Optional[geo.FrameTransform] = None,
        contents: Optional[Iterable[Any]] = None,
    ):
        assert transform is None or isinstance(transform, geo.FrameTransform)
        assert contents is None or isinstance(contents, list)

        if transform is None:
            transform = geo.FrameTransform.identity()
        if contents is None:
            contents = []

        self._contents = set()
        self.add_contents(contents)

        self.transform = transform
        self._tree = None

    def _get_tree(self) -> TransformTree:
        if self._tree is None:
            raise ValueError("Node does not belong to a tree")
        return self._tree

    def _set_tree(self, value: TransformTree):
        if self._tree is not None and self._tree != value:
            raise ValueError("Node already belongs to a different tree")
        self._tree = value

    @property
    def tree(self) -> TransformTree:
        return self._get_tree()

    def add_contents(self, contents: Iterable[TransformNodeContent]):
        for content in contents:
            self.add_content(content)

    def add_content(self, content: TransformNodeContent):
        self._contents.add(content)
        content._set_node(self)

    def remove_content(self, content: TransformNodeContent):
        self._contents.remove(content)
        content._set_node(None)

    def add(self, node: Union[TransformNode, TransformDriver, TransformNodeContent]):
        if isinstance(node, TransformNode) or isinstance(node, TransformDriver):
            self._get_tree().add(node=node, parent=self)
        elif isinstance(node, TransformNodeContent):
            self.add_content(node)
        else:
            raise ValueError(f"Cannot add {type(node)} to TransformTreeNode")

    def __str__(self):
        return self.contents[0] if len(self.contents) > 0 else "TransformTreeNode"
        # return f"TransformTreeNode({self.transform}, {self.contents})"

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self._contents.__iter__()


class TransformTree:
    def __init__(self):
        self._g = nx.DiGraph()
        self._root_node = TransformNode()
        self._root_node._set_tree(self)

    def _add_tree_edge(self, parent: TransformNode, child: TransformNode):
        self._g.add_edge(parent, child, forward=True)
        self._g.add_edge(child, parent, forward=False)

    def _remove_tree_edge(self, parent: TransformNode, child: TransformNode):
        self._g.remove_edge(parent, child)
        self._g.remove_edge(child, parent)

    def get_parent(self, node: TransformNode) -> Optional[TransformNode]:
        try:
            return next(
                (x for x in self._g.neighbors(node) if not self._g[node][x]["forward"])
            )
        except StopIteration:
            return None

    def get_children(self, node: TransformNode) -> List[TransformNode]:
        return [x for x in self._g.neighbors(node) if self._g[node][x]["forward"]]

    def _forward_view(self) -> nx.DiGraph:
        return nx.subgraph_view(
            self._g, filter_edge=lambda u, v: self._g[u][v]["forward"]
        )

    def _reverse_view(self) -> nx.DiGraph:
        return nx.reverse_view(self._forward_view())

    def is_ancestor(self, ancestor: TransformNode, descendant: TransformNode) -> bool:
        return nx.has_path(self._reverse_view(), descendant, ancestor)

    def add(
        self,
        node: Union[TransformNode, TransformDriver],
        parent: Optional[TransformNode] = None,
    ):
        if parent is None:
            parent = self._root_node

        if isinstance(node, TransformDriver):
            node._add_as_child_of(parent)
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

    def get_transform(
        self,
        source: Union[TransformTree, TransformNode, None],
        target: Union[TransformTree, TransformNode, None],
    ) -> geo.FrameTransform:
        if source is None:
            source = self._root_node
        if target is None:
            target = self._root_node

        if isinstance(source, TransformTree):
            source = source._root_node
        if isinstance(target, TransformTree):
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
            forward = self._g[path[i]][path[i + 1]]["forward"]
            if forward:
                transform = transform @ path[i + 1].transform
            else:
                transform = transform @ path[i].transform.inverse()

        return transform

    def __iter__(self):
        return self._g.__iter__()

    def draw(self):
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self._g)
        g_forward = self._forward_view()
        nx.draw(
            g_forward,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            linewidths=1,
            font_color="black",
        )
        plt.show()