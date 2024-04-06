from deepdrr import geo
from typing import List, Optional, Any, Set
from abc import ABC, abstractmethod

from deepdrr.serial.render import *
from .transform_manager import *

class Primitive:
    pass

class PrimitiveInstance:
    def __init__(self, primitive: Primitive, transform_node: TransformTreeNode, morph_weights: List[float]):
        self.primitive = primitive
        self.transform_node = transform_node
        self.morph_weights = morph_weights

class Camera:
    pass

# class InstanceData:
#     primitive_id: int
#     transform: geo.FrameTransform
#     morph_weights: List[float]

class Scene(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_camera(self) -> Camera:
        ...

    @abstractmethod
    def get_primitives(self) -> List[Primitive]:
        ...

    @abstractmethod
    def instance_snapshot(self) -> List[RenderInstance]:
        ...

class GraphScene(Scene):

    def __init__(self, graph: TransformTree):
        self.graph = graph
        self._primitives = None
        self._camera = None

    def get_camera(self) -> Camera:
        if self._camera is None:
            for node in self.graph:
                for content in node.contents:
                    if isinstance(content, Camera):
                        self._camera = content
        return self._camera

    def _get_instances(self) -> List[PrimitiveInstance]:
        instances = []
        # get the list of instances
        for node in self.graph:
            for content in node.contents:
                if isinstance(content, PrimitiveInstance):
                    instances.append(content)
        # look up the primitives for each instance
        # if primitives have already been cached, ensure that no new primitives have been added
        if self._primitives is not None:
            for instance in instances:
                if instance.primitive not in self._primitives:
                    raise ValueError("Adding new primitives to the graph is not allowed after scene construction")
                
        return instances

    def get_primitives(self) -> List[Primitive]:
        if self._primitives is None:
            prim_set = set()
            for instance in self._get_instances():
                prim_set.add(instance.primitive)
            self._primitives = list(prim_set)
            self._prim_to_id = {prim: i for i, prim in enumerate(self._primitives)}
        # pick an arbitrary order, make primitive lookup table
        # cache the return value
        return self._primitives

    def instance_snapshot(self) -> List[RenderInstance]:
        self.get_primitives()
        instances = self._get_instances()
        render_instance_list = []
        for instance in instances:
            render_instance_list.append(RenderInstance(
                primitive_id=self._prim_to_id[instance.primitive],
                transform=geo_to_render_matrix4x4(self.graph.get_transform(None, instance.transform_node)),
                morph_weights=instance.morph_weights
            ))
        return render_instance_list

class TransformDriver(ABC):
    # drives the transform graph
    pass

class MobileCArm(TransformDriver):

    # isocenter node field
    # camera node field

    def __init__(self, graph: TransformTree, parent):
        # make a node, attach it to parent
        # make a sub-node, attach it to node
        # assign a camera to the sub-node
        pass

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_alpha: Optional[float] = None,
        delta_beta: Optional[float] = None,
        delta_gamma: Optional[float] = None,
        degrees: bool = True,
    ) -> None:
        # move the nodes
        pass

class Renderable(TransformDriver):
    pass

class Mesh(Renderable):
    # not allowed to change vertex data or material data after construction
    pass

class Volume(Renderable):
    # not allowed to change volume data or material data after construction
    pass


class RenderProfile(ABC):
    pass

class DRRRenderProfile(RenderProfile):
    pass

class RasterizeRenderProfile(RenderProfile):
    pass

