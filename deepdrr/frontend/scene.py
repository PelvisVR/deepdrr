from __future__ import annotations

from .node_content import *


class Scene(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_camera(self) -> Camera: ...

    @abstractmethod
    def get_render_primitives(self) -> List[RenderPrimitive]: ...

    @abstractmethod
    def instance_snapshot(self) -> List[RenderInstance]: ...


class GraphScene(Scene):

    def __init__(self, graph: TransformTree, camera: Optional[Camera] = None):
        super().__init__()
        self._prim_to_id = None
        self.graph = graph
        self._primitives = None
        self._camera = None
        self._render_primitives = None

        if camera is not None:
            self.set_camera(camera)

    def get_camera(self) -> Camera:
        if self._camera is None:
            for node in self.graph:
                for content in node:
                    if isinstance(content, Camera):
                        if self._camera is not None:
                            raise ValueError(
                                "Multiple cameras in scene, use set_camera to specify"
                            )
                        self._camera = content
        return self._camera

    def set_camera(self, camera: Camera):
        # make sure camera is in the graph
        for node in self.graph:
            for content in node:
                if content is camera:
                    self._camera = camera
                    return
        raise ValueError("Camera not in scene graph")

    def _get_instances(self) -> List[PrimitiveInstance]:
        instances = []
        # get the list of instances
        for node in self.graph:
            for content in node:
                if isinstance(content, PrimitiveInstance):
                    instances.append(content)
        # look up the primitives for each instance
        # if primitives have already been cached, ensure that no new primitives have been added
        if self._primitives is not None:
            for instance in instances:
                if instance.primitive not in self._primitives:
                    raise ValueError(
                        "Adding new primitives to the graph is not allowed after scene construction"
                    )

        return instances

    def _get_primitives(self) -> List[Primitive]:
        if self._primitives is None:
            prim_set = set()
            for instance in self._get_instances():
                prim_set.add(instance.primitive)
            self._primitives = list(prim_set)
            self._prim_to_id = {prim: i for i, prim in enumerate(self._primitives)}
        # pick an arbitrary order, make primitive lookup table
        # cache the return value
        return self._primitives

    def get_render_primitives(self) -> List[RenderPrimitive]:
        if self._render_primitives is None:
            self._render_primitives = [
                primitive.to_render_primitive() for primitive in self._get_primitives()
            ]
        return self._render_primitives

    def instance_snapshot(self) -> List[RenderInstance]:
        instances = self._get_instances()
        return [instance.to_render_instance(self.get_prim_id) for instance in instances]

    def get_prim_id(self, primitive: Primitive):
        if self._prim_to_id is None:
            self._get_primitives()
        return self._prim_to_id[primitive]
