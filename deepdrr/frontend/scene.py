from __future__ import annotations

from .primitive import *

class Scene(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_camera(self) -> Camera: ...

    @abstractmethod
    def get_primitives(self) -> List[Primitive]: ...

    @abstractmethod
    def get_instances(self) -> List[PrimitiveInstance]: ...


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
        if self._camera is None:
            raise ValueError("No camera in scene")
        return self._camera

    def set_camera(self, camera: Camera):
        # make sure camera is in the graph
        for node in self.graph:
            for content in node:
                if content is camera:
                    self._camera = camera
                    return
        raise ValueError("Camera not in scene graph")

    def get_instances(self) -> List[PrimitiveInstance]:
        instances = []
        # get the list of instances
        for node in self.graph:
            for content in node:
                if isinstance(content, PrimitiveInstance):
                    content._set_scene(self)
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

    def get_primitives(self) -> List[Primitive]:
        if self._primitives is None:
            prim_set = set()
            for instance in self.get_instances():
                prim_set.add(instance.primitive)
            self._primitives = list(prim_set)
            self._prim_to_id = {prim: i for i, prim in enumerate(self._primitives)}
        # pick an arbitrary order, make primitive lookup table
        # cache the return value
        return self._primitives

    def get_prim_id(self, primitive: Primitive):
        if self._prim_to_id is None:
            self.get_primitives()
        return self._prim_to_id[primitive]


class GraphSceneContent(TransformNodeContent):
    _scene: Optional[GraphScene] = None

    def _set_scene(self, scene: Optional[GraphScene]):
        self._scene = scene

    @property
    def scene(self) -> Optional[GraphScene]:
        return self._scene


class PrimitiveInstance(GraphSceneContent):

    def __init__(
        self,
        primitive: Primitive,
        morph_weights: Optional[List[float]] = None,
        tags: Optional[Iterable[str]] = None,
    ):
        self.primitive = primitive
        self.morph_weights = morph_weights
        if tags is None:
            tags = set()
        self.tags = tags

    def to_render_instance(
        self,
    ) -> RenderInstance:
        transform = self.node.tree.get_transform(None, self.node)
        return RenderInstance(
            primitive_id=self._scene.get_prim_id(self.primitive),
            transform=geo_to_render_matrix4x4(transform),
            morph_weights=self.morph_weights,
        )


class Camera(GraphSceneContent):
    intrinsic: geo.CameraIntrinsicTransform
    near: float
    far: float

    def __init__(
        self,
        intrinsic: geo.CameraIntrinsicTransform,
        near: float,
        far: float,
    ):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far

    def to_render_camera_intrinsic(self) -> RenderCameraIntrinsic:
        return RenderCameraIntrinsic(
            fx=self.intrinsic.fx,
            fy=self.intrinsic.fy,
            cx=self.intrinsic.cx,
            cy=self.intrinsic.cy,
            near=self.near,
            far=self.far,
        )

    def to_render_camera(self) -> RenderCamera:
        return RenderCamera(
            transform=geo_to_render_matrix4x4(self.node.transform),
            intrinsic=self.to_render_camera_intrinsic(),
        )
