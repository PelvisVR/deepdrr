from __future__ import annotations

from .renderable import *
from .devices import *


class PrimitiveInstance(TransformNodeContent):

    def __init__(
        self,
        primitive: Primitive,
        morph_weights: Optional[List[float]] = None,
    ):
        self.primitive = primitive
        self.morph_weights = morph_weights

    def to_render_instance(
        self, get_prim_id: Callable[[Primitive], int]
    ) -> RenderInstance:
        transform = self.node.tree.get_transform(None, self.node)
        return RenderInstance(
            primitive_id=get_prim_id(self.primitive),
            transform=geo_to_render_matrix4x4(transform),
            morph_weights=self.morph_weights,
        )


class Camera(TransformNodeContent):
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
