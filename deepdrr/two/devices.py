from __future__ import annotations

from .scene import Camera
from .transform_manager import *
from .. import utils


class Device(TransformDriver, ABC):
    _camera: Camera

    def __init__(self, camera: Camera):
        assert camera is not None
        self._camera = camera

    @property
    def camera(self) -> Camera:
        return self._camera


class MobileCArm(Device):

    def __init__(
        self,
        world_from_isocenter: Optional[geo.FrameTransform] = None,
        isocenter_from_camera: Optional[geo.FrameTransform] = None,
        alpha: float = 0,
        beta: float = 0,
        gamma: float = 0,  # rotation of detector about principle ray
        degrees: bool = True,
        horizontal_movement: float = 200,  # width of window in X and Y planes.
        vertical_travel: float = 430,  # width of window in Z plane.
        min_alpha: float = -40,
        max_alpha: float = 110,
        # note that this would collide with the patient. Suggested to limit to +/- 45
        min_beta: float = -225,
        max_beta: float = 225,
        source_to_detector_distance: float = 1020,
        # vertical component of the source point offset from the isocenter of rotation, in -Z. Previously called `isocenter_distance`
        source_to_isocenter_vertical_distance: float = 530,
        # horizontal offset of the principle ray from the isocenter of rotation, in +Y. Defaults to 9, but should be 200 in document.
        source_to_isocenter_horizontal_offset: float = 0,
        # horizontal distance from principle ray to inner C-arm circumference. Used for visualization
        immersion_depth: float = 730,
        # distance from central ray to edge of arm. Used for visualization
        free_space: float = 820,
        sensor_height: int = 1536,
        sensor_width: int = 1536,
        pixel_size: float = 0.194,
        rotate_camera_left: bool = True,  # make it so that down in the image corresponds to -x, so that patient images appear as expected (when gamma=0).
        enforce_isocenter_bounds: bool = False,  # Allow the isocenter to travel arbitrarily far from the device origin
    ):

        self.alpha = utils.radians(alpha, degrees=degrees)
        self.beta = utils.radians(beta, degrees=degrees)
        self.gamma = utils.radians(gamma, degrees=degrees)
        self.horizontal_movement = horizontal_movement
        self.vertical_travel = vertical_travel
        self.min_alpha = utils.radians(min_alpha, degrees=degrees)
        self.max_alpha = utils.radians(max_alpha, degrees=degrees)
        self.min_beta = utils.radians(min_beta, degrees=degrees)
        self.max_beta = utils.radians(max_beta, degrees=degrees)
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_vertical_distance = (
            source_to_isocenter_vertical_distance
        )
        self.source_to_isocenter_horizontal_offset = (
            source_to_isocenter_horizontal_offset
        )
        self.immersion_depth = immersion_depth
        self.free_space = free_space
        self.pixel_size = pixel_size
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
            sensor_size=(sensor_width, sensor_height),
            pixel_size=pixel_size,
            source_to_detector_distance=self.source_to_detector_distance,
        )
        camera = Camera(
            intrinsic=self.camera_intrinsics,
            near=1,
            far=source_to_detector_distance,
        )
        super().__init__(camera)
        self._node_isocenter = TransformNode(transform=world_from_isocenter)
        self._node_camera = TransformNode(
            transform=isocenter_from_camera, contents=[self.camera]
        )
        self._camera.transform_node = self._node_camera

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

    def _add_as_child_of(self, node: TransformNode):
        node.add(self._node_isocenter)
        self._node_isocenter.add(self._node_camera)

    def add(self, node: TransformNode):
        self._node_isocenter.add(node)

    @property
    def base_node(self) -> TransformNode:
        return self._node_isocenter

    @property
    def world_from_isocenter(self) -> geo.FrameTransform:
        return self._node_isocenter.transform

    @world_from_isocenter.setter
    def world_from_isocenter(self, value: geo.FrameTransform):
        self._node_isocenter.transform = value

    @property
    def isocenter_from_camera(self) -> geo.FrameTransform:
        return self._node_camera.transform

    @isocenter_from_camera.setter
    def isocenter_from_camera(self, value: geo.FrameTransform):
        self._node_camera.transform = value
