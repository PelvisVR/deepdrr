from typing import Optional, Tuple, Union, List

import logging
import numpy as np
from numpy.lib.utils import source
from scipy.spatial.transform import Rotation

from . import geo
from . import utils

pv, pv_available = utils.try_import_pyvista()


logger = logging.getLogger(__name__)

PI = np.float32(np.pi)
DEFAULT_MIN_ISOCENTER = [None, -100, -215]
DEFAULT_MAX_ISOCENTER = [None, 100, 215]
DEFAULT_MIN_ALPHA = -2 * PI / 3 # -120
DEFAULT_MAX_ALPHA = 2 * PI / 3 # 120
DEFAULT_MIN_BETA = -PI / 4 # -45
DEFAULT_MAX_BETA = PI / 4 # 45
DEFAULT_SENSOR_HEIGHT = 960
DEFAULT_SENSOR_WIDTH = 1240
DEFAULT_PIXEL_SIZE = 0.31
DEFAULT_SOURCE_TO_DETECTOR_DISTANCE = 1020


def make_detector_rotation(phi: float, theta: float, rho: float):
    """Make the rotation matrix for a CArm detector at the given angles.

    Args:
        phi (float): phi.
        theta (float): theta.
        rho (float): rho

    Returns:
        np.ndarray: Rotation matrix.
    """
    # rotation around phi and theta
    sin_p = np.sin(phi)
    neg_cos_p = -np.cos(phi)
    z = 0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    omc = 1 - cos_t

    # Rotation by theta about vector [sin(phi), -cos(phi), z].
    R = np.array([
        [
            sin_p * sin_p * omc + cos_t,
            sin_p * neg_cos_p * omc - z * sin_t, 
            sin_p * z * omc + neg_cos_p * sin_t,
        ],
        [
            sin_p * neg_cos_p * omc + z * sin_t,
            neg_cos_p * neg_cos_p * omc + cos_t,
            neg_cos_p * z * omc - sin_p * sin_t,
        ],
        [
            sin_p * z * omc - neg_cos_p * sin_t,
            neg_cos_p * z * omc + sin_p * sin_t,
            z * z * omc + cos_t,
        ]])
    
    rho = -phi + np.pi * 0.5 + rho
    R_principle = np.array([[np.cos(rho), -np.sin(rho), 0],
                            [np.sin(rho), np.cos(rho), 0],
                            [0, 0, 1]])
    R = np.matmul(R_principle, R)

    return R


def pose_vector_angles(pose: geo.Vector3D) -> Tuple[float, float]:
    """Get the C-arm angles alpha, beta corrsponding the the pose vector.

    TODO(killeen): make a part of the MobileCArm object, to convert from a world-space vector.

    Args:
        pose (geo.Vector3D): the vector pointing from the isocenter (or camera) to the detector.

    Returns:
        Tuple[float, float]: carm angulation (alpha, beta) in radians.
    """
    x, y, z = pose
    alpha = np.arctan2(y, np.sqrt(x * x + z * z))
    beta = -np.arctan2(x, z)
    return alpha, beta


class MobileCArm(object):

    isocenter: geo.Point3D # the isocenter point in the device frame
    alpha: float # alpha angle in radians
    beta: float # beta angle in radians

    def __init__(
        self,
        isocenter: geo.Point3D = [0, 0, 0],
        alpha: float = 0,
        beta: float = 0,
        world_from_device: Optional[geo.FrameTransform] = None,
        horizontal_movement: float = 200, # width of window in X and Y planes.
        vertical_travel: float = 430, # width of window in Z plane.
        min_alpha: float = -40,
        max_alpha: float = 110,
        min_beta: float = -225, # note that this would collide with the patient. Suggested to limit to +/- 45
        max_beta: float = 225,
        degrees: bool = True,
        source_to_detector_distance: float = 1020,
        source_to_isocenter_vertical_distance: float = 530, # vertical component of the source point offset from the isocenter of rotation, in -Z. Previously called `isocenter_distance`
        source_to_isocenter_horizontal_offset: float = 200, # horizontal offset of the principle ray from the isocenter of rotation, in +Y
        immersion_depth: float = 730, # horizontal distance from principle ray to inner C-arm circumference. Used for visualization
        free_space: float = 820, # distance from central ray to edge of arm. Used for visualization
        sensor_height: int = 1536,
        sensor_width: int = 1536,
        pixel_size: float = 0.194,
    ) -> None:
        """Make a CArm device.

        All units in mm.

        The C-arm frame is centered in the isocenter movement window, with -Y pointed toward the "Arm" and Z pointing up.
        
        The actual point being visualized is offset from the isocenter.

        The angulation and orbital angles are reversed from figure 2 in Kausch et al: https://pubmed.ncbi.nlm.nih.gov/32533315/

        TODO(killeen): the C-Arm is really a fancy CameraProjection. Inherit from that?

        Rather than incorporating a swivel, the device allows for horizontal translations, based on += 12 degrees (by default).
        
        Defaults based on the Siemens Cios Fusion device. Specifications here:
        https://www.lomisa.com/app/download/10978681598/ARCO_EN_C_SIEMENS_CIOS_FUSION.pdf?t=1490962065 

        Args:
            isocenter_distance (float): the distance from the X-ray source to the isocenter of the CAarm. (The center of rotation).
            isocenter (Point3D): isocenter of the C-arm in carm-space. This is the center about which rotations are performed.
            alpha (float): initial LAO/RAO angle of the C-Arm. alpha > 0 is in the RAO direction. This is the angle along arm of the C-arm.
            beta (float): initial CRA/CAU angulation of the C-arm. beta > 0 is in the CAU direction.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
            world_from_device: (Optional[geo.FrameTransform], optional): Transform that defines the device coordinate space in world coordinates. None is the identity transform. Defaults to None.
            camera_intrinsics: (Optional[Union[geo.CameraIntrinsicTransform, dict]], optional): either a CameraIntrinsicTransform instance or kwargs for CameraIntrinsicTransform.from_sizes
        """
        self.isocenter = geo.point(isocenter)
        self.alpha = utils.radians(alpha, degrees=degrees)
        self.beta = utils.radians(beta, degrees=degrees)
        self.world_from_device = geo.frame_transform(world_from_device)
        self.horizontal_movement = horizontal_movement
        self.vertical_travel = vertical_travel
        self.min_alpha = utils.radians(min_alpha, degrees=degrees)
        self.max_alpha = utils.radians(max_alpha, degrees=degrees)
        self.min_beta = utils.radians(min_beta, degrees=degrees)
        self.max_beta = utils.radians(max_beta, degrees=degrees)
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_vertical_distance = source_to_isocenter_vertical_distance
        self.source_to_isocenter_horizontal_offset = source_to_isocenter_horizontal_offset
        self.immersion_depth = immersion_depth
        self.free_space = free_space
        self.camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
            sensor_size=(sensor_width, sensor_height),
            pixel_size=pixel_size,
            source_to_detector_distance=self.source_to_detector_distance,
        )

        self._static_mesh = None

    def __str__(self):
        return f'MobileCArm(isocenter={np.array_str(np.array(self.isocenter))}, alpha={np.degrees(self.alpha)}, beta={np.degrees(self.beta)}, degrees=True)'

    @property
    def max_isocenter(self) -> np.ndarray:
        return np.array([self.horizontal_movement, self.horizontal_movement, self.vertical_travel]) / 2

    @property
    def min_isocenter(self) -> np.ndarray:
        return -self.max_isocenter

    @property
    def device_from_world(self) -> geo.FrameTransform:
        return self.world_from_device.inv

    @property
    def isocenter_in_world(self) -> geo.Point3D:
        return self.world_from_device @ self.isocenter

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_alpha: Optional[float] = None,
        delta_beta: Optional[float] = None,
        degrees: bool = True,
    ) -> None:
        """Move the C-arm to the specified pose.

        TODO: don't let out-of-bounds movement pass silently.

        Args:
            delta_isocenter (Optional[geo.Vector3D], optional): isocenter (Point3D): isocenter of the C-arm in C-arm-space. This is the center about which rotations are performed. Defaults to None.
            delta_alpha (Optional[float], optional): change in alpha. Defaults to None.
            delta_beta (Optional[float], optional): change in beta. Defaults to None.
            degrees (bool, optional): whether the given angles are in degrees. Defaults to False.

        """
        if delta_isocenter is not None:
            self.isocenter += geo.vector(delta_isocenter)
            self.isocenter = geo.point(np.clip(self.isocenter, self.min_isocenter, self.max_isocenter))

        if delta_alpha is not None:
            assert np.isscalar(delta_alpha)
            self.alpha += utils.radians(float(delta_alpha), degrees=degrees)
            self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)

        if delta_beta is not None:
            assert np.isscalar(delta_beta)
            self.beta += utils.radians(float(delta_beta), degrees=degrees)
            self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    def move_to(
        self,
        isocenter: Optional[geo.Point3D] = None,
        alpha: Optional[geo.Point3D] = None,
        beta: Optional[geo.Point3D] = None,
        degrees: bool = True,
    ) -> None:
        """Move to the specified point.

        Args:
            isocenter (Optional[geo.Point3D], optional): the desired isocenter in carm coordinates. Defaults to None.
            alpha (Optional[geo.Point3D], optional): the desired alpha angulation. Defaults to None.
            beta (Optional[geo.Point3D], optional): the desired secondary angulation. Defaults to None.
            degrees (bool, optional): whether angles are in degrees or radians. Defaults to False.
        """
        if isocenter is not None:
            self.isocenter = geo.point(isocenter)
            self.isocenter = geo.point(np.clip(self.isocenter, self.min_isocenter, self.max_isocenter))
        if alpha is not None:
            self.alpha = utils.radians(float(alpha), degrees=degrees)
            self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)
        if beta is not None:
            self.beta = utils.radians(float(beta), degrees=degrees)
            self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    @property
    def device_from_arm(self) -> geo.FrameTransform:
        # First, rotate points, then translate back by the isocenter.
        rot = Rotation.from_euler('xy', [self.alpha, self.beta]).as_matrix()
        return geo.FrameTransform.from_rt(rot, self.isocenter)

    @property
    def arm_from_device(self) -> geo.FrameTransform:
        """Transformation from the device frame (which doesn't move) to the arm frame (which rotates and translates with the arm, origin at the isocenter).
        """
        return self.device_from_arm.inv

    @property
    def camera3d_from_device(self) -> geo.FrameTransform:
        camera3d_from_arm = geo.FrameTransform.from_rt(translation=geo.point(
            0, -self.source_to_isocenter_horizontal_offset, self.source_to_isocenter_vertical_distance))
        return camera3d_from_arm @ self.arm_from_device

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        """Rigid transformation of the C-arm camera pose."""
        return self.camera3d_from_device @ self.device_from_world

    def get_camera3d_from_world(self) -> geo.FrameTransform:
        return self.camera3d_from_world

    def get_camera_projection(self) -> geo.CameraProjection:
        return geo.CameraProjection(self.camera_intrinsics, self.get_camera3d_from_world())

    @property
    def viewpoint(self) -> geo.Point3D:
        """Get the point along the principle ray, where objects of interest should ideally be placed.

        Returns:
            geo.Point3D: the viewpoint in the device frame.
        """
        return self.device_from_arm @ geo.point(0, self.source_to_isocenter_horizontal_offset, 0)

    @property
    def viewpoint_in_world(self) -> geo.Point3D:
        return self.world_from_device @ self.viewpoint

    @property
    def principle_ray(self) -> geo.Vector3D:
        """Unit vector along principle ray."""
        return self.device_from_arm @ geo.vector(0, 0, 1)

    @property
    def principle_ray_in_world(self) -> geo.Vector3D:
        return self.world_from_device @ self.principle_ray

    # shape parameters
    source_height = 200
    source_radius = 200
    detector_height = 100
    arm_width = 100
    def _make_mesh(
        self,
        full=True,
        include_labels: bool = False
    ) -> pv.PolyData:
        """Make the mesh of the C-arm, centered and upright.

        This DOES NOT use the current isocenter, alpha, or beta.

        Returns:
            pv.PolyData: [description]
        """
        assert pv_available, f'PyVista not available for obtaining MobileCArm surface model. Try: `pip install pyvista`'
        if include_labels:
            logger.warning(f'C-arm mesh labels not supported yet')

        source_point = geo.point(0, self.source_to_isocenter_horizontal_offset, -self.source_to_isocenter_horizontal_offset)
        center_point = geo.point(0, self.source_to_isocenter_horizontal_offset, 0)

        mesh = pv.Line(
            list(source_point),
            list(source_point + geo.vector(0, 0, self.source_to_detector_distance)),
        ) + pv.Line(
            list(center_point + geo.vector(-100, 0, 0)),
            list(center_point + geo.vector(100, 0, 0)),
        ) + pv.Line(
            list(center_point + geo.vector(0, -100, 0, 0)),
            list(center_point + geo.vector(0, 100, 0)),
        )

        if full:
            # Source
            mesh += pv.Cylinder(
                center=source_point,
                direction=[0, 0, 1],
                radius=self.source_radius,
                height=self.source_height,
            )

            # Sensor
            mesh += pv.Box(
                bounds=[
                    -self.pixel_size * self.sensor_width / 2,
                    self.pixel_size * self.sensor_width / 2,
                    -self.pixel_size * self.sensor_height / 2 + self.source_to_isocenter_horizontal_offset,
                    self.pixel_size * self.sensor_height / 2 + self.source_to_isocenter_horizontal_offset,
                    -self.source_to_isocenter_vertical_distance + self.source_to_detector_distance,
                    -self.source_to_isocenter_vertical_distance + self.source_to_detector_distance + self.detector_height,
                ],
            )

            # Arm
            arm = pv.ParametricTorus(
                ringradius=self.source_to_isocenter_vertical_distance,
                crosssectionradius=self.arm_width / 2,
            )
            arm.clip(normal='y', inplace=True)
            arm.rotate_y(90)
            mesh += arm

        return mesh

    def get_mesh_in_world(self, full=False) -> pv.PolyData:
        """Get the pyvista mesh for the C-arm, in its world-space orientation.

        Raises:
            RuntimeError: if pyvista is not available.

        Returns:
            pv.PolyData: a mesh somewhat resembling the C-arm device.
        """
        if self._static_mesh is None:
            self._static_mesh = self._make_mesh(full=full)

        mesh = self.static_mesh.copy()
        mesh.rotate_x(-np.degrees(self.alpha))
        mesh.rotate_y(-np.degrees(self.beta))
        mesh.translate(self.isocenter)
        mesh.transform(geo.get_data(self.world_from_carm))

        # TODO: add operating window.

        return mesh

    def jitter(self):
        # semi-realistic jitter about the axis.
        raise NotImplementedError
    

class CArm(object):
    """C-arm device for positioning a camera in space."""
    def __init__(
        self,
        isocenter_distance: float,
        isocenter: Optional[geo.Point3D] = None,
        phi: float = 0,
        theta: float = 0,
        rho: float = 0,
        degrees: bool = False,
    ) -> None:
        logger.warning('Previously, device.CArm used phi, theta as device angulation. This was incorrect. To accomplish something similar, use device.MobileCArm instead.')

        self.isocenter_distance = isocenter_distance
        self.isocenter = geo.point(0, 0, 0) if isocenter is None else isocenter
        self.phi, self.theta, self.rho = utils.radians(phi, theta, rho, degrees=degrees)

    def move_to(
        self, 
        isocenter: Optional[geo.Point3D] = None,
        phi: Optional[float] = None,
        theta: Optional[float] = None,
        rho: Optional[float] = None,
        degrees: bool = False,
    ) -> None:
        """Move the C-arm to the specified pose.

        Args:
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
        """
        if isocenter is not None:
            self.isocenter = geo.point(isocenter)
        if phi is not None:
            self.phi = utils.radians(float(phi), degrees=degrees)
        if theta is not None:
            self.theta = utils.radians(float(theta), degrees=degrees)
        if rho is not None:
            self.rho = utils.radians(float(rho), degrees=degrees)

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_phi: Optional[float] = None,
        delta_theta: Optional[float] = None,
        delta_rho: Optional[float] = None,
        degrees: bool = False,
        min_isocenter: Optional[geo.Point3D] = None,
        max_isocenter: Optional[geo.Point3D] = None,
        min_phi: Optional[float] = None,
        max_phi: Optional[float] = None,
        min_theta: Optional[float] = None,
        max_theta: Optional[float] = None,
    ) -> None:
        """Move the C-arm by the specified deltas.

        Clips the internal state by the provided values if not None.

        Args:
            delta_isocenter (Vector3D): offset for the isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
        """
        if delta_isocenter is not None:
            self.isocenter += geo.vector(delta_isocenter)
        if min_isocenter is not None or max_isocenter is not None:
            # TODO: check min_isocenter < max_isocenter
            self.isocenter = geo.point(np.clip(self.isocenter, min_isocenter, max_isocenter))

        if delta_phi is not None:
            assert np.isscalar(delta_phi)
            self.phi += utils.radians(float(delta_phi), degrees=degrees)
        if min_phi is not None or max_phi is not None:
            self.phi = np.clip(self.phi, min_phi, max_phi)

        if delta_theta is not None:
            assert np.isscalar(delta_theta)
            self.theta += utils.radians(float(delta_theta), degrees=degrees)
        if min_theta is not None or max_theta is not None:
            self.theta = np.clip(self.theta, min_theta, max_theta)

        if delta_rho is not None:
            self.rho += utils.radians(float(delta_rho), degrees=degrees)

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        return self.get_camera3d_from_world(self.isocenter, self.phi, self.theta, self.rho, degrees=False)

    def get_camera3d_from_world(
        self,
        isocenter: geo.Point3D,
        phi: float,
        theta: float,
        rho: Optional[float] = 0,
        degrees: bool = False,
    ) -> geo.FrameTransform:
        """Get the FrameTransform for the C-Arm device at the given pose.

        This ignores the internal state except for the isocenter_distance.
        
        Args:
            isocenter (geo.Point3D): isocenter of the device.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (Optional[float], optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
            offset (Optional[Vector3D], optional): world-space offset to add to the initial C-arm isocenter. Defaults to None.

        Returns:
            FrameTransform: the extrinsic matrix or "camera3d_from_world" frame transformation for the oriented C-Arm camera.
        """
        #  TODO: A staticmethod function may be more appropriate.
        phi, theta, rho = utils.radians(phi, theta, rho, degrees=degrees)

        # get the rotation corresponding to the c-arm, then translate to the camera-center frame, along z-axis.
        R = make_detector_rotation(phi, theta, rho)
        t = np.array([0, 0, self.isocenter_distance])
        camera3d_from_isocenter = geo.FrameTransform.from_rt(R, t)
        isocenter_from_world = geo.FrameTransform.from_origin(isocenter)

        return camera3d_from_isocenter @ isocenter_from_world
