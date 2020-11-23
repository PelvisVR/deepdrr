"""Define the 3D geometry primitives that the rest of DeepDRR would use, in homogeneous coordinates.
"""

from __future__ import annotations

from typing import Union, Tuple, Optional, Type, List, Generic, TypeVar

from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
import numpy as np


from . import utils


def _to_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert an array to homogeneous points or vectors.

    Args:
        x (np.ndarray): array with objects on the last axis.
        is_point (bool, optional): if True, the array represents a point, otherwise it represents a vector. Defaults to True.

    Returns:
        np.ndarray: array containing the homogeneous point/vector(s).
    """
    if is_point:
        return np.concatenate([x, np.ones_like(x[..., -1:])], axis=-1)
    else:
        return np.concatenate([x, np.zeros_like(x[..., -1:])], axis=-1)


def _from_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert array containing homogeneous data to raw form.

    Args:
        x (np.ndarray): array containing homogenous
        is_point (bool, optional): whether the objects are points (true) or vectors (False). Defaults to True.

    Returns:
        np.ndarray: the raw data representing the point/vector(s).
    """
    if is_point:
        return (x / x[..., -1:])[..., :-1]
    else:
        assert np.all(np.isclose(x[..., :-1], 0)), 'not a homogeneous vector: {x}'
        return x[..., :-1]


T = TypeVar('T')


class HomogeneousObject(ABC):
    """Any of the objects that rely on homogeneous transforms, all of which wrap a single array called `data`."""

    dtype = np.float32

    def __init__(
            self,
            data: np.ndarray,
    ) -> None:
        self.data = np.array(data, dtype=self.dtype)

    @classmethod
    @abstractmethod
    def from_array(
            cls: Type[T],
            x: np.ndarray,
    ) -> T:
        """Create a homogeneous object from its non-homogeous representation as an array."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Get the dimension of the space the object lives in."""
        pass

    @abstractmethod
    def to_array(self):
        """Get the non-homogeneous representation of the object."""
        pass

    def __array__(self):
        return self.to_array()
            
    def __str__(self):
        return np.array_str(np.array(self), suppress_small=True)

    def __repr__(self):
        s = '  ' + str(np.array_str(self.data)).replace('\n', '\n  ')
        return f"{self.__class__.__name__}(\n{s}\n)"

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)




class Homogeneous(HomogeneousObject):
    """A Homogeneous point or vector in any dimension."""
    def __init__(
            self,
            data: np.ndarray,
    ) -> None:
        """Instantiate the homogeneous point or vector and check its dimension."""
        super().__init__(data)
        
        if self.data.shape != (self.dim + 1,):
            raise ValueError(f'invalid shape for {self.dim}D object in homogeneous coordinates: {self.data.shape}')

    def to_array(self):
        return _from_homogeneous(self.data, vector=(self.data[-1] == 0))

    
class Point(Homogeneous):
    def __init__(self, data: np.ndarray) -> None:
        assert data[-1] == 1
        super().__init__(data)
        
    @classmethod
    def from_array(
            cls: Type[T],
            x: np.ndarray,
    ) -> T:
        x = np.array(x, dtype=cls.dtype)
        data = _to_homogeneous(x, is_point=True)
        return cls(data)

    @classmethod
    def from_any(
            cls: Type[T],
            other: Union[np.ndarray, Point],
    ):
        """ If other is not a point, make it one. """
        return other if issubclass(type(other), Point) else cls.from_array(other)

    def __sub__(
            self: Point,
            other: Point,
    ) -> Union[Vector2D, Vector3D]:
        """ Subtract two points, obtaining a vector. """
        other = self.from_any(other)
        return vector(self.data - other.data)

    def __add__(self, other):
        """ Can add a vector to a point, but cannot add two points. """
        if issubclass(type(other), Vector):
            return type(self)(other.data + self.data)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other


class Vector(Homogeneous):
    def __init__(self, data: np.ndarray) -> None:
        assert data[-1] == 0
        super().__init__(data)
        
    @classmethod
    def from_array(
            cls: Type[T],
            v: np.ndarray,
    ) -> T:
        v = np.array(v, dtype=cls.dtype)
        data = _to_homogeneous(v, is_point=False)
        return cls(data)

    @classmethod
    def from_any(
            cls: Type[T],
            other: Union[np.ndarray, Vector],
    ):
        """ If other is not a Vector, make it one. """
        return other if issubclass(type(other), Vector) else cls.from_array(other)
    
    def __mul__(self, other: Union[int, float]):
        """ Vectors can be multiplied by scalars. """
        if isinstance(other, (int, float)):
            return type(self)(other * self.data)
        else:
            return NotImplemented

    def __matmul__(self, other: Vector):
        """ Inner product between two Vectors. """
        other = self.from_any(other)
        return type(self)(self.data @ other.data)

    def __add__(self, other: Vector) -> Vector:
        """ Two vectors can be added to make another vector. """
        other = self.from_any(other)
        return type(self)(self.data + other.data)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other: Vector):
        return self + (-other)

    def __rmul__(self, other: Vector):
        return self * other

    def __rsub__(self, other: Vector):
        return (-self) + other

    def __radd__(self, other: Vector):
        return self + other


class Homogeneous2D(Homogeneous):
    dim = 2


class Homogeneous3D(Homogeneous):
    dim = 3


class Point2D(Point, Homogeneous2D):
    """ Homogeneous point in 2D, represented as an array with [x, y, 1] """


class Vector2D(Vector, Homogeneous2D):
    """ Homogeneous vector in 2D, represented as an array with [x, y, 0] """
    

class Point3D(Point, Homogeneous3D):
    """ Homogeneous point in 3D, represented as an array with [x, y, z, 1] """


class Vector3D(Vector, Homogeneous3D):
    """ Homogeneous vector in 3D, represented as an array with [x, y, z, 0] """


PointOrVector = TypeVar('PointOrVector', Point2D, Point3D, Vector2D, Vector3D)
PointOrVector2D = TypeVar('PointOrVector2D', Point2D, Vector2D)
PointOrVector3D = TypeVar('PointOrVector3D', Point3D, Vector3D)


def _array(x: Union[List[np.ndarray], List[float]]) -> np.ndarray:
    """Parse args into a numpy array."""
    if len(x) == 1:
        return np.array(x[0])
    elif len(x) == 2 or len(x) == 3:
        return np.array(x)
    else:
        raise ValueError(f'could not parse args: {x}')


def point(*x: Union[np.ndarray, float, Point2D, Point3D]) -> Union[Point2D, Point3D]:
    if len(x) == 1 and isinstance(x[0], (Point2D, Point3D)):
        return x[0]

    x = _array(x)
    if x.shape == (2,):
        return Point2D.from_array(x)
    elif x.shape == (3,):
        return Point3D.from_array(x)
    else:
        raise ValueError(f'invalid data for point: {x}')
    

def vector(*v: Union[np.ndarray, float, Vector2D, Vector3D]) -> Union[Vector2D, Vector3D]:
    if len(v) == 1 and isinstance(v[0], (Vector2D, Vector3D)):
        return v[0]

    v = _array(v)
    if v.shape == (2,):
        return Vector2D.from_array(v)
    elif v.shape == (3,):
        return Vector3D.from_array(v)
    else:
        raise ValueError(f'invalid data for vector: {v}')


class Transform(HomogeneousObject):
    def to_array(self):
        return self.data

    @classmethod
    def from_array(cls, data: np.ndarray) -> FrameTransform:
        return cls(data)

    def __matmul__(
            self,
            other: Union[FrameTransform, PointOrVector],
    ) -> Union[FrameTransform, PointOrVector]:  # TODO: output type will match input type
        assert other.dim == self.dim, 'dimensions must match between other ({other.dim}) and self ({self.dim})'
        return type(other)(self.data @ other.data)

    def __call__(
            self,
            other: PointOrVector,
    ) -> PointOrVector:
        return self @ other
    
    @property
    def inv(self):
        raise NotImplementedError("inverse operation not necessarily well-defined for all transforms")


class FrameTransform(Transform):
    """Defines a rigid (affine) transformation from one frame to another.

    So that, for a point `x` in world-coordinates `F(x)` (or `F @ x`) is the same point in `F`'s
    coordinates. Note that if `x` is a numpy array, it is assumed to be a point.

    FrameTransforms can also be composed using `@`. If frame 1 `F_W1` is a frame transform from world to frame
    1, `F_12` is a frame transform from frame 1 to frame 2, `F_W1` to and y is a point in frame 2 coordinates, then
    ```
    F_W1 @ F_12 @ y
    ```
    is the point's representation in world coordinates. Similarly, if x is a point in world coordinates:
    ```
    F_12.inv @ F_W1.inv @ x
    ```
    is the point's representation in frame 2.

    In order to maximize readability, the suggested naming convention for frames should be as above. 
    As an example, if there is a volume with an index frame (indices into the volume), an anatomical frame (e.g. LPS), 
    both of which are situated somewhere in world-space, `F_world_lps` should be the LPS frame, `F_lps_ijk` 
    should be the index frame in the LPS system. In this setup, then, given an index-space point [i,j,k], the corresponding world-space representation is
    `[x,y,z] = F_world_lps @ F_lps_ijk @ [i,j,k]`. 
    
    In this setup, an inverse simply flips the two subscripted frames, so one would denote `F_lps_world = F_world_lps.inv`. 
    Thus, if `[x,y,z]` is a world-space representation of a point, `F_lps_ijk.inv @ F_world_lps.inv @ [x,y,z]` 
    is the point's representation in index-space.

    The idea here is that the frame being transformed to comes first, so that (if no inverses are present) one can glance
    at the outermost frame to see what frame the point is in. This also allows one to easily verify that frametransforms rightly 
    go next to one another by checking whether the inner frames match.

    The "F2_to_F1" convention for naming frames is confusing and should be avoided. Instead, this would be `F_F1_F2` (hence the confusion).

    Alternatively, use the convention `F1_from_F2`. This maintains the handy ordering properties as above but is a little more readable, as the "from" 
    separates frame names.

    Note that a FrameTransform is dimension-independent, but its dimension must match the objects it transforms.

    """
    def __init__(
            self,
            data: np.ndarray,    # the homogeneous frame transformation matrix
    ) -> None:
        super().__init__(data)
        
        assert np.all(self.data[-1, :-1] == 0) and self.data[-1, -1] == 1, f'not a rigid transformation:\n{self.data}'

    @property
    def dim(self):
        return self.data.shape[0] - 1
    
    @classmethod
    def from_rt(
        cls,
        rotation: Union[Rotation, np.ndarray],
        translation: Union[Point3D, np.ndarray],
    ) -> FrameTransform:
        R = rotation.as_matrix() if isinstance(rotation, Rotation) else rotation
        t = np.array(translation)

        dim = t.shape[0]
        assert R.shape == (dim, dim)

        data = np.concatenate(
            [
                np.concatenate([R, t[:, np.newaxis]], axis=1),
                np.concatenate([np.zeros((1, dim)), [[1]]], axis=1)
            ],
            axis=0
        )

        return cls(data)

    @classmethod
    def from_scaling(
            cls: Type[FrameTransform],
            scaling: Union[int, float, np.ndarray],
            translation: Optional[np.ndarray] = None,
    ) -> FrameTransform:
        """Create a frame based on scaling dimensions. Assumes dim = 3.

        Args:
            cls (Type[FrameTransform]): the class.
            scaling (Union[int, float, np.ndarray]): coefficient to scale by, or one for each dimension.

        Returns:
            FrameTransform: 
        """
        scaling = np.array(scaling) * np.ones(3)
        translation = np.zeros(3) if translation is None else translation
        return cls.from_matrices(np.diag(scaling), translation)

    @classmethod
    def from_translation(
        cls,
        t: np.ndarray,
    ) -> FrameTransform:
        return cls.from_matrices(np.eye(t.shape[0]), t)

    @classmethod
    def identity(
            cls: Type[FrameTransform],
            dim: int = 3
    ):
        return cls.from_matrices(np.identity(dim), np.zeros(dim))

    @property
    def R(self):
        return self.data[0:3, 0:3]

    @property
    def t(self):
        return self.data[0:3, 3]

    @property
    def inv(self):
        return FrameTransform.from_matrices(self.R.T, -(self.R.T @ self.t))



class CameraIntrinsicTransform(FrameTransform):
    dim = 2
    """The camera intrinsic matrix, which is fundamentally a FrameTransform in 2D, namely `index_from_camera`.

    The intrinsic matrix transfroms to the index-space of the image (as mapped on the sensor) from the 

    Note that focal lengths are often measured in world units (e.g. millimeters.) The conversion can be taken
    from the size of a pixel.
    
    References:
    - Szeliski's "Computer Vision."
    - https://ksimek.github.io/2013/08/13/intrinsic/
    
    """

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert self.data.shape == (3,3), 'unrecognized shape'

    @classmethod
    def from_parameters(
        cls,
        optical_center: Point2D,
        focal_length: Union[float, Tuple[float, float]] = 1,
        shear: float = 0,
        aspect_ratio: Optional[float] = None,
    ) -> CameraIntrinsicTransform:
        """Creates a camera intrinsic matrix.

        Args:
            optical_center (Point2D): the index-space point where the isocenter (or pinhole) is centered.
            focal_length (Union[float, Tuple[float, float]]): the focal length in index units. Can be a tubple (f_x, f_y), 
                or a scalar used for both, or a scalar modified by aspect_ratio, in index units.
            shear (float): the shear `s` of the camera.
            aspect_ratio (Optional[float], optional): the aspect ratio `a` (for use with one focal length). If not provided, aspect 
                ratio is 1. Defaults to None.

        Returns:
            CameraIntrinsicTransform: The camera intrinsic matrix as
                [[f_x, s, c_x],
                 [0, f_y, c_y],
                 [0,   0,   1]]
                or
                [[f, s, c_x],
                 [0, a f, c_y],
                 [0,   0,   1]]

        """
        optical_center = point(optical_center)
        assert optical_center.dim == 2, 'center point not in 2D'

        cx, cy = np.array(optical_center)

        if aspect_ratio is None:
            fx, fy = utils.tuplify(focal_length, 2)
        else:
            assert isinstance(focal_length, (float, int)), 'cannot use aspect ratio if both focal lengths provided'
            fx, fy = (focal_length, aspect_ratio * focal_length)
        
        data = np.ndarray(
            [[fx, shear, cx],
             [0, fy, cy],
             [0, 0, 1]], 
            np.float32)

        return cls(data)

    @classmethod
    def from_sizes(
        cls, 
        sensor_size: Union[int, Tuple[int, int]],
        pixel_size: Union[float, Tuple[float, float]],
        source_to_detector_distance: int,
    ) -> CameraIntrinsicTransform:
        """Generate the camera from human-readable parameters.

        This is the recommended way to create the camera.

        Args:
            sensor_size (Union[float, Tuple[float, float]]): (width, height) of the sensor, or a single value for both, in pixels.
            pixel_size (Union[float, Tuple[float, float]]): (width, height) of a pixel, or a single value for both, in world units (e.g. mm).
            source_to_detector_distance (int): distance from source to detector in world units.

        Returns:
            
        """
        sensor_size = utils.tuplify(sensor_size, 2)
        pixel_size = utils.tuplify(pixel_size, 2)
        fx = source_to_detector_distance / pixel_size[0]
        fy = source_to_detector_distance / pixel_size[1]
        optical_center = point(sensor_size[0] / 2, sensor_size[1] / 2)
        return cls.from_parameters(
            optical_center=optical_center,
            focal_length=(fx, fy))


class CameraProjection(HomogeneousObject):
    def __init__(
        self,
        intrinsic: Union[CameraIntrinsicTransform, np.ndarray],
        extrinsic: Union[FrameTransform, np.ndarray],
    ) -> None:
        """Create a camera in 3D space.

        Args:
            intrinsic (CameraIntrinsicTransform): the camera intrinsic matrix, or a mapping to 2D image index coordinates 
                from camera coordinates, i.e. index_from_camera2d.
            extrinsic (FrameTransform): the camera extrinsic matrix, or simply a FrameTransform to camera coordinates
                 from world coordinates, i.e. camera3d_from_world.

        """
        self.intrinsic = intrinsic if isinstance(intrinsic, CameraIntrinsicTransform) else CameraIntrinsicTransform(intrinsic)
        self.extrinsic = extrinsic if isinstance(extrinsic, FrameTransform) else FrameTransform(extrinsic)
        
        index_from_camera2d = self.intrinsic
        camera2d_from_camera3d = Transform(np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1))
        camera3d_from_world = self.extrinsic

        self.index_from_world = index_from_camera2d @ camera2d_from_camera3d @ camera3d_from_world

        
