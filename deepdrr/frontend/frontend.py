from __future__ import annotations

from deepdrr import geo
from typing import List, Optional, Any, Set
from abc import ABC, abstractmethod


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
from .cache import *

from deepdrr import geo

from deepdrr.serial.render import *
from .transform_manager import *


class Primitive:

    def __init__(self):
        self._enabled = True

    @abstractmethod
    def get_render_primitive(self) -> RenderPrimitive: ...

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value


class PrimitiveInstance:

    def __init__(
        self,
        primitive: Primitive,
        transform_node: TransformNode,
        morph_weights: List[float],
    ):
        self.primitive = primitive
        self.transform_node = transform_node
        self.morph_weights = morph_weights


class Camera:
    pass


class PrimitiveMesh(Primitive):
    material_name: str
    density: float
    priority: int
    additive: bool
    subtractive: bool

    def __init__(
        self,
        material_name: str,
        density: float,
        priority: int,
        additive: bool,
        subtractive: bool,
    ):
        super().__init__()
        self.material_name = material_name
        self.density = density
        self.priority = priority
        self.additive = additive
        self.subtractive = subtractive

    @abstractmethod
    def get_render_primitive(self) -> RenderPrimitive: ...


class StlMesh(PrimitiveMesh):
    path: str

    def __init__(
        self,
        path: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path = path

    def get_render_primitive(self) -> RenderPrimitive:
        return RenderPrimitive(
            data=RenderStlMesh(
                url=Url("file://" + Path(self.path).resolve()),
                material_name=self.material_name,
                density=self.density,
                priority=self.priority,
                additive=self.additive,
                subtractive=self.subtractive,
            )
        )


class PrimitiveVolume(Primitive):
    priority: int

    def __init__(self, priority: int = 0):
        super().__init__()
        self.priority = priority

    @abstractmethod
    def get_render_primitive(self) -> RenderPrimitive: ...

    @abstractmethod
    def save_h5(
        self,
        path: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def to_memory_volume(self) -> MemoryVolume: ...


class H5Volume(PrimitiveVolume):
    path: str

    def __init__(self, path: str, priority: int = 0):
        super().__init__(priority)
        self.path = path

    def get_render_primitive(self) -> RenderPrimitive:
        return RenderPrimitive(
            data=RenderH5Volume(
                url=Url("file://" + Path(self.path).resolve()),
                priority=self.priority,
            )
        )

    def save_h5(
        self,
        path: Optional[str] = None,
    ) -> str:
        if path is None:
            return self.path
        shutil.copy(self.path, path)
        return path

    def to_memory_volume(self) -> MemoryVolume:
        return MemoryVolume.from_h5(self.path)


class MemoryVolume(PrimitiveVolume):
    data: np.ndarray
    materials: Dict[str, np.ndarray]
    anatomical_from_IJK: geo.FrameTransform
    anatomical_coordinate_system: Optional[str]
    serialize_path: Optional[str]

    def __init__(
        self,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        anatomical_from_IJK: geo.FrameTransform,
        anatomical_coordinate_system: Optional[str],
        serialize_path: Optional[str],
        priority: int = 0,
    ):
        super().__init__(priority)
        self.data = data
        self.materials = materials
        self.anatomical_coordinate_system = anatomical_coordinate_system
        self.serialize_path = serialize_path
        self._anatomical_from_IJK = anatomical_from_IJK

    @property
    def anatomical_from_IJK(self) -> geo.FrameTransform:
        if self._anatomical_from_IJK is None:
            self._anatomical_from_IJK = geo.FrameTransform.identity()
        return self._anatomical_from_IJK

    @anatomical_from_IJK.setter
    def anatomical_from_IJK(self, value: geo.FrameTransform):
        self._anatomical_from_IJK = value

    def get_render_primitive(self) -> RenderPrimitive:
        saved_path = self.save_h5(self.serialize_path)

        return H5Volume(
            path=saved_path,
            priority=self.priority,
        ).get_render_primitive()

    def save_h5(
        self,
        path: Optional[str] = None,
    ) -> str:
        def save_h5(p):
            write_h5_file(
                p,
                self.data,
                self.materials,
                self.anatomical_from_IJK,
                self.anatomical_coordinate_system,
            )

        saved_path = save_or_cache_file(path, save_h5)

        if self.serialize_path is None:
            log.info(f"Saved H5 file to {saved_path}")

        return saved_path

    @classmethod
    def from_h5(cls, path: str):
        data, materials, anatomical_from_IJK, anatomical_coordinate_system = load_h5(
            path
        )

        return cls(
            data=data,
            materials=materials,
            anatomical_from_IJK=anatomical_from_IJK,
            anatomical_coordinate_system=anatomical_coordinate_system,
            serialize_path=path,
        )

    @classmethod
    def from_nrrd(cls, path: str):
        data, materials, anatomical_from_IJK, anatomical_coordinate_system = load_nrrd(
            path
        )

        return cls(
            data=data,
            materials=materials,
            anatomical_from_IJK=anatomical_from_IJK,
            anatomical_coordinate_system=anatomical_coordinate_system,
            serialize_path=path,
        )

    def to_memory_volume(self) -> MemoryVolume:
        return self


class StlMesh(Primitive):
    url: Url
    material_name: str
    density: float
    priority: int
    addtive: bool
    subtractive: bool


# class InstanceData:
#     primitive_id: int
#     transform: geo.FrameTransform
#     morph_weights: List[float]


class Scene(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_camera(self) -> Camera: ...

    @abstractmethod
    def get_primitives(self) -> List[Primitive]: ...

    @abstractmethod
    def instance_snapshot(self) -> List[RenderInstance]: ...


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
                    raise ValueError(
                        "Adding new primitives to the graph is not allowed after scene construction"
                    )

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
            render_instance_list.append(
                RenderInstance(
                    primitive_id=self._prim_to_id[instance.primitive],
                    transform=geo_to_render_matrix4x4(
                        self.graph.get_transform(None, instance.transform_node)
                    ),
                    morph_weights=instance.morph_weights,
                )
            )
        return render_instance_list


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

    @abstractmethod
    def _add_to(self, node: TransformNode): ...

    @abstractmethod
    def add(self, node: TransformNode): ...

class Mesh(Renderable):
    def __init__(
        self,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        mesh: PrimitiveMesh = None,
        enabled: bool = True,
    ):
        self._mesh = mesh
        self._node_anatomical = TransformNode(
            transform=world_from_anatomical, contents=[mesh]
        )
        self.enabled = enabled

    @classmethod
    def from_stl(cls, path: str, **kwargs):
        return cls(
            mesh=StlMesh(path=path, **kwargs),
            **kwargs,
        )

    def _add_to(self, node: TransformNode):
        node.add(self._node_anatomical)

    def add(self, node: TransformNode):
        self._node_anatomical.add(node)

    @property
    def enabled(self) -> bool:
        return self.mesh.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.mesh.enabled = value

    @property
    def world_from_anatomical(self) -> geo.FrameTransform:
        return self._node_anatomical.transform

    @world_from_anatomical.setter
    def world_from_anatomical(self, value: geo.FrameTransform):
        self._node_anatomical.transform = value

    @property
    def mesh(self) -> PrimitiveMesh:
        return self._mesh


class Volume(Renderable):
    def __init__(
        self,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        volume: PrimitiveVolume = None,
        enabled: bool = True,
    ):
        self._volume = volume
        self._node_anatomical = TransformNode(
            transform=world_from_anatomical, contents=[volume]
        )
        self.enabled = enabled

    @classmethod
    def from_h5(cls, path: str, in_memory=False, **kwargs):
        volume = H5Volume(path)
        if in_memory:
            volume = volume.to_memory_volume()
        return cls(
            volume=volume,
            **kwargs,
        )

    @classmethod
    def from_nrrd(cls, path: str, **kwargs):
        return cls(
            volume=MemoryVolume.from_nrrd(path),
            **kwargs,
        )

    def _add_to(self, node: TransformNode):
        node.add(self._node_anatomical)

    def add(self, node: TransformNode):
        self._node_anatomical.add(node)

    @property
    def enabled(self) -> bool:
        return self.volume.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self.volume.enabled = value

    @property
    def world_from_anatomical(self) -> geo.FrameTransform:
        return self._node_anatomical.transform

    @world_from_anatomical.setter
    def world_from_anatomical(self, value: geo.FrameTransform):
        self._node_anatomical.transform = value

    @property
    def volume(self) -> PrimitiveVolume:
        return self._volume

    @property
    def memory_volume(self) -> MemoryVolume:
        if isinstance(self._volume, MemoryVolume):
            return self._volume
        self._volume = self._volume.to_memory_volume()
        return self._volume

    def to_memory_volume(self):
        self._volume = self._volume.to_memory_volume()

    @property
    def anatomical_from_IJK(self) -> geo.FrameTransform:
        return self.memory_volume.anatomical_from_IJK

    @anatomical_from_IJK.setter
    def anatomical_from_IJK(self, value: geo.FrameTransform):
        self.memory_volume.anatomical_from_IJK = value

    def save_h5(self, path: str):
        self._volume.save_h5(path, self.anatomical_from_IJK)


class RenderProfile(ABC):
    pass


class DRRRenderProfile(RenderProfile):
    pass


class RasterizeRenderProfile(RenderProfile):
    pass
