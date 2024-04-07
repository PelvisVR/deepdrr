from __future__ import annotations

from deepdrr import geo
from typing import List, Optional, Any, Set
from abc import ABC, abstractmethod
from .. import utils


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

    def __init__(self, tag: Optional[str] = None):
        self._enabled = True
        self.tag = tag

    @abstractmethod
    def to_render_primitive(self) -> RenderPrimitive: ...

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value


class PrimitiveMesh(Primitive):
    material_name: str
    density: float
    priority: int
    subtractive: bool

    def __init__(
        self,
        material_name: str = "iron",
        density: float = 1,
        priority: int = 0,
        subtractive: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.material_name = material_name
        self.density = density
        self.priority = priority
        self.subtractive = subtractive

    @abstractmethod
    def to_render_primitive(self) -> RenderPrimitive: ...


class StlMesh(PrimitiveMesh):
    path: str

    def __init__(
        self,
        path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path

    def to_render_primitive(self) -> RenderPrimitive:
        return RenderPrimitive(
            data=RenderStlMesh(
                url=Url("file://" + str(Path(self.path).resolve())),
                material_name=self.material_name,
                density=self.density,
                priority=self.priority,
                subtractive=self.subtractive,
            )
        )


class PrimitiveVolume(Primitive):
    priority: int

    def __init__(self, priority: int = 0):
        super().__init__()
        self.priority = priority

    @abstractmethod
    def to_render_primitive(self) -> RenderPrimitive: ...

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

    def to_render_primitive(self) -> RenderPrimitive:
        return RenderPrimitive(
            data=RenderH5Volume(
                url=Url("file://" + str(Path(self.path).resolve())),
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
        serialize_path: Optional[str] = None,
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

    def to_render_primitive(self) -> RenderPrimitive:
        saved_path = self.save_h5(self.serialize_path)

        return H5Volume(
            path=saved_path,
            priority=self.priority,
        ).to_render_primitive()

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
            serialize_path=None,  # use cache folder
        )

    def to_memory_volume(self) -> MemoryVolume:
        return self
