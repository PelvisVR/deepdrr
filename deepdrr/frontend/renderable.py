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
from .. import utils

from .primitive import *


class Renderable(TransformDriver):
    pass


class Mesh(Renderable):
    def __init__(
        self,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        mesh: PrimitiveMesh = None,
        enabled: bool = True,
    ):
        self._mesh = mesh
        self._instance = PrimitiveInstance(primitive=mesh)
        self._node_anatomical = TransformNode(
            transform=world_from_anatomical, contents=[self._instance]
        )
        self.enabled = enabled

    @classmethod
    def from_stl(
        cls,
        path: str,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        enabled: bool = True,
        **kwargs,
    ):
        return cls(
            mesh=StlMesh(path=path, **kwargs),
            world_from_anatomical=world_from_anatomical,
            enabled=enabled,
        )

    def _add_as_child_of(self, node: TransformNode):
        node.add(self._node_anatomical)

    def add(self, node: TransformNode):
        self._node_anatomical.add(node)

    @property
    def base_node(self) -> TransformNode:
        return self._node_anatomical

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
        self._instance = PrimitiveInstance(primitive=volume)
        self._node_anatomical = TransformNode(
            transform=world_from_anatomical, contents=[self._instance]
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

    def _add_as_child_of(self, node: TransformNode):
        node.add(self._node_anatomical)

    def add(self, node: TransformNode):
        self._node_anatomical.add(node)

    @property
    def base_node(self) -> TransformNode:
        return self._node_anatomical

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
