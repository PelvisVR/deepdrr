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


class Scene(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_camera(self) -> Camera: ...

    @abstractmethod
    def get_render_primitives(self) -> List[RenderPrimitive]: ...

    @abstractmethod
    def instance_snapshot(self) -> List[RenderInstance]: ...


class GraphScene(Scene):

    def __init__(self, graph: TransformTree):
        self.graph = graph
        self._primitives = None
        self._camera = None
        self._render_primitives = None

    def get_camera(self) -> Camera:
        if self._camera is None:
            for node in self.graph:
                for content in node:
                    if isinstance(content, Camera):
                        self._camera = content
        return self._camera

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

    def get_render_primitives(self) -> List[RenderPrimitive]:
        if self._render_primitives is None:
            self._render_primitives = [
                primitive.to_render_primitive() for primitive in self.get_primitives()
            ]
        return self._render_primitives

    def instance_snapshot(self) -> List[RenderInstance]:
        self.get_primitives()
        instances = self._get_instances()
        return [instance.to_render_instance(self.get_prim_id) for instance in instances]

    def get_prim_id(self, primitive: Primitive):
        return self._prim_to_id[primitive]
