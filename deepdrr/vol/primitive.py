from __future__ import annotations
from typing import Any, Union, Tuple, List, Optional, Dict, Type

import logging
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread
import nrrd
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

from .. import load_dicom
from .. import geo
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients
from .renderable import Renderable

vtk, nps, vtk_available = utils.try_import_vtk()


log = logging.getLogger(__name__)


class Primitive(object):
    data: pv.PolyData
    morph_targets: List[np.ndarray]
    material: str
    density: float

    def __init__(
        self,
        material: str,
        density: float,
        mesh: pv.PolyData,
        morph_targets: Optional[np.ndarray] = None,
        subtractive: bool = False,
    ) -> None:
        self.data = mesh
        self.morph_targets = morph_targets if morph_targets is not None else []
        for mt in self.morph_targets:
            assert mt.shape[0] == self.data.n_points
        self.material = material
        self.density = density
        self.subtractive = subtractive

    def compute_vertices(self):
        """Compute the vertices of the mesh in local coordinates, including the morph targets."""
        if len(self.morph_targets) == 0:
            return np.array(self.data.points, dtype=np.float32)
        else:
            return np.array(self.data.points + self.morph_targets * self.morph_weights, dtype=np.float32)
    
    def triangles(self, flip_winding_order=True):
        if flip_winding_order: # TODO bad
            return self.data.faces.reshape((-1, 4))[..., 1:][..., [0, 2, 1]].astype(np.int32)
        else:
            return self.data.faces.reshape((-1, 4))[..., 1:][..., [0, 1, 2]].astype(np.int32)   # flip winding order
