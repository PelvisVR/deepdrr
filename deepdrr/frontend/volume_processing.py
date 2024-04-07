from __future__ import annotations
from pathlib import Path

import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
import nrrd
import numpy as np
import h5py
from .. import load_dicom
import logging

from deepdrr import geo
import nibabel as nib


log = logging.getLogger(__name__)

def _convert_hounsfield_to_density(hu_values: np.ndarray, smooth_air: bool = False):
    # Use two linear interpolations from data: (HU,g/cm^3)
    # use for lower HU: density = 0.001029*HU + 1.03
    # use for upper HU: density = 0.0005886*HU + 1.03

    # set air densities
    if smooth_air:
        hu_values[hu_values <= -900] = -1000
    # hu_values[hu_values > 600] = 5000;
    densities = np.maximum(
        np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0
    )
    return densities

def segment_materials(
    hu_values: np.ndarray,
    use_thresholding: bool = True,
) -> Dict[str, np.ndarray]:
    """Segment the materials in a volume, potentially caching.

    If cache_dir is None, then

    Args:
        hu_values (np.ndarray): volume data in Hounsfield Units.
        use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.
        use_cached (bool, optional): use the cached segmentation, if it exists. Defaults to True.
        save_cache (bool, optional): save the segmentation to cache_dir. Defaults to True.
        cache_dir (Optional[Path], optional): where to look for the segmentation cache. If None, no caching performed. Defaults to None.
        cache_name (str, optional): Name of cache file. Must be provided if use_cached or cache_dir is True. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: materials segmentation.
    """
    materials = _segment_materials(
        hu_values, use_thresholding=use_thresholding
    )
    return materials


def _segment_materials(
    hu_values: np.ndarray,
    use_thresholding: bool = True,
) -> Dict[str, np.ndarray]:
    """Segment the materials.

    Meant for internal use, particularly to be overriden by volumes with different materials.

    Args:
        hu_values (np.ndarray): volume data in Hounsfield Units.
        use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.

    Returns:
        Dict[str, np.ndarray]: materials segmentation.
    """
    if use_thresholding:
        return load_dicom.conv_hu_to_materials_thresholding(hu_values)
    else:
        return load_dicom.conv_hu_to_materials(hu_values)


def parse_nrrd_header(header) -> geo.FrameTransform:
    anatomical_from_IJK = np.concatenate(
        [
            header["space directions"],
            header["space origin"].reshape(-1, 1),
        ],
        axis=1,
    )
    anatomical_from_IJK = np.concatenate(
        [anatomical_from_IJK, [[0, 0, 0, 1]]], axis=0
    ).astype(np.float32)

    anatomical_coordinate_system = {
        "right-anterior-superior": "RAS",
        "left-posterior-superior": "LPS",
    }.get(header.get("space", "right-anterior-superior"))

    return anatomical_from_IJK, anatomical_coordinate_system


def load_nrrd(
    nrrd_path: str,
    use_thresholding: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, str]:

    nrrd_path = Path(nrrd_path)

    hu_values, header = nrrd.read(nrrd_path)

    anatomical_from_IJK, anatomical_coordinate_system = parse_nrrd_header(header)

    data = _convert_hounsfield_to_density(hu_values)
    materials = segment_materials(
        hu_values,
        use_thresholding=use_thresholding,
    )

    return data, materials, anatomical_from_IJK, anatomical_coordinate_system


def h5_from_nrrd(
    nrrd_path: str,
    h5_path: str,
    use_thresholding: bool = True,
):
    data, materials, anatomical_from_IJK, anatomical_coordinate_system = load_nrrd(
        nrrd_path
    )

    write_h5_file(h5_path, data, materials, anatomical_from_IJK, anatomical_coordinate_system)


def write_h5_file(
    h5_path: str,
    data: np.ndarray,
    materials: Dict[str, np.ndarray],
    anatomical_from_IJK: np.ndarray,
    anatomical_coordinate_system: str,
):
    h5_path = str(h5_path)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=data, compression="lzf")
        seg_grp = f.create_group("segmentation")
        for material, segmentation in materials.items():
            seg_grp.create_dataset(material, data=segmentation, compression="lzf")

        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("anatomical_from_IJK", data=anatomical_from_IJK)
        meta_grp.create_dataset("anatomical_coordinate_system", data=anatomical_coordinate_system)


def load_h5(h5_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, str]:
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        data = f["data"][()]
        materials = {k: v[()] for k, v in f["segmentation"].items()}
        anatomical_from_IJK = f["meta"]["anatomical_from_IJK"][()]
        anatomical_coordinate_system = f["meta"]["anatomical_coordinate_system"][()].decode("utf-8")
    return data, materials, anatomical_from_IJK, anatomical_coordinate_system


def h5_from_nifti(
        nifti_path: str,
        h5_path: str,
        use_thresholding: bool = True,
    ):

    nifti_path = Path(nifti_path)
    h5_path = Path(h5_path)

    log.info(f"loading NiFti volume from {nifti_path}")
    img = nib.load(nifti_path)
    if img.header.get_xyzt_units()[0] not in ["mm", "unknown"]:
        log.warning(
            f'got NifTi xyz units: {img.header.get_xyzt_units()[0]}. (Expected "mm").'
        )
    
    anatomical_from_IJK = geo.FrameTransform(img.affine)

    hu_values = img.get_fdata()
    data = _convert_hounsfield_to_density(hu_values)
    if materials is None:
        materials = segment_materials(
            hu_values,
            anatomical_from_IJK,
            use_thresholding=use_thresholding,
        )

    write_h5_file(h5_path, data, materials, anatomical_from_IJK, "RAS")
