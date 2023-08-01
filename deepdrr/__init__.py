try:
    import cupy
except ImportError:
    raise ImportError("""CuPy must be installed to use DeepDRR.
Please install the version of CuPy for your CUDA Toolkit version by following the instructions here: https://cupy.dev/
Or by installing deepdrr with the optional CuPy extra for your CUDA Toolkit version:
e.g. pip install deepdrr[cupy11x] for CUDA v11.2 ~ 11.8
""")

from . import vis, geo, projector, device, annotations, utils
from .projector import Projector
from .vol import Volume, Mesh
from .device import CArm, MobileCArm
from .annotations import LineAnnotation
from .logging import setup_log


__all__ = [
    "MobileCArm",
    "CArm",
    "Volume",
    "Mesh",
    "Projector",
    "vis",
    "geo",
    "projector",
    "device",
    "annotations",
    "utils",
    "setup_log",
]
