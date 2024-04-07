
from __future__ import annotations
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

from deepdrr import geo

CACHE_DIR_ENV_VAR = "DEEPDRR_CACHE_DIR"

def deepdrr_cache_dir() -> Path:
    if os.environ.get(CACHE_DIR_ENV_VAR) is not None:
        root = Path(os.environ.get(CACHE_DIR_ENV_VAR)).expanduser()
    else:
        root = Path.home() / ".cache" / "deepdrr"

    if not root.exists():
        root.mkdir(parents=True)

    return root

def save_or_cache_file(path: str, func: Callable[[str], None]) -> str:
    with tempfile.NamedTemporaryFile() as f:
        func(f.name)
        
        if path is not None:
            shutil.remove(path)
            shutil.copy(f.name, path)
        else:
            with open(f.name, "rb") as f:
                sha1 = hashlib.sha1()
                sha1.update(f.read())
                dig = sha1.hexdigest()
            path = deepdrr_cache_dir() / dig
            shutil.copy(f.name, path)
    
    return path