import numpy as np
from pydantic import BaseModel

from annotated_types import Len
from pydantic import BaseModel
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated, Literal, Union

from pydantic_core import Url

from deepdrr import geo

class RenderMatrix4x4(BaseModel):
    data: Annotated[list[float], Len(min_length=16, max_length=16)]

def geo_to_render_matrix4x4(geo: geo.FrameTransform) -> RenderMatrix4x4:
    return RenderMatrix4x4(data=geo.data.astype(np.float32).flatten().tolist())
                           
def render_matrix4x4_to_geo(render: RenderMatrix4x4) -> geo.FrameTransform:
    return geo.FrameTransform(np.array(render.data, dtype=np.float32).reshape(4, 4))

class RenderStlMesh(BaseModel):
    prim_data_type: Literal["StlMesh"] = "StlMesh"
    url: Url
    material_name: str
    density: float
    priority: int
    addtive: bool
    subtractive: bool

class RenderH5Volume(BaseModel):
    prim_data_type: Literal["H5Volume"] = "H5Volume"
    url: Url
    priority: int

class RenderPrimitive(BaseModel):
    # primitive_id: str
    data: Annotated[Union[RenderStlMesh, RenderH5Volume], Field(discriminator="prim_data_type")]

class RenderInstance(BaseModel):
    # primitive_id: str
    primitive_id: int
    transform: RenderMatrix4x4
    morph_weights: list

class CameraIntrinsic(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float

class RenderCamera(BaseModel):
    transform: RenderMatrix4x4
    intrinsic: CameraIntrinsic

class FrameSettings(BaseModel):
    mode: str

class RenderFrame(BaseModel):
    frame_settings: FrameSettings
    camera: RenderCamera
    instances: list[RenderInstance]

class DRRRenderSettings(BaseModel):
    render_type: Literal["DRR"] = "DRR"
    width: int
    height: int
    step: float
    mode: str
    spectrum: str
    add_noise: bool
    photon_count: int
    collected_energy: bool
    neglog: bool
    intensity_upper_bound: float
    attenuate_outside_volume: bool
    max_mesh_hits: int

class RasterizeRenderSettings(BaseModel):
    render_type: Literal["Rasterize"] = "Rasterize"
    width: int
    height: int
    max_mesh_hits: int

class RenderSettings(BaseModel):
    settings: Annotated[Union[DRRRenderSettings, RasterizeRenderSettings], Field(discriminator="render_type")]

class RenderSequence(BaseModel):
    render_settings: RenderSettings
    primitives: list[RenderPrimitive]
    frames: list[RenderFrame]
