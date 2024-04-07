from __future__ import annotations

from .frontend import *
from .renderer import Renderer
from .transform_manager import *


class RenderProfile(ABC):

    def __init__(
        self,
        settings: RenderSettingsUnion,
        renderer: Renderer,
        scene: Scene,
    ):
        self._settings = settings
        self._renderer = renderer
        self._scene = scene

    @property
    def settings(self):
        return self._settings

    @property
    def renderer(self):
        return self._renderer

    @property
    def scene(self):
        return self._scene

    def __enter__(self):
        render_settings = RenderSettings(settings=self._settings)
        if not self._renderer.is_initialized:
            self._renderer.init(self._scene, render_settings)
        self._renderer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._renderer.__exit__(exc_type, exc_value, traceback)

    def render_single(self, frame_settings: FrameSettings):
        camera = self._scene.get_camera()
        instances = self._scene.instance_snapshot()
        render_frame = RenderFrame(
            frame_settings=frame_settings,
            camera=camera.to_render_camera(),
            instances=instances,
        )
        self._renderer.render_frames([render_frame])


class DRRRenderProfile(RenderProfile):

    def render_drr(self):
        frame_settings = FrameSettings(mode="drr")
        self.render_single(frame_settings)


class RasterizeRenderProfile(RenderProfile):

    def render_seg(self):
        frame_settings = FrameSettings(mode="seg")
        self.render_single(frame_settings)

    def render_travel(self):
        frame_settings = FrameSettings(mode="travel")
        self.render_single(frame_settings)

    def render_hits(self):
        frame_settings = FrameSettings(mode="hits")
        self.render_single(frame_settings)
