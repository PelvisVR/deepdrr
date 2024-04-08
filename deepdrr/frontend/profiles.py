from __future__ import annotations

from .scene import *
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
        self._

    @property
    def settings(self):
        return self._settings

    @property
    def renderer(self):
        return self._renderer

    @property
    def scene(self):
        return self._scene

    def _get_camera(self) -> Camera:
        return self._scene.get_camera()

    def _get_primitives(self) -> List[Primitive]:
        return self._scene.get_primitives()

    def _get_render_primitives(self) -> List[RenderPrimitive]:
        primitives = self._get_primitives()
        return [primitive.to_render_primitive() for primitive in primitives]

    def _get_instances(self) -> List[PrimitiveInstance]:
        return self._scene.get_instances()

    def _get_instance_snapshot(self) -> List[RenderInstance]:
        instances = self._get_instances()
        return [instance.to_render_instance() for instance in instances]

    def __enter__(self):
        render_settings = RenderSettings(settings=self._settings)
        if not self._renderer.is_initialized:
            self._renderer.init(self._scene, render_settings)
        self._renderer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._renderer.__exit__(exc_type, exc_value, traceback)

    def render_frame(self, frame_settings: FrameSettings):
        camera = self._get_camera()
        instances = self._get_instance_snapshot()
        render_frame = RenderFrame(
            frame_settings=frame_settings,
            camera=camera.to_render_camera(),
            instances=instances,
        )
        render_batch = RenderBatch([render_frame])
        self._renderer.render_batches([render_batch])

    def render_tag_batch(self, frame_settings: FrameSettings, tags: Optional[List[str]] = None):
        camera = self._get_camera()
        render_camera = camera.to_render_camera()
        instances = self._get_instances()

        batch = []
        if tags is not None:
            for tag in tags:
                tagged_instances = [instance for instance in instances if tag in instance.tags]
                render_instances = [instance.to_render_instance() for instance in tagged_instances]
                render_frame = RenderFrame(
                    frame_settings=frame_settings,
                    camera=render_camera,
                    instances=render_instances,
                )
                batch.append(render_frame)
        else:
            render_instances = [instance.to_render_instance() for instance in instances]
            render_frame = RenderFrame(
                frame_settings=frame_settings,
                camera=render_camera,
                instances=render_instances,
            )
            batch.append(render_frame)

        render_batch = RenderBatch(batch)
        self._renderer.render_batches([render_batch])


class DRRRenderProfile(RenderProfile):

    def render_drr(self):
        frame_settings = FrameSettings(mode="drr")
        self.render_single(frame_settings)


class RasterizeRenderProfile(RenderProfile):

    def render_seg(self, tags: Optional[List[str]] = None):
        frame_settings = FrameSettings(mode="seg")
        self.render_tag_batch(frame_settings, tags)

    def render_travel(self, tags: Optional[List[str]] = None):
        frame_settings = FrameSettings(mode="travel")
        self.render_tag_batch(frame_settings, tags)

    def render_hits(self, tags: Optional[List[str]] = None):
        frame_settings = FrameSettings(mode="hits")
        self.render_tag_batch(frame_settings, tags)
