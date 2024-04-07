from __future__ import annotations

from .scene import Scene
from .render_serial import *
from .transform_manager import *
from .backend import *


class Renderer(ABC):

    @abstractmethod
    def init(self, scene: Scene, render_settings: RenderSettings):
        ...

    @property
    @abstractmethod
    def is_initialized(self): ...

    @abstractmethod
    def render_frames(self, frames: List[RenderFrame]):
        ...

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        ...


class SynchronousRenderer(Renderer):

    def __init__(self):
        self._scene = None
        self._backend = None
        self._render_settings = None

    @property
    def is_initialized(self):
        return self._scene is not None

    def init(self, scene: Scene, render_settings: RenderSettings):
        if self.is_initialized:
            raise ValueError("Renderer already initialized")
        assert isinstance(scene, Scene), f"Expected Scene, got {type(scene)}"
        assert isinstance(
            render_settings, RenderSettings
        ), f"Expected RenderSettings, got {type(render_settings)}"
        self._scene = scene
        self._backend = Backend.create(render_settings)
        self._render_settings = render_settings

    def __enter__(self):
        if not self.is_initialized:
            raise ValueError("Renderer not initialized")
        self._backend.init(self._render_settings, self._scene.get_render_primitives())
        self._backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._backend.__exit__(exc_type, exc_value, traceback)

    def render_frames(self, frames: List[RenderFrame]):
        return self._backend.render_frames(frames)


class ProcessRenderer(Renderer):
    pass


class DeferredRenderer(Renderer):
    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        self._frames = []
        self._scene = None
        self._render_settings = None

    @property
    def is_initialized(self):
        return self._scene is not None

    def init(self, scene: Scene, render_settings: RenderSettings):
        assert isinstance(scene, Scene), f"Expected Scene, got {type(scene)}"
        assert isinstance(
            render_settings, RenderSettings
        ), f"Expected RenderSettings, got {type(render_settings)}"
        if self.is_initialized:
            raise ValueError("Renderer already initialized")
        self._scene = scene
        self._render_settings = render_settings

    def __enter__(self):
        if not self.is_initialized:
            raise ValueError("Renderer not initialized")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self._path, "w") as f:
            rs = RenderSequence(
                render_settings=self._render_settings,
                primitives=self._scene.get_render_primitives(),
                frames=self._frames,
            )
            # f.write(rs.model_dump_json())
            f.write(rs.model_dump_json(indent=2))

    def render_frames(self, frames: List[RenderFrame]):
        self._frames.extend(frames)
