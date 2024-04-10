from __future__ import annotations

from .scene import Scene
from .render_serial import *
from .transform_manager import *


class Backend(ABC):
    def __init__(self):
        pass

    # def process_sequence(self, sequence: RenderSequence) -> List[Any]:
    #     # initialize the backend with the render settings and the primitives
    #     # render the sequence
    #     self.init(sequence.render_settings, sequence.primitives)
    #     return self.render_batches(sequence.frames)

    @abstractmethod
    def init(self, render_settings: RenderSettings, primitives: List[RenderPrimitive]): ...

    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback): ...

    @abstractmethod
    def render_batches(self, frames: List[RenderFrame]): ...

    @classmethod
    def create(cls, render_settings: RenderSettings) -> Backend:
        if render_settings.settings.render_type == "DRR":
            return DRRBackend()
        elif render_settings.settings.render_type == "Rasterize":
            return RasterizeBackend()
        else:
            raise ValueError(
                f"Unknown render type {render_settings.settings.render_type}"
            )


class DRRBackend(Backend):

    def __init__(self):
        pass

    def init(self, render_settings: RenderSettings, primitives: List[RenderPrimitive]):
        # TODO
        pass

    def init_render_backend(self, render_settings: RenderSettings):
        # TODO
        pass

    def init_primitives(self, primitives: List[RenderPrimitive]):
        # TODO
        pass

    def __enter__(self):
        print("Entering DRR Backend")
        # TODO
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting DRR Backend")
        # TODO
        pass

    def render_batches(self, batches: List[RenderBatch]):
        print(f"Rendering {len(batches)} frames")
        # TODO
        pass
        # render_settings = self._render_settings
        # for frame in frames:
        #     frame_settings = frame.frame_settings

    # def render_batch(self, batch: RenderBatch):
    #     return self.render_batches([batch])[0]


class RasterizeBackend(Backend):

    def __init__(self):
        pass

    def init(self, render_settings: RenderSettings, primitives: List[RenderPrimitive]):
        # TODO
        pass

    def init_render_backend(self, render_settings: RenderSettings):
        # TODO
        pass

    def init_primitives(self, primitives: List[RenderPrimitive]):
        # TODO
        pass

    def __enter__(self):
        # TODO
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO
        pass

    def render_batches(self, batches: List[RenderBatch]):
        # TODO
        pass

    # def render_batch(self, frame: RenderFrame):
    #     return self.render_batches([frame])[0]
