from deepdrr.utils import test_utils
from deepdrr.frontend.frontend import *
from deepdrr.frontend.renderer import *
from deepdrr.frontend.profiles import *

f = test_utils.download_sampledata("CT-chest")
mf = Path("tests/resources/suzanne.stl")
mf2 = Path("tests/resources/threads.stl")
print(f)

volume = Volume.from_nrrd(f)

mesh1 = Mesh.from_stl(mf, tag="mesh1")
mesh2 = Mesh.from_stl(mf2, tag="mesh2")

carm = MobileCArm()
carm

graph = TransformTree()
graph.add(carm)
graph.add(volume)
graph.add(mesh1)
graph.add(mesh2)

scene = GraphScene(graph)

drr_json_path = Path("drr.json")
drr_settings = DRRRenderSettings(
    width=512,
    height=512,
    neglog=True,
)
drr_prof = DRRRenderProfile(
    settings=drr_settings,
    renderer=DeferredRenderer(drr_json_path),
    # renderer=SynchronousRenderer(),
    scene=scene,
)
rast_settings = RasterizeRenderSettings(
    width=512,
    height=512,
    max_mesh_hits=32,
)
rasterize_prof = RasterizeRenderProfile(
    settings=rast_settings, renderer=SynchronousRenderer(), scene=scene
)

mesh1.world_from_anatomical = geo.FrameTransform.from_translation([-30, 50, 200])

with drr_prof, rasterize_prof:
    for frame_idx in range(10):
        mesh2.world_from_anatomical = geo.FrameTransform.from_translation(
            [0, 0, 10 * frame_idx]
        )
        carm.move_by(
            delta_isocenter=[0, 0, 10],
            delta_alpha=5 * frame_idx,
            delta_beta=0,
            delta_gamma=0,
            degrees=False,
        )
        drr = drr_prof.render_drr()
        segs = rasterize_prof.render_seg()
        hits = rasterize_prof.render_hits()
