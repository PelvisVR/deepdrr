#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
from PIL import Image
import pytest
import copy

import pyvista as pv
import logging
import pyrender
from deepdrr.pyrenderdrr.material import DRRMaterial
from deepdrr.utils.mesh_utils import *

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames]
                   for funcargs in funcarglist]
    )


class TestSingleVolume:
    d = Path(__file__).resolve().parent
    truth = d / "reference"
    output_dir = d / "output"
    output_dir.mkdir(exist_ok=True)
    file_path = test_utils.download_sampledata("CT-chest")

    params = {
        "test_simple": [dict()],
        "test_mesh": [dict()],
        "test_mesh_only": [dict()],
        "test_translate": [
            dict(t=[0, 0, 0]),
            dict(t=[100, 0, 0]),
            dict(t=[0, 100, 0]),
            dict(t=[0, 0, 100]),
        ],
        "test_rotate_x": [dict(x=0), dict(x=30), dict(x=45), dict(x=90), dict(x=180)],
        "test_angle": [dict(alpha=0, beta=90)],
    }

    def load_volume(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        volume.rotate(Rotation.from_euler("x", -90, degrees=True))
        return volume

    def project(self, volume, carm, name):

        try: 
            truth_img = np.array(Image.open(self.truth / name))
        except FileNotFoundError:
            print(f"Truth image not found: {self.truth / name}")
            # pytest.skip("Truth image not found")
            pytest.fail("Truth image not found")

        with deepdrr.Projector(
            volume=volume,
            carm=carm,
            step=0.1,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=200,
            spectrum="90KV_AL40",
            photon_count=100000,
            scatter_num=0,
            threads=8,
            neglog=True,
        ) as projector:
            image = projector.project()
            from timer_util import FPS
            # fps = FPS()
            # for i in range(50):
            #     image = projector.project()
            #     if fps_count := fps():
            #         print(f"FPS2 {fps_count}")

        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(self.output_dir / name)
        # diff
        Image.fromarray(np.abs(image - truth_img)).save(self.output_dir / f"diff_{name}")
        assert np.allclose(image, truth_img, atol=1)
        print(f"Test {name} passed")
        return image

    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_simple.png")

    def test_mesh(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        # load 10cmcube.stl from resources folder
        # stl = pv.read("tests/resources/10cmrighttri.stl")
        stl3 = pv.read("tests/resources/10cmcube.stl")
        stl3.scale([100, 100, 100], inplace=True)
        stl3.rotate_z(60, inplace=True)
        # stl3.translate([0, 00, 0], inplace=True)

        stl2 = pv.read("tests/resources/10cmcube.stl")
        stl2.scale([200, 200, 200], inplace=True)
        # stl2.translate([0, 30, 0], inplace=True)
        stl = pv.read("tests/resources/suzanne.stl")
        stl.scale([200]*3, inplace=True)
        stl.translate([0, -200, 0], inplace=True)
        # stl = pv.read("tests/resources/suzanne.stl")
        morph_targets = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [-1, 0, 1],
            [0, 0, 1],
                                  ]).reshape(1, -1, 3)
        # scale from m to mm
        # mesh = deepdrr.Mesh("titanium", 7, stl, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("y", 90, degrees=True)))
        # mesh = deepdrr.Mesh("air", 0, stl, morph_targets=morph_targets, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True)))
        prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("titanium", density=2, subtractive=True))
        # prim = pyrender.Mesh.from_trimesh(polydata_to_trimesh(stl), material=DRRMaterial("bone", density=2, subtractive=True))
        mesh = deepdrr.Mesh(mesh=prim, world_from_anatomical=geo.FrameTransform.from_rotation(geo.Rotation.from_euler("x", 90, degrees=True) * geo.Rotation.from_euler("y", 30, degrees=True)))

        # prim2 = deepdrr.Primitive("titanium", 2, stl2, subtractive=True)
        prim2 = trimesh_to_pyrender_mesh(polydata_to_trimesh(stl2), material=DRRMaterial("lung", density=2, subtractive=True))
        mesh2 = deepdrr.Mesh(mesh=prim2, world_from_anatomical=geo.FrameTransform.from_translation([30, 50, 200]))

        # prim3 = deepdrr.Primitive("titanium", 0, stl2, subtractive=True)
        prim3 = polydata_to_pyrender_mesh(stl2, material=DRRMaterial("titanium", density=0, subtractive=True))
        mesh3 = deepdrr.Mesh(mesh=prim3, world_from_anatomical=geo.FrameTransform.from_translation([-30, 20, -70]))
        # mesh = deepdrr.Mesh("polyethylene", 1.05, stl)
        # mesh.morph_weights = np.array([-10])
        
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        # self.project([volume], carm, "test_mesh.png")
        # self.project([mesh, mesh2, mesh3], carm, "test_mesh.png")
        self.project([volume, mesh, mesh2, mesh3], carm, "test_mesh.png")


    def test_translate(self, t):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        volume.translate(t)
        self.project(
            volume, carm, f"test_translate_{int(t[0])}_{int(t[1])}_{int(t[2])}.png"
        )

    def test_rotate_x(self, x):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        volume.rotate(Rotation.from_euler(
            "x", x, degrees=True), volume.center_in_world)
        self.project(volume, carm, f"test_rotate_x={int(x)}.png")

    def test_angle(self, alpha, beta):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        carm.move_to(alpha=alpha, beta=beta, degrees=True)
        self.project(
            volume, carm, f"test_angle_alpha={int(alpha)}_beta={int(beta)}.png"
        )


if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    # set projector log level to debug
    logging.basicConfig(level=logging.DEBUG)
    test = TestSingleVolume()
    test.test_mesh()
    # volume = test.load_volume()
    # carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    # test.project(volume, carm, "test.png")
