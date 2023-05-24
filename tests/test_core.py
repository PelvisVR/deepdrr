#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
from PIL import Image

import pyvista as pv

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames]
                   for funcargs in funcarglist]
    )


class TestSingleVolume:
    truth = Path.cwd() / "reference"
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    file_path = test_utils.download_sampledata("CT-chest")

    params = {
        "test_simple": [dict()],
        "test_mesh": [dict()],
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

    def project(self, volume, carm, name, meshes=None):
        with deepdrr.Projector(
            volume=volume,
            carm=carm,
            step=0.1,  # stepsize along projection ray, measured in voxels
            mode="linear",
            max_block_index=200,
            spectrum="90KV_AL40",
            photon_count=100000,
            add_scatter=False,
            threads=8,
            neglog=True,
            meshes=meshes,
        ) as projector:
            image = projector.project()

        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(self.output_dir / name)
        try: 
            truth_img = np.array(Image.open(self.truth / name))
            assert np.allclose(image, truth_img, atol=1)
        except FileNotFoundError:
            print(f"Truth image not found: {self.truth / name}")
        return image

    def test_simple(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
        self.project(volume, carm, "test_simple.png")

    def test_mesh(self):
        volume = deepdrr.Volume.from_nrrd(self.file_path)
        # load 10cmcube.stl from resources folder
        stl = pv.read("resources/suzanne.stl")
        # scale from m to mm
        stl.scale([100]*3, inplace=True)
        mesh = deepdrr.Mesh("steel", stl)
        carm = deepdrr.MobileCArm(isocenter=volume.center_in_world, sensor_width=300, sensor_height=200, pixel_size=0.6)
        self.project(volume, carm, "test_mesh.png", meshes=[mesh])

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
    test = TestSingleVolume()
    test.test_mesh()
    # volume = test.load_volume()
    # carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    # test.project(volume, carm, "test.png")
