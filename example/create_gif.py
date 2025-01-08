
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from geometry_perception_utils.io_utils import create_directory
from geometry_perception_utils.vispy_utils import plot_color_plc
from imageio.v2 import imwrite
import time

assert len(sys.argv) == 4, f"Expected 3 arguments, got {len(sys.argv) - 1}."

input_xyz_npy = Path(sys.argv[1]).resolve()
tmp_dir = Path(sys.argv[2]).resolve()
output_fn = Path(sys.argv[3]).resolve()

assert os.path.exists(
    input_xyz_npy), f"NPY file {input_xyz_npy} does not exist."

xyz_rgb_wc = np.load(input_xyz_npy)
vis = dict(
    elevation=45,
    azimuth=0,
    up="-y",
    roll=0,
    shape=(512, 512),
)

tmp_dir = create_directory(f"{tmp_dir}", delete_prev=True, ignore_request=True)
for azimuth in range(0, 360, 15):
    vis['azimuth'] = azimuth
    canvas = plot_color_plc(
        xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T, return_canvas=True, **vis)
    img = canvas.render()
    canvas.close()
    # fn = f"{cfg.open_eqa.data_dir}/{cfg.open_eqa.scene_name}/xyz_rgb_wc.png"
    fn = f'{tmp_dir}/{time.time_ns()}.png'
    imwrite(fn, img)

os.system(f"convert -delay 10 -loop 0 {tmp_dir}/*.png {output_fn}")
