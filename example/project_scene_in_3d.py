import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
import open_eqa_utils
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from imageio.v2 import imread


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    list_frames = [Path(f).stem for f in os.listdir(cfg.open_eqa.rgb_dir)]
    _xyz_rgb_wc = []

    for fr in tqdm(list_frames[0:-1:5]):
        rgb = imread(f"{cfg.open_eqa.rgb_dir}/{fr}.jpg")
        depth = np.load(f"{cfg.open_eqa.depth_dir}/{fr}.npy")
        cam_pose = np.load(f"{cfg.open_eqa.poses_dir}/{fr}.npy")

        xyz_cc, m = project_pp_depth_from_hfov(depth, cfg.open_eqa.hfov)
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
        xyz_color = get_color_array(rgb)[:, m]/255

        _xyz_rgb_wc.append(np.vstack((xyz_wc, xyz_color)))

    xyz_rgb_wc = np.hstack(_xyz_rgb_wc)
    plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)


if __name__ == "__main__":
    main()
