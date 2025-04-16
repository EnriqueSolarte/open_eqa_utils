import hydra
from geometry_perception_utils.io_utils import get_abs_path, load_module
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imageio.v2 import imread
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.image_utils import get_color_array, get_map_from_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.pcl_utils import max_min_normalization
from geometry_perception_utils.config_utils import save_cfg
import matplotlib.pyplot as plt
import open_eqa_utils


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    save_cfg(cfg, resolve=True)
    rgb_dir = f"{cfg.open_eqa.rgb_dir}"
    depth_dir = f"{cfg.open_eqa.depth_dir}"
    poses_dir = f"{cfg.open_eqa.poses_dir}"

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    # Scales considering the whole scene (run examples/scene_coordinates/scene_xyz.py)
    scene_scale_max = [1.865, 1.065, 2.355]
    scene_scale_min = [-9.175, - 2.055, - 6.345]

    cam_projection = load_module(cfg.open_eqa.dataset.cam_projection_module)

    plt.figure(figsize=(5, 5))
    for idx in tqdm(list_idx[0:-1:10]):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        sc_map = np.zeros_like(rgb, dtype=np.float32)
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = cam_projection(depth_map, cfg.open_eqa)
        xyz_sc = get_color_array(sc_map)
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        sc_colors, _, _ = max_min_normalization(
            xyz_wc, scene_scale_max, scene_scale_min)
        xyz_sc[:, m] = sc_colors
        map_color = get_map_from_array(xyz_sc, rgb.shape)
        plt.clf()
        plt.imshow(np.hstack((map_color, rgb/255)))
        plt.draw()
        plt.waitforbuttonpress(0.01)
        plt.show()


if __name__ == "__main__":
    main()
