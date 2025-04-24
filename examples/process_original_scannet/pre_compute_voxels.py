import hydra
from geometry_perception_utils.io_utils import get_abs_path
from pathlib import Path
import os
from tqdm import tqdm
from imageio.v2 import imread
import numpy as np
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_K
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.dense_voxel_grid import VoxelGrid3D, VoxelGrid2D
import logging
import open_eqa_utils
import pickle
from open_eqa_utils.utils import check_file_exists
import yaml


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    rgb_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/rgb"
    depth_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/depth"
    poses_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/poses"

    with open(f"{cfg.pre_proc_dir}/{cfg.scene_name}/intrinsics.yaml", "r") as f:
        intrinsics = yaml.safe_load(f)

    K = np.array(intrinsics["K"]).reshape(3, 3)
    list_frames = [Path(f).stem for f in os.listdir(rgb_dir)]
    _xyz_rgb_wc = []

    # check if the files exist
    bins_voxel_map_2d_fn = f"{cfg.pre_proc_dir}/{cfg.scene_name}/bins_voxel_map_2d.pkl"
    check_file_exists(bins_voxel_map_2d_fn)
    bins_voxel_map_3d_fn = f"{cfg.pre_proc_dir}/{cfg.scene_name}/bins_voxel_map_3d.pkl"
    check_file_exists(bins_voxel_map_3d_fn)

    voxel2d = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel3d = VoxelGrid3D(cfg.voxel_grid_3d)

    for frame in tqdm(list_frames, desc="Creating voxel maps..."):
        rgb = imread(f"{rgb_dir}/{frame}.jpg")
        depth = np.load(f"{depth_dir}/{frame}.npy")
        cam_pose = np.load(f"{poses_dir}/{frame}.npy")

        xyz_cc, m = project_pp_depth_from_K(depth, K)
        if m.sum(0) == 0:
            continue
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
        xyz_color = get_color_array(rgb)[:, m]/255

        _xyz_rgb_wc = np.vstack((xyz_wc, xyz_color))

        _ = voxel2d.project_xyz(_xyz_rgb_wc[:3, :])
        _ = voxel3d.project_xyz(_xyz_rgb_wc[:3, :])

    # save voxel maps as pickle
    pickle.dump(voxel2d.get_bins(), open(bins_voxel_map_2d_fn, "wb"))
    logging.info(f"2D voxel map saved @ {bins_voxel_map_2d_fn}")
    pickle.dump(voxel3d.get_bins(), open(bins_voxel_map_3d_fn, "wb"))
    logging.info(f"3D voxel map saved @ {bins_voxel_map_2d_fn}")


if __name__ == "__main__":
    main()
