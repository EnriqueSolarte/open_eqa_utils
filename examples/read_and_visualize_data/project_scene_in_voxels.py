import hydra
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc, plot_list_pcl
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
import open_eqa_utils
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from imageio.v2 import imread
from geometry_perception_utils.dense_voxel_grid import VoxelGrid2D, VoxelGrid3D
from geometry_perception_utils.io_utils import load_module
import pickle
import logging


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    # Setting scene_name on runtime
    list_scenes = cfg.open_eqa.dataset.scene_list
    scene = list_scenes[0]
    cfg.open_eqa.dataset.prefix = scene

    # Reading RGB frames
    list_frames = [Path(f).stem for f in os.listdir(cfg.open_eqa.rgb_dir)]

    # Loading pre-computed Voxels grid from saved pickle bin files
    bins_2d = pickle.load(Path(cfg.open_eqa.bins_voxels_2d_fn).open('rb'))
    bins_3d = pickle.load(Path(cfg.open_eqa.bins_voxels_3d_fn).open('rb'))
    voxel2d = VoxelGrid2D.from_bins(*bins_2d)
    voxel3d = VoxelGrid3D.from_bins(*bins_3d)

    # Load module for camera registration to world coordinates
    xyz_wc_registration = load_module(
        cfg.open_eqa.dataset.xyz_wc_registration_module)
    logging.info(
        f"registration module: {cfg.open_eqa.dataset.xyz_wc_registration_module}"
    )

    global_xyz_rgb = []
    skip_frames = cfg.get('skip_frames', 3)
    max_frames = cfg.get('max_frames', -1)
    for fr in tqdm(list_frames[0:max_frames:skip_frames]):
        # Read RGB, Depth, Camera Pose and register them to WC
        xyz_rgb_wc = xyz_wc_registration(cfg.open_eqa, fr)
        if xyz_rgb_wc is None:
            continue
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(
            xyz_rgb_wc[:3])
        # Mapping from vx domain --> xyz Euclidean domain
        # print(xyz_idx.shape, xyz_wc_vx.shape, xyz_idx.max())
        # Mapping xyz --> vx
        # print(all_xyz_idx.shape, xyz_wc.shape, all_xyz_idx.max())

        local_xyz = np.vstack([xyz_wc_vx, xyz_rgb_wc[3:, xyz_idx]])
        global_xyz_rgb.append(local_xyz)
        # plot_list_pcl([x[:3] for x in global_xyz_rgb], shape=(512, 512))
    xyz_rgb_wc = np.hstack(global_xyz_rgb)
    plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)


if __name__ == "__main__":
    main()
