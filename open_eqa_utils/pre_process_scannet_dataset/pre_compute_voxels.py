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
from open_eqa_utils.utils import check_file_exists, get_registered_xyz_rgb_wc_scannet


def process_voxels_scannet(cfg):
    list_frames = [Path(f).stem for f in os.listdir(cfg.open_eqa.rgb_dir)]
    _xyz_rgb_wc = []

    # check if the files exist
    fn = f"{cfg.open_eqa.data_dir}/{cfg.open_eqa.scene_name}/bins_voxel_map_2d.pkl"
    check_file_exists(fn)
    fn = f"{cfg.open_eqa.data_dir}/{cfg.open_eqa.scene_name}/bins_voxel_map_3d.pkl"
    check_file_exists(fn)

    voxel2d = VoxelGrid2D(cfg.voxel_grid_2d)
    voxel3d = VoxelGrid3D(cfg.voxel_grid_3d)

    for fr in tqdm(list_frames, desc="Creating voxel maps..."):
        _xyz_rgb_wc = get_registered_xyz_rgb_wc_scannet(cfg.open_eqa, fr)
        _ = voxel2d.project_xyz(_xyz_rgb_wc[:3, :])
        _ = voxel3d.project_xyz(_xyz_rgb_wc[:3, :])

    # save voxel maps as pickle
    fn = f"{cfg.open_eqa.data_dir}/{cfg.open_eqa.scene_name}/bins_voxel_map_2d.pkl"
    pickle.dump(voxel2d.get_bins(), open(fn, "wb"))
    logging.info(f"2D voxel map saved @ {fn}")
    fn = f"{cfg.open_eqa.data_dir}/{cfg.open_eqa.scene_name}/bins_voxel_map_3d.pkl"
    pickle.dump(voxel3d.get_bins(), open(fn, "wb"))
    logging.info(f"3D voxel map saved @ {fn}")


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    list_scenes = cfg.scannet_scene_list
    for scene in tqdm(list_scenes, desc="Processing scenes"):
        logging.info(f"Processing scene: {scene}")
        cfg.open_eqa.scene_name = scene
        process_voxels_scannet(cfg)
        return


if __name__ == "__main__":
    main()
