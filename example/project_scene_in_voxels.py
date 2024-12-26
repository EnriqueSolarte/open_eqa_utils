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
from open_eqa_utils.pre_process_hm3d_dataset.pre_compute_voxels import get_registered_xyz_rgb_wc
from geometry_perception_utils.dense_voxel_grid import VoxelGrid2D, VoxelGrid3D
import pickle


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    list_frames = [Path(f).stem for f in os.listdir(cfg.open_eqa.rgb_dir)]

    # Loading pre-computed Voxels grid from saved pickle bin files
    bins_2d = pickle.load(Path(cfg.open_eqa.bins_voxels_2d_fn).open('rb'))
    bins_3d = pickle.load(Path(cfg.open_eqa.bins_voxels_3d_fn).open('rb'))
    voxel2d = VoxelGrid2D.from_bins(*bins_2d)
    voxel3d = VoxelGrid3D.from_bins(*bins_3d)

    global_xyz_rgb = []
    for fr in tqdm(list_frames):
        # Read RGB, Depth, Camera Pose and register them to WC
        xyz_rgb_wc = get_registered_xyz_rgb_wc(cfg.open_eqa, fr)
        
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(xyz_rgb_wc[:3])
        # Mapping from vx domain --> xyz Euclidean domain
        # print(xyz_idx.shape, xyz_wc_vx.shape, xyz_idx.max())
        # Mapping xyz --> vx
        # print(all_xyz_idx.shape, xyz_wc.shape, all_xyz_idx.max())
        
        local_xyz = np.vstack([xyz_wc_vx, xyz_rgb_wc[3:, xyz_idx]])
        global_xyz_rgb.append(local_xyz)
        
    xyz_rgb_wc = np.hstack(global_xyz_rgb)
    plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)


if __name__ == "__main__":
    main()
