import hydra
from geometry_perception_utils.io_utils import get_abs_path, load_module
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imageio.v2 import imread
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.dense_voxel_grid.voxel_grid_2d import VoxelGrid2D
from geometry_perception_utils.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D
from matplotlib.colors import hsv_to_rgb
import open_eqa_utils
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.pcl_utils import max_min_normalization


def load_voxels_map(cfg):
    # Voxel map for the data
    voxel_3d_map_fn = f"{cfg.open_eqa.bins_voxels_3d_fn}"

    bins_3d_map = np.load(f"{voxel_3d_map_fn}",  allow_pickle=True)
    voxel_3d_map = VoxelGrid3D.from_bins(
        u_bins=bins_3d_map[0], v_bins=bins_3d_map[1], c_bins=bins_3d_map[2])
    return voxel_3d_map


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    save_cfg(cfg, resolve=True)

    voxel_map = load_voxels_map(cfg)

    rgb_dir = f"{cfg.open_eqa.rgb_dir}"
    depth_dir = f"{cfg.open_eqa.depth_dir}"
    poses_dir = f"{cfg.open_eqa.poses_dir}"

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    cam_projection = load_module(cfg.open_eqa.dataset.cam_projection_module)

    list_xyz = []
    for idx in tqdm(list_idx[0:-1:10]):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = cam_projection(depth_map, cfg.open_eqa)
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)

        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel_map.project_xyz(xyz_wc)
        list_xyz.append(xyz_wc_vx)

    xyz_wc = np.hstack(list_xyz)
    # plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)

    sc_color, max_scale, min_scale = max_min_normalization(xyz_wc[:3, :])
    print(f"max_scale: {max_scale})")
    print(f"min_scale: {min_scale})")
    plot_color_plc(xyz_wc.T, sc_color.T)


if __name__ == "__main__":
    main()
