import hydra
from geometry_perception_utils.io_utils import get_abs_path, load_module, save_yaml_dict, create_directory
from geometry_perception_utils.config_utils import save_cfg
from pathlib import Path
from tqdm import tqdm
from imageio.v2 import imread
import numpy as np
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.image_utils import get_color_array, get_map_from_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from geometry_perception_utils.pcl_utils import max_min_normalization
from geometry_perception_utils.dense_voxel_grid.voxel_grid_3d import VoxelGrid3D
import open_eqa_utils
import matplotlib.pyplot as plt


def pre_compute_scales(cfg):

    voxel_3d_map_fn = f"{cfg.open_eqa.bins_voxels_3d_fn}"

    bins_3d_map = np.load(f"{voxel_3d_map_fn}",  allow_pickle=True)
    voxel_map = VoxelGrid3D.from_bins(
        u_bins=bins_3d_map[0], v_bins=bins_3d_map[1], c_bins=bins_3d_map[2])

    depth_dir = f"{cfg.open_eqa.depth_dir}"
    poses_dir = f"{cfg.open_eqa.poses_dir}"

    list_idx = [fn.stem for fn in Path(depth_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    # Camera projection (both dataset has different projection)
    cam_projection = load_module(cfg.open_eqa.dataset.cam_projection_module)

    list_xyz = []
    for idx in tqdm(list_idx):
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = cam_projection(depth_map, cfg.open_eqa)
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)

        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel_map.project_xyz(xyz_wc)
        list_xyz.append(xyz_wc_vx)

    xyz_wc = np.hstack(list_xyz)
    # plot_color_plc(xyz_rgb_wc[:3, :].T, xyz_rgb_wc[3:, :].T)

    _, max_scale, min_scale = max_min_normalization(xyz_wc[:3, :])
    print(f"max_scale: {max_scale})")
    print(f"min_scale: {min_scale})")

    cfg_scene = dict(cfg.open_eqa.intrinsics)
    cfg_scene['max_scale'] = [float(s) for s in max_scale]
    cfg_scene['min_scale'] = [float(s) for s in min_scale]
    save_yaml_dict(cfg.open_eqa.intrinsics_fn, cfg_scene)
    print(f"Saved scales in {cfg.open_eqa.intrinsics_fn}")


def save_scene_coordinates_maps(cfg):
    rgb_dir = f"{cfg.open_eqa.rgb_dir}"
    depth_dir = f"{cfg.open_eqa.depth_dir}"
    poses_dir = f"{cfg.open_eqa.poses_dir}"

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    # scales
    scene_scale_max = cfg.open_eqa.intrinsics.max_scale
    scene_scale_min = cfg.open_eqa.intrinsics.min_scale

    cam_projection = load_module(cfg.open_eqa.dataset.cam_projection_module)

    # Create directories
    create_directory(cfg.open_eqa.sc_map_dir)
    create_directory(cfg.open_eqa.sc_map_vis_dir)
    
    for idx in tqdm(list_idx):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        sc_map = np.zeros_like(rgb, dtype=np.float32)
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = cam_projection(depth_map, cfg.open_eqa)
        xyz_sc = get_color_array(sc_map)
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)

        # Save map scene coordinates
        xyz_sc[:, m] = xyz_wc
        np.save(f"{cfg.open_eqa.sc_map_dir}/{idx}.npy", xyz_sc)

        # Save map scene coordinates visualization
        sc_colors_vis, _, _ = max_min_normalization(
            xyz_wc, scene_scale_max, scene_scale_min)
        xyz_sc[:, m] = sc_colors_vis
        map_color_vis = get_map_from_array(xyz_sc, rgb.shape)
        plt.figure(0)
        plt.clf()
        plt.imshow(np.hstack((map_color_vis, rgb/255)))
        plt.draw()
        plt.title(f"Scene coordinates map {idx}")
        plt.axis('off')
        plt.tight_layout()
        # Reduce white space
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
        plt.savefig(f"{cfg.open_eqa.sc_map_vis_dir}/{idx}.jpg", dpi=80, bbox_inches='tight', pad_inches=0)
        plt.close()


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    save_cfg(cfg, resolve=True)
    # pre_compute_scales(cfg)
    save_scene_coordinates_maps(cfg)


if __name__ == "__main__":
    main()
