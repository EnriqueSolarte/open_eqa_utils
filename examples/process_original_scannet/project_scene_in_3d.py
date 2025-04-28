import hydra
from geometry_perception_utils.io_utils import get_abs_path, load_module
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc, plot_list_pcl
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous, eulerAnglesToRotationMatrix
import open_eqa_utils
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from imageio.v2 import imread, imwrite
import yaml
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_K
import pickle
from geometry_perception_utils.dense_voxel_grid import VoxelGrid2D, VoxelGrid3D


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    rgb_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/rgb"
    depth_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/depth"
    poses_dir = f"{cfg.pre_proc_dir}/{cfg.scene_name}/poses"

    bins_2d = pickle.load(
        Path(f"{cfg.pre_proc_dir}/{cfg.scene_name}/bins_voxel_map_2d.pkl").open('rb'))
    bins_3d = pickle.load(
        Path(f"{cfg.pre_proc_dir}/{cfg.scene_name}/bins_voxel_map_3d.pkl").open('rb'))
    voxel2d = VoxelGrid2D.from_bins(*bins_2d)
    voxel3d = VoxelGrid3D.from_bins(*bins_3d)

    list_idx = [fn.stem for fn in Path(rgb_dir).iterdir()]
    list_idx = sorted(list_idx, key=lambda x: int(x))

    # scales
    intrinsics_fn = f"{cfg.pre_proc_dir}/{cfg.scene_name}/intrinsics.yaml"
    with open(f"{intrinsics_fn}", "r") as f:
        intrinsics = yaml.safe_load(f)

    K = np.array(intrinsics["K"]).reshape(3, 3)

    global_xyz_rgb = []
    for idx in tqdm(list_idx[0:-1:10]):
        rgb = imread(f"{rgb_dir}/{idx}.jpg")
        depth_map = np.load(f"{depth_dir}/{idx}.npy")
        cam_pose = np.load(f"{poses_dir}/{idx}.npy")

        xyz, m = project_pp_depth_from_K(depth_map, K)
        if np.sum(m) < 10:
            continue
        xyz_rgb = get_color_array(rgb)[:, m]/255
        xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz)
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(
            xyz_wc[:3])

        local_xyz = np.vstack([xyz_wc_vx, xyz_rgb[:, xyz_idx]])
        global_xyz_rgb.append(local_xyz)
        # plot_list_pcl([x[:3] for x in global_xyz_rgb], shape=(512, 512))
    xyz_rgb_wc = np.hstack(global_xyz_rgb)

    args = dict(
        elevation=90,
        azimuth=0,
        up="z",
        roll=0,
        return_canvas=True,
        scale_factor=10,
    )
    
    xyz = xyz_rgb_wc[:3, :] - np.mean(xyz_rgb_wc[:3, :], axis=1, keepdims=True)
    canvas = plot_color_plc(
        xyz.T, xyz_rgb_wc[3:, :].T, **args)
    img = canvas.render()[:, :, :3]
    fn = f"{cfg.pre_proc_dir}/{cfg.scene_name}/test.jpg"
    imwrite(f"{fn}", img)
    print(f"Saved image to {fn}")


if __name__ == "__main__":
    main()
