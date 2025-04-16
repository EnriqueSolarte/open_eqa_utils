import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import open_eqa_utils
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path
import pickle
from geometry_perception_utils.dense_voxel_grid import VoxelGrid3D
from geometry_perception_utils.io_utils import load_module
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
import os
from open_eqa_utils.utils import save_visualization


def render_scene(cfg):
    list_frames = [Path(f).stem for f in os.listdir(cfg.open_eqa.rgb_dir)]

    # Loading pre-computed Voxels grid from saved pickle bin files
    bins_3d = pickle.load(Path(cfg.open_eqa.bins_voxels_3d_fn).open('rb'))
    voxel3d = VoxelGrid3D.from_bins(*bins_3d)

    xyz_wc_registration = load_module(
        cfg.open_eqa.dataset.xyz_wc_registration_module)
    global_xyz_rgb = []

    skip_frames = cfg.get('skip_frames', int(list_frames.__len__()*0.1))
    max_frames = cfg.get('max_frames', -1)

    cam_0 = np.load(f"{cfg.open_eqa.poses_dir}/{list_frames[0]}.npy")
    for fr in tqdm(list_frames[0:max_frames:skip_frames]):
        xyz_rgb_wc = xyz_wc_registration(cfg.open_eqa, fr)
        if xyz_rgb_wc is None:
            continue
        xyz_wc_vx, vx_idx, xyz_idx, all_xyz_idx = voxel3d.project_xyz(
            xyz_rgb_wc[:3])
        # Mapping from vx domain --> xyz Euclidean domain
        # print(xyz_idx.shape, xyz_wc_vx.shape, xyz_idx.max())
        # Mapping xyz --> vx
        # print(all_xyz_idx.shape, xyz_wc.shape, all_xyz_idx.max())

        xyz_cam_0 = np.linalg.inv(
            cam_0)[:3, :] @ extend_array_to_homogeneous(xyz_wc_vx)
        local_xyz = np.vstack([xyz_cam_0, xyz_rgb_wc[3:, xyz_idx]])
        global_xyz_rgb.append(local_xyz)
        # plot_list_pcl([x[:3] for x in global_xyz_rgb], shape=(512, 512))
    xyz_rgb_wc = np.hstack(global_xyz_rgb)
    prefix = f"{cfg.open_eqa.dataset.prefix}"
    output_fn = f"{cfg.log_dir}/{prefix}_vis_scene.gif"
    save_visualization(xyz_rgb_wc, cfg, output_fn)


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    render_scene(cfg)


if __name__ == '__main__':
    main()
