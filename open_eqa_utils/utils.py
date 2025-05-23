from pathlib import Path
import json
import logging
from geometry_perception_utils.image_utils import imread, get_color_array
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov, project_pp_depth_from_K
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.io_utils import create_directory, save_yaml_dict
from open_eqa_utils.pre_process_scannet_dataset.scannet_open_eqa.SensorData import SensorData
import numpy as np
import os
import open_eqa_utils


def load_q_and_a(json_path: Path):
    with open(json_path, "r") as f:
        q_and_a = json.load(f)
    return q_and_a


def check_file_exists(fn):
    if Path(fn).exists():
        logging.warning(f"File already exists @ {fn}")
        input("The file will be overwritten. Press Enter to continue...")


def create_scene_dir(scene_dir, delete_prev=True):
    data_dir = create_directory(
        scene_dir, delete_prev=delete_prev, ignore_request=True)
    rgb_dir = create_directory(
        f"{data_dir}/rgb", delete_prev=delete_prev, ignore_request=True)
    depth_dir = create_directory(
        f"{data_dir}/depth", delete_prev=delete_prev, ignore_request=True)
    poses_dir = create_directory(
        f"{data_dir}/poses", delete_prev=delete_prev, ignore_request=True)
    return rgb_dir, depth_dir, poses_dir


def get_registered_xyz_rgb_wc_scannet(cfg, frame):

    rgb = imread(f"{cfg.rgb_dir}/{frame}.jpg")
    depth = np.load(f"{cfg.depth_dir}/{frame}.npy")
    cam_pose = np.load(f"{cfg.poses_dir}/{frame}.npy")

    K = np.array(cfg.intrinsics.K).reshape(3, 3)
    xyz_cc, m = project_pp_depth_from_K(depth, K)
    if m.sum(0) == 0:
        return None
    xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
    xyz_color = get_color_array(rgb)[:, m]/255

    return np.vstack((xyz_wc, xyz_color))


def get_registered_xyz_rgb_wc_hm3d(cfg, frame):

    rgb = imread(f"{cfg.rgb_dir}/{frame}.jpg")
    depth = np.load(f"{cfg.depth_dir}/{frame}.npy")
    cam_pose = np.load(f"{cfg.poses_dir}/{frame}.npy")

    xyz_cc, m = project_pp_depth_from_hfov(depth, cfg.intrinsics.sensor_hfov)
    if m.sum(0) == 0:
        return None
    xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
    xyz_color = get_color_array(rgb)[:, m]/255

    return np.vstack((xyz_wc, xyz_color))


def cam_projection_hm3d(depth, cfg):
    xyz_cc, m = project_pp_depth_from_hfov(depth, cfg.intrinsics.sensor_hfov)
    return xyz_cc, m


def cam_projection_scannet(depth, cfg):
    K = np.array(cfg.intrinsics.K).reshape(3, 3)
    xyz_cc, m = project_pp_depth_from_K(depth, K)
    return xyz_cc, m


def save_visualization(xyz_rgb_wc, cfg, output_fn):
    logging.info(f"Saving visualization to {cfg.log_dir}")
    prefix = f"{cfg.open_eqa.dataset.prefix}"
    xyz_fn = f"{cfg.log_dir}/{prefix}_xyz_rgb_wc.npy"
    np.save(xyz_fn, xyz_rgb_wc)

    script = f"{Path(open_eqa_utils.__file__).parent}/create_gif.py"

    command = f"{cfg.python} {script} {xyz_fn} {cfg.log_dir}/{prefix}_tmp/ {output_fn}"
    print("... Creating GIF visualization (May take a few seconds)")
    os.system(command)
    logging.info(f"Saved visualization to {output_fn}")
