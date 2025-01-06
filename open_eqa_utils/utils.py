from pathlib import Path
import json
import logging
from geometry_perception_utils.image_utils import imread, get_color_array
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_hfov, project_pp_depth_from_K
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.io_utils import create_directory, save_yaml_dict
from open_eqa_utils.pre_process_scannet_dataset.scannet_open_eqa.SensorData import SensorData
import numpy as np


def load_q_and_a(json_path: Path):
    with open(json_path, "r") as f:
        q_and_a = json.load(f)
    return q_and_a


def check_file_exists(fn):
    if Path(fn).exists():
        logging.warning(f"File already exists @ {fn}")
        input("The file will be overwritten. Press Enter to continue...")


def create_scene_dir(scene_dir):
    data_dir = create_directory(scene_dir)
    rgb_dir = create_directory(f"{data_dir}/rgb")
    depth_dir = create_directory(f"{data_dir}/depth")
    poses_dir = create_directory(f"{data_dir}/poses")
    return rgb_dir, depth_dir, poses_dir


def get_registered_xyz_rgb_wc_scannet(cfg, frame):

    rgb = imread(f"{cfg.rgb_dir}/{frame}.jpg")
    depth = np.load(f"{cfg.depth_dir}/{frame}.npy")
    cam_pose = np.load(f"{cfg.poses_dir}/{frame}.npy")

    K = np.array(cfg.intrinsics.K).reshape(3, 3)
    xyz_cc, m = project_pp_depth_from_K(depth, K)
    xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
    xyz_color = get_color_array(rgb)[:, m]/255

    return np.vstack((xyz_wc, xyz_color))


def get_registered_xyz_rgb_wc_hm3d(cfg, frame):

    rgb = imread(f"{cfg.rgb_dir}/{frame}.jpg")
    depth = np.load(f"{cfg.depth_dir}/{frame}.npy")
    cam_pose = np.load(f"{cfg.poses_dir}/{frame}.npy")

    xyz_cc, m = project_pp_depth_from_hfov(depth, cfg.intrinsics.sensor_hfov)
    xyz_wc = cam_pose[:3, :] @ extend_array_to_homogeneous(xyz_cc)
    xyz_color = get_color_array(rgb)[:, m]/255

    return np.vstack((xyz_wc, xyz_color))
