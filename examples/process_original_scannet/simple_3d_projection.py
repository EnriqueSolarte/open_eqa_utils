
import os
from pathlib import Path
from imageio.v2 import imread, imwrite
import numpy as np
import cv2
from matplotlib import pyplot as plt
from geometry_perception_utils.pinhole_utils import project_pp_depth_from_K
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from geometry_perception_utils.image_utils import get_color_array
from geometry_perception_utils.vispy_utils import plot_color_plc
from tqdm import tqdm

DATA_DIR = '/media/datasets/ScanNet/ScanNet_full_version/scannet_sensor_data/scene0011_00'
RGB_DIR = f"{DATA_DIR}/color"
DEPTH_DIR = f"{DATA_DIR}/depth"
K_FN = f"{DATA_DIR}/intrinsic/intrinsic_depth.txt"
POSES_DIR = f"{DATA_DIR}/pose"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def read_txt_file(pose_fn):
    with open(pose_fn, 'r') as f:
        lines = f.readlines()
    cam_pose = np.array(
        [[float(v) for v in l.strip().split(' ')] for l in lines])
    return cam_pose


def main():
    list_frames = [f.stem for f in Path(RGB_DIR).iterdir()]
    xyz_rgb_wc = []
    for fr in tqdm(list_frames[0:-1:100]):
        rgb = imread(f"{RGB_DIR}/{fr}.jpg")
        depth_16 = imread(f"{DEPTH_DIR}/{fr}.png")
        depth = depth_16 / 1000
         # depth = (x / 10 * 65535).astype(np.uint16)
        H, W = depth.shape
        pose = read_txt_file(f"{POSES_DIR}/{fr}.txt")
        K = read_txt_file(K_FN)[:3, :3]
        rgb_image = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        xyz, m = project_pp_depth_from_K(depth, K)
        if np.sum(m) == 0:
            continue
        xyz_wc = pose[:3, :] @ extend_array_to_homogeneous(xyz)
        xyz_rgb = get_color_array(rgb_image)[:, m]/255
        xyz_rgb_wc.append(np.vstack((xyz_wc, xyz_rgb)))
        # plot_color_plc(xyz_wc.T, xyz_rgb.T)
        

    _xyz_wc = np.hstack(xyz_rgb_wc)
    plot_color_plc(_xyz_wc[:3].T, _xyz_wc[3:].T)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(rgb_image)
    # plt.subplot(122)
    # plt.imshow(depth)
    # plt.show()
    # pass


if __name__ == '__main__':
    main()
