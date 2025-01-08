# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

import os
import struct
import zlib

import cv2
import imageio
import numpy as np
import png
import logging
from tqdm import tqdm, trange

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" *
                self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" *
                self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(
                struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(
        self, output_path, frame_skip=1, num_frames=None
    ):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if num_frames is None:
            num_frames = len(self.frames)
        logging.info(
            "exporting {} of {} depth frames to {}".format(
                num_frames, len(self.frames) // frame_skip, output_path
            )
        )
        # 2025-01-06: This code is from original OpenQEA code. It is not used in the current version.
        # for i, f in enumerate(range(0, len(self.frames), frame_skip)):
        #     if i + 1 > num_frames:
        #         break
        #     output_filename = os.path.join(output_path, f"{f:06d}-depth.png")
        #     if os.path.exists(output_filename):
        #         continue  # skip existing
        #     if image_size is not None:
        #         depth = cv2.resize(
        #             depth,
        #             (image_size[1], image_size[0]),
        #             interpolation=cv2.INTER_NEAREST,
        #         )
        #     with open(output_filename, "wb") as f:  # write 16-bit
        #         writer = png.Writer(
        #             width=depth.shape[1], height=depth.shape[0], bitdepth=16
        #         )
        #         depth = depth.reshape(-1, depth.shape[1]).tolist()
        #         writer.write(f, depth)
        for f in trange(0, num_frames, frame_skip, desc="Exporting depth maps.."):
            _, ret = self.get_and_check_camera_pose(f)
            if not ret:
                continue
            output_filename = os.path.join(output_path, f"{f:06d}.npy")
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            np.save(output_filename, depth/self.depth_shift)

    def export_color_images(
        self, output_path, image_size=None, frame_skip=1, num_frames=None
    ):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if num_frames is None:
            num_frames = len(self.frames)
        logging.info(
            "exporting {} of {} color frames to {}".format(
                num_frames, len(self.frames) // frame_skip, output_path
            )
        )
        # for i, f in enumerate(range(0, len(self.frames), frame_skip)):
        #     if i + 1 > num_frames:
        #         break
        #     output_filename = os.path.join(output_path, f"{f:06d}-rgb.png")
        #     if os.path.exists(output_filename):
        #         continue  # skip existing
        for f in trange(0, num_frames, frame_skip, desc="Exporting RGB images..."):
            _, ret = self.get_and_check_camera_pose(f)
            if not ret:
                continue
            output_filename = os.path.join(output_path, f"{f:06d}.jpg")
            color = self.frames[f].decompress_color(
                self.color_compression_type)
            if image_size is not None:
                _image_size = (image_size[1], image_size[0])
            else:
                # scaling to depth image size for direct mapping in 3D
                _image_size = (self.depth_height, self.depth_width)
            color = cv2.resize(
                color,
                (_image_size[1], _image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            imageio.imwrite(output_filename, color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def get_and_check_camera_pose(self, frame_idx):
        cam_pose = self.frames[frame_idx].camera_to_world
        if np.isnan(cam_pose).any() or np.isinf(cam_pose).any():
            return None, False
        return cam_pose, True

    def export_poses(self, output_path, frame_skip=1, num_frames=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if num_frames is None:
            num_frames = len(self.frames)
        logging.info(
            "exporting {} of {} camera poses to {}".format(
                num_frames, len(self.frames) // frame_skip, output_path
            )
        )
        # for i, f in enumerate(range(0, len(self.frames), frame_skip)):
        #     if i + 1 > num_frames:
        #         break
        # if os.path.exists(output_filename):
        #     continue  # skip existing
        # self.save_mat_to_file(
        #     self.frames[f].camera_to_world, output_filename)
        for f in trange(0, num_frames, frame_skip, desc="Exporting RGB images..."):
            output_filename = os.path.join(output_path, f"{f:06d}.npy")
            cam_pose, ret = self.get_and_check_camera_pose(f)
            if not ret:
                logging.warning(
                    f"Found nan or inf values in the camera pose for frame {f}. Skipping..."
                )
                continue
            np.save(output_filename, cam_pose)

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(
                output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(
                output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(
                output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(
                output_path, "extrinsic_depth.txt")
        )
