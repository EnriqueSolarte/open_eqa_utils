import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_yaml_dict
import open_eqa_utils
import logging
import json
from pathlib import Path
from tqdm import tqdm
from open_eqa_utils.pre_process_scannet_dataset.scannet_open_eqa.SensorData import SensorData
from open_eqa_utils.utils import create_scene_dir
from open_eqa_utils.pre_process_scannet_dataset.pre_process_scannet_v0 import save_intrinsics_file_scannet


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.info("*" * 50)
    logging.info(f"Data will be saved @: {cfg.pre_proc_dir}")

    scene = cfg.scene_name

    logging.info(f"Processing scene: {scene}")
    scene_dir = Path(cfg.pre_proc_dir) / scene
    rgb_dir, depth_dir, poses_dir = create_scene_dir(scene_dir)

    sens_file = Path(cfg.scannet_dir) / scene / f"{scene}.sens"
    sd = SensorData(sens_file)

    save_intrinsics_file_scannet(scene_dir/"intrinsics.yaml", sd)
    sd.export_poses(poses_dir)
    sd.export_depth_images(depth_dir)
    sd.export_color_images(rgb_dir)


if __name__ == '__main__':
    main()
