import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_yaml_dict
import open_eqa_utils
import logging
import json
from pathlib import Path
from tqdm import tqdm
from open_eqa_utils.pre_process_scannet_dataset.scannet_open_eqa.SensorData import SensorData
from open_eqa_utils.utils import create_scene_dir


def save_intrinsics_file_scannet(intrinsic_fn, sd: SensorData):
    k = sd.intrinsic_depth[:3, :3].flatten()
    intrinsics = dict(
        width=int(sd.depth_width),
        height=int(sd.depth_height),
        K=[float(k) for k in k]
    )
    save_yaml_dict(intrinsic_fn, intrinsics)


def get_list_scene_data(cfg):
    # map scene name -> *sens file in ScanNet
    scenes = {}
    for folder in sorted(cfg.scannet_scene_list):
        scene_name = folder.split("/")[-1].split("-")[-1]
        sens_file = Path(cfg.scannet_dirs["scans"]) / \
            scene_name / f"{scene_name}.sens"
        if sens_file.exists():
            scenes[folder] = sens_file
        else:
            sens_file = Path(
                cfg.scannet_dirs["scans_test"]) / scene_name / f"{scene_name}.sens"
            assert sens_file.exists(), f"Scene {folder} does not exist"
            scenes[folder] = sens_file
    return scenes


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.info("*" * 50)
    logging.info(f"Total number of scenes: {cfg.scannet_scene_list.__len__()}")
    [logging.info(f"Scannet dirs - {k}: {v}")
     for k, v in cfg.scannet_dirs.items()]
    logging.info(f"Process directory: {cfg.pre_proc_dir}")
    list_scene_data = get_list_scene_data(cfg)

    for scene in tqdm(list_scene_data, desc="Processing scenes"):
        logging.info(f"Processing scene: {scene}")
        scene_dir = Path(cfg.pre_proc_dir) / scene
        rgb_dir, depth_dir, poses_dir = create_scene_dir(scene_dir)

        sens_file = list_scene_data[scene]
        sd = SensorData(sens_file)

        save_intrinsics_file_scannet(scene_dir/"intrinsics.yaml", sd)
        sd.export_poses(poses_dir)
        sd.export_depth_images(depth_dir)
        sd.export_color_images(rgb_dir)


if __name__ == '__main__':
    main()
