import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory
import utils_open_eqa
from tqdm import tqdm
import os
import habitat_sim
from pathlib import Path
import pickle
from utils_open_eqa.pre_process_hm3d_dataset.orig_open_eqa.config import make_cfg
from geometry_perception_utils.geometry_utils import eulerAnglesToRotationMatrix
import logging
from imageio.v2 import imwrite
import numpy as np
from pyquaternion import Quaternion


os.environ['GLOG_minloglevel'] = "3"
os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

default_settings = {
    "sensor_hfov": 90,
    "sensor_width": 1920,
    "sensor_height": 1080,
}


def get_cam_pose(translation, quaternion):
    """
    Returns the camera pose given the agent states.
    """
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = translation
    q = Quaternion(x=quaternion.x, y=quaternion.y,
                   z=quaternion.z, w=quaternion.w)
    cam_pose[:3, :3] = q.rotation_matrix
    return cam_pose


def load_habitat_sim(single_agent_state, hm3d_dir) -> habitat_sim.Simulator:
    data = pickle.load(single_agent_state.open("rb"))
    # due to different paths and error in some scenes pickled files
    glb_file = data['scene_id'].split('val')[-1]
    scene_id = f"{hm3d_dir}/{glb_file}"

    assert os.path.exists(scene_id), f"Scene {scene_id} does not exist"

    agent_state = data["agent_state"]
    sensor_position = (
        agent_state.sensor_states["rgb"].position[1] - agent_state.position[1]
    )
    settings = {
        "scene_id": scene_id,
        "sensor_hfov": 90,
        "sensor_width": 1920,
        "sensor_height": 1080,
        "sensor_position": sensor_position,
    }
    cfg = make_cfg(settings | default_settings)
    return habitat_sim.Simulator(cfg)


def get_sensor_wc():
    sensor_wc = np.eye(4)
    R = eulerAnglesToRotationMatrix((np.deg2rad(180), 0, 0))
    sensor_wc[:3, :3] = R
    return sensor_wc


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.info("*" * 50)
    logging.info(f"Total number of scenes: {cfg.hm3d_scene_list.__len__()}")
    logging.info(f"Agent States dir: {cfg.states_dir}")
    logging.info(f"Process directory: {cfg.pre_proc_dir}")
    list_scene_dir = [f"{cfg.states_dir}/{s}" for s in cfg.hm3d_scene_list]

    sensor_wc = get_sensor_wc()

    for raw_scene_dir in tqdm(list_scene_dir, desc="Processing scenes"):
        list_states = [
            f"{raw_scene_dir}/{s}" for s in os.listdir(raw_scene_dir) if s.endswith(".pkl")]

        # load habitat sim from cfg in one of the agent states
        single_agent_state = Path(cfg.states_dir).rglob("*.pkl").__next__()
        sim = load_habitat_sim(single_agent_state, cfg.hm3d_dir)

        logging.info(
            f"Processing scene: {Path(raw_scene_dir).stem} - agent positions: {list_states.__len__()}")

        data_dir = create_directory(
            f"{cfg.pre_proc_dir}/{Path(raw_scene_dir).stem}")
        rgb_dir = create_directory(f"{data_dir}/rgb")
        depth_dir = create_directory(f"{data_dir}/depth")
        semantic_dir = create_directory(f"{data_dir}/semantic")
        poses_dir = create_directory(f"{data_dir}/poses")

        for idx, path in tqdm(enumerate(list_states), desc="Processing agent states"):
            # set agent state
            data = pickle.load(Path(path).open("rb"))
            agent = sim.get_agent(0)
            agent.set_state(data["agent_state"])

            # save render data
            obs = sim.get_sensor_observations()
            frame = Path(path).stem

            # Save RGB
            imwrite(f"{rgb_dir}/{frame}.jpg",
                    obs["rgb"][:, :, :3])

            if cfg.get("rgb_only", False):
                continue

            np.save(f"{semantic_dir}/{frame}.npy", obs['semantic'])
            np.save(f"{depth_dir}/{frame}.npy", obs['depth'])

            # Save camera pose
            cam_pose_fn = f"{poses_dir}/{frame}.npy"
            t = sim.agents[0].get_state().position
            q = sim.agents[0].get_state().rotation
            cam_pose = get_cam_pose(t, q) @ sensor_wc
            if idx == 0:
                # The very first frame is the WC
                cam_pose_0 = cam_pose
            cam_pose = np.linalg.inv(cam_pose_0) @ cam_pose
            np.save(cam_pose_fn, cam_pose)

        sim.close()


if __name__ == "__main__":
    main()
