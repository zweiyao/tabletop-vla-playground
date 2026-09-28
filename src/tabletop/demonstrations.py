"""Deterministic task setup and physics-only demonstration / success criteria."""
import numpy as np
from scipy.spatial.transform import Rotation
from .skills import Skills

TASKS = {"red": "pick up the red cube", "green": "pick up the green cube",
         "blue": "pick up the blue cube", "home": "return to the initial end effector position and orientation with the gripper open"}


def prepare(env, seed, task):
    env.reset_seed(seed)
    skills = Skills(env)
    skills.wait(15)
    home = {"position": skills.eef.copy(), "orientation": skills.orientation.copy()}
    rng = np.random.default_rng(seed + 700000)
    # Home always starts away from home; half the pick demonstrations also vary the start.
    if task == "home" or rng.random() < 0.5:
        skills.move(np.r_[skills.eef[:2], 1.10])
        skills.orientation = Rotation.from_rotvec(rng.uniform([-0.12, -0.12, -0.5], [0.12, 0.12, 0.5])).as_matrix() @ home["orientation"]
        skills.move(rng.uniform([-0.13, -0.22, 0.93], [0.13, 0.22, 1.13]))
        skills.wait(10)
    return home


def demonstrate(env, task, home):
    skills = Skills(env)
    skills.orientation = home["orientation"].copy()
    if task == "home":
        skills.move(np.r_[skills.eef[:2], 1.10])
        skills.move(np.r_[home["position"][:2], 1.10])
        skills.move(home["position"])
        skills.wait(25)
    else:
        skills.pick(task)
        skills.wait(20)


def success(env, task, home):
    obs = env._get_observations(force_update=True)
    if task == "home":
        site = env.robots[0].eef_site_id["right"]
        rotation = env.sim.data.site_xmat[site].reshape(3, 3)
        distance = np.linalg.norm(obs["robot0_eef_pos"] - home["position"])
        angle = Rotation.from_matrix(rotation @ home["orientation"].T).magnitude()
        return bool(distance < 0.015 and angle < 0.10 and np.diff(obs["robot0_gripper_qpos"])[0] < -0.06)
    return bool(env._check_grasp(env.robots[0].gripper["right"], env.cubes[task])
                and env.object_pos(task)[2] > 0.90 and env.object_speed(task) < 0.04)
