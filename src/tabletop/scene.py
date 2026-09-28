"""Three cubes and a Panda. Object state is private to skills and evaluation."""
import os
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

COLORS = {"red": [1, 0.03, 0.03, 1], "green": [0.03, 0.8, 0.03, 1], "blue": [0.03, 0.12, 1, 1]}
REGIONS = {"left": [0.0, -0.23], "center": [0.0, 0.0], "right": [0.0, 0.23]}
HALF_SIZE = 0.02
TABLE_Z = 0.8


class TabletopEnv(ManipulationEnv):
    def __init__(self, seed=0):
        config = load_composite_controller_config(controller="BASIC")
        config["body_parts"]["right"]["type"] = "OSC_POSE"
        super().__init__(
            robots="Panda", controller_configs=config, initialization_noise=None,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
            control_freq=20, horizon=10000, ignore_done=True, hard_reset=False,
            render_gpu_device_id=int(os.environ["MUJOCO_EGL_DEVICE_ID"]), seed=seed,
        )

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](0.8))
        arena = TableArena(table_full_size=(0.8, 0.8, 0.05), table_offset=(0, 0, TABLE_Z))
        arena.set_origin([0, 0, 0])
        self.cubes = {name: BoxObject(name=name, size=[HALF_SIZE] * 3,
                      rgba=rgba, density=500, friction=[1, 0.005, 0.0001])
                      for name, rgba in COLORS.items()}
        self.sampler = UniformRandomSampler(
            name="cubes", mujoco_objects=list(self.cubes.values()),
            x_range=[-0.10, 0.10], y_range=[-0.16, 0.16], rotation=0,
            ensure_object_boundary_in_range=True, ensure_valid_placement=True,
            reference_pos=(0, 0, TABLE_Z), z_offset=0.005, rng=self.rng,
        )
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot],
                                      mujoco_objects=list(self.cubes.values()))

    def _setup_references(self):
        super()._setup_references()
        self.body_ids = {name: self.sim.model.body_name2id(obj.root_body)
                         for name, obj in self.cubes.items()}

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for pos, quat, obj in self.sampler.sample().values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.r_[pos, quat])

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def object_pos(self, name):
        return self.sim.data.body_xpos[self.body_ids[name]].copy()

    def object_speed(self, name):
        return float(np.linalg.norm(self.sim.data.get_body_xvelp(self.cubes[name].root_body)))

    def images(self, size=512):
        return {name: self.sim.render(width=size, height=size, camera_name=camera)[::-1].copy()
                for name, camera in {"front": "frontview", "wrist": "robot0_eye_in_hand"}.items()}

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(int(seed))
        self.sampler.rng = self.rng
        return self.reset()

    def pi05_observation(self):
        obs = self._get_observations()
        quat = np.asarray(obs["robot0_eef_quat"], dtype=float).copy()  # xyzw, world frame
        w = np.clip(quat[3], -1, 1)
        denominator = np.sqrt(1 - w * w)
        axis_angle = np.zeros(3) if np.isclose(denominator, 0) else quat[:3] * (2 * np.arccos(w) / denominator)
        # Match RLinf's LIBERO evaluator: raw MuJoCo frames rotated by 180 degrees.
        def camera(name):
            return np.ascontiguousarray(self.sim.render(width=256, height=256, camera_name=name)[::-1, ::-1])
        return {"image": camera("agentview"), "wrist_image": camera("robot0_eye_in_hand"),
                "state": np.concatenate([obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]]).astype(np.float32)}
