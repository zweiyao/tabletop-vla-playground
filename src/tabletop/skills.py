"""Ground-truth-position skills; all motion uses physics and the OSC controller."""
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import COLORS, REGIONS, HALF_SIZE, TABLE_Z


class SkillError(RuntimeError):
    pass


class Skills:
    def __init__(self, env, stop=None, on_step=None):
        self.env = env
        self.stop = stop if stop is not None else threading.Event()
        self.on_step = on_step
        self.grip = -1.0
        self.held = None
        self.steps = 0
        self.max_steps = 650
        self.site = env.robots[0].eef_site_id["right"]
        self.orientation = self.env.sim.data.site_xmat[self.site].reshape(3, 3).copy()

    @property
    def eef(self):
        return self.env.sim.data.site_xpos[self.site].copy()

    def tick(self, target):
        if self.stop.is_set():
            raise SkillError("用户已停止")
        if self.steps >= self.max_steps:
            raise SkillError("技能超时")
        current_rot = self.env.sim.data.site_xmat[self.site].reshape(3, 3)
        rot_error = Rotation.from_matrix(self.orientation @ current_rot.T).as_rotvec()
        action = np.r_[np.clip((target - self.eef) / 0.05, -0.5, 0.5),
                       np.clip(rot_error / 0.5, -0.5, 0.5), self.grip]
        self.env.step(action)
        self.steps += 1
        if self.on_step:
            self.on_step(self.steps)

    def move(self, target, tolerance=0.003, limit=110):
        target = np.asarray(target, dtype=float)
        for _ in range(limit):
            self.tick(target)
            if np.linalg.norm(target - self.eef) < tolerance:
                return
        raise SkillError(f"未能到达目标位置，误差 {np.linalg.norm(target-self.eef):.3f}m")

    def wait(self, steps, target=None):
        target = self.eef if target is None else np.asarray(target)
        for _ in range(steps):
            self.tick(target)

    def is_grasping(self, obj):
        return bool(self.env._check_grasp(self.env.robots[0].gripper["right"], self.env.cubes[obj]))

    def pick(self, obj):
        if self.held:
            raise SkillError("夹爪已有物体，请先放下")
        pos = self.env.object_pos(obj)
        self.grip = -1
        self.move(np.r_[self.eef[:2], 1.08])
        self.move(pos + [0, 0, 0.15])
        self.move(pos + [0, 0, 0.002])
        self.grip = 1
        self.wait(25)
        self.move(pos + [0, 0, 0.15])
        self.wait(10)
        if not self.is_grasping(obj) or self.env.object_pos(obj)[2] < TABLE_Z + 0.10:
            raise SkillError("抓取失败：未检测到稳定夹持及抬升")
        self.held = obj

    def release_at(self, xyz):
        obj = self.held
        offset = self.eef - self.env.object_pos(obj)
        target = np.asarray(xyz) + offset
        self.move(np.r_[self.eef[:2], 1.10])
        self.move(np.r_[target[:2], 1.10])
        self.move(target + [0, 0, 0.004])
        self.grip = -1
        self.wait(20)
        self.move(target + [0, 0, 0.16])
        self.wait(30)
        self.held = None
        return obj

    def execute(self, skill, obj, target=None):
        self.steps = 0
        if obj not in COLORS:
            raise SkillError("不存在的积木")
        if skill == "pick":
            self.pick(obj)
            return {"success": True, "steps": self.steps, "skill": skill, "object": obj}
        if skill not in ("place", "stack"):
            raise SkillError("未知技能")
        if skill == "place" and target not in REGIONS:
            raise SkillError("未知放置区域")
        if skill == "stack" and (target not in COLORS or target == obj):
            raise SkillError("无效堆叠目标")
        if self.held and self.held != obj:
            raise SkillError("夹爪中是另一块积木")
        if not self.held:
            self.pick(obj)
        if skill == "place":
            xyz = np.r_[REGIONS[target], TABLE_Z + HALF_SIZE]
            if any(np.linalg.norm(self.env.object_pos(n)[:2] - xyz[:2]) < 0.055
                   for n in COLORS if n != obj):
                raise SkillError("目标区域被占用")
        else:
            xyz = self.env.object_pos(target) + [0, 0, 2 * HALF_SIZE]
        self.release_at(xyz)
        pos = self.env.object_pos(obj)
        success = (np.linalg.norm(pos[:2] - xyz[:2]) < 0.018
                   and abs(pos[2] - xyz[2]) < 0.008
                   and self.env.object_speed(obj) < 0.02
                   and not self.is_grasping(obj))
        if not success:
            raise SkillError("放置后的位置或稳定性检查未通过")
        return {"success": True, "steps": self.steps, "skill": skill, "object": obj, "target": target}
