import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor

# robot
R_W = 0.0625                    # wheel radius [m]
B = 0.2529                      # wheel base [m]
V_MAX = 0.8                     
W_MAX = 2.0                     
ACC = 3.0                       

# observation
SECTOR_DEG = 10                 # each lidar sector covers this many degrees
RANGE_MAX = 5.0                 # lidar max range [m]
GOAL_SCALE = 20.0               

# reward
K_PROGRESS = 10.0               # reward getting closer tpo the goal
STEP_COST = 0.02                # small penatly for each step
R_GOAL = 100.0                  # goal reward
R_COLLISION = -50.0             # penalty for collision
R_NEAR = 0.3                    # penalty for being too close to something
NEAR_DIST = 0.4                 # penalty starts when the nearest lidar hit is closer than this [m]
R_STALL = -20.0                 # penalty if the robot has not got closer to the goal within STALL_S seconds
STALL_S = 20.0

START = (1.0, -6.5, 0.0)        # x, y, yaw (facing east)
GOAL = (19.5, -6.5)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

class RoseyNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, steps_per_action=8, max_episode_s=90.0, goal_radius=0.6, fast=True, jitter=False):
        super().__init__()
        self.robot = Supervisor()
        self.node = self.robot.getSelf()
        self.dt = int(self.robot.getBasicTimeStep())
        self.n = steps_per_action
        self.dt_act = self.n * self.dt / 1000.0                  # one RL step = 0.064 s
        self.max_steps = int(max_episode_s / self.dt_act)
        self.goal_radius = goal_radius
        self.jitter = jitter                                     
        if fast:
            self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self.motor_l = self.robot.getDevice("left_wheel_motor")
        self.motor_r = self.robot.getDevice("right_wheel_motor")
        self.motor_l.setPosition(float("inf"))
        self.motor_r.setPosition(float("inf"))

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.n * self.dt)
        n_rays = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        self.n_sectors = max(1, round(math.degrees(fov) / SECTOR_DEG))   # 17 sectors at 170 degrees
        self._sector = np.arange(n_rays) * self.n_sectors // n_rays

        self.node.enableContactPointsTracking(self.dt, False)    # collision = contact point
        self.goal_node = self.robot.getFromDef("GOAL")           
        self.goal = GOAL
        self.v_cmd = self.w_cmd = 0.0
        self.steps = 0
        self.d_best = 0.0
        self.last_best_step = 0

        low = np.concatenate([np.zeros(self.n_sectors), [0, -1, -1, 0, -1]]).astype(np.float32)
        high = np.concatenate([np.ones(self.n_sectors), [1, 1, 1, 1, 1]]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _yaw(self):
        R = self.node.getOrientation()
        return math.atan2(R[3], R[0])

    def _set_wheels(self, v, w):
        self.motor_l.setVelocity((v - w * B / 2) / R_W)
        self.motor_r.setVelocity((v + w * B / 2) / R_W)

    def _advance(self, v_tgt, w_tgt):
        step = ACC * self.dt / 1000.0
        for _ in range(self.n):
            self.v_cmd += max(-step, min(step, v_tgt - self.v_cmd))
            self._set_wheels(self.v_cmd, w_tgt)
            if self.robot.step(self.dt) == -1:
                raise RuntimeError("Webots ended the simulation")
        self.w_cmd = w_tgt

    def _observe(self):
        r = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        r = np.minimum(np.where(np.isfinite(r) & (r >= 0.1), r, RANGE_MAX), RANGE_MAX)
        sectors = np.full(self.n_sectors, RANGE_MAX)
        np.minimum.at(sectors, self._sector, r)                  # nearest hit in each sector

        pos, yaw = self.node.getPosition(), self._yaw()
        d = math.hypot(self.goal[0] - pos[0], self.goal[1] - pos[1])
        bearing = wrap(math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0]) - yaw)
        obs = np.concatenate([
            sectors / RANGE_MAX,
            [min(d, GOAL_SCALE) / GOAL_SCALE, math.sin(bearing), math.cos(bearing),
             self.v_cmd / V_MAX, self.w_cmd / W_MAX],
        ])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
        return obs, d, (pos[0], pos[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        sx, sy, yaw = options.get("start", START)
        if self.jitter and "start" not in options:
            sx += self.np_random.uniform(-0.3, 0.3)
            sy += self.np_random.uniform(-0.5, 0.5)
            yaw += self.np_random.uniform(-0.26, 0.26)
        self.node.getField("translation").setSFVec3f([sx, sy, R_W])
        self.node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, yaw])
        self.node.resetPhysics()
        if self.goal_node is not None:
            self.goal_node.getField("translation").setSFVec3f([GOAL[0], GOAL[1], 0.5])
        self.v_cmd = self.w_cmd = 0.0
        self.steps = 0
        self._set_wheels(0.0, 0.0)
        self._advance(0.0, 0.0)
        obs, self.d_best, _ = self._observe()
        self.last_best_step = 0
        return obs, {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._advance(V_MAX * (a[0] + 1.0) / 2.0, W_MAX * a[1])   # forward only
        self.steps += 1

        obs, d, xy = self._observe()
        reward = -STEP_COST
        if d < self.d_best - 0.01:                               # reward only a new record, so detours around obstacles cost little
            reward += K_PROGRESS * (self.d_best - d)
            self.d_best = d
            self.last_best_step = self.steps
        r_min = float(obs[:self.n_sectors].min()) * RANGE_MAX
        if r_min < NEAR_DIST:                                    # too close: penalty grows the closer it gets, teaches keeping distance
            reward -= R_NEAR * (NEAR_DIST - r_min) / NEAR_DIST
        stalled = (self.steps - self.last_best_step) * self.dt_act > STALL_S

        success = d < self.goal_radius
        collided = (not success) and len(self.node.getContactPoints(False)) > 0
        terminated = success or collided or stalled
        truncated = (not terminated) and self.steps >= self.max_steps
        if success:
            reward += R_GOAL
        elif collided:
            reward += R_COLLISION
        elif stalled:
            reward += R_STALL

        info = {"success": success, "collision": collided, "stalled": stalled,
                "x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4)}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._set_wheels(0.0, 0.0)
