"""Gymnasium-miljø for Rosey i Webots.

Kjøres som extern controller: Rosey sin controller i world-filen er satt til <extern>,
og skriptene smoke_test.py / train_ppo.py / eval_ppo.py kobler seg til Webots.

Oppgave: kjør fra tilfeldig startpose til et tilfeldig mål uten å treffe hindringer.
Observasjon (41 tall): 36 lidar-sektorer, avstand til mål, sin/cos av retning til mål,
forrige v- og w-kommando. Handling: [v, w] i [-1, 1].
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from controller import Supervisor
except ImportError as e:
    raise ImportError(
        "Fant ikke Webots sin 'controller'-modul. Sett WEBOTS_HOME og legg "
        "$WEBOTS_HOME/lib/controller/python i PYTHONPATH (se Using Python i Webots-dokumentasjonen)."
    ) from e

# --- Rosey (samme verdier som i slip_controller.py) ---
R_W = 0.0625                    # hjulradius [m]
B = 0.2529                      # hjulavstand [m]
LIDAR_X = 0.19731               # lidar-posisjon langs robotens x-akse [m]
V_MAX = 0.8                     # toppfart [m/s]
W_MAX = 2.0                     # maks svingrate [rad/s]
ACC = 3.0                       # maks fartsendring [m/s^2]
V_MIN = 0.2                     # slip er udefinert under denne farten [m/s]
BODY_X, BODY_Y = 0.26, 0.38     # lidar-treff innenfor dette er roboten selv

# --- observasjon ---
N_SECTORS = 36
RANGE_MAX = 5.0                 # lidar-avstander klippes her [m]
GOAL_SCALE = 10.0               # avstand til mål normaliseres med denne [m]
COLLISION_RANGE = 0.15          # reserve hvis kontaktpunkter ikke er tilgjengelige [m]

# --- belønning ---
K_PROGRESS = 10.0               # per meter nærmere målet
STEP_COST = 0.01                # per steg
R_GOAL = 20.0
R_COLLISION = -10.0

# Arena i ugv_slip_baseline.wbt: vegger ved +-5 m og fire hindringer som sirkler (x, y, radius)
BASELINE_ARENA = {
    "half": 4.95,
    "obstacles": [(3.0, 2.5, 0.45), (-2.5, -3.0, 0.70), (-3.0, 2.5, 0.25), (2.0, -2.5, 0.25)],
}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class RoseyNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, steps_per_action=8, max_episode_s=40.0, goal_radius=0.6,
                 arena=BASELINE_ARENA, fast=True):
        super().__init__()
        self.robot = Supervisor()
        self.node = self.robot.getSelf()
        self.dt = int(self.robot.getBasicTimeStep())            # [ms]
        self.n = steps_per_action
        self.dt_act = self.n * self.dt / 1000.0                 # tid per RL-steg [s]
        self.max_steps = int(max_episode_s / self.dt_act)
        self.goal_radius = goal_radius
        self.arena = arena
        if fast:
            self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        self.motor_l = self.robot.getDevice("left_wheel_motor")
        self.motor_r = self.robot.getDevice("right_wheel_motor")
        self.motor_l.setPosition(float("inf"))
        self.motor_r.setPosition(float("inf"))
        self.enc_l = self.robot.getDevice("left_wheel_sensor")
        self.enc_r = self.robot.getDevice("right_wheel_sensor")
        self.enc_l.enable(self.dt)
        self.enc_r.enable(self.dt)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.n * self.dt)                     # oppdateres én gang per RL-steg
        n_rays = self.lidar.getHorizontalResolution()
        ang = math.pi - np.arange(n_rays) * 2 * math.pi / n_rays   # stråle 0 peker bakover, så med klokka
        self._cos, self._sin = np.cos(ang), np.sin(ang)
        self._sector = (((ang + math.pi) % (2 * math.pi)) / (2 * math.pi) * N_SECTORS).astype(int) % N_SECTORS

        try:
            self.node.enableContactPointsTracking(self.dt, False)
            self._contacts = True
        except Exception:
            self._contacts = False

        self.goal_node = self.robot.getFromDef("GOAL")           # valgfritt: flyttes til målet hvis den finnes
        self.goal = (0.0, 0.0)
        self.v_cmd = self.w_cmd = 0.0
        self.steps = 0
        self.d_prev = 0.0

        low = np.concatenate([np.zeros(N_SECTORS), [0, -1, -1, 0, -1]]).astype(np.float32)
        high = np.concatenate([np.ones(N_SECTORS), [1, 1, 1, 1, 1]]).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ---------- hjelpefunksjoner ----------
    def _yaw(self):
        R = self.node.getOrientation()
        return math.atan2(R[3], R[0])

    def _set_wheels(self, v, w):
        self.motor_l.setVelocity((v - w * B / 2) / R_W)
        self.motor_r.setVelocity((v + w * B / 2) / R_W)

    def _free(self, x, y, margin):
        h = self.arena["half"] - margin
        if abs(x) > h or abs(y) > h:
            return False
        return all(math.hypot(x - ox, y - oy) > r + margin for ox, oy, r in self.arena["obstacles"])

    def _sample_xy(self, margin=0.5):
        for _ in range(1000):
            x, y = self.np_random.uniform(-self.arena["half"], self.arena["half"], size=2)
            if self._free(x, y, margin):
                return float(x), float(y)
        raise RuntimeError("Fant ingen ledig posisjon i arenaen")

    def _advance(self, v_tgt, w_tgt, measure=False):
        """Kjør n Webots-steg med fartsramp. Gir enkoder- og sann fart hvis measure=True."""
        if measure:
            p0, yaw0 = self.node.getPosition(), self._yaw()
            l0, r0 = self.enc_l.getValue(), self.enc_r.getValue()
        step = ACC * self.dt / 1000.0
        for _ in range(self.n):
            self.v_cmd += max(-step, min(step, v_tgt - self.v_cmd))
            self._set_wheels(self.v_cmd, w_tgt)
            if self.robot.step(self.dt) == -1:
                raise RuntimeError("Webots avsluttet simuleringen")
        self.w_cmd = w_tgt
        if not measure:
            return {}
        p1, yaw1 = self.node.getPosition(), self._yaw()
        ymid = yaw0 + wrap(yaw1 - yaw0) / 2
        v_true = ((p1[0] - p0[0]) * math.cos(ymid) + (p1[1] - p0[1]) * math.sin(ymid)) / self.dt_act
        v_enc = R_W * ((self.enc_l.getValue() - l0) + (self.enc_r.getValue() - r0)) / 2 / self.dt_act
        slip = (v_enc - v_true) / v_true if abs(v_true) > V_MIN else None   # None når slip er udefinert
        # rundet til 6 desimaler: fjerner avrundingsstøy fra akkumulerte enkoderverdier
        return {"v_enc": round(float(v_enc), 6), "v_true": round(float(v_true), 6),
                "slip": None if slip is None else round(float(slip), 6)}

    def _observe(self):
        r = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        r = np.minimum(np.where(np.isfinite(r) & (r >= 0.1), r, RANGE_MAX), RANGE_MAX)
        x, y = LIDAR_X + r * self._cos, r * self._sin
        r = np.where((np.abs(x) < BODY_X) & (np.abs(y) < BODY_Y), RANGE_MAX, r)   # fjern roboten selv
        sectors = np.full(N_SECTORS, RANGE_MAX)
        np.minimum.at(sectors, self._sector, r)

        pos, yaw = self.node.getPosition(), self._yaw()
        dx, dy = self.goal[0] - pos[0], self.goal[1] - pos[1]
        d = math.hypot(dx, dy)
        bearing = wrap(math.atan2(dy, dx) - yaw)
        obs = np.concatenate([
            sectors / RANGE_MAX,
            [min(d, GOAL_SCALE) / GOAL_SCALE, math.sin(bearing), math.cos(bearing),
             self.v_cmd / V_MAX, self.w_cmd / W_MAX],
        ])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
        return obs, d, float(r.min()), (pos[0], pos[1])

    def _collided(self, min_range):
        if self._contacts:
            return len(self.node.getContactPoints(False)) > 0   # chassiset ligger over gulvet: all kontakt er kollisjon
        return min_range < COLLISION_RANGE

    # ---------- Gymnasium ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        sx, sy = options["start"][:2] if "start" in options else self._sample_xy()
        yaw = options["start"][2] if "start" in options and len(options["start"]) > 2 \
            else float(self.np_random.uniform(-math.pi, math.pi))
        if "goal" in options:
            gx, gy = options["goal"]
        else:
            while True:
                gx, gy = self._sample_xy()
                if math.hypot(gx - sx, gy - sy) > 2.0:
                    break

        self.node.getField("translation").setSFVec3f([sx, sy, R_W])
        self.node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, yaw])
        self.node.resetPhysics()
        if self.goal_node is not None:
            self.goal_node.getField("translation").setSFVec3f([gx, gy, 0.5])
        self.goal = (gx, gy)
        self.v_cmd = self.w_cmd = 0.0
        self.steps = 0
        self._set_wheels(0.0, 0.0)
        self._advance(0.0, 0.0)                    # ett RL-steg uten kommando, så lidar og posisjon oppdateres
        obs, self.d_prev, _, _ = self._observe()
        return obs, {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_tgt = V_MAX * (a[0] + 1.0) / 2.0         # kun forover
        w_tgt = W_MAX * a[1]
        meas = self._advance(v_tgt, w_tgt, measure=True)
        self.steps += 1

        obs, d, min_range, xy = self._observe()
        reward = K_PROGRESS * (self.d_prev - d) - STEP_COST
        self.d_prev = d

        success = d < self.goal_radius
        collided = (not success) and self._collided(min_range)
        terminated = success or collided
        truncated = (not terminated) and self.steps >= self.max_steps
        if success:
            reward += R_GOAL
        elif collided:
            reward += R_COLLISION

        info = {"success": success, "collision": collided, "dist": round(float(d), 6),
                "x": round(float(xy[0]), 6), "y": round(float(xy[1]), 6),
                "v_cmd": round(float(self.v_cmd), 6), **meas}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._set_wheels(0.0, 0.0)
