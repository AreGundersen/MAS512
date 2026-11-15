import argparse
import csv
import math
import os
import re
import numpy as np
from stable_baselines3 import PPO
from rosey_env_simple import RoseyNavEnv, GOAL

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/ppo_simple_v4")
p.add_argument("--world", default="../worlds/nursing_home_simple.wbt")
args = p.parse_args()

# the 5 standard tests: (x, y, yaw) in the world frame; y spans -7.0 ... -5.8, yaw is kept small (<= 0.15 rad = 9 deg)
TESTS = [(1.0, -6.5, 0.0), (1.0, -7.5, 0.5), (1.3, -6.8, 0.10), (0.8, -5.8, -0.10), (1.5, -6.0, 0.15)]

OUT = "results/simple_trajectories"

os.makedirs("results", exist_ok=True)
env = RoseyNavEnv()
model = PPO.load(args.model, device="cpu")
paths, rows = [], []
for i, start in enumerate(TESTS, 1):
    obs, _ = env.reset(options={"start": start})
    xy, done, k = [(start[0], start[1])], False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        xy.append((info["x"], info["y"]))
        k += 1
        done = term or trunc
    a = np.array(xy)
    length = float(np.sum(np.hypot(*np.diff(a, axis=0).T)))
    res = "goal" if info["success"] else "collision" if info["collision"] else "stalled" if info.get("stalled") else "timeout"
    print(f"test {i}: {res}, time {k * env.dt_act:.1f} s, length {length:.1f} m, start {start[0]:.2f},{start[1]:.2f}")
    paths.append((a, info["success"]))
    rows.append([i, round(start[0], 2), round(start[1], 2), round(start[2], 2), res, round(k * env.dt_act, 1), round(length, 1)])
env.close()

with open(OUT + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "start_x", "start_y", "start_yaw", "result", "time_s", "length_m"])
    w.writerows(rows)
print(f"\nsuccess: {sum(s for _, s in paths)}/{len(TESTS)}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# footprint (width, depth) for PROTOs without a size field
DEFAULT_SIZE = {"HospitalBed": (2.15, 1.05), "Chair": (0.6, 0.6), "Sofa": (1.95, 1.0), "Armchair": (0.95, 0.95),
                "PottedTree": (0.65, 0.65), "WashingMachine": (0.7, 0.7), "Table": (1.9, 0.95)}


def world_rects(path):
    """Read walls and furniture from the scenery group of the .wbt file as (type, x, y, width, depth, yaw)."""
    lines = open(path, encoding="utf-8").read().split("\n")
    a = next(k for k, l in enumerate(lines) if l.startswith("DEF scenery Group"))
    b = next(k for k, l in enumerate(lines) if l.startswith("DEF ROSEY"))
    lines, out, i = lines[a:b], [], 0
    while i < len(lines):
        m = re.match(r"^    (\w+) \{$", lines[i])                 # nodes directly under the scenery group
        if m:
            j = i + 1
            while lines[j] != "    }":
                j += 1
            body = lines[i + 1:j]

            def field(key):
                for l in body:
                    mm = re.match(rf"^      {key} (.+)$", l)
                    if mm:
                        return [float(v) for v in mm.group(1).split()]
            t, rot, size = field("translation"), field("rotation"), field("size")
            typ = m.group(1)
            if t is not None and typ not in ("Solid", "Shape"):    # skip floor and goal pole
                w_, d_ = (size[0], size[1]) if size else DEFAULT_SIZE[typ]
                out.append((typ, t[0], t[1], w_, d_, rot[3] * (1 if rot[2] >= 0 else -1) if rot else 0.0))
            i = j
        i += 1
    return out


fig, ax = plt.subplots(figsize=(13, 6.5))
try:                                                             # draw walls and furniture from the world file
    for typ, cx, cy, w_, d_, yaw in world_rects(args.world):
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(cx + u * c - v * s, cy + u * s + v * c) for u, v in
               ((-w_ / 2, -d_ / 2), (w_ / 2, -d_ / 2), (w_ / 2, d_ / 2), (-w_ / 2, d_ / 2))]
        ax.add_patch(Polygon(pts, closed=True, fc="0.35" if typ in ("Wall", "Window") else "0.7", ec="k", lw=0.5))
except Exception as e:
    print("could not draw the floor plan:", e)
for i, (a, ok) in enumerate(paths, 1):
    ax.plot(a[:, 0], a[:, 1], lw=2, label=f"test {i}" + ("" if ok else " (failed)"))
    ax.plot(*a[0], "o", ms=6, color=ax.lines[-1].get_color())
ax.plot(*GOAL, "r*", ms=18, mec="k", label="goal")
ax.set_xlim(-0.5, 22.5)
ax.set_ylim(-10.5, 0.5)
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"{len(TESTS)} test run(s) of the trained PPO model (dots = start positions)")
ax.legend(loc="lower right", ncol=3)
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print("saved " + OUT + ".png")
