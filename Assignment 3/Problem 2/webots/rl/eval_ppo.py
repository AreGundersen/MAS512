"""Evaluer en trent modell og logg banen og slip per steg (til plottene i oppgave 4c).

Bruk:
  python eval_ppo.py --model models/ppo_rosey --episodes 10
  python eval_ppo.py --model models/ppo_rosey --start 1 0.6 0 --goal 3.5 4
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from rosey_env import RoseyNavEnv

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/ppo_rosey")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
p.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
p.add_argument("--csv", default="results/rl_eval.csv")
args = p.parse_args()

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
env = RoseyNavEnv()
model = PPO.load(args.model)
options = {}
if args.start:
    options["start"] = tuple(args.start)
if args.goal:
    options["goal"] = tuple(args.goal)

wins, slips, lengths = 0, [], []
with open(args.csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "t", "x", "y", "v_cmd", "v_enc", "v_true", "slip", "dist"])
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep, options=options)
        done, k = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            k += 1
            w.writerow([ep, round(k * env.dt_act, 3), info["x"], info["y"], info["v_cmd"],
                        info["v_enc"], info["v_true"], info["slip"], info["dist"]])
            if info["slip"] is not None:                     # None når farten er for lav til å definere slip
                slips.append(info["slip"])
            done = term or trunc
        wins += bool(info["success"])
        lengths.append(k * env.dt_act)
        print(f"episode {ep}: {'mål' if info['success'] else 'kollisjon' if info['collision'] else 'tidsavbrudd'}"
              f" etter {k * env.dt_act:.1f} s")

print(f"\nsuksess: {wins}/{args.episodes}, gjennomsnittlig tid {np.mean(lengths):.1f} s")
if slips:
    print(f"slip: snitt {np.mean(slips):+.3f}, maks {max(slips, key=abs):+.3f}")
print("logg:", args.csv)
env.close()
