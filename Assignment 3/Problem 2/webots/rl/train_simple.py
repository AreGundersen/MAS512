"""Train PPO in nursing_home_simple.wbt (straight-line reward).

Usage:  python train_simple.py --steps 800000 --name ppo_simple_v4
Webots: open worlds/nursing_home_simple.wbt (extern controller), reload the world, press play (fast mode).
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rosey_env_simple import RoseyNavEnv

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=800_000)
p.add_argument("--name", default="ppo_simple")
args = p.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = Monitor(RoseyNavEnv(jitter=True), filename=os.path.join("logs", args.name))
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.005,
            device="cpu", verbose=1)
model.learn(total_timesteps=args.steps,
            callback=CheckpointCallback(save_freq=10_000, save_path="models", name_prefix=args.name))
model.save(os.path.join("models", args.name))
print("saved models/" + args.name + ".zip")
