"""Tren PPO på Rosey. Samme hyperparametre som i foreleserens notebook.

Bruk: python train_ppo.py --steps 100000
Start Webots med worlds/ugv_rl_train.wbt først.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rosey_env import RoseyNavEnv

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--name", default="ppo_rosey")
p.add_argument("--check", action="store_true", help="kjør SB3 sin check_env først")
args = p.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = RoseyNavEnv()
if args.check:
    check_env(env)
env = Monitor(env, filename=os.path.join("logs", args.name))     # logger belønning og lengde per episode

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)
model.learn(
    total_timesteps=args.steps,
    callback=CheckpointCallback(save_freq=10_000, save_path="models", name_prefix=args.name),
)
model.save(os.path.join("models", args.name))
print("lagret models/" + args.name + ".zip")
