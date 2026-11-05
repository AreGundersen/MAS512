"""Første test: kobler til Webots, kjører tilfeldige handlinger og måler hastighet.

Krever ikke torch eller stable-baselines3. Start Webots med worlds/ugv_rl_train.wbt først.
"""
import time

import numpy as np

from rosey_env import RoseyNavEnv

env = RoseyNavEnv()
print("observasjon:", env.observation_space.shape, " handling:", env.action_space.shape)

try:
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    print("gymnasium check_env: OK")
except Exception as e:                      # advarsler skal ikke stoppe testen
    print("gymnasium check_env:", repr(e))

t0, total_steps, ep = time.time(), 0, 0
while total_steps < 400:
    obs, _ = env.reset(seed=ep)
    done, ret, n = False, 0.0, 0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ret, n, done = ret + r, n + 1, term or trunc
    total_steps += n
    why = "mål" if info["success"] else "kollisjon" if info["collision"] else "tidsavbrudd"
    print(f"episode {ep}: {n} steg, belønning {ret:6.1f}, slutt: {why}, avstand {info['dist']:.2f} m")
    ep += 1

rate = total_steps / (time.time() - t0)
print(f"\n{rate:.1f} RL-steg per sekund")
print(f"100 000 steg tar ca. {100_000 / rate / 3600:.1f} timer, 300 000 steg ca. {300_000 / rate / 3600:.1f} timer")
env.close()
