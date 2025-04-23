import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from train import SingleLegEnv

env = SingleLegEnv()
model = PPO.load("ppo_single_leg")

model_data = mujoco.MjData(env.model)

with mujoco.viewer.launch_passive(env.model, model_data) as viewer:
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        viewer.sync()
        time.sleep(0.01)
        if done:
            obs, _ = env.reset()