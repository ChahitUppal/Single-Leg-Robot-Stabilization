import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from train import SingleLegEnv

env = SingleLegEnv()
model = PPO.load("ppo_single_leg")

model_data = mujoco.MjData(env.model)

with mujoco.viewer.launch_passive(env.model, model_data) as viewer:
    obs, _ = env.reset()
    step_count = 0

    while viewer.is_running():
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        # Give 1s for the robot to stabilize before first push
        if 100 < step_count < 150:
            force = np.array([15.0, 0.0, 0.0])  # gentler but visible lateral push
            mujoco.mj_applyFT(
                env.model,
                env.data,
                force.reshape(3, 1),
                np.zeros((3, 1)),
                env.data.xpos[env.model.body('torso').id],
                env.model.body('torso').id,
                env.data.qfrc_applied
            )
            print(f"Applied push: {force}")

        mujoco.mj_step(env.model, env.data)
        viewer.sync()
        time.sleep(0.01)
        step_count += 1

        if done:
            print("Resetting due to fall")
            obs, _ = env.reset()
            time.sleep(1.0)  # Pause briefly so you can see the reset
            step_count = 0