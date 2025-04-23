import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class SingleLegEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("mujoco_model.xml")
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.model.nq + self.model.nv,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.05 * np.random.randn(self.model.nq)
        self.data.qvel[:] = 0.05 * np.random.randn(self.model.nv)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)

        # Apply occasional random push
        if np.random.rand() < 0.02:
            force = np.random.uniform(-20, 20, size=3)
            mujoco.mj_applyFT(
                self.model,                   # model
                self.data,                    # data
                force.reshape(3, 1),          # force (3x1)
                np.zeros((3, 1)),             # torque (3x1)
                self.data.xpos[self.model.body('torso').id],  # point of application
                self.model.body('torso').id, # body ID
                self.data.qfrc_applied        # target force array
            )

        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        com_height = self.data.qpos[2]
        com_vel = self.data.qvel[2]
        torque_penalty = 0.01 * np.sum(np.square(action))
        reward = com_height - 0.1 * abs(com_vel) - torque_penalty

        done = com_height < 0.4

        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

# Wrap with Monitor to track rewards and length
env = Monitor(SingleLegEnv())

# Train PPO agent with tensorboard logging
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")
model.learn(total_timesteps=100_000)
model.save("ppo_single_leg")
