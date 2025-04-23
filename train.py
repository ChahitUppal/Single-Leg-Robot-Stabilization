import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

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
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # Reward = stay upright
        height = self.data.qpos[2]
        reward = height
        done = height < 0.5  # fell over

        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

env = SingleLegEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_single_leg")