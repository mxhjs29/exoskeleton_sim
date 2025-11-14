import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import time
from typing import Optional

class BaseEnv(gym.Env):
    def __init__(self,cfg):
        self.clip_actions = False
        self.render_mode = "human"
        self.headless = cfg.run.headless
        #仿真频率
        self.sim_timestep_inv = cfg.env.sim_timestep_inv
        self.sim_timestep = 1.0 / self.sim_timestep_inv
        self.control_freq_inv = cfg.env.control_frequency_inv
        self.cur_t = 0
        self.dt = self.sim_timestep * self.control_freq_inv
        self.fast_forward = cfg.run.fast_forward
        #可视化参数
        self.viewer = None
        self.renderer = None
        self.camera = -1
        self.paused = False
        self.disable_reset = False
        self.follow = False  #镜头跟随

    def reset(self,seed:Optional[int] = None, options = None):
        """

        :param seed: The random seed for the environment. Defaults to None.
        :param options: Additional options for resetting the environment; they are plugged into gym.Env.reset(). Defaults to None.
        :return: tuple: A tuple containing the observation and info after the reset.
        """
        super().reset(seed=seed, options=options)
        self.cur_t = 0
        observation = self.compute_observations()
        info = {}
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.render()
        return observation,info

    def compute_observations(self):
        #等待子类填充
        raise NotImplementedError

    def compute_info(self):
        raise NotImplementedError

    def step(self,action:np.ndarray):
        """
        :param action: The action to take in the environment. Must be compatible with the action space.
        :return:
            observation: The current observation of the environment - cf compute_observations().
            reward: Scalar - the reward obtained from the environment. It is a weighted sum of several reward components.
            terminated: A boolean indicating whether the episode is terminated, i.e. a fail state has been reached.
            truncated: A boolean indicating whether the episode is truncated, i.e. the maximum number of steps has been reached.
            info: Additional information about the step - cf post_physics_step().
        """
        self.pre_physics_step(action)
        self.physics_step(action)
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.render()

        return observation, reward, terminated, truncated, info

    def pre_physics_step(self,action):
        raise NotImplementedError

    def physics_step(self,action):
        raise NotImplementedError

    def post_physics_step(self,action):
        raise NotImplementedError

    def render(self):
        """
            Renders the environment.
                If the environment is not set to headless mode, it creates a viewer and updates the rendering based on the render mode.

            Returns:
                If render mode is "rgb_array", returns the rendered pixels as an array.
        """
        if not self.headless:
            if self.viewer is None and self.renderer is None:
                self.create_viewer()

            if self.render_mode == "human":
                self.viewer.sync()
                if self.follow:
                    self.viewer.cam.lookat = self.mj_data.qpos[:3]
                if not self.fast_forward:
                    time.sleep(1. / 100)

            if self.render_mode == "rgb_array":
                self.renderer.update_scene(self.mj_data,cama=self.camera)
                pixels = self.renderer.render()
                return pixels

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def seed(self,seed:Optional[int] = None):
        super().reset(seed=seed)

    def create_sim(self, xml_path:str):
        """
        Creates a simulation environment using the specified XML file.

        Args:
            xml_path (str): The path to the XML file.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_timestep

    def _create_renderer(self):
        self.renderer = mujoco.Renderer(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data)

    def create_viewer(self):
        if not self.headless and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.mj_model,self.mj_data,key_callback=self.key_callback)

        if not self.headless and self.render_mode == "rgb_array":
            self._create_renderer()

    def key_callback(self,keycode):
        #键盘输入
        print(keycode)
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Paused {self.paused}")
        elif chr(keycode) == "R":
            self.reset()
        elif chr(keycode) == "M":
            self.disable_reset = not self.disable_reset
            print(f"Disable reset {self.disable_reset}")
        elif chr(keycode) == "F":
            self.follow = not self.follow
            print(f"Follow {self.follow}")

