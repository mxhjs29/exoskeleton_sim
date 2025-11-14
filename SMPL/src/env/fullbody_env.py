import os
import sys
from typing import List, Tuple, Dict

from sympy.physics.units import velocity

sys.path.append(os.getcwd())

import numpy as np
from collections import OrderedDict
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation as sRot

from SMPL.src.env.base_env import BaseEnv
import SMPL.src.utils.np_transform_utils as npt_utils

class FullBodyEnv(BaseEnv):
    def __init__(self,cfg):
        self.cfg = cfg
        super().__init__(cfg=self.cfg)
        self.setup_configs(cfg)

        self.create_sim(cfg.run.xml_path)
        self.setup_fullbody_params()
        self.reward_info = {}

        self.observation_space = gym.spaces.Box(
            low = -np.inf * np.ones(self.get_obs_size()),
            high = np.inf * np.ones(self.get_obs_size()),
            dtype = self.dtype,
        )

        self.action_space = gym.spaces.Box(
            low = -np.ones(self.get_act_size()),
            high = np.ones(self.get_act_size()),
            dtype =self.dtype,
        )

    def setup_configs(self,cfg) -> None:
        #设置肌肉PD控制参数
        self._kp_scale = cfg.env.kp_scale
        self._kd_scale = cfg.env.kd_scale
        self.control_mode = cfg.run.control_mode
        self.max_episode_length = 300 #每回合最大步数
        self.dtype = np.float32

    def setup_fullbody_params(self):
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):
            body_names = self.mj_model.body(i).name
            self.mj_body_names.append(body_names)

        self.body_names = ['root','torso','pelvis','femur_r','tibia_r','talus_r','calcn_r','toes_r','patella_r',
                           'femur_l','tibia_l','talus_l','calcn_l','toes_l','patella_l']
        self.num_bodies = len(self.body_names)
        self.num_vel_limit = self.num_bodies * 3  #关注这个参数的作用
        self.robot_body_idxes = [
            self.mj_body_names.index(name) for name in self.body_names
        ]  #肌肉骨骼模型，记录[1 2 3 4...]

        self.robot_idx_start = self.robot_body_idxes[0]
        self.robot_idx_end = self.robot_body_idxes[-1] + 1 #这里加1，后面就不用加1了

        #这是在算什么？最后两个关节自由度相同，则是在计算模型中所有关节（joint）所占用的广义坐标（qpos）的总维度（或总数量）,
        # 在搬运任务中，箱子/外骨骼的位置都要被考虑
        self.qpos_lim = np.max(self.mj_model.jnt_qposadr) + self.mj_model.jnt_qposadr[-1] - self.mj_model.jnt_qposadr[-2]
        self.qvel_lim = np.max(self.mj_model.jnt_dofadr) + self.mj_model.jnt_dofadr[-1] - self.mj_model.jnt_dofadr[-2]

        geom_type_id = mujoco.mju_str2Type("geom")
        self.floor_idx = mujoco.mj_name2id(self.mj_model, geom_type_id, "ground-plane")

    def get_obs_size(self) -> int:
        return self.get_self_obs_size()

    def get_act_size(self) -> int:
        return 80  #后面需要更改数值

    def compute_observations(self) -> np.ndarray:
        obs = self.compute_proprioception()
        return obs

    def get_self_obs_size(self) -> int:
        """
            Returns the size of the proprioceptive observations.
            有哪些观测还需要更改
        """
        inputs = self.cfg.run.proprioceptive_inputs
        tally = 0
        if "root_height" in inputs:
            tally += 1
        if "root_tilt" in inputs:
            tally += 4  #四元数形式
        if "local_body_pos" in inputs:
            tally += 3 * self.num_bodies - 3 #减3是因为根节点相对自己永远是0
        if "local_body_rot" in inputs:
            tally += 6 * self.num_bodies #每个body为什么对应6个变量，6D旋转表示法
        if "local_body_vel" in inputs:
            tally += 3 * self.num_bodies
        if "local_body_ang_vel" in inputs:
            tally += 3 * self.num_bodies
        if "muscle_len" in inputs:
            tally += self.mj_model.nu
        if "muscle_vel" in inputs:
            tally += self.mj_model.nu
        if "muscle_force" in inputs:
            tally += self.mj_model.nu
        if "feet_contacts" in inputs:
            tally += 4

        return tally

    def compute_proprioception(self) -> np.ndarray:
        """
        Computes proprioceptive observations for the current simulation state.

        Updates the humanoid's body and actuator states, and generates observations
        based on the configured inputs.

        Returns:
            np.ndarray: Flattened array of proprioceptive observations.

        Notes:
            - The observations are also stored in the `self.proprioception` attribute.
        """
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]


        body_vel = self.get_body_linear_vel()[None,]
        body_ang_vel = self.get_body_angular_vel()[None,]

        obs_dict = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)

        # 从 MuJoCo 仿真数据中读取根组件的姿态四元数，将其分量顺序从 MuJoCo 的
        #   `(w, x, y, z)` 格式调整为 SciPy 库所期望的 `(x, y, z, w)`
        #   格式，然后使用这个调整后的四元数创建一个 SciPy 的 `Rotation`
        #   对象，以便后续进行更方便的旋转计算（例如，在紧接着的下一行代码中，就用
        #   `.as_euler("xyz")` 将其转换为了欧拉角）。
        root_rot = sRot.from_quat(self.mj_data.qpos[[4,5,6,3]])
        root_rot_euler = root_rot.as_euler("xyz")

        fullbody_obs = OrderedDict()

        inputs = self.cfg.run.proprioceptive_inputs

        if "root_height" in inputs:
            fullbody_obs["root_height"] = obs_dict["root_h_obs"]
        if "root_tilt" in inputs:
            fullbody_obs["root_tilt"] = np.array([np.cos(root_rot_euler[0]), np.sin(root_rot_euler[0]), np.cos(root_rot_euler[1]), np.sin(root_rot_euler[1])])
        if "local_body_pos" in inputs:
            fullbody_obs["local_body_pos"] = obs_dict["local_body_pos"][0]
        if "local_body_rot" in inputs:
            fullbody_obs["local_body_rot"] = obs_dict["local_body_rot_obs"][0] # 6 * num_bodies
        if "local_body_vel" in inputs:
            fullbody_obs["local_body_vel"] = obs_dict["local_body_vel"][0] # 3 * num_bodies
        if "local_body_ang_vel" in inputs:
            fullbody_obs["local_body_ang_vel"] = obs_dict["local_body_ang_vel"][0] # 3 * num_bodies
        # 有外骨骼的话，得从mj_data.actuator_length里面去除
        if "muscle_len" in inputs:
            fullbody_obs["muscle_len"] = np.nan_to_num(self.mj_data.actuator_length.copy()) # num_actuators
        if "muscle_vel" in inputs:
            fullbody_obs["muscle_vel"] = np.nan_to_num(self.mj_data.actuator_velocity.copy()) # num_actuators
        if "muscle_force" in inputs:
            fullbody_obs["muscle_force"] = np.nan_to_num(self.mj_data.actuator_force.copy()) # num_actuators
        if "feet_contacts" in inputs:
            fullbody_obs["feet_contacts"] = self.get_touch() # 4
        self.proprioception = fullbody_obs

        return np.concatenate([v.ravel() for v in fullbody_obs.values()], axis=0, dtype=self.dtype)

    def get_body_xpos(self):
        return np.array([self.mj_data.xpos.copy()[i] for i in self.robot_body_idxes])

    def get_body_xquat(self):
        return np.array([self.mj_data.xquat.copy()[i] for i in self.robot_body_idxes])

    def get_body_linear_vel(self):
        # mj_data.sensordata存放顺序是什么？
        return  self.mj_data.sensordata[: self.num_vel_limit].reshape(self.num_bodies, 3).copy()

    def get_body_angular_vel(self):
        #为什么是2倍
        return self.mj_data.sensordata[self.num_vel_limit : 2*self.num_vel_limit].reshape(self.num_bodies, 3).copy()

    def get_touch(self):
        return  self.mj_data.sensordata[2*self.num_bodies :].copy()

    def get_qpos(self):
        return self.mj_data.qpos.copy()[: self.qpos_lim]

    def get_qvel(self):
        return self.mj_data.qvel.copy()[: self.qvel_lim]

    def get_root_pos(self):
        return self.get_body_xpos()[0].copy()

    def compute_reward(self,action):
        reward = 0
        return reward

    def compute_reset(self) -> Tuple[bool,bool]:
    #     计算termination and truncation，判断是任务完成还是达到回合最大步数
        if self.cur_t > self.max_episode_length:
            return False,True
        else:
            return False,False

    def pre_physics_step(self,action):
        pass

    def physics_step(self,action: np.ndarray = None) -> None:
        """
        Executes a physics step in the simulation with the given action.

        Depending on the control mode, computes muscle activations and applies them
        to the simulation. Tracks power usage during the step.

        Args:
            action (np.ndarray): The action to apply. If None, a random action is sampled.
        """
        self.curr_power_usage = []
        if action is None:
            action = self.action_space.sample()

        if self.control_mode == "PD":
    #         肌肉采用PD控制,网络输出action转换为肌肉目标长度
            target_lengths = action_to_target_length(action, self.mj_model)

        for i in range(self.control_freq_inv):
            if not self.paused:
                if self.control_mode == "PD":
                    # 将目标长度转换为目标激活程度
                    muscle_activity = target_length_to_activation(target_lengths, self.mj_data, self.mj_model)
                    # 使某些肌肉无力,模仿病态
                    if self.cfg.run.deactivate_muscles:
                        inactive_muscles = ["tibant_l", "tibant_r"]
                        muscle_activity = self.deactivate_muscles(muscle_activity, inactive_muscles)

                elif self.control_mode == "direct":
                    muscle_activity = (action + 1)/2.0

                else:
                    raise NotImplementedError

                # 存在外骨骼时,注意赋值范围
                self.mj_data.ctrl[:] = muscle_activity
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.curr_power_usage.append(self.compute_energy_reward(muscle_activity))

    def deactivate_muscles(self,muscle_activity : np.ndarray, targetted_muscles: List[str]) -> np.ndarray:
        muscle_names = get_actuator_names(self.mj_model)
        indexes = [muscle_names.index(muscle) for muscle in targetted_muscles]
        for idx in indexes:
            muscle_activity[idx] = 0.0
        return muscle_activity

    def post_physics_step(self,action):
        """
        Processes the environment state after the physics step.

        Increments the simulation time, computes observations, reward, and checks
        for termination or truncation conditions. Collects and returns additional
        information about the reward components.

        Args:
            action (np.ndarray): The action applied at the current step.

        Returns:
            Tuple:
                - obs (np.ndarray): Current observations.
                - reward (float): Reward for the current step.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information, including raw reward components.
        """
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(action)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info.update(self.reward_info)
        return obs, reward, terminated, truncated, info

    def init_fullbody(self):
        # 位姿初始化
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        self.mj_data.qpos[2] = 0.94
        self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])

    def reset_fullbody(self):
        self.init_fullbody()

    def forward_sim(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self,seed=None, options = None):
        self.reset_fullbody()
        self.forward_sim()
        return super().reset(seed=seed, options=options)

    def compute_energy_reward(self, action: np.ndarray) -> float:
        l1_energy = np.abs(action).sum()
        l2_energy = np.linalg.norm(action)
        energy_reward = -l1_energy - l2_energy
        # energy_reward = np.exp(self.reward_specs["k_energy"] * energy_reward)
        return 0

def compute_self_observations(body_pos: np.ndarray, body_rot: np.ndarray, body_vel: np.ndarray, body_ang_vel: np.ndarray) -> OrderedDict:
    """
    Computes observations of the agent's local body state relative to its root.

    Args:
        body_pos (np.ndarray): Global positions of the bodies.
        body_rot (np.ndarray): Global rotations of the bodies in quaternion format.
        body_vel (np.ndarray): Linear velocities of the bodies.
        body_ang_vel (np.ndarray): Angular velocities of the bodies.

    Returns:
        OrderedDict: Dictionary containing:
            - `root_h_obs`: Root height observation.
            - `local_body_pos`: Local body positions excluding root.
            - `local_body_rot_obs`: Local body rotations in tangent-normalized format.
            - `local_body_vel`: Local body velocities.
            - `local_body_ang_vel`: Local body angular velocities.
    """
    obs = OrderedDict()

    root_pos = body_pos[:,0,:]
    root_rot = body_rot[:,0,:]

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:,2:3]
    obs["root_h_obs"] = root_h

    heading_rot_inv_expand = heading_rot_inv[...,None,:]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1],axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    # 计算各个部分的相对位置,朝向统一
    root_pos_expand = root_pos[...,None,:]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    # 不包含根节点位置,因为一直是0
    obs["local_body_pos"] = local_body_pos[...,3:]

    # 相对角度计算
    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    # 相对速度
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"] = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    return obs

def get_actuator_names(model) -> list:
    actuators = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in ["name_numericadr", "name_textadr", "name_tupleadr", "name_keyadr", "name_pluginadr",
                       "name_sensoradr"]:
                v = getattr(model, el)
                if np.any(v):
                    if end_p is None:
                        end_p = v[0]
                    else:
                        end_p = min(end_p, v[0])
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i + 1]
        name = model.names[model.name_actuatoradr[i]:end_p].decode("utf-8").rstrip('\x00')
        actuators.append(name)
    return actuators

def force_to_activation(forces,data,model):
    # 根据力计算对应的目标激活程度
    activations=[]

    # 有外骨骼的时候,注意循环次数
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        # 这三个是什么参数
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator,:9]
        prmg = model.actuator_gainprm[idx_actuator,:9]
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1,mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        activations.append(np.clip((forces[idx_actuator] - bias) / gain, 0, 1))

    return activations

def target_length_to_force(lengths: np.ndarray, data, model):
    # 使用PD控制使目标长度转换为对应的力
    forces = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        peak_force = model.actuator_biasprm[idx_actuator, 2]
        kp = 5 * peak_force
        kd = 0.1 * kp
        force = (kp * (lengths[idx_actuator] - length) - kd * velocity)
        clipped_force = np.clip(force, -peak_force, 0)
        forces.append(clipped_force)

    return forces

def target_length_to_activation(lengths: np.ndarray, data, model):
#     将目标长度转为目标激活程度
    forces = target_length_to_force(lengths,data,model)
    activations = force_to_activation(forces,data,model)
    return np.clip(activations,0,1)

def action_to_target_length(action: np.ndarray, model) -> list:
    # 将策略网络输出[-1,1]线性放缩成肌肉的目标长度
    target_lengths = []
    for idx_actuator in range(model.nu):
        hi = model.actuator_lengthrange[idx_actuator,1]
        lo = 0
        target_lengths.append((action[idx_actuator] + 1) / 2 * (hi - lo) + lo)
    return target_lengths

def compute_imitation_reward(
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    rwd_specs: dict,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Computes the imitation reward based on differences in positions and velocities
    between the current and reference states.

    Args:
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        rwd_specs (dict): Reward specifications containing:
            - `"k_pos"`: Scaling factor for position reward.
            - `"k_vel"`: Scaling factor for velocity reward.
            - `"w_pos"`: Weight for position reward.
            - `"w_vel"`: Weight for velocity reward.

    Returns:
        Tuple:
            - reward (float): Weighted sum of position and velocity rewards.
            - reward_raw (Dict[str, np.ndarray]): Dictionary of raw reward components:
                - `"r_body_pos"`: Body position reward.
                - `"r_vel"`: Velocity reward.
    """
    k_pos, k_vel = rwd_specs["k_pos"], rwd_specs["k_vel"]
    w_pos, w_vel = rwd_specs["w_pos"], rwd_specs["w_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(axis=-1).mean(axis=-1)
    r_body_pos = np.exp(-k_pos * diff_body_pos_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(axis=-1).mean(axis=-1)
    r_vel = np.exp(-k_vel * diff_global_vel_dist)

    reward = w_pos * r_body_pos + w_vel * r_vel
    reward_raw = {
        "r_body_pos": r_body_pos,
        "r_vel": r_vel,
    }

    return reward[0], reward_raw