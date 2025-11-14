import numpy as np
import pandas as pd
from  myosuite.utils import gym
import yaml
from myosuite.envs.obs_vec_dict import ObsVecDict
from myosuite.utils.implement_for import implement_for
from myosuite.utils.prompt_utils import prompt,Prompt
from myosuite.envs.env_variants import gym_registry_specs, register_env_variant, register
from SMPL.src.utils.common.model_load import *
import pickle
import SMPL.src.utils.np_transform_utils as npt_utils
from scipy.spatial.transform import Rotation as sRot
from collections import OrderedDict
import torch 
import joblib
from SMPL.src.assistance_policy import ExoNet
import math
from pettingzoo import ParallelEnv

file_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data/initial_state_with_exo.yaml"
with open(file_path,'r') as file:
    initial_joint = yaml.safe_load(file)
joint_qpos = np.array(initial_joint['qpos'])
joint_qvel = np.array(initial_joint['qvel'])

obs_true_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/data/obs_truth'

class MARL_EXO_Env(ParallelEnv, ObsVecDict):
    DEFAULT_CREDIT = """\
            MyoSuite: a collection of environments/tasks to be solved by musculoskeletal models | https://sites.google.com/view/myosuite
            Code: https://github.com/MyoHub/myosuite/stargazers (add a star to support the project)
        """
    metadata = {"render_modes": ['human','rgb_array'], "render_fps": 30}
    def __init__(self, model_path, seed=None,env_credits=DEFAULT_CREDIT,max_episode_steps = 2500, n_stacks=4, **kwargs):
        prompt("MyoSuite:> For environment credits, please cite -")
        prompt(env_credits, color="cyan", type=Prompt.ONCE)

        self.seed(seed)
        self.agents = ["human","exo"]
        self.n_stacks = n_stacks
        self.stacked_obs = None
        self.sim = SimScene.get_sim(model_path)
        # 观测关节
        self.obs_joints = ['pelvis_tx','ankle_angle_r', 'hip_flexion_l','ankle_angle_l','lumbar_rotation']
        self.obs_joints_id = {name: self.sim.model.joint(name).id for name in self.obs_joints}
        self.actuators = ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 'addmagIsch_r', 'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'bflh140_l', 'bflh140_r', 'bfsh140_l', 'bfsh140_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'ercspn_l', 'ercspn_r', 'extobl_l', 'extobl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'gaslat140_l', 'gaslat140_r', 'gasmed_l', 'gasmed_r', 'gem_l', 'gem_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 'glmin3_l', 'glmin3_r', 'grac_l', 'grac_r', 'iliacus_l', 'iliacus_r', 'intobl_l', 'intobl_r', 'obtext_l', 'obtext_r', 'obtint_l', 'obtint_r', 'pect_l', 'pect_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 'perlong_r', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'quadfem_l', 'quadfem_r', 'recfem_l', 'recfem_r', 'rectab_l', 'rectab_r', 'sart_l', 'sart_r', 'semimem_l', 'semimem_r', 'semiten_l', 'semiten_r', 'soleus_l', 'soleus_r', 'tfl_l', 'tfl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 'tibpost_r', 'vasint_l', 'vasint_r', 'vaslat140_l', 'vaslat140_r', 'vasmed_l', 'vasmed_r']
        self.actuators_id = {name: self.sim.model.actuator(name).id for name in self.actuators}
        self.hip_joints = ['hip_flexion_r','hip_adduction_r','hip_rotation_r',
                          'hip_flexion_l','hip_adduction_l','hip_rotation_l',
                          'jActuatedRightHip_rotx', 'jActuatedLeftHip_rotx']
        self.hip_joints_id = {name: self.sim.model.joint(name).id for name in self.hip_joints}
        # 需要记录激活程度的肌肉
        self.muscle_tendon = ['soleus_r_tendon','gasmed_r_tendon','tibant_r_tendon','recfem_r_tendon',
                              'vasmed_r_tendon','vaslat140_r_tendon','semiten_r_tendon','bflh140_r_tendon','glmed1_r_tendon',
                              'soleus_l_tendon', 'gasmed_l_tendon', 'tibant_l_tendon', 'recfem_l_tendon',
                              'vasmed_l_tendon', 'vaslat140_l_tendon', 'semiten_l_tendon', 'bflh140_l_tendon','glmed1_l_tendon'
                              ]
        self.muscle_tendon_id = {name: self.sim.model.tendon(name).id for name in self.muscle_tendon}
        self.muscle_activation = {name: [] for name in self.muscle_tendon}
        # 初始化，exo关节数据有没有
        # self.sim.data.qpos[:] = joint_qpos[0]
        qvel = np.zeros(len(joint_qvel[0]))
        # self.sim.data.qvel[:] = qvel
        self.sim.forward()
        ObsVecDict.__init__(self)
        self.dataset_relative_pos, self.dataset_relative_quat, self.grap_actuators, self.grap_actuators_id = self._load_dataset(self.sim)
        self.flag_animation = 0
        self.end_flag = 0
        self.max_episode_steps = max_episode_steps
        self.calculate_hight_weight()
        self.x = []
        self.y_r = []
        self.y_l = []
        self.qfrc_norm_r = 0
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = None,
               weighted_reward_keys: dict = {},
               render_bool: bool = True,
               length_frame: int = 469,
               frame_skip: int = 2,
               rwd_viz: bool = False,
               device_id: int = 0,
               scale: int = 5,
               apply_muscle_activate: bool = True,
               offline_render: bool = False,
               record_exo_force: bool = False,
            ):

        if self.sim is None:
            raise TypeError("sim must be instantiated for setup to run")
        # resolve view
        self.mujoco_render_frames = True
        self.offline_render = offline_render
        self.device_id = device_id
        self.rwd_viz = rwd_viz
        self.viewer_setup()
        # resolve action space
        self.frame_skip = frame_skip
        self.render_bool = render_bool
        
        self.scale = scale
        self.apply_muscle_activate = apply_muscle_activate
        # resolve initial state
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        # resolve rewards
        self.rwd_dict = {}
        self.rwd_keys_wt = weighted_reward_keys
        # resolve obs
        self.obs_dict = {}
        self.obs_keys = obs_keys
        # resolve proprio
        self.body_names = ['pelvis','femur_r','tibia_r','talus_r','calcn_r','toes_r','patella_r',
                           'femur_l','tibia_l','talus_l','calcn_l','toes_l','patella_l','torso']
        self.robot_body_idxes = [
            self.sim.model.body(name).id for name in self.body_names
        ]
        self.index = 2
        length_pos, qpos_gt, qvel_gt, local_body_pos, local_body_rot, local_body_vel, local_body_angle_vel = obs_data_load(obs_true_path, self.index)
        self.length_frame = length_frame
        self.qpos_gt = qpos_gt
        self.qvel_gt = qvel_gt
        self.local_body_pos_gt = local_body_pos
        self.local_body_rot_gt = local_body_rot
        self.local_body_vel_gt = local_body_vel
        self.local_body_angle_vel_gt = local_body_angle_vel
        self.count = 0
        done = False
        self.record_exo_force = record_exo_force
        self.qfrc_actuator_r = 0
        self.qfrc_actuator_l = 0
        self.n = 0
        self.N = 2
        #外骨骼数据采集，助力策略加载
        self.timestamp = []
        self.theta_l = []
        self.dtheta_l = []
        self.theta_r = []
        self.dtheta_r = []
        self.tau_l = []
        self.tau_r = []
        self.run_exo = True
        self.assistance_model = ExoNet()
        self.assistance_model.load_state_dict(torch.load("/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/exo_model.pth",weights_only=True))
        self.assistance_model.eval()
        self.scaler_X = joblib.load("/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/scaler_X.pkl")
        self.scaler_y = joblib.load("/home/chenshuo/PycharmProjects/move_sim/SMPL/data/assistance_policy/scaler_y.pkl")
        self.qfrc_actuator_r_avrage = []
        self.qfrc_actuator_l_avrage = []
        observation = self.reset()
        assert not done,"Check initialization. Simulation starts in a done state."
        self.observation_space = {
            "human": gym.spaces.Box(low=-np.inf * np.ones(len(observation[0])),
                                    high=np.inf * np.ones(len(observation[0])),
                                    dtype=np.float64),
            "exo": gym.spaces.Box(low=-np.inf * np.ones(len(observation[1])),
                                    high=np.inf * np.ones(len(observation[1])),
                                    dtype=np.float64)
        }
        act_low = -np.ones(self.sim.model.na - 18) 
        act_high = np.ones(self.sim.model.na - 18) 
        self.action_space = {
            "human": gym.spaces.Box(act_low,act_high,dtype=np.float64),
            "exo": gym.spaces.Box(-np.array([1,1]),np.array([2,2]),dtype=np.float64)
        }

        return

    def activation_2_color(self, activation):
        # 确保激活度在 [0, 1] 范围内
        activation = np.clip(activation, 0.0, 1.0)
        if activation <= 0.5:
            # 从蓝 (0,0,1) 渐变到绿 (0,1,0)
            r = 0.0
            g = 0.0
            b = 1.0  # 在0.5时达到0.0
        else:
            # 从绿 (0,1,0) 渐变到红 (1,0,0)
            r = 1.0  # 在1.0时达到1.0
            g = 0.0  # 在1.0时达到0.0
            b = 0.0
        return [r, g, b, 1.0]  # RGBA，A(alpha)固定为1.0

    def activation_to_color(self, model, data):
        color = []
        N = len(data.ctrl) - 2
        activation = data.ctrl
        # 根据激活度计算颜色
        for i in range(N):
            color.append(self.activation_2_color(activation[i + 2]))
        for i in range(N):
            model.tendon_rgba[i] = color[i]

    def record_muscle_activation(self):
        for name in self.muscle_tendon:
            self.muscle_activation[name].append(self.sim.data.ctrl[self.muscle_tendon_id[name]])
        if self.flag_animation >= self.length_frame - 1:
            df = pd.DataFrame(self.muscle_activation)
            file_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/muscle_activation.csv'
            df.to_csv(file_path, sep="\t", index=False)
            self.muscle_activation = {name: [] for name in self.muscle_tendon}
            print("记录一次肌肉激活")

    def step(self,a, **kwargs):
        step = 3
        if (self.flag_animation == 345):
            step = 100
        if (self.flag_animation == 147):
            step = 100
        if (self.flag_animation == 148):
            step = 3
        if (self.flag_animation == 346):
            step = 3
        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = (a + 1.) / 2.
        

        if self.flag_animation == 250:
            self.end_flag = 1

        if (self.flag_animation < 147 and self.end_flag == 0):
            command = 0
            self.sim.data.ctrl[2:] = grasp_handle(self.sim.data.ctrl[2:], self.grap_actuators, self.grap_actuators_id, command,a)

        if (self.flag_animation >= 147 and self.end_flag == 0):
            command = 1
            # 更改手部肌肉发力
            self.sim.data.ctrl[2:] = grasp_handle(self.sim.data.ctrl[2:], self.grap_actuators, self.grap_actuators_id, command,a)

        if (self.flag_animation >= 250 and self.end_flag == 1):
            # action = np.ones(self.sim.model.nu) * 0
            command = 0
            self.sim.data.ctrl[2:] = grasp_handle(self.sim.data.ctrl[2:], self.grap_actuators, self.grap_actuators_id, command,a)

        self.qfrc_actuator_r = (self.sim.data.qfrc_actuator[self.hip_joints_id['hip_flexion_r']]
                                  - self.sim.data.qfrc_bias[self.hip_joints_id['hip_flexion_r']]
                                  + self.sim.data.qfrc_passive[self.hip_joints_id['hip_flexion_r']])
        self.qfrc_actuator_l = (self.sim.data.qfrc_actuator[self.hip_joints_id['hip_flexion_l']]
                                - self.sim.data.qfrc_bias[self.hip_joints_id['hip_flexion_l']]
                                + self.sim.data.qfrc_passive[self.hip_joints_id['hip_flexion_l']])
        if self.record_exo_force:
            if np.abs(self.qfrc_actuator_l) < 50:
                self.exo_torque_l = np.clip(0.1 * self.qfrc_actuator_l, -5, 5)
            elif np.abs(self.qfrc_actuator_l) < 150:
                self.exo_torque_l = np.clip(0.2 * self.qfrc_actuator_l, -20, 20)
            else:
                self.exo_torque_l = self.qfrc_actuator_l * 0.4

            if np.abs(self.qfrc_actuator_r) < 50:
                self.exo_torque_r = np.clip(0.1 * self.qfrc_actuator_r, -5, 5)
            elif np.abs(self.qfrc_actuator_r) < 150:
                self.exo_torque_r = np.clip(0.2 * self.qfrc_actuator_r, -20, 20)
            else:
                self.exo_torque_r =  self.qfrc_actuator_r * 0.4

            self.timestamp.append(self.time)
            self.theta_l.append(self.sim.data.qpos[self.hip_joints_id['jActuatedLeftHip_rotx']])
            self.dtheta_l.append(self.sim.data.qvel[self.hip_joints_id['jActuatedLeftHip_rotx']])
            self.tau_l.append(self.exo_torque_l)
            self.theta_r.append(self.sim.data.qpos[self.hip_joints_id['jActuatedRightHip_rotx']])
            self.dtheta_r.append(self.sim.data.qvel[self.hip_joints_id['jActuatedRightHip_rotx']])
            self.tau_r.append(self.exo_torque_r)
            if self.flag_animation >= self.length_frame - 1:
                self.timestamp = [t - self.timestamp[0] for t in self.timestamp]
                data_exo = {
                    'timestamp': self.timestamp,
                    'theta_l': self.theta_l,
                    'dtheta_l': self.dtheta_l,
                    'tau_l': self.tau_l,
                    'theta_r': self.theta_r,
                    'dtheta_r': self.dtheta_r,
                    'tau_r': self.tau_r
                }
                exo_data_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/data/obs_truth/exo_im_data.csv'
                with open(exo_data_path, 'w') as f:
                    writer = pd.DataFrame(data_exo)
                    writer.to_csv(f, sep="\t", index=False)
                print("保存一次外骨骼数据")

        if self.run_exo:
            x_real = np.array([[self.sim.data.qpos[self.hip_joints_id['jActuatedLeftHip_rotx']],
                                self.sim.data.qvel[self.hip_joints_id['jActuatedLeftHip_rotx']],
                                self.sim.data.qpos[self.hip_joints_id['jActuatedRightHip_rotx']],
                                self.sim.data.qvel[self.hip_joints_id['jActuatedRightHip_rotx']]
                                ]])
            x_scaled = self.scaler_X.transform(x_real)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
            with torch.no_grad():
                y_pred = self.assistance_model(x_tensor).numpy()
            y_real = self.scaler_y.inverse_transform(y_pred)
            self.exo_torque_l = y_real[0, 0]
            self.exo_torque_r = y_real[0, 1]
            self.sim.data.ctrl[0] = self.exo_torque_l
            self.sim.data.ctrl[1] = self.exo_torque_r
        else:
            self.sim.data.ctrl[0] = 0
            self.sim.data.ctrl[1] = 0

        if self.count % step == 0 and self.flag_animation < self.length_frame:
            # mocap的更新频率要慢一些
            self.applied_uplimb_mocap(self.dataset_relative_pos, self.dataset_relative_quat, self.flag_animation)
            self.flag_animation += 1
                
        self.plot_draw()
        self.activation_to_color(self.sim.model, self.sim.data)
        self.record_muscle_activation()
        self.sim.advance(substeps=self.frame_skip, render=self.render_bool)
        self.sim.forward()

        self.count = self.count + 1
        if (self.offline_render):
            self.frame = self.render_offscreen()
        else:
            self.frame = None
        self.muscle_activate = []
        return self.forward(**kwargs)

    @implement_for("gym", None, "0.24")
    def forward(self, **kwargs):
        return self._forward(**kwargs)

    @implement_for("gym", "0.24", None)
    def forward(self, **kwargs):
        obs, reward, done, truncated,info = self._forward(**kwargs)
        terminal = done
        print(reward)
        return obs, reward, terminal, truncated, info

    @implement_for("gymnasium")
    def forward(self, **kwargs):
        obs, reward, done, truncated,info = self._forward(**kwargs)
        terminal = done
        # print(self.flag_animation,reward)
        return obs, reward, terminal, truncated, info

    def _forward(self,**kwargs):
        if self.mujoco_render_frames:
            self.mj_render()
        # obs
        obs = self.get_obs(**kwargs)
        self.stacked_obs.append(obs)
        if len(self.stacked_obs) > self.n_stacks:
            self.stacked_obs.pop(0)
        stack_obs = self._get_stacked_obs()

        # rwd
        self.expand_dims(self.obs_dict)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)
        #final step
        env_info = self.get_env_info()
        if self.count >= self.max_episode_steps:
            truncated = True
        else:
            truncated = False
        if self.flag_animation >= self.length_frame-1:
            self.flag_animation = 0
        return stack_obs, env_info['rwd_pose'], env_info['done'], truncated, env_info

    def get_obs(self,update_proprioception=True, update_exteroception=False, **kwargs):
        self.get_obs_dict(self.obs_dict)
        t,obs = self.obsdict2obsvec(self.obs_dict,self.obs_keys)
        return obs

    def _get_stacked_obs(self):
        return np.concatenate(self.stacked_obs, axis=0, dtype=np.float64)

    def get_proprioception(self):
        obs_proprio = OrderedDict()
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]
        body_vel = self.get_body_linear_vel()[None,]
        body_ang_vel = self.get_body_angular_vel()[None,]
        obs_proprio = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)
        return obs_proprio
    
    def get_body_xpos(self):
        return np.array([self.sim.data.xpos.copy()[i] for i in self.robot_body_idxes])

    def get_body_xquat(self):
        return np.array([self.sim.data.xquat.copy()[i] for i in self.robot_body_idxes])

    def get_body_linear_vel(self):
        return np.array([self.sim.data.cvel.copy()[i][:3] for i in self.robot_body_idxes])

    def get_body_angular_vel(self):
        return np.array([self.sim.data.cvel.copy()[i][3:] for i in self.robot_body_idxes])

    def get_exteroception(self,**kwargs):
        return self.get_visuals(**kwargs)

    def get_env_info(self):
        env_info = {
            'time': self.obs_dict['time'][()],  # MDP(t)
            'rwd_pose': self.rwd_dict['R_p'][()],  # MDP(t)
            'done': self.rwd_dict['done'][()],  # MDP(t)
            'obs_dict': self.obs_dict,  # MDP(t)
            'rwd_dict': self.rwd_dict,  # MDP(t)
            'state': self.get_env_state(),  # MDP(t)
            'stack_obs': self.stacked_obs,
            'muscle_activate': self.muscle_activate,
            'frame': self.frame
        }
        return env_info

    def seed(self,seed=None):
        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_input_seed(self):
        return self.input_seed

    def _reset(self,reset_qpos=None, options=None, reset_qvel=None, seed=None):

        length_pos, qpos_gt, qvel_gt, local_body_pos, local_body_rot, local_body_vel, local_body_angle_vel = obs_data_load(obs_true_path, self.index)
        self.length_frame = 469
        self.qpos_gt = qpos_gt
        self.qvel_gt = qvel_gt
        self.local_body_pos_gt = local_body_pos
        self.local_body_rot_gt = local_body_rot
        self.local_body_vel_gt = local_body_vel
        self.local_body_angle_vel_gt = local_body_angle_vel
        self.count = 0
        self.flag_animation = 0
        self.end_flag = 0
        self.sim.reset()
        self.sim.advance(substeps=self.frame_skip, render=self.render_bool)
        self.sim.forward()
        obs = self.get_obs()
        self.stacked_obs = [obs.copy() for _ in range(self.n_stacks)]
        return self._get_stacked_obs()

    @implement_for("gym", None, "0.26")
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs)

    @implement_for("gym", "0.26", None)
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs), {}

    @implement_for("gymnasium")
    def reset(self, reset_qpos=None, reset_qvel=None, seed=None, options=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, seed=seed, **kwargs), {}

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip

    @property
    def time(self):
        return self.sim.data.time

    @property
    def id(self):
        return self.spec.id

    @implement_for("gym")
    def _horizon(self):
        return self.spec.max_episode_steps  # paths could have early termination before horizon

    @implement_for("gymnasium")
    def _horizon(self):
        return gym_registry_specs()[
            self.spec.id].max_episode_steps  # gymnasium unwrapper overrides specs (https://github.com/Farama-Foundation/Gymnasium/issues/871)

    @property
    def horizon(self):
        return self._horizon()

    def get_env_state(self):
        time = self.sim.data.time
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        act = self.sim.data.act.ravel().copy() if self.sim.model.na > 0 else None
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap > 0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite > 0 else None
        site_quat = self.sim.model.site_quat[:].copy() if self.sim.model.nsite > 0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        body_quat = self.sim.model.body_quat[:].copy()
        return dict(time=time,
                    qpos=qp,
                    qvel=qv,
                    act=act,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    site_pos=site_pos,
                    site_quat=site_quat,
                    body_pos=body_pos,
                    body_quat=body_quat)

    def mj_render(self):
        self.sim.renderer.render_to_window()


    def viewer_setup(self,distance=2.5,azimuth=90,elevation=-30, lookat=None, render_actuator=None, render_tendon=None):
        self.sim.renderer.set_free_camera_settings(
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            lookat=lookat
        )
        self.sim.renderer.set_viewer_settings(
            render_actuator=render_actuator,
            render_tendon=render_tendon
        )

    def get_obs_dict(self, obs_dict=None):
        # 观测字典
        obs_dict['frame'] = np.array([self.flag_animation])
        obs_dict['qpos_gt'] = self.qpos_gt[self.count]
        obs_dict['time'] = np.array([self.time])
        qpos_obs = self.sim.data.qpos[self.obs_joints_id['pelvis_tx']:self.obs_joints_id['lumbar_rotation']+1]
        obs_dict['qpos'] = qpos_obs.ravel().copy()
        obs_dict['qpos_error'] = obs_dict['qpos_gt'] - obs_dict['qpos']
        obs_dict['activation'] = self.sim.data.act[:].copy()
        obs_dict['qfrc_constraint'] = self.sim.data.qfrc_constraint.ravel().copy()
        obs_dict['qfrc_constraint'] = np.nan_to_num(obs_dict['qfrc_constraint'], nan=0, posinf=300, neginf=-300)
        obs_dict['exo_torque_t'] = self.sim.data.ctrl[:2].copy()
        #外骨骼关节角度设定值，需要从外骨骼控制器中获取
        if self.count >= 2:
            obs_dict['exo_joint_set_t2'] = self.obs_dict['exo_joint_set_t1']
        else:
            obs_dict['exo_joint_set_t2'] = np.array([0,0])
        if self.count >=1:
            obs_dict['exo_joint_set_t1'] = self.obs_dict['exo_joint_set_t']
        else:
            obs_dict['exo_joint_set_t1'] = np.array([0,0])
        obs_dict['exo_joint_set_t'] = np.array([0,0])
        #添加proprioception观测，pelvis，femur_r，tibia_r，femur_l，tibia_l，torso
        #添加proprioception的真值
        obs_dict['local_body_pos_gt'] = self.local_body_pos_gt[self.count]
        obs_dict['local_body_rot_gt'] = self.local_body_rot_gt[self.count]
        obs_dict['local_body_vel_gt'] = self.local_body_vel_gt[self.count]
        obs_dict['local_body_angle_vel_gt'] = self.local_body_angle_vel_gt[self.count]
        obs_proprio =  self.get_proprioception()
        obs_dict.update(obs_proprio)
        obs_dict['proprio_error'] = np.concatenate([
            (obs_dict['local_body_pos_gt'] - obs_dict['local_body_pos']).ravel(),
            (obs_dict['local_body_rot_gt'] - obs_dict['local_body_rot_obs']).ravel(),
            (obs_dict['local_body_vel_gt'] - obs_dict['local_body_vel']).ravel(),
            (obs_dict['local_body_angle_vel_gt'] - obs_dict['local_body_angle_vel']).ravel()],
            axis=-1)
        return None

    def get_reward_dict(self,obs_dict):
        # reward_dict包含肌肉骨骼模型、外骨骼模型的奖励，分为各个部分存放，按需调取
        # 肌肉骨骼模型：R_p R_proprio R_a
        # 外骨骼模型：R_p R_proprio R_a R_t R_eas
        # R_p: 位置奖励，关节位置误差
        # R_proprio: 本体感知奖励
        # R_a: 肌肉激活奖励
        # R_t: 外骨骼能量消耗
        # R_eas: 外骨骼助力平滑奖励
        reward_dict = self.rwd_keys_wt.copy()

        pos_error = obs_dict['qpos_error']
        pos_error_reward = self.rwd_keys_wt['w_pos_err'] * math.exp( -self.rwd_keys_wt['theta_pos_err'] * np.sum(pos_error ** 2)) 
        reward_dict['R_p'] =   pos_error_reward

        proprio_error = obs_dict['proprio_error']
        proprio_error_reward = self.rwd_keys_wt['w_proprio_err'] * math.exp( -self.rwd_keys_wt['theta_proprio_err'] * np.sum(proprio_error ** 2)) 
        reward_dict['R_proprio'] = proprio_error_reward

        #肌肉激活的奖励数值大小可能与其他奖励差距较大，需要调节权重
        activation = obs_dict['activation']
        activation_reward = self.rwd_keys_wt['w_activation'] * math.exp( -self.rwd_keys_wt['theta_activation'] * np.sum(activation ** 2))
        reward_dict['R_a'] = activation_reward

        exo_torque = obs_dict['exo_torque_t']
        exo_energy = np.sum(np.abs(exo_torque * self.sim.data.qvel[self.hip_joints_id['jActuatedRightHip_rotx']:self.hip_joints_id['jActuatedLeftHip_rotx']+1]))
        reward_dict['R_t'] = self.rwd_keys_wt['w_exo_energy'] * math.exp( -self.rwd_keys_wt['theta_exo_energy'] * exo_energy)
   
        exo_smooth = np.sum((obs_dict['exo_joint_set_t'] - 2 * obs_dict['exo_joint_set_t1'] + obs_dict['exo_joint_set_t2']) ** 2)
        reward_dict['R_eas'] = self.rwd_keys_wt['w_exo_smooth'] * math.exp( -self.rwd_keys_wt['theta_exo_smooth'] * exo_smooth)

        if (self.flag_animation >= self.length_frame - 1):
            reward_dict['done'] = True
        else:
            reward_dict['done'] = False
        return reward_dict

    def render_offscreen(self, width=800, height=1000, camera_id=1):
        if not self.sim or not self.sim.renderer:
            raise RuntimeError("Simulator or renderer not initialized.")
        # 进行无头渲染
        try:
            frame = self.sim.renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=camera_id
            )
        except Exception as e:
            raise RuntimeError(f"Rendering failed: {e}")
        return frame

    def applied_uplimb_mocap(self, dataset_relative_pos, dataset_relative_quat, flag_animation):
        self.sim.data.mocap_pos[22] = dataset_relative_pos['base_pos'][0][:, flag_animation, :].squeeze()
        self.sim.data.mocap_quat[22] = dataset_relative_quat['base_rot'][0][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[23] = self.sim.data.mocap_pos[22].copy() + dataset_relative_pos['body'][0][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[23] = dataset_relative_quat['base_rot'][0][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[32] = self.sim.data.mocap_pos[23].copy() + dataset_relative_pos['right_arm'][0][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[32] = dataset_relative_quat['right_arm'][1][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[33] = self.sim.data.mocap_pos[32].copy() + dataset_relative_pos['right_arm'][1][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[33] = dataset_relative_quat['right_arm'][2][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[34] = self.sim.data.mocap_pos[33].copy() + dataset_relative_pos['right_arm'][2][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[34] = dataset_relative_quat['right_arm'][3][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[35] = self.sim.data.mocap_pos[23].copy() + dataset_relative_pos['left_arm'][0][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[35] = dataset_relative_quat['left_arm'][1][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[36] = self.sim.data.mocap_pos[35].copy() + dataset_relative_pos['left_arm'][1][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[36] = dataset_relative_quat['left_arm'][2][:, flag_animation, :].squeeze()
        self.sim.data.mocap_pos[37] = self.sim.data.mocap_pos[36].copy() + dataset_relative_pos['left_arm'][2][:, flag_animation,:].squeeze()
        self.sim.data.mocap_quat[37] = dataset_relative_quat['left_arm'][3][:, flag_animation, :].squeeze()

    def plot_draw(self):
        if self.count == 0:
            self.x = []
            self.y_r = []
            self.y_l = []
            self.exo_t_l = []
            self.exo_t_r = []
        self.qfrc_norm_r += np.linalg.norm(self.qfrc_actuator_r)
        self.x.append(self.count)
        self.y_r.append(self.qfrc_actuator_r.__float__())
        self.y_l.append(self.qfrc_actuator_l.__float__())
        self.exo_t_r.append(self.exo_torque_r.__float__())
        self.exo_t_l.append(self.exo_torque_l.__float__())
        if self.flag_animation >= self.length_frame - 1:
            self.n += 1
            self.qfrc_actuator_l_avrage = ([x * (self.n - 1) / self.n for x in self.qfrc_actuator_l_avrage]
                                           + [x * 1 / self.n for x in self.y_l])
            self.qfrc_actuator_r_avrage = ([x * (self.n - 1) / self.n for x in self.qfrc_actuator_r_avrage]
                                           + [x * 1 / self.n for x in self.y_r])
            if self.n == self.N:
                if self.run_exo:
                    qfrc_actuator_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/qfrc_actuator_with_exo.yaml"
                else:
                    qfrc_actuator_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/qfrc_actuator_with_zero_exo.yaml"
                data_to_save_yaml = {
                    'x': self.x,
                    'qfrc_actuator_r': self.y_r,
                    'qfrc_actuator_l': self.y_l,
                    'exo_torque_l': self.exo_t_l,
                    'exo_torque_r': self.exo_t_r,
                }
                with open(qfrc_actuator_with_exo_path,'w') as file:
                    yaml.safe_dump(data_to_save_yaml, file, default_flow_style=False)
                self.n = 0
                print("已记录", self.N, "次平均值", qfrc_actuator_with_exo_path)

    def calculate_hight_weight(self):
        foot_pos = self.sim.data.site_xpos[self.sites_id['SMPL_10']].copy()
        head_pos = self.sim.data.site_xpos[self.sites_id['head_max']].copy()

        height = head_pos[2] - foot_pos[2]
        total_mass = 0.0
        body_records = []
        # max_pos = sim.data.body_xpos[bodys_id['']]
        # 遍历所有body（包括worldbody）
        for body_id in range(self.sim.model.nbody):
            # 获取body名称
            body_name = self.sim.model.body(body_id).name

            # 跳过特定前缀的body
            if body_name.startswith(('mocap', 'SMPL','Exo','table')):
                continue

            # 获取质量（注意：worldbody的mass为0）
            mass = self.sim.model.body_mass[body_id]
            if (mass == 0.001):
                continue
            # 记录有效body信息
            body_records.append({
                'name': body_name,
                'id': body_id,
                'mass': mass
            })
            # 累加总质量
            total_mass += mass
        print("human_height", height)
        print("human_mass", total_mass)



    def _load_dataset(self,sim):
        data_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data"
        data_path_with_base_quat = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/imitation_data"
        left_leg = tuple([0, 1, 4, 7, 10])
        right_leg = tuple([0, 2, 5, 8, 11])
        right_arm = tuple([12, 17, 19, 21, 41])
        left_arm = tuple([12, 16, 18, 20, 26])
        body = tuple([0, 12])
        joints = ['box_move_tx', 'pelvis_tx', 'lumbar_rotation', 'sternoclavicular_r2', 'md5_flexion_mirror',
                  'elbow_flexion']
        actuators = ['FPL', 'FDS5', 'FDS4', 'FDS3', 'FDS2', 'FDP5', 'FDP4', 'FDP3', 'FDP2', 'FDS5_mirror',
                     'FDS4_mirror','FDS3_mirror', 'FDS2_mirror', 'FDP5_mirror', 'FDP4_mirror', 'FDP3_mirror'
                    , 'FDP2_mirror', 'FPL_mirror']
        bodys = ['SMPL_0', 'SMPL_1', 'SMPL_2', 'SMPL_4', 'SMPL_5', 'SMPL_7', 'SMPL_8', 'SMPL_10', 'SMPL_11',
                 'SMPL_12', 'SMPL_16', 'SMPL_18', 'SMPL_20', 'SMPL_17', 'SMPL_19', 'SMPL_21', 'SMPL_41', 'SMPL_26',
                 'mocap_0', 'mocap_1', 'mocap_2', 'mocap_4', 'mocap_5', 'mocap_7', 'mocap_8', 'mocap_10',
                 'mocap_11',
                 'mocap_12', 'mocap_16', 'mocap_18', 'mocap_20', 'mocap_17', 'mocap_19', 'mocap_21', 'pelvis',
                 'grap_1', 'grap_2', 'box_move', 'table', 'proximal_thumb', 'secondmc', 'thirdmc', 'fourthmc',
                 'fifthmc',
                 'proximal_thumb_mirror', 'secondmc_mirror', 'thirdmc_mirror', 'fourthmc_mirror', 'fifthmc_mirror']
        sites = ['SMPL_10', 'head_max']
        muscle_dict = {

            'FDS': ['FDS5', 'FDS4', 'FDS3', 'FDS2'],
            'FDP': ['FDP5', 'FDP4', 'FDP3', 'FDP2'],
            'FDS_mirror': ['FDS5_mirror', 'FDS4_mirror', 'FDS3_mirror', 'FDS2_mirror'],
            'FDP_mirror': ['FDP5_mirror', 'FDP4_mirror', 'FDP3_mirror', 'FDP2_mirror'],

        }
        dataset_relative_pos = {'left_leg': [], 'right_leg': [], 'right_arm': [], 'left_arm': [], 'body': [],
                                'base_pos': []}
        dataset_relative_quat = {'left_leg': [], 'right_leg': [], 'right_arm': [], 'left_arm': [], 'body': [],
                                 'base_rot': []}

        model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml"
        sim, joints_id, actuators_id, bodys_id, self.sites_id = load_model(model_path, joints, actuators, bodys, sites)
        for key, value in muscle_dict.items():
            N_muscle = len(value)
            for i in range(N_muscle):
                muscle_dict[key][i] = actuators_id[muscle_dict[key][i]]

        name = "bend.pkl"
        file_path = os.path.join(data_path, name)
        file_path_quat = os.path.join(data_path_with_base_quat, name)
        with open(file_path, "rb") as f:
            two_modata = pickle.load(f)
        with open(file_path_quat, "rb") as f:
            two_modata_with_quat = pickle.load(f)
        two_modata_with_quat = two_modata
        base_relative_pos = {'left_leg': [], 'left_leg_norm': [],
                             'right_leg': [], 'right_leg_norm': [],
                             'left_leg': [], 'left_leg_norm': [],
                             'right_arm': [], 'right_arm_norm': [],
                             'left_arm': [], 'left_arm_norm': [],
                             'body': [], 'body_norm': []}
        base_relative_quat = {'left_leg': [], 'right_leg': [], 'right_arm': [], 'left_arm': [], 'body': []}
        base_pos = {'left_leg': [], 'right_leg': [], 'right_arm': [], 'left_arm': [], 'body': []}
        base_quat = {'left_leg': [], 'right_leg': [], 'right_arm': [], 'left_arm': [], 'body': []}
        base_camera = two_modata_with_quat['base_rot_relative'][0, 0, :]
        angle = -90
        angle = math.radians(angle)
        sim.forward()
        for i in range(len(body) - 1):
            sim.reset()
            sim.data.qpos[:] = np.zeros_like(sim.data.qpos[:].copy())
            sim.data.qvel[:] = np.zeros_like(sim.data.qvel[:].copy())
            sim.forward()
            body_father_left = 'SMPL_' + str(body[i])
            body_child_left = 'SMPL_' + str(body[i + 1])

            dataset_body = two_modata[body][i]
            dataset_body_norm = dataset_body / np.linalg.norm(dataset_body, axis=-1, keepdims=True)

            relative_body = sim.data.xpos[bodys_id[body_child_left]].copy() \
                            - sim.data.xpos[bodys_id[body_father_left]].copy()

            relative_body_quat = qbetween_np(np.array([0, 0, 1]), relative_body)
            dataset_body_scale = np.linalg.norm(relative_body) * dataset_body_norm

            relative_left_broad = np.broadcast_to(np.array([0, 0, 1]), dataset_body_scale.shape).copy()
            body_rot = qbetween_np(relative_left_broad, dataset_body_scale)

            base_relative_pos['body'].append(relative_body.tolist())
            base_relative_pos['body_norm'].append(np.linalg.norm(relative_body).tolist())
            base_relative_quat['body'].append(relative_body_quat.tolist())

            base_pos['body'].append(sim.data.xpos[bodys_id[body_father_left]].copy().tolist())
            base_quat['body'].append(sim.data.xquat[bodys_id[body_father_left]].copy().tolist())

            if (i == len(body) - 2):
                base_pos['body'].append(sim.data.xpos[bodys_id[body_child_left]].copy().tolist())
                base_quat['body'].append(sim.data.xquat[bodys_id[body_child_left]].copy().tolist())

            dataset_relative_pos['body'].append(dataset_body_scale)
            dataset_relative_quat['body'].append(body_rot)

        for i in range(len(left_arm) - 1):
            left_arm_father_left = 'SMPL_' + str(left_arm[i])
            right_arm_father_right = 'SMPL_' + str(right_arm[i])
            left_arm_child_left = 'SMPL_' + str(left_arm[i + 1])
            right_arm_child_right = 'SMPL_' + str(right_arm[i + 1])

            left_leg_father_left = 'SMPL_' + str(left_leg[i])
            right_leg_father_right = 'SMPL_' + str(right_leg[i])
            left_leg_child_left = 'SMPL_' + str(left_leg[i + 1])
            right_leg_child_right = 'SMPL_' + str(right_leg[i + 1])

            dataset_left_arm = two_modata[left_arm][i]
            dataset_right_arm = two_modata[right_arm][i]
            dataset_left_leg = two_modata[left_leg][i]
            dataset_right_leg = two_modata[right_leg][i]

            plv_pos = two_modata['base_pose']
            shoulder_dif = two_modata['base_rot_relative']

            dataset_left_arm_norm = dataset_left_arm / np.linalg.norm(dataset_left_arm, axis=-1, keepdims=True)
            dataset_right_arm_norm = dataset_right_arm / np.linalg.norm(dataset_right_arm, axis=-1, keepdims=True)
            dataset_left_leg_norm = dataset_left_leg / np.linalg.norm(dataset_left_leg, axis=-1, keepdims=True)
            dataset_right_leg_norm = dataset_right_leg / np.linalg.norm(dataset_right_leg, axis=-1, keepdims=True)

            dataset_base_rot = qbetween_np(np.broadcast_to(np.array([0, -1, 0]), shoulder_dif.shape).copy(),
                                           shoulder_dif)
            if (left_leg_father_left == 'SMPL_0'):
                error = sim.data.xpos[bodys_id[left_leg_father_left]].copy() - plv_pos[0, 0, :]
                dataset_base_pos = plv_pos + error
                dataset_relative_pos['base_pos'].append(dataset_base_pos)
                dataset_relative_quat['base_rot'].append(dataset_base_rot)

            # leg_process
            relative_left_leg = sim.data.xpos[bodys_id[left_leg_child_left]].copy() \
                                - sim.data.xpos[bodys_id[left_leg_father_left]].copy()
            relative_right_leg = sim.data.xpos[bodys_id[right_leg_child_right]].copy() \
                                 - sim.data.xpos[bodys_id[right_leg_father_right]].copy()
            if (left_leg_father_left == 'SMPL_7'):
                base_vector = np.array([1, 0, 0])
            else:
                base_vector = np.array([0, 0, -1])
            relative_left_leg_quat = qbetween_np(base_vector, relative_left_leg)
            relative_right_leg_quat = qbetween_np(base_vector, relative_right_leg)
            dataset_left_leg_scale = np.linalg.norm(relative_left_leg) * dataset_left_leg_norm
            dataset_right_leg_scale = np.linalg.norm(relative_right_leg) * dataset_right_leg_norm
            relative_left_broad = np.broadcast_to(base_vector, dataset_left_leg_scale.shape).copy()
            relative_right_broad = np.broadcast_to(base_vector, dataset_right_leg_scale.shape).copy()
            left_leg_rot = qbetween_np(relative_left_broad, dataset_left_leg_scale)
            right_leg_rot = qbetween_np(relative_right_broad, dataset_right_leg_scale)
            base_relative_pos['left_leg'].append(relative_left_leg.tolist())
            base_relative_pos['right_leg'].append(relative_right_leg.tolist())
            base_relative_pos['left_leg_norm'].append(np.linalg.norm(relative_left_leg).tolist())
            base_relative_pos['right_leg_norm'].append(np.linalg.norm(relative_right_leg).tolist())
            base_relative_quat['left_leg'].append(relative_left_leg_quat.tolist())
            base_relative_quat['right_leg'].append(relative_right_leg_quat.tolist())
            base_pos['left_leg'].append(sim.data.xpos[bodys_id[left_leg_father_left]].copy().tolist())
            base_pos['right_leg'].append(sim.data.xpos[bodys_id[right_leg_father_right]].copy().tolist())
            base_quat['left_leg'].append(sim.data.xquat[bodys_id[left_leg_father_left]].copy().tolist())
            base_quat['right_leg'].append(sim.data.xquat[bodys_id[right_leg_father_right]].copy().tolist())
            if (i == len(left_arm) - 2):
                base_pos['left_leg'].append(sim.data.xpos[bodys_id[left_leg_child_left]].copy().tolist())
                base_pos['right_leg'].append(sim.data.xpos[bodys_id[right_leg_child_right]].copy().tolist())
                base_quat['left_leg'].append(sim.data.xquat[bodys_id[left_leg_child_left]].copy().tolist())
                base_quat['right_leg'].append(sim.data.xquat[bodys_id[right_leg_child_right]].copy().tolist())

            dataset_relative_pos['left_leg'].append(dataset_left_leg_scale)
            dataset_relative_pos['right_leg'].append(dataset_right_leg_scale)
            dataset_relative_quat['left_leg'].append(left_leg_rot)
            dataset_relative_quat['right_leg'].append(right_leg_rot)

            # arm_process
            relative_left_arm = sim.data.xpos[bodys_id[left_arm_child_left]].copy() \
                                - sim.data.xpos[bodys_id[left_arm_father_left]].copy()
            relative_right_arm = sim.data.xpos[bodys_id[right_arm_child_right]].copy() \
                                 - sim.data.xpos[bodys_id[right_arm_father_right]].copy()
            if (left_arm_father_left == 'SMPL_12'):
                base_vector_left = np.array([0, -1, 0])
                base_vector_right = np.array([0, 1, 0])
            else:
                base_vector_left = np.array([0, 0, -1])
                base_vector_right = np.array([0, 0, -1])
            relative_left_arm_quat = qbetween_np(base_vector_left, relative_left_arm)
            relative_right_arm_quat = qbetween_np(base_vector_right, relative_right_arm)
            dataset_left_arm_scale = np.linalg.norm(relative_left_arm) * dataset_left_arm_norm
            dataset_right_arm_scale = np.linalg.norm(relative_right_arm) * dataset_right_arm_norm
            relative_left_broad = np.broadcast_to(base_vector_left, dataset_left_arm_scale.shape).copy()
            relative_right_broad = np.broadcast_to(base_vector_right, dataset_right_arm_scale.shape).copy()
            left_arm_rot = qbetween_np(relative_left_broad, dataset_left_arm_scale)
            right_arm_rot = qbetween_np(relative_right_broad, dataset_right_arm_scale)
            base_relative_pos['left_arm'].append(relative_left_arm.tolist())
            base_relative_pos['right_arm'].append(relative_right_arm.tolist())
            base_relative_pos['left_arm_norm'].append(np.linalg.norm(relative_left_arm).tolist())
            base_relative_pos['right_arm_norm'].append(np.linalg.norm(relative_right_arm).tolist())
            base_relative_quat['left_arm'].append(relative_left_arm_quat.tolist())
            base_relative_quat['right_arm'].append(relative_right_arm_quat.tolist())
            base_pos['left_arm'].append(sim.data.xpos[bodys_id[left_arm_father_left]].copy().tolist())
            base_pos['right_arm'].append(sim.data.xpos[bodys_id[right_arm_father_right]].copy().tolist())
            base_quat['left_arm'].append(sim.data.xquat[bodys_id[left_arm_father_left]].copy().tolist())
            base_quat['right_arm'].append(sim.data.xquat[bodys_id[right_arm_father_right]].copy().tolist())
            if (i == len(left_arm) - 2):
                base_pos['left_arm'].append(sim.data.xpos[bodys_id[left_arm_child_left]].copy().tolist())
                base_pos['right_arm'].append(sim.data.xpos[bodys_id[right_arm_child_right]].copy().tolist())
                base_quat['left_arm'].append(sim.data.xquat[bodys_id[left_arm_child_left]].copy().tolist())
                base_quat['right_arm'].append(sim.data.xquat[bodys_id[right_arm_child_right]].copy().tolist())

            dataset_relative_pos['left_arm'].append(dataset_left_arm_scale)
            dataset_relative_pos['right_arm'].append(dataset_right_arm_scale)
            dataset_relative_quat['left_arm'].append(left_arm_rot)
            dataset_relative_quat['right_arm'].append(right_arm_rot)

        print("dataset处理完")
        return dataset_relative_pos,dataset_relative_quat, muscle_dict, actuators_id


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
    obs["local_body_angle_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    return obs


def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    # env_names = [spec for spec in gym.envs.registry]
    # for name in sorted(env_names):
    #     print(name)
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

