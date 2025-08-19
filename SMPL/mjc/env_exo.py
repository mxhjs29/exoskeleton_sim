import collections
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from  myosuite.utils import gym
import yaml
from myosuite.envs.obs_vec_dict import ObsVecDict
from myosuite.utils.implement_for import implement_for
from myosuite.utils.prompt_utils import prompt,Prompt
from myosuite.envs.env_variants import gym_registry_specs, register_env_variant, register
from SMPL.common.model_load import *
import sys
import pickle
sys.path.append("/home/chenshuo/PycharmProjects/move_sim/SMPL")

plt.rcParams['axes.unicode_minus'] = False
file_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/initial_state_with_exo.yaml"
with open(file_path,'r') as file:
    initial_joint = yaml.safe_load(file)
joint_qpos = np.array(initial_joint['qpos'])
joint_qvel = np.array(initial_joint['qvel'])

obs_true_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/mjc/obs_data'

class EnvCarry_WithExo(gym.Env, gym.utils.EzPickle, ObsVecDict):
    DEFAULT_CREDIT = """\
            MyoSuite: a collection of environments/tasks to be solved by musculoskeletal models | https://sites.google.com/view/myosuite
            Code: https://github.com/MyoHub/myosuite/stargazers (add a star to support the project)
        """
    metadata = {"render_modes": ['human','rgb_array'], "render_fps": 60}
    def __init__(self, model_path, seed=None,env_credits=DEFAULT_CREDIT,max_episode_steps = 2500, n_stacks=4, **kwargs):
        prompt("MyoSuite:> For environment credits, please cite -")
        prompt(env_credits, color="cyan", type=Prompt.ONCE)

        self.seed(seed)
        self.n_stacks = n_stacks

        self.stacked_obs = None
        self.sim = SimScene.get_sim(model_path)
        # 后续改为肌肉对应的actuator？
        self.obs_joints = ['pelvis_tx', 'lumbar_rotation']
        self.obs_joints_id = {name: self.sim.model.joint(name).id for name in self.obs_joints}
        self.actuators = ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagDist_r', 'addmagIsch_l', 'addmagIsch_r', 'addmagMid_l', 'addmagMid_r', 'addmagProx_l', 'addmagProx_r', 'bflh140_l', 'bflh140_r', 'bfsh140_l', 'bfsh140_r', 'edl_l', 'edl_r', 'ehl_l', 'ehl_r', 'ercspn_l', 'ercspn_r', 'extobl_l', 'extobl_r', 'fdl_l', 'fdl_r', 'fhl_l', 'fhl_r', 'gaslat140_l', 'gaslat140_r', 'gasmed_l', 'gasmed_r', 'gem_l', 'gem_r', 'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 'glmed1_r', 'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r', 'glmin3_l', 'glmin3_r', 'grac_l', 'grac_r', 'iliacus_l', 'iliacus_r', 'intobl_l', 'intobl_r', 'obtext_l', 'obtext_r', 'obtint_l', 'obtint_r', 'pect_l', 'pect_r', 'perbrev_l', 'perbrev_r', 'perlong_l', 'perlong_r', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'quadfem_l', 'quadfem_r', 'recfem_l', 'recfem_r', 'rectab_l', 'rectab_r', 'sart_l', 'sart_r', 'semimem_l', 'semimem_r', 'semiten_l', 'semiten_r', 'soleus_l', 'soleus_r', 'tfl_l', 'tfl_r', 'tibant_l', 'tibant_r', 'tibpost_l', 'tibpost_r', 'vasint_l', 'vasint_r', 'vaslat140_l', 'vaslat140_r', 'vasmed_l', 'vasmed_r']
        self.actuators_id = {name: self.sim.model.actuator(name).id for name in self.actuators}
        self.knee_joints = ['walker_knee_l_translation1','walker_knee_l_translation2','walker_knee_l_translation3',
                            'knee_angle_l','walker_knee_l_rotation2','walker_knee_l_rotation3',
                            'walker_knee_r_rotation3','walker_knee_r_rotation2','knee_angle_r',
                            'walker_knee_r_translation3','walker_knee_r_translation2','walker_knee_r_translation1']
        self.knee_joints_id = {name: self.sim.model.joint(name).id for name in self.knee_joints}
        self.exo_joints = ['jActuatedRightHip_rotx','jActuatedLeftHip_rotx']
        self.exo_joints_id = {name: self.sim.model.joint(name).id for name in self.exo_joints}
        self.hip_joints = ['hip_flexion_r','hip_adduction_r','hip_rotation_r',
                          'hip_flexion_l','hip_adduction_l','hip_rotation_l']
        self.hip_joints_id = {name: self.sim.model.joint(name).id for name in self.hip_joints}
        # 初始化，exo关节数据有没有
        self.sim.data.qpos[:] = joint_qpos[0]
        qvel = np.zeros(len(joint_qvel[0]))
        self.sim.data.qvel[:] = qvel
        self.sim.forward()
        ObsVecDict.__init__(self)
        self.dataset_relative_pos, self.dataset_relative_quat, self.grap_actuators, self.grap_actuators_id = self._load_dataset(self.sim)
        self.flag_animation = 0
        self.end_flag = 0
        self.max_episode_steps = max_episode_steps
        self.exo_force_r = 0
        self.exo_force_l = 0
        self.calculate_hight_weight()
        self.x = []
        self.y_r = []
        self.y_l = []
        self.qfrc_norm_r = 0
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = None,
               weighted_reward_keys: dict = {},
               proprio_keys: list = None,
               render_bool: bool = True,
               length_data: int = 10000,
               frame_skip: int = 1,
               normalize_act: bool = True,
               rwd_viz: bool = False,
               device_id: int = 0,
               scale: int = 5,
               apply_muscle_activate: bool = True,
               kp: int = 100,
               kv: int = 5,
               offline_render: bool = False,
               ratio: int = 0,
               apply_exo_force: bool = True,
            ):

        if self.sim is None:
            raise TypeError("sim must be instantiated for setup to run")
        # resolve view
        self.mujoco_render_frames = False
        self.offline_render = offline_render
        self.device_id = device_id
        self.rwd_viz = rwd_viz
        self.viewer_setup()
        self.length_data = length_data
        # resolve action space
        self.frame_skip = frame_skip
        self.ratio = ratio
        self.render_bool = render_bool
        self.normalize_act = normalize_act
        act_low = -np.ones(self.sim.model.na - 18) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,0].copy()
        act_high = np.ones(self.sim.model.na - 18) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,1].copy()
        self.action_space = gym.spaces.Box(act_low,act_high,dtype=np.float64)
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
        # obs_keys=['qpos']
        self.obs_keys = obs_keys
        # resolve proprio
        self.proprio_dict = {}
        self.proprio_dict = {}
        self.proprio_keys = proprio_keys if type(proprio_keys) == list or proprio_keys == None else [proprio_keys]

        self.index = 0
        length_pos, qpos_gt, qvel_gt = obs_data_load(obs_true_path, self.index)
        self.length_data = length_pos
        self.qpos_gt = qpos_gt
        self.qvel_gt = qvel_gt
        self.count = 0
        done = False
        self.kp = kp
        self.kv = kv
        self.apply_exo_force = apply_exo_force
        self.qfrc_actuator_r = 0
        self.qfrc_actuator_l = 0
        self.exo_force_r = 0
        self.exo_force_l = 0
        self.exo_force_r_last = 0
        self.exo_force_l_last = 0
        self.n = 0
        self.N = 2
        self.run_exo = False
        self.qfrc_actuator_r_avrage = []
        self.qfrc_actuator_l_avrage = []
        observation = self.reset()
        assert not done,"Check initialization. Simulation starts in a done state."
        self.observation_space = gym.spaces.Box(
                                    low=-np.inf * np.ones(len(observation[0])),
                                    high=np.inf * np.ones(len(observation[0])),
                                    dtype=np.float64)

        return

    def step(self,a, **kwargs):

        a = np.clip(a, self.action_space.low, self.action_space.high)
        a = (a + 1.) / 2.
        if self.count % 2 == 0 and self.flag_animation < self.length_data:
            # mocap的更新频率要慢一些
            self.applied_uplimb_mocap(self.dataset_relative_pos,self.dataset_relative_quat,self.flag_animation)
            self.flag_animation += 1

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

        if self.apply_exo_force:
            self.qfrc_actuator_r = (self.sim.data.qfrc_actuator[self.hip_joints_id['hip_flexion_r']]
                                  - self.sim.data.qfrc_bias[self.hip_joints_id['hip_flexion_r']]
                                  + self.sim.data.qfrc_passive[self.hip_joints_id['hip_flexion_r']])
            self.qfrc_actuator_l = (self.sim.data.qfrc_actuator[self.hip_joints_id['hip_flexion_l']]
                                    - self.sim.data.qfrc_bias[self.hip_joints_id['hip_flexion_l']]
                                    + self.sim.data.qfrc_passive[self.hip_joints_id['hip_flexion_l']])
            if np.abs(self.qfrc_actuator_l) < 50:
                self.exo_force_l = np.clip(0.05 * self.qfrc_actuator_l, -5, 5)
            elif np.abs(self.qfrc_actuator_l) < 150:
                self.exo_force_l = np.clip(0.2 * self.qfrc_actuator_l, -30, 30)
            else:
                self.exo_force_l = self.qfrc_actuator_l * 0.4

            if np.abs(self.qfrc_actuator_r) < 50:
                self.exo_force_r = np.clip(0.05 * self.qfrc_actuator_r, -5, 5)
            elif np.abs(self.qfrc_actuator_r) < 150:
                self.exo_force_r = np.clip(0.2 * self.qfrc_actuator_r, -30, 30)
            else:
                self.exo_force_r =  self.qfrc_actuator_r * 0.4

            self.exo_force_l =  0 * self.exo_force_l_last +  self.exo_force_l
            self.exo_force_r =   0 * self.exo_force_r_last +  self.exo_force_r
            self.exo_force_l_last = self.exo_force_l
            self.exo_force_r_last = self.exo_force_r
            if self.run_exo:
                self.sim.data.ctrl[0] = self.exo_force_l
                self.sim.data.ctrl[1] = self.exo_force_r
            else:
                self.sim.data.ctrl[0] = 0
                self.sim.data.ctrl[1] = 0

        self.plot_draw()
        if self.apply_muscle_activate:
            self.sim.advance(substeps=self.frame_skip, render=self.render_bool)
            self.sim.forward()
            self.count = self.count + 1
            if (self.offline_render):
                self.frame = self.render_offscreen()
            else:
                self.frame = None
            self.muscle_activate = []

        self.last_ctrl = a

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
        if self.flag_animation >= self.length_data-1:
            self.flag_animation = 0
        return stack_obs, env_info['rwd_pose'] + env_info['rwd_vel'], env_info['done'], truncated, env_info

    def get_obs(self,update_proprioception=True, update_exteroception=False, **kwargs):
        self.obs_dict = self.get_obs_dict()
        if update_proprioception:
            self.proprio_dict = self.get_proprioception(self.obs_dict)[2]
        t,obs = self.obsdict2obsvec(self.obs_dict,self.obs_keys)
        return obs

    def _get_stacked_obs(self):
        return np.concatenate(self.stacked_obs, axis=0, dtype=np.float64)

    def get_proprioception(self,obs_dict=None):
        if self.proprio_keys == None:
            return None, None, None
        if obs_dict==None: obs_dict = self.obs_dict
        proprio_vec = np.zeros(0)
        proprio_dict = {}
        proprio_dict['time'] = obs_dict['time']

        for key in self.proprio_keys:
            proprio_vec = np.concatenate([proprio_vec, obs_dict[key]])
            proprio_dict[key] = obs_dict[key]

        return proprio_dict['time'], proprio_vec, proprio_dict

    def get_exteroception(self,**kwargs):
        return self.get_visuals(**kwargs)

    def get_env_info(self):
        env_info = {
            'time': self.obs_dict['time'][()],  # MDP(t)
            'rwd_pose': self.rwd_dict['position'][()],  # MDP(t)
            'rwd_control': self.rwd_dict['control'][()],  # MDP(t)
            'rwd_vel': self.rwd_dict['velocity'][()],
            # 'solved': self.rwd_dict['solved'][()],      # MDP(t)
            'done': self.rwd_dict['done'][()],  # MDP(t)
            'obs_dict': self.obs_dict,  # MDP(t)
            'proprio_dict': self.proprio_dict,  # MDP(t)
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

        length_pos, qpos_gt, qvel_gt = obs_data_load(obs_true_path, self.index)
        combined_array = np.vstack(qpos_gt)
        mean = np.mean(combined_array, axis=0)
        std_dev = np.std(combined_array, axis=0)
        self.length_data = length_pos
        self.qpos_gt = qpos_gt
        self.qvel_gt = qvel_gt
        self.count = 0
        self.flag_animation = 0
        self.end_flag = 0
        self.exo_force_r_last = 0
        self.exo_force_l_last = 0
        qpos = joint_qpos
        qvel = np.zeros(len(joint_qvel[0]))
        self.sim.reset()
        self.sim.data.qpos[:] = qpos[0]
        self.sim.data.qvel[:] = qvel
        self.sim.advance(substeps=self.frame_skip, render=self.render_bool)
        self.sim.forward()
        obs = self.get_obs()
        self.stacked_obs = [obs] * self.n_stacks
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
        return self.sim_obsd.data.time

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

    def set_env_state(self,state_dict):
        time = state_dict['time']
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        act = state_dict['act'] if 'act' in state_dict.keys() else None
        self.sim.set_state(time=time, qpos=qp, qvel=qv, act=act)
        if self.sim.model.nmocap>0:
            self.sim.data.mocap_pos[:] = state_dict['mocap_pos']
            self.sim.data.mocap_quat[:] = state_dict['mocap_quat']

        if self.sim.model.nsite>0:
            self.sim.model.site_pos[:] = state_dict['site_pos']
            self.sim.model.site_quat[:] = state_dict['site_quat']
        self.sim.model.body_pos[:] = state_dict['body_pos']
        self.sim.model.body_quat[:] = state_dict['body_quat']
        self.sim.forward()

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

    def get_obs_dict(self):
        obs_dict = {}
        obs_dict['frame'] = np.array([self.flag_animation])
        obs_dict['qpos_gt'] = self.qpos_gt[self.flag_animation]
        obs_dict['qvel_gt'] = self.qvel_gt[self.flag_animation] * self.dt
        obs_dict['time'] = np.array([self.sim.data.time])
        qpos_obs = self.sim.data.qpos[self.obs_joints_id['pelvis_tx']:self.obs_joints_id['lumbar_rotation']+1]
        obs_dict['qpos'] = qpos_obs.ravel().copy()
        qvel_obs = self.sim.data.qvel[self.obs_joints_id['pelvis_tx']:self.obs_joints_id['lumbar_rotation']+1]
        obs_dict['qvel'] = qvel_obs.ravel().copy() * self.dt
        obs_dict['qpos_error'] = obs_dict['qpos_gt'] - obs_dict['qpos']
        obs_dict['qvel_error'] =  - obs_dict['qvel']
        obs_dict['act'] = self.sim.data.act[:].copy() if self.sim.model.nu > 0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['qfrc_constraint'] = self.sim.data.qfrc_constraint.ravel().copy()
        obs_dict['qfrc_constraint'] = np.nan_to_num(obs_dict['qfrc_constraint'], nan=0, posinf=100, neginf=-100)
        return obs_dict

    def get_reward_dict(self,obs_dict):
        reward_dict = self.rwd_keys_wt.copy()
        else_error = obs_dict['qpos_error']
        else_error_reward = - self.rwd_keys_wt['position'] * np.linalg.norm(else_error)
        reward_dict['position'] =   else_error_reward
        reward_dict['velocity'] = -self.rwd_keys_wt['velocity'] * np.linalg.norm( obs_dict['qvel_error'])
        reward_dict['control'] = -self.rwd_keys_wt['control'] * np.linalg.norm(self.sim.data.ctrl[:].copy())
        if (self.flag_animation >= self.length_data - 1):
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
        self.qfrc_norm_r += np.linalg.norm(self.qfrc_actuator_r)
        self.x.append(self.count)
        self.y_r.append(self.qfrc_actuator_r.__float__())
        self.y_l.append(self.qfrc_actuator_l.__float__())
        if self.flag_animation >= self.length_data - 1:
            self.n += 1
            self.qfrc_actuator_l_avrage = ([x * (self.n - 1) / self.n for x in self.qfrc_actuator_l_avrage]
                                           + [x * 1 / self.n for x in self.y_l])
            self.qfrc_actuator_r_avrage = ([x * (self.n - 1) / self.n for x in self.qfrc_actuator_r_avrage]
                                           + [x * 1 / self.n for x in self.y_r])
            if self.n == self.N:
                if self.run_exo:
                    qfrc_actuator_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/qfrc_actuator_with_exo.yaml"
                else:
                    qfrc_actuator_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/qfrc_actuator_with_zero_exo.yaml"
                    # qfrc_actuator_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/qfrc_actuator_without_exo.yaml"
                data_to_save_yaml = {
                    'x': self.x,
                    'qfrc_actuator_r': self.y_r,
                    'qfrc_actuator_l': self.y_l,
                }
                with open(qfrc_actuator_with_exo_path,'w') as file:
                    yaml.safe_dump(data_to_save_yaml, file, default_flow_style=False)
                self.n = 0
                print("已记录", self.N, "次平均值")

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
        data_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL"
        data_path_with_base_quat = "/home/chenshuo/PycharmProjects/move_sim/SMPL"
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

        model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/mjc/mj_fullbody_with_exo_carrying.xml"
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

register_env_with_variants(
    id="EnvCarry_WithoutExo-v0",
    entry_point="SMPL.mjc.env_without_exo:EnvCarry_WithoutExoV0",
    max_episode_steps=6000,
    kwargs={
        'model_path': '/home/chenshuo/PycharmProjects/move_sim/SMPL/mjc/mj_fullbody_without_exo_carrying.xml',
        'normalize_act': True,
        'obs_keys': ['qpos','qvel'],
        'weighted_reward_keys': {'position': 1,
                                 'velocity': 1,
                                 'control': 1}
    }
)