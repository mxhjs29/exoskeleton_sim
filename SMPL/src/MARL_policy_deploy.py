"""
在实机上使用部署的策略进行推理
无需 Ray 训练框架，只需加载权重即可
"""

import pickle
import numpy as np
from pathlib import Path
import torch
import importlib
from SMPL.src.env.MARL_env import MARL_EXO_Env
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

def raw_env_creator(config):
    return MARL_EXO_Env(render_mode = config.get("render_mode","human"),
                            render_bool = config.get("render_bool",False),
                            frame_skip = config.get("frame_skip",2),
                            model_path = config.get("model_path",None),
                            n_stacks = config.get("n_stacks",1),
                            human_obs_keys = config.get("human_obs_keys"),
                            exo_obs_keys = config.get("exo_obs_keys"),
                            weighted_reward_keys = config.get("weighted_reward_keys")
                        )

def env_creator(config):
    return ParallelPettingZooEnv(raw_env_creator(config))

register_env("MARL_EXO_ENV", env_creator)

class RobotPolicyInference():
    """轻量级的实机推理引擎"""
    
    def __init__(self, deploy_dir: str = "/home/chenshuo/PycharmProjects/move_sim/deployed_policy"):
        """
        初始化实机推理引擎
        
        Args:
            deploy_dir: 部署文件目录
        """
        self.deploy_path = Path(deploy_dir)
        self.exo_policy = None
        self.human_policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] 使用设备: {self.device}")
        self._load_policies()
    
    def _load_single_policy(self, policy_name: str):
        """从 RLlib 导出的策略目录加载模型与权重"""
        print(f"[INFO] 加载 {policy_name} ...")

        ctor_file = (
            self.deploy_path / f"learner_group/learner/rl_module/{policy_name}/class_and_ctor_args.pkl"
        )
        state_file = (
            self.deploy_path / f"learner_group/learner/rl_module/{policy_name}/module_state.pkl"
        )

        if not ctor_file.exists() or not state_file.exists():
            raise FileNotFoundError(f"[ERROR] {policy_name} 文件缺失：{ctor_file} 或 {state_file}")


        with open(ctor_file, "rb") as f:
            ctor_info = pickle.load(f)
        with open(state_file, "rb") as f:
            module_state = pickle.load(f)

        module_class = ctor_info["class"]
        args, kwargs = ctor_info["ctor_args_and_kwargs"]

        if isinstance(module_class, str):
            module_name, class_name = module_class.rsplit(".", 1)
            module_lib = importlib.import_module(module_name)
            module_class = getattr(module_lib, class_name)

        model = module_class(*args, **kwargs)

        if isinstance(module_state, dict):
            for k, v in module_state.items():
                if isinstance(v, np.ndarray):
                    module_state[k] = torch.from_numpy(v)
        else:
            raise ValueError(f"[ERROR] {policy_name} 的 state 文件不是有效 state_dict。")
        model.load_state_dict(module_state)
        model = model.to(self.device)

        for name, param in model.named_parameters():
            print(f"{name:50s}  {tuple(param.shape)}")

        print(f"  ✓ {policy_name} 权重已成功加载。")
        return model

    def _load_policies(self):
        self.exo_policy = self._load_single_policy("exo_policy")
        self.human_policy = self._load_single_policy("human_policy")
        print("[INFO] 所有策略加载完成。")
    
    def _logits_to_action(self, logits: torch.Tensor,
                      deterministic: bool = True) -> torch.Tensor:
        """
        logits: shape [B, 2*act_dim] = [mean, log_std]
        返回:   shape [B, act_dim]，已经被 squash 到 [-1,1] 并线性映射到 [low, high]
        """
        B, two_d = logits.shape
        act_dim = two_d // 2
        mean, log_std = torch.split(logits, act_dim, dim=1)
        std = torch.exp(log_std).clamp_min(1e-6)

        if deterministic:
            a_pre_tanh = mean
        else:
            eps = torch.randn_like(mean)
            a_pre_tanh = mean + std * eps

        # 1) squash 到 [-1, 1]
        a = torch.tanh(a_pre_tanh)

        return a

    def compute_exo_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # 如训练用了 MeanStdFilter，这里先做同样的标准化：
        # observation = self.exo_filter(observation, update=False)

        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.exo_policy({"obs": obs})
            logits = out["action_dist_inputs"]     # shape [1, 2*act_dim]
            act = self._logits_to_action(logits, deterministic=deterministic)
        return act.squeeze(0).cpu().numpy()


    def compute_human_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # observation = self.human_filter(observation, update=False)

        obs = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.human_policy({"obs": obs})
            logits = out["action_dist_inputs"]     # shape [1, 2*98]
            act = self._logits_to_action(logits, deterministic=deterministic)
        return act.squeeze(0).cpu().numpy()
    
    def compute_actions(self, observations: dict) -> dict:
        """
        计算所有策略的动作
        
        Args:
            observations: 各策略的观测值字典
            
        Returns:
            各策略的动作字典
        """
        actions = {}
        actions["exo"] = self.compute_exo_action(observations["exo"])
        actions["human"] = self.compute_human_action(observations["human"])
        return actions

def butter_lowpass_zero_phase(x, cutoff_norm=0.1, order=3):
    b, a = butter(order, Wn=cutoff_norm, btype="low", analog=False)
    y = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[1]):
        y[:, i] = filtfilt(b, a, x[:, i])
    return y

def standardize(obs):
    obs_human_mean = obs["human"].mean()
    obs_human_std = obs["human"].std()
    obs_exo_mean = obs["exo"].mean()
    obs_exo_std = obs["exo"].std()
    obs["human"] = (obs["human"] - obs_human_mean) / (obs_human_std)
    obs["exo"] = (obs["exo"] - obs_exo_mean) / (obs_exo_std)
    return obs

if __name__ == "__main__":
    # 初始化推理引擎
    human_obs_keys = ['qpos','local_body_pos','local_body_vel','local_body_rot_obs','local_body_angle_vel']
    exo_obs_keys = ['exo_joint_t','exo_joint_vel_t']
    weighted_reward_keys = {'w_pos_err': 0.6,
                        'w_proprio_err': 0.6,
                        'w_activation': 0.4,
                        'w_exo_energy':0.2,
                        'w_exo_smooth':0.2,
                        'theta_pos_err': 0.9,
                        'theta_proprio_err': 0.55,
                        'theta_activation': 0.1,
                        'theta_exo_energy':0.1,
                        'theta_exo_smooth':3}
    env = MARL_EXO_Env(render_mode = "human",
                            render_bool = False,
                            frame_skip = 2,
                            model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml",
                            n_stacks = 2,
                            human_obs_keys = human_obs_keys,
                            exo_obs_keys = exo_obs_keys,
                            weighted_reward_keys = weighted_reward_keys 
                        )
    obs,info = env.reset()
    policies = RobotPolicyInference("/home/chenshuo/PycharmProjects/move_sim/deployed_policy")
    exo_trace = []
    return_exo = 0
    return_human = 0
    while True:
        actions = policies.compute_actions(obs)
        exo_trace.append(actions["exo"])
        obs, rewards, terminated, truncated, info = env.step(actions)
        return_exo += rewards["exo"]
        return_human += rewards["human"]
        if terminated["__all__"]:
            break
    env.close()
    print(f"Exo Return: {return_exo}")
    print(f"Human Return: {return_human}")
    exo_trace = np.array(exo_trace)
    exo_sg = savgol_filter(exo_trace, window_length=21, polyorder=3, axis=0, mode="interp")
    exo_bw = butter_lowpass_zero_phase(exo_trace, cutoff_norm=0.1, order=3)
    # plt.plot(exo_trace, label="original exo action")
    # plt.plot(exo_sg, label="smoothed exo action")
    plt.plot(exo_bw, label="butterworth filtered exo action")
    plt.title("Exo Action Trace")
    plt.xlabel("Timestep")
    plt.ylabel("Exo Action Value")
    plt.grid(True)
    plt.show()

    
    