"""
在实机上使用部署的策略进行推理
无需 Ray 训练框架，只需加载权重即可
"""

import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import importlib
from SMPL.src.env.MARL_env import MARL_EXO_Env
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

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
        print(f"  ✓ {policy_name} 权重已成功加载。")
        return model

    def _load_policies(self):
        self.exo_policy = self._load_single_policy("exo_policy")
        self.human_policy = self._load_single_policy("human_policy")
        print("[INFO] 所有策略加载完成。")
    
    def compute_exo_action(self, observation: np.ndarray) -> np.ndarray:
        """
        计算 exo 策略的动作
        
        Args:
            observation: exo 的观测值
            
        Returns:
            exo 的动作
        """
        # 这里添加实际的推理逻辑
        # 将观测转换为 torch tensor，通过网络，得到动作
        obs = torch.from_numpy(observation)
        action = self.exo_policy(obs)
        return action
    
    def compute_human_action(self, observation: np.ndarray) -> np.ndarray:
        """
        计算 human 策略的动作
        
        Args:
            observation: human 的观测值
            
        Returns:
            human 的动作
        """
        # 这里添加实际的推理逻辑
        # 将观测转换为 torch tensor，通过网络，得到动作
        obs = torch.from_numpy(observation)
        action = self.human_policy(obs)
        return action


if __name__ == "__main__":
    # 初始化推理引擎
    policies = RobotPolicyInference("/home/chenshuo/PycharmProjects/move_sim/deployed_policy")
    
    