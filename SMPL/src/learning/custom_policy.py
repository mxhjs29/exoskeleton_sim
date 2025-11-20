import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class HumanModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_size = int(np.product(obs_space.shape))

        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        # 策略输出（logits），长度 = num_outputs（离散动作）或 action_dim（连续动作）
        self.policy_head = nn.Linear(256, num_outputs)
        # 价值函数输出 V(s)
        self.value_head = nn.Linear(256, 1)

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        # RLlib 已经帮你把 obs flatten 成 obs_flat 了
        x = input_dict["obs_flat"]
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        logits = self.policy_head(x)
        # 保存 value 给 value_function() 用
        self._last_value = self.value_head(x).squeeze(-1)

        return logits, state

    def value_function(self):
        # 必须返回 shape=[B] 的 1D tensor
        return self._last_value
    
class ExoModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_size = int(np.product(obs_space.shape))

        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        # 策略输出（logits），长度 = num_outputs（离散动作）或 action_dim（连续动作）
        self.policy_head = nn.Linear(64, num_outputs)
        # 价值函数输出 V(s)
        self.value_head = nn.Linear(64, 1)

        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        # RLlib 已经帮你把 obs flatten 成 obs_flat 了
        x = input_dict["obs_flat"]
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.policy_head(x)
        # 保存 value 给 value_function() 用
        self._last_value = self.value_head(x).squeeze(-1)

        return logits, state

    def value_function(self):
        # 必须返回 shape=[B] 的 1D tensor
        return self._last_value