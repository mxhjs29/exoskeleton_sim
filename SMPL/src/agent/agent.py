import math
import time
import os
import torch
import numpy as np

import torch.multiprocessing as multiprocessing

import gymnasium as gym

from SMPL.src.learning.memory import Memory
from SMPL.src.learning.trajbatch import TrajBatch
from SMPL.src.learning.logger_rl import LoggerRL
from SMPL.src.learning.learning_utils import to_test, to_cpu, rescale_actions
import random
random.seed(0)

from typing import Any, Optional, List

os.environ["OMP_NUM_THREADS"] = "1"

done = multiprocessing.Event()
class Agent():
    def __init__(self,
                 env: gym.Env,
                 policy_net: torch.nn.Module,
                 value_net: torch.nn.Module,
                 dtype: torch.dtype,
                 device: torch.device,
                 gamma: float,
                 mean_action: bool = False,
                 headless: bool = False,
                 num_threads: int = 1,
                 clip_obs: bool = False,
                 clip_actions: bool = False,
                 clip_obs_range: Optional[List[float]] = None,
                 ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.np_dtype = np.float32
        self.gamma = gamma
        self.mean_action = mean_action
        self.headless = headless
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.num_steps = 0




















