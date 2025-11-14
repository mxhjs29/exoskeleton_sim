from SMPL.src.env.fullbody_env import FullBodyEnv
from omegaconf import DictConfig, OmegaConf

import hydra
import os
import numpy as np
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config",
)
def main(cfg: DictConfig) -> None:

    env = FullBodyEnv(cfg)
    while True:
        action = np.zeros(env.action_space.shape)
        observation, reward, terminated, truncated, info = env.step(action)
        print("observation",observation)



if __name__ == "__main__":
    print("start running")
    main()


