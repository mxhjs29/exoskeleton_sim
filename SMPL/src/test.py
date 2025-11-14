import numpy as np
from myosuite.envs.env_variants import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import time
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco.viewer
from triton.language.extra.cuda.libdevice import fast_logf


model_without_exo_weld_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_without_exo_carrying_policy_test.xml"
model_with_exo_weld_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml"

def create_env(test_exo, vision):
    if test_exo:
        register(
            id="EnvCarry_WithExo-v1",
            entry_point="SMPL.src.env.env_exo:EnvCarry_WithExo",
            kwargs={
                'obs_keys': ['qpos'],
                'model_path': model_with_exo_weld_path,
                'normalize_act': True,
                'max_episode_steps': 2500,
                'render_bool': vision,
                'weighted_reward_keys': {'w_pos_err': 1,
                                         'w_proprio_err': 0.001,
                                         'w_activation': 0.1,
                                         'w_exo_energy':0.01,
                                         'w_exo_smooth':0.01,
                                         'theta_pos_err': 1,
                                         'theta_proprio_err': 0.001,
                                         'theta_activation': 0.1,
                                         'theta_exo_energy':0.01,
                                         'theta_exo_smooth':0.01}
            },
        )
    else:
        register(
            id="EnvCarry_WithoutExo-v1",
            entry_point="SMPL.src.env.env_without_exo_weld:EnvCarry_WithoutExo",
            kwargs={
                'obs_keys': ['qpos'],
                'model_path': model_without_exo_weld_path,
                'normalize_act': True,
                'max_episode_steps': 2500,
                'render_bool': vision,
                'weighted_reward_keys': {'position': 1,
                                         'velocity': 0.001,
                                         'control': 0.1}
            },
        )
        


policy_test = True  
test_exo = True
clip_obs = 10
vision = False

PPO_policy_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/model/best_model/ppo_bend_weld_without_exo/best_model.zip"

qfrc_without_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/qfrc_actuator_without_exo.yaml"
qfrc_with_zero_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/qfrc_actuator_with_zero_exo.yaml"
qfrc_with_exo_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/torque_record/qfrc_actuator_with_exo.yaml"



def main():
    if policy_test:
        create_env(test_exo, vision)
        if test_exo:
            env = make_vec_env('EnvCarry_WithExo-v1', n_envs=1)
            env_norm = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=clip_obs)
        else:
            env = make_vec_env('EnvCarry_WithoutExo-v1', n_envs=1)
            env_norm = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=clip_obs)
        obs = env_norm.reset()
        model = PPO.load(PPO_policy_path,device='cpu')
        while True:
            # action , _ = model.predict(obs)
            action = np.zeros((1, 98))
            obs, rewards, dones, info = env_norm.step(action)

    else:
        model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        # print(data.model.body_mass[0])
        for body_id in range(data.model.nbody):
            body_name = data.model.body(body_id).name
            if body_name == "box_move":
                print("box_mass: ",data.model.body_mass[body_id])

        with open(qfrc_with_zero_exo_path, 'r') as file:
            qfrc_without_exo = yaml.safe_load(file)
        x = np.array(qfrc_without_exo['x'])
        qfrc_with_zero_exo_r = np.array(qfrc_without_exo['qfrc_actuator_r'])
        qfrc_with_zero_exo_l = np.array(qfrc_without_exo['qfrc_actuator_l'])

        with open(qfrc_with_exo_path, 'r') as file:
            qfrc_without_exo = yaml.safe_load(file)
        qfrc_with_exo_r = np.array(qfrc_without_exo['qfrc_actuator_r'])
        qfrc_with_exo_l = np.array(qfrc_without_exo['qfrc_actuator_l'])
        exo_l = np.array(qfrc_without_exo['exo_torque_l'])
        exo_r = np.array(qfrc_without_exo['exo_torque_r'])

        qfrc_with_zero_exo_l = np.abs(qfrc_with_zero_exo_l)
        qfrc_with_exo_l = np.abs(qfrc_with_exo_l)
        qfrc_with_zero_exo_r = np.abs(qfrc_with_zero_exo_r)
        qfrc_with_exo_r = np.abs(qfrc_with_exo_r)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.set_title("right_hip:qfrc_actuator")
        ax1.plot(x,qfrc_with_zero_exo_r,color="red")
        ax1.plot(x,qfrc_with_exo_r,color="green")
        ax1.plot(x,exo_r,color="blue")
        ax1.grid(True)

        ax2.set_title("left_hip:qfrc_actuator")
        ax2.plot(x, qfrc_with_zero_exo_l, color="red")
        ax2.plot(x, qfrc_with_exo_l, color="green")
        ax2.plot(x,exo_l,color="blue")
        ax2.grid(True)

        # 计算均值
        without_exo_left = 0
        with_exo_left = 0
        without_exo_right = 0
        with_exo_right = 0
        for i in range(len(x)):
            without_exo_left += qfrc_with_zero_exo_l[i]
            with_exo_left += qfrc_with_exo_l[i]
            without_exo_right += qfrc_with_zero_exo_r[i]
            with_exo_right += qfrc_with_exo_r[i]
        without_exo_left /= len(x)
        with_exo_left /= len(x)
        without_exo_right /= len(x)
        with_exo_right /= len(x)
        print("x",len(x))
        print("without_exo_left：",without_exo_left)
        print("with_exo_left:",with_exo_left)
        print("without_exo_right",without_exo_right)
        print("with_exo_right",with_exo_right)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

