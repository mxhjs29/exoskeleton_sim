import numpy as np
from myosuite.envs.env_variants import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback

register(
    id="EnvCarry_WithoutExo-v1",
    entry_point="SMPL.mjc.env_without_exo:EnvCarry_WithoutExo",
    kwargs={
        'obs_keys': ['qpos','qpos_error','qvel'],
        'model_path': '/home/chenshuo/PycharmProjects/move_sim/SMPL/mjc/mj_fullbody_without_exo_carrying.xml',
        'normalize_act': True,
        'max_episode_steps': 2500,
        'render_bool': False,
        'weighted_reward_keys': {'position': 1,
                                 'velocity': 0.001,
                                 'control': 0.1}
    },
)

tensorboard_log_path = './tb_logs'
tb_log_name = "bend"
log_path = './callback_log/ppo_bend_without_exo'
best_model_save_path = './best_model/ppo_bend_without_exo'
model_save_path = './saved_model/ppo_bend_without_exo'
checkpoint_model_name = "ppo_bend_without_exo"

steps_each_episode = 779
num_episode = 20
du = 5
N = num_episode * steps_each_episode
eval_frequency = N * du
save_frequency = eval_frequency * 2

env = make_vec_env('EnvCarry_WithoutExo-v1',n_envs=2)
eval_env = make_vec_env('EnvCarry_WithoutExo-v1',n_envs=2)
obs = env.reset()
clip_obs = 10


env_norm = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs = clip_obs)
eval_env_norm = VecNormalize(eval_env,norm_obs=True,norm_reward=True,clip_obs=clip_obs)

eval_callback = EvalCallback(eval_env=eval_env_norm, callback_on_new_best=None,
                             callback_after_eval=None, n_eval_episodes=11,
                             eval_freq=eval_frequency, log_path = log_path,
                             deterministic=True, render=False,
                             verbose=2, warn=True,
                             best_model_save_path=best_model_save_path)
checkpoint = CheckpointCallback(verbose=2, save_replay_buffer=True,
                                save_path=model_save_path, save_freq=save_frequency,
                                save_vecnormalize=True,name_prefix=checkpoint_model_name)

policy_kwargs = dict(
    net_arch=[512, 512, 512, 128]
)
model = PPO('MlpPolicy',env_norm,use_sde=False,
            verbose=1,device='cpu',
            clip_range=0.2,
            n_steps=N,
            batch_size=steps_each_episode,
            n_epochs=30,
            ent_coef=0.0001,
            tensorboard_log='./tb_logs/',
            policy_kwargs=policy_kwargs)


# print(model.policy)
model.learn(total_timesteps = np.inf,
            reset_num_timesteps=False,
            tb_log_name=tb_log_name,
            callback=[eval_callback,checkpoint])
model.save("ppo_bend")
# #
del model



