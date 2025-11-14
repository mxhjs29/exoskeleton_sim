import numpy as np
from myosuite.envs.env_variants import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback

model_with_exo_weld_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml"

def create_env(test_exo, vision):
    register(
        id="EnvCarry_WithExo-v1",
        entry_point="SMPL.src.env.env_exo:EnvCarry_WithExo",
        kwargs={
            'obs_keys': ['qpos', 'local_body_pos','local_body_rot_obs','local_body_vel','local_body_ang_vel'],
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

tensorboard_log_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/log/tb_logs'
tb_log_name = "lifting_exo_test"
log_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/log/callback_log/lifting_exo_test/'
best_model_save_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/model/best_model/lifting_exo_test'
model_save_path = '/home/chenshuo/PycharmProjects/move_sim/SMPL/model/trained_models/lifting_exo_test'
checkpoint_model_name = "lifting_exo_test_checkpoint"

steps_each_episode = 779
num_episode = 20
du = 5
N = num_episode * steps_each_episode
eval_frequency = N * du
save_frequency = eval_frequency * 2

env = make_vec_env('EnvCarry_WithoutExoWeld-v1',n_envs=2)
eval_env = make_vec_env('EnvCarry_WithoutExoWeld-v1',n_envs=2)
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
            verbose=1,device='cuda',
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
model.save("new_ppo_bend_weld")
# #
del model

# obs = env.reset()
# print(obs)
# while True:
#     action,_states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(obs)
#     env.render("human")

