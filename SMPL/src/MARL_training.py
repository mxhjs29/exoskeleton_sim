import warnings
warnings.filterwarnings("ignore")
import os
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_IGNORE_RECONSTRUCTION_WARNS"] = "1"
from SMPL.src.env.MARL_env import MARL_EXO_Env
import ray
import numpy as np
import csv
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from SMPL.src.learning.custom_policy import HumanModel,ExoModel
from ray.tune import Tuner
from ray import air
import ray.tune as tune
from SMPL.src.Custom_CallBack import RewardLoggingCallbacks
from itertools import count
from collections import deque
import shutil
from ray.rllib.policy.policy import PolicySpec
os.environ["RAY_disable_metrics_export"] = "1"

render_bool = False
model_path = "/home/chenshuo/PycharmProjects/move_sim/SMPL/data/xml/mj_fullbody_with_exo_carrying_policy_test.xml"
n_stacks = 2
frame_skip = 2
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

def main():
    policies = register_env_policy()

    print("---------------------自定义环境已注册，策略已实例化-------------------")

    best = param_search(policies)

    print("----------------------------网格搜索已完成-------------------------")
    policies = register_env_policy()
    ray.init()
    save_dir = "final_policy_checkpoints"
    best_dir = "best_policy"
    abs_save_dir = os.path.abspath(save_dir)
    abs_best_dir = os.path.abspath(best_dir)
    os.makedirs(abs_save_dir, exist_ok=True)
    os.makedirs(abs_best_dir, exist_ok=True)
    
    # ===== 创建训练日志目录和文件 =====
    log_dir = "final_training_logs"
    abs_log_dir = os.path.abspath(log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)
    
    # 保存最优超参数
    optimal_hyperparams_file = os.path.join(abs_log_dir, "optimal_hyperparams.txt")
    with open(optimal_hyperparams_file, "w") as f:
        f.write("=== 最优超参数组合 ===\n")
        for k, v in best.config["env_config"]["weighted_reward_keys"].items():
            f.write(f"{k}: {v}\n")
    
    # 创建 CSV 文件记录每次迭代的指标
    metrics_csv_file = os.path.join(abs_log_dir, "training_metrics.csv")
    _init_metrics_csv(metrics_csv_file)
    
    best_reward = 0
    final_config = PPOConfig()
    final_config = final_config.update_from_dict(config_dict = best.config)
    algo = final_config.build()
    algo.restore(best.checkpoint.path)
    save_interval = 10    # 每 100 次 iteration 保存一次
    max_iters = 6    # 可以设置你想训练多少 iteration（或无穷）
    last_50_ckpts = deque(maxlen=50)
    for i in count(2):
        result = algo.train()
        reward = result["env_runners"]["module_episode_returns_mean"]["exo_policy"]
        print_training_summary(result)
        
        # ===== 记录训练指标到 CSV =====
        _log_training_metrics(result, metrics_csv_file, i)

        # ===== 1) 保存历史最优模型 =====
        if reward > best_reward:
            best_reward = reward
            algo.save(f"file://{abs_best_dir}")
            print(f"[BEST] Updated at iter {i}, reward={reward}")

        # ===== 2) 保存最近 50 次模型（rolling buffer）====
        if i % save_interval == 0:
            ckpt = algo.save(f"file://{abs_save_dir}")
            last_50_ckpts.append(ckpt)
            print("[SAVE] Saved checkpoint")

            # 删除被挤掉的最旧 checkpoint
            if len(last_50_ckpts) == last_50_ckpts.maxlen:
                try:
                    oldest = last_50_ckpts[0]["checkpoint_path"]
                    shutil.rmtree(oldest, ignore_errors=True)
                    print("[CLEAN] Removed old checkpoint")
                except Exception as e:
                    print(f"[WARN] Failed to clean old ckpt: {e}")
                finally:
                    # 无论成功失败，都移除最旧的元素，防止死循环
                    last_50_ckpts.popleft()

    print(f"[BEST] save_path{abs_best_dir}")
    print(f"[LOGS] Training metrics saved to {metrics_csv_file}")

    ray.shutdown()
    print("------------------------训练结束----------------------------")

def register_env_policy():
    #注册环境、自定义网络
    register_env("MARL_EXO_ENV", env_creator)
    ModelCatalog.register_custom_model("human_model", HumanModel)
    ModelCatalog.register_custom_model("exo_model", ExoModel)

    #创建临时环境
    temp_env = raw_env_creator({
        "render_mode" : "human",
        "render_bool" : render_bool,
        "model_path" : model_path,
        "n_stacks" : n_stacks,
        "human_obs_keys": human_obs_keys,
        "exo_obs_keys" : exo_obs_keys,
        "weighted_reward_keys" : weighted_reward_keys
    })

    agents = temp_env.possible_agents
    print("agents: ",agents)
    obs_spaces = {agent: temp_env.observation_space(agent) for agent in agents}
    act_spaces = {agent: temp_env.action_space(agent) for agent in agents}
    print("obs_space",obs_spaces)
    print("act_space",act_spaces)
    temp_env.close()

    policies= {
        "human_policy": PolicySpec(
            policy_class=None,
            observation_space=obs_spaces["human"],
            action_space=act_spaces["human"],
            config={"model": {"custom_model": "human_model",
                              "free_log_std": True,
                              "log_std_init": -1.0,
                              "log_std_clip": [-5.0, 1.5],},
                    "entropy_coeff_schedule": [(0, 0.02), (5e7, 0.01), (1e9, 0.002)],}
        ),
        "exo_policy": PolicySpec(
            policy_class=None,
            observation_space=obs_spaces["exo"],
            action_space=act_spaces["exo"],
            config={"model": {"custom_model": "exo_model",
                              "vf_share_layers": False,
                              "free_log_std": True,
                              "log_std_init": -0.5,   
                              "log_std_clip": [-5.0, 2.0],},
                    "entropy_coeff_schedule": [(0, 0.02), (5e8, 0.01), (1e10, 0.002)],}
        ),
    }
    return policies

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

def policy_mapping_fn(agent_id, episode, **kwargs):
    return f"{agent_id}_policy"

def param_search(policies):
     #RLlib训练配置
    base_cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner = True,
            enable_env_runner_and_connector_v2 = True
        )
        .resources(
            num_gpus=0,                    # 或 1
            num_cpus_for_main_process=2    # 新版本推荐字段名
        )
        .environment(
            env = "MARL_EXO_ENV",
            env_config = {
                "render_mode": "human",
                "render_bool": render_bool,
                "frame_skip": frame_skip,
                "model_path": model_path,
                "n_stacks": n_stacks,
                "human_obs_keys": human_obs_keys,
                "exo_obs_keys": exo_obs_keys,
                "weighted_reward_keys": weighted_reward_keys
            }   
        )
        .framework("torch")
        .env_runners(
            num_env_runners=4,           
            num_envs_per_env_runner=4,   
            num_cpus_per_env_runner=2,   
            rollout_fragment_length="auto",
            sample_timeout_s=120,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            lambda_=0.95,
            clip_param=0.2,
            grad_clip=0.5,
            vf_loss_coeff=1.5,
            vf_clip_param=10.0,
            kl_coeff=0.2,
            kl_target=0.01
        )
        .update_from_dict({
            "train_batch_size": 65536,
            "sgd_minibatch_size": 2048,
            "num_epochs": 20,
        })
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .callbacks(RewardLoggingCallbacks)
    )
    base_cfg.observation_filter = "MeanStdFilter"
    base_cfg.batch_mode = "complete_episodes"
    param_space = base_cfg.to_dict()
    wr = param_space["env_config"]["weighted_reward_keys"]
    wr["w_pos_err"]     = tune.grid_search([0.5])
    # wr["w_proprio_err"] = tune.grid_search([0.5, 0.6, 0.7])
    # wr["w_activation"]  = tune.grid_search([0.3, 0.4, 0.5])
    # wr["w_exo_energy"]  = tune.grid_search([0.05, 0.1, 0.2])
    # wr["w_exo_smooth"]  = tune.grid_search([0.1, 0.2, 0.3])

    relative_path = "ray_results"
    abs_path = os.path.abspath(relative_path)
    run_cfg = air.RunConfig(
        name="marl_exo_reward_grid",
        stop={"training_iteration": 1},             # 每个网格训练 30 iter，可按需调整
        storage_path=f"file://{abs_path}",                     # 结果输出目录
        verbose=1
    )

    # #网格搜索
    tuner = tune.Tuner(
        "PPO",
        param_space=param_space,
        run_config=run_cfg,
        tune_config=tune.TuneConfig(reuse_actors=True)
    )
    results = tuner.fit()
    best = results.get_best_result(metric="env_runners/module_episode_returns_mean/exo_policy", mode="max")
    print("最优参数组合：")
    for k,v in best.config["env_config"]["weighted_reward_keys"].items():
        print(k,v)
    ray.shutdown()
    return best

def print_training_summary(result: dict):
    """适配 RLlib 新 API 栈 Result.metrics 结构的训练信息打印函数。

    支持两种输入：
    - 直接传 algo.train() 返回的 dict
    - 传 Tune 的 Result.metrics（外层有 .metrics 时也能自动兼容）
    """

    # 既兼容 Result(metrics=...)，也兼容直接传 metrics dict
    metrics = result.get("metrics", result)

    timers       = metrics.get("timers", {}) or {}
    env_runners  = metrics.get("env_runners", {}) or {}
    learners     = metrics.get("learners", {}) or {}
    perf         = metrics.get("perf", {}) or {}
    fault        = metrics.get("fault_tolerance", {}) or {}

    # --------- 基本信息 ----------
    iteration = metrics.get("training_iteration")
    if iteration is None:
        iteration = timers.get("training_iteration")

    num_env_steps_lifetime = (
        metrics.get("num_env_steps_sampled_lifetime")
        or env_runners.get("num_env_steps_sampled_lifetime")
    )
    env_tput = env_runners.get("num_env_steps_sampled_lifetime_throughput")

    # --------- Episode 级指标 ----------
    ep_ret_mean = env_runners.get("episode_return_mean")
    ep_ret_max  = env_runners.get("episode_return_max")
    ep_ret_min  = env_runners.get("episode_return_min")
    ep_len_mean = env_runners.get("episode_len_mean")
    ep_len_max  = env_runners.get("episode_len_max")
    ep_len_min  = env_runners.get("episode_len_min")
    num_episodes = env_runners.get("num_episodes")

    # 多智能体 / 多策略回报
    agent_returns  = env_runners.get("agent_episode_returns_mean", {}) or {}
    policy_returns = env_runners.get("module_episode_returns_mean", {}) or {}

    # 自定义 reward 细节（human/exo 分开）
    custom_rewards = env_runners.get("custom", {}) or {}

    # --------- Learner / 优化器信息 ----------
    # 汇总信息（__all_modules__）
    agg_learner = learners.get("__all_modules__", {}) or {}
    # 单个策略的学习信息（human_policy / exo_policy）
    per_module_stats = {
        mid: info for mid, info in learners.items()
        if mid != "__all_modules__"
    }

    # --------- 打印 ----------
    print("\n================== TRAINING SUMMARY ==================")
    print(f"Iteration:                     {iteration}")
    print("------------------------------------------------------")

    print("Episode Metrics:")
    print(f"  Num Episodes (this iter):    {num_episodes}")
    print(f"  Episode Return Mean:         {ep_ret_mean}")
    print(f"  Episode Return Max:          {ep_ret_max}")
    print(f"  Episode Return Min:          {ep_ret_min}")
    print(f"  Episode Length Mean:         {ep_len_mean}")
    print(f"  Episode Length Max:          {ep_len_max}")
    print(f"  Episode Length Min:          {ep_len_min}")
    print("------------------------------------------------------")

    print("Agent Episode Returns (per-agent):")
    for agent_id, rew in agent_returns.items():
        print(f"  {agent_id:15s}: {rew}")
    print("------------------------------------------------------")

    print("Policy Episode Returns (per-policy/module):")
    for module_id, rew in policy_returns.items():
        print(f"  {module_id:15s}: {rew}")
    print("------------------------------------------------------")

    # 自定义 reward 组件（你现在有 human / exo 的各个 reward term）
    if custom_rewards:
        print("Custom Reward Components:")
        for agent_id, comp in custom_rewards.items():
            print(f"  [{agent_id}]")
            for k, v in comp.items():
                print(f"      {k:15s}: {v}")
        print("------------------------------------------------------")

    # Learner 优化信息
    print("Learner / Optimization Stats (per policy):")
    for module_id, info in per_module_stats.items():
        print(f"  {module_id}:")
        # 新栈里这些字段在 learners[module_id] 直接给出
        stats = {
            "policy_loss":      info.get("policy_loss"),
            "vf_loss":          info.get("vf_loss"),
            "mean_kl_loss":     info.get("mean_kl_loss"),
            "entropy":          info.get("entropy"),
            "vf_explained_var": info.get("vf_explained_var"),
            "curr_kl_coeff":    info.get("curr_kl_coeff"),
            "curr_entropy_coeff": info.get("curr_entropy_coeff"),
            "lr":               info.get("default_optimizer_learning_rate"),
            "num_params":       info.get("num_trainable_parameters"),
        }
        for k, v in stats.items():
            print(f"      {k:20s}: {v}")
        print("------------------------------------------------------")

    # 系统性能
    print("System / Performance:")
    print(f"  Total Env Steps (lifetime):  {num_env_steps_lifetime}")
    print(f"  Env Throughput (steps/s):    {env_tput}")
    print(f"  CPU Util (%):                {perf.get('cpu_util_percent')}")
    print(f"  RAM Util (%):                {perf.get('ram_util_percent')}")
    print(f"  GPU Util (%):                {perf.get('gpu_util_percent0')}")
    print(f"  VRAM Util (%):               {perf.get('vram_util_percent0')}")
    print(f"  Healthy Workers:             {fault.get('num_healthy_workers')}")
    print(f"  Remote Worker Restarts:      {fault.get('num_remote_worker_restarts')}")
    print("======================================================\n")


def _init_metrics_csv(csv_file):
    """初始化训练指标 CSV 文件"""
    import csv
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration",
            "exo_reward",
            "human_reward",
            "episode_return_mean",
            "episode_len_mean",
            "policy_loss_human",
            "policy_loss_exo",
            "vf_loss_human",
            "vf_loss_exo",
            "entropy_human",
            "entropy_exo",
            "num_episodes",
        ])


def _log_training_metrics(result: dict, csv_file: str, iteration: int):
    """将训练指标记录到 CSV 文件"""
    import csv
    
    metrics = result.get("metrics", result)
    env_runners = metrics.get("env_runners", {}) or {}
    learners = metrics.get("learners", {}) or {}
    
    # 提取指标
    exo_reward = env_runners.get("module_episode_returns_mean", {}).get("exo_policy")
    human_reward = env_runners.get("module_episode_returns_mean", {}).get("human_policy")
    ep_ret_mean = env_runners.get("episode_return_mean")
    ep_len_mean = env_runners.get("episode_len_mean")
    num_episodes = env_runners.get("num_episodes")
    
    # 提取 learner 指标
    human_stats = learners.get("human_policy", {}) or {}
    exo_stats = learners.get("exo_policy", {}) or {}
    
    policy_loss_human = human_stats.get("policy_loss")
    policy_loss_exo = exo_stats.get("policy_loss")
    vf_loss_human = human_stats.get("vf_loss")
    vf_loss_exo = exo_stats.get("vf_loss")
    entropy_human = human_stats.get("entropy")
    entropy_exo = exo_stats.get("entropy")
    
    # 写入 CSV
    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                exo_reward,
                human_reward,
                ep_ret_mean,
                ep_len_mean,
                policy_loss_human,
                policy_loss_exo,
                vf_loss_human,
                vf_loss_exo,
                entropy_human,
                entropy_exo,
                num_episodes,
            ])
    except Exception as e:
        print(f"[WARNING] Failed to write metrics to CSV: {e}")



if __name__=="__main__":
    main()


