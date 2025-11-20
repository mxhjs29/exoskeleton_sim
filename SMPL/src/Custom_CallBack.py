import os
import csv
from typing import Dict, Any, Tuple

# 新 API：从 callbacks.callbacks 导入 RLlibCallback
from ray.rllib.callbacks.callbacks import RLlibCallback  # 官方推荐路径
# 也可以：from ray.rllib.callbacks import RLlibCallback

class RewardLoggingCallbacks(RLlibCallback):
    """
    适配 RLlib 新 API 栈的回调：
    - 统计每回合的人/外骨骼 reward 子项的和与均值
    - 写入一个 CSV（默认写到当前进程的 cwd；也可通过 env_config['csv_dir'] 指定目录）
    - 通过 MetricsLogger 上报自定义指标（便于在 progress.csv / 结果字典中查看）
    """

    def __init__(
        self,
        agents: Tuple[str, str] = ("human", "exo"),
        csv_filename: str = "reward_terms_by_episode.csv",
        smoothing_window: int = 50,  # 用于 metrics_logger 的滑动平均窗口
    ):
        super().__init__()
        self.agents = agents
        self.csv_filename = csv_filename
        self.smoothing_window = smoothing_window

        # 用于在回合生命周期里保存临时统计（避免依赖 episode.user_data）
        # key = id(episode)，value = dict(累加器等)
        self._ep_state: Dict[int, Dict[str, Any]] = {}

    # ---- 事件：回合开始 -------------------------------------------------------
    def on_episode_start(self, *, episode, env_runner=None, metrics_logger=None, **kwargs):

        ep_key = id(episode)

        # 为本回合初始化累加器
        self._ep_state[ep_key] = {
            "sum_R_p": 0.0,
            "sum_R_proprio": 0.0,
            "sum_R_a": 0.0,
            "sum_R_t": 0.0,
            "sum_R_eas": 0.0,
            "steps": 0,
        }

        # 确定 CSV 路径：优先从 env_config['csv_dir']，否则用当前工作目录
        csv_dir = None
        if env_runner is not None:
            try:
                # 新 API 中可以从 env_runner.config 访问到 env_config
                csv_dir = getattr(env_runner.config, "env_config", {}).get("csv_dir")
            except Exception:
                csv_dir = None
        if not csv_dir:
            csv_dir = os.getcwd()

        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, self.csv_filename)
        self._ep_state[ep_key]["csv_path"] = csv_path

        # 若 CSV 不存在则写表头（一次性）
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode_id", "steps",
                    "human_R_p_sum","human_R_proprio_sum","human_R_a_sum",
                    "exo_R_t_sum","exo_R_eas_sum",
                    "human_R_p_mean","human_R_proprio_mean","human_R_a_mean",
                    "exo_R_t_mean","exo_R_eas_mean"
                ])

    # ---- 事件：每一步 ---------------------------------------------------------
    def on_episode_step(self, *, episode, env_runner=None, metrics_logger=None, **kwargs):
        ep_key = id(episode)
        st = self._ep_state.get(ep_key)
        if st is None:
            return

        # 取最后一步（indices=-1）的 infos；只要我们关心的 agent
        # get_infos 会返回 {agent_id: info_dict} 的字典
        infos = {}
        try:
            infos = episode.get_infos(-1, agent_ids=self.agents)  # 多智能体 API（新栈）
            # 说明文档：MultiAgentEpisode.get_infos(indices, agent_ids=..., env_steps=True) :contentReference[oaicite:3]{index=3}
        except Exception:
            infos = {}

        human_info = infos.get("human") or {}
        exo_info   = infos.get("exo")   or {}

        # 累加 reward 子项
        st["sum_R_p"]       += float(human_info.get("R_p", 0.0))
        st["sum_R_proprio"] += float(human_info.get("R_proprio", 0.0))
        st["sum_R_a"]       += float(human_info.get("R_a", 0.0))

        st["sum_R_t"]       += float(exo_info.get("R_t", 0.0))
        st["sum_R_eas"]     += float(exo_info.get("R_eas", 0.0))

        st["steps"] += 1  # 以环境步为粒度计数

    # ---- 事件：回合结束 -------------------------------------------------------
    def on_episode_end(self, *, episode, env_runner=None, metrics_logger=None, **kwargs):
        ep_key = id(episode)
        st = self._ep_state.get(ep_key)
        if st is None:
            return

        steps = max(1, int(st["steps"]))
        # 计算均值
        ep_metrics = {
            "human_R_p_mean":       st["sum_R_p"]       / steps,
            "human_R_proprio_mean": st["sum_R_proprio"] / steps,
            "human_R_a_mean":       st["sum_R_a"]       / steps,
            "exo_R_t_mean":         st["sum_R_t"]       / steps,
            "exo_R_eas_mean":       st["sum_R_eas"]     / steps,
        }

        # 输出一行便于在日志里观察
        ep_id = getattr(episode, "id_", None)  # 新 Episode 对象通常使用 id_ 字段描述回合ID
        if ep_id is None:
            ep_id = getattr(episode, "episode_id", "?")  # 兜底
        print(
            f"[Episode {ep_id}] "
            f"human: R_p={ep_metrics['human_R_p_mean']:.5f}, "
            f"R_proprio={ep_metrics['human_R_proprio_mean']:.5f}, "
            f"R_a={ep_metrics['human_R_a_mean']:.5f} | "
            f"exo: R_t={ep_metrics['exo_R_t_mean']:.5f}, "
            f"R_eas={ep_metrics['exo_R_eas_mean']:.5f}"
        )

        # 记录到 CSV（每回合一行）
        try:
            with open(st["csv_path"], "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep_id, steps,
                    st["sum_R_p"],
                    st["sum_R_proprio"],
                    st["sum_R_a"],
                    st["sum_R_t"],
                    st["sum_R_eas"],
                    ep_metrics["human_R_p_mean"],
                    ep_metrics["human_R_proprio_mean"],
                    ep_metrics["human_R_a_mean"],
                    ep_metrics["exo_R_t_mean"],
                    ep_metrics["exo_R_eas_mean"],
                ])
        except Exception as e:
            # 避免文件系统问题终止训练
            print(f"[RewardLoggingCallbacks] Failed to write CSV: {e}")

        # 通过 MetricsLogger 上报自定义指标（新 API 推荐方式）
        # 这些键会被聚合到结果字典下（通常在 'env_runners' 节点），便于 progress.csv/TensorBoard/W&B 等查看。:contentReference[oaicite:4]{index=4}
        if metrics_logger is not None:
            metrics_logger.log_value(("custom", "human", "R_p_mean"),       ep_metrics["human_R_p_mean"],       window=self.smoothing_window)
            metrics_logger.log_value(("custom", "human", "R_proprio_mean"), ep_metrics["human_R_proprio_mean"], window=self.smoothing_window)
            metrics_logger.log_value(("custom", "human", "R_a_mean"),       ep_metrics["human_R_a_mean"],       window=self.smoothing_window)
            metrics_logger.log_value(("custom", "exo",   "R_t_mean"),       ep_metrics["exo_R_t_mean"],         window=self.smoothing_window)
            metrics_logger.log_value(("custom", "exo",   "R_eas_mean"),     ep_metrics["exo_R_eas_mean"],       window=self.smoothing_window)

        # 清理本回合缓存，防止内存积累
        self._ep_state.pop(ep_key, None)
