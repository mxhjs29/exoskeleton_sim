# MARL 训练数据记录指南

## 📊 概述

你已在 `MARL_training.py` 中实现了基于最优超参数组合训练的模型的训练数据记录功能。

## 🎯 记录的内容

### 1. 最优超参数记录
**文件**: `final_training_logs/optimal_hyperparams.txt`

记录所有最优的 reward 权重配置：
```
=== 最优超参数组合 ===
w_pos_err: 0.4
w_proprio_err: 0.4
w_activation: 0.4
...
```

### 2. 训练指标记录
**文件**: `final_training_logs/training_metrics.csv`

每个训练 iteration 记录以下指标：
- `iteration`: 迭代次数
- `exo_reward`: 外骨骼策略奖励
- `human_reward`: 人类策略奖励
- `episode_return_mean`: 平均回合回报
- `episode_len_mean`: 平均回合长度
- `policy_loss_human`: 人类策略损失
- `policy_loss_exo`: 外骨骼策略损失
- `vf_loss_human`: 人类价值函数损失
- `vf_loss_exo`: 外骨骼价值函数损失
- `entropy_human`: 人类策略熵
- `entropy_exo`: 外骨骼策略熵
- `num_episodes`: 本迭代生成的回合数

### 3. Episode 级奖励细节
**文件**: `reward_terms_by_episode.csv`

（来自 `RewardLoggingCallbacks`）记录每个 episode 的详细奖励分量：
- Human: R_p（位置错误）, R_proprio（本体感觉），R_a（激活）
- Exo: R_t（扭矩），R_eas（平滑性）

## 📁 输出目录结构

```
.
├── final_training_logs/                      # 本次最优超参训练的日志
│   ├── optimal_hyperparams.txt              # 最优超参数
│   └── training_metrics.csv                 # 训练指标
├── final_policy_checkpoints/                # 定期保存的检查点
├── best_policy/                             # 历史最优模型
├── best_policy/checkpoint_*/                # 具体检查点文件
├── reward_terms_by_episode.csv              # Episode 级奖励细节
└── ray_results/                             # 网格搜索结果
    └── marl_exo_reward_grid/
        └── PPO_MARL_EXO_ENV_*/
            ├── progress.csv                 # 网格搜索每个试验的进度
            └── checkpoint_*/                # 每个试验的检查点
```

## 🔄 工作流程

### 第一阶段：网格搜索
```python
best = param_search(policies)
# 运行所有超参数组合，每个 7 iteration
# 选择最优的组合
```
输出：
- 所有超参数组合的训练结果在 `ray_results/marl_exo_reward_grid/`
- 每个试验的 `progress.csv` 记录了各自的训练指标

### 第二阶段：最优超参数训练
```python
final_config = PPOConfig().update_from_dict(best.config)
algo = final_config.build()
algo.restore(best.checkpoint.path)
# 继续训练 6 iteration
```
输出：
- `optimal_hyperparams.txt`: 最优超参数
- `training_metrics.csv`: 详细的训练指标
- `best_policy/`: 最好的模型
- `final_policy_checkpoints/`: 定期保存的检查点

## 📈 如何使用这些数据

### 分析训练曲线
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取最优超参数训练的指标
df = pd.read_csv('final_training_logs/training_metrics.csv')

# 绘制奖励曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['iteration'], df['exo_reward'], label='Exo')
plt.plot(df['iteration'], df['human_reward'], label='Human')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.title('Policy Returns')

plt.subplot(1, 3, 2)
plt.plot(df['iteration'], df['policy_loss_human'], label='Human')
plt.plot(df['iteration'], df['policy_loss_exo'], label='Exo')
plt.xlabel('Iteration')
plt.ylabel('Policy Loss')
plt.legend()
plt.title('Policy Loss')

plt.subplot(1, 3, 3)
plt.plot(df['iteration'], df['entropy_human'], label='Human')
plt.plot(df['iteration'], df['entropy_exo'], label='Exo')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.title('Policy Entropy')

plt.tight_layout()
plt.savefig('final_training_logs/training_analysis.png', dpi=150)
plt.show()
```

### 对比网格搜索和最优超参数训练
```python
# 读取网格搜索结果
grid_results = pd.read_csv('ray_results/marl_exo_reward_grid/PPO_MARL_EXO_ENV_*/progress.csv')

# 读取最优超参数训练
optimal_training = pd.read_csv('final_training_logs/training_metrics.csv')

# 比较最终性能
print(f"网格搜索最优: {grid_results['episode_return_mean'].max()}")
print(f"最优超参数继续训练: {optimal_training['episode_return_mean'].iloc[-1]}")
```

## 🚀 主要改进功能

1. **自动化日志记录**：无需手动收集数据
2. **结构化存储**：使用 CSV 格式便于后续分析
3. **完整信息保存**：从超参数到训练指标全覆盖
4. **多层次数据**：网格搜索 → 最优超参 → Episode 细节

## ⚙️ 配置调整

如果需要修改日志行为：

```python
# 在 main() 函数中修改
log_dir = "final_training_logs"  # 修改日志目录
metrics_csv_file = os.path.join(abs_log_dir, "training_metrics.csv")

# 在 _init_metrics_csv() 中添加更多字段
writer.writerow([
    "iteration",
    "exo_reward",
    # 添加其他你需要的指标...
])
```

## 📝 示例输出

### optimal_hyperparams.txt
```
=== 最优超参数组合 ===
w_pos_err: 0.4
w_proprio_err: 0.4
w_activation: 0.4
w_exo_energy: 0.2
w_exo_smooth: 0.2
theta_pos_err: 0.5
theta_proprio_err: 0.3
theta_activation: 0.1
theta_exo_energy: 0.1
theta_exo_smooth: 4
```

### training_metrics.csv 示例
```
iteration,exo_reward,human_reward,episode_return_mean,episode_len_mean,...
0,-150.32,-120.45,-270.77,250.5,...
1,-145.20,-118.60,-263.80,251.2,...
2,-142.15,-115.30,-257.45,252.1,...
...
```

## 🔧 故障排除

**日志文件没有被创建？**
- 检查 `final_training_logs/` 目录是否有写入权限
- 确保你的代码成功完成了网格搜索阶段

**CSV 文件为空？**
- 检查 `algo.train()` 是否正常工作
- 验证 result 字典的结构是否与预期匹配

**缺少某些字段？**
- 在 `_log_training_metrics()` 中添加相应的字段提取逻辑
- 同时在 `_init_metrics_csv()` 中的表头中添加对应列

---

**更新时间**: 2025年11月20日  
**相关文件**: `MARL_training.py`, `Custom_CallBack.py`
